"""Jarvis v3 listener — tmux mode.

Streams mic audio to Deepgram for STT + turn detection, then routes
transcribed text to tmux send-keys for Claude Code injection.

Barge-in during TTS uses Silero VAD (local, CPU) to detect speech,
pauses Deepgram streaming to avoid echo feedback, and resumes after
TTS is cancelled.
"""

import asyncio
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from jarvis.config import CHUNK_SIZE, SAMPLE_RATE, VAD_START_THRESHOLD, get_config
from jarvis.pipeline import PipelineResources
from jarvis.transcriber import DeepgramTranscriber
from jarvis.vad import SileroVAD

logger = logging.getLogger(__name__)

VOICE_MODE_FLAG = Path.home() / ".claude" / ".voice-mode"
SPEAKING_FLAG = Path.home() / ".claude" / ".speaking"
TTS_QUEUE = Path.home() / ".claude" / ".tts-queue"

EXIT_PHRASES = {"stop jarvis", "jarvis stop", "esci jarvis"}

# Filler words to strip from transcriptions (en + it)
_FILLERS = re.compile(
    r"\b(uh|uhm|um|umm|ah|eh|er|erm|hmm|hm|like|you know|i mean|basically|actually"
    r"|allora|cioè|ehm|beh|mah|diciamo|praticamente)\b",
    re.IGNORECASE,
)

# Filler-only patterns for barge-in classification (en + it)
_BARGEIN_FILLER = re.compile(
    r"^(uh|uhm|um|umm|ah|eh|er|erm|hmm|hm|mm|mhm|uh huh|oh|okay|ok|right|yeah"
    r"|yes|no|sure|huh|wow|whoa|ooh|aah"
    r"|sì|no|ok|va bene|ah|eh|oh|mah|beh|ehm|allora)\.?$",
    re.IGNORECASE,
)


def _clean_transcript(text: str) -> str:
    """Remove filler words and deduplicate consecutive repeated words."""
    text = _FILLERS.sub("", text)
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text


def _find_device(name: str, direction: str = "input") -> int | None:
    key = "max_input_channels" if direction == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if name in d["name"] and d[key] > 0:
            return i
    return None


def _validate_tmux_target(target: str):
    if not shutil.which("tmux"):
        print("Error: tmux required. Install: brew install tmux", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(
        ["tmux", "has-session", "-t", target.split(":")[0]], capture_output=True
    )
    if result.returncode != 0:
        session = target.split(":")[0]
        print(
            f"Error: tmux session '{session}' not found.\n"
            f"  tmux new-session -s {session}\n"
            f"  claude",
            file=sys.stderr,
        )
        sys.exit(1)


def _send_to_tmux(text: str, target: str):
    subprocess.run(["tmux", "send-keys", "-t", target, "--", text, "Enter"], check=False)


class JarvisListener:
    """tmux-mode listener using Deepgram streaming STT."""

    def __init__(self, resources: PipelineResources, tmux_target: str):
        self.resources = resources
        self.tmux_target = tmux_target
        self.config = get_config()

        self._running = False
        self._tts_playing = False
        self._tts_settling = False

        # Deepgram transcriber
        listener_cfg = self.config.get("listener", {})
        debounce_s = listener_cfg.get("debounce_ms", 300) / 1000.0
        self.transcriber = DeepgramTranscriber(debounce_s=debounce_s)

        # Barge-in
        self._response_cancel_event: asyncio.Event | None = None
        self._bargein_enabled = listener_cfg.get("bargein_enabled", True)
        self._bargein_smart = listener_cfg.get("bargein_smart", True)
        chunk_ms = listener_cfg.get("chunk_ms", 30)
        bargein_ms = listener_cfg.get("bargein_sustain_ms", 800)
        self._bargein_threshold = max(1, int(bargein_ms / chunk_ms))
        self._bargein_count = 0
        self._bargein_vad = SileroVAD()
        self._bargein_buffer: list[np.ndarray] = []

        # Acknowledgment
        self._ack_enabled = listener_cfg.get("ack_enabled", True)

        # Track when last utterance was sent (for continuation detection)
        self._last_send_time: float = 0.0
        self._continuation_window_s = listener_cfg.get("continuation_window_s", 2.0)
        self._is_continuation = False

    async def run(self):
        loop = asyncio.get_running_loop()
        self._running = True

        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.warning("Mic: %s", status)
            loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

        # Resolve devices
        listener_cfg = self.config.get("listener", {})
        input_device = None
        input_name = listener_cfg.get("input_device")
        if input_name:
            input_device = _find_device(input_name, "input")
            if input_device is not None:
                logger.info("Input: [%d] %s", input_device, sd.query_devices(input_device)["name"])

        output_name = listener_cfg.get("output_device")
        if output_name:
            out_idx = _find_device(output_name, "output")
            if out_idx is not None:
                sd.default.device = (sd.default.device[0], out_idx)
                logger.info("Output: [%d] %s", out_idx, sd.query_devices(out_idx)["name"])

        stream = sd.InputStream(
            device=input_device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=mic_callback,
        )
        stream.start()

        VOICE_MODE_FLAG.touch()
        await self.transcriber.start()
        utterance_task = asyncio.create_task(self._utterance_worker())
        tts_task = asyncio.create_task(self._watch_tts_queue())

        print(
            "Jarvis v3 active (Deepgram STT + Kokoro TTS). "
            "Speak naturally. Say 'stop Jarvis' to quit.",
            file=sys.stderr,
        )

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Settling after barge-in: discard mic input while sd.stop() completes
                if self._tts_settling:
                    continue

                # During TTS: run Silero VAD for barge-in (don't stream to Deepgram)
                if self._tts_playing:
                    if self._bargein_enabled and self._response_cancel_event:
                        vad_prob = self._bargein_vad.process_chunk(chunk)
                        self._bargein_buffer.append(chunk)
                        if vad_prob >= VAD_START_THRESHOLD:
                            self._bargein_count += 1
                            if self._bargein_count >= self._bargein_threshold:
                                if self._bargein_smart:
                                    await self._classify_bargein()
                                else:
                                    logger.info(
                                        "Barge-in: speech detected for %d chunks",
                                        self._bargein_count,
                                    )
                                    self._response_cancel_event.set()
                                self._bargein_count = 0
                        else:
                            self._bargein_count = 0
                        # Keep buffer trimmed
                        max_buf = self._bargein_threshold * 3
                        if len(self._bargein_buffer) > max_buf:
                            self._bargein_buffer = self._bargein_buffer[-max_buf:]
                    continue

                # Normal: stream audio to Deepgram
                await self.transcriber.send_audio(chunk)

        except asyncio.CancelledError:
            pass
        finally:
            stream.stop()
            stream.close()
            await self.transcriber.stop()
            utterance_task.cancel()
            tts_task.cancel()
            for f in (VOICE_MODE_FLAG, SPEAKING_FLAG, TTS_QUEUE):
                f.unlink(missing_ok=True)
            print("\nJarvis v3 stopped.", file=sys.stderr)

    async def _utterance_worker(self):
        """Consume complete utterances from Deepgram and send to tmux."""
        while self._running:
            try:
                text = await asyncio.wait_for(
                    self.transcriber.ready_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            await self._process_utterance(text)

    async def _process_utterance(self, raw_text: str):
        """Clean, filter, and send an utterance to tmux."""
        # Continuation detection
        elapsed = time.monotonic() - self._last_send_time
        self._is_continuation = (
            self._last_send_time > 0
            and elapsed < self._continuation_window_s
            and not self._tts_playing
        )
        if self._is_continuation:
            logger.info("Continuation detected (%.1fs since last send)", elapsed)

        # Clean transcript (regex only — Deepgram transcripts are already clean)
        text = _clean_transcript(raw_text)
        if raw_text != text:
            logger.debug("Cleaned: '%s' → '%s'", raw_text, text)

        lower = text.lower().rstrip(".!?,")
        if lower in EXIT_PHRASES:
            print("Jarvis: Goodbye.", file=sys.stderr)
            self._running = False
            return

        if len(lower.split()) <= 1:
            logger.debug("Filtered single word: %s", text)
            return

        # Play acknowledgment (skip on continuations)
        if self._ack_enabled and not self._is_continuation:
            await self._play_ack()
        self._is_continuation = False

        self._last_send_time = time.monotonic()
        print(f"[voice] {text}", file=sys.stderr)
        await asyncio.get_running_loop().run_in_executor(
            None, _send_to_tmux, text, self.tmux_target
        )

    async def _play_ack(self):
        """Play a pre-cached acknowledgment clip."""
        from jarvis.speaker import get_random_ack, get_sample_rate

        audio = get_random_ack()
        if audio is None:
            return

        try:
            sr = get_sample_rate()
            sd.play(audio, samplerate=sr)
            duration = len(audio) / sr
            await asyncio.sleep(duration)
        except Exception:
            logger.exception("Ack playback failed")

    async def _classify_bargein(self):
        """Smart barge-in: send buffered audio to Deepgram for classification.

        If the transcribed text is a filler word, resume TTS. Otherwise cancel it.
        Since Deepgram is streaming, we send the buffered audio and wait briefly
        for a transcript to classify.
        """
        try:
            audio = np.concatenate(self._bargein_buffer)
            self._bargein_buffer.clear()

            # Send buffered audio to Deepgram for transcription
            await self.transcriber.send_audio(audio)

            # Wait briefly for a transcript
            try:
                await asyncio.wait_for(
                    self.transcriber.speech_detected.wait(), timeout=0.8
                )
                self.transcriber.speech_detected.clear()
            except asyncio.TimeoutError:
                logger.info("Barge-in: no transcript received, cancelling TTS")
                self._response_cancel_event.set()
                return

            # Check accumulated text for filler classification
            text = self.transcriber._accumulated.strip()
            if not text:
                logger.info("Barge-in: empty transcript, resuming TTS")
                return

            lower = text.lower().rstrip(".!?,")
            if _BARGEIN_FILLER.match(lower):
                logger.info("Barge-in: filler ('%s'), resuming TTS", text)
                return

            # Intentional interruption
            logger.info("Barge-in: intentional ('%s'), cancelling TTS", text)
            self._response_cancel_event.set()

        except Exception:
            logger.exception("Barge-in classification failed, cancelling TTS")
            self._response_cancel_event.set()

    async def _watch_tts_queue(self):
        """Poll for TTS text queued by voice output hook."""
        while self._running:
            try:
                if TTS_QUEUE.exists():
                    text = TTS_QUEUE.read_text().strip()
                    TTS_QUEUE.unlink(missing_ok=True)
                    if text:
                        await self._play_tts(text)
            except OSError:
                pass
            await asyncio.sleep(0.2)

    async def _play_tts(self, text: str):
        from jarvis.speaker import get_sample_rate, render, sanitize_for_tts

        text = sanitize_for_tts(text)
        if not text:
            return

        try:
            self._tts_playing = True
            SPEAKING_FLAG.touch()
            self._response_cancel_event = asyncio.Event()

            async with self.resources.mlx_lock:
                audio = await asyncio.get_running_loop().run_in_executor(
                    None, render, text
                )

            if audio is None or self._response_cancel_event.is_set():
                return

            sr = get_sample_rate()
            sd.play(audio, samplerate=sr)
            duration = len(audio) / sr
            start = time.monotonic()

            while not self._response_cancel_event.is_set():
                if time.monotonic() - start >= duration:
                    break
                await asyncio.sleep(0.05)

            if self._response_cancel_event.is_set():
                sd.stop()
                self._tts_settling = True
                await asyncio.sleep(0.15)
                self._tts_settling = False
                self._bargein_vad.reset()
                logger.info("TTS interrupted, settled")
            else:
                logger.info("TTS finished")

        except Exception:
            logger.exception("TTS failed")
        finally:
            self._tts_playing = False
            self._tts_settling = False
            self._bargein_count = 0
            self._bargein_buffer.clear()
            self._response_cancel_event = None
            SPEAKING_FLAG.unlink(missing_ok=True)

    def stop(self):
        self._running = False


def run_jarvis(tmux_target: str = "claude:0"):
    _validate_tmux_target(tmux_target)
    resources = PipelineResources()
    listener = JarvisListener(resources, tmux_target)

    import signal

    def handle_signal(sig, frame):
        listener.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    asyncio.run(listener.run())
