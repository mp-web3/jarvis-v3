"""Jarvis v3 listener — tmux mode.

VAD-gated STT with tmux output. Reuses PipelineResources and SpeechDetector
from pipeline.py but routes output to tmux send-keys instead of WebSocket.

Key improvements over v2:
- asyncio.Lock for MLX safety (not flag-based)
- 4-state VAD machine (not binary)
- EOU detection (not fixed silence timer)
- Event-based barge-in cancellation
"""

import asyncio
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from jarvis.config import CHUNK_SIZE, MIN_SEGMENT_DURATION, SAMPLE_RATE, get_config
from jarvis.pipeline import PipelineEvent, PipelineResources, SpeechDetector

logger = logging.getLogger(__name__)

VOICE_MODE_FLAG = Path.home() / ".claude" / ".voice-mode"
SPEAKING_FLAG = Path.home() / ".claude" / ".speaking"
TTS_QUEUE = Path.home() / ".claude" / ".tts-queue"
SPEAKER_EMBEDDING = Path(__file__).parent.parent / "speaker_embedding_ecapa.npy"

EXIT_PHRASES = {"stop jarvis", "jarvis stop", "esci jarvis"}


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
    """tmux-mode listener using shared pipeline components."""

    def __init__(self, resources: PipelineResources, tmux_target: str):
        self.resources = resources
        self.tmux_target = tmux_target
        self.detector = SpeechDetector()
        self.config = get_config()

        self._running = False
        self._tts_playing = False

        # Transcription
        self._transcription_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._accumulated_text = ""
        self._is_accumulating = False

        # Barge-in
        self._response_cancel_event: asyncio.Event | None = None

        # Debounce for tmux (combine rapid utterances)
        listener_cfg = self.config.get("listener", {})
        debounce_ms = listener_cfg.get("debounce_ms", 300)
        self._debounce_s = debounce_ms / 1000.0
        self._text_buffer: list[str] = []
        self._debounce_task: asyncio.Task | None = None

    async def run(self):
        loop = asyncio.get_running_loop()
        self._running = True
        self.detector.start_listening()

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
        transcription_task = asyncio.create_task(self._transcription_worker())
        tts_task = asyncio.create_task(self._watch_tts_queue())

        print(
            "Jarvis v3 active (Parakeet TDT + Kokoro + Silero VAD + EOU). "
            "Speak naturally. Say 'stop Jarvis' to quit.",
            file=sys.stderr,
        )

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if self._tts_playing:
                    continue

                events = self.detector.process_chunk(chunk)
                await self._handle_events(events)

        except asyncio.CancelledError:
            pass
        finally:
            stream.stop()
            stream.close()
            transcription_task.cancel()
            tts_task.cancel()
            for f in (VOICE_MODE_FLAG, SPEAKING_FLAG, TTS_QUEUE):
                f.unlink(missing_ok=True)
            print("\nJarvis v3 stopped.", file=sys.stderr)

    async def _handle_events(self, events: list[str]):
        if PipelineEvent.SPEECH_START in events:
            self._is_accumulating = True
            self._accumulated_text = ""
            # Interrupt TTS on barge-in
            if self._tts_playing and self._response_cancel_event:
                self._response_cancel_event.set()

        for event in events:
            if event == PipelineEvent.TRANSCRIBE:
                await self._queue_transcription()
            elif event == PipelineEvent.RESPOND:
                await self._finalize_and_send()

    async def _queue_transcription(self):
        segment = self.detector.current_segment
        if segment is None or len(segment) < SAMPLE_RATE * MIN_SEGMENT_DURATION:
            return
        self.detector.segment_count += 1
        try:
            self._transcription_queue.put_nowait(
                (self.detector.segment_count, segment)
            )
        except asyncio.QueueFull:
            logger.warning("Transcription queue full")

    async def _transcription_worker(self):
        from jarvis.transcriber import transcribe

        while True:
            segment_id, audio = await self._transcription_queue.get()
            try:
                async with self.resources.mlx_lock:
                    text = await asyncio.get_running_loop().run_in_executor(
                        None, transcribe, audio, SAMPLE_RATE
                    )
                if text and text.strip():
                    logger.info("Transcribed #%d: %s", segment_id, text)
                    if self._is_accumulating:
                        if self._accumulated_text:
                            self._accumulated_text += " " + text.strip()
                        else:
                            self._accumulated_text = text.strip()
            except Exception:
                logger.exception("Transcription error")
            finally:
                self._transcription_queue.task_done()

    async def _finalize_and_send(self):
        await self._transcription_queue.join()

        self._is_accumulating = False
        if not self._accumulated_text:
            return

        text = self._accumulated_text.strip()
        self._accumulated_text = ""

        lower = text.lower().rstrip(".!?,")
        if lower in EXIT_PHRASES:
            print("Jarvis: Goodbye.", file=sys.stderr)
            self._running = False
            return

        if len(lower.split()) <= 1:
            logger.debug("Filtered single word: %s", text)
            return

        print(f"[voice] {text}", file=sys.stderr)
        await asyncio.get_running_loop().run_in_executor(
            None, _send_to_tmux, text, self.tmux_target
        )

    async def _watch_tts_queue(self):
        """Poll for TTS text queued by voice output hook."""
        from jarvis.speaker import get_sample_rate, render

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
        from jarvis.speaker import get_sample_rate, render

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
                await asyncio.sleep(0.1)
                logger.info("TTS interrupted")
            else:
                logger.info("TTS finished")

        except Exception:
            logger.exception("TTS failed")
        finally:
            self._tts_playing = False
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
