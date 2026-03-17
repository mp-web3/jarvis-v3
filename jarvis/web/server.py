"""Jarvis v3 web server — browser-based voice I/O with echo cancellation.

Runs ML models server-side (Parakeet STT, Kokoro TTS, Silero VAD, SmartTurn EOU)
and streams audio to/from a browser client over WebSocket. The browser provides
hardware-accelerated AEC via getUserMedia({ echoCancellation: true }), enabling
speaker-free operation without headphones.

Output goes to Claude via tmux (same as listener.py). Claude's response comes
back through the voice-output-hook -> .tts-queue -> TTS -> WebSocket -> browser.
"""

import asyncio
import base64
import io
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from jarvis.audio_buffer import split_audio_into_chunks
from jarvis.config import (
    CHUNK_SIZE,
    MAX_TRANSCRIPTION_QUEUE_SIZE,
    MIN_SEGMENT_DURATION,
    SAMPLE_RATE,
    get_config,
)
from jarvis.pipeline import PipelineEvent, PipelineResources, SpeechDetector

logger = logging.getLogger(__name__)

TTS_QUEUE = Path.home() / ".claude" / ".tts-queue"
SPEAKING_FLAG = Path.home() / ".claude" / ".speaking"
VOICE_MODE_FLAG = Path.home() / ".claude" / ".voice-mode"

STATIC_DIR = Path(__file__).parent.parent.parent / "web"

EXIT_PHRASES = {"stop jarvis", "jarvis stop", "esci jarvis"}

_FILLERS = re.compile(
    r"\b(uh|uhm|um|umm|ah|eh|er|erm|hmm|hm|like|you know|i mean|basically|actually"
    r"|allora|cioè|ehm|beh|mah|diciamo|praticamente)\b",
    re.IGNORECASE,
)


def _clean_transcript(text: str) -> str:
    """Remove filler words and deduplicate consecutive repeated words."""
    text = _FILLERS.sub("", text)
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text


def _send_to_tmux(text: str, target: str):
    subprocess.run(
        ["tmux", "send-keys", "-t", target, "--", text, "Enter"],
        check=False,
    )


def _validate_tmux_target(target: str):
    if not shutil.which("tmux"):
        raise RuntimeError("tmux required — brew install tmux")
    result = subprocess.run(
        ["tmux", "has-session", "-t", target.split(":")[0]],
        capture_output=True,
    )
    if result.returncode != 0:
        session = target.split(":")[0]
        raise RuntimeError(
            f"tmux session '{session}' not found. "
            f"Start it: tmux new-session -s {session}"
        )


def _encode_audio(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_float32_audio(data: str) -> np.ndarray | None:
    try:
        audio_bytes = base64.b64decode(data.encode("ascii"))
        if len(audio_bytes) % 4 != 0:
            return None
        return np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception:
        return None


class WebVoicePipeline:
    """Per-connection voice pipeline for browser-based I/O."""

    def __init__(
        self,
        ws: WebSocket,
        resources: PipelineResources,
        tmux_target: str,
    ):
        self.ws = ws
        self.resources = resources
        self.tmux_target = tmux_target
        self.detector = SpeechDetector(
            text_provider=lambda: self._accumulated_text,
        )
        self.config = get_config()

        self._transcription_queue: asyncio.Queue = asyncio.Queue(
            maxsize=MAX_TRANSCRIPTION_QUEUE_SIZE,
        )
        self._accumulated_text = ""
        self._is_accumulating = False
        self._transcription_task: asyncio.Task | None = None

        self._tts_task: asyncio.Task | None = None
        self._tts_cancel = asyncio.Event()

        listener_cfg = self.config.get("listener", {})
        self._polish_enabled = listener_cfg.get("polish_enabled", False)

    async def start(self):
        self._transcription_task = asyncio.create_task(
            self._transcription_worker(),
        )
        self._tts_task = asyncio.create_task(self._watch_tts_queue())
        VOICE_MODE_FLAG.touch()
        await self._send_state()
        logger.info("Web pipeline started")

    async def shutdown(self):
        self._tts_cancel.set()
        for task in (self._transcription_task, self._tts_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        for flag in (VOICE_MODE_FLAG, SPEAKING_FLAG):
            flag.unlink(missing_ok=True)
        TTS_QUEUE.unlink(missing_ok=True)
        logger.info("Web pipeline shutdown")

    async def handle_message(self, payload: dict):
        event = payload.get("event")

        if event == "start":
            self.detector.start_listening()
            await self._send_state()
        elif event == "stop":
            if payload.get("target") == "playback":
                self._tts_cancel.set()
            else:
                self.detector.stop_listening()
            await self._send_state()
        elif event == "media":
            await self._handle_audio(payload.get("audio"))
        elif event == "interrupt":
            self._tts_cancel.set()
            await self._send_state()

    async def _handle_audio(self, audio_data: str | None):
        if not audio_data:
            return
        audio = _decode_float32_audio(audio_data)
        if audio is None or len(audio) == 0:
            return
        for chunk in split_audio_into_chunks(audio, CHUNK_SIZE):
            events = self.detector.process_chunk(chunk)
            if events:
                await self._handle_events(events)
        await self._send_state()

    async def _handle_events(self, events: list[str]):
        if PipelineEvent.SPEECH_START in events:
            self._is_accumulating = True
            self._accumulated_text = ""

        for event in events:
            if event == PipelineEvent.TRANSCRIBE:
                await self._queue_transcription()
            elif event == PipelineEvent.RESPOND:
                await self._finalize_and_send()
            elif event in (
                PipelineEvent.SPEECH_START,
                PipelineEvent.SPEECH_END,
            ):
                await self._send_json({"event": event})

    async def _queue_transcription(self):
        segment = self.detector.current_segment
        if segment is None or len(segment) < SAMPLE_RATE * MIN_SEGMENT_DURATION:
            return
        self.detector.segment_count += 1
        try:
            self._transcription_queue.put_nowait(
                (self.detector.segment_count, segment),
            )
        except asyncio.QueueFull:
            logger.warning("Transcription queue full")

    async def _transcription_worker(self):
        from jarvis.transcriber import transcribe

        while True:
            segment_id, audio = await self._transcription_queue.get()
            try:
                stt_start = time.monotonic()
                async with self.resources.mlx_lock:
                    text = await asyncio.get_running_loop().run_in_executor(
                        None, transcribe, audio, SAMPLE_RATE,
                    )
                stt_latency = time.monotonic() - stt_start

                await self._send_json({
                    "event": "metrics",
                    "metrics": {"stt": {"latency": stt_latency}},
                })

                if text and text.strip():
                    logger.info(
                        "STT #%d (%.2fs): %s", segment_id, stt_latency, text,
                    )
                    if self._is_accumulating:
                        if self._accumulated_text:
                            self._accumulated_text += " " + text.strip()
                        else:
                            self._accumulated_text = text.strip()

                        await self._send_json({
                            "event": "text",
                            "role": "user",
                            "text": text.strip(),
                            "partial": True,
                        })
            except Exception:
                logger.exception("Transcription error")
            finally:
                self._transcription_queue.task_done()

    async def _finalize_and_send(self):
        await self._transcription_queue.join()
        self._is_accumulating = False

        if not self._accumulated_text:
            return

        raw_text = self._accumulated_text.strip()
        self._accumulated_text = ""

        if self._polish_enabled:
            from jarvis.polisher import polish

            async with self.resources.mlx_lock:
                text = await asyncio.get_running_loop().run_in_executor(
                    None, polish, raw_text,
                )
        else:
            text = _clean_transcript(raw_text)

        if raw_text != text:
            logger.debug("Cleaned: '%s' -> '%s'", raw_text, text)

        lower = text.lower().rstrip(".!?,")
        if lower in EXIT_PHRASES:
            logger.info("Exit phrase detected")
            self.detector.stop_listening()
            await self._send_json({"event": "stop", "target": "listening"})
            return

        if len(lower.split()) <= 1:
            logger.debug("Filtered single word: %s", text)
            return

        await self._send_json({
            "event": "text",
            "role": "user",
            "text": text,
            "complete": True,
        })

        logger.info("[voice] %s", text)
        await asyncio.get_running_loop().run_in_executor(
            None, _send_to_tmux, text, self.tmux_target,
        )

    async def _watch_tts_queue(self):
        """Poll for TTS text queued by voice output hook."""
        while True:
            try:
                if TTS_QUEUE.exists():
                    text = TTS_QUEUE.read_text().strip()
                    TTS_QUEUE.unlink(missing_ok=True)
                    if text:
                        self._tts_cancel.clear()
                        await self._send_tts(text)
            except asyncio.CancelledError:
                raise
            except OSError:
                pass
            await asyncio.sleep(0.2)

    async def _send_tts(self, text: str):
        """Render TTS and send WAV bytes over WebSocket."""
        from jarvis.speaker import get_sample_rate, render

        try:
            SPEAKING_FLAG.touch()

            tts_start = time.monotonic()
            async with self.resources.mlx_lock:
                audio = await asyncio.get_running_loop().run_in_executor(
                    None, render, text,
                )
            tts_latency = time.monotonic() - tts_start

            if audio is None or self._tts_cancel.is_set():
                return

            sr = get_sample_rate()
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            await self._send_json({
                "event": "metrics",
                "metrics": {"tts": {"latency": tts_latency}},
            })

            if not self._tts_cancel.is_set():
                await self._send_json({
                    "event": "media",
                    "mime": "audio/wav",
                    "audio": _encode_audio(wav_bytes),
                    "index": 0,
                })
                logger.info(
                    "TTS sent (%d bytes, %.2fs render)",
                    len(wav_bytes),
                    tts_latency,
                )

        except Exception:
            logger.exception("TTS failed")
        finally:
            SPEAKING_FLAG.unlink(missing_ok=True)

    async def _send_state(self):
        state = self.detector.get_state()
        await self._send_json({"event": "state", **state})

    async def _send_json(self, payload: dict):
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception:
            logger.debug("WebSocket send failed")


def create_app(tmux_target: str = "claude:0") -> FastAPI:
    """Create FastAPI application with shared pipeline resources."""
    _validate_tmux_target(tmux_target)
    resources = PipelineResources()
    app = FastAPI(title="Jarvis v3 Web")

    @app.get("/")
    async def index():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text())

    @app.get("/correlator.js")
    async def correlator():
        js_path = STATIC_DIR / "correlator.js"
        return Response(
            content=js_path.read_text(),
            media_type="application/javascript",
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info("Client connected")

        pipeline = WebVoicePipeline(websocket, resources, tmux_target)
        await pipeline.start()

        try:
            while True:
                message = await websocket.receive_text()
                await pipeline.handle_message(json.loads(message))
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("WebSocket error")
        finally:
            await pipeline.shutdown()

    return app


def run_web(
    tmux_target: str = "claude:0",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Entry point — start the web server."""
    import uvicorn

    app = create_app(tmux_target)
    logger.info("Starting Jarvis web server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
