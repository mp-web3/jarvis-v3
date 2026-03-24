"""OpenAI Realtime API — managed STT + TTS for Jarvis v3.

Replaces the local MLX pipeline:
  - Silero VAD (vad.py)        → server-side VAD in Realtime API
  - Parakeet TDT (transcriber.py) → gpt-4o-transcribe via WebSocket
  - Kokoro TTS (speaker.py)    → OpenAI /v1/audio/speech

API ref: https://platform.openai.com/docs/guides/realtime-transcription
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import re
from math import gcd

import numpy as np

logger = logging.getLogger(__name__)

# STT — OpenAI Realtime transcription WebSocket
_STT_WS_URL = "wss://api.openai.com/v1/realtime"
_STT_MODEL = "gpt-4o-transcribe"

# TTS — OpenAI Audio API
_TTS_URL = "https://api.openai.com/v1/audio/speech"
_TTS_MODEL = "tts-1"
_TTS_DEFAULT_VOICE = "nova"
_TTS_SAMPLE_RATE = 24000  # OpenAI TTS native output rate

# Acknowledgment phrases pre-rendered at startup
_ACK_PHRASES = [
    "Okay.",
    "Right.",
    "Got it.",
    "Let me check.",
    "One moment.",
    "Sure.",
    "On it.",
]

# Markdown sanitisation rules for TTS (ported from speaker.py)
_TTS_SANITIZE = [
    (re.compile(r"```[\s\S]*?```"), ""),
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"\*(.+?)\*"), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    (re.compile(r"^[-*]\s+", re.MULTILINE), ""),
    (re.compile(r"^\d+\.\s+", re.MULTILINE), ""),
    (re.compile(r"^\|?[-:|]+\|[-:|]+\|?\s*$", re.MULTILINE), ""),
    (re.compile(r"\|"), ", "),
    (re.compile(r",\s*,"), ","),
    (re.compile(r"[→←]"), ""),
    (re.compile(r"—"), ", "),
    (re.compile(r"--+"), ", "),
    (re.compile(r"https?://\S+"), ""),
    (re.compile(r"^\s*,\s*", re.MULTILINE), ""),
    (re.compile(r",\s*$", re.MULTILINE), ""),
    (re.compile(r"\s{2,}"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def sanitize_for_tts(text: str) -> str:
    """Strip markdown and special characters so TTS reads naturally."""
    for pattern, replacement in _TTS_SANITIZE:
        text = pattern.sub(replacement, text)
    return text.strip()


def render_tts(text: str, voice: str = _TTS_DEFAULT_VOICE) -> tuple[np.ndarray, int]:
    """Render text via OpenAI TTS API.

    Returns (float32 audio, sample_rate). Runs synchronously — call in executor.
    """
    import httpx
    import soundfile as sf

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    resp = httpx.post(
        _TTS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": _TTS_MODEL, "input": text, "voice": voice, "response_format": "wav"},
        timeout=30.0,
    )
    resp.raise_for_status()

    audio, sr = sf.read(io.BytesIO(resp.content))
    return audio.astype(np.float32), int(sr)


def resample_for_device(audio: np.ndarray, src_sr: int) -> tuple[np.ndarray, int]:
    """Resample audio to match the default sounddevice output device rate."""
    import sounddevice as sd
    from scipy.signal import resample_poly

    out_dev = sd.default.device[1] if sd.default.device[1] is not None else sd.default.device
    device_rate = int(sd.query_devices(out_dev, "output")["default_samplerate"])
    if device_rate == src_sr:
        return audio, src_sr
    g = gcd(device_rate, src_sr)
    resampled = resample_poly(audio, device_rate // g, src_sr // g).astype(np.float32)
    return resampled, device_rate


def preload_acknowledgments(voice: str = _TTS_DEFAULT_VOICE) -> list[tuple[np.ndarray, int]]:
    """Pre-render acknowledgment phrases via TTS API.

    Returns list of (float32 audio, sample_rate) tuples, resampled for device.
    """
    acks: list[tuple[np.ndarray, int]] = []
    for phrase in _ACK_PHRASES:
        try:
            audio, sr = render_tts(phrase, voice=voice)
            audio, sr = resample_for_device(audio, sr)
            acks.append((audio, sr))
        except Exception:
            logger.warning("Failed to pre-render ack '%s', skipping", phrase)
    logger.info("Pre-rendered %d/%d acknowledgments", len(acks), len(_ACK_PHRASES))
    return acks


def get_random_ack(cache: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int] | None:
    """Return a random pre-rendered acknowledgment clip, or None if cache is empty."""
    return random.choice(cache) if cache else None


class RealtimeSTT:
    """Streaming STT via OpenAI Realtime transcription API with server-side VAD.

    Usage::

        stt = RealtimeSTT(vad_threshold=0.5, silence_ms=500)
        await stt.connect()
        receive_task = asyncio.create_task(stt.receive_loop())

        # Send mic audio
        await stt.send_audio(chunk)  # float32 16kHz mono numpy array

        # Read events from stt.event_queue:
        #   {"type": "speech_started"}
        #   {"type": "speech_stopped"}
        #   {"type": "transcript", "text": "..."}
        #   {"type": "error", "message": "..."}

        receive_task.cancel()
        await stt.close()
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        silence_ms: int = 500,
        prefix_padding_ms: int = 300,
    ):
        self._vad_threshold = vad_threshold
        self._silence_ms = silence_ms
        self._prefix_padding_ms = prefix_padding_ms
        self._ws = None
        self._running = False
        self.event_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def connect(self) -> None:
        """Open WebSocket and configure transcription session."""
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets package required: uv pip install websockets")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        url = f"{_STT_WS_URL}?intent=transcription"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        self._ws = await websockets.connect(url, additional_headers=headers)
        self._running = True

        # Configure transcription session with server-side VAD
        await self._ws.send(json.dumps({
            "type": "transcription_session.update",
            "session": {
                "model": _STT_MODEL,
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": _STT_MODEL},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self._vad_threshold,
                    "prefix_padding_ms": self._prefix_padding_ms,
                    "silence_duration_ms": self._silence_ms,
                },
            },
        }))
        logger.info("Realtime API connected (model=%s, silence=%dms)", _STT_MODEL, self._silence_ms)

    async def send_audio(self, chunk: np.ndarray) -> None:
        """Send a float32 16kHz mono chunk to the API."""
        if not self._ws or not self._running:
            return
        pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        b64 = base64.b64encode(pcm16.tobytes()).decode()
        try:
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64,
            }))
        except Exception:
            logger.warning("send_audio failed — connection may have dropped")

    async def receive_loop(self) -> None:
        """Process incoming WebSocket events; push normalised dicts to event_queue.

        Run as an asyncio task. Exits cleanly when the connection closes or
        self._running is set to False.
        """
        try:
            async for raw in self._ws:
                if not self._running:
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                if etype == "input_audio_buffer.speech_started":
                    await self.event_queue.put({"type": "speech_started"})

                elif etype == "input_audio_buffer.speech_stopped":
                    await self.event_queue.put({"type": "speech_stopped"})

                elif etype == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "").strip()
                    if text:
                        await self.event_queue.put({"type": "transcript", "text": text})

                elif etype == "error":
                    msg = str(event.get("error", event))
                    logger.error("Realtime API error: %s", msg)
                    await self.event_queue.put({"type": "error", "message": msg})

                # session.created / session.updated / other events — silently ignored

        except Exception:
            if self._running:
                logger.exception("Realtime receive_loop error")

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
