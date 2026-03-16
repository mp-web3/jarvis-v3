"""TTS using mlx-audio Kokoro — pure MLX, no PyTorch GPU contention.

Singleton model with explicit preload. Resamples to device native rate.
Bilingual: English + Italian.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_sample_rate = 24000


def _get_model(model_name: str | None = None):
    global _model, _sample_rate
    if _model is None:
        from mlx_audio.tts.utils import load_model
        from jarvis.config import TTS_MODEL, TTS_SAMPLE_RATE

        name = model_name or TTS_MODEL
        logger.info("Loading TTS model: %s", name)
        _model = load_model(name)
        _sample_rate = TTS_SAMPLE_RATE
        logger.info("TTS model ready")
    return _model


def preload(model_name: str | None = None):
    """Pre-load model and warm up pipeline (avoids first-call latency)."""
    from jarvis.config import TTS_VOICE

    model = _get_model(model_name)
    for _ in model.generate(".", voice=TTS_VOICE, speed=1.0, lang_code="a"):
        pass
    logger.info("TTS pipeline warmed up")


def get_sample_rate() -> int:
    """Return actual playback sample rate (resampled if device differs)."""
    import sounddevice as sd

    out_dev = sd.default.device[1] if sd.default.device[1] is not None else sd.default.device
    device_rate = int(sd.query_devices(out_dev, "output")["default_samplerate"])
    return device_rate if device_rate != _sample_rate else _sample_rate


def render(
    text: str,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
    lang: str | None = None,
) -> np.ndarray | None:
    """Render text to PCM float32 audio. Returns None on failure."""
    import mlx.core as mx
    from jarvis.config import get_config

    model = _get_model()

    # Bilingual voice selection
    if lang is not None:
        config = get_config()
        voices_cfg = config.get("tts", {}).get("voices", {})
        if lang in voices_cfg:
            voice = voices_cfg[lang].get("voice", voice)
            lang_code = voices_cfg[lang].get("lang_code", lang_code)

    segments = []
    for result in model.generate(text, voice=voice, speed=speed, lang_code=lang_code):
        audio = result.audio
        if isinstance(audio, mx.array):
            audio = np.array(audio)
        segments.append(audio.flatten())

    if not segments:
        logger.warning("TTS produced no audio for: %s", text[:60])
        return None

    audio = np.concatenate(segments).astype(np.float32)

    # Resample to device native rate (48kHz for AirPods)
    import sounddevice as sd

    out_dev = sd.default.device[1] if sd.default.device[1] is not None else sd.default.device
    device_rate = int(sd.query_devices(out_dev, "output")["default_samplerate"])
    if device_rate != _sample_rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(device_rate, _sample_rate)
        audio = resample_poly(audio, device_rate // g, _sample_rate // g).astype(np.float32)

    return audio


def speak(text: str, lang: str = "en"):
    """Render and play TTS audio (blocking)."""
    import sounddevice as sd

    audio = render(text, lang=lang)
    if audio is None:
        return
    sd.play(audio, samplerate=get_sample_rate())
    sd.wait()
