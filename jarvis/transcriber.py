"""STT using Parakeet TDT on Apple Silicon via MLX.

Transducer architecture — outputs blanks on silence, cannot hallucinate.
Singleton model with explicit preload for Metal GPU safety.
"""

import logging
import tempfile

import numpy as np

logger = logging.getLogger(__name__)

_model = None


def _get_model(model_name: str | None = None):
    global _model
    if _model is None:
        from parakeet_mlx import from_pretrained
        from jarvis.config import STT_MODEL

        name = model_name or STT_MODEL
        logger.info("Loading Parakeet model: %s", name)
        _model = from_pretrained(name)
        logger.info("Parakeet model ready")
    return _model


def preload(model_name: str | None = None):
    """Pre-load model on main thread for safe Metal GPU init."""
    _get_model(model_name)


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe audio to text. Returns empty string if no speech."""
    import soundfile as sf

    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, sample_rate)
        result = model.transcribe(f.name)

    text = result.text.strip() if result.text else ""
    if text:
        logger.info("Transcribed: %s", text[:80])
    return text
