"""Voice Activity Detection — Silero VAD for barge-in during TTS.

With Deepgram handling STT + VAD + turn detection, Silero VAD is only
needed to detect speech during TTS playback for barge-in cancellation.
"""

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from jarvis.config import (
    SAMPLE_RATE,
    SILERO_VAD_MODEL,
    VAD_ALPHA,
    VAD_CONTEXT_SIZE,
    VAD_STATE_SHAPE,
)

logger = logging.getLogger(__name__)


class SileroVAD:
    """Generic VAD using Silero ONNX model."""

    def __init__(self, model_path: str = SILERO_VAD_MODEL):
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = Path(__file__).parent.parent / model_path
        if not resolved.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {resolved}")

        self.session = ort.InferenceSession(
            str(resolved), providers=["CPUExecutionProvider"]
        )
        self.state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros((1, VAD_CONTEXT_SIZE), dtype=np.float32)
        self.smoothed_prob = 0.0
        logger.info("Silero VAD loaded from %s", resolved)

    def process_chunk(self, chunk: np.ndarray) -> float:
        audio_input = np.concatenate([self.context, chunk.reshape(1, -1)], axis=1)
        output, self.state = self.session.run(
            None,
            {
                "input": audio_input,
                "state": self.state,
                "sr": np.array([SAMPLE_RATE], dtype=np.int64),
            },
        )
        self.context = audio_input[:, -VAD_CONTEXT_SIZE:]
        raw_prob = float(output[0][0])
        self.smoothed_prob = VAD_ALPHA * raw_prob + (1.0 - VAD_ALPHA) * self.smoothed_prob
        return self.smoothed_prob

    def reset(self):
        self.state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros((1, VAD_CONTEXT_SIZE), dtype=np.float32)
        self.smoothed_prob = 0.0
