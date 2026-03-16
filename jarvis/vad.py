"""Voice Activity Detection — dual mode.

Silero VAD (generic, ONNX, CPU) for always-on detection.
pVAD (FireRedChat, ONNX, CPU) for speaker-verified detection when enrollment exists.

Adapted from:
- offline-voice-ai: Silero VAD + 4-state machine + EOU detection
- jarvis-v2: pVAD with speaker embedding
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from jarvis.config import (
    EOU_CONFIDENCE_THRESHOLD,
    EOU_MIN_SAMPLES,
    EOU_MODEL,
    EOU_OPTIMAL_SAMPLES,
    PVAD_ACTIVATION_THRESHOLD,
    PVAD_MIN_SILENCE_FRAMES,
    PVAD_MIN_SPEECH_FRAMES,
    PVAD_MODEL_DIR,
    SAMPLE_RATE,
    SILERO_VAD_MODEL,
    VAD_ALPHA,
    VAD_CONTEXT_SIZE,
    VAD_STATE_SHAPE,
)

logger = logging.getLogger(__name__)

PVAD_FRAME_SAMPLES = 160  # 10ms at 16kHz


class SpeechState(str, Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"


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


class PersonalizedVAD:
    """Speaker-verified VAD using FireRedChat pVAD ONNX model.

    Falls back to generic mode (zeros embedding) without enrollment.
    """

    def __init__(
        self,
        speaker_embedding: Optional[np.ndarray] = None,
        model_dir: str = PVAD_MODEL_DIR,
        threshold: float = PVAD_ACTIVATION_THRESHOLD,
    ):
        resolved = Path(model_dir)
        if not resolved.is_absolute():
            resolved = Path(__file__).parent.parent / model_dir
        model_path = resolved / "pvad.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"pVAD model not found at {model_path}. "
                "Download from HuggingFace scmkrd/FireRedTTS-1DGPT"
            )

        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._spkemb = (
            speaker_embedding
            if speaker_embedding is not None
            else np.zeros((1, 192), dtype=np.float32)
        )
        self._threshold = threshold

        self._mel_buffer = np.zeros((1, 80, 15), dtype=np.float32)
        self._gru_buffer = np.zeros((2, 1, 256), dtype=np.float32)
        self._speech_count = 0
        self._silence_count = 0
        self._speaking = False
        self._ema_prob = 0.0
        self._ema_alpha = 0.8
        self._min_speech = PVAD_MIN_SPEECH_FRAMES
        self._min_silence = PVAD_MIN_SILENCE_FRAMES

        has_enrollment = speaker_embedding is not None
        logger.info(
            "pVAD loaded (enrollment=%s, threshold=%.2f)", has_enrollment, threshold
        )

    def process_chunk(self, chunk: np.ndarray) -> bool:
        audio_int16 = (chunk * np.iinfo(np.int16).max).astype(np.float32)
        for i in range(0, len(audio_int16) - PVAD_FRAME_SAMPLES + 1, PVAD_FRAME_SAMPLES):
            frame = audio_int16[i : i + PVAD_FRAME_SAMPLES].reshape(1, PVAD_FRAME_SAMPLES)
            outputs = self._session.run(
                None,
                {
                    "input_audio": frame,
                    "spkemb": self._spkemb,
                    "mel_buffer": self._mel_buffer,
                    "gru_buffer": self._gru_buffer,
                },
            )
            raw_prob = outputs[1][0].tolist()[0]
            self._mel_buffer = outputs[2]
            self._gru_buffer = outputs[3]

            self._ema_prob = self._ema_alpha * raw_prob + (1 - self._ema_alpha) * self._ema_prob
            if self._ema_prob >= self._threshold:
                self._speech_count += 1
                self._silence_count = 0
                if not self._speaking and self._speech_count >= self._min_speech:
                    self._speaking = True
            else:
                self._silence_count += 1
                self._speech_count = 0
                if self._speaking and self._silence_count >= self._min_silence:
                    self._speaking = False

        return self._speaking

    @property
    def probability(self) -> float:
        return self._ema_prob

    def reset(self):
        self._mel_buffer = np.zeros((1, 80, 15), dtype=np.float32)
        self._gru_buffer = np.zeros((2, 1, 256), dtype=np.float32)
        self._speech_count = 0
        self._silence_count = 0
        self._speaking = False
        self._ema_prob = 0.0


class EndOfUtteranceDetector:
    """Detect end of utterance using SmartTurn ONNX model.

    Smarter than fixed silence timers — uses ML to determine
    if the user has finished their thought even during pauses.
    """

    def __init__(self, model_path: str = EOU_MODEL):
        resolved = Path(model_path)
        if not resolved.is_absolute():
            resolved = Path(__file__).parent.parent / model_path

        if not resolved.exists():
            logger.warning("EOU model not found at %s — disabled", resolved)
            self._session = None
            self._extractor = None
            self.audio_buffer = np.array([], dtype=np.float32)
            return

        from transformers import WhisperFeatureExtractor

        self._extractor = WhisperFeatureExtractor(chunk_length=8)

        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(resolved), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.audio_buffer = np.array([], dtype=np.float32)
        logger.info("EOU detector loaded from %s", resolved)

    @property
    def available(self) -> bool:
        return self._session is not None

    def add_audio(self, chunk: np.ndarray):
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        if len(self.audio_buffer) > EOU_OPTIMAL_SAMPLES:
            self.audio_buffer = self.audio_buffer[-EOU_OPTIMAL_SAMPLES:]

    def has_enough_audio(self) -> bool:
        return len(self.audio_buffer) >= EOU_MIN_SAMPLES

    def detect(self) -> dict:
        if not self.available or not self.has_enough_audio():
            return {"ended": False, "confidence": 0.0}
        try:
            audio_length = min(len(self.audio_buffer), EOU_OPTIMAL_SAMPLES)
            audio = self.audio_buffer[-audio_length:]
            inputs = self._extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=EOU_OPTIMAL_SAMPLES,
                truncation=True,
                do_normalize=True,
            )
            features = np.expand_dims(
                inputs.input_features.squeeze(0).astype(np.float32), axis=0
            )
            outputs = self._session.run(None, {"input_features": features})
            confidence = float(outputs[0][0].item())
            return {
                "ended": confidence > EOU_CONFIDENCE_THRESHOLD,
                "confidence": confidence,
            }
        except Exception as e:
            logger.exception("EOU detection error")
            return {"ended": False, "confidence": 0.0}

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
