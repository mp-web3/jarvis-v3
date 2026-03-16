"""Jarvis v3 configuration — loaded from config.yaml with defaults."""

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_config: dict | None = None


def get_config() -> dict:
    global _config
    if _config is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH) as f:
                _config = yaml.safe_load(f)
        else:
            _config = {}
    return _config


# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512  # 32ms at 16kHz (matches offline-voice-ai)

# VAD — Silero (generic, always-on)
SILERO_VAD_MODEL = "models/silero_vad.onnx"
VAD_ALPHA = 0.1
VAD_START_THRESHOLD = 0.3
VAD_SPEAKING_THRESHOLD = 0.5
VAD_STOP_THRESHOLD = 0.3
VAD_QUIET_THRESHOLD = 0.05
VAD_STATE_SHAPE = (2, 1, 128)
VAD_CONTEXT_SIZE = 64

# pVAD — FireRedChat (personalized, speaker-verified)
PVAD_MODEL_DIR = "models/pvad"
PVAD_ACTIVATION_THRESHOLD = 0.85
PVAD_MIN_SPEECH_FRAMES = 10
PVAD_MIN_SILENCE_FRAMES = 20

# End-of-utterance detection (SmartTurn)
EOU_MODEL = "models/smart_turn_v3.onnx"
EOU_MIN_SAMPLES = 4 * SAMPLE_RATE
EOU_OPTIMAL_SAMPLES = 8 * SAMPLE_RATE
EOU_CONFIDENCE_THRESHOLD = 0.9

# Speech segmentation
SAFETY_CHUNKS_BEFORE = 4
MIN_SEGMENT_DURATION = 0.3

# STT
STT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# TTS
TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0
TTS_SAMPLE_RATE = 24000

# LLM sentence streaming
LLM_SENTENCE_DELIMITERS = ".!?"
LLM_MIN_TOKENS_FOR_TTS = 3

# Processing
MAX_TRANSCRIPTION_QUEUE_SIZE = 256
