"""Jarvis v3 configuration — loaded from config.yaml with defaults."""

import os
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
CHUNK_SIZE = 512  # 32ms at 16kHz

# Deepgram STT
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = get_config().get("stt", {}).get("model", "nova-3")
DEEPGRAM_LANGUAGE = get_config().get("stt", {}).get("language", "en")
DEEPGRAM_UTTERANCE_END_MS = int(
    get_config().get("stt", {}).get("utterance_end_ms", 1000)
)
DEEPGRAM_INTERIM_RESULTS = get_config().get("stt", {}).get("interim_results", True)

# VAD — Silero (for barge-in during TTS only)
SILERO_VAD_MODEL = "models/silero_vad.onnx"
VAD_ALPHA = 0.1
VAD_START_THRESHOLD = 0.3
VAD_STATE_SHAPE = (2, 1, 128)
VAD_CONTEXT_SIZE = 64

# TTS
TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"
TTS_SPEED = float(get_config().get("tts", {}).get("speed", 1.2))
TTS_SAMPLE_RATE = 24000
