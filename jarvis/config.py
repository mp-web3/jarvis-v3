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

# VAD thresholds — used for local barge-in energy gating
VAD_START_THRESHOLD = 0.3
VAD_SPEAKING_THRESHOLD = 0.5
VAD_STOP_THRESHOLD = 0.3
VAD_QUIET_THRESHOLD = 0.05

# Speech segmentation
SAFETY_CHUNKS_BEFORE = 4
MIN_SEGMENT_DURATION = 0.3

# OpenAI Realtime API — STT (server-side VAD + gpt-4o-transcribe)
OPENAI_STT_MODEL: str = get_config().get("stt", {}).get("model", "gpt-4o-transcribe")
OPENAI_VAD_THRESHOLD: float = float(
    get_config().get("listener", {}).get("vad_threshold", 0.5)
)
OPENAI_VAD_SILENCE_MS: int = int(
    get_config().get("listener", {}).get("silence_ms", 500)
)
OPENAI_VAD_PREFIX_PADDING_MS: int = 300

# OpenAI TTS (/v1/audio/speech)
OPENAI_TTS_MODEL: str = get_config().get("tts", {}).get("model", "tts-1")
OPENAI_TTS_VOICE: str = get_config().get("tts", {}).get("voice", "nova")
OPENAI_TTS_SAMPLE_RATE = 24000

# LLM sentence streaming
LLM_SENTENCE_DELIMITERS = ".!?"
LLM_MIN_TOKENS_FOR_TTS = 3

# Processing
MAX_TRANSCRIPTION_QUEUE_SIZE = 256
