"""Tests for jarvis.config."""

import os
from unittest.mock import patch

import pytest


def test_config_loads_defaults():
    """Config module provides expected default constants."""
    from jarvis.config import (
        CHANNELS,
        CHUNK_SIZE,
        SAMPLE_RATE,
        TTS_MODEL,
        TTS_VOICE,
        VAD_ALPHA,
        VAD_START_THRESHOLD,
    )

    assert SAMPLE_RATE == 16000
    assert CHANNELS == 1
    assert CHUNK_SIZE == 512
    assert TTS_MODEL == "mlx-community/Kokoro-82M-bf16"
    assert TTS_VOICE == "af_heart"
    assert 0 < VAD_ALPHA < 1
    assert 0 < VAD_START_THRESHOLD < 1


def test_deepgram_config_defaults():
    """Deepgram config has reasonable defaults."""
    from jarvis.config import (
        DEEPGRAM_INTERIM_RESULTS,
        DEEPGRAM_LANGUAGE,
        DEEPGRAM_MODEL,
        DEEPGRAM_UTTERANCE_END_MS,
    )

    assert DEEPGRAM_MODEL in ("nova-3", "flux-general-en")
    assert DEEPGRAM_LANGUAGE in ("en", "en-US")
    assert isinstance(DEEPGRAM_UTTERANCE_END_MS, int)
    assert DEEPGRAM_UTTERANCE_END_MS > 0
    assert isinstance(DEEPGRAM_INTERIM_RESULTS, bool)


def test_deepgram_api_key_from_env():
    """DEEPGRAM_API_KEY reads from environment."""
    # The module reads os.environ at import time, so we test the mechanism
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    from jarvis.config import DEEPGRAM_API_KEY
    assert DEEPGRAM_API_KEY == key


def test_get_config_returns_dict():
    """get_config returns a dict (from yaml or empty)."""
    from jarvis.config import get_config
    config = get_config()
    assert isinstance(config, dict)


def test_get_config_has_expected_sections():
    """config.yaml should have the expected top-level sections."""
    from jarvis.config import get_config
    config = get_config()
    # These sections exist in config.yaml
    assert "stt" in config
    assert "tts" in config
    assert "listener" in config
