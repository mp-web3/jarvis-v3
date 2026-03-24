"""Tests for jarvis.transcriber."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from jarvis.transcriber import DeepgramTranscriber, _float32_to_int16_bytes


class TestFloat32ToInt16Bytes:
    def test_silence(self):
        """Silence (zeros) converts to zero bytes."""
        audio = np.zeros(512, dtype=np.float32)
        result = _float32_to_int16_bytes(audio)
        assert len(result) == 512 * 2  # int16 = 2 bytes per sample
        assert all(b == 0 for b in result)

    def test_max_amplitude(self):
        """Full-scale signal clips correctly."""
        audio = np.ones(10, dtype=np.float32)
        result = _float32_to_int16_bytes(audio)
        values = np.frombuffer(result, dtype=np.int16)
        assert np.all(values == 32767)

    def test_negative_amplitude(self):
        """Negative full-scale clips correctly."""
        audio = np.full(10, -1.0, dtype=np.float32)
        result = _float32_to_int16_bytes(audio)
        values = np.frombuffer(result, dtype=np.int16)
        assert np.all(values == -32767)

    def test_round_trip_preserves_shape(self):
        """Output length is correct for various input sizes."""
        for size in [0, 1, 512, 4096]:
            audio = np.random.randn(size).astype(np.float32) * 0.5
            result = _float32_to_int16_bytes(audio)
            assert len(result) == size * 2


class TestDeepgramTranscriber:
    def test_init_raises_without_api_key(self):
        """Transcriber raises if DEEPGRAM_API_KEY is not set."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", ""):
            with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
                DeepgramTranscriber()

    def test_init_with_api_key(self):
        """Transcriber initializes with API key set."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()
            assert t._accumulated == ""
            assert t.ready_queue.empty()

    def test_flush_utterance(self):
        """Flushing puts accumulated text on the ready queue."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()
            t._accumulated = "Hello, how are you?"
            t._flush_utterance()
            assert t._accumulated == ""
            assert not t.ready_queue.empty()
            assert t.ready_queue.get_nowait() == "Hello, how are you?"

    def test_flush_utterance_empty(self):
        """Flushing empty text doesn't put anything on the queue."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()
            t._accumulated = "   "
            t._flush_utterance()
            assert t.ready_queue.empty()

    def test_handle_transcript_final(self):
        """Final transcripts are accumulated."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()
            # Cancel debounce tasks so they don't interfere
            t._restart_debounce = MagicMock()

            # Simulate a final transcript message
            msg = MagicMock()
            msg.type = "Results"
            msg.is_final = True
            alt = MagicMock()
            alt.transcript = "Hello world"
            msg.channel.alternatives = [alt]

            t._handle_transcript(msg)
            assert t._accumulated == "Hello world"

    def test_handle_transcript_accumulates(self):
        """Multiple final transcripts are concatenated."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()
            t._restart_debounce = MagicMock()

            for text in ["Hello", "world"]:
                msg = MagicMock()
                msg.type = "Results"
                msg.is_final = True
                alt = MagicMock()
                alt.transcript = text
                msg.channel.alternatives = [alt]
                t._handle_transcript(msg)

            assert t._accumulated == "Hello world"

    def test_handle_transcript_interim_not_accumulated(self):
        """Interim transcripts are not accumulated."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()

            msg = MagicMock()
            msg.type = "Results"
            msg.is_final = False
            alt = MagicMock()
            alt.transcript = "Hello"
            msg.channel.alternatives = [alt]

            t._handle_transcript(msg)
            assert t._accumulated == ""

    def test_handle_utterance_end(self):
        """Utterance end flushes accumulated text to ready queue."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()
            t._accumulated = "Can you help me with this?"
            t._handle_utterance_end()
            assert t._accumulated == ""
            assert t.ready_queue.get_nowait() == "Can you help me with this?"

    def test_on_message_routes_correctly(self):
        """_on_message dispatches to the right handler."""
        with patch("jarvis.transcriber.DEEPGRAM_API_KEY", "test-key"):
            t = DeepgramTranscriber()
            t._restart_debounce = MagicMock()

            # Test UtteranceEnd
            t._accumulated = "Some text"
            msg = MagicMock()
            msg.type = "UtteranceEnd"
            # UtteranceEnd messages don't have channel
            del msg.channel
            t._on_message(msg)
            assert t.ready_queue.get_nowait() == "Some text"
