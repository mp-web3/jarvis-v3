"""Tests for jarvis.vad (SileroVAD for barge-in)."""

import numpy as np
import pytest

from jarvis.config import CHUNK_SIZE, SAMPLE_RATE


class TestSileroVAD:
    @pytest.fixture
    def vad(self):
        """Create a SileroVAD instance (requires model file)."""
        from jarvis.vad import SileroVAD
        return SileroVAD()

    def test_init(self, vad):
        """SileroVAD initializes with zero state."""
        assert vad.smoothed_prob == 0.0

    def test_process_silence(self, vad):
        """Silent audio returns low probability."""
        chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        prob = vad.process_chunk(chunk)
        assert 0.0 <= prob <= 1.0
        assert prob < 0.1  # silence should be low

    def test_process_noise(self, vad):
        """Random noise returns a probability (may not be speech)."""
        rng = np.random.default_rng(42)
        chunk = rng.standard_normal(CHUNK_SIZE).astype(np.float32) * 0.01
        prob = vad.process_chunk(chunk)
        assert 0.0 <= prob <= 1.0

    def test_reset(self, vad):
        """Reset clears state to zero."""
        chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        vad.process_chunk(chunk)
        vad.reset()
        assert vad.smoothed_prob == 0.0

    def test_smoothing(self, vad):
        """Probability is smoothed (doesn't jump instantly)."""
        silence = np.zeros(CHUNK_SIZE, dtype=np.float32)
        # Process several silent chunks
        probs = []
        for _ in range(10):
            probs.append(vad.process_chunk(silence))
        # All should be low and relatively stable
        assert all(p < 0.1 for p in probs)

    def test_missing_model_raises(self):
        """Missing model file raises FileNotFoundError."""
        from jarvis.vad import SileroVAD
        with pytest.raises(FileNotFoundError):
            SileroVAD(model_path="/nonexistent/model.onnx")
