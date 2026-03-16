"""Audio buffering for speech segments.

Pre-buffer captures audio before VAD triggers (ring buffer equivalent).
Active segment accumulates during speech.

Adapted from offline-voice-ai's AudioBuffer.
"""

from typing import Optional

import numpy as np

from jarvis.config import CHUNK_SIZE, SAFETY_CHUNKS_BEFORE, SAMPLE_RATE
from jarvis.vad import SpeechState


class AudioBuffer:
    """Manages audio buffering with safety margins for segment capture."""

    def __init__(self):
        self.pre_buffer: list[np.ndarray] = []
        self.active_segment: list[np.ndarray] = []
        self.is_capturing = False

    def add_chunk(self, chunk: np.ndarray, state: SpeechState):
        chunk = chunk.copy()

        if state == SpeechState.QUIET:
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)

        elif state == SpeechState.STARTING:
            if not self.is_capturing:
                self.is_capturing = True
                self.active_segment = self.pre_buffer.copy()
            self.active_segment.append(chunk)
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)

        elif state in (SpeechState.SPEAKING, SpeechState.STOPPING):
            self.active_segment.append(chunk)

    def get_segment(self) -> Optional[np.ndarray]:
        if not self.active_segment:
            return None
        segment = np.concatenate(self.active_segment)
        self.active_segment = []
        self.is_capturing = False
        return segment


def split_audio_into_chunks(
    audio: np.ndarray, chunk_size: int = CHUNK_SIZE
) -> list[np.ndarray]:
    num_chunks = len(audio) // chunk_size
    return [audio[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
