"""Voice pipeline orchestrator — the core of Jarvis v3.

Combines:
- offline-voice-ai: 4-state VAD machine, asyncio.Lock for MLX, Event-based cancellation,
  transcription queue, sentence-streaming TTS
- jarvis-v2: Parakeet TDT (no hallucinations), pVAD speaker verification,
  tmux integration, bilingual TTS, barge-in

This module handles the web (WebSocket) pipeline. The tmux CLI listener
reuses the same SpeechDetector and pipeline resources.
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from jarvis.audio_buffer import AudioBuffer, split_audio_into_chunks
from jarvis.config import (
    MAX_TRANSCRIPTION_QUEUE_SIZE,
    MIN_SEGMENT_DURATION,
    SAMPLE_RATE,
    CHUNK_SIZE,
    VAD_START_THRESHOLD,
    VAD_SPEAKING_THRESHOLD,
    VAD_STOP_THRESHOLD,
    VAD_QUIET_THRESHOLD,
    EOU_CONFIDENCE_THRESHOLD,
)
from jarvis.vad import EndOfUtteranceDetector, SileroVAD, SpeechState

logger = logging.getLogger(__name__)


class PipelineEvent:
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TRANSCRIBE = "transcribe"
    RESPOND = "respond"


class PipelineResources:
    """Global resources initialized once. Singleton per process."""

    def __init__(self):
        from jarvis.transcriber import preload as preload_stt
        from jarvis.speaker import preload as preload_tts
        from jarvis.polisher import preload as preload_polish

        logger.info("Initializing pipeline resources...")

        # Pre-load models on main thread (Metal GPU safety)
        preload_stt()
        preload_tts()
        preload_polish()

        # MLX lock — prevents concurrent Parakeet STT + Kokoro TTS on Metal
        self.mlx_lock = asyncio.Lock()

        logger.info("Pipeline resources ready")


class SpeechDetector:
    """4-state VAD machine with audio segmentation and EOU detection.

    State machine: QUIET -> STARTING -> SPEAKING -> STOPPING -> QUIET

    3-layer turn detection:
    1. Silero VAD — detects speech start/stop (~30ms)
    2. SmartTurn v3 — audio-level end-of-utterance (~12ms)
    3. STT semantic — checks transcribed text for completeness (zero cost)
    """

    # Words that signal the speaker hasn't finished
    _INCOMPLETE_ENDINGS = frozenset({
        "and", "but", "or", "so", "because", "that", "which", "who",
        "where", "when", "if", "while", "the", "a", "an", "to", "of",
        "in", "with", "for", "on", "at", "by", "from", "is", "are",
        "was", "were", "not", "then", "also", "just", "about",
    })

    def __init__(self, text_provider=None):
        self.vad = SileroVAD()
        self.eou = EndOfUtteranceDetector()
        self.buffer = AudioBuffer()

        self.state = SpeechState.QUIET
        self.is_listening = False
        self.is_responding = False
        self.user_speaking = False

        self.segment_count = 0
        self.current_segment: Optional[np.ndarray] = None

        # Layer 3: STT semantic turn detection
        self._text_provider = text_provider  # Callable[[], str] | None
        self._semantic_wait_chunks = 0
        self._semantic_max_wait = 30  # ~1s at 30ms/chunk

    def start_listening(self):
        self.is_listening = True
        self.user_speaking = False
        logger.info("Detector started")

    def stop_listening(self):
        self.is_listening = False
        self.user_speaking = False
        logger.info("Detector stopped (segments: %d)", self.segment_count)

    def process_chunk(self, chunk: np.ndarray) -> list[str]:
        if not self.is_listening:
            return []

        vad_prob = self.vad.process_chunk(chunk)
        self.buffer.add_chunk(chunk, self.state)

        if self.eou.available:
            self.eou.add_audio(chunk)

        return self._update_state(vad_prob)

    def _update_state(self, vad_prob: float) -> list[str]:
        events = []
        prev_state = self.state

        if self.state == SpeechState.QUIET:
            if vad_prob >= VAD_START_THRESHOLD:
                self.state = SpeechState.STARTING
                self.user_speaking = True
                events.append(PipelineEvent.SPEECH_START)

        elif self.state == SpeechState.STARTING:
            if vad_prob >= VAD_SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
            elif vad_prob < VAD_QUIET_THRESHOLD:
                self.state = SpeechState.QUIET
                self.user_speaking = False

        elif self.state == SpeechState.SPEAKING:
            if vad_prob < VAD_STOP_THRESHOLD:
                self.state = SpeechState.STOPPING
                self.current_segment = self.buffer.get_segment()
                if self.current_segment is not None:
                    events.append(PipelineEvent.TRANSCRIBE)

        elif self.state == SpeechState.STOPPING:
            vad_quiet = vad_prob < VAD_QUIET_THRESHOLD

            eou_confirms = not self.eou.available
            if self.eou.available and vad_quiet and self.eou.has_enough_audio():
                result = self.eou.detect()
                eou_confirms = result["ended"] and result["confidence"] > EOU_CONFIDENCE_THRESHOLD
                if eou_confirms:
                    logger.info("EOU detected (conf: %.2f)", result["confidence"])

            if vad_quiet and eou_confirms:
                # Layer 3: semantic check — is the transcribed text complete?
                should_wait = False
                if self._text_provider and not self._text_complete():
                    if self._semantic_wait_chunks < self._semantic_max_wait:
                        self._semantic_wait_chunks += 1
                        if self._semantic_wait_chunks == 1:
                            text = self._text_provider() or ""
                            logger.info(
                                "Semantic: text incomplete, extending wait: '...%s'",
                                text[-50:],
                            )
                        should_wait = True
                    else:
                        logger.info("Semantic: wait timeout, sending anyway")

                if not should_wait:
                    self._semantic_wait_chunks = 0
                    self.state = SpeechState.QUIET
                    events.append(PipelineEvent.SPEECH_END)
                    if self.user_speaking:
                        events.append(PipelineEvent.RESPOND)
                        self.user_speaking = False
                    if self.eou.available:
                        self.eou.reset()
                    self.current_segment = None
            elif vad_prob > VAD_SPEAKING_THRESHOLD:
                # User resumed speaking (just a pause)
                self.state = SpeechState.SPEAKING
                self.current_segment = None
                self._semantic_wait_chunks = 0

        if prev_state != self.state:
            logger.debug(
                "State: %s -> %s (vad: %.3f)", prev_state.value, self.state.value, vad_prob
            )

        return events

    def _text_complete(self) -> bool:
        """Layer 3: check if accumulated text looks like a complete utterance."""
        if not self._text_provider:
            return True

        text = (self._text_provider() or "").strip()
        if not text:
            return False  # No text yet — wait for transcription

        # Ends with sentence-ending punctuation → definitely complete
        if text[-1] in ".!?":
            return True

        words = text.split()

        # Very short fragment (<3 words) without punctuation — likely incomplete
        if len(words) < 3:
            return False

        # Ends with conjunction/preposition/article → likely mid-sentence
        last_word = words[-1].lower().rstrip(",;:")
        if last_word in self._INCOMPLETE_ENDINGS:
            return False

        # No clear signal either way — assume complete
        # (Parakeet sometimes omits final punctuation on complete sentences)
        return True

    def get_state(self) -> dict:
        return {
            "state": self.state.value,
            "vad_prob": float(self.vad.smoothed_prob),
            "segments": self.segment_count,
            "listening": self.is_listening,
            "responding": self.is_responding,
        }
