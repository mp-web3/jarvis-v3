"""STT using Deepgram streaming WebSocket.

Replaces local Parakeet TDT with cloud Deepgram API. Handles connection
lifecycle, audio streaming, transcript accumulation, and utterance-end detection.
"""

import asyncio
import logging

import numpy as np

from jarvis.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_INTERIM_RESULTS,
    DEEPGRAM_LANGUAGE,
    DEEPGRAM_MODEL,
    DEEPGRAM_UTTERANCE_END_MS,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


def _float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 audio [-1, 1] to int16 PCM bytes for Deepgram."""
    int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return int16.tobytes()


class DeepgramTranscriber:
    """Streams mic audio to Deepgram and yields complete utterances.

    Accumulates is_final transcript segments. When Deepgram signals utterance_end
    (or a debounce timeout fires), the accumulated text is pushed to ready_queue.
    """

    def __init__(self, debounce_s: float = 0.5):
        if not DEEPGRAM_API_KEY:
            raise RuntimeError(
                "DEEPGRAM_API_KEY not set. Export it in your shell: "
                "export DEEPGRAM_API_KEY='your-key'"
            )

        self._debounce_s = debounce_s
        self._connection = None
        self._ctx = None
        self._connected = False

        # Transcript accumulation
        self._accumulated = ""
        self._debounce_task: asyncio.Task | None = None

        # Complete utterances ready for processing
        self.ready_queue: asyncio.Queue[str] = asyncio.Queue()

        # Event fired on each is_final transcript (for external consumers)
        self.speech_detected = asyncio.Event()

    async def start(self):
        """Connect to Deepgram and start listening for transcripts."""
        from deepgram import AsyncDeepgramClient
        from deepgram.core.events import EventType

        client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)

        self._ctx = client.listen.v2.connect(
            model=DEEPGRAM_MODEL,
            encoding="linear16",
            sample_rate=str(SAMPLE_RATE),
            language=DEEPGRAM_LANGUAGE,
            smart_format="true",
            punctuate="true",
            interim_results=str(DEEPGRAM_INTERIM_RESULTS).lower(),
            utterance_end_ms=str(DEEPGRAM_UTTERANCE_END_MS),
        )
        self._connection = await self._ctx.__aenter__()

        self._connection.on(EventType.OPEN, self._on_open)
        self._connection.on(EventType.MESSAGE, self._on_message)
        self._connection.on(EventType.CLOSE, self._on_close)
        self._connection.on(EventType.ERROR, self._on_error)

        await self._connection.start_listening()
        logger.info(
            "Deepgram connected (model=%s, lang=%s, utterance_end=%dms)",
            DEEPGRAM_MODEL,
            DEEPGRAM_LANGUAGE,
            DEEPGRAM_UTTERANCE_END_MS,
        )

    async def send_audio(self, chunk: np.ndarray):
        """Send a float32 audio chunk to Deepgram."""
        if not self._connected or self._connection is None:
            return
        try:
            from deepgram.extensions.types.sockets import ListenV2MediaMessage

            audio_bytes = _float32_to_int16_bytes(chunk)
            await self._connection.send_media(
                ListenV2MediaMessage(data=audio_bytes)
            )
        except Exception:
            logger.exception("Failed to send audio to Deepgram")

    async def stop(self):
        """Cleanly close the Deepgram connection."""
        self._cancel_debounce()
        # Flush any remaining accumulated text
        if self._accumulated.strip():
            self._flush_utterance()

        if self._connection is not None:
            try:
                from deepgram.extensions.types.sockets import (
                    ListenV2ControlMessage,
                )

                await self._connection.send_control(
                    ListenV2ControlMessage(type="CloseStream")
                )
            except Exception:
                pass
        if self._ctx is not None:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._connected = False
        self._connection = None
        logger.info("Deepgram disconnected")

    def _on_open(self, _):
        self._connected = True
        logger.info("Deepgram WebSocket opened")

    def _on_close(self, _):
        self._connected = False
        logger.info("Deepgram WebSocket closed")

    def _on_error(self, error):
        logger.error("Deepgram error: %s", error)

    def _on_message(self, message):
        msg_type = getattr(message, "type", "")

        if msg_type == "Results" or hasattr(message, "channel"):
            self._handle_transcript(message)
        elif msg_type == "UtteranceEnd":
            self._handle_utterance_end()

    def _handle_transcript(self, message):
        try:
            channel = getattr(message, "channel", None)
            if channel is None:
                return
            alternatives = getattr(channel, "alternatives", [])
            if not alternatives:
                return
            transcript = getattr(alternatives[0], "transcript", "")
            is_final = getattr(message, "is_final", False)

            if not transcript:
                return

            if is_final:
                if self._accumulated:
                    self._accumulated += " " + transcript
                else:
                    self._accumulated = transcript
                logger.info("Transcript (final): %s", transcript)
                self.speech_detected.set()
                self._restart_debounce()
            else:
                logger.debug("Transcript (interim): %s", transcript)
        except Exception:
            logger.exception("Error handling transcript message")

    def _handle_utterance_end(self):
        logger.info("Utterance end detected by Deepgram")
        self._cancel_debounce()
        if self._accumulated.strip():
            self._flush_utterance()

    def _flush_utterance(self):
        """Push accumulated text to the ready queue."""
        text = self._accumulated.strip()
        self._accumulated = ""
        self.speech_detected.clear()
        if text:
            try:
                self.ready_queue.put_nowait(text)
            except asyncio.QueueFull:
                logger.warning("Ready queue full, dropping utterance: %s", text[:60])

    def _restart_debounce(self):
        """Reset the debounce timer. If no new is_final arrives within
        debounce_s, flush the utterance (fallback for missing utterance_end)."""
        self._cancel_debounce()
        loop = asyncio.get_event_loop()
        self._debounce_task = loop.create_task(self._debounce_timer())

    def _cancel_debounce(self):
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            self._debounce_task = None

    async def _debounce_timer(self):
        try:
            await asyncio.sleep(self._debounce_s)
            if self._accumulated.strip():
                logger.info("Debounce timeout — flushing utterance")
                self._flush_utterance()
        except asyncio.CancelledError:
            pass
