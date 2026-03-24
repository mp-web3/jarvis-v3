"""Pipeline resources — shared TTS model and MLX lock.

With Deepgram handling STT, the pipeline only needs to manage TTS (Kokoro MLX)
and the MLX lock to prevent concurrent Metal GPU access.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class PipelineResources:
    """Global resources initialized once. Singleton per process."""

    def __init__(self):
        from jarvis.speaker import preload as preload_tts, preload_acknowledgments

        logger.info("Initializing pipeline resources...")

        preload_tts()
        preload_acknowledgments()

        # MLX lock — only TTS uses MLX now (no STT contention)
        self.mlx_lock = asyncio.Lock()

        logger.info("Pipeline resources ready")
