<!-- docs-synced-at: deepgram-migration -->
# Jarvis v3 — Codebase Overview

Voice interface for Claude Code. Deepgram cloud STT (streaming WebSocket) + Kokoro MLX TTS (local). Silero VAD for barge-in during TTS only.

## Architecture

```
tmux mode (jarvis start):
  Mic -> Deepgram WebSocket (STT + VAD + turn detection)
    -> transcript callback -> tmux send-keys -> Claude Code
                                |
  Speaker <- sounddevice <- Kokoro TTS <- voice-output-hook (.tts-queue file)

  During TTS: Silero VAD (local, CPU) detects barge-in -> cancel TTS

web mode (jarvis web):
  Browser mic (AEC on) -> WebSocket -> Deepgram WebSocket (STT)
    -> transcript callback -> tmux send-keys -> Claude Code
                                |
  Browser speaker <- WebSocket <- WAV <- Kokoro TTS <- voice-output-hook (.tts-queue)
```

Key design: Deepgram handles STT + VAD + end-of-utterance in one WebSocket.
asyncio.Lock guards MLX ops (TTS only — no more STT contention).
Barge-in uses Silero VAD + asyncio.Event for clean TTS cancellation.

## Tech Stack
- Python 3.12+, uv, hatchling
- deepgram-sdk (STT) — Deepgram Nova-3/Flux streaming WebSocket
- mlx-audio (TTS) — Kokoro 82M, pure MLX
- onnxruntime — Silero VAD (CPU, for barge-in only)
- sounddevice + soundfile — audio I/O
- scipy — resampling (24kHz -> device native rate)
- FastAPI + uvicorn — web/phone interface (optional)

## Modules

| Module | Description |
|--------|-------------|
| `jarvis/cli.py` | CLI entry point (start, test, say, web, status) |
| `jarvis/config.py` | Configuration constants + YAML loader |
| `jarvis/pipeline.py` | PipelineResources (TTS preload + MLX lock) |
| `jarvis/listener.py` | tmux-mode listener using Deepgram streaming STT |
| `jarvis/transcriber.py` | DeepgramTranscriber — WebSocket client, transcript accumulation, utterance-end detection |
| `jarvis/vad.py` | SileroVAD (barge-in during TTS playback) |
| `jarvis/speaker.py` | Kokoro TTS wrapper, resampling, bilingual |
| `jarvis/web/server.py` | Web mode: FastAPI + WebSocket server, browser AEC |
| `tests/` | pytest test suite |

## Key Files
- `config.yaml` — tunable parameters (devices, thresholds, models, voices)
- `models/silero_vad.onnx` — Silero VAD (not in git, for barge-in only)
- `web/index.html` — browser client (AEC, correlator, playback queue, barge-in)
- `web/correlator.js` — AudioWorklet for echo detection (cosine similarity)
- `reference/` — source repos used for adaptation (offline-voice-ai, pipecat-macos)

## Environment
- `DEEPGRAM_API_KEY` — required for STT (export in shell)
