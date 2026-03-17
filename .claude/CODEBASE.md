<!-- docs-synced-at: initial -->
# Jarvis v3 — Codebase Overview

Local voice interface for Claude Code on Apple Silicon. Parakeet TDT (STT) + Kokoro (TTS) + dual VAD (Silero generic + FireRedChat personalized) + SmartTurn EOU detection. Adapted from jarvis-v2 + shubhdotai/offline-voice-ai patterns.

## Architecture

```
tmux mode (jarvis start):
  Mic -> Silero VAD (4-state machine) -> EOU detection -> Parakeet TDT (STT)
    -> tmux send-keys -> Claude Code
                            |
  Speaker <- sounddevice <- Kokoro TTS <- voice-output-hook (.tts-queue file)

web mode (jarvis web):
  Browser mic (AEC on) -> WebSocket -> Silero VAD -> EOU -> Parakeet STT
    -> tmux send-keys -> Claude Code
                            |
  Browser speaker <- WebSocket <- WAV <- Kokoro TTS <- voice-output-hook (.tts-queue)
```

Key design: asyncio.Lock guards all MLX ops (STT + TTS never concurrent).
Barge-in uses asyncio.Event for clean cancellation of TTS playback.

## Tech Stack
- Python 3.12+, uv, hatchling
- parakeet-mlx (STT) — transducer, no hallucinations
- mlx-audio (TTS) — Kokoro 82M, pure MLX
- onnxruntime — Silero VAD + pVAD + SmartTurn EOU (all CPU, no GPU contention)
- transformers — WhisperFeatureExtractor for EOU model
- sounddevice + soundfile — audio I/O
- scipy — resampling (24kHz -> device native rate)
- FastAPI + uvicorn — web/phone interface (optional)

## Modules

| Module | Description | Docs |
|--------|-------------|------|
| `jarvis/cli.py` | CLI entry point (start, test, say, web, status) | — |
| `jarvis/config.py` | Configuration constants + YAML loader | — |
| `jarvis/pipeline.py` | PipelineResources, SpeechDetector (4-state VAD machine) | [docs/listener.md](docs/listener.md) |
| `jarvis/listener.py` | tmux-mode listener using pipeline components | [docs/listener.md](docs/listener.md) |
| `jarvis/vad.py` | SileroVAD, PersonalizedVAD, EndOfUtteranceDetector | [docs/pvad.md](docs/pvad.md) |
| `jarvis/audio_buffer.py` | Pre-buffer + active segment capture | [docs/listener.md](docs/listener.md) |
| `jarvis/transcriber.py` | Parakeet TDT wrapper | [docs/audio.md](docs/audio.md) |
| `jarvis/speaker.py` | Kokoro TTS wrapper, resampling, bilingual | [docs/audio.md](docs/audio.md) |
| `jarvis/web/server.py` | Web mode: FastAPI + WebSocket server, browser AEC | [docs/web-mode-spec.md](docs/web-mode-spec.md) |

## Key Files
- `config.yaml` — tunable parameters (devices, thresholds, models, voices)
- `models/silero_vad.onnx` — Silero VAD (not in git)
- `models/smart_turn_v3.onnx` — SmartTurn EOU detector (not in git)
- `models/pvad/pvad.onnx` — pVAD model (not in git)
- `speaker_embedding_ecapa.npy` — speaker enrollment (not in git)
- `web/index.html` — browser client (AEC, correlator, playback queue, barge-in)
- `web/correlator.js` — AudioWorklet for echo detection (cosine similarity)
- `reference/` — source repos used for adaptation (offline-voice-ai, pipecat-macos)
