<!-- docs-synced-at: initial -->
# Jarvis v3 — Codebase Overview

Local voice interface for Claude Code on Apple Silicon. Parakeet TDT for speech-to-text, Kokoro via mlx-audio for text-to-speech, FireRedChat pVAD for speaker-verified voice activity detection. Runs entirely on-device — no cloud APIs for audio.

## Architecture

```
Mic -> pVAD (voice activity) -> Parakeet TDT (STT) -> tmux send-keys -> Claude Code
                                                                           |
Speaker <- sounddevice <- Kokoro mlx-audio (TTS) <- Claude hook (voice-output-hook)
```

Web mode (phone):
```
Phone mic -> MediaRecorder -> WebSocket -> ffmpeg decode -> Parakeet STT
  -> claude -p (full context) -> Kokoro TTS -> WAV -> WebSocket -> Audio element
```

## Tech Stack
- Python 3.12+, uv package manager, hatchling build
- parakeet-mlx (STT) — transducer architecture, outputs blanks on silence
- mlx-audio (TTS) — Kokoro 82M, pure MLX, no PyTorch GPU contention
- onnxruntime (pVAD) — FireRedChat pvad.onnx on CPU
- sounddevice + soundfile — audio I/O
- scipy — resampling (24kHz Kokoro -> 48kHz AirPods)
- FastAPI + uvicorn — web/phone interface
- spacy — (v2 dep, evaluate if still needed)

## Modules

| Module | Description | Docs |
|--------|-------------|------|
| `jarvis/cli.py` | CLI entry point (start, test, say, status) | — |
| `jarvis/listener.py` | Main async loop: mic -> VAD -> STT -> tmux, barge-in, AEC | [docs/listener.md](docs/listener.md) |
| `jarvis/transcriber.py` | Parakeet TDT wrapper, model caching | [docs/audio.md](docs/audio.md) |
| `jarvis/speaker.py` | Kokoro TTS wrapper, resampling, bilingual | [docs/audio.md](docs/audio.md) |
| `jarvis/pvad.py` | Personalized VAD (ONNX), speaker embedding | [docs/pvad.md](docs/pvad.md) |
| `web/server.py` | FastAPI WebSocket server for phone | [docs/web.md](docs/web.md) |
| `web/index.html` | Push-to-talk mobile UI | [docs/web.md](docs/web.md) |

## Key Files
- `config.yaml` — all tunable parameters (devices, thresholds, models, voices)
- `models/pvad/pvad.onnx` — pVAD model (not in git, download from HuggingFace)
- `speaker_embedding_ecapa.npy` — speaker enrollment (not in git, user-specific)
