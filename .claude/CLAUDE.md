# Jarvis v3 — Voice Interface for Claude Code

## Commands
- `uv pip install -e .` — install in dev mode
- `uv pip install -e ".[web]"` — install with web server deps
- `jarvis start` — start voice listener (requires tmux session)
- `jarvis test` — record 5s + transcribe (mic test)
- `jarvis say "text"` — TTS test
- `jarvis status` — show config and model status

## Process
- Before starting, read `.claude/CODEBASE.md` and relevant `.claude/docs/` files
- Do NOT use the Explore agent by default — read documentation first
- After code changes, update documentation if interfaces or behavior changed
- Metal GPU ops (STT, TTS) must never run concurrently — single-thread executor
- Test with both headphones and laptop speakers when touching audio pipeline
- pVAD model (`models/pvad/pvad.onnx`) is not in git — download from HuggingFace

## Available Documentation
- `.claude/CODEBASE.md` — project overview, architecture, module index
- `.claude/docs/listener.md` — main loop, VAD gating, barge-in, AEC
- `.claude/docs/audio.md` — STT (Parakeet TDT) and TTS (Kokoro MLX)
- `.claude/docs/pvad.md` — personalized VAD design and tuning
- `.claude/docs/web.md` — phone interface (FastAPI + WebSocket)

## Available Knowledge
- `.claude/knowledge/mlx-audio.md` — MLX audio stack conventions and gotchas
- `.claude/knowledge/v2-lessons.md` — what worked and failed in jarvis-v2
