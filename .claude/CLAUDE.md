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
- MLX ops guarded by `asyncio.Lock` in pipeline — never bypass this
- Test with both headphones and laptop speakers when touching audio pipeline
- ONNX models not in git — see `jarvis status` for what's missing

## Available Documentation
- `.claude/CODEBASE.md` — project overview, architecture, module index
- `.claude/docs/listener.md` — main loop, VAD gating, barge-in
- `.claude/docs/audio.md` — STT (Parakeet TDT) and TTS (Kokoro MLX)
- `.claude/docs/pvad.md` — personalized VAD + Silero VAD + EOU detection
- `.claude/docs/web.md` — phone interface (FastAPI + WebSocket)

## Available Knowledge
- `.claude/knowledge/mlx-audio.md` — MLX audio stack conventions and gotchas
- `.claude/knowledge/v2-lessons.md` — what worked and failed in jarvis-v2

## Reference Code
- `reference/offline-voice-ai/` — source repo for pipeline patterns (MIT)
- `reference/pipecat-macos/` — Pipecat local agent example (BSD-2)
