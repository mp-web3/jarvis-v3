---
tags: [knowledge, jarvis, lessons, v2]
created: 2026-03-16
last_reviewed: 2026-03-16
type: knowledge
---

# Jarvis v2 Lessons Learned

## What Worked Well
- Parakeet TDT eliminated Whisper hallucinations entirely — transducer is the right architecture for streaming STT
- Kokoro on MLX: natural voice, no PyTorch GPU contention, fast render
- pVAD with speaker enrollment: filters background speakers effectively
- Ring buffer (800ms): captures sentence starts before VAD activates
- Debounce (300ms): prevents split sentences from becoming separate prompts
- Sequential TTS queue: handles multi-part Claude responses cleanly
- Explicit device selection in config: prevents macOS from hijacking input to AirPods mic

## What Failed
- **Barge-in with laptop speakers**: WebRTC AEC reduces both echo AND your voice to ~0.001 RMS. Energy thresholds, mic-vs-loopback ratio, pVAD during TTS — all unreliable. Only solutions: hardware AEC (ReSpeaker) or headphones.
- **Software echo cancellation approaches tried**: WebRTC AEC, energy thresholds on raw mic, mic/loopback ratio with 300ms smoothing, pVAD during TTS. None worked without headphones.
- **iOS audio playback**: autoplay policy blocks `Audio.play()` even after user gesture unlock in some cases
- **Web conversation resume**: `claude -p` stderr doesn't reliably expose conversation_id

## Architecture Decisions to Preserve
- Async main loop with sounddevice callback -> asyncio.Queue (jitter-free audio capture)
- Singleton model pattern with explicit `preload()` (Metal safety)
- Flag files for inter-process communication (~/.claude/.voice-mode, .speaking, .tts-queue)
- Config-driven everything (config.yaml): thresholds, devices, models, voices

## Things to Reconsider in v3
- spacy dependency (en_core_web_sm) — was it actually used? Not imported in any v2 module
- Temp WAV file for Parakeet — check if newer parakeet-mlx accepts numpy directly
- `claude -p` subprocess for web mode — consider Claude Agent SDK for streaming responses
- Flag file IPC (.tts-queue polling) — consider Unix socket or named pipe for lower latency
- AEC via `aec-audio-processing` package — evaluate if still needed given headphone-only barge-in
