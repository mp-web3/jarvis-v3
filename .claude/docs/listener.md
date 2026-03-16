---
tags: [module, listener, vad, barge-in, aec]
created: 2026-03-16
last_reviewed: 2026-03-16
type: module-doc
---

# Listener Module — `jarvis/listener.py`

Main async loop that orchestrates mic capture, VAD gating, STT, and tmux injection. Also handles TTS playback queue and barge-in detection.

## Core Flow

1. Open mic InputStream (sounddevice, 16kHz mono, 30ms chunks)
2. Each chunk -> pVAD `process_chunk()` -> boolean: target speaker speaking?
3. On speech start: drain ring buffer (800ms pre-roll) into speech buffer
4. On silence (700ms): concatenate speech buffer -> `transcribe()` in executor
5. Transcribed text -> debounce (300ms) -> `tmux send-keys` to Claude Code pane
6. Exit phrases: "stop jarvis", "jarvis stop", "esci jarvis"

## Barge-in (interrupting TTS)

Uses pVAD probability directly during TTS playback:
- Threshold: `bargein_prob_threshold` (0.85) AND `rms >= 0.002`
- Sustain: N consecutive chunks of detected voice before interrupting
- Grace: brief pauses don't reset the counter
- On trigger: `sd.stop()`, 150ms settle, reset pVAD, resume listening
- **Only works with headphone output** — laptop speakers feed echo into mic

## AEC (Acoustic Echo Cancellation)

Optional WebRTC AEC via `aec-audio-processing` package:
- BlackHole 2ch loopback captures system audio as reference signal
- Processes mic in 10ms frames (160 samples at 16kHz)
- Reduces echo but also attenuates real speech — insufficient for barge-in without headphones

## TTS Queue

- Voice output hook writes text to `~/.claude/.tts-queue`
- `_watch_tts_queue()` polls every 200ms, plays sequentially via `_play_tts()`
- Sets `~/.claude/.speaking` flag during playback (hook reads this)
- Sets `~/.claude/.voice-mode` flag while listener is active

## Key Interfaces

```python
class JarvisListener:
    def __init__(self, config: dict, tmux_target: str): ...
    async def run(self): ...          # Main loop — call with asyncio.run()
    def stop(self): ...               # Signal graceful shutdown

def run_jarvis(config: dict, tmux_target: str = "claude:0"): ...  # Entry point
```

## Design Decisions

- **Async with sounddevice callback**: mic callback pushes to asyncio.Queue, main loop awaits. This keeps audio capture jitter-free.
- **Ring buffer**: 800ms deque captures the start of speech before pVAD triggers (pVAD has ~200ms latency to activate).
- **Debounce**: 300ms after utterance end before sending to tmux. Prevents split sentences from becoming separate prompts.
- **Single-word filter**: drops transcriptions of 1 word (usually noise artifacts).
- **Energy gate**: RMS < 0.0005 on full utterance = ambient noise, skip.
