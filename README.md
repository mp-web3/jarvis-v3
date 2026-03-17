# Jarvis v3

Fully local voice interface for Claude Code on Apple Silicon. Talk to Claude through your terminal, no cloud APIs for speech.

Mic → Parakeet TDT 0.6B (STT) → Qwen 1.5B (transcript polish) → tmux injection → Claude Code → Kokoro 82M (TTS) → speaker.

## What it does

You speak into the mic. Jarvis transcribes your speech, cleans it up, types it into a tmux pane where Claude Code is running, waits for the response, and reads it back to you.

The tmux approach means it works with any CLI tool, not just Claude Code.

## Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| STT | Parakeet TDT 0.6B (MLX) | Speech-to-text, no hallucinations on silence |
| TTS | Kokoro 82M (MLX) | Text-to-speech, bilingual en/it |
| VAD | Silero (ONNX) | Generic voice activity detection |
| pVAD | FireRedChat (ONNX) | Speaker-verified VAD, filters for your voice |
| EOU | SmartTurn v3 (ONNX) | ML-based end-of-utterance prediction |
| Polish | Qwen 1.5B (4-bit, MLX) | Strips filler words, fixes grammar, deduplicates |

All inference runs on Metal via MLX. No cloud APIs, no internet required for speech.

## How v3 was built

v3 is a synthesis of two codebases:

1. **jarvis-v2** had the right models (Parakeet, Kokoro, personalized VAD) but was a monolithic 520-line file doing everything. Binary listening state, fixed silence timer, broken barge-in.

2. **[offline-voice-ai](https://github.com/shubhdotai/offline-voice-ai)** had the right architecture: 4-state VAD machine, async queues, proper concurrency, SmartTurn EOU detection. But it used Whisper (hallucinates on silence) and lacked speaker verification.

v3 took the **architecture** from offline-voice-ai and the **components** from jarvis-v2.

## Key improvements over v2

**SmartTurn end-of-utterance.** Replaced the fixed 700ms silence timer with an ML model that predicts when you're done talking. Considers spectral context, not just silence duration. You can pause mid-sentence to think and it waits.

**Transcript polishing.** Qwen 1.5B cleans up raw STT output before Claude sees it. Strips filler words ("um", "like"), deduplicates repeated phrases, fixes minor grammar. Adds ~300-500ms per call, noticeably improves Claude's response quality.

**Smart barge-in.** Separate Silero VAD monitors the mic during TTS playback. Sustained speech cancels TTS, enters a settling state to avoid re-triggering, then resumes normal listening.

**4-state VAD machine.** QUIET → STARTING → SPEAKING → STOPPING. Starting requires sustained frames above threshold (prevents noise triggers). Stopping uses SmartTurn for EOU prediction. Accumulation buffer stitches speech across short pauses.

**Modular architecture.** 10 modules (~1270 lines) instead of one 520-line file. asyncio.Lock for Metal GPU safety, Event-based cancellation, transcription queue with accumulation.

## Modules

| Module | File | Purpose |
|--------|------|---------|
| CLI | `cli.py` | Commands: `start`, `test`, `say`, `status` |
| Config | `config.py` | Constants + YAML loader |
| Pipeline | `pipeline.py` | Resource management, SpeechDetector (4-state VAD) |
| Listener | `listener.py` | Mic → VAD → STT → tmux, TTS queue, barge-in |
| VAD | `vad.py` | SileroVAD, PersonalizedVAD, EndOfUtteranceDetector |
| Audio buffer | `audio_buffer.py` | Pre-buffer + active segment capture |
| Transcriber | `transcriber.py` | Parakeet TDT wrapper |
| Speaker | `speaker.py` | Kokoro TTS, bilingual, resampling |
| Polisher | `polisher.py` | Hybrid regex + Qwen 1.5B transcript cleanup |

## Setup

Requires Python 3.12+, Apple Silicon Mac, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mp-web3/jarvis-v3.git
cd jarvis-v3
uv sync
```

Download ONNX models (not included in repo):
- `models/silero_vad.onnx` — [Silero VAD](https://github.com/snakers4/silero-vad)
- `models/smart_turn_v3.onnx` — [SmartTurn](https://github.com/shubhdotai/offline-voice-ai)
- `models/pvad/pvad.onnx` — FireRedChat personalized VAD (optional)

Configure `config.yaml` with your audio devices:

```yaml
listener:
  input_device: "MacBook Air Microphone"
  output_device: "AirPods"
```

Set up a tmux session for Claude Code:

```bash
tmux new-session -s claude
claude  # start Claude Code in this pane
```

## Usage

```bash
jarvis start          # start listening
jarvis status         # check model status
jarvis say "hello"    # test TTS
```

## Known limitations

- **Barge-in requires headphones.** Without them, TTS output hits the mic and triggers false barge-in. Echo cancellation is not yet ported from v2.
- **Post-interrupt flow.** After barge-in, the interrupted text fragment can cause premature processing. Needs accumulation logic.

## License

MIT
