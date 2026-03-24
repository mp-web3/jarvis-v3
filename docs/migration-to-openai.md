# Migration: Local MLX Pipeline → OpenAI Managed Voice API

## What Changed

Jarvis v3 originally ran 6 local models on Apple Silicon via MLX/ONNX:

| Component | Old (local) | New (managed) |
|---|---|---|
| STT | Parakeet TDT 0.6B (MLX) — `transcriber.py` | OpenAI `gpt-4o-transcribe` via Realtime API WebSocket |
| TTS | Kokoro 82M (MLX) — `speaker.py` | OpenAI `tts-1` via `/v1/audio/speech` HTTP |
| VAD | Silero + FireRedChat ONNX — `vad.py` | Server-side VAD in OpenAI Realtime API |
| EOU | SmartTurn v3 ONNX — `pipeline.py` | Server-side silence detection in Realtime API |
| Polish | Qwen 1.5B (MLX) — `polisher.py` | Removed (API transcription quality removes the need) |

The new module is `jarvis/realtime_api.py`. It provides:
- `RealtimeSTT` — WebSocket client for OpenAI Realtime transcription with server-side VAD
- `render_tts()` — HTTP call to `/v1/audio/speech`, returns float32 audio + sample rate
- `preload_acknowledgments()` — pre-renders short filler phrases at startup

## Dependencies

**Removed:**
- `parakeet-mlx` — Parakeet TDT STT
- `mlx-audio[tts]` — Kokoro TTS
- `mlx-lm` — Qwen polish model
- `onnxruntime` — Silero VAD, SmartTurn EOU, pVAD
- `transformers` — WhisperFeatureExtractor for EOU model
- `spacy` + `en_core_web_sm` — sentence tokenisation in polisher

**Added:**
- `httpx>=0.27` — HTTP client for TTS API calls
- `websockets>=13` — WebSocket client for Realtime STT

**Kept:**
- `sounddevice`, `numpy`, `soundfile`, `pyyaml`, `scipy` — audio I/O and resampling

## Setup

### 1. API Key

Export your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Add to `~/.zshrc` to persist.

### 2. Reinstall dependencies

```bash
cd ~/jarvis-v3
uv sync
```

The install is now faster (~5 packages vs ~15) and has no ONNX model downloads.

### 3. ONNX models no longer needed

The following files in `models/` are no longer used and can be deleted:

```
models/silero_vad.onnx
models/smart_turn_v3.onnx
models/pvad/
speaker_embedding_ecapa.npy
```

## Configuration

`config.yaml` keys changed:

| Old key | New key | Notes |
|---|---|---|
| `stt.model` (Parakeet HF path) | `stt.model` (`gpt-4o-transcribe`) | Same key, new value |
| `tts.model` (Kokoro HF path) | `tts.model` (`tts-1` or `tts-1-hd`) | Same key, new value |
| `tts.voice` (Kokoro voice code) | `tts.voice` (OpenAI voice name) | `af_heart` → `nova`; see voices below |
| `tts.lang_code`, `tts.speed`, `tts.voices` | removed | Not used by OpenAI TTS |
| `pvad.*` | removed | Server-side VAD replaces local pVAD |
| `listener.polish_enabled`, `listener.polish_model` | removed | Polisher removed |
| `listener.loopback_device`, `listener.aec_delay_ms` | removed | Not needed with API approach |
| `listener.silence_ms` | `listener.silence_ms` | Now maps to OpenAI `silence_duration_ms` |
| *(new)* | `listener.vad_threshold` | OpenAI server VAD threshold (default: 0.5) |

### Available OpenAI TTS voices

`alloy`, `echo`, `fable`, `onyx`, **`nova`** (default), `shimmer`

## Architecture After Migration

```
tmux mode:
  Mic → sounddevice → RealtimeSTT (WebSocket, server-side VAD)
    → transcript event → tmux send-keys → Claude Code
                              |
  Speaker ← sounddevice ← render_tts() (HTTP) ← voice-output-hook (.tts-queue)
```

The local VAD state machine (`pipeline.py` / `vad.py`) is replaced by event-driven
callbacks from `RealtimeSTT.event_queue`:
- `{"type": "speech_started"}` — server VAD detected speech
- `{"type": "speech_stopped"}` — server VAD detected silence
- `{"type": "transcript", "text": "..."}` — final transcript ready

## Modules Status After This Migration Step

| Module | Status | Notes |
|---|---|---|
| `realtime_api.py` | new | OpenAI STT + TTS client |
| `config.py` | updated | OpenAI constants, MLX/ONNX constants removed |
| `pyproject.toml` | updated | MLX/ONNX deps removed, httpx + websockets added |
| `config.yaml` | updated | stt/tts keys updated, pvad/polish removed |
| `transcriber.py` | pending replacement | Still references `STT_MODEL` (removed from config) |
| `speaker.py` | pending replacement | Still references `TTS_MODEL`, `TTS_VOICE` etc. (removed) |
| `vad.py` | pending replacement | Still references EOU/pVAD constants (removed) |
| `polisher.py` | pending removal | References `POLISH_MODEL` (removed) |
| `pipeline.py` | pending update | Needs new listener integration |
| `listener.py` | pending rewrite | Needs to use `RealtimeSTT` instead of local VAD pipeline |
| `cli.py` | pending update | `status` command references old model constants |

## Cost Estimate

OpenAI pricing (as of early 2026):
- Realtime transcription: ~$0.006/min audio
- TTS (`tts-1`): ~$15 per 1M characters

At 30 minutes of daily voice use: ~$0.18/day transcription + ~$1–2/month TTS.
Total: ~$5–8/month.
