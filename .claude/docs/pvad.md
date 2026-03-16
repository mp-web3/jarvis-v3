---
tags: [module, pvad, vad, speaker-verification]
created: 2026-03-16
last_reviewed: 2026-03-16
type: module-doc
---

# Personalized VAD — `jarvis/pvad.py`

FireRedChat personalized voice activity detection. ONNX model runs on CPU — no GPU contention with STT/TTS.

## How It Works

1. Audio chunk (30ms at 16kHz) split into 10ms frames (160 samples)
2. Each frame -> ONNX model with speaker embedding -> raw probability
3. EMA smoothing (alpha 0.8) on raw probability
4. Hysteresis state machine: speech_count / silence_count vs thresholds

## Key Interface

```python
class PersonalizedVAD:
    def __init__(self, config: dict, speaker_embedding: np.ndarray | None = None): ...
    def process_chunk(self, chunk: np.ndarray) -> bool: ...  # True = target speaker speaking
    def reset(self): ...                                      # Clear all state (after TTS interrupt)
    @property
    def probability(self) -> float: ...                       # Current EMA-smoothed probability
```

## Speaker Enrollment

- ECAPA-TDNN 192-dimensional embedding from SpeechBrain
- Stored as `speaker_embedding_ecapa.npy` in project root
- Without enrollment: pVAD runs as generic VAD (zeros embedding)
- With enrollment: only responds to enrolled speaker's voice

## ONNX Model Inputs/Outputs

| Input | Shape | Description |
|-------|-------|-------------|
| `input_audio` | `(1, 160)` | 10ms frame as float32 (int16-scaled) |
| `spkemb` | `(1, 192)` | Speaker embedding (zeros if no enrollment) |
| `mel_buffer` | `(1, 80, 15)` | Mel spectrogram state |
| `gru_buffer` | `(2, 1, 256)` | GRU hidden state |

| Output | Index | Description |
|--------|-------|-------------|
| Detection | 0 | Binary detection |
| Probability | 1 | Raw probability `[0, 1]` |
| Mel buffer | 2 | Updated mel state |
| GRU buffer | 3 | Updated GRU state |

## Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `activation_threshold` | 0.85 | Higher = fewer false positives, slower activation |
| `min_speech_frames` | 10 | Frames (10ms each) of speech before `_speaking = True` |
| `min_silence_frames` | 20 | Frames of silence before `_speaking = False` |
| `_ema_alpha` | 0.8 | Higher = more responsive, noisier |

## Model Location
- `models/pvad/pvad.onnx` — from HuggingFace `scmkrd/FireRedTTS-1DGPT`
- `models/pvad/spkrec-ecapa-voxceleb/` — SpeechBrain ECAPA-TDNN (for enrollment)
