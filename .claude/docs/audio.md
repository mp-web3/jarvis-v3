---
tags: [module, stt, tts, parakeet, kokoro]
created: 2026-03-16
last_reviewed: 2026-03-16
type: module-doc
---

# Audio Modules — STT + TTS

## Transcriber (`jarvis/transcriber.py`)

Parakeet TDT wrapper. Transducer architecture — outputs blanks on silence, structurally cannot hallucinate (unlike Whisper CTC/attention models).

```python
def preload(config: dict): ...                          # Pre-load on main thread (Metal GPU init)
def transcribe(audio: np.ndarray, sr: int, config: dict) -> str: ...  # Returns text or ""
```

- Model: `mlx-community/parakeet-tdt-0.6b-v3` (configurable)
- Writes temp WAV file for `model.transcribe()` — parakeet-mlx requires file path
- Singleton `_model` — loaded once, reused across calls
- Must preload on main thread before any executor calls (Metal GPU safety)

## Speaker (`jarvis/speaker.py`)

Kokoro 82M TTS via mlx-audio. Pure MLX — no PyTorch, no GPU contention with Parakeet.

```python
def preload(config: dict): ...                                    # Pre-load + warm up pipeline
def render(text: str, config: dict, lang: str = "en") -> np.ndarray | None: ...  # Text -> PCM
def speak(text: str, config: dict, lang: str = "en"): ...        # Render + play (blocking)
def get_sample_rate() -> int: ...                                 # Actual playback rate
```

- Bilingual: English (`af_heart`, lang_code `a`) and Italian (`if_sara`, lang_code `i`)
- Resamples from Kokoro native 24kHz to device rate (48kHz for AirPods) via `scipy.signal.resample_poly`
- Without resampling: choppy Bluetooth audio
- Warm-up in `preload()` generates a single "." to force pipeline init — avoids first-call latency

## Design Decisions

- **Separate preload/render pattern**: models must init on main thread (Metal), but render runs in ThreadPoolExecutor. Preload forces eager init.
- **No concurrent GPU ops**: both STT and TTS use Metal. Listener sets `_tts_playing` flag to block STT during TTS render.
- **Temp file for STT**: parakeet-mlx API requires file path, not numpy array. Using `NamedTemporaryFile(delete=True)`.
