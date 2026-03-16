---
tags: [knowledge, mlx, audio, apple-silicon]
created: 2026-03-16
last_reviewed: 2026-03-16
type: knowledge
---

# MLX Audio Stack — Conventions and Gotchas

## Metal GPU Safety
- Metal operations must init on the main thread — preload models before spawning executors
- STT (Parakeet) and TTS (Kokoro) both use Metal — never run concurrently
- Use `ThreadPoolExecutor(max_workers=1)` to serialize GPU ops in async code

## Parakeet TDT (STT)
- Import: `from parakeet_mlx import from_pretrained`
- API: `model.transcribe(file_path)` — requires WAV file, not numpy array
- Returns `result.text` (string, may be empty)
- Transducer architecture: outputs blanks on silence — no hallucination filters needed
- Model: `mlx-community/parakeet-tdt-0.6b-v3`

## Kokoro / mlx-audio (TTS)
- Import: `from mlx_audio.tts.utils import load_model`
- API: `model.generate(text, voice=voice, speed=speed, lang_code=lang_code)` — yields results
- Each result has `.audio` (mx.array) — convert with `np.array(result.audio).flatten()`
- Native sample rate: 24000 Hz
- Voices: `af_heart` (en), `if_sara` (it)
- Lang codes: `a` (American English), `i` (Italian)
- Warm up by generating "." on load — forces pipeline creation, avoids first-call latency

## Resampling
- AirPods run at 48kHz, Kokoro outputs 24kHz
- Use `scipy.signal.resample_poly(audio, up, down)` where up/down = gcd-reduced ratio
- Without resampling: choppy Bluetooth audio, clean on wired/speakers

## sounddevice
- `sd.InputStream` callback runs on a separate thread — use `loop.call_soon_threadsafe` to push to asyncio
- `sd.play()` is non-blocking — poll or await duration
- `sd.stop()` interrupts playback immediately
- Device selection: use explicit index from `sd.query_devices()` to prevent macOS auto-switching
- AirPods as default input returns zeros in A2DP mode — always set input device explicitly
