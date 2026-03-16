---
tags: [module, web, fastapi, websocket, phone]
created: 2026-03-16
last_reviewed: 2026-03-16
type: module-doc
---

# Web Module — `web/server.py` + `web/index.html`

Phone voice interface via push-to-talk. Reuses the same STT/TTS models as the CLI listener but routes through `claude -p` instead of tmux.

## Architecture

```
Phone browser -> HTTPS (Tailscale) -> FastAPI
  -> WebSocket:
     - Binary (audio blob) -> ffmpeg decode -> Parakeet STT
     - STT text -> claude -p (full Claude Code context) -> response text
     - Response -> Kokoro TTS -> WAV bytes -> WebSocket -> Audio element
     - JSON: { type: "clear" } -> reset conversation
```

## Server (`web/server.py`)

- `decode_audio(data)` — ffmpeg converts webm/opus or mp4/aac to 16kHz mono float32
- `encode_wav(audio, sr)` — float32 to 16-bit WAV for browser playback
- `call_claude(prompt, conversation_id)` — subprocess `claude -p` with voice system prompt appended
- GPU executor: `ThreadPoolExecutor(max_workers=1)` — serializes STT/TTS to prevent Metal conflicts
- HTTPS via self-signed cert (`cert.pem`, `key.pem`) for iOS mic access

## Client (`web/index.html`)

- Push-to-talk: hold button to record, release to send
- States: idle -> recording -> processing -> playing -> idle
- iOS audio unlock: pre-creates Audio element on first gesture
- WebSocket reconnect on disconnect (2s retry)
- Chat-style message display (user/assistant bubbles)

## Known Issues (from v2)

- **iOS audio playback**: WAV generated but sometimes not played (autoplay policy edge case)
- **Conversation resume**: `conversation_id` extraction from claude stderr is incomplete (placeholder code)
- **Voice system prompt**: appended to every message — consider moving to `--system-prompt` flag if available

## Key Config

| Setting | Default | Description |
|---------|---------|-------------|
| `web.port` | 8765 | Server port |
| TLS certs | `web/cert.pem`, `web/key.pem` | Required for iOS mic access |
| Tailscale | `mattias-macbook-air.tailc4bc95.ts.net` | Phone access URL |
