---
tags: [spec, web, aec, implementation]
created: 2026-03-17
type: spec
status: phase-1-done
---

# Web Mode — Speaker-Free Jarvis via Browser AEC

## Problem

Jarvis v3 requires AirPods/headphones because TTS output from speakers feeds back into
the mic and triggers false barge-in. Software AEC was exhaustively tested in v2 (WebRTC,
energy thresholds, mic/loopback ratio, pVAD during TTS) — all failed. See
`knowledge/v2-lessons.md` for the full post-mortem.

## Solution

Run the voice pipeline server-side (ML models stay on Mac) but capture mic audio and play
TTS audio in the browser. Browsers provide hardware-accelerated echo cancellation via
`getUserMedia({ echoCancellation: true })` — Chrome's AEC3, maintained by Google for
Meet/WebRTC. This is fundamentally better than Python-level AEC because the browser has
direct access to the OS audio output path and adapts in real-time.

A secondary correlator layer (AudioWorklet) computes cosine similarity between mic and TTS
reference signals to catch any echo that leaks through AEC3.

## Same Repo, Not Separate Project

Web mode belongs in jarvis-v3. It is NOT a separate project. Reasons:

1. **Shared core** — reuses pipeline.py, vad.py, transcriber.py, speaker.py, polisher.py,
   audio_buffer.py, config.py. Every ML model and detection algorithm is shared.
2. **Already scaffolded** — `web/` dir exists, `pyproject.toml` has `[web]` optional deps,
   CLAUDE.md references `docs/web.md`, CLI has extensible subparser.
3. **Single config** — `config.yaml` already has the relevant fields.
4. **Same deployment** — runs on the same Mac as `jarvis start`. No container, no remote
   server, no infra. Just a different I/O transport.

The only thing that changes is the transport layer: instead of
`mic → sounddevice → pipeline → sd.play()`, it's
`browser mic → WebSocket → pipeline → WebSocket → browser playback`.

Duplicating the ML pipeline into a separate repo would be wrong — it's the same codebase
with a different front door.

## Architecture

```
Browser (AEC + mic capture)                  Mac (ML models)
┌─────────────────────────┐                  ┌────────────────────────────┐
│                         │   WebSocket      │                            │
│  getUserMedia(AEC: on)  │ ──float32 PCM──> │  SpeechDetector (4-state)  │
│                         │                  │  ├─ Silero VAD             │
│  correlator.js worklet  │                  │  ├─ SmartTurn EOU          │
│  (echo detection)       │                  │  └─ semantic turn check    │
│                         │                  │                            │
│  Playback queue         │ <──WAV bytes──── │  Parakeet STT → tmux      │
│  (barge-in detection)   │                  │                            │
│                         │   interrupt msg  │  .tts-queue → Kokoro TTS   │
│  Interruption detector  │ ──────────────>  │  (voice-output-hook feeds  │
│                         │                  │   the queue, same as tmux  │
└─────────────────────────┘                  │   mode)                    │
                                             └────────────────────────────┘
```

### Data flow — speech to Claude

1. Browser captures mic audio at 16kHz mono with `echoCancellation: true`
2. AudioWorklet (correlator) passes through mic audio, flags echo frames
3. Client sends non-echo float32 PCM chunks over WebSocket as base64
4. Server feeds chunks through `SpeechDetector.process_chunk()`
5. On TRANSCRIBE event: segment → `transcriber.transcribe()` (MLX locked)
6. On RESPOND event: accumulated text → `tmux send-keys` to Claude pane

### Data flow — Claude to voice

1. Claude responds in the tmux pane
2. `voice-output-hook` writes response text to `~/.claude/.tts-queue` (unchanged)
3. Server's TTS watcher reads `.tts-queue`, renders via `speaker.render()` (MLX locked)
4. Server sends WAV bytes over WebSocket to browser
5. Browser enqueues audio in playback queue, plays sequentially
6. Barge-in: browser detects user speech during playback via RMS threshold,
   sends interrupt message, server cancels remaining TTS

### What stays the same as tmux mode

- PipelineResources (MLX lock, model preload)
- SpeechDetector (4-state VAD, EOU, semantic turn check)
- Transcription pipeline (Parakeet TDT, polish with Qwen)
- tmux injection (_send_to_tmux)
- TTS queue watching (.tts-queue file polling)
- TTS rendering (Kokoro via speaker.render())
- Config system (config.yaml)

### What changes

| Aspect | tmux mode (listener.py) | web mode (web/server.py) |
|--------|------------------------|-------------------------|
| Mic input | sounddevice InputStream | WebSocket float32 from browser |
| Audio output | sd.play() | WebSocket WAV bytes to browser |
| AEC | None (needs headphones) | Browser getUserMedia AEC3 + correlator |
| Barge-in detection | SileroVAD on mic during TTS | Browser-side RMS + interrupt message |
| Process model | Single async loop | FastAPI + WebSocket per connection |
| Client | Terminal (no UI) | Browser page |

## File Plan

```
jarvis-v3/
├── jarvis/
│   ├── web/                    # NEW — web mode package
│   │   ├── __init__.py
│   │   └── server.py           # FastAPI app, WebSocket handler, TTS watcher
│   └── cli.py                  # ADD `jarvis web` subcommand
├── web/                        # NEW — static client files (served by FastAPI)
│   ├── index.html              # Browser client (adapted from reference)
│   └── correlator.js           # AudioWorklet for echo detection (from reference)
└── config.yaml                 # ADD web section (host, port)
```

### jarvis/web/server.py — Server

Responsibilities:
- FastAPI app with `/` (serves index.html), `/correlator.js`, `/ws` (WebSocket)
- One `WebVoicePipeline` per WebSocket connection
- `WebVoicePipeline` owns a `SpeechDetector` instance (reused from pipeline.py)
- Audio processing: decode base64 float32 → split into chunks → feed SpeechDetector
- Transcription: same worker pattern as listener.py (async queue + MLX lock)
- tmux output: same `_send_to_tmux` function from listener.py
- TTS watcher: same `.tts-queue` polling, but `render()` → WAV bytes → WebSocket
  instead of `sd.play()`
- State broadcasting: send VAD state, metrics to client

Key interfaces:

```python
class WebVoicePipeline:
    def __init__(self, ws: WebSocket, resources: PipelineResources, tmux_target: str): ...
    async def start(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def handle_message(self, payload: dict) -> None: ...

# Internal methods (mirror listener.py structure):
    async def _handle_audio(self, audio_data: str) -> None: ...
    async def _handle_events(self, events: list[str]) -> None: ...
    async def _queue_transcription(self) -> None: ...
    async def _transcription_worker(self) -> None: ...
    async def _finalize_and_send(self) -> None: ...
    async def _watch_tts_queue(self) -> None: ...
    async def _send_tts(self, text: str) -> None: ...
    async def _send_state(self) -> None: ...
```

Note: unlike the reference server.py which has its own LLMHandler for direct LLM calls,
our server routes through tmux → Claude Code. The LLM response comes back asynchronously
via the voice-output-hook → .tts-queue. This means we don't need an LLMHandler at all.

### web/index.html — Browser Client

Adapted from `reference/offline-voice-ai/index.html`. Keep:
- `getUserMedia({ echoCancellation: true, noiseSuppression: true, autoGainControl: true })`
- AudioWorklet correlator setup (mic + TTS reference → echo detection)
- Playback queue with sequential audio playback
- Interruption detector (RMS-based, sustained frames before triggering)
- Rolling audio buffer (captures pre-interruption frames)
- WebSocket protocol (start, stop, media, state, metrics, interrupt events)

Change:
- Remove LLM-specific conversation display (we can't show Claude's streamed text —
  it goes through tmux, not the WebSocket). Show transcript of user speech only.
- Simplify UI — this is a local tool, not a product. Minimal: status indicator,
  transcript, start/stop button.
- Add connection URL configurability (localhost default, but support `--host` flag)

### web/correlator.js — Echo Detection AudioWorklet

Copy from `reference/offline-voice-ai/correlator.js` as-is. It's a standalone worklet that:
1. Takes two inputs: mic (input 0) and TTS reference (input 1)
2. Computes cosine similarity between them
3. Reports { corr, micRms, refRms } via port.postMessage
4. Passes mic audio through to output

No modifications needed.

### CLI addition

```python
# In cli.py, add:
def cmd_web(args):
    """Start Jarvis web server (browser-based voice I/O)."""
    from jarvis.web.server import run_web

    config = get_config()
    target = args.target or config.get("jarvis", {}).get("tmux_target", "claude:0")
    host = args.host or config.get("web", {}).get("host", "0.0.0.0")
    port = args.port or config.get("web", {}).get("port", 8000)
    run_web(tmux_target=target, host=host, port=port)
```

Usage: `jarvis web` or `jarvis web --port 8080 --target claude:0`

### Config addition

```yaml
# Append to config.yaml:
web:
  host: "0.0.0.0"
  port: 8000
```

## WebSocket Protocol

Same as reference implementation. Messages are JSON with `event` field.

### Client → Server

| Event | Fields | When |
|-------|--------|------|
| `start` | — | User clicks Start Listening |
| `stop` | `target?` ("playback" or omit) | User clicks Stop, or stop playback only |
| `media` | `audio` (base64 float32) | Every ~30ms while listening |
| `interrupt` | — | User barge-in detected client-side |

### Server → Client

| Event | Fields | When |
|-------|--------|------|
| `state` | `state`, `vad_prob`, `listening`, `responding` | After every state change |
| `text` | `role`, `text`, `partial?`, `complete?` | User transcription |
| `media` | `audio` (base64 WAV), `mime`, `index` | TTS audio chunk |
| `metrics` | `metrics: { stt?, llm?, tts? }` | Performance data |
| `interrupt` | — | Server-side TTS interruption |
| `speech_start` | — | VAD detected speech start |
| `speech_end` | — | VAD detected speech end |

## Implementation Order

### Phase 1 — Minimal working web mode (1 session)

1. Create `jarvis/web/__init__.py` (empty)
2. Create `jarvis/web/server.py` with:
   - FastAPI app, static file serving
   - WebSocket handler with `WebVoicePipeline`
   - Audio input processing (reuse SpeechDetector)
   - Transcription pipeline (reuse transcriber)
   - tmux output (reuse _send_to_tmux)
   - TTS queue watcher → WAV over WebSocket
3. Copy `correlator.js` from reference to `web/`
4. Adapt `index.html` from reference to `web/` (strip LLM UI, keep AEC + playback)
5. Add `jarvis web` to CLI
6. Add `web:` section to config.yaml
7. Test: `jarvis web`, open browser, speak, verify audio reaches Claude via tmux

### Phase 2 — Polish and barge-in (1 session)

1. Verify browser AEC works without headphones (the whole point)
2. Test correlator echo detection quality
3. Test client-side barge-in during TTS playback
4. Add polish (Qwen cleanup) to web transcription pipeline
5. Handle edge cases: WebSocket disconnect, multiple clients, TTS queue race

### Phase 3 — Optional (future)

- HTTPS + mDNS for phone/tablet access on local network
- Mobile-friendly UI
- Show Claude's response text (would require hooking into Claude's output stream,
  not just TTS — currently the voice-output-hook only fires for TTS-eligible text)

## Testing Plan

| Test | Method | Pass criteria |
|------|--------|---------------|
| Server starts | `jarvis web` | FastAPI serves at localhost:8000 |
| Browser connects | Open page | WebSocket connects, state = quiet |
| Mic capture | Click Start | Browser requests mic, audio frames sent |
| VAD detection | Speak | State transitions: quiet → starting → speaking → stopping |
| Transcription | Speak a sentence | Text appears in transcript area |
| tmux injection | Speak a sentence | Text appears in Claude tmux pane |
| TTS playback | Wait for Claude response | Audio plays in browser |
| AEC (the point) | Speak during TTS with speakers | No false barge-in triggered |
| Barge-in | Speak intentionally during TTS | Playback stops, new speech processed |
| Correlator | Play TTS, observe echo metrics | corr > 0.3 during playback, echo frames suppressed |

## Risks

1. **Browser AEC quality varies** — Chrome's AEC3 is excellent, Safari's is inconsistent.
   Firefox uses WebRTC's AEC but lower quality. Recommend Chrome.
2. **Latency** — WebSocket adds ~1-5ms per frame vs direct sounddevice. Negligible for
   30ms chunks but worth measuring.
3. **Single client** — TTS queue file is shared state. If tmux mode and web mode run
   simultaneously, they'll race on `.tts-queue`. Phase 1: enforce single mode. Phase 2:
   consider a proper queue (Unix socket or asyncio.Queue bridging).
4. **No Claude response text** — The voice-output-hook writes to `.tts-queue` but doesn't
   expose the text to the web client. User sees transcript of their speech but not Claude's
   text response (only hears it). Acceptable for v1.

## Dependencies

Already in `pyproject.toml` under `[project.optional-dependencies] web`:
- `fastapi>=0.115`
- `uvicorn[standard]>=0.34`

No new dependencies needed.
