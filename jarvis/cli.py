#!/usr/bin/env python3
"""Jarvis v3 CLI — voice interface for Claude Code."""

import argparse
import logging
import sys
from pathlib import Path


def cmd_start(args):
    """Start Jarvis listener (hands-free voice I/O for Claude Code)."""
    from jarvis.config import get_config
    from jarvis.listener import run_jarvis

    config = get_config()
    target = args.target or config.get("jarvis", {}).get("tmux_target", "claude:0")
    run_jarvis(tmux_target=target)


def cmd_test(args):
    """Quick mic + transcription test."""
    import numpy as np
    import sounddevice as sd

    from jarvis.config import SAMPLE_RATE
    from jarvis.transcriber import transcribe

    duration = 5
    print(f"Recording {duration} seconds...", file=sys.stderr)
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]

    print("Transcribing with Parakeet TDT...", file=sys.stderr)
    text = transcribe(audio, SAMPLE_RATE)
    print(f"Text: {text}" if text else "(no speech detected)")


def cmd_say(args):
    """Text-to-speech test."""
    from jarvis.speaker import speak

    text = args.text if args.text else sys.stdin.read()
    if text.strip():
        speak(text.strip(), lang=args.lang)


def cmd_web(args):
    """Start Jarvis web server (browser-based voice I/O with AEC)."""
    from jarvis.config import get_config
    from jarvis.web.server import run_web

    config = get_config()
    target = args.target or config.get("jarvis", {}).get("tmux_target", "claude:0")
    host = args.host or config.get("web", {}).get("host", "0.0.0.0")
    port = args.port or config.get("web", {}).get("port", 8000)
    run_web(tmux_target=target, host=host, port=port)


def cmd_status(args):
    """Show Jarvis v3 status."""
    from pathlib import Path

    from jarvis.config import STT_MODEL, TTS_MODEL, TTS_VOICE, PVAD_MODEL_DIR, SILERO_VAD_MODEL, EOU_MODEL

    print("Jarvis v3")
    print(f"  STT: Parakeet TDT ({STT_MODEL})")
    print(f"  TTS: Kokoro ({TTS_MODEL})")
    print(f"  TTS voice: {TTS_VOICE}")

    silero_path = Path(__file__).parent.parent / SILERO_VAD_MODEL
    print(f"  Silero VAD: {'found' if silero_path.exists() else 'MISSING'}")

    pvad_path = Path(__file__).parent.parent / PVAD_MODEL_DIR / "pvad.onnx"
    print(f"  pVAD model: {'found' if pvad_path.exists() else 'MISSING'}")

    eou_path = Path(__file__).parent.parent / EOU_MODEL
    print(f"  EOU model: {'found' if eou_path.exists() else 'MISSING'}")

    emb_path = Path(__file__).parent.parent / "speaker_embedding_ecapa.npy"
    print(f"  Speaker enrollment: {'yes' if emb_path.exists() else 'no'}")


def main():
    parser = argparse.ArgumentParser(prog="jarvis", description="Jarvis v3 — Voice I/O for Claude Code")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command")

    start_p = sub.add_parser("start", help="Start Jarvis listener")
    start_p.add_argument("--target", default=None, help="tmux target pane")

    sub.add_parser("test", help="Mic + transcription test")

    say_p = sub.add_parser("say", help="Text-to-speech")
    say_p.add_argument("text", nargs="?", default=None, help="Text (or pipe stdin)")
    say_p.add_argument("--lang", default="en", help="Language (en, it)")

    web_p = sub.add_parser("web", help="Start web server (browser AEC)")
    web_p.add_argument("--target", default=None, help="tmux target pane")
    web_p.add_argument("--host", default=None, help="Bind address (default: 0.0.0.0)")
    web_p.add_argument("--port", type=int, default=None, help="Port (default: 8000)")

    sub.add_parser("status", help="Show status")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Also log to file for remote debugging
    fh = logging.FileHandler(str(Path(__file__).parent.parent / "jarvis.log"))
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    commands = {"start": cmd_start, "test": cmd_test, "say": cmd_say, "web": cmd_web, "status": cmd_status}

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
