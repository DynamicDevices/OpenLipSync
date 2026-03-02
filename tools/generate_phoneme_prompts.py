#!/usr/bin/env python3
"""
Pre-generate phoneme test audio for the realtime harness.

Creates one file per segment in data/phoneme_prompts/:
- segment_01: 1 s silence (WAV)
- segment_02..15: TTS of the target sound repeated REPEAT_COUNT times (e.g. "e e e e e", "thin thin ..." for /θ/) as MP3.

At run time, --speak plays the clip for each segment in the background while the mic
captures; the harness checks that the expected viseme is detected (peak activation, top-2).

TODO: Replace TTS clips with human-recorded phoneme clips (same filenames) for better
      pass rate and more natural test; see TODO.md (phoneme test audio).

Usage (from project root):
  uv run --extra realtime python tools/generate_phoneme_prompts.py
"""

from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "phoneme_prompts"

# Segment 1 = silence (we write a WAV). Segments 2-15 = TTS text for the sound only.
# Use "thin" for TH so TTS produces the /θ/ sound, not "tee aitch".
PHONEME_TEXTS = [
    None,  # 1: silence, no TTS
    "e", "ah", "eh", "oh", "oo", "p", "f", "thin", "t", "k", "sh", "s", "n", "r",
]
REPEAT_COUNT = 5  # Say each sound this many times per segment for a stronger test
SAMPLE_RATE = 16000

VOICE = "en-GB-SoniaNeural"


def write_silence_wav(path: Path, duration_sec: float = 1.0) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        n = int(SAMPLE_RATE * duration_sec)
        w.writeframes(b"\x00\x00" * n)


async def main() -> None:
    try:
        import edge_tts
    except ImportError:
        print(
            "edge-tts is not installed. From the project root, run:\n"
            "  uv run --extra realtime python tools/generate_phoneme_prompts.py\n"
            "(--extra realtime installs edge-tts into the environment, then runs this script.)",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(PHONEME_TEXTS)} files to {OUT_DIR}")

    # Segment 1: silence
    seg1 = OUT_DIR / "segment_01.wav"
    write_silence_wav(seg1)
    print(f"  {seg1.name} (silence)")

    for i, text in enumerate(PHONEME_TEXTS[1:], start=2):
        path = OUT_DIR / f"segment_{i:02d}.mp3"
        # Repeat the sound REPEAT_COUNT times so each segment is a solid test
        repeated = " ".join([text] * REPEAT_COUNT)
        communicate = edge_tts.Communicate(repeated, VOICE)
        await communicate.save(str(path))
        print(f"  {path.name} ({text} x{REPEAT_COUNT})")

    print("Done. Run phoneme-test with --speak: each segment plays this sound while the mic captures and checks the viseme.")


if __name__ == "__main__":
    asyncio.run(main())
