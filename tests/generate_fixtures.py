"""Regenerate the committed speaker audio fixtures. LOCAL ONLY — never run in CI.

Synthesises two distinct voices once with edge-tts + ffmpeg and writes 16 kHz
mono 16-bit WAVs into ``tests/fixtures/``. Those files are committed so the test
suite never depends on edge-tts or the network in CI.

Usage (needs ``edge-tts`` and ``ffmpeg`` on PATH)::

    python tests/generate_fixtures.py
"""
import os
import subprocess

from speaker_clips import ENROLL_A, GUEST_B, VERIFY_A

VOICE_A = "en-US-GuyNeural"      # authorised household member
VOICE_B = "en-US-JennyNeural"    # un-enrolled guest

CLIPS = [
    (ENROLL_A[0], VOICE_A, "The quick brown fox jumps over the lazy dog."),
    (ENROLL_A[1], VOICE_A,
     "Speaker verification identifies household members by voice."),
    (VERIFY_A, VOICE_A,
     "Voice commands are only accepted from authorised speakers."),
    (GUEST_B, VOICE_B,
     "This guest voice should not be accepted by the system."),
]


def _synth(path: str, voice: str, text: str) -> None:
    mp3 = path.replace(".wav", ".mp3")
    subprocess.run(
        ["edge-tts", "--voice", voice, "--text", text, "--write-media", mp3],
        check=True, timeout=60,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", "-f", "wav", path],
        check=True, timeout=30,
    )
    os.unlink(mp3)


def main() -> None:
    for path, voice, text in CLIPS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _synth(path, voice, text)
        print("wrote", os.path.basename(path))


if __name__ == "__main__":
    main()
