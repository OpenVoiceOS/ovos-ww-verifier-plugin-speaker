"""Committed speaker audio fixtures (synthesised once, reused in CI).

The WAVs under ``tests/fixtures/`` are generated locally by
``tests/generate_fixtures.py`` and committed to the repo, so the test suite
never calls edge-tts or touches the network in CI.

Two speakers:
- ``ENROLL_A`` / ``VERIFY_A`` — the authorised household member (voice A).
- ``GUEST_B`` — an un-enrolled guest (voice B) that must be rejected.
"""
import os
import wave

_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

ENROLL_A = [
    os.path.join(_DIR, "speaker_a_enroll_1.wav"),
    os.path.join(_DIR, "speaker_a_enroll_2.wav"),
]
VERIFY_A = os.path.join(_DIR, "speaker_a_verify.wav")
GUEST_B = os.path.join(_DIR, "speaker_b_guest.wav")


def wav_to_pcm(path: str) -> bytes:
    """Return the raw PCM frames of a WAV fixture."""
    with wave.open(path, "rb") as wf:
        return wf.readframes(wf.getnframes())
