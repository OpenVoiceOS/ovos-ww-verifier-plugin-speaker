"""Real-model end-to-end tests: only authorised speakers trigger the wake word.

For **every** speaker-embedding model in ``speakeronnx.MODEL_REGISTRY`` this:

1. Synthesises two distinct voices with edge-tts (authorised speaker A, guest B).
2. Enrols speaker A in a real ``SpeakerVerifier`` backed by that model.
3. Confirms the model separates the speakers (same-speaker cosine > guest cosine).
4. Drives a real ``DinkumVoiceLoop`` (via ovoscope's ``MiniVoiceLoop``) and asserts
   that A's audio emits ``recognizer_loop:record_begin`` while B's audio is
   suppressed — i.e. an un-enrolled speaker cannot trigger the wake word.

Absolute cosine scales differ enormously between architectures (e.g. titanet
scores ~0.9, campplus ~0.15), so the acceptance threshold is calibrated per model
to the midpoint of the two measured scores; the separation assertion in step 3 is
what actually proves the model discriminates.

Heavy + networked: requires edge-tts, ffmpeg, and HuggingFace model downloads.
Each is skipped gracefully when unavailable.
"""
import os
import subprocess
import wave

import numpy as np
import pytest

try:
    from ovoscope.voice_loop import MiniVoiceLoop, MockHotWordEngine
    HAS_OVOSCOPE = True
except ImportError:
    HAS_OVOSCOPE = False

from speakeronnx import MODEL_REGISTRY, cosine
from ovos_ww_verifier_plugin_speaker import SpeakerVerifier

pytestmark = pytest.mark.skipif(
    not HAS_OVOSCOPE, reason="ovoscope (>=0.19.0a1) not installed"
)

VOICE_A = "en-US-GuyNeural"      # authorised household member
VOICE_B = "en-US-JennyNeural"    # un-enrolled guest

ENROLL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Speaker verification identifies household members by voice.",
]
VERIFY_TEXT_A = "Voice commands are only accepted from authorised speakers."
VERIFY_TEXT_B = "This guest voice should not be accepted by the system."


def _have(tool: str) -> bool:
    try:
        subprocess.run([tool, "--version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _tts_to_wav(text: str, voice: str, path: str) -> None:
    """Synthesise *text* in *voice* to a 16 kHz mono 16-bit WAV at *path*."""
    mp3 = path.replace(".wav", ".mp3")
    subprocess.run(
        ["edge-tts", "--voice", voice, "--text", text, "--write-media", mp3],
        check=True, timeout=60, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", "-f", "wav", path],
        check=True, timeout=30, capture_output=True,
    )
    os.unlink(mp3)


def _pcm(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        return wf.readframes(wf.getnframes())


@pytest.fixture(scope="module")
def voices(tmp_path_factory):
    """Generate (and cache for the module) the edge-tts speaker clips."""
    if not (_have("edge-tts") and _have("ffmpeg")):
        pytest.skip("edge-tts / ffmpeg not available")

    d = tmp_path_factory.mktemp("voices")
    enroll_a = []
    for i, text in enumerate(ENROLL_TEXTS):
        p = str(d / f"a_enroll_{i}.wav")
        _tts_to_wav(text, VOICE_A, p)
        enroll_a.append(p)
    verify_a = str(d / "a_verify.wav")
    _tts_to_wav(VERIFY_TEXT_A, VOICE_A, verify_a)
    guest_b = str(d / "b_guest.wav")
    _tts_to_wav(VERIFY_TEXT_B, VOICE_B, guest_b)

    return {
        "enroll_a": enroll_a,
        "verify_a": verify_a,
        "guest_b": guest_b,
    }


def _triggers_wakeword(verifier, audio_pcm: bytes) -> bool:
    """Feed *audio_pcm* through a real DinkumVoiceLoop gated by *verifier*.

    Returns True if ``recognizer_loop:record_begin`` was emitted.
    """
    ww = MockHotWordEngine("hey_mycroft", trigger_after=1)
    with MiniVoiceLoop(ww_instances={"hey_mycroft": ww}, verifiers=[verifier]) as vl:
        msgs = vl.feed_chunks([audio_pcm])
        return any(m.msg_type == "recognizer_loop:record_begin" for m in msgs)


@pytest.mark.parametrize("model_alias", sorted(MODEL_REGISTRY.keys()))
def test_only_authorised_speaker_triggers_wakeword(model_alias, voices, tmp_path):
    """For each embedding model, A triggers the WW and guest B is rejected."""
    profiles_path = str(tmp_path / "profiles.json")
    verifier = SpeakerVerifier(config={
        "model": model_alias,
        "profiles_path": profiles_path,
    })

    try:
        verifier.enroll("A", voices["enroll_a"])
        embedder = verifier._get_embedder()
        profile = verifier._get_profiles()["A"]
        score_a = float(cosine(embedder.embed(voices["verify_a"]), profile))
        score_b = float(cosine(embedder.embed(voices["guest_b"]), profile))
    except Exception as e:  # model download / runtime unavailable
        pytest.skip(f"{model_alias}: model unavailable ({e})")

    # The model must place the enrolled speaker above the guest.
    assert score_a > score_b, (
        f"{model_alias} failed to separate speakers: "
        f"A={score_a:.3f} <= B={score_b:.3f}"
    )

    # Calibrate the acceptance threshold to sit between the two scores.
    verifier._threshold = (score_a + score_b) / 2.0

    assert _triggers_wakeword(verifier, _pcm(voices["verify_a"])), (
        f"{model_alias}: authorised speaker A did NOT trigger the wake word"
    )
    assert not _triggers_wakeword(verifier, _pcm(voices["guest_b"])), (
        f"{model_alias}: un-enrolled guest B WAS able to trigger the wake word"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
