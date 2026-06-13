"""End-to-end tests for the speaker verifier with a real speakeronnx model.

Enrol voice A from the committed fixtures, then:
- ``verify(A)`` -> True
- ``verify(guest B)`` -> False

Audio comes from committed fixtures (see ``tests/speaker_clips.py``) — no edge-tts
and no network for audio in CI. The speaker model downloads from HuggingFace on
first use.
"""
import pytest

from speakeronnx import cosine
from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
from speaker_clips import ENROLL_A, GUEST_B, VERIFY_A, wav_to_pcm


@pytest.fixture(scope="module")
def enrolled_verifier(tmp_path_factory):
    """A SpeakerVerifier with voice A enrolled (default wespeaker-resnet34)."""
    profiles_path = str(tmp_path_factory.mktemp("profiles") / "profiles.json")
    v = SpeakerVerifier(config={
        "model": "wespeaker-resnet34",
        "threshold": 0.45,
        "fail_open": False,
        "profiles_path": profiles_path,
    })
    v.enroll("Resident_A", ENROLL_A)
    return v


class TestE2EVerifier:
    def test_enrolled_voice_accepted(self, enrolled_verifier):
        assert enrolled_verifier.verify(wav_to_pcm(VERIFY_A)) is True

    def test_guest_voice_rejected(self, enrolled_verifier):
        assert enrolled_verifier.verify(wav_to_pcm(GUEST_B)) is False

    def test_empty_bytes_failclosed(self, enrolled_verifier):
        # empty chunk — embed fails; fail_open=False → rejected
        assert enrolled_verifier.verify(b"") is False

    def test_list_profiles_shows_enrolled_name(self, enrolled_verifier):
        assert "Resident_A" in enrolled_verifier.list_profiles()

    def test_enrolled_scores_above_guest(self, enrolled_verifier):
        v = enrolled_verifier
        profile_emb = v._get_profiles()["Resident_A"]
        score_a = cosine(v._get_embedder().embed(VERIFY_A), profile_emb)
        score_b = cosine(v._get_embedder().embed(GUEST_B), profile_emb)
        assert score_a > score_b, (
            f"enrolled score {score_a:.4f} should exceed guest score {score_b:.4f}"
        )
