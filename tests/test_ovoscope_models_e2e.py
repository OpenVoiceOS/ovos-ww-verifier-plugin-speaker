"""Real-model end-to-end tests: only authorised speakers trigger the wake word.

For **every** speaker-embedding model in ``speakeronnx.MODEL_REGISTRY`` this:

1. Enrols speaker A (committed fixtures) in a real ``SpeakerVerifier``.
2. Confirms the model separates the speakers (same-speaker cosine > guest cosine).
3. Drives a real ``DinkumVoiceLoop`` (via ovoscope's ``MiniVoiceLoop``) and asserts
   that A's audio emits ``recognizer_loop:record_begin`` while the guest's audio is
   suppressed — i.e. an un-enrolled speaker cannot trigger the wake word.

Absolute cosine scales differ enormously between architectures (titanet ~0.9,
campplus ~0.15), so the acceptance threshold is calibrated per model to the
midpoint of the two measured scores; the separation assertion in step 2 is what
proves the model discriminates.

Audio comes from committed fixtures (no edge-tts in CI); the model weights
download from HuggingFace on first use.
"""
import pytest

from ovoscope.voice_loop import MiniVoiceLoop, MockHotWordEngine
from speakeronnx import MODEL_REGISTRY, cosine
from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
from speaker_clips import ENROLL_A, GUEST_B, VERIFY_A, wav_to_pcm


def _triggers_wakeword(verifier, audio_pcm: bytes) -> bool:
    """Feed *audio_pcm* through a real DinkumVoiceLoop gated by *verifier*.

    Returns True if ``recognizer_loop:record_begin`` was emitted.
    """
    ww = MockHotWordEngine("hey_mycroft", trigger_after=1)
    with MiniVoiceLoop(ww_instances={"hey_mycroft": ww}, verifiers=[verifier]) as vl:
        msgs = vl.feed_chunks([audio_pcm])
        return any(m.msg_type == "recognizer_loop:record_begin" for m in msgs)


@pytest.mark.parametrize("model_alias", sorted(MODEL_REGISTRY.keys()))
def test_only_authorised_speaker_triggers_wakeword(model_alias, tmp_path):
    """For each embedding model, A triggers the WW and guest B is rejected."""
    verifier = SpeakerVerifier(config={
        "model": model_alias,
        "profiles_path": str(tmp_path / "profiles.json"),
    })
    verifier.enroll("A", ENROLL_A)

    embedder = verifier._get_embedder()
    profile = verifier._get_profiles()["A"]
    score_a = float(cosine(embedder.embed(VERIFY_A), profile))
    score_b = float(cosine(embedder.embed(GUEST_B), profile))

    # The model must place the enrolled speaker above the guest.
    assert score_a > score_b, (
        f"{model_alias} failed to separate speakers: "
        f"A={score_a:.3f} <= B={score_b:.3f}"
    )

    # Calibrate the acceptance threshold to sit between the two scores.
    verifier._threshold = (score_a + score_b) / 2.0

    assert _triggers_wakeword(verifier, wav_to_pcm(VERIFY_A)), (
        f"{model_alias}: authorised speaker A did NOT trigger the wake word"
    )
    assert not _triggers_wakeword(verifier, wav_to_pcm(GUEST_B)), (
        f"{model_alias}: un-enrolled guest B WAS able to trigger the wake word"
    )
