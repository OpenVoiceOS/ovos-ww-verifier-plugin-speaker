"""End-to-end tests for the speaker verifier plugin.

Requires: edge-tts, ffmpeg, speakeronnx installed.

Enroll voice A (en-US-GuyNeural) from 2 clips, then:
- verify(clip3 of A) -> True
- verify(clip1 of B / en-US-JennyNeural) -> False
"""

import os
import subprocess
import tempfile
import unittest
import wave

import numpy as np
import pytest


VOICE_A = "en-US-GuyNeural"
VOICE_B = "en-US-JennyNeural"

ENROLL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Speaker verification identifies household members by voice.",
]
VERIFY_TEXT_A = "Voice commands are only accepted from authorised speakers."
VERIFY_TEXT_B = "This guest voice should not be accepted by the system."


def _tts_to_wav(text: str, voice: str, path: str, timeout: int = 60) -> None:
    mp3 = path.replace(".wav", ".mp3")
    subprocess.run(
        ["edge-tts", "--voice", voice, "--text", text, "--write-media", mp3],
        check=True, timeout=timeout, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", "-f", "wav", path],
        check=True, timeout=30, capture_output=True,
    )
    os.unlink(mp3)


def _skip_if_no_tools():
    for tool in ["edge-tts", "ffmpeg"]:
        try:
            subprocess.run([tool, "--version"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip(f"{tool} not available")


def _wav_to_pcm_bytes(path: str) -> bytes:
    """Read a WAV file and return raw PCM bytes."""
    with wave.open(path, "rb") as wf:
        return wf.readframes(wf.getnframes())


@pytest.fixture(scope="module")
def audio_clips(tmp_path_factory):
    _skip_if_no_tools()
    d = tmp_path_factory.mktemp("verifier_audio")

    clips = {}
    for i, text in enumerate(ENROLL_TEXTS):
        path = str(d / f"enroll_A{i+1}.wav")
        _tts_to_wav(text, VOICE_A, path)
        clips[f"enroll_A{i+1}"] = path

    path_a = str(d / "verify_A.wav")
    _tts_to_wav(VERIFY_TEXT_A, VOICE_A, path_a)
    clips["verify_A"] = path_a

    path_b = str(d / "verify_B.wav")
    _tts_to_wav(VERIFY_TEXT_B, VOICE_B, path_b)
    clips["verify_B"] = path_b

    return clips


@pytest.fixture(scope="module")
def enrolled_verifier(audio_clips, tmp_path_factory):
    """Return a SpeakerVerifier with voice A enrolled."""
    profiles_path = str(tmp_path_factory.mktemp("profiles") / "profiles.json")
    from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
    v = SpeakerVerifier(config={
        "model": "wespeaker-resnet34",
        "threshold": 0.45,
        "fail_open": False,
        "profiles_path": profiles_path,
    })
    v.enroll("Resident_A", [audio_clips["enroll_A1"], audio_clips["enroll_A2"]])
    return v, audio_clips


class TestE2EVerifier:
    def test_enrolled_voice_accepted(self, enrolled_verifier):
        v, clips = enrolled_verifier
        pcm = _wav_to_pcm_bytes(clips["verify_A"])
        result = v.verify(pcm)
        assert result is True, "Enrolled voice A should be accepted"

    def test_guest_voice_rejected(self, enrolled_verifier):
        v, clips = enrolled_verifier
        pcm = _wav_to_pcm_bytes(clips["verify_B"])
        result = v.verify(pcm)
        assert result is False, "Guest voice B should be rejected"

    def test_empty_bytes_failclosed(self, enrolled_verifier):
        v, clips = enrolled_verifier
        # empty chunk — embed will fail; fail_open=False → rejected
        result = v.verify(b"")
        assert result is False

    def test_list_profiles_shows_enrolled_name(self, enrolled_verifier):
        v, _ = enrolled_verifier
        assert "Resident_A" in v.list_profiles()

    def test_cosine_scores_printed(self, enrolled_verifier):
        """Print actual scores for PR 'How to test' section."""
        v, clips = enrolled_verifier
        from speakeronnx import cosine

        emb_a = v._get_embedder().embed(clips["verify_A"])
        emb_b = v._get_embedder().embed(clips["verify_B"])
        profiles = v._get_profiles()
        profile_emb = profiles["Resident_A"]

        score_a = cosine(emb_a, profile_emb)
        score_b = cosine(emb_b, profile_emb)

        print(f"\n[e2e verifier] score(enrolled_A, profile)={score_a:.4f}")
        print(f"[e2e verifier] score(guest_B, profile)={score_b:.4f}")
        print(f"[e2e verifier] threshold=0.45  A_accepted={score_a>=0.45}  B_accepted={score_b>=0.45}")

        assert score_a > score_b, (
            f"Enrolled voice score {score_a:.4f} should be > guest score {score_b:.4f}"
        )
