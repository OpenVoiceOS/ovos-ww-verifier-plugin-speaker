"""Unit tests for the OVOS speaker verifier plugin.

No real model downloads; speakeronnx is mocked at the embed level.
"""

import json
import os
import struct
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

import numpy as np


def _make_wav_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    """Generate synthetic 16-bit PCM WAV bytes (440 Hz sine)."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()


def _random_emb(dim: int = 256) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestProfilePersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._path = os.path.join(self._tmp, "profiles.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_verifier(self, extra_cfg=None):
        from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
        cfg = {"profiles_path": self._path, "model": "wespeaker-resnet34"}
        if extra_cfg:
            cfg.update(extra_cfg)
        return SpeakerVerifier(config=cfg)

    def test_empty_profiles_returns_empty_list(self):
        v = self._make_verifier()
        self.assertEqual(v.list_profiles(), [])

    def test_enroll_persists_profile(self):
        """Enroll with a mocked embedder; profile should be saved to disk."""
        fake_emb = _random_emb()

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = fake_emb
            v = self._make_verifier()
            # Provide a real wav path (content ignored — embed is mocked)
            fd, wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                emb = v.enroll("Alice", [wav, wav])
            finally:
                os.unlink(wav)

        self.assertIn("Alice", v.list_profiles())
        # Stored on disk
        with open(self._path) as f:
            raw = json.load(f)
        self.assertIn("Alice", raw)
        self.assertEqual(len(raw["Alice"]), 256)

    def test_enroll_mean_of_multiple_clips(self):
        """Mean embedding of two identical clips == the clip embedding."""
        fake_emb = _random_emb()

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = fake_emb
            v = self._make_verifier()
            fd, wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                emb = v.enroll("Bob", [wav, wav])
            finally:
                os.unlink(wav)

        # mean of [fake_emb, fake_emb] normalised should equal fake_emb
        np.testing.assert_allclose(emb, fake_emb, atol=1e-5)

    def test_remove_profile(self):
        fake_emb = _random_emb()
        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = fake_emb
            v = self._make_verifier()
            fd, wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                v.enroll("Alice", [wav])
            finally:
                os.unlink(wav)
        self.assertIn("Alice", v.list_profiles())
        v.remove_profile("Alice")
        self.assertNotIn("Alice", v.list_profiles())

    def test_remove_nonexistent_profile_returns_false(self):
        v = self._make_verifier()
        self.assertFalse(v.remove_profile("Nobody"))


class TestVerifyLogic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._path = os.path.join(self._tmp, "profiles.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_verifier(self, extra_cfg=None):
        from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
        cfg = {"profiles_path": self._path, "model": "wespeaker-resnet34",
               "threshold": 0.45}
        if extra_cfg:
            cfg.update(extra_cfg)
        return SpeakerVerifier(config=cfg)

    def _enroll_profile(self, verifier, name: str, emb: np.ndarray):
        """Directly write a profile without going through SpeakerEmbedder."""
        from ovos_ww_verifier_plugin_speaker import _load_profiles, _save_profiles
        raw = _load_profiles(self._path)
        raw[name] = emb.tolist()
        _save_profiles(raw, self._path)
        verifier._invalidate_profile_cache()

    def test_fail_open_no_profiles(self):
        """With no profiles and fail_open=True, verify returns True."""
        v = self._make_verifier({"fail_open": True})
        chunk = _make_wav_bytes()
        self.assertTrue(v.verify(chunk))

    def test_fail_closed_no_profiles(self):
        """With no profiles and fail_open=False, verify returns False."""
        v = self._make_verifier({"fail_open": False})
        chunk = _make_wav_bytes()
        self.assertFalse(v.verify(chunk))

    def test_accept_enrolled_speaker(self):
        """If embedder returns the enrolled embedding, verify should accept."""
        profile_emb = _random_emb()
        v = self._make_verifier({"threshold": 0.45})
        self._enroll_profile(v, "Alice", profile_emb)

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = profile_emb  # exact match
            v._embedder = instance
            chunk = _make_wav_bytes()
            self.assertTrue(v.verify(chunk))

    def test_reject_unknown_speaker(self):
        """Embedding orthogonal to all profiles should be rejected."""
        profile_emb = np.array([1.0] + [0.0] * 255, dtype=np.float32)
        unknown_emb = np.array([0.0, 1.0] + [0.0] * 254, dtype=np.float32)
        v = self._make_verifier({"threshold": 0.45})
        self._enroll_profile(v, "Alice", profile_emb)

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = unknown_emb
            v._embedder = instance
            chunk = _make_wav_bytes()
            with patch("ovos_ww_verifier_plugin_speaker.LOG.info") as info:
                self.assertFalse(v.verify(chunk))
        line = next(c.args[0] for c in info.call_args_list if "speaker verifier rejected" in c.args[0])
        self.assertIn("best match Alice score 0.000", line)
        self.assertIn("threshold 0.45", line)

    def test_multi_profile_any_match_accepts(self):
        """If any enrolled profile matches, verify accepts."""
        alice_emb = _random_emb()
        bob_emb = _random_emb()
        # Make caller look like Bob
        caller_emb = bob_emb.copy()
        v = self._make_verifier({"threshold": 0.45})
        self._enroll_profile(v, "Alice", alice_emb)
        self._enroll_profile(v, "Bob", bob_emb)

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = caller_emb
            v._embedder = instance
            chunk = _make_wav_bytes()
            self.assertTrue(v.verify(chunk))

    def test_per_profile_threshold_override(self):
        """Per-profile threshold overrides global threshold."""
        # Profile with 0.0 cosine to caller — would be rejected globally
        profile_emb = np.array([1.0] + [0.0] * 255, dtype=np.float32)
        caller_emb = np.array([0.0, 1.0] + [0.0] * 254, dtype=np.float32)  # cosine=0
        v = self._make_verifier({
            "threshold": 0.45,
            "per_profile_thresholds": {"GuestPass": -1.0},  # accept any score
        })
        self._enroll_profile(v, "GuestPass", profile_emb)

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = caller_emb
            v._embedder = instance
            chunk = _make_wav_bytes()
            self.assertTrue(v.verify(chunk))

    def test_malformed_audio_applies_fail_open(self):
        """If embed raises (too-short audio), fail_open policy is applied."""
        profile_emb = _random_emb()
        v = self._make_verifier({"fail_open": True})
        self._enroll_profile(v, "Alice", profile_emb)

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.side_effect = ValueError("too short")
            v._embedder = instance
            # Pass empty bytes
            self.assertTrue(v.verify(b""))

    def test_malformed_audio_applies_fail_closed(self):
        """If embed raises and fail_open=False, verify returns False."""
        profile_emb = _random_emb()
        v = self._make_verifier({"fail_open": False})
        self._enroll_profile(v, "Alice", profile_emb)

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.side_effect = ValueError("too short")
            v._embedder = instance
            self.assertFalse(v.verify(b""))

    def test_profile_cache_invalidated_after_enroll(self):
        """list_profiles() reflects newly enrolled name immediately."""
        v = self._make_verifier()
        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = _random_emb()
            v._embedder = instance
            fd, wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                v.enroll("Charlie", [wav])
            finally:
                os.unlink(wav)
        self.assertIn("Charlie", v.list_profiles())


class TestEnrollEdgeCases(unittest.TestCase):
    """Edge cases in enrollment and configuration defaults."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._path = os.path.join(self._tmp, "profiles.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_verifier(self, extra_cfg=None):
        from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
        cfg = {"profiles_path": self._path, "model": "wespeaker-resnet34"}
        if extra_cfg:
            cfg.update(extra_cfg)
        return SpeakerVerifier(config=cfg)

    def test_enroll_single_clip(self):
        """Enroll with one clip; stored profile equals that clip's embedding."""
        emb = _random_emb()
        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = emb
            v = self._make_verifier()
            fd, wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                result = v.enroll("Solo", [wav])
            finally:
                os.unlink(wav)
        np.testing.assert_allclose(result, emb, atol=1e-5)

    def test_enroll_overwrites_existing_profile(self):
        """Re-enrolling the same name replaces the old profile entry."""
        emb1 = _random_emb()
        emb2 = _random_emb()
        embs = [emb1, emb2]

        def _side(path):
            return embs.pop(0)

        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.side_effect = _side
            v = self._make_verifier()
            for _ in range(2):
                fd, wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    v.enroll("Alice", [wav])
                finally:
                    os.unlink(wav)
        # Only one "Alice" entry should be present
        self.assertEqual(v.list_profiles().count("Alice"), 1)

    def test_remove_profile_leaves_others(self):
        """Removing one profile does not affect other enrolled profiles."""
        emb = _random_emb()
        with patch("speakeronnx.SpeakerEmbedder") as MockEmb:
            instance = MockEmb.return_value
            instance.embed.return_value = emb
            v = self._make_verifier()
            for name in ("Alice", "Bob"):
                fd, wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    v.enroll(name, [wav])
                finally:
                    os.unlink(wav)
        v.remove_profile("Alice")
        self.assertNotIn("Alice", v.list_profiles())
        self.assertIn("Bob", v.list_profiles())

    def test_default_config_values(self):
        """SpeakerVerifier uses documented defaults when config is minimal."""
        from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
        v = SpeakerVerifier(config={"profiles_path": self._path})
        self.assertAlmostEqual(v._threshold, 0.45)
        self.assertTrue(v._fail_open)
        self.assertEqual(v._model_alias, "wespeaker-resnet34")
        self.assertEqual(v._sample_rate, 16000)
        self.assertEqual(v._sample_width, 2)
        self.assertEqual(v._channels, 1)


if __name__ == "__main__":
    unittest.main()
