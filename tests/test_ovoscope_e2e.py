"""End-to-end bus-sequence tests driving the speaker verifier through a listener.

Where ``test_e2e.py`` calls ``SpeakerVerifier.verify()`` directly with real
edge-tts/speakeronnx embeddings, these tests prove the *integration* the plugin
exists for: when wired into a real ``DinkumVoiceLoop`` as a wake-word verifier, a
rejected speaker actually **suppresses** the ``recognizer_loop:record_begin`` bus
event, and an accepted speaker lets recording proceed.

They use ovoscope's ``MiniVoiceLoop`` harness, which runs a real
``DinkumVoiceLoop`` on a ``FakeBus`` and consults the verifier chain inside
``_detect_ww``.  No speaker model is downloaded:

- The fail-open / fail-closed policy (empty roster) needs no embedder at all.
- The enrolled accept / reject paths inject a tiny fake embedder and profile so
  the real ``verify()`` cosine comparison runs deterministically and offline.
"""
import os
import tempfile
import unittest

import inspect

import numpy as np

try:
    from ovoscope.voice_loop import MiniVoiceLoop, MockHotWordEngine
    HAS_OVOSCOPE = True
except ImportError:
    HAS_OVOSCOPE = False

# Suppression only happens if the installed ovos-dinkum-listener consults the
# verifier chain inside _detect_ww (the hotword-verifier gate, dinkum >=0.6.0a1).
# Accept-path tests work regardless; reject-path tests need the gate.
try:
    from ovos_dinkum_listener.voice_loop.voice_loop import DinkumVoiceLoop
    HAS_VERIFY_GATE = "self.hotwords.verify" in inspect.getsource(
        DinkumVoiceLoop._detect_ww
    )
except Exception:
    HAS_VERIFY_GATE = False

from ovos_ww_verifier_plugin_speaker import SpeakerVerifier

SILENT_CHUNK = b"\x00" * 512

_GATE_REASON = "ovos-dinkum-listener lacks the hotword-verifier gate (<0.6.0a1)"


class _FakeEmbedder:
    """Stand-in ``SpeakerEmbedder`` returning a fixed embedding (no ONNX model)."""

    def __init__(self, vec):
        self._vec = np.array(vec, dtype=np.float32)

    def embed(self, wav_path):
        return self._vec


@unittest.skipUnless(HAS_OVOSCOPE, "ovoscope (>=0.19.0a1) not installed")
class TestSpeakerVerifierBusGate(unittest.TestCase):
    """Drive the real SpeakerVerifier through MiniVoiceLoop and assert the bus."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._profiles_path = os.path.join(self._tmp, "profiles.json")

    def _drive(self, verifier):
        """Feed PCM through a real DinkumVoiceLoop gated by *verifier*."""
        ww = MockHotWordEngine("hey_mycroft", trigger_after=3)
        with MiniVoiceLoop(
            ww_instances={"hey_mycroft": ww}, verifiers=[verifier]
        ) as vl:
            return vl, vl.feed_chunks([SILENT_CHUNK] * 5)

    def _verifier(self, **cfg):
        cfg.setdefault("profiles_path", self._profiles_path)
        return SpeakerVerifier(config=cfg)

    # -- empty roster: fail-open policy ---------------------------------

    def test_fail_open_empty_roster_accepts(self):
        """No enrolled profiles + fail_open=True → recording begins."""
        vl, msgs = self._drive(self._verifier(fail_open=True))
        vl.assert_wakeword_detected(msgs)

    @unittest.skipUnless(HAS_VERIFY_GATE, _GATE_REASON)
    def test_fail_closed_empty_roster_suppresses(self):
        """No enrolled profiles + fail_open=False → detection suppressed."""
        vl, msgs = self._drive(self._verifier(fail_open=False))
        vl.assert_wakeword_suppressed(msgs)

    # -- enrolled roster: cosine decision (offline, fake embedder) -------

    def test_enrolled_speaker_accepted(self):
        """An embedding matching an enrolled profile lets recording begin."""
        v = self._verifier(threshold=0.45)
        v._profiles = {"Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        v._embedder = _FakeEmbedder([1.0, 0.0, 0.0])  # cosine 1.0 >= 0.45
        vl, msgs = self._drive(v)
        vl.assert_wakeword_detected(msgs)

    @unittest.skipUnless(HAS_VERIFY_GATE, _GATE_REASON)
    def test_unenrolled_speaker_rejected(self):
        """An embedding below threshold for every profile suppresses recording."""
        v = self._verifier(threshold=0.45)
        v._profiles = {"Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        v._embedder = _FakeEmbedder([0.0, 1.0, 0.0])  # cosine 0.0 < 0.45
        vl, msgs = self._drive(v)
        vl.assert_wakeword_suppressed(msgs)

    @unittest.skipUnless(HAS_VERIFY_GATE, _GATE_REASON)
    def test_per_profile_threshold_rejects(self):
        """A stricter per-profile threshold can reject an otherwise-close match."""
        v = self._verifier(
            threshold=0.45, per_profile_thresholds={"Alice": 0.99}
        )
        v._profiles = {"Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        # cosine ~0.71 — above the global 0.45 but below Alice's 0.99
        v._embedder = _FakeEmbedder([1.0, 1.0, 0.0])
        vl, msgs = self._drive(v)
        vl.assert_wakeword_suppressed(msgs)

    def test_malformed_audio_applies_fail_open_policy(self):
        """An embedder error falls back to the fail-open policy (accept)."""
        class _Boom:
            def embed(self, wav_path):
                raise RuntimeError("too short")

        v = self._verifier(threshold=0.45, fail_open=True)
        v._profiles = {"Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        v._embedder = _Boom()
        vl, msgs = self._drive(v)
        vl.assert_record_begin_emitted(msgs)


if __name__ == "__main__":
    unittest.main()
