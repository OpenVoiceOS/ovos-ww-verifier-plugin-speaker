"""The OVOS-SPEAKER-1 surface, driven over a real message bus.

A real ``ovos-messagebus`` process on a free port, a real ``MessageBusClient``
inside the verifier, and a second real ``MessageBusClient`` playing the
administration client. Every assertion below is on a payload that travelled
over a websocket and came back, not on a return value.

The one thing that is not real is the embedding model. ``speakeronnx``
downloads an ONNX model on first use, which a test suite cannot do offline, so
a deterministic stand-in embedder is injected exactly as ``test_ovoscope_e2e``
already does. The bus, the client, the topic names, the payload decoding, the
base64 WAV handling, the gate and the ``response`` derivation are all the
shipped ones.
"""
import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import wave

import numpy as np
from ovos_bus_client import Message, MessageBusClient

from ovos_ww_verifier_plugin_speaker import SpeakerVerifier

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _b64_wav(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A real, readable WAV of silence, for the size and shape cases."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class _FakeEmbedder:
    """A stand-in ``SpeakerEmbedder``: one vector per file, deterministic."""

    def __init__(self, vectors, default):
        self._vectors = vectors
        self._default = np.array(default, dtype=np.float32)

    def embed(self, wav_path):
        size = os.path.getsize(wav_path)
        vec = self._vectors.get(size)
        if vec is None:
            return self._default
        return np.array(vec, dtype=np.float32)


class _RealBusCase(unittest.TestCase):
    """One real bus daemon and two real clients for the whole class."""

    @classmethod
    def setUpClass(cls):
        # A hard requirement, not a best-effort skip: these tests exist to
        # prove the surface over a real websocket, and a skip here would read
        # as a pass while nothing was driven at all. ovos-messagebus is in the
        # `test` extra for this reason.
        if shutil.which("ovos-messagebus") is None:
            raise AssertionError(
                "ovos-messagebus is not installed; install the 'test' extra")
        cls.port = _free_port()
        cls.tmp = tempfile.mkdtemp(prefix="ovos-speaker-bus-")
        env = dict(os.environ)
        env["XDG_CONFIG_HOME"] = os.path.join(cls.tmp, "config")
        env["XDG_DATA_HOME"] = os.path.join(cls.tmp, "data")
        env["XDG_CACHE_HOME"] = os.path.join(cls.tmp, "cache")
        os.makedirs(os.path.join(env["XDG_CONFIG_HOME"], "mycroft"),
                    exist_ok=True)
        with open(os.path.join(env["XDG_CONFIG_HOME"], "mycroft",
                               "mycroft.conf"), "w") as handle:
            json.dump({"lang": "en-us",
                       "websocket": {"host": "127.0.0.1", "port": cls.port,
                                     "route": "/core"}}, handle)
        cls.log = open(os.path.join(cls.tmp, "bus.log"), "wb")
        cls.proc = subprocess.Popen(["ovos-messagebus"], env=env,
                                    stdout=cls.log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        deadline = time.time() + 30
        while time.time() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", cls.port)) == 0:
                    break
            time.sleep(0.25)
        else:
            raise AssertionError(
                f"the real bus did not listen on 127.0.0.1:{cls.port}")

        cls.client = MessageBusClient(host="127.0.0.1", port=cls.port,
                                      route="/core")
        cls.client.run_in_thread()
        assert cls.client.connected_event.wait(15), "client did not connect"

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        except Exception:
            pass
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.log.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- per-test verifier ------------------------------------------------

    def _verifier(self, **cfg):
        """A verifier serving the surface on its own real bus client."""
        profiles_path = os.path.join(
            tempfile.mkdtemp(dir=self.tmp), "profiles.json")
        config = {"profiles_path": profiles_path, "bus_surface": False}
        config.update(cfg)
        verifier = SpeakerVerifier(config=config)
        bus = MessageBusClient(host="127.0.0.1", port=self.port, route="/core")
        bus.run_in_thread()
        self.assertTrue(bus.connected_event.wait(15),
                        "the verifier's bus client did not connect")
        verifier.attach_bus(bus)
        self.addCleanup(bus.close)
        return verifier

    def _ask(self, topic, data, timeout=20):
        reply = self.client.wait_for_response(
            Message(topic, data), reply_type=topic + ".response",
            timeout=timeout)
        self.assertIsNotNone(reply, f"no answer on {topic}.response")
        return reply


class TestTheSurfaceAnswers(_RealBusCase):
    """Every topic answers, over the websocket, on its own .response."""

    def test_list_on_an_empty_roster_is_an_empty_list(self):
        self._verifier()
        reply = self._ask("ovos.speaker.list", {})
        self.assertEqual(reply.data["speakers"], [])
        self.assertIsNone(reply.data["error"])

    def test_the_reply_topic_is_the_request_topic_plus_response(self):
        """OVOS-MSG-1 5.3: a response is a reply suffixed with .response."""
        self._verifier()
        reply = self._ask("ovos.speaker.list", {})
        self.assertEqual(reply.msg_type, "ovos.speaker.list.response")

    def test_the_reply_swaps_the_routing_pair(self):
        """OVOS-MSG-1 5.2: the new source is the request's destination.

        The handler must derive the reply from the request rather than name
        itself, and this is the observable that says it did.
        """
        self._verifier()
        request = Message("ovos.speaker.list", {},
                          {"source": "control-panel",
                           "destination": "speaker-roster"})
        reply = self.client.wait_for_response(
            request, reply_type="ovos.speaker.list.response", timeout=20)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.context.get("source"), "speaker-roster")
        self.assertEqual(reply.context.get("destination"), "control-panel")

    def test_the_session_of_the_request_reaches_the_reply(self):
        """OVOS-MSG-1 4.1: session propagates onto every derived Message."""
        self._verifier()
        request = Message("ovos.speaker.list", {},
                          {"session": {"session_id": "sess-42"}})
        reply = self.client.wait_for_response(
            request, reply_type="ovos.speaker.list.response", timeout=20)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.context["session"]["session_id"], "sess-42")


class TestTheGateIsClosedByDefault(_RealBusCase):
    """The mutating topics refuse, loudly, unless the deployment opens them."""

    def test_enroll_is_refused_and_says_so(self):
        verifier = self._verifier()
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice",
                           "clips": [_b64_wav(os.path.join(
                               _FIXTURES, "speaker_a_enroll_1.wav"))]})
        self.assertEqual(reply.data["error"], "enrollment disabled")
        self.assertFalse(reply.data["enrolled"])
        self.assertEqual(verifier.list_profiles(), [],
                         "a refused enrollment must not touch the roster")

    def test_delete_is_refused_and_says_so(self):
        self._verifier()
        reply = self._ask("ovos.speaker.delete", {"name": "Alice"})
        self.assertEqual(reply.data["error"], "enrollment disabled")
        self.assertFalse(reply.data["removed"])

    def test_a_refusal_is_an_answer_and_not_silence(self):
        """A silent refusal reads exactly like an absent service."""
        self._verifier()
        reply = self.client.wait_for_response(
            Message("ovos.speaker.enroll", {"name": "Alice", "clips": ["x"]}),
            reply_type="ovos.speaker.enroll.response", timeout=20)
        self.assertIsNotNone(reply, "the closed gate stayed silent")

    def test_list_and_verify_are_not_gated(self):
        """Neither changes the roster, so neither is behind the gate."""
        self._verifier()
        self.assertIsNone(self._ask("ovos.speaker.list", {}).data["error"])
        clip = base64.b64encode(_silence_wav(0.5)).decode("ascii")
        reply = self._ask("ovos.speaker.verify", {"clip": clip})
        self.assertIsNone(reply.data["error"])


class TestEnrollmentWithTheGateOpen(_RealBusCase):
    """With the deployment's consent, the roster changes over the bus."""

    def _open_verifier(self):
        verifier = self._verifier(allow_bus_enrollment=True)
        verifier._embedder = _FakeEmbedder({}, [1.0, 0.0, 0.0])
        return verifier

    def test_enroll_then_list_shows_the_name(self):
        verifier = self._open_verifier()
        clips = [_b64_wav(os.path.join(_FIXTURES, "speaker_a_enroll_1.wav")),
                 _b64_wav(os.path.join(_FIXTURES, "speaker_a_enroll_2.wav"))]
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice", "clips": clips})
        self.assertIsNone(reply.data["error"])
        self.assertTrue(reply.data["enrolled"])
        self.assertEqual(reply.data["clips"], 2)
        self.assertEqual(self._ask("ovos.speaker.list", {}).data["speakers"],
                         ["Alice"])
        self.assertEqual(verifier.list_profiles(), ["Alice"])

    def test_enrolling_a_name_twice_replaces_the_profile(self):
        """OVOS-SPEAKER-1 4.1: a second enrollment replaces, never fails."""
        self._open_verifier()
        clip = [_b64_wav(os.path.join(_FIXTURES, "speaker_a_enroll_1.wav"))]
        self._ask("ovos.speaker.enroll", {"name": "Alice", "clips": clip})
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice", "clips": clip})
        self.assertIsNone(reply.data["error"])
        self.assertTrue(reply.data["enrolled"])
        self.assertEqual(self._ask("ovos.speaker.list", {}).data["speakers"],
                         ["Alice"])

    def test_delete_removes_the_profile(self):
        self._open_verifier()
        clip = [_b64_wav(os.path.join(_FIXTURES, "speaker_a_enroll_1.wav"))]
        self._ask("ovos.speaker.enroll", {"name": "Alice", "clips": clip})
        reply = self._ask("ovos.speaker.delete", {"name": "Alice"})
        self.assertIsNone(reply.data["error"])
        self.assertTrue(reply.data["removed"])
        self.assertEqual(self._ask("ovos.speaker.list", {}).data["speakers"],
                         [])

    def test_deleting_a_name_nobody_enrolled_is_not_an_error(self):
        """OVOS-SPEAKER-1 4.2: the roster is in the state the client asked."""
        self._open_verifier()
        reply = self._ask("ovos.speaker.delete", {"name": "Nobody"})
        self.assertIsNone(reply.data["error"])
        self.assertFalse(reply.data["removed"])


class TestVerifyAnswers(_RealBusCase):
    """The read-only question, and what it says about an empty roster."""

    def test_an_empty_roster_is_named_as_such(self):
        """A null speaker also means "nobody enrolled matches", so the empty
        roster gets its own field (OVOS-SPEAKER-1 4.4)."""
        self._verifier(fail_open=True)
        clip = base64.b64encode(_silence_wav(0.5)).decode("ascii")
        reply = self._ask("ovos.speaker.verify", {"clip": clip})
        self.assertTrue(reply.data["roster_empty"])
        self.assertIsNone(reply.data["speaker"])
        self.assertIsNone(reply.data["score"])
        self.assertTrue(reply.data["accepted"],
                        "fail_open=True accepts while nobody is enrolled")

    def test_the_empty_roster_answer_follows_the_deployment(self):
        """The same roster, the opposite answer, because accepted is what the
        verifier would tell the audio input service."""
        self._verifier(fail_open=False)
        clip = base64.b64encode(_silence_wav(0.5)).decode("ascii")
        reply = self._ask("ovos.speaker.verify", {"clip": clip})
        self.assertTrue(reply.data["roster_empty"])
        self.assertFalse(reply.data["accepted"])

    def test_an_enrolled_voice_is_named_and_accepted(self):
        verifier = self._verifier(allow_bus_enrollment=True, threshold=0.5)
        verifier._embedder = _FakeEmbedder({}, [1.0, 0.0, 0.0])
        clip = _b64_wav(os.path.join(_FIXTURES, "speaker_a_enroll_1.wav"))
        self._ask("ovos.speaker.enroll", {"name": "Alice", "clips": [clip]})
        reply = self._ask("ovos.speaker.verify",
                          {"clip": _b64_wav(os.path.join(
                              _FIXTURES, "speaker_a_verify.wav"))})
        self.assertEqual(reply.data["speaker"], "Alice")
        self.assertTrue(reply.data["accepted"])
        self.assertFalse(reply.data["roster_empty"])
        self.assertGreaterEqual(reply.data["score"], 0.5)

    def test_a_voice_that_does_not_match_is_named_and_refused(self):
        verifier = self._verifier(allow_bus_enrollment=True, threshold=0.5)
        guest = os.path.join(_FIXTURES, "speaker_b_guest.wav")
        enroll = os.path.join(_FIXTURES, "speaker_a_enroll_1.wav")
        verifier._embedder = _FakeEmbedder(
            {os.path.getsize(guest): [0.0, 1.0, 0.0]}, [1.0, 0.0, 0.0])
        self._ask("ovos.speaker.enroll",
                  {"name": "Alice", "clips": [_b64_wav(enroll)]})
        reply = self._ask("ovos.speaker.verify", {"clip": _b64_wav(guest)})
        self.assertEqual(reply.data["speaker"], "Alice")
        self.assertFalse(reply.data["accepted"])
        self.assertFalse(reply.data["roster_empty"])

    def test_verify_does_not_change_the_roster(self):
        verifier = self._verifier()
        clip = base64.b64encode(_silence_wav(0.5)).decode("ascii")
        self._ask("ovos.speaker.verify", {"clip": clip})
        self.assertEqual(verifier.list_profiles(), [])


class TestTheErrorVocabulary(_RealBusCase):
    """OVOS-SPEAKER-1 7: the exact strings a client acts on."""

    def test_audio_that_is_not_a_wav_is_a_bad_clip(self):
        self._verifier(allow_bus_enrollment=True)
        junk = base64.b64encode(b"this is not a wav file").decode("ascii")
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice", "clips": [junk]})
        self.assertEqual(reply.data["error"], "bad clip")
        self.assertFalse(reply.data["enrolled"])

    def test_audio_that_is_not_base64_is_a_bad_clip(self):
        self._verifier(allow_bus_enrollment=True)
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice", "clips": ["not base64 at all!!"]})
        self.assertEqual(reply.data["error"], "bad clip")

    def test_no_clips_is_no_audio(self):
        self._verifier(allow_bus_enrollment=True)
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice", "clips": []})
        self.assertEqual(reply.data["error"], "no audio")

    def test_a_verify_with_no_clip_is_no_audio(self):
        self._verifier()
        reply = self._ask("ovos.speaker.verify", {})
        self.assertEqual(reply.data["error"], "no audio")
        self.assertFalse(reply.data["accepted"],
                         "nothing was measured, so nothing was accepted")

    def test_a_missing_name_is_a_bad_request(self):
        self._verifier(allow_bus_enrollment=True)
        clip = base64.b64encode(_silence_wav(0.2)).decode("ascii")
        reply = self._ask("ovos.speaker.enroll", {"clips": [clip]})
        self.assertEqual(reply.data["error"], "bad request")

    def test_audio_past_the_bound_is_too_large(self):
        self._verifier(allow_bus_enrollment=True, max_audio_bytes=1024)
        clip = base64.b64encode(_silence_wav(1.0)).decode("ascii")
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice", "clips": [clip]})
        self.assertEqual(reply.data["error"], "clip too large")

    def test_a_failed_enrollment_reports_enrolled_false(self):
        """OVOS-SPEAKER-1 7: an error and a truthful outcome, never both."""
        verifier = self._verifier(allow_bus_enrollment=True)
        junk = base64.b64encode(b"nope").decode("ascii")
        reply = self._ask("ovos.speaker.enroll",
                          {"name": "Alice", "clips": [junk]})
        self.assertFalse(reply.data["enrolled"])
        self.assertEqual(reply.data["clips"], 0)
        self.assertEqual(verifier.list_profiles(), [])


if __name__ == "__main__":
    unittest.main()
