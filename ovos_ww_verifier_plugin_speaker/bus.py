"""The OVOS-SPEAKER-1 bus surface over a :class:`SpeakerVerifier`.

This module holds the four handlers the specification fixes —
``ovos.speaker.enroll``, ``ovos.speaker.delete``, ``ovos.speaker.list`` and
``ovos.speaker.verify`` — and nothing else. The speaker logic stays in the
verifier: a handler decodes a payload, calls a method the ``ovos-speaker-enroll``
console script also calls, and answers.

Why the handlers live here and not in the plugin template
---------------------------------------------------------
``HotWordVerifier`` in ovos-plugin-manager takes a config and nothing else, and
the audio input service builds a verifier with ``plug_cls(config=cfg)``. Giving
the template a bus is a base-class change that core merges and every installed
verifier carries. This surface needs none of that: the verifier attaches a bus
client of its own, which is what ``RedisAudioTransformer`` in
ovos-audio-transformer-redis-publish already does.

Why the mutating topics are closed by default
---------------------------------------------
The plugin exists to refuse a wake word that does not come from an enrolled
voice. An enrollment topic lets anything that can publish on the bus add a
trusted voice, and OVOS-MSG-1 section 3.4 makes ``source`` an opaque string the
caller chose, so a handler cannot tell an administration client from any other
publisher and must not try. ``ovos.speaker.enroll`` and ``ovos.speaker.delete``
are therefore behind ``allow_bus_enrollment``, which is off unless the
deployment turns it on, and a closed gate answers ``enrollment disabled``
rather than staying silent: a silent refusal reads exactly like an absent
service.
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
import wave
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ovos_utils.log import LOG

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ovos_ww_verifier_plugin_speaker import SpeakerVerifier

# OVOS-SPEAKER-1 section 2
TOPIC_ENROLL = "ovos.speaker.enroll"
TOPIC_DELETE = "ovos.speaker.delete"
TOPIC_LIST = "ovos.speaker.list"
TOPIC_VERIFY = "ovos.speaker.verify"

# OVOS-SPEAKER-1 section 7
ERR_DISABLED = "enrollment disabled"
ERR_BAD_REQUEST = "bad request"
ERR_BAD_CLIP = "bad clip"
ERR_TOO_LARGE = "clip too large"
ERR_NO_AUDIO = "no audio"
ERR_INTERNAL = "internal error"

# The default bound of section 4.5. Five seconds of 16 kHz 16-bit mono is about
# 160 kB, so this accepts roughly fifty such clips before it refuses.
DEFAULT_MAX_AUDIO_BYTES = 8 * 1024 * 1024


class _ClipError(Exception):
    """A clip the specification says to refuse, carrying its error string."""

    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


class SpeakerBusSurface:
    """The four OVOS-SPEAKER-1 handlers, bound to one verifier and one bus."""

    def __init__(self, verifier: "SpeakerVerifier", bus,
                 allow_enrollment: bool = False,
                 max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES):
        self.verifier = verifier
        self.bus = bus
        self.allow_enrollment = bool(allow_enrollment)
        self.max_audio_bytes = int(max_audio_bytes)

    def register(self) -> None:
        """Subscribe to all four topics.

        All four or none: OVOS-SPEAKER-1 section 2. A client cannot discover a
        partial surface, and a request nobody answers is indistinguishable from
        an absent service.
        """
        self.bus.on(TOPIC_ENROLL, self.handle_enroll)
        self.bus.on(TOPIC_DELETE, self.handle_delete)
        self.bus.on(TOPIC_LIST, self.handle_list)
        self.bus.on(TOPIC_VERIFY, self.handle_verify)
        LOG.info("speaker bus surface registered: enroll and delete are "
                 f"{'open' if self.allow_enrollment else 'closed'}")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def handle_enroll(self, message) -> None:
        data = message.data or {}
        name = data.get("name")
        clips = data.get("clips")

        if not self.allow_enrollment:
            self._answer(message, {"name": name, "enrolled": False, "clips": 0,
                                   "error": ERR_DISABLED})
            return
        if not isinstance(name, str) or not name:
            self._answer(message, {"name": name, "enrolled": False, "clips": 0,
                                   "error": ERR_BAD_REQUEST})
            return
        if clips is None or (isinstance(clips, list) and not clips):
            self._answer(message, {"name": name, "enrolled": False, "clips": 0,
                                   "error": ERR_NO_AUDIO})
            return
        if not isinstance(clips, list):
            self._answer(message, {"name": name, "enrolled": False, "clips": 0,
                                   "error": ERR_BAD_REQUEST})
            return

        paths: List[str] = []
        try:
            paths = self._decode_clips(clips)
            self.verifier.enroll(name, paths)
        except _ClipError as err:
            self._answer(message, {"name": name, "enrolled": False, "clips": 0,
                                   "error": err.error})
            return
        except Exception:
            LOG.exception(f"speaker enrollment failed for {name!r}")
            self._answer(message, {"name": name, "enrolled": False, "clips": 0,
                                   "error": ERR_INTERNAL})
            return
        finally:
            _unlink_all(paths)

        self._answer(message, {"name": name, "enrolled": True,
                               "clips": len(paths), "error": None})

    def handle_delete(self, message) -> None:
        data = message.data or {}
        name = data.get("name")

        if not self.allow_enrollment:
            self._answer(message, {"name": name, "removed": False,
                                   "error": ERR_DISABLED})
            return
        if not isinstance(name, str) or not name:
            self._answer(message, {"name": name, "removed": False,
                                   "error": ERR_BAD_REQUEST})
            return
        try:
            removed = self.verifier.remove_profile(name)
        except Exception:
            LOG.exception(f"speaker deletion failed for {name!r}")
            self._answer(message, {"name": name, "removed": False,
                                   "error": ERR_INTERNAL})
            return
        # Deleting a name that is not enrolled is not an error: the roster is
        # in the state the client asked for (OVOS-SPEAKER-1 section 4.2).
        self._answer(message, {"name": name, "removed": bool(removed),
                               "error": None})

    def handle_list(self, message) -> None:
        try:
            speakers = self.verifier.list_profiles()
        except Exception:
            LOG.exception("speaker listing failed")
            self._answer(message, {"speakers": [], "error": ERR_INTERNAL})
            return
        self._answer(message, {"speakers": list(speakers), "error": None})

    def handle_verify(self, message) -> None:
        data = message.data or {}
        clip = data.get("clip")

        if clip is None:
            self._answer(message, self._verify_failure(ERR_NO_AUDIO))
            return
        if not isinstance(clip, str):
            self._answer(message, self._verify_failure(ERR_BAD_REQUEST))
            return

        paths: List[str] = []
        try:
            paths = self._decode_clips([clip])
            speaker, score, accepted, roster_empty = self.verifier.identify(paths[0])
        except _ClipError as err:
            self._answer(message, self._verify_failure(err.error))
            return
        except Exception:
            LOG.exception("speaker verification failed")
            self._answer(message, self._verify_failure(ERR_INTERNAL))
            return
        finally:
            _unlink_all(paths)

        self._answer(message, {"speaker": speaker, "score": score,
                               "accepted": bool(accepted),
                               "roster_empty": bool(roster_empty),
                               "error": None})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verify_failure(self, error: str) -> Dict[str, Any]:
        """The verify reply for a request that never reached the model.

        ``accepted`` is False rather than the fail-open answer: nothing was
        measured, so nothing was accepted.
        """
        try:
            roster_empty = not self.verifier.list_profiles()
        except Exception:
            roster_empty = True
        return {"speaker": None, "score": None, "accepted": False,
                "roster_empty": bool(roster_empty), "error": error}

    def _decode_clips(self, clips: List[Any]) -> List[str]:
        """Decode base64 WAV clips to temporary files.

        Raises :class:`_ClipError` with the specification's own error string
        for a clip that is not a string, not base64, larger than the bound, or
        not a readable WAV.
        """
        total = 0
        paths: List[str] = []
        try:
            for clip in clips:
                if not isinstance(clip, str):
                    raise _ClipError(ERR_BAD_REQUEST)
                try:
                    raw = base64.b64decode(clip, validate=True)
                except (binascii.Error, ValueError):
                    raise _ClipError(ERR_BAD_CLIP)
                total += len(raw)
                if total > self.max_audio_bytes:
                    raise _ClipError(ERR_TOO_LARGE)
                paths.append(_write_wav(raw))
        except BaseException:
            _unlink_all(paths)
            raise
        return paths

    def _answer(self, message, data: Dict[str, Any]) -> None:
        """Answer with the ``response`` derivation of OVOS-MSG-1 section 5.3.

        Derived from the request, never built from this component's identity,
        so the source and destination swap of section 5.2 and the session
        propagation of section 4.1 hold without a rule here.
        """
        self.bus.emit(message.response(data))


def _write_wav(raw: bytes) -> str:
    """Write *raw* to a temporary file, refusing anything that is not a WAV."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="ovos-speaker-clip-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
    except BaseException:
        _unlink_all([path])
        raise
    try:
        with wave.open(path, "rb") as wav:
            if wav.getnframes() == 0:
                raise _ClipError(ERR_BAD_CLIP)
    except _ClipError:
        _unlink_all([path])
        raise
    except Exception:
        # wave raises wave.Error for a bad header and EOFError for a truncated
        # file. Either way the bytes are not a clip this can enroll from.
        _unlink_all([path])
        raise _ClipError(ERR_BAD_CLIP)
    return path


def _unlink_all(paths: List[str]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass
