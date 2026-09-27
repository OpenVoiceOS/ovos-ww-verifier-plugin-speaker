"""ovos-ww-verifier-plugin-speaker — OVOS wake word verifier based on speaker identity.

Purpose
-------
Ignore wake words from speakers who are not enrolled as authorised household members.
After a wake word is detected by any HotWordEngine, this verifier extracts a speaker
embedding from the captured audio and compares it against enrolled profiles.  Only
speakers whose cosine similarity to any enrolled profile meets or exceeds the
configured threshold are accepted.

Fail-open semantics
-------------------
When no profiles are enrolled (empty roster), the verifier defaults to *accepting
all* activations — ``fail_open=True`` in config.  Set ``fail_open: false`` to
reject all activations until at least one profile is enrolled.

Enrollment
----------
Profiles are stored as a JSON file containing mean embeddings per enrolled speaker
name under the XDG data directory (``~/.local/share/ovos_speaker_verifier/``).
Each profile is the L2-normalised mean of embeddings extracted from the enrollment
WAV clips.  The more clips, the more robust the profile.

Audio input
-----------
The ``verify(chunk)`` method accepts raw 16-bit PCM bytes (any sample rate — audio
is passed via stdlib wave loading internally).  When driven by the listener the
chunk is the accumulated *wake-word* audio window (the "hey mycroft" utterance
itself), not the command that follows it.  Audio shorter than ~0.5 seconds may
produce unreliable embeddings, so a clearly-spoken wake word is the sweet spot.

Configuration
-------------
Plugin config keys (nested under the verifier's entry point in OVOS config):

``model`` (str):
    speakeronnx model alias or path.  Default ``"wespeaker-resnet34"``.
``threshold`` (float):
    Global cosine similarity acceptance threshold.  Default ``0.45``.
``fail_open`` (bool):
    Accept all activations when no profiles are enrolled.  Default ``True``.
``profiles_path`` (str, optional):
    Override default XDG path for the profiles JSON file.
``per_profile_thresholds`` (dict, optional):
    Per-name overrides, e.g. ``{"Alice": 0.5, "Bob": 0.4}``.  Falls back to
    ``threshold`` for names not listed.
``bus_surface`` (bool):
    Serve the OVOS-SPEAKER-1 topics.  Default ``True``.  When no bus is passed
    to the constructor the plugin opens a client of its own; a failure to
    connect is logged and the verifier keeps verifying.
``allow_bus_enrollment`` (bool):
    Let ``ovos.speaker.enroll`` and ``ovos.speaker.delete`` change the roster.
    Default ``False``.  While it is off both answer ``enrollment disabled``.
    Read on before turning it on: anything that can publish on the bus can then
    add or remove a trusted voice, and a handler cannot tell an administration
    client from any other publisher (OVOS-MSG-1 3.4).
``max_audio_bytes`` (int):
    The bound on the decoded audio one request may carry.  Default 8 MiB.
    Past it the answer is ``clip too large``.

Bus surface
-----------
``ovos.speaker.enroll``, ``ovos.speaker.delete``, ``ovos.speaker.list`` and
``ovos.speaker.verify``, each answered by its ``.response``, as OVOS-SPEAKER-1
fixes them.  That specification is proposed, not merged
(OpenVoiceOS/architecture#277); the topic names follow it and change with it.  The handlers are in :mod:`ovos_ww_verifier_plugin_speaker.bus` and
call the same methods the console script calls.
"""

from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

# NOTE: ``speakeronnx`` pulls in onnxruntime (a multi-second import). It is
# imported lazily inside the methods that need it so that merely loading this
# plugin class — as OPM does for every installed plugin at startup — stays cheap.
from ovos_plugin_manager.templates.hotwords import HotWordVerifier
from ovos_utils.log import LOG

if False:  # type-checking only; avoids the heavy import at module load
    from speakeronnx import SpeakerEmbedder

# ---------------------------------------------------------------------------
# XDG data directory
# ---------------------------------------------------------------------------
_XDG_DATA_HOME = os.environ.get(
    "XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")
)
_DEFAULT_PROFILES_DIR = os.path.join(_XDG_DATA_HOME, "ovos_speaker_verifier")
_DEFAULT_PROFILES_PATH = os.path.join(_DEFAULT_PROFILES_DIR, "profiles.json")


# ---------------------------------------------------------------------------
# Profile persistence helpers
# ---------------------------------------------------------------------------

def _load_profiles(path: str) -> Dict[str, List[float]]:
    """Load profiles dict {name: mean_embedding_list} from JSON."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_profiles(profiles: Dict[str, List[float]], path: str) -> None:
    """Persist profiles dict to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profiles, f)


def _bytes_to_wav_file(chunk: bytes, sample_rate: int = 16000,
                       sample_width: int = 2, channels: int = 1) -> str:
    """Write raw PCM bytes to a temporary WAV file; return the path."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(chunk)
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Main verifier class
# ---------------------------------------------------------------------------

class SpeakerVerifier(HotWordVerifier):
    """OVOS wake word verifier that accepts only enrolled speakers.

    Implements the ``HotWordVerifier`` interface from
    ``ovos_plugin_manager.templates.hotwords``.

    Parameters
    ----------
    config : dict, optional
        Plugin configuration dict.  See module docstring for keys.
    """

    def __init__(self, config: Optional[Dict] = None, bus=None):
        super().__init__(config=config)
        cfg = self.config

        self._model_alias: str = cfg.get("model", "wespeaker-resnet34")
        self._threshold: float = float(cfg.get("threshold", 0.45))
        self._fail_open: bool = bool(cfg.get("fail_open", True))
        self._per_profile_thresholds: Dict[str, float] = cfg.get(
            "per_profile_thresholds", {}
        )
        profiles_path = cfg.get("profiles_path", _DEFAULT_PROFILES_PATH)
        self._profiles_path: str = profiles_path

        # Lazy-load ONNX session on first verify call to avoid import-time cost
        self._embedder: Optional[SpeakerEmbedder] = None

        # Cache of profile embeddings: {name: np.ndarray}
        self._profiles: Optional[Dict[str, np.ndarray]] = None

        # Raw PCM audio settings for verify(chunk: bytes)
        self._sample_rate: int = int(cfg.get("sample_rate", 16000))
        self._sample_width: int = int(cfg.get("sample_width", 2))  # bytes
        self._channels: int = int(cfg.get("channels", 1))

        # OVOS-SPEAKER-1 bus surface
        self.bus = None
        self._bus_surface = None
        self._owns_bus = False
        if bus is not None:
            self.attach_bus(bus)
        elif bool(cfg.get("bus_surface", True)):
            self._connect_bus()

    # ------------------------------------------------------------------
    # The OVOS-SPEAKER-1 bus surface
    # ------------------------------------------------------------------

    def attach_bus(self, bus) -> None:
        """Serve the four OVOS-SPEAKER-1 topics on *bus*.

        The handlers live in :mod:`ovos_ww_verifier_plugin_speaker.bus` and
        call the same methods ``ovos-speaker-enroll`` calls, so the console
        script and the bus are two thin callers of one API.

        ``ovos.speaker.enroll`` and ``ovos.speaker.delete`` answer
        ``enrollment disabled`` unless ``allow_bus_enrollment`` is on. The bus
        inside a device is not authenticated and OVOS-MSG-1 section 3.4 makes
        ``source`` a string the caller chose, so a handler cannot tell an
        administration client from any other publisher; the gate is the
        boundary, and it is closed unless a deployment opens it.
        """
        from ovos_ww_verifier_plugin_speaker.bus import (
            DEFAULT_MAX_AUDIO_BYTES,
            SpeakerBusSurface,
        )

        cfg = self.config
        self.bus = bus
        self._bus_surface = SpeakerBusSurface(
            self, bus,
            allow_enrollment=bool(cfg.get("allow_bus_enrollment", False)),
            max_audio_bytes=int(cfg.get("max_audio_bytes",
                                        DEFAULT_MAX_AUDIO_BYTES)),
        )
        self._bus_surface.register()

    def _connect_bus(self) -> None:
        """Open a bus client of this plugin's own and serve the surface on it.

        ``HotWordVerifier`` takes a config and no bus, and the audio input
        service builds a verifier with ``plug_cls(config=cfg)``, so a verifier
        that wants a bus opens one. ``RedisAudioTransformer`` in
        ovos-audio-transformer-redis-publish does the same.

        A failure to connect is logged and never raised: the verifier's real
        job is to answer the audio input service, and it must keep doing that
        on a device with no bus, in the console script, and under a test.
        """
        try:
            from ovos_bus_client import MessageBusClient
            bus = MessageBusClient()
            bus.run_in_thread()
            self._owns_bus = True
            self.attach_bus(bus)
        except Exception as err:
            LOG.warning(f"speaker verifier could not open a bus client, the "
                        f"enrollment surface is not served: {err}")

    def shutdown(self) -> None:
        """Close the bus client this plugin opened, if it opened one."""
        if self._owns_bus and self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                LOG.exception("speaker verifier failed to close its bus client")
        self._owns_bus = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedder(self) -> "SpeakerEmbedder":
        if self._embedder is None:
            from speakeronnx import SpeakerEmbedder
            self._embedder = SpeakerEmbedder(model=self._model_alias)
        return self._embedder

    def _get_profiles(self) -> Dict[str, np.ndarray]:
        """Load and cache profiles from disk."""
        if self._profiles is None:
            raw = _load_profiles(self._profiles_path)
            self._profiles = {
                name: np.array(emb, dtype=np.float32)
                for name, emb in raw.items()
            }
        return self._profiles

    def _invalidate_profile_cache(self) -> None:
        self._profiles = None

    def _threshold_for(self, name: str) -> float:
        return float(self._per_profile_thresholds.get(name, self._threshold))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def verify(self, chunk: bytes) -> bool:
        """Verify whether the speaker in *chunk* is an enrolled household member.

        Parameters
        ----------
        chunk:
            Raw PCM audio bytes from the wake word detection window.
            Expected format: 16-bit signed, mono, 16 kHz (configurable via
            ``sample_rate``, ``sample_width``, ``channels`` config keys).

        Returns
        -------
        bool
            True if an enrolled speaker is recognised, False otherwise.
            Returns True unconditionally when no profiles are enrolled and
            ``fail_open=True`` (the default).
        """
        profiles = self._get_profiles()

        if not profiles:
            return self._fail_open

        wav_path = _bytes_to_wav_file(
            chunk,
            sample_rate=self._sample_rate,
            sample_width=self._sample_width,
            channels=self._channels,
        )
        try:
            embedder = self._get_embedder()
            emb = embedder.embed(wav_path)
        except Exception:
            # Malformed / too-short audio — apply fail-open/closed policy
            return self._fail_open
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        name, score, accepted = self._decide(emb, profiles)
        if accepted:
            LOG.info(f"speaker verifier accepted {name} (score {score:.3f}, threshold {self._threshold_for(name)})")
        else:
            LOG.info(f"speaker verifier rejected unknown speaker (best match {name} score {score:.3f}, "
                     f"threshold {self._threshold_for(name)})")
        return accepted

    def _decide(self, emb, profiles: Dict[str, np.ndarray]
                ) -> Tuple[str, float, bool]:
        """Score *emb* against *profiles* and say whether it is accepted.

        The one place the acceptance rule lives, so that the answer
        ``ovos.speaker.verify`` gives on the bus is the answer the audio input
        service gets, and not a second rule that can drift from it.

        @param emb: the embedding of the audio under test
        @param profiles: the enrolled roster, which must not be empty
        @return: the best matching name, its score, and whether it is accepted
        """
        from speakeronnx import cosine
        scores = {name: cosine(emb, profile_emb)
                  for name, profile_emb in profiles.items()}
        for name, score in scores.items():
            if score >= self._threshold_for(name):
                return name, float(score), True
        best = max(scores, key=scores.get)
        return best, float(scores[best]), False

    def identify(self, wav_path: str) -> Tuple[Optional[str], Optional[float],
                                               bool, bool]:
        """Say which enrolled speaker a WAV file matches.

        This is what ``ovos.speaker.verify`` answers with. It reads the roster
        and changes nothing.

        @param wav_path: a readable WAV file
        @return: the best matching name or None, its score or None, whether
                 the deployment accepts this voice, and whether the roster is
                 empty
        """
        profiles = self._get_profiles()
        if not profiles:
            # The same rule verify() applies: with nobody enrolled, the
            # deployment's fail-open setting is the answer, and there is no
            # name and no score to report.
            return None, None, self._fail_open, True
        emb = self._get_embedder().embed(wav_path)
        name, score, accepted = self._decide(emb, profiles)
        return name, score, accepted, False

    def enroll(self, name: str, wav_paths: List[str]) -> np.ndarray:
        """Enroll a speaker by extracting and averaging embeddings from WAV clips.

        Parameters
        ----------
        name:
            Display name for this speaker profile (e.g. ``"Alice"``).
        wav_paths:
            List of paths to WAV files containing speech from this speaker.
            More clips → more robust profile.  Recommended: 5–30 s total.

        Returns
        -------
        np.ndarray
            The stored L2-normalised mean embedding for this speaker.
        """
        embedder = self._get_embedder()
        embeddings = []
        for path in wav_paths:
            emb = embedder.embed(path)
            embeddings.append(emb)

        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        # Persist
        raw = _load_profiles(self._profiles_path)
        raw[name] = mean_emb.tolist()
        _save_profiles(raw, self._profiles_path)
        self._invalidate_profile_cache()

        return mean_emb

    def remove_profile(self, name: str) -> bool:
        """Remove an enrolled speaker profile by name.

        Returns True if the profile existed and was removed.
        """
        raw = _load_profiles(self._profiles_path)
        if name not in raw:
            return False
        del raw[name]
        _save_profiles(raw, self._profiles_path)
        self._invalidate_profile_cache()
        return True

    def list_profiles(self) -> List[str]:
        """Return the names of all enrolled speakers."""
        return list(_load_profiles(self._profiles_path).keys())
