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
is passed via stdlib wave loading internally).  Audio shorter than ~0.5 seconds
may produce unreliable embeddings; prefer ~1–5 seconds of post-WW audio.

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

from speakeronnx import SpeakerEmbedder, cosine
from ovos_plugin_manager.templates.hotwords import HotWordVerifier

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

    def __init__(self, config: Optional[Dict] = None):
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedder(self) -> SpeakerEmbedder:
        if self._embedder is None:
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

        for name, profile_emb in profiles.items():
            score = cosine(emb, profile_emb)
            if score >= self._threshold_for(name):
                return True
        return False

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
