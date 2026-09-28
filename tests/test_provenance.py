"""Enrollment provenance: clip count and total seconds per profile (T-6528)."""
import json
import os
import wave

import numpy as np
import pytest

from ovos_ww_verifier_plugin_speaker import (
    SpeakerVerifier,
    _load_profiles,
    _meta_path,
)


def _write_wav(path, seconds, sample_rate=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return path


class _FakeEmbedder:
    def embed(self, path):
        return np.ones(8, dtype=np.float32)


@pytest.fixture
def verifier(tmp_path):
    v = SpeakerVerifier({"profiles_path": str(tmp_path / "profiles.json")})
    v._embedder = _FakeEmbedder()
    return v


def test_enroll_records_clip_count_and_seconds(verifier, tmp_path):
    clips = [_write_wav(str(tmp_path / f"c{i}.wav"), 2.0) for i in range(3)]
    verifier.enroll("Alice", clips)

    info = verifier.profile_info("Alice")
    assert info["clips"] == 3
    assert info["seconds"] == pytest.approx(6.0, abs=0.01)


def test_profiles_file_keeps_the_list_shape(verifier, tmp_path):
    """The embedding stays a bare list, so an older install still loads it."""
    verifier.enroll("Alice", [_write_wav(str(tmp_path / "a.wav"), 1.0)])

    raw = _load_profiles(str(tmp_path / "profiles.json"))
    assert isinstance(raw["Alice"], list)
    assert all(isinstance(x, float) for x in raw["Alice"])
    # and the loader still yields arrays
    assert verifier._get_profiles()["Alice"].dtype == np.float32


def test_provenance_lives_in_a_sidecar_file(verifier, tmp_path):
    verifier.enroll("Alice", [_write_wav(str(tmp_path / "a.wav"), 1.0)])
    meta = json.load(open(_meta_path(str(tmp_path / "profiles.json"))))
    assert meta["Alice"]["clips"] == 1


def test_profile_without_provenance_reports_none(verifier, tmp_path):
    """A profile enrolled before this change has no count and must not crash."""
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"Legacy": [0.1] * 8}))
    assert verifier.profile_info("Legacy") == {"clips": None, "seconds": None}


def test_unknown_profile_has_no_info(verifier):
    assert verifier.profile_info("Nobody") is None


def test_remove_profile_drops_its_provenance(verifier, tmp_path):
    verifier.enroll("Alice", [_write_wav(str(tmp_path / "a.wav"), 1.0)])
    assert verifier.remove_profile("Alice") is True
    meta = json.load(open(_meta_path(str(tmp_path / "profiles.json"))))
    assert "Alice" not in meta
    assert verifier.profile_info("Alice") is None


def test_list_profiles_detailed(verifier, tmp_path):
    verifier.enroll("Alice", [_write_wav(str(tmp_path / "a.wav"), 2.0)])
    path = tmp_path / "profiles.json"
    raw = _load_profiles(str(path))
    raw["Legacy"] = [0.1] * 8
    path.write_text(json.dumps(raw))
    verifier._invalidate_profile_cache()

    rows = {r["name"]: r for r in verifier.list_profiles_detailed()}
    assert rows["Alice"]["clips"] == 1
    assert rows["Alice"]["seconds"] == pytest.approx(2.0, abs=0.01)
    assert rows["Legacy"]["clips"] is None
    assert verifier.list_profiles() == ["Alice", "Legacy"]


def test_reenroll_replaces_provenance(verifier, tmp_path):
    verifier.enroll("Alice", [_write_wav(str(tmp_path / "a.wav"), 1.0)])
    clips = [_write_wav(str(tmp_path / f"b{i}.wav"), 1.0) for i in range(4)]
    verifier.enroll("Alice", clips)
    assert verifier.profile_info("Alice")["clips"] == 4
