# ovos-ww-verifier-plugin-speaker

OVOS wake word verifier plugin that accepts voice commands only from enrolled household members.

After a wake word engine detects an activation, this verifier extracts a speaker embedding from
the captured audio and compares it against enrolled profiles. Activations from unrecognised
speakers are silently dropped.

## Use case

Alice and Bob live together and use OVOS at home. They enroll their voices once. When a guest
visits, their "Hey Mycroft" triggers the wake word detector — but the speaker verifier rejects
it before any intent is processed. Alice and Bob's commands go through normally.

## Privacy note

Speaker profiles are stored as fixed-length numeric vectors (embeddings) in a local JSON file
under `~/.local/share/ovos_speaker_verifier/profiles.json`. No audio is retained after
embedding extraction. Embeddings cannot be reversed into audio.

## Install

```bash
pip install ovos-ww-verifier-plugin-speaker
```

## Enroll household members

```bash
ovos-speaker-enroll Alice clip1.wav clip2.wav clip3.wav
ovos-speaker-enroll Bob morning_command.wav evening_command.wav
```

More clips (5–30 s total per person) → more robust profile.

## OVOS configuration

Add to `~/.config/mycroft/mycroft.conf` (or OpenVoiceOS equivalent):

```json
{
  "hotwords": {
    "hey mycroft": {
      "module": "...",
      "verifier": "ovos-ww-verifier-speaker",
      "verifier_config": {
        "model": "wespeaker-resnet34",
        "threshold": 0.45,
        "fail_open": true
      }
    }
  }
}
```

## Configuration keys

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | str | `"wespeaker-resnet34"` | speakeronnx model alias or `.onnx` path |
| `threshold` | float | `0.45` | Cosine similarity acceptance threshold |
| `fail_open` | bool | `true` | Accept all activations when no profiles enrolled |
| `profiles_path` | str | `~/.local/share/ovos_speaker_verifier/profiles.json` | Override profile storage path |
| `per_profile_thresholds` | dict | `{}` | Per-name threshold overrides, e.g. `{"Alice": 0.5}` |
| `sample_rate` | int | `16000` | PCM sample rate of audio chunks passed to `verify()` |
| `sample_width` | int | `2` | PCM sample width in bytes (2 = 16-bit) |
| `channels` | int | `1` | PCM channel count |

## Threshold tuning

The default threshold of `0.45` works well for the WeSpeaker ResNet34 model with clean
microphone audio. Lower it (e.g. `0.35`) for noisier environments or distant microphones.
Raise it (e.g. `0.55`) for stricter security.

## Python API

```python
from ovos_ww_verifier_plugin_speaker import SpeakerVerifier

v = SpeakerVerifier(config={"threshold": 0.45, "fail_open": False})
v.enroll("Alice", ["alice1.wav", "alice2.wav"])

# In wake word callback:
accepted = v.verify(pcm_bytes)  # True if Alice spoke
```

## Dependencies

- `speakeronnx` (onnxruntime + numpy + huggingface_hub)
- `ovos-plugin-manager`
