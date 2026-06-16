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

Wake-word verifiers are loaded by [`ovos-dinkum-listener`](https://github.com/OpenVoiceOS/ovos-dinkum-listener)
(≥ 0.6.0) from `listener.ww_verifiers`. Each key is a verifier plugin's
entry-point name; its value is that plugin's config. Add to
`~/.config/mycroft/mycroft.conf` (or the OpenVoiceOS equivalent):

```json
{
  "listener": {
    "ww_verifiers": {
      "ovos-ww-verifier-speaker": {
        "model": "wespeaker-resnet34",
        "threshold": 0.45,
        "fail_open": true
      }
    }
  }
}
```

> **Installing the plugin enables it.** The listener runs every installed
> verifier whose config does not set `"enabled": false`. With no entry in
> `ww_verifiers` the plugin still loads with its defaults — and because
> `fail_open` defaults to `true`, it accepts everything until you enroll at
> least one profile. Enroll first, then tune. To install the plugin without
> activating it, set `"enabled": false`:
>
> ```json
> {"listener": {"ww_verifiers": {"ovos-ww-verifier-speaker": {"enabled": false}}}}
> ```

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

## Supported models

The `model` key accepts any alias from
[`speakeronnx`](https://github.com/TigreGotico/speakeronnx)'s registry (models are
downloaded from HuggingFace on first use and cached):

| Alias | Architecture |
|---|---|
| `wespeaker-resnet34` *(default)* | WeSpeaker ResNet34 r-vector |
| `wespeaker-ecapa512` | WeSpeaker ECAPA-TDNN-512 |
| `wespeaker-resnet293` | WeSpeaker ResNet293 (large) |
| `campplus` | WeSpeaker CAM++ |
| `campplus-zh-en` | CAM++ (zh/en) |
| `eres2net` | ERes2Net |
| `titanet-small` | NVIDIA TitaNet-Small |
| `titanet-large` | NVIDIA TitaNet-Large |
| `redimnet-b2` | ReDimNet-B2 |

## Threshold tuning

**The acceptance `threshold` is model-specific — it does not transfer between
models.** Cosine-similarity scales differ enormously across architectures (in our
tests the same enrolled-vs-guest pair scored ~0.95 / 0.89 on `titanet-small` but
~0.17 / 0.14 on `campplus`). The default `0.45` is calibrated for the default
`wespeaker-resnet34`; **if you change `model`, you must re-tune `threshold`.**

To pick a value, enrol a speaker, then compare `verify()` scores for genuine vs.
guest clips and choose a threshold that sits between them
(`tests/test_ovoscope_models_e2e.py` calibrates this per model automatically). For
a given model, lower the threshold for noisier or distant-microphone setups and
raise it for stricter security.

## Python API

```python
from ovos_ww_verifier_plugin_speaker import SpeakerVerifier

v = SpeakerVerifier(config={"threshold": 0.45, "fail_open": False})
v.enroll("Alice", ["alice1.wav", "alice2.wav"])

# In wake word callback:
accepted = v.verify(pcm_bytes)  # True if Alice spoke
```

## Testing

```bash
pip install -e ".[test]"
pytest tests/test_unit.py tests/test_ovoscope_e2e.py   # fast, offline
```

- `test_unit.py` — verifier policy logic (enrolment, thresholds, fail-open).
- `test_ovoscope_e2e.py` — drives the verifier through a real listener
  (`ovoscope.MiniVoiceLoop`) and asserts a rejected speaker suppresses
  `recognizer_loop:record_begin` on the bus. Fast; no model download.
- `test_e2e.py` / `test_ovoscope_models_e2e.py` — real-model tests over **every**
  `speakeronnx` model, using `edge-tts` synthetic voices to confirm only the
  enrolled speaker triggers the wake word. Require `edge-tts` + `ffmpeg` and
  download models; they skip automatically when unavailable.

## Dependencies

- `speakeronnx` (onnxruntime + numpy + huggingface_hub)
- `ovos-plugin-manager`

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
