"""CLI enrollment helper: ``ovos-speaker-enroll``."""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ovos-speaker-enroll",
        description=(
            "Enroll a household member so the OVOS speaker verifier "
            "will accept their voice commands."
        ),
    )
    parser.add_argument("name", help="Speaker name (e.g. Alice)")
    parser.add_argument("wavs", nargs="+", help="One or more WAV files")
    parser.add_argument(
        "--model", default="wespeaker-resnet34",
        help="speakeronnx model alias (default: wespeaker-resnet34)"
    )
    parser.add_argument(
        "--profiles-path", default=None,
        help="Override default profiles JSON path"
    )
    args = parser.parse_args()

    cfg: dict = {"model": args.model}
    if args.profiles_path:
        cfg["profiles_path"] = args.profiles_path

    from ovos_ww_verifier_plugin_speaker import SpeakerVerifier
    verifier = SpeakerVerifier(config=cfg)
    emb = verifier.enroll(args.name, args.wavs)

    print(f"Enrolled '{args.name}' from {len(args.wavs)} clip(s).")
    print(f"Profile dim={emb.shape[0]}  norm={float(__import__('numpy').linalg.norm(emb)):.6f}")
    print(f"Profiles saved to: {verifier._profiles_path}")
    print(f"Enrolled speakers: {verifier.list_profiles()}")


if __name__ == "__main__":
    main()
