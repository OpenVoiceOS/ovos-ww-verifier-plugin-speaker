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

    # The console script enrolls and exits. It must not open a bus client:
    # its boundary is the shell account that ran it, which is the boundary the
    # OVOS-SPEAKER-1 gate exists to replace for a client that has no shell.
    cfg: dict = {"model": args.model, "bus_surface": False}
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
