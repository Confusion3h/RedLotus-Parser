#!/usr/bin/env python3
"""
Confu V1.0 — CLI entry point.

Examples
--------
  # Batch file transcription + PII anonymization
  python main.py file path/to/meeting.wav

  # Live microphone (default 10 seconds)
  python main.py live --duration 10

  # Rehydrate anonymized text from mapping_dict.json
  python main.py rehydrate "Ciao [FULLNAME_0], il tuo CF è [CF_0]"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from confu_core import ConfuV1Pipeline


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_result(result, mapping_path: str, show_raw: bool = False) -> None:
    print("\n" + "=" * 60)
    print(f"SOURCE : {result.source}")
    if show_raw:
        print("-" * 60)
        print("RAW TRANSCRIPT")
        print(result.raw_transcript or "(vuoto)")
    print("-" * 60)
    print("ANONYMIZED TEXT (safe to paste into cloud AI)")
    print(result.anonymized_text or "(vuoto)")
    print("-" * 60)
    print(f"MAPPING ({len(result.mapping)} entries) → {mapping_path}")
    if result.mapping:
        print(json.dumps(result.mapping, ensure_ascii=False, indent=2))
    print("=" * 60 + "\n")


def cmd_file(args: argparse.Namespace) -> int:
    pipeline = ConfuV1Pipeline(
        stt_model_size=args.model,
        mapping_path=args.mapping,
        riva_uri=args.riva_uri,
    )
    try:
        result = pipeline.process_audio_file(args.audio_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_result(result, mapping_path=args.mapping, show_raw=args.show_raw)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    if args.duration is not None and args.duration < 0:
        print("ERROR: --duration must be >= 0", file=sys.stderr)
        return 2

    pipeline = ConfuV1Pipeline(
        stt_model_size=args.model,
        mapping_path=args.mapping,
        riva_uri=args.riva_uri,
    )

    def on_partial(raw: str, anon: str) -> None:
        print(f"  [chunk] raw={raw!r}")
        print(f"          anon={anon!r}")

    try:
        result = pipeline.start_live_transcription(
            duration_seconds=args.duration,
            on_partial=on_partial if args.verbose else None,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_result(result, mapping_path=args.mapping, show_raw=args.show_raw)
    return 0


def cmd_rehydrate(args: argparse.Namespace) -> int:
    mapping_path = Path(args.mapping)
    mapping: dict[str, str] = {}
    if mapping_path.exists():
        try:
            data = json.loads(mapping_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid mapping JSON ({mapping_path}): {exc}", file=sys.stderr)
            return 2
        if not isinstance(data, dict):
            print(
                f"ERROR: mapping file must be a JSON object: {mapping_path}",
                file=sys.stderr,
            )
            return 2
        mapping = {str(k): str(v) for k, v in data.items()}
    else:
        print(f"WARN: mapping file not found: {mapping_path}", file=sys.stderr)

    pipeline = ConfuV1Pipeline(mapping_path=args.mapping, load_models=False)
    restored = pipeline.rehydrate_text(
        args.text, mapping_dict=mapping if mapping else None
    )

    print("\nREHYDRATED TEXT")
    print(restored)
    print()
    return 0


def _add_shared_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """
    Shared CLI flags.

    When ``suppress`` is True (subparsers), omit defaults so values set on the
    top-level parser are not overwritten by argparse subparser defaults.
    """
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=default if suppress else False,
        help="Debug logging / live chunk prints",
    )
    parser.add_argument(
        "--model",
        default=default if suppress else "medium",
        help="faster-whisper model size (tiny|base|small|medium|large-v3|large-v3-turbo)",
    )
    parser.add_argument(
        "--mapping",
        default=default if suppress else "mapping_dict.json",
        help="Path to volatile local mapping JSON",
    )
    parser.add_argument(
        "--riva-uri",
        default=default if suppress else None,
        help="Optional NVIDIA Riva gRPC endpoint (e.g. localhost:50051)",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        default=default if suppress else False,
        help="Also print the uncensored raw transcript (local only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="confu",
        description="Confu V1.0 — Local STT + Italian PII Anonymizer",
    )
    _add_shared_flags(parser, suppress=False)

    sub = parser.add_subparsers(dest="command", required=True)

    p_file = sub.add_parser("file", help="Batch: transcribe + anonymize an audio file")
    _add_shared_flags(p_file, suppress=True)
    p_file.add_argument("audio_path", help="Path to .mp3 / .wav / .m4a")
    p_file.set_defaults(func=cmd_file)

    p_live = sub.add_parser("live", help="Live: microphone STT + anonymize")
    _add_shared_flags(p_live, suppress=True)
    p_live.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Capture duration in seconds (default: 10)",
    )
    p_live.set_defaults(func=cmd_live)

    p_reh = sub.add_parser(
        "rehydrate", help="Restore PII from placeholders via mapping"
    )
    _add_shared_flags(p_reh, suppress=True)
    p_reh.add_argument("text", help="Anonymized text containing [PLACEHOLDER_N] tokens")
    p_reh.set_defaults(func=cmd_rehydrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Ensure shared flags exist even if only provided on one side
    for name, value in {
        "verbose": False,
        "model": "medium",
        "mapping": "mapping_dict.json",
        "riva_uri": None,
        "show_raw": False,
    }.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
