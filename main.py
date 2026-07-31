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


def _print_result(result, show_raw: bool = False) -> None:
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
    print(f"MAPPING ({len(result.mapping)} entries) → mapping_dict.json")
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
    _print_result(result, show_raw=args.show_raw)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
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
    _print_result(result, show_raw=args.show_raw)
    return 0


def cmd_rehydrate(args: argparse.Namespace) -> int:
    mapping_path = Path(args.mapping)
    mapping: dict[str, str] = {}
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        print(f"WARN: mapping file not found: {mapping_path}", file=sys.stderr)

    pipeline = ConfuV1Pipeline(mapping_path=args.mapping, load_models=False)
    # Prefer CLI mapping file contents if loaded
    if mapping:
        restored = pipeline.rehydrate_text(args.text, mapping_dict=mapping)
    else:
        restored = pipeline.rehydrate_text(args.text)

    print("\nREHYDRATED TEXT")
    print(restored)
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging / live chunk prints"
    )
    shared.add_argument(
        "--model",
        default="medium",
        help="faster-whisper model size (tiny|base|small|medium|large-v3|large-v3-turbo)",
    )
    shared.add_argument(
        "--mapping",
        default="mapping_dict.json",
        help="Path to volatile local mapping JSON",
    )
    shared.add_argument(
        "--riva-uri",
        default=None,
        help="Optional NVIDIA Riva gRPC endpoint (e.g. localhost:50051)",
    )
    shared.add_argument(
        "--show-raw",
        action="store_true",
        help="Also print the uncensored raw transcript (local only)",
    )

    parser = argparse.ArgumentParser(
        prog="confu",
        description="Confu V1.0 — Local STT + Italian PII Anonymizer",
        parents=[shared],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_file = sub.add_parser(
        "file",
        parents=[shared],
        help="Batch: transcribe + anonymize an audio file",
    )
    p_file.add_argument("audio_path", help="Path to .mp3 / .wav / .m4a")
    p_file.set_defaults(func=cmd_file)

    p_live = sub.add_parser(
        "live",
        parents=[shared],
        help="Live: microphone STT + anonymize",
    )
    p_live.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Capture duration in seconds (default: 10)",
    )
    p_live.set_defaults(func=cmd_live)

    p_reh = sub.add_parser(
        "rehydrate",
        parents=[shared],
        help="Restore PII from placeholders via mapping",
    )
    p_reh.add_argument("text", help="Anonymized text containing [PLACEHOLDER_N] tokens")
    p_reh.set_defaults(func=cmd_rehydrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
