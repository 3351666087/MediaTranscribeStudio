"""Strict-JSON command line interface for the speaker-scaling benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .runner import (
    DEFAULT_MODES,
    DEFAULT_SAMPLES_PER_SPEAKER,
    DEFAULT_SPEAKER_COUNTS,
    run_benchmark,
)


class CliUsageError(ValueError):
    """Invalid CLI input that should be returned as JSON."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def _csv_tokens(value: str, field: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in value.split(","))
    if not tokens or any(not token for token in tokens):
        raise CliUsageError(f"{field} must be a non-empty comma-separated list")
    if len(set(tokens)) != len(tokens):
        raise CliUsageError(f"{field} must not contain duplicates")
    return tokens


def parse_speaker_counts(value: str) -> tuple[int, ...]:
    tokens = _csv_tokens(value, "--speaker-counts")
    counts: list[int] = []
    for token in tokens:
        try:
            count = int(token)
        except ValueError as exc:
            raise CliUsageError(
                "--speaker-counts values must be positive integers"
            ) from exc
        if count < 1:
            raise CliUsageError(
                "--speaker-counts values must be positive integers"
            )
        counts.append(count)
    return tuple(counts)


def parse_modes(value: str) -> tuple[str, ...]:
    modes = _csv_tokens(value, "--modes")
    invalid = tuple(mode for mode in modes if mode not in DEFAULT_MODES)
    if invalid:
        raise CliUsageError(
            "--modes values must be manual, auto, or hybrid"
        )
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        prog="python -m benchmarks.speaker_scaling",
        description=(
            "Benchmark backend.speaker_pipeline._cluster using synthetic "
            "embeddings only; this is not a DER benchmark."
        ),
    )
    parser.add_argument(
        "--speaker-counts",
        default=",".join(str(value) for value in DEFAULT_SPEAKER_COUNTS),
    )
    parser.add_argument(
        "--samples-per-speaker",
        type=int,
        default=DEFAULT_SAMPLES_PER_SPEAKER,
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser


def strict_json_dumps(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )


def _write_output(path: Path, text: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(destination)


def _error_payload(exc: BaseException, *, kind: str) -> dict[str, object]:
    return {
        "schemaVersion": "speaker-scaling-benchmark/error-v1",
        "ok": False,
        "error": {
            "kind": kind,
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        speaker_counts = parse_speaker_counts(args.speaker_counts)
        modes = parse_modes(args.modes)
        if args.samples_per_speaker < 1:
            raise CliUsageError(
                "--samples-per-speaker must be a positive integer"
            )
        if args.repeat < 1:
            raise CliUsageError("--repeat must be a positive integer")
        payload = run_benchmark(
            speaker_counts=speaker_counts,
            samples_per_speaker=args.samples_per_speaker,
            modes=modes,
            repeat=args.repeat,
        )
        text = strict_json_dumps(payload)
        if args.output_json is not None:
            _write_output(args.output_json, text)
    except CliUsageError as exc:
        text = strict_json_dumps(_error_payload(exc, kind="usage"))
        sys.stdout.write(text + "\n")
        return 2
    except Exception as exc:  # Strict JSON is more useful than a CLI traceback.
        text = strict_json_dumps(_error_payload(exc, kind="benchmark"))
        sys.stdout.write(text + "\n")
        return 1

    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
