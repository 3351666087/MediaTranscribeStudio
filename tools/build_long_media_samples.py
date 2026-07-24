"""Build deterministic short-window evidence from complete long media."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.long_media_samples import build_long_media_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--random-window-count", type=int, default=2)
    parser.add_argument("--change-window-count", type=int, default=3)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--reference-rttm", type=Path)
    parser.add_argument("--reference-recording-id")
    parser.add_argument("--target-speaker-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_long_media_matrix(
        args.source,
        output_root=args.output_root,
        window_seconds=args.window_seconds,
        random_seed=args.random_seed,
        random_window_count=args.random_window_count,
        change_window_count=args.change_window_count,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        reference_rttm=args.reference_rttm,
        reference_recording_id=args.reference_recording_id,
        target_speaker_count=args.target_speaker_count,
    )
    print(
        json.dumps(
            {
                "libraryId": manifest["libraryId"],
                "sourceCount": len(manifest["sources"]),
                "caseCount": len(manifest["cases"]),
                "outputRoot": str(args.output_root.expanduser().resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
