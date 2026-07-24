"""Export SRT, WebVTT, and ASS sidecars from a persisted sample transcript."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.output_orchestration import transcript_subtitle_segments
from backend.subtitles import CuePolicy, SubtitleFormat, arrange_cues, export_subtitles


def export_sample_subtitles(
    transcript_path: Path,
    output_root: Path,
    *,
    include_speaker_labels: bool = False,
) -> list[Path]:
    document = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = transcript_subtitle_segments(document)
    arrangement = arrange_cues(
        segments,
        policy=CuePolicy(include_speaker_labels=include_speaker_labels),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    stem = str(document.get("jobId") or transcript_path.stem)
    paths: list[Path] = []
    for subtitle_format in (
        SubtitleFormat.SRT,
        SubtitleFormat.WEBVTT,
        SubtitleFormat.ASS,
    ):
        suffix = ".vtt" if subtitle_format is SubtitleFormat.WEBVTT else (
            f".{subtitle_format.value}"
        )
        path = output_root / f"{stem}{suffix}"
        path.write_text(
            export_subtitles(arrangement, subtitle_format),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-speaker-labels", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = export_sample_subtitles(
        args.transcript,
        args.output_root,
        include_speaker_labels=args.include_speaker_labels,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
