"""Run the real Python assembler -> Java PDF renderer integration matrix.

The fixture is synthetic and contains no meeting transcript.  The command
prints structural metrics only, so it is safe to attach to CI logs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from reporting.java_pdf_client import JavaPdfClient  # noqa: E402
from reporting.report_document_assembler import (  # noqa: E402
    ReportDocumentAssembler,
    canonical_speaker_ids,
)


DEFAULT_COUNTS = (1, 2, 5, 8, 13)
FIXTURE_TIMESTAMP = "2026-07-21T00:00:00Z"


def _parse_counts(value: str) -> tuple[int, ...]:
    counts: list[int] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            count = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"speaker count must be an integer: {text!r}"
            ) from exc
        if count < 1:
            raise argparse.ArgumentTypeError(
                "speaker counts must be positive integers"
            )
        if count not in counts:
            counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError(
            "at least one positive speaker count is required"
        )
    return tuple(counts)


def _speaker_scores(
    winner: str,
    speaker_ids: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "speaker": f"legacy-{index}",
            "score": 0.94 if speaker_id == winner else 0.10 + index * 0.001,
        }
        for index, speaker_id in enumerate(speaker_ids, start=1)
    ]


def _segments(speaker_count: int) -> list[dict[str, Any]]:
    speaker_ids = canonical_speaker_ids(speaker_count)
    return [
        {
            "id": f"segment-{index:03d}",
            "start": (index - 1) * 1.5,
            "end": (index - 1) * 1.5 + 1.2,
            "speaker": f"legacy-{index}",
            "raw_text": f"动态人数离线回归片段{index}。",
            "normalized_text": f"动态人数离线回归片段{index}。",
            "display_text": f"动态人数离线回归片段{index}。",
            "confidence": 0.95,
            "speaker_scores": _speaker_scores(speaker_id, speaker_ids),
            "boundary_confidence": 0.94,
        }
        for index, speaker_id in enumerate(speaker_ids, start=1)
    ]


def _run_case(
    *,
    assembler: ReportDocumentAssembler,
    client: JavaPdfClient,
    source_path: Path,
    output_root: Path,
    speaker_count: int,
) -> dict[str, Any]:
    speaker_ids = canonical_speaker_ids(speaker_count)
    document = assembler.assemble(
        _segments(speaker_count),
        source_path=source_path,
        duration_ms=speaker_count * 1_500 + 1_000,
        document_id=f"real-java-e2e-{speaker_count}-speakers",
        generated_at=FIXTURE_TIMESTAMP,
        speaker_count_mode="auto",
        speaker_count_confidence=0.97,
        privacy={
            "containsRealMeetingText": False,
            "exportApproved": True,
        },
    )

    started = time.perf_counter()
    outcome = client.render(
        document,
        job_id=f"real-java-e2e-n{speaker_count}",
        output_directory=output_root / f"n{speaker_count}",
    )
    elapsed_seconds = time.perf_counter() - started

    quality = json.loads(
        outcome.artifact_paths["qualityReportPath"].read_text(encoding="utf-8")
    )
    persisted = json.loads(
        outcome.report_document_path.read_text(encoding="utf-8")
    )
    persisted_speaker_ids = tuple(
        speaker["id"] for speaker in persisted["speakers"]
    )
    if persisted_speaker_ids != speaker_ids:
        raise RuntimeError(
            f"N={speaker_count}: persisted speaker set changed"
        )
    if persisted["speakerPolicy"]["resolvedCount"] != speaker_count:
        raise RuntimeError(
            f"N={speaker_count}: resolved speaker count changed"
        )
    for segment in persisted["segments"]:
        score_ids = tuple(
            score["speakerId"]
            for score in segment["evidence"]["speaker"]["scores"]
        )
        if score_ids != speaker_ids:
            raise RuntimeError(
                f"N={speaker_count}: speaker score vector cardinality changed"
            )
    if quality.get("hardGatesPassed") is not True:
        raise RuntimeError(f"N={speaker_count}: hard gates did not pass")
    if len(quality.get("hardGates", [])) != 13:
        raise RuntimeError(f"N={speaker_count}: expected 13 hard gates")
    if len(quality.get("facets", [])) != 14:
        raise RuntimeError(f"N={speaker_count}: expected 14 aesthetic facets")

    screenshot_directory = outcome.artifact_paths["screenshotsDirectory"]
    return {
        "speakerCount": speaker_count,
        "status": outcome.result["status"],
        "elapsedSeconds": round(elapsed_seconds, 3),
        "pages": len(tuple(screenshot_directory.glob("*.png"))),
        "qualityScore": quality["score"],
        "hardGateCount": len(quality["hardGates"]),
        "facetCount": len(quality["facets"]),
        "pdfBytes": outcome.artifact_paths["pdfPath"].stat().st_size,
        "reportDocumentSha256": outcome.report_document_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the real ReportDocumentAssembler -> Java OpenHTMLtoPDF/"
            "PDFBox chain using synthetic dynamic-speaker fixtures."
        )
    )
    parser.add_argument(
        "--jar",
        type=Path,
        default=APP_ROOT / "pdf-renderer" / "target" / "pdf-renderer.jar",
        help="Path to the shaded Java renderer JAR.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dedicated directory for generated integration evidence.",
    )
    parser.add_argument(
        "--counts",
        type=_parse_counts,
        default=DEFAULT_COUNTS,
        help="Comma-separated positive speaker counts (default: 1,2,5,8,13).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete an existing output root before running.",
    )
    args = parser.parse_args()

    jar_path = args.jar.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root == APP_ROOT or APP_ROOT in output_root.parents:
        raise RuntimeError(
            "integration evidence must be written outside the source repository"
        )
    if output_root.exists():
        if not args.replace:
            raise RuntimeError(
                f"output root already exists; pass --replace: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    source_path = output_root / "synthetic-source.wav"
    source_path.write_bytes(b"offline-synthetic-media")
    assembler = ReportDocumentAssembler(
        pipeline_version="e2e-2026.07.21",
    )
    client = JavaPdfClient.from_jar(
        jar_path,
        allowed_output_root=output_root,
        timeout_seconds=180.0,
    )

    cases = [
        _run_case(
            assembler=assembler,
            client=client,
            source_path=source_path,
            output_root=output_root,
            speaker_count=count,
        )
        for count in args.counts
    ]
    result = {
        "status": "passed",
        "fixtureDate": "2026-07-21",
        "rendererJar": str(jar_path),
        "outputRoot": str(output_root),
        "counts": list(args.counts),
        "cases": cases,
    }
    summary_path = output_root / "e2e-summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
