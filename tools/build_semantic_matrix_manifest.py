#!/usr/bin/env python3
"""Bind completed production outputs into a final-state semantic matrix."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (
    atomic_write_json_no_replace,
    read_json_strict,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--worker-output-root", required=True, type=Path)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{field} is not a file: {path}")
    return path


def _expected_languages(row: Mapping[str, Any]) -> list[str]:
    expected = row.get("expectedLanguages")
    if isinstance(expected, list):
        values = [
            str(value).strip()
            for value in expected
            if isinstance(value, str) and value.strip()
        ]
        if len(values) != len(expected) or len(set(values)) != len(values):
            raise ValueError(f"{row.get('id')}.expectedLanguages is invalid")
        return values
    language = row.get("language")
    if (
        isinstance(language, str)
        and language.strip()
        and language.strip().casefold() not in {"auto", "mul", "und"}
    ):
        return [language.strip()]
    return []


def _truth_eligible(
    row: Mapping[str, Any],
    field: str,
    *,
    fallback: bool,
) -> bool:
    truth = row.get("truthEligibility")
    if not isinstance(truth, Mapping) or field not in truth:
        return fallback
    value = truth.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{row.get('id')}.truthEligibility.{field} is invalid")
    return value


def _completed_case(
    row: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    case_id = str(row.get("id") or "").strip()
    if not case_id:
        raise ValueError("sample case id is missing")
    output = (output_root / case_id).resolve()
    checkpoint_path = output / "checkpoint.v2.json"
    checkpoint = read_json_strict(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{case_id} checkpoint is invalid")
    semantic = checkpoint.get("semantic")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("mode") != "candidate-composition"
        or semantic.get("status") != "completed"
        or semantic.get("autoApply") is not True
    ):
        raise ValueError(
            f"{case_id} did not complete mandatory candidate composition"
        )
    transcript = _path(
        str(output / "semantic" / "input-transcript.v2.json"),
        f"{case_id}.transcript",
    )
    lattice = _path(
        semantic.get("inputLatticePath"),
        f"{case_id}.inputLatticePath",
    )
    arbitration = _path(
        semantic.get("arbitrationPath"),
        f"{case_id}.arbitrationPath",
    )
    composition = _path(
        semantic.get("artifactPath"),
        f"{case_id}.artifactPath",
    )
    pipeline_metrics = _path(
        str(output / "pipeline-metrics.v1.json"),
        f"{case_id}.pipelineMetrics",
    )
    expected_count = row.get("expectedSpeakerCount")
    count_eligible = _truth_eligible(
        row,
        "speakerCount",
        fallback=(
            isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and expected_count >= 0
        ),
    )
    if not count_eligible:
        expected_count = None
    transcript_eligible = _truth_eligible(
        row,
        "asr",
        fallback=isinstance(
            row.get("scoringTranscript") or row.get("expectedTranscript"),
            str,
        ),
    )
    expected_transcript = (
        str(
            row.get("scoringTranscript")
            or row.get("expectedTranscript")
            or ""
        )
        if transcript_eligible
        else ""
    )
    scenarios = row.get("scenario")
    if not isinstance(scenarios, list) or not scenarios:
        scenarios = ["production-final-state"]
    return {
        "id": case_id,
        "scenarios": [str(value) for value in scenarios],
        "expected": {
            "speakerCount": expected_count,
            "languages": _expected_languages(row),
            "transcript": expected_transcript,
        },
        "artifacts": {
            "transcript": str(transcript),
            "lattice": str(lattice),
            "arbitration": str(arbitration),
            "composition": str(composition),
            "pipelineMetrics": str(pipeline_metrics),
            "checkpoint": str(checkpoint_path.resolve()),
        },
        "agentSemanticReview": {
            "status": "pending",
            "meaningPreservation": "not-reviewed",
            "speakerCoherence": "not-reviewed",
            "languageCoherence": "not-reviewed",
            "translationCompleteness": "not-reviewed",
            "notes": [],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = read_json_strict(args.sample_manifest.resolve())
    if not isinstance(manifest, Mapping):
        raise ValueError("sample manifest must contain an object")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sample manifest cases must be non-empty")
    by_id = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("id")
    }
    selected = args.case or list(by_id)
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ValueError(f"unknown sample cases: {', '.join(unknown)}")
    output_root = args.worker_output_root.resolve()
    value = {
        "schemaVersion": "1.0.0",
        "matrixId": args.matrix_id.strip(),
        "cases": [
            _completed_case(by_id[case_id], output_root=output_root)
            for case_id in selected
        ],
    }
    if not value["matrixId"]:
        raise ValueError("matrix id must not be blank")
    atomic_write_json_no_replace(args.output.resolve(), value)
    print(f"{len(value['cases'])} completed cases bound to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
