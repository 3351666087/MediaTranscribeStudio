#!/usr/bin/env python3
"""Compose one semantic arbitration and score its final state against truth."""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.semantic_composition import build_semantic_composition
from tools.evaluate_semantic_shadow import (
    _load_case,
    compare_shadow_metrics,
    score_shadow_state,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--lattice", required=True, type=Path)
    parser.add_argument("--arbitration", required=True, type=Path)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def _review_open_count(path: Path | None) -> int | None:
    if path is None:
        return None
    review = read_json_strict(path.resolve())
    count = review.get("openCount") if isinstance(review, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("review queue openCount is invalid")
    return count


def _final_segments(
    document: Mapping[str, Any],
    composition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_segments = {
        str(segment["id"]): dict(segment)
        for segment in document["segments"]
        if isinstance(segment, Mapping)
    }
    output: list[dict[str, Any]] = []
    for final in composition["segments"]:
        source = source_segments[str(final["id"])]
        output.append(
            {
                **source,
                "startMs": final["startMs"],
                "endMs": final["endMs"],
                "speakerId": final["speakerId"],
                "language": final["language"],
                "normalizedText": final["finalText"],
                "displayText": final["finalText"],
                "overlapping": final["overlapping"],
                "humanLocked": final["humanLocked"],
            }
        )
    return output


def main() -> int:
    args = _parser().parse_args()
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    manifest_path = args.manifest.resolve()
    transcript_path = args.transcript.resolve()
    lattice_path = args.lattice.resolve()
    arbitration_path = args.arbitration.resolve()
    manifest = read_json_strict(manifest_path)
    document = read_json_strict(transcript_path)
    lattice = read_json_strict(lattice_path)
    arbitration = read_json_strict(arbitration_path)
    if not all(
        isinstance(value, Mapping)
        for value in (manifest, document, lattice, arbitration)
    ):
        raise ValueError("semantic composition evidence must contain JSON objects")

    case = _load_case(manifest, args.case_id)
    composition = build_semantic_composition(
        document,
        lattice,
        arbitration,
    )
    review_count = _review_open_count(args.review_queue)
    baseline_segments = [
        dict(segment)
        for segment in document["segments"]
        if isinstance(segment, Mapping)
    ]
    composed_segments = _final_segments(document, composition)
    composed_document = copy.deepcopy(dict(document))
    composed_document["speakerPolicy"] = {
        **dict(document["speakerPolicy"]),
        "resolvedCount": composition["speakerPolicy"]["resolvedCount"],
        "speakerIds": list(composition["speakerPolicy"]["speakerIds"]),
    }
    baseline = score_shadow_state(
        case=case,
        document=document,
        segments=baseline_segments,
        review_open_count=review_count,
    )
    final = score_shadow_state(
        case=case,
        document=composed_document,
        segments=composed_segments,
        review_open_count=review_count,
    )
    comparisons, overall = compare_shadow_metrics(baseline, final)
    changed_segments = [
        {
            "segmentId": final_segment["id"],
            "baselineText": source_segment["normalizedText"],
            "finalText": final_segment["normalizedText"],
        }
        for source_segment, final_segment in zip(
            baseline_segments,
            composed_segments,
            strict=True,
        )
        if source_segment["normalizedText"] != final_segment["normalizedText"]
    ]
    blockers = [
        "composition-evaluation-is-not-human-adjudication",
        "open-review-items-not-resolved",
        "frozen-production-thresholds-not-evaluated",
        "single-development-case-cannot-authorize-general-release",
    ]
    if overall == "unchanged":
        blockers.append("no-measured-semantic-gain")
    elif overall == "regressed":
        blockers.append("semantic-hard-domain-regression")
    report = {
        "schemaVersion": "1.0.0",
        "evaluationType": "semantic-composition-final-state",
        "productionAdjudication": False,
        "releaseApproved": False,
        "case": {
            "id": case.get("id"),
            "evaluationSplit": case.get("evaluationSplit"),
            "language": case.get("language"),
            "expectedSpeakerCount": case.get("expectedSpeakerCount"),
        },
        "policy": {
            "truthVisibleToSemanticModel": False,
            "composeOnlyHashBoundSelectedCandidateIds": True,
            "simulateHumanApproval": False,
            "mutateProductionTranscript": False,
            "nonCompensatingDomains": True,
        },
        "semantic": {
            "model": arbitration["model"],
            "promptVersion": arbitration["promptVersion"],
            "status": arbitration["status"],
            "selectedGroupCount": arbitration["metrics"]["selectedGroupCount"],
            "changedSegmentCount": len(changed_segments),
            "changedSegments": changed_segments,
        },
        "baselineMetrics": baseline,
        "finalMetrics": final,
        "metricComparisons": comparisons,
        "overallOutcome": overall,
        "promotionEligible": False,
        "blockingReasons": blockers,
        "evidence": {
            "manifest": {
                "path": str(manifest_path),
                "fileSha256": sha256_file(manifest_path),
                "canonicalSha256": canonical_json_sha256(manifest),
            },
            "transcript": {
                "path": str(transcript_path),
                "fileSha256": sha256_file(transcript_path),
                "canonicalSha256": canonical_json_sha256(document),
            },
            "lattice": {
                "path": str(lattice_path),
                "fileSha256": sha256_file(lattice_path),
                "canonicalSha256": canonical_json_sha256(lattice),
            },
            "arbitration": {
                "path": str(arbitration_path),
                "fileSha256": sha256_file(arbitration_path),
                "canonicalSha256": canonical_json_sha256(arbitration),
            },
            "compositionCanonicalSha256": canonical_json_sha256(composition),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json_no_replace(
        output / "semantic-composition.v1.json",
        composition,
    )
    atomic_write_json_no_replace(
        output / "semantic-composition-quality-report.v1.json",
        report,
    )
    print(
        f"{overall}: {len(changed_segments)} segments changed; "
        f"releaseApproved=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
