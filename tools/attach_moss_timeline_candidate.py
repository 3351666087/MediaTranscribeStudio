#!/usr/bin/env python3
"""Attach one pinned MOSS result as a real semantic timeline challenger."""

from __future__ import annotations

import argparse
import json
import sys
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
    validate_strict_json,
)
from backend.semantic_candidate_generation import (
    SemanticCandidateGenerationRegistry,
    build_timeline_challenger_result,
)
from backend.semantic_candidate_lattice import (
    validate_semantic_candidate_lattice,
)
from backend.semantic_composition import validate_semantic_job_arbitration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--lattice", required=True, type=Path)
    parser.add_argument("--arbitration", required=True, type=Path)
    parser.add_argument("--moss-segments", required=True, type=Path)
    parser.add_argument("--moss-challenge-report", required=True, type=Path)
    parser.add_argument("--pyannote-result", type=Path)
    parser.add_argument("--pyannote-audit", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def _segments(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_strict_json(value)
    if not isinstance(value, list) or not value:
        raise ValueError("MOSS segments must be a non-empty JSON array")
    output: list[dict[str, Any]] = []
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            raise ValueError(f"MOSS segment {index} must be an object")
        output.append(
            {
                "startMs": segment["startMs"],
                "endMs": segment["endMs"],
                "speaker": segment["speakerId"],
                "text": segment["rawText"],
            }
        )
    return output


def main() -> int:
    args = _parser().parse_args()
    transcript = read_json_strict(args.transcript.resolve())
    lattice = validate_semantic_candidate_lattice(
        read_json_strict(args.lattice.resolve()),
        expected_source_media_sha256=transcript["source"]["sha256"],
        expected_transcript_sha256=canonical_json_sha256(transcript),
    )
    arbitration = validate_semantic_job_arbitration(
        read_json_strict(args.arbitration.resolve()),
        expected_job_id=transcript["jobId"],
        expected_lattice=lattice,
    )
    challenge = read_json_strict(args.moss_challenge_report.resolve())
    if (
        challenge.get("status") != "completed"
        or challenge.get("failureCode") is not None
        or challenge.get("input", {}).get("sha256")
        != transcript["source"]["sha256"]
    ):
        raise ValueError(
            "MOSS challenge is incomplete or bound to different source media"
        )
    model = challenge.get("model")
    if not isinstance(model, dict):
        raise ValueError("MOSS challenge model identity is missing")
    moss_segments_path = args.moss_segments.resolve()
    result = build_timeline_challenger_result(
        turns=_segments(moss_segments_path),
        source_duration_ms=transcript["source"]["durationMs"],
        system_id=model["repoId"],
        revision=model["revision"],
        artifact_sha256=sha256_file(moss_segments_path),
        model_manifest_sha256=model["manifestSha256"],
        local_speaker_field="speaker",
    )
    candidate_results = [
        {
            "producer": result["producer"],
            "payload": result["candidates"][0]["payload"],
        }
    ]
    pyannote_summary = None
    if (args.pyannote_result is None) != (args.pyannote_audit is None):
        raise ValueError(
            "Pyannote result and audit must be supplied together"
        )
    if args.pyannote_result is not None and args.pyannote_audit is not None:
        pyannote_result_path = args.pyannote_result.resolve()
        pyannote_result = read_json_strict(pyannote_result_path)
        pyannote_audit = read_json_strict(args.pyannote_audit.resolve())
        if (
            pyannote_audit.get("status") != "completed"
            or pyannote_audit.get("source", {}).get("sha256")
            != transcript["source"]["sha256"]
            or pyannote_audit.get("source", {}).get("durationMs")
            != transcript["source"]["durationMs"]
        ):
            raise ValueError(
                "Pyannote challenger is incomplete or rebound to other media"
            )
        pyannote_model = pyannote_audit.get("model")
        if not isinstance(pyannote_model, dict):
            raise ValueError("Pyannote challenger model identity is missing")
        pyannote_candidate = build_timeline_challenger_result(
            turns=pyannote_result["speakerTurns"],
            source_duration_ms=transcript["source"]["durationMs"],
            system_id=pyannote_model["repoId"],
            revision=pyannote_model["revision"],
            artifact_sha256=sha256_file(pyannote_result_path),
            model_manifest_sha256=pyannote_model["manifestSha256"],
            local_speaker_field="localSpeaker",
        )
        candidate_results.append(
            {
                "producer": pyannote_candidate["producer"],
                "payload": pyannote_candidate["candidates"][0]["payload"],
            }
        )
        pyannote_summary = {
            "revision": pyannote_model["revision"],
            "manifestSha256": pyannote_model["manifestSha256"],
            "resultFileSha256": sha256_file(pyannote_result_path),
            "speakerCount": pyannote_audit["output"]["speakerCount"],
            "regularTurnCount": pyannote_audit["output"][
                "regularTurnCount"
            ],
            "elapsedSeconds": pyannote_audit["runtime"]["elapsedSeconds"],
        }
    combined_result = {"candidateResults": candidate_results}
    registry = SemanticCandidateGenerationRegistry(
        {
            "timeline-challenger": (
                lambda request, document, current_lattice: combined_result
            )
        }
    )
    generation = registry.fulfill(
        transcript,
        lattice,
        arbitration,
    )
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json_no_replace(
        output / "semantic-candidate-generation.v1.json",
        generation,
    )
    atomic_write_json_no_replace(
        output / "semantic-candidate-lattice.extended.v1.json",
        generation["outputLattice"],
    )
    report = {
        "schemaVersion": "1.0.0",
        "artifactType": "moss-semantic-timeline-candidate-audit",
        "jobId": transcript["jobId"],
        "sourceMediaSha256": transcript["source"]["sha256"],
        "inputLatticeSha256": lattice["latticeSha256"],
        "outputLatticeSha256": generation["outputLattice"]["latticeSha256"],
        "generationArtifactSha256": canonical_json_sha256(generation),
        "status": generation["status"],
        "fulfilledRequestCount": generation["metrics"][
            "fulfilledRequestCount"
        ],
        "unfulfilledRequestCount": generation["metrics"][
            "unfulfilledRequestCount"
        ],
        "moss": {
            "revision": model["revision"],
            "manifestSha256": model["manifestSha256"],
            "segmentsFileSha256": sha256_file(moss_segments_path),
            "speakerCount": challenge["outputValidation"]["speakerCount"],
            "segmentCount": challenge["outputValidation"]["segmentCount"],
            "elapsedSeconds": challenge["supervision"]["elapsedSeconds"],
        },
        "pyannote": pyannote_summary,
        "claimsFinalQualityImprovement": False,
        "conclusion": "timeline-candidate-added-requires-rearbitration",
    }
    atomic_write_json_no_replace(
        output / "audit-report.v1.json",
        report,
    )
    print(
        f"{generation['status']}: MOSS timeline added; "
        f"{generation['metrics']['unfulfilledRequestCount']} requests remain"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
