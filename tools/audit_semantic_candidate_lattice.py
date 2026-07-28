#!/usr/bin/env python3
"""Audit real transcript artifacts without persisting transcript text in reports."""

from __future__ import annotations

import argparse
import copy
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (
    atomic_write_json,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.semantic_candidate_lattice import (
    SemanticCandidateLatticeError,
    build_semantic_candidate_lattice_from_document,
    validate_semantic_candidate_lattice,
)


REPORT_SCHEMA_VERSION = "1.0.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return result[:120] or "case"


def _domain_summary(domain: dict[str, Any]) -> dict[str, Any]:
    groups = domain["groups"]
    return {
        "domain": domain["domain"],
        "status": domain["status"],
        "unavailableReason": domain["unavailableReason"],
        "groupCount": domain["groupCount"],
        "availableGroupCount": domain["availableGroupCount"],
        "unavailableGroupCount": domain["unavailableGroupCount"],
        "candidateCount": sum(group["candidateCount"] for group in groups),
        "eligibleCandidateCount": sum(
            group["eligibleCandidateCount"] for group in groups
        ),
        "unavailableReasons": sorted(
            {
                group["unavailableReason"]
                for group in groups
                if group["unavailableReason"] is not None
            }
        ),
    }


def _tamper_checks(lattice: dict[str, Any]) -> dict[str, bool]:
    payload_tampered = copy.deepcopy(lattice)
    first_candidate = payload_tampered["domains"][0]["groups"][0][
        "candidates"
    ][0]
    first_candidate["payload"]["classification"] = (
        "no-transcribable-speech"
        if first_candidate["payload"]["classification"]
        == "transcribable-speech"
        else "transcribable-speech"
    )
    identity_tampered = copy.deepcopy(lattice)
    identity_tampered["domains"][0]["groups"][0]["candidates"][0][
        "producers"
    ][0]["revision"] = "tampered"
    binding_tampered = copy.deepcopy(lattice)
    binding_tampered["binding"]["transcriptSha256"] = "f" * 64
    derived_tampered = copy.deepcopy(lattice)
    derived_tampered["domains"][0]["status"] = "available"

    results: dict[str, bool] = {}
    for name, value in (
        ("payloadTamperRejected", payload_tampered),
        ("producerIdentityTamperRejected", identity_tampered),
        ("transcriptBindingTamperRejected", binding_tampered),
        ("derivedAvailabilityTamperRejected", derived_tampered),
    ):
        try:
            validate_semantic_candidate_lattice(value)
        except SemanticCandidateLatticeError:
            results[name] = True
        else:
            results[name] = False
    return results


def audit_transcripts(
    transcript_paths: list[Path],
    *,
    output_directory: Path,
    persist_lattices: bool,
) -> dict[str, Any]:
    if not transcript_paths:
        raise ValueError("at least one transcript path is required")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for transcript_path in transcript_paths:
        canonical_path = transcript_path.expanduser().resolve(strict=True)
        document = read_json_strict(canonical_path)
        lattice = build_semantic_candidate_lattice_from_document(document)
        validate_semantic_candidate_lattice(
            lattice,
            expected_source_media_sha256=str(document["source"]["sha256"]),
            expected_transcript_sha256=canonical_json_sha256(document),
        )
        case_id = _safe_name(
            str(document.get("jobId") or canonical_path.stem)
        )
        if case_id in seen_ids:
            raise ValueError(f"duplicate audit case ID: {case_id}")
        seen_ids.add(case_id)
        lattice_path: Path | None = None
        if persist_lattices:
            lattice_path = (
                output_directory
                / "lattices"
                / case_id
                / "semantic-candidate-lattice.v1.json"
            )
            atomic_write_json(lattice_path, lattice)
        tamper_checks = _tamper_checks(lattice)
        cases.append(
            {
                "caseId": case_id,
                "transcript": {
                    "path": str(canonical_path),
                    "fileSha256": sha256_file(canonical_path),
                    "canonicalSha256": canonical_json_sha256(document),
                    "segmentCount": len(document["segments"]),
                },
                "lattice": {
                    "latticeId": lattice["latticeId"],
                    "latticeSha256": lattice["latticeSha256"],
                    "path": str(lattice_path) if lattice_path is not None else None,
                    "persistedFileSha256": (
                        sha256_file(lattice_path)
                        if lattice_path is not None
                        else None
                    ),
                    "sourceMediaSha256": lattice["binding"][
                        "sourceMediaSha256"
                    ],
                    "availability": lattice["availability"],
                    "domains": [
                        _domain_summary(domain)
                        for domain in lattice["domains"]
                    ],
                },
                "validation": {
                    "runtimeRebuildPassed": True,
                    "sourceBindingPassed": True,
                    "transcriptBindingPassed": True,
                    **tamper_checks,
                },
            }
        )
    report = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "reportType": "semantic-candidate-lattice-audit",
        "generatedAt": _utc_now(),
        "privacy": {
            "transcriptTextPersistedInReport": False,
            "candidatePayloadPersistedOnlyWhenRequested": persist_lattices,
        },
        "caseCount": len(cases),
        "allCasesValid": all(
            all(case["validation"].values()) for case in cases
        ),
        "allRequiredDomainsAvailable": all(
            case["lattice"]["availability"][
                "allRequiredDomainsAvailable"
            ]
            for case in cases
        ),
        "cases": cases,
    }
    atomic_write_json(output_directory / "audit-report.v1.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript",
        action="append",
        type=Path,
        required=True,
        help="Path to a transcript-document.v2.json artifact; repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persist-lattices", action="store_true")
    args = parser.parse_args()
    report = audit_transcripts(
        args.transcript,
        output_directory=args.output.expanduser().resolve(),
        persist_lattices=args.persist_lattices,
    )
    print(
        f"audited {report['caseCount']} transcript(s); "
        f"allCasesValid={str(report['allCasesValid']).lower()}; "
        "allRequiredDomainsAvailable="
        f"{str(report['allRequiredDomainsAvailable']).lower()}"
    )
    return 0 if report["allCasesValid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
