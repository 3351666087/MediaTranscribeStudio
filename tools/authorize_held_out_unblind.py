"""Seal a completed blind review before held-out identity or truth access.

The authorizer reads only the public package manifest, reviewer manifest,
review template, and a completed copy of that template. It deliberately never
opens the identity vault and accepts no reference, truth, oracle, or scoring
input. The resulting no-replace artifact authorizes a separate post-review
process to unblind identities and begin objective held-out scoring.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
    validate_strict_json,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_TYPE = "held-out-unblind-authorization"
REVIEW_SOURCES = frozenset({"codex-agent", "human"})
SEVERITIES = frozenset({"blocker", "major", "minor", "pass"})
REVIEW_DIMENSIONS = (
    "speakerTimelineReview",
    "rawAsrReview",
    "finalTranscriptReview",
    "subtitlePdfReview",
)
_REVIEW_CASE_FIELDS = frozenset(
    {
        "reviewCaseId",
        "candidateOrder",
        "preferredCandidateId",
        "tie",
        *REVIEW_DIMENSIONS,
        "notes",
    }
)


class HeldOutUnblindAuthorizationError(ValueError):
    """Raised when the sealed blind review is incomplete or rebound."""


def _text(value: Any, *, field: str, maximum: int = 2000) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or any(
            ord(character) < 32 and character not in "\t\n\r"
            for character in value
        )
    ):
        raise HeldOutUnblindAuthorizationError(
            f"{field} must be trimmed non-empty text"
        )
    return value


def _sha256(value: Any, *, field: str) -> str:
    digest = _text(value, field=field, maximum=64)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise HeldOutUnblindAuthorizationError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return digest


def _load_object(path: Path, *, field: str) -> tuple[Path, dict[str, Any]]:
    if path.is_symlink():
        raise HeldOutUnblindAuthorizationError(f"{field} must be a regular file")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HeldOutUnblindAuthorizationError(
            f"{field} must be a regular file"
        ) from exc
    if not resolved.is_file():
        raise HeldOutUnblindAuthorizationError(f"{field} must be a regular file")
    try:
        value = read_json_strict(resolved)
        validate_strict_json(value)
    except Exception as exc:
        raise HeldOutUnblindAuthorizationError(f"{field} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise HeldOutUnblindAuthorizationError(f"{field} must be an object")
    return resolved, copy.deepcopy(dict(value))


def _contained_file(root: Path, relative: Any, *, field: str) -> Path:
    text = _text(relative, field=field, maximum=500)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise HeldOutUnblindAuthorizationError(
            f"{field} must be a safe relative path"
        )
    resolved_root = root.resolve(strict=True)
    unresolved = resolved_root
    for part in logical.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise HeldOutUnblindAuthorizationError(
                f"{field} must not traverse a symbolic link"
            )
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise HeldOutUnblindAuthorizationError(
            f"{field} escapes the package"
        ) from exc
    if not path.is_file():
        raise HeldOutUnblindAuthorizationError(
            f"{field} must resolve to a regular file"
        )
    return path


def _validate_dimension(value: Any, *, field: str) -> dict[str, Any]:
    required = {"severity", "reason", "evidence"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise HeldOutUnblindAuthorizationError(
            f"{field} must contain severity, reason, and evidence"
        )
    severity = value.get("severity")
    if severity not in SEVERITIES:
        raise HeldOutUnblindAuthorizationError(f"{field}.severity is invalid")
    reason = _text(value.get("reason"), field=f"{field}.reason", maximum=4000)
    raw_evidence = value.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise HeldOutUnblindAuthorizationError(
            f"{field}.evidence must be a non-empty array"
        )
    evidence = [
        _text(item, field=f"{field}.evidence[{index}]", maximum=1000)
        for index, item in enumerate(raw_evidence)
    ]
    return {"severity": severity, "reason": reason, "evidence": evidence}


def _validate_completed_review(
    value: Mapping[str, Any],
    *,
    template: Mapping[str, Any],
    reviewer_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    expected_top = {
        "schemaVersion",
        "artifactType",
        "packageId",
        "automaticScoring",
        "cases",
    }
    if set(value) != expected_top:
        raise HeldOutUnblindAuthorizationError(
            "completed review form fields are invalid"
        )
    for field in ("schemaVersion", "artifactType", "packageId", "automaticScoring"):
        if value.get(field) != template.get(field):
            raise HeldOutUnblindAuthorizationError(
                f"completed review form {field} differs from its template"
            )
    if value.get("artifactType") != "blind-e2e-human-review-form":
        raise HeldOutUnblindAuthorizationError(
            "completed review form artifactType is unsupported"
        )
    if value.get("automaticScoring") is not False:
        raise HeldOutUnblindAuthorizationError(
            "completed review must remain a manual blind decision"
        )
    if template.get("packageId") != reviewer_manifest.get("packageId"):
        raise HeldOutUnblindAuthorizationError(
            "review template is rebound to another package"
        )
    raw_template_cases = template.get("cases")
    raw_review_cases = value.get("cases")
    raw_manifest_cases = reviewer_manifest.get("cases")
    if not all(
        isinstance(items, list)
        for items in (raw_template_cases, raw_review_cases, raw_manifest_cases)
    ):
        raise HeldOutUnblindAuthorizationError("blind review cases are invalid")
    assert isinstance(raw_template_cases, list)
    assert isinstance(raw_review_cases, list)
    assert isinstance(raw_manifest_cases, list)
    if len(raw_review_cases) != len(raw_template_cases) or len(raw_review_cases) != len(
        raw_manifest_cases
    ):
        raise HeldOutUnblindAuthorizationError(
            "completed review must decide every blind case exactly once"
        )

    manifest_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_manifest_cases):
        if not isinstance(raw, Mapping):
            raise HeldOutUnblindAuthorizationError(
                f"reviewer manifest cases[{index}] is invalid"
            )
        case_id = _text(
            raw.get("reviewCaseId"),
            field=f"reviewer manifest cases[{index}].reviewCaseId",
        )
        if case_id in manifest_by_id:
            raise HeldOutUnblindAuthorizationError(
                "reviewer manifest contains duplicate case IDs"
            )
        manifest_by_id[case_id] = raw

    template_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_template_cases):
        if not isinstance(raw, Mapping) or set(raw) != _REVIEW_CASE_FIELDS:
            raise HeldOutUnblindAuthorizationError(
                f"review template cases[{index}] is invalid"
            )
        case_id = _text(
            raw.get("reviewCaseId"),
            field=f"review template cases[{index}].reviewCaseId",
        )
        if case_id in template_by_id:
            raise HeldOutUnblindAuthorizationError(
                "review template contains duplicate case IDs"
            )
        template_by_id[case_id] = raw
    if set(template_by_id) != set(manifest_by_id):
        raise HeldOutUnblindAuthorizationError(
            "review template is rebound to different blind cases"
        )

    completed: list[dict[str, Any]] = []
    severity_counts = {severity: 0 for severity in sorted(SEVERITIES)}
    seen: set[str] = set()
    for index, raw in enumerate(raw_review_cases):
        if not isinstance(raw, Mapping) or set(raw) != _REVIEW_CASE_FIELDS:
            raise HeldOutUnblindAuthorizationError(
                f"completed review cases[{index}] fields are invalid"
            )
        case_id = _text(
            raw.get("reviewCaseId"),
            field=f"completed review cases[{index}].reviewCaseId",
        )
        if case_id in seen or case_id not in template_by_id:
            raise HeldOutUnblindAuthorizationError(
                "completed review case IDs are duplicated or unknown"
            )
        seen.add(case_id)
        manifest_case = manifest_by_id[case_id]
        raw_candidates = manifest_case.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise HeldOutUnblindAuthorizationError(
                f"reviewer manifest {case_id} candidates are invalid"
            )
        candidate_order = [
            _text(
                candidate.get("reviewCandidateId")
                if isinstance(candidate, Mapping)
                else None,
                field=f"reviewer manifest {case_id} candidate",
            )
            for candidate in raw_candidates
        ]
        if len(candidate_order) != len(set(candidate_order)):
            raise HeldOutUnblindAuthorizationError(
                f"reviewer manifest {case_id} repeats a candidate"
            )
        template_case = template_by_id[case_id]
        if (
            template_case.get("candidateOrder") != candidate_order
            or template_case.get("preferredCandidateId") is not None
            or template_case.get("tie") is not None
            or any(template_case.get(dimension) is not None for dimension in REVIEW_DIMENSIONS)
            or template_case.get("notes") != []
        ):
            raise HeldOutUnblindAuthorizationError(
                f"review template {case_id} is not the original blank reviewer form"
            )
        if raw.get("candidateOrder") != candidate_order:
            raise HeldOutUnblindAuthorizationError(
                f"completed review {case_id} candidate order changed"
            )
        tie = raw.get("tie")
        preferred = raw.get("preferredCandidateId")
        if tie is True:
            if preferred is not None:
                raise HeldOutUnblindAuthorizationError(
                    f"completed review {case_id} tie cannot prefer one candidate"
                )
        elif tie is False:
            if preferred not in candidate_order:
                raise HeldOutUnblindAuthorizationError(
                    f"completed review {case_id} preference is invalid"
                )
        else:
            raise HeldOutUnblindAuthorizationError(
                f"completed review {case_id} tie decision is incomplete"
            )
        dimensions = {
            dimension: _validate_dimension(
                raw.get(dimension),
                field=f"completed review {case_id}.{dimension}",
            )
            for dimension in REVIEW_DIMENSIONS
        }
        for dimension in dimensions.values():
            severity_counts[str(dimension["severity"])] += 1
        notes = raw.get("notes")
        if not isinstance(notes, list):
            raise HeldOutUnblindAuthorizationError(
                f"completed review {case_id}.notes must be an array"
            )
        normalized_notes = [
            _text(note, field=f"completed review {case_id}.notes[{note_index}]")
            for note_index, note in enumerate(notes)
        ]
        completed.append(
            {
                "reviewCaseId": case_id,
                "candidateOrder": candidate_order,
                "preferredCandidateId": preferred,
                "tie": tie,
                **dimensions,
                "notes": normalized_notes,
            }
        )
    if seen != set(template_by_id):
        raise HeldOutUnblindAuthorizationError(
            "completed review omitted one or more blind cases"
        )
    return {
        "schemaVersion": value["schemaVersion"],
        "artifactType": value["artifactType"],
        "packageId": value["packageId"],
        "automaticScoring": False,
        "cases": completed,
    }, severity_counts


def authorize_held_out_unblind(
    *,
    package_root: Path,
    completed_review_path: Path,
    held_out_freeze_manifest_sha256: str,
    expected_package_manifest_sha256: str,
    reviewer_source: str,
    reviewer: str,
    reviewed_at: str,
    output_path: Path,
) -> dict[str, Any]:
    """Publish review-complete authorization without opening private evidence."""

    if package_root.is_symlink():
        raise HeldOutUnblindAuthorizationError(
            "blind package root must not be a symbolic link"
        )
    try:
        root = package_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HeldOutUnblindAuthorizationError(
            "blind package root must be a directory"
        ) from exc
    if not root.is_dir():
        raise HeldOutUnblindAuthorizationError(
            "blind package root must be a directory"
        )
    output = output_path.expanduser().absolute()
    completed_path = completed_review_path.expanduser().absolute()
    for path, field in (
        (output, "authorization output"),
        (completed_path, "completed review"),
    ):
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError:
            continue
        raise HeldOutUnblindAuthorizationError(
            f"{field} must remain outside the immutable blind package"
        )

    expected_package_sha256 = _sha256(
        expected_package_manifest_sha256,
        field="expectedPackageManifestSha256",
    )
    freeze_manifest_sha256 = _sha256(
        held_out_freeze_manifest_sha256,
        field="heldOutFreezeManifestSha256",
    )
    manifest_path, package = _load_object(
        root / "package-manifest.v1.json",
        field="package manifest",
    )
    package_manifest_sha256 = sha256_file(manifest_path)
    if package_manifest_sha256 != expected_package_sha256:
        raise HeldOutUnblindAuthorizationError(
            "package manifest SHA-256 differs from the externally frozen digest"
        )
    if package.get("artifactType") != "blind-e2e-package-files":
        raise HeldOutUnblindAuthorizationError(
            "package manifest artifactType is unsupported"
        )
    if package.get("referenceInputsAccepted") is not False:
        raise HeldOutUnblindAuthorizationError(
            "blind package accepted reference inputs"
        )
    package_id = _text(
        package.get("packageId"), field="package manifest packageId"
    )
    reviewer_evidence = package.get("reviewerPacket")
    identity_commitment = package.get("identityVault")
    if not isinstance(reviewer_evidence, Mapping) or not isinstance(
        identity_commitment, Mapping
    ):
        raise HeldOutUnblindAuthorizationError(
            "package manifest evidence bindings are invalid"
        )
    if set(identity_commitment) != {
        "relativePath",
        "fileSha256",
        "canonicalSha256",
    }:
        raise HeldOutUnblindAuthorizationError(
            "package identity-vault commitment fields are invalid"
        )
    identity_file_sha256 = _sha256(
        identity_commitment.get("fileSha256"),
        field="package identity-vault fileSha256",
    )
    identity_canonical_sha256 = _sha256(
        identity_commitment.get("canonicalSha256"),
        field="package identity-vault canonicalSha256",
    )
    reviewer_manifest_path = _contained_file(
        root,
        reviewer_evidence.get("relativePath"),
        field="package reviewer manifest path",
    )
    if sha256_file(reviewer_manifest_path) != reviewer_evidence.get("fileSha256"):
        raise HeldOutUnblindAuthorizationError(
            "reviewer manifest file SHA-256 does not match"
        )
    _, reviewer_manifest = _load_object(
        reviewer_manifest_path,
        field="reviewer manifest",
    )
    reviewer_body = dict(reviewer_manifest)
    reviewer_canonical = reviewer_body.pop("canonicalSha256", None)
    if (
        reviewer_canonical != canonical_json_sha256(reviewer_body)
        or reviewer_canonical != reviewer_evidence.get("canonicalSha256")
    ):
        raise HeldOutUnblindAuthorizationError(
            "reviewer manifest canonical SHA-256 does not match"
        )
    if reviewer_manifest.get("packageId") != package_id:
        raise HeldOutUnblindAuthorizationError(
            "reviewer manifest is rebound to another package"
        )
    reviewer_identity_commitment = reviewer_manifest.get("identityCommitment")
    if not isinstance(reviewer_identity_commitment, Mapping) or set(
        reviewer_identity_commitment
    ) != {"fileSha256", "canonicalSha256"}:
        raise HeldOutUnblindAuthorizationError(
            "reviewer manifest identity commitment is invalid"
        )
    if (
        _sha256(
            reviewer_identity_commitment.get("fileSha256"),
            field="reviewer identity commitment fileSha256",
        )
        != identity_file_sha256
        or _sha256(
            reviewer_identity_commitment.get("canonicalSha256"),
            field="reviewer identity commitment canonicalSha256",
        )
        != identity_canonical_sha256
    ):
        raise HeldOutUnblindAuthorizationError(
            "package and reviewer identity commitments differ"
        )
    blindness = reviewer_manifest.get("blindnessPolicy")
    if not isinstance(blindness, Mapping) or any(
        blindness.get(field) is not False
        for field in (
            "candidateIdentityPersisted",
            "originalPathPersisted",
            "referenceAnswerPersisted",
            "automaticScorePersisted",
            "referenceInputsAccepted",
        )
    ):
        raise HeldOutUnblindAuthorizationError(
            "reviewer manifest blindness policy is invalid"
        )
    review_template_evidence = reviewer_manifest.get("reviewForm")
    if not isinstance(review_template_evidence, Mapping):
        raise HeldOutUnblindAuthorizationError(
            "reviewer manifest review-form binding is invalid"
        )
    template_path = _contained_file(
        reviewer_manifest_path.parent,
        review_template_evidence.get("path"),
        field="review form template path",
    )
    if sha256_file(template_path) != review_template_evidence.get("sha256"):
        raise HeldOutUnblindAuthorizationError(
            "review form template SHA-256 does not match"
        )
    _, template = _load_object(template_path, field="review form template")
    completed_file, completed_review = _load_object(
        completed_path,
        field="completed review",
    )
    normalized_review, severity_counts = _validate_completed_review(
        completed_review,
        template=template,
        reviewer_manifest=reviewer_manifest,
    )

    source = _text(reviewer_source, field="reviewerSource", maximum=40)
    if source not in REVIEW_SOURCES:
        raise HeldOutUnblindAuthorizationError("reviewerSource is unsupported")
    reviewer_name = _text(reviewer, field="reviewer", maximum=200)
    reviewed_text = _text(reviewed_at, field="reviewedAt", maximum=80)
    try:
        reviewed_time = datetime.fromisoformat(reviewed_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HeldOutUnblindAuthorizationError(
            "reviewedAt must be an ISO-8601 timestamp"
        ) from exc
    if reviewed_time.tzinfo is None:
        raise HeldOutUnblindAuthorizationError(
            "reviewedAt must include a timezone"
        )

    identity_path = _text(
        identity_commitment.get("relativePath"),
        field="identity vault commitment path",
        maximum=500,
    )
    identity_logical = PurePosixPath(identity_path)
    if (
        identity_logical.is_absolute()
        or ".." in identity_logical.parts
        or "\\" in identity_path
        or not identity_logical.parts
        or identity_logical.parts[0] != "identity-vault"
        or len(identity_logical.parts) < 2
    ):
        raise HeldOutUnblindAuthorizationError(
            "identity vault commitment path is invalid"
        )
    body: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "packageId": package_id,
        "authorizationScope": "identity-unblind-and-post-review-objective-scoring",
        "reviewerSource": source,
        "reviewer": reviewer_name,
        "reviewedAt": reviewed_text,
        "packageManifest": {
            "path": str(manifest_path),
            "fileSha256": package_manifest_sha256,
        },
        "heldOutFreezeManifest": {
            "fileSha256": freeze_manifest_sha256,
            "readByAuthorizer": False,
        },
        "reviewerManifest": {
            "path": str(reviewer_manifest_path),
            "fileSha256": sha256_file(reviewer_manifest_path),
            "canonicalSha256": reviewer_canonical,
        },
        "reviewTemplate": {
            "path": str(template_path),
            "fileSha256": sha256_file(template_path),
        },
        "completedReview": {
            "path": str(completed_file),
            "fileSha256": sha256_file(completed_file),
            "canonicalSha256": canonical_json_sha256(normalized_review),
        },
        "identityVaultCommitment": {
            "relativePath": identity_path,
            "fileSha256": identity_file_sha256,
            "canonicalSha256": identity_canonical_sha256,
        },
        "counts": {
            "reviewedCases": len(normalized_review["cases"]),
            "dimensionDecisions": len(normalized_review["cases"])
            * len(REVIEW_DIMENSIONS),
            "severityByDimension": severity_counts,
        },
        "isolationPolicy": {
            "identityVaultReadByAuthorizer": False,
            "heldOutFreezeManifestReadByAuthorizer": False,
            "referenceTruthReadByAuthorizer": False,
            "automaticScoresReadByAuthorizer": False,
            "reviewCompletedBeforeUnblind": True,
        },
    }
    body["canonicalSha256"] = canonical_json_sha256(body)
    atomic_write_json_no_replace(output, body)
    return body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--completed-review", type=Path, required=True)
    parser.add_argument("--held-out-freeze-manifest-sha256", required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument(
        "--reviewer-source",
        choices=sorted(REVIEW_SOURCES),
        required=True,
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        authorization = authorize_held_out_unblind(
            package_root=arguments.package_root,
            completed_review_path=arguments.completed_review,
            held_out_freeze_manifest_sha256=(
                arguments.held_out_freeze_manifest_sha256
            ),
            expected_package_manifest_sha256=(
                arguments.expected_package_manifest_sha256
            ),
            reviewer_source=arguments.reviewer_source,
            reviewer=arguments.reviewer,
            reviewed_at=arguments.reviewed_at,
            output_path=arguments.output,
        )
    except (OSError, HeldOutUnblindAuthorizationError) as exc:
        print(f"held-out unblind authorization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(authorization, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
