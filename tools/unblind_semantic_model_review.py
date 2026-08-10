#!/usr/bin/env python3
"""Unblind one sealed semantic-model review and publish its comparison.

The identity vault is not opened until the sealed review, package manifest,
and every anonymous reviewer packet have passed their hash, canonical JSON,
and exact coverage checks.  The resulting comparison is immutable evidence;
this tool never reads or mutates production configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    validate_strict_json,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_TYPE = "semantic-model-blind-review-comparison"
REVIEW_ARTIFACT_TYPE = "codex-semantic-blind-review"
MANIFEST_ARTIFACT_TYPE = "anonymous-semantic-blind-review-manifest"
PACKET_ARTIFACT_TYPE = "anonymous-semantic-blind-review-packet"
VAULT_ARTIFACT_TYPE = "anonymous-semantic-blind-review-identity-vault"
EXPECTED_PACKET_COUNT = 22
MAX_CANDIDATE_COUNT = 99
SEVERITIES = ("blocker", "major", "minor", "pass")
_SEVERITY_SET = frozenset(SEVERITIES)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
# Ollama model references are not artifact identifiers: tags may contain
# uppercase quantization markers, underscores, slashes, colons, and @ digests.
# Keep the reference constrained to a portable, non-whitespace token while
# retaining the stricter `_IDENTIFIER` contract for review/case IDs.
_MODEL_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]*$")
_CASE_ALIAS = re.compile(r"^case-[0-9]{3}$")
_CANDIDATE_ALIAS = re.compile(r"^candidate-[0-9]{2}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024


class SemanticModelUnblindError(ValueError):
    """Raised when sealed blind-review evidence cannot be unblinded safely."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticModelUnblindError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise SemanticModelUnblindError(f"non-finite JSON number: {value}")


def _load_json_snapshot(
    path: Path,
    *,
    label: str,
) -> tuple[Path, dict[str, Any], str, int]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise SemanticModelUnblindError(f"{label} must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SemanticModelUnblindError(f"{label} is missing: {path}") from exc
    if not resolved.is_file():
        raise SemanticModelUnblindError(f"{label} must be a regular file")
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise SemanticModelUnblindError(f"{label} could not be read") from exc
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise SemanticModelUnblindError(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
        validate_strict_json(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, SemanticModelUnblindError):
            raise
        raise SemanticModelUnblindError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SemanticModelUnblindError(f"{label} must contain an object")
    return resolved, value, hashlib.sha256(raw).hexdigest(), len(raw)


def _object(
    value: Any,
    *,
    field: str,
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticModelUnblindError(f"{field} must be an object")
    result = {str(key): item for key, item in value.items()}
    missing = sorted(required - set(result))
    unknown = sorted(set(result) - required)
    if missing or unknown:
        raise SemanticModelUnblindError(
            f"{field} fields are invalid: missing={missing}, unknown={unknown}"
        )
    return result


def _text(value: Any, *, field: str, maximum: int = 1000) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise SemanticModelUnblindError(f"{field} must be trimmed non-empty text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SemanticModelUnblindError(f"{field} contains control characters")
    return value


def _sha256(value: Any, *, field: str) -> str:
    result = _text(value, field=field, maximum=80).casefold()
    if result.startswith("sha256:"):
        result = result.removeprefix("sha256:")
    if _SHA256.fullmatch(result) is None:
        raise SemanticModelUnblindError(f"{field} must be a SHA-256 digest")
    return result


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SemanticModelUnblindError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _canonical_body(value: Mapping[str, Any], *, field: str) -> str:
    body = dict(value)
    declared = _sha256(body.pop("canonicalSha256", None), field=f"{field}.canonicalSha256")
    actual = canonical_json_sha256(body)
    if actual != declared:
        raise SemanticModelUnblindError(f"{field} canonical SHA-256 does not match")
    return declared


def _resolve_package_root(package_root: Path) -> Path:
    candidate = package_root.expanduser()
    if candidate.is_symlink():
        raise SemanticModelUnblindError("package root must not be a symbolic link")
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise SemanticModelUnblindError("package root is missing") from exc
    if not root.is_dir():
        raise SemanticModelUnblindError("package root must be a directory")
    return root


def _package_member(root: Path, relative_path: Any, *, field: str) -> Path:
    relative = Path(_text(relative_path, field=field, maximum=500))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SemanticModelUnblindError(f"{field} must remain within the package")
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise SemanticModelUnblindError(
                f"{field} must not traverse a symbolic link"
            )
    try:
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SemanticModelUnblindError(f"{field} escapes the package") from exc
    if not resolved.is_file():
        raise SemanticModelUnblindError(f"{field} must resolve to a regular file")
    return resolved


def _validate_review_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    review = _object(
        value,
        field="sealed review",
        required=frozenset(
            {
                "schemaVersion",
                "artifactType",
                "reviewId",
                "blindPackage",
                "reviewPolicy",
                "reviewer",
                "cases",
                "bodySourceFileSha256",
                "validation",
                "canonicalSha256",
            }
        ),
    )
    if review["schemaVersion"] != SCHEMA_VERSION:
        raise SemanticModelUnblindError("sealed review schemaVersion is unsupported")
    if review["artifactType"] != REVIEW_ARTIFACT_TYPE:
        raise SemanticModelUnblindError("sealed review artifactType is unsupported")
    _canonical_body(review, field="sealed review")
    review_id = _text(review["reviewId"], field="sealed review.reviewId", maximum=200)
    if _IDENTIFIER.fullmatch(review_id) is None:
        raise SemanticModelUnblindError("sealed review.reviewId is invalid")
    _sha256(
        review["bodySourceFileSha256"],
        field="sealed review.bodySourceFileSha256",
    )

    blind_package = _object(
        review["blindPackage"],
        field="sealed review.blindPackage",
        required=frozenset(
            {
                "manifestFileSha256",
                "packetCount",
                "candidatesPerPacket",
                "manifestCanonicalSha256",
                "reviewerPacketSetSha256",
            }
        ),
    )
    if _positive_int(blind_package["packetCount"], field="review packetCount") != EXPECTED_PACKET_COUNT:
        raise SemanticModelUnblindError("sealed review must bind exactly 22 packets")
    candidate_count = _positive_int(
        blind_package["candidatesPerPacket"],
        field="review candidatesPerPacket",
    )
    if candidate_count < 2 or candidate_count > MAX_CANDIDATE_COUNT:
        raise SemanticModelUnblindError(
            "sealed review candidate count must be between 2 and 99"
        )
    for name in (
        "manifestFileSha256",
        "manifestCanonicalSha256",
        "reviewerPacketSetSha256",
    ):
        _sha256(blind_package[name], field=f"sealed review.blindPackage.{name}")

    policy = _object(
        review["reviewPolicy"],
        field="sealed review.reviewPolicy",
        required=frozenset(
            {
                "automaticScoringUsed",
                "referenceTranscriptUsed",
                "identityVaultReadBeforeSealing",
                "allowedSeverities",
                "preferenceRule",
            }
        ),
    )
    if (
        policy["automaticScoringUsed"] is not False
        or policy["referenceTranscriptUsed"] is not False
        or policy["identityVaultReadBeforeSealing"] is not False
    ):
        raise SemanticModelUnblindError(
            "sealed review violates its blind manual-review policy"
        )
    if policy["allowedSeverities"] != list(SEVERITIES):
        raise SemanticModelUnblindError("sealed review severity policy is invalid")
    _text(policy["preferenceRule"], field="sealed review.preferenceRule", maximum=4000)

    reviewer = _object(
        review["reviewer"],
        field="sealed review.reviewer",
        required=frozenset({"source", "actor", "reviewedAt"}),
    )
    if reviewer["source"] not in {"human", "codex-agent"}:
        raise SemanticModelUnblindError("sealed review reviewer source is invalid")
    _text(reviewer["actor"], field="sealed review.reviewer.actor", maximum=200)
    _text(reviewer["reviewedAt"], field="sealed review.reviewer.reviewedAt", maximum=100)

    validation = _object(
        review["validation"],
        field="sealed review.validation",
        required=frozenset(
            {
                "reviewedCaseCount",
                "assessedCandidateCount",
                "allCandidatesAssessedExactlyOnce",
                "identityVaultRead",
                "sealedBeforeUnblind",
            }
        ),
    )
    if validation["reviewedCaseCount"] != EXPECTED_PACKET_COUNT:
        raise SemanticModelUnblindError("sealed review validation case count is invalid")
    if validation["assessedCandidateCount"] != (
        EXPECTED_PACKET_COUNT * candidate_count
    ):
        raise SemanticModelUnblindError(
            "sealed review validation candidate count is invalid"
        )
    if validation["allCandidatesAssessedExactlyOnce"] is not True:
        raise SemanticModelUnblindError(
            "sealed review does not cover every candidate exactly once"
        )
    if validation["identityVaultRead"] is not False:
        raise SemanticModelUnblindError(
            "sealed review declares that the identity vault was already read"
        )
    if validation["sealedBeforeUnblind"] is not True:
        raise SemanticModelUnblindError("sealedBeforeUnblind must be true")
    cases = review["cases"]
    if not isinstance(cases, list) or len(cases) != EXPECTED_PACKET_COUNT:
        raise SemanticModelUnblindError("sealed review must contain exactly 22 cases")
    return review


def _validate_manifest(
    value: Mapping[str, Any],
    *,
    file_sha256: str,
    review: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _object(
        value,
        field="blind manifest",
        required=frozenset(
            {
                "schemaVersion",
                "artifactType",
                "publicationPolicy",
                "manifestWrittenLast",
                "packetCount",
                "candidatesPerPacket",
                "files",
                "canonicalSha256",
            }
        ),
    )
    if manifest["schemaVersion"] != SCHEMA_VERSION:
        raise SemanticModelUnblindError("blind manifest schemaVersion is unsupported")
    if manifest["artifactType"] != MANIFEST_ARTIFACT_TYPE:
        raise SemanticModelUnblindError("blind manifest artifactType is unsupported")
    if (
        manifest["publicationPolicy"] != "atomic-directory-no-replace"
        or manifest["manifestWrittenLast"] is not True
    ):
        raise SemanticModelUnblindError("blind manifest publication policy is invalid")
    canonical = _canonical_body(manifest, field="blind manifest")
    binding = review["blindPackage"]
    assert isinstance(binding, Mapping)
    if file_sha256 != _sha256(
        binding.get("manifestFileSha256"),
        field="sealed review manifest file binding",
    ):
        raise SemanticModelUnblindError("blind manifest file SHA-256 does not match review")
    if canonical != _sha256(
        binding.get("manifestCanonicalSha256"),
        field="sealed review manifest canonical binding",
    ):
        raise SemanticModelUnblindError(
            "blind manifest canonical SHA-256 does not match review"
        )
    if manifest["packetCount"] != EXPECTED_PACKET_COUNT:
        raise SemanticModelUnblindError("blind manifest must contain 22 packets")
    candidate_count = _positive_int(
        binding.get("candidatesPerPacket"),
        field="sealed review candidatesPerPacket binding",
    )
    if manifest["candidatesPerPacket"] != candidate_count:
        raise SemanticModelUnblindError(
            "blind manifest candidate count does not match review"
        )
    raw_files = manifest["files"]
    if not isinstance(raw_files, list) or len(raw_files) != EXPECTED_PACKET_COUNT + 1:
        raise SemanticModelUnblindError("blind manifest file inventory is incomplete")
    packet_rows: list[dict[str, Any]] = []
    vault_row: dict[str, Any] | None = None
    paths: set[str] = set()
    required_file = frozenset(
        {"role", "relativePath", "canonicalSha256", "fileSha256", "sizeBytes"}
    )
    for index, raw in enumerate(raw_files):
        row = _object(raw, field=f"blind manifest files[{index}]", required=required_file)
        relative = _text(
            row["relativePath"],
            field=f"blind manifest files[{index}].relativePath",
            maximum=500,
        )
        if relative in paths:
            raise SemanticModelUnblindError("blind manifest repeats a file path")
        paths.add(relative)
        _sha256(row["canonicalSha256"], field=f"blind manifest files[{index}].canonicalSha256")
        _sha256(row["fileSha256"], field=f"blind manifest files[{index}].fileSha256")
        _positive_int(row["sizeBytes"], field=f"blind manifest files[{index}].sizeBytes")
        if row["role"] == "reviewer-packet":
            packet_rows.append(row)
        elif row["role"] == "identity-vault" and vault_row is None:
            vault_row = row
        else:
            raise SemanticModelUnblindError("blind manifest contains an unsupported file role")
    if len(packet_rows) != EXPECTED_PACKET_COUNT or vault_row is None:
        raise SemanticModelUnblindError(
            "blind manifest must contain 22 packets and one identity vault"
        )
    if vault_row["relativePath"] != "identity-vault.json":
        raise SemanticModelUnblindError("blind manifest identity-vault path is invalid")
    expected_packet_paths = {
        f"reviewer/case-{index:03d}.json"
        for index in range(1, EXPECTED_PACKET_COUNT + 1)
    }
    if {str(row["relativePath"]) for row in packet_rows} != expected_packet_paths:
        raise SemanticModelUnblindError("blind manifest packet path coverage is invalid")
    packet_rows.sort(key=lambda row: str(row["relativePath"]))
    return packet_rows, vault_row


def _validate_packet(
    value: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    file_sha256: str,
    size_bytes: int,
    candidate_count: int,
) -> dict[str, Any]:
    packet = _object(
        value,
        field="reviewer packet",
        required=frozenset(
            {
                "schemaVersion",
                "artifactType",
                "caseAlias",
                "sharedInputCanonicalSha256",
                "sharedInput",
                "candidates",
                "canonicalSha256",
            }
        ),
    )
    if packet["schemaVersion"] != SCHEMA_VERSION or packet["artifactType"] != PACKET_ARTIFACT_TYPE:
        raise SemanticModelUnblindError("reviewer packet schema or artifact type is invalid")
    canonical = _canonical_body(packet, field="reviewer packet")
    if canonical != _sha256(
        evidence.get("canonicalSha256"), field="manifest packet canonical SHA-256"
    ):
        raise SemanticModelUnblindError("reviewer packet canonical SHA-256 is rebound")
    if file_sha256 != _sha256(
        evidence.get("fileSha256"), field="manifest packet file SHA-256"
    ):
        raise SemanticModelUnblindError("reviewer packet file SHA-256 is rebound")
    if size_bytes != evidence.get("sizeBytes"):
        raise SemanticModelUnblindError("reviewer packet size is rebound")
    case_alias = _text(packet["caseAlias"], field="reviewer packet caseAlias", maximum=20)
    if _CASE_ALIAS.fullmatch(case_alias) is None:
        raise SemanticModelUnblindError("reviewer packet caseAlias is invalid")
    expected_relative = f"reviewer/{case_alias}.json"
    if evidence.get("relativePath") != expected_relative:
        raise SemanticModelUnblindError("reviewer packet path and alias differ")
    shared_input = packet["sharedInput"]
    if not isinstance(shared_input, Mapping):
        raise SemanticModelUnblindError("reviewer packet sharedInput is invalid")
    if canonical_json_sha256(shared_input) != _sha256(
        packet["sharedInputCanonicalSha256"],
        field="reviewer packet sharedInputCanonicalSha256",
    ):
        raise SemanticModelUnblindError("reviewer packet shared input SHA-256 does not match")
    raw_candidates = packet["candidates"]
    if not isinstance(raw_candidates, list) or len(raw_candidates) != candidate_count:
        raise SemanticModelUnblindError(
            "reviewer packet candidate count does not match manifest"
        )
    candidates: list[dict[str, Any]] = []
    aliases: set[str] = set()
    required_candidate = frozenset(
        {"candidateAlias", "resultCanonicalSha256", "result"}
    )
    for index, raw in enumerate(raw_candidates):
        candidate = _object(
            raw,
            field=f"reviewer packet candidates[{index}]",
            required=required_candidate,
        )
        alias = _text(
            candidate["candidateAlias"],
            field=f"reviewer packet candidates[{index}].candidateAlias",
            maximum=20,
        )
        if _CANDIDATE_ALIAS.fullmatch(alias) is None or alias in aliases:
            raise SemanticModelUnblindError("reviewer packet candidate aliases are invalid")
        aliases.add(alias)
        result = candidate["result"]
        if not isinstance(result, Mapping):
            raise SemanticModelUnblindError("reviewer packet candidate result is invalid")
        result_sha = _sha256(
            candidate["resultCanonicalSha256"],
            field=f"reviewer packet candidates[{index}].resultCanonicalSha256",
        )
        if canonical_json_sha256(result) != result_sha:
            raise SemanticModelUnblindError("reviewer packet result SHA-256 does not match")
        candidates.append(
            {"candidateAlias": alias, "resultCanonicalSha256": result_sha}
        )
    expected_aliases = {
        f"candidate-{index:02d}"
        for index in range(1, candidate_count + 1)
    }
    if aliases != expected_aliases:
        raise SemanticModelUnblindError("reviewer packet candidate coverage is incomplete")
    return {
        "caseAlias": case_alias,
        "fileSha256": file_sha256,
        "canonicalSha256": canonical,
        "sharedInputCanonicalSha256": str(packet["sharedInputCanonicalSha256"]),
        "candidates": candidates,
    }


def _validate_review_cases(
    review: Mapping[str, Any],
    *,
    packets: Mapping[str, Mapping[str, Any]],
    candidate_count: int,
) -> dict[str, dict[str, Any]]:
    raw_cases = review["cases"]
    assert isinstance(raw_cases, list)
    reviewed: dict[str, dict[str, Any]] = {}
    required_case = frozenset(
        {"caseAlias", "preferredCandidateAliases", "tie", "assessmentGroups"}
    )
    required_group = frozenset(
        {"candidateAliases", "severity", "reason", "evidence", "recommendedAction"}
    )
    required_evidence = frozenset({"startMs", "endMs", "observation"})
    assessed_count = 0
    for index, raw in enumerate(raw_cases):
        case = _object(raw, field=f"sealed review cases[{index}]", required=required_case)
        case_alias = _text(case["caseAlias"], field=f"sealed review cases[{index}].caseAlias", maximum=20)
        packet = packets.get(case_alias)
        if case_alias in reviewed or packet is None:
            raise SemanticModelUnblindError("sealed review case aliases are duplicated or unknown")
        candidate_aliases = {
            str(row["candidateAlias"])
            for row in packet["candidates"]
            if isinstance(row, Mapping)
        }
        preferred_raw = case["preferredCandidateAliases"]
        if (
            not isinstance(preferred_raw, list)
            or not preferred_raw
            or any(not isinstance(item, str) for item in preferred_raw)
            or len(set(preferred_raw)) != len(preferred_raw)
            or not set(preferred_raw) <= candidate_aliases
        ):
            raise SemanticModelUnblindError("sealed review preferred candidates are invalid")
        tie = case["tie"]
        if not isinstance(tie, bool):
            raise SemanticModelUnblindError("sealed review tie must be a boolean")
        if (tie and len(preferred_raw) < 2) or (not tie and len(preferred_raw) != 1):
            raise SemanticModelUnblindError("sealed review tie and preference disagree")
        raw_groups = case["assessmentGroups"]
        if not isinstance(raw_groups, list) or not raw_groups:
            raise SemanticModelUnblindError("sealed review assessmentGroups are invalid")
        severities: dict[str, str] = {}
        for group_index, raw_group in enumerate(raw_groups):
            group = _object(
                raw_group,
                field=f"sealed review {case_alias}.assessmentGroups[{group_index}]",
                required=required_group,
            )
            aliases = group["candidateAliases"]
            if (
                not isinstance(aliases, list)
                or not aliases
                or any(not isinstance(item, str) for item in aliases)
                or len(set(aliases)) != len(aliases)
                or not set(aliases) <= candidate_aliases
            ):
                raise SemanticModelUnblindError("sealed review assessment aliases are invalid")
            severity = group["severity"]
            if severity not in _SEVERITY_SET:
                raise SemanticModelUnblindError("sealed review assessment severity is invalid")
            _text(group["reason"], field="sealed review assessment reason", maximum=4000)
            _text(
                group["recommendedAction"],
                field="sealed review recommendedAction",
                maximum=4000,
            )
            evidence = _object(
                group["evidence"],
                field="sealed review assessment evidence",
                required=required_evidence,
            )
            start_ms = _positive_int(evidence["startMs"], field="review evidence startMs", allow_zero=True)
            end_ms = _positive_int(evidence["endMs"], field="review evidence endMs")
            if end_ms <= start_ms:
                raise SemanticModelUnblindError("sealed review evidence time range is invalid")
            _text(evidence["observation"], field="review evidence observation", maximum=4000)
            for alias in aliases:
                if alias in severities:
                    raise SemanticModelUnblindError(
                        "sealed review assesses one candidate more than once"
                    )
                severities[alias] = str(severity)
        if set(severities) != candidate_aliases:
            raise SemanticModelUnblindError(
                "sealed review must assess every packet candidate exactly once"
            )
        assessed_count += len(severities)
        reviewed[case_alias] = {
            "preferredCandidateAliases": tuple(str(item) for item in preferred_raw),
            "tie": tie,
            "severities": severities,
        }
    if set(reviewed) != set(packets) or assessed_count != (
        EXPECTED_PACKET_COUNT * candidate_count
    ):
        raise SemanticModelUnblindError("sealed review packet coverage is incomplete")
    return reviewed


def _validate_vault(
    value: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    file_sha256: str,
    size_bytes: int,
    packets: Mapping[str, Mapping[str, Any]],
    candidate_count: int,
) -> tuple[str, str, list[dict[str, str]], dict[str, dict[str, str]]]:
    vault = _object(
        value,
        field="identity vault",
        required=frozenset(
            {
                "schemaVersion",
                "artifactType",
                "seedSha256",
                "inputSetCanonicalSha256",
                "modelSetCanonicalSha256",
                "cases",
                "canonicalSha256",
            }
        ),
    )
    if vault["schemaVersion"] != SCHEMA_VERSION or vault["artifactType"] != VAULT_ARTIFACT_TYPE:
        raise SemanticModelUnblindError("identity vault schema or artifact type is invalid")
    canonical = _canonical_body(vault, field="identity vault")
    if canonical != _sha256(evidence.get("canonicalSha256"), field="manifest vault canonical SHA-256"):
        raise SemanticModelUnblindError("identity vault canonical SHA-256 is rebound")
    if file_sha256 != _sha256(evidence.get("fileSha256"), field="manifest vault file SHA-256"):
        raise SemanticModelUnblindError("identity vault file SHA-256 is rebound")
    if size_bytes != evidence.get("sizeBytes"):
        raise SemanticModelUnblindError("identity vault size is rebound")
    _sha256(vault["seedSha256"], field="identity vault seedSha256")
    case_set_sha256 = _sha256(
        vault["inputSetCanonicalSha256"], field="identity vault inputSetCanonicalSha256"
    )
    model_set_sha256 = _sha256(
        vault["modelSetCanonicalSha256"], field="identity vault modelSetCanonicalSha256"
    )
    raw_cases = vault["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_PACKET_COUNT:
        raise SemanticModelUnblindError("identity vault must contain exactly 22 cases")
    alias_bindings: dict[str, dict[str, str]] = {}
    input_binding: list[dict[str, str]] = []
    model_set: list[dict[str, str]] | None = None
    seen_case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _object(
            raw_case,
            field=f"identity vault cases[{index}]",
            required=frozenset({"caseAlias", "caseId", "caseSha256", "candidates"}),
        )
        case_alias = _text(case["caseAlias"], field="identity vault caseAlias", maximum=20)
        packet = packets.get(case_alias)
        if packet is None or case_alias in alias_bindings:
            raise SemanticModelUnblindError("identity vault case aliases are duplicated or unknown")
        case_id = _text(case["caseId"], field="identity vault caseId", maximum=200)
        if case_id in seen_case_ids:
            raise SemanticModelUnblindError("identity vault case IDs must be unique")
        seen_case_ids.add(case_id)
        case_sha = _sha256(case["caseSha256"], field="identity vault caseSha256")
        input_binding.append(
            {
                "caseId": case_id,
                "caseSha256": case_sha,
                "sharedInputCanonicalSha256": str(packet["sharedInputCanonicalSha256"]),
            }
        )
        packet_candidates = {
            str(item["candidateAlias"]): str(item["resultCanonicalSha256"])
            for item in packet["candidates"]
            if isinstance(item, Mapping)
        }
        raw_candidates = case["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) != candidate_count:
            raise SemanticModelUnblindError("identity vault candidate coverage is invalid")
        case_aliases: dict[str, str] = {}
        current_models: list[dict[str, str]] = []
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            candidate = _object(
                raw_candidate,
                field=f"identity vault {case_alias}.candidates[{candidate_index}]",
                required=frozenset(
                    {"candidateAlias", "model", "modelId", "digest", "resultCanonicalSha256"}
                ),
            )
            alias = _text(candidate["candidateAlias"], field="identity vault candidateAlias", maximum=20)
            if alias in case_aliases or alias not in packet_candidates:
                raise SemanticModelUnblindError("identity vault candidate aliases are invalid")
            model = _text(candidate["model"], field="identity vault model", maximum=200)
            model_id = _text(candidate["modelId"], field="identity vault modelId", maximum=200)
            if _MODEL_REFERENCE.fullmatch(model_id) is None:
                raise SemanticModelUnblindError("identity vault modelId is invalid")
            digest = _sha256(candidate["digest"], field="identity vault model digest")
            result_sha = _sha256(
                candidate["resultCanonicalSha256"],
                field="identity vault resultCanonicalSha256",
            )
            if result_sha != packet_candidates[alias]:
                raise SemanticModelUnblindError(
                    "identity vault alias/result SHA binding does not match packet"
                )
            case_aliases[alias] = model_id
            current_models.append(
                {"model": model, "modelId": model_id, "digest": f"sha256:{digest}"}
            )
        current_models.sort(
            key=lambda row: (row["model"], row["modelId"], row["digest"])
        )
        if model_set is None:
            model_set = current_models
            if len({row["modelId"] for row in model_set}) != len(model_set):
                raise SemanticModelUnblindError("identity vault model IDs must be unique")
        elif current_models != model_set:
            raise SemanticModelUnblindError(
                "identity vault cases do not contain one identical complete model set"
            )
        alias_bindings[case_alias] = case_aliases
    if set(alias_bindings) != set(packets):
        raise SemanticModelUnblindError("identity vault case coverage is incomplete")
    if canonical_json_sha256(input_binding) != case_set_sha256:
        raise SemanticModelUnblindError("identity vault case-set SHA-256 does not match")
    assert model_set is not None
    if canonical_json_sha256(model_set) != model_set_sha256:
        raise SemanticModelUnblindError("identity vault model-set SHA-256 does not match")
    return case_set_sha256, model_set_sha256, model_set, alias_bindings


def _aggregate_results(
    *,
    reviews: Mapping[str, Mapping[str, Any]],
    aliases: Mapping[str, Mapping[str, str]],
    models: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    counts = {
        str(model["modelId"]): Counter({severity: 0 for severity in SEVERITIES})
        for model in models
    }
    preferred_credit = {
        str(model["modelId"]): Fraction(0, 1) for model in models
    }
    for case_alias in sorted(reviews):
        review = reviews[case_alias]
        case_aliases = aliases.get(case_alias)
        if case_aliases is None:
            raise SemanticModelUnblindError("unblind alias coverage is incomplete")
        severities = review["severities"]
        assert isinstance(severities, Mapping)
        for candidate_alias, severity in severities.items():
            model_id = case_aliases.get(str(candidate_alias))
            if model_id not in counts:
                raise SemanticModelUnblindError("unblind model alias is unknown")
            counts[model_id][str(severity)] += 1
        preferred = review["preferredCandidateAliases"]
        assert isinstance(preferred, tuple)
        weight = Fraction(1, len(preferred))
        for candidate_alias in preferred:
            model_id = case_aliases.get(str(candidate_alias))
            if model_id not in preferred_credit:
                raise SemanticModelUnblindError("preferred model alias is unknown")
            preferred_credit[model_id] += weight

    model_by_id = {str(model["modelId"]): model for model in models}

    def ranking_key(model_id: str) -> tuple[Any, ...]:
        severity = counts[model_id]
        return (
            severity["blocker"],
            severity["major"],
            severity["minor"],
            -severity["pass"],
            -preferred_credit[model_id],
        )

    ordered_ids = sorted(model_by_id, key=lambda model_id: (ranking_key(model_id), model_id))
    best_key = ranking_key(ordered_ids[0])
    tied = sorted(model_id for model_id in ordered_ids if ranking_key(model_id) == best_key)
    unique_winner = tied[0] if len(tied) == 1 else None
    aggregate: list[dict[str, Any]] = []
    previous_key: tuple[Any, ...] | None = None
    rank = 0
    for position, model_id in enumerate(ordered_ids, start=1):
        key = ranking_key(model_id)
        if key != previous_key:
            rank = position
            previous_key = key
        credit = preferred_credit[model_id]
        share = credit / EXPECTED_PACKET_COUNT
        model = model_by_id[model_id]
        aggregate.append(
            {
                "rank": rank,
                "registryModelId": model_id,
                "model": str(model["model"]),
                "digest": str(model["digest"]),
                "caseCount": EXPECTED_PACKET_COUNT,
                "severityCounts": {
                    severity: counts[model_id][severity] for severity in SEVERITIES
                },
                "preferredCaseCredit": {
                    "numerator": credit.numerator,
                    "denominator": credit.denominator,
                },
                "weightedPreferredShare": round(float(share), 12),
            }
        )
    return aggregate, unique_winner, tied


def unblind_semantic_model_review(
    *,
    package_root: Path,
    sealed_review_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate, unblind, rank, and atomically publish one comparison."""

    output = output_path.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"comparison output already exists: {output}")
    root = _resolve_package_root(package_root)

    review_path, review_value, review_file_sha, _review_size = _load_json_snapshot(
        sealed_review_path, label="sealed review"
    )
    review = _validate_review_envelope(review_value)
    review_package = review["blindPackage"]
    assert isinstance(review_package, Mapping)
    candidate_count = int(review_package["candidatesPerPacket"])
    review_canonical_sha = _sha256(
        review["canonicalSha256"], field="sealed review canonical SHA-256"
    )

    manifest_path = _package_member(root, "manifest.json", field="blind manifest path")
    (
        _manifest_resolved,
        manifest_value,
        manifest_file_sha,
        _manifest_size,
    ) = _load_json_snapshot(manifest_path, label="blind manifest")
    packet_rows, vault_row = _validate_manifest(
        manifest_value,
        file_sha256=manifest_file_sha,
        review=review,
    )
    manifest_canonical_sha = _sha256(
        manifest_value["canonicalSha256"], field="blind manifest canonical SHA-256"
    )

    packets: dict[str, dict[str, Any]] = {}
    for row in packet_rows:
        packet_path = _package_member(
            root,
            row["relativePath"],
            field=f"reviewer packet {row['relativePath']}",
        )
        _resolved, packet_value, packet_file_sha, packet_size = _load_json_snapshot(
            packet_path, label=f"reviewer packet {row['relativePath']}"
        )
        packet = _validate_packet(
            packet_value,
            evidence=row,
            file_sha256=packet_file_sha,
            size_bytes=packet_size,
            candidate_count=candidate_count,
        )
        case_alias = str(packet["caseAlias"])
        if case_alias in packets:
            raise SemanticModelUnblindError("reviewer packet aliases must be unique")
        packets[case_alias] = packet
    packet_set_binding = [
        {
            "caseAlias": case_alias,
            "fileSha256": str(packets[case_alias]["fileSha256"]),
            "canonicalSha256": str(packets[case_alias]["canonicalSha256"]),
        }
        for case_alias in sorted(packets)
    ]
    reviewer_packet_set_sha = canonical_json_sha256(packet_set_binding)
    if reviewer_packet_set_sha != _sha256(
        review_package.get("reviewerPacketSetSha256"),
        field="sealed review reviewer packet set SHA-256",
    ):
        raise SemanticModelUnblindError(
            "reviewer packet set SHA-256 does not match sealed review"
        )
    reviews = _validate_review_cases(
        review,
        packets=packets,
        candidate_count=candidate_count,
    )

    # This is the first point at which identity-vault contents may be opened.
    vault_path = _package_member(
        root,
        vault_row["relativePath"],
        field="identity vault path",
    )
    _vault_resolved, vault_value, vault_file_sha, vault_size = _load_json_snapshot(
        vault_path, label="identity vault"
    )
    case_set_sha, model_set_sha, vault_models, aliases = _validate_vault(
        vault_value,
        evidence=vault_row,
        file_sha256=vault_file_sha,
        size_bytes=vault_size,
        packets=packets,
        candidate_count=candidate_count,
    )
    vault_canonical_sha = _sha256(
        vault_value["canonicalSha256"], field="identity vault canonical SHA-256"
    )
    aggregate, winner, tied = _aggregate_results(
        reviews=reviews,
        aliases=aliases,
        models=vault_models,
    )
    models = [
        {
            "registryModelId": str(model["modelId"]),
            "model": str(model["model"]),
            "digest": str(model["digest"]),
        }
        for model in sorted(vault_models, key=lambda row: str(row["modelId"]))
    ]
    review_id = str(review["reviewId"])
    blind_batch_id = f"semantic-blind-{manifest_file_sha[:24]}"
    value: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "comparisonId": f"{review_id}.unblind-comparison",
        "reviewId": review_id,
        "automaticScoringUsed": False,
        "sealedBeforeUnblind": True,
        "candidateIdentitiesHiddenDuringReview": True,
        "blindBatchId": blind_batch_id,
        "caseSetSha256": case_set_sha,
        "modelSetSha256": model_set_sha,
        "caseCount": EXPECTED_PACKET_COUNT,
        "candidateCount": candidate_count,
        "models": models,
        "rankingPolicy": {
            "order": [
                "blocker-count-ascending",
                "major-count-ascending",
                "minor-count-ascending",
                "pass-count-descending",
                "weighted-preferred-share-descending",
            ],
            "tiePreferenceWeight": "one case credit divided equally among preferred candidates",
            "automaticMetricsUsed": False,
        },
        "aggregate": aggregate,
        "outcome": "unique-winner" if winner is not None else "tie",
        "uniqueWinnerRegistryModelId": winner,
        "tiedRegistryModelIds": tied,
        "evidence": {
            "sealedReviewPath": str(review_path),
            "sealedReviewFileSha256": review_file_sha,
            "sealedReviewCanonicalSha256": review_canonical_sha,
            "manifestFileSha256": manifest_file_sha,
            "manifestCanonicalSha256": manifest_canonical_sha,
            "identityVaultFileSha256": vault_file_sha,
            "identityVaultCanonicalSha256": vault_canonical_sha,
            "reviewerPacketSetSha256": reviewer_packet_set_sha,
        },
    }
    value["canonicalSha256"] = canonical_json_sha256(value)
    atomic_write_json_no_replace(output, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--sealed-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        value = unblind_semantic_model_review(
            package_root=arguments.package_root,
            sealed_review_path=arguments.sealed_review,
            output_path=arguments.output,
        )
    except (OSError, ValueError) as exc:
        print(f"semantic model review unblind failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
