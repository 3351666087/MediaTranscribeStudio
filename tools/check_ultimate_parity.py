#!/usr/bin/env python3
"""Fail-closed Ultimate parity and cutover audit for MediaTranscribeStudio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat as stat_module
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
ALLOWED_GATE_STATES = {"not_started", "in_progress", "blocked", "passed"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
GIT_MODE_RE = re.compile(r"^[0-7]{6}$")
RFC3339_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
PROTECTED_BLOB_MODES = {"100644", "100755"}
CONTENT_TRANSFORMING_GIT_ATTRIBUTES = (
    "filter",
    "working-tree-encoding",
    "ident",
)
FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
CJK_RE = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff"
    r"\u31c0-\u31ef\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\uff00-\uffef]"
)
DEFAULT_EXCLUDED_PARTS = {
    ".git",
    ".codex",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "target",
}
REAL_MEDIA_ARTIFACT_CONTRACTS: dict[str, dict[str, str]] = {
    "transcript": {
        "extension": ".json",
        "mediaType": "application/json",
        "format": "transcript-document-v2",
    },
    "report-document": {
        "extension": ".json",
        "mediaType": "application/json",
        "format": "report-document-v1",
    },
    "pdf": {
        "extension": ".pdf",
        "mediaType": "application/pdf",
        "format": "pdf",
    },
    "pdf-inspection": {
        "extension": ".json",
        "mediaType": "application/json",
        "format": "pdfbox-inspection-v1",
    },
    "quality-report": {
        "extension": ".json",
        "mediaType": "application/json",
        "format": "pdf-quality-report-v1",
    },
    "artifact-manifest": {
        "extension": ".json",
        "mediaType": "application/json",
        "format": "pdf-artifact-manifest-v1",
    },
    "contact-sheet": {
        "extension": ".png",
        "mediaType": "image/png",
        "format": "png",
    },
}
REAL_MEDIA_METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "speakerCount": (
        "expectedCount",
        "detectedCount",
        "resolvedCount",
        "absoluteError",
        "reviewRequired",
    ),
    "diarization": (
        "segmentCount",
        "auditedSegmentCount",
        "speakerAssignmentErrors",
        "unresolvedSpeakerSegments",
        "humanAuditCompleted",
    ),
    "boundary": (
        "segmentCount",
        "invalidIntervals",
        "nonMonotonicIntervals",
        "outOfBoundsIntervals",
    ),
    "asr": (
        "segmentCount",
        "auditedSegmentCount",
        "emptySegments",
        "unresolvedTextSegments",
        "sourceLanguagePreserved",
    ),
    "semantic": (
        "reviewedRevisionCount",
        "unresolvedRevisionCount",
        "rawTranscriptImmutable",
        "speakerLocksPreserved",
        "humanApproved",
    ),
    "efficiency": (
        "mediaDurationSeconds",
        "wallClockSeconds",
        "realTimeFactor",
        "peakRamMb",
        "peakVramMb",
    ),
    "pdf": (
        "pageCount",
        "qualityScore",
        "hardGateFailureCount",
        "facetFailureCount",
        "pdfBoxValidated",
    ),
}


class ManifestError(ValueError):
    """Raised when the audit manifest is malformed or unsafe."""


def repository_root() -> Path:
    """Return the repository root based on this checker, not the caller's CWD."""

    return Path(__file__).resolve().parents[1]


def load_json_with_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes.decode("utf-8", errors="strict"))
    except FileNotFoundError as exc:
        raise ManifestError(f"JSON file does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read valid UTF-8 JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"JSON root must be an object: {path}")
    return value, raw_bytes


def load_json(path: Path) -> dict[str, Any]:
    value, _ = load_json_with_bytes(path)
    return value


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        return False
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    return (
        not path.is_absolute()
        and ".." not in parts
        and (not parts or ":" not in parts[0])
    )


def _resolve_under(root: Path, relative: str) -> Path:
    if not _is_safe_relative_path(relative):
        raise ManifestError(f"Unsafe relative path: {relative!r}")
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(relative)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ManifestError(f"Path escapes its configured root: {relative!r}")
    return candidate


def _require_string(
    errors: list[str], owner: Mapping[str, Any], key: str, context: str
) -> None:
    value = owner.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context}.{key} must be a non-empty string")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not RFC3339_TIMESTAMP_RE.fullmatch(value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _normalize_evaluation_time(value: datetime) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _validate_authorization(
    errors: list[str], value: Any, context: str, expected_scope: str
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return
    if not isinstance(value.get("approved"), bool):
        errors.append(f"{context}.approved must be a boolean")
    if value.get("scope") != expected_scope:
        errors.append(f"{context}.scope must be {expected_scope!r}")
    for key in ("approvedBy", "approvedAt", "changeTicket"):
        item = value.get(key)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            errors.append(f"{context}.{key} must be null or a non-empty string")
    approved_at = value.get("approvedAt")
    if approved_at is not None and _parse_timestamp(approved_at) is None:
        errors.append(
            f"{context}.approvedAt must be an RFC3339 timestamp with an "
            "explicit timezone"
        )


def _validate_protected_legacy_baseline(
    errors: list[str],
    protected_paths: Any,
    value: Any,
) -> None:
    context = "policy.protectedLegacyBaseline"
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return
    source_commit = value.get("sourceCommit")
    if not isinstance(source_commit, str) or not GIT_SHA_RE.fullmatch(source_commit):
        errors.append(f"{context}.sourceCommit must have a 40-char SHA")
    entries = value.get("entries")
    if not isinstance(entries, dict) or not entries:
        errors.append(f"{context}.entries must be a non-empty object")
        return
    if isinstance(protected_paths, list) and set(entries) != set(protected_paths):
        errors.append(
            f"{context}.entries must exactly cover policy.protectedLegacyPaths"
        )
    for relative, entry in entries.items():
        entry_context = f"{context}.entries[{relative!r}]"
        if not _is_safe_relative_path(relative):
            errors.append(f"{entry_context} uses an unsafe path")
        if not isinstance(entry, dict):
            errors.append(f"{entry_context} must be an object")
            continue
        git_type = entry.get("gitType")
        git_mode = entry.get("gitMode")
        object_id = entry.get("gitObjectId")
        if git_type not in {"blob", "tree"}:
            errors.append(f"{entry_context}.gitType must be 'blob' or 'tree'")
        if not isinstance(git_mode, str) or not GIT_MODE_RE.fullmatch(git_mode):
            errors.append(f"{entry_context}.gitMode must be a six-digit Git mode")
        elif git_type == "tree" and git_mode != "040000":
            errors.append(f"{entry_context}.gitMode must be '040000' for a tree")
        elif git_type == "blob" and git_mode not in PROTECTED_BLOB_MODES:
            errors.append(
                f"{entry_context}.gitMode must be a regular-file mode, never a symlink"
            )
        if not isinstance(object_id, str) or not GIT_SHA_RE.fullmatch(object_id):
            errors.append(f"{entry_context}.gitObjectId must have a 40-char SHA")


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return structural and policy errors without consulting runtime evidence."""

    errors: list[str] = []
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        errors.append(f"schemaVersion must be {SCHEMA_VERSION!r}")
    for key in ("auditId", "title", "updatedAt"):
        _require_string(errors, manifest, key, "manifest")

    serialized = json.dumps(manifest, ensure_ascii=False)
    if CJK_RE.search(serialized):
        errors.append("manifest must be English-only and contain no CJK characters")

    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        return errors + ["policy must be an object"]
    if policy.get("defaultFailClosed") is not True:
        errors.append("policy.defaultFailClosed must be true")
    if policy.get("forbidEvidenceInsideRepository") is not True:
        errors.append("policy.forbidEvidenceInsideRepository must be true")
    max_age = policy.get("maxEvidenceAgeHours")
    if not isinstance(max_age, int) or isinstance(max_age, bool) or max_age <= 0:
        errors.append("policy.maxEvidenceAgeHours must be a positive integer")
    _require_string(errors, policy, "evidenceRootEnv", "policy")

    required_capabilities = policy.get("requiredCapabilityIds")
    required_gates = policy.get("requiredGateIds")
    parity_gates = policy.get("parityGateIds")
    release_gates = policy.get("releaseGateIds")
    protected_paths = policy.get("protectedLegacyPaths")
    for key, value in (
        ("requiredCapabilityIds", required_capabilities),
        ("requiredGateIds", required_gates),
        ("parityGateIds", parity_gates),
        ("releaseGateIds", release_gates),
        ("protectedLegacyPaths", protected_paths),
    ):
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))
        ):
            errors.append(f"policy.{key} must be a non-empty unique string array")

    if isinstance(protected_paths, list):
        for value in protected_paths:
            if not _is_safe_relative_path(value):
                errors.append(f"policy.protectedLegacyPaths contains unsafe path {value!r}")
    _validate_protected_legacy_baseline(
        errors,
        protected_paths,
        policy.get("protectedLegacyBaseline"),
    )

    protected_refs = policy.get("protectedMainRefs")
    if not isinstance(protected_refs, dict) or not protected_refs:
        errors.append("policy.protectedMainRefs must be a non-empty object")
    else:
        for ref_name, expected_sha in protected_refs.items():
            if not isinstance(ref_name, str) or not ref_name.startswith("refs/"):
                errors.append(f"invalid protected Git ref name: {ref_name!r}")
            if not isinstance(expected_sha, str) or not GIT_SHA_RE.fullmatch(expected_sha):
                errors.append(f"protected Git ref {ref_name!r} must have a 40-char SHA")

    authorization = policy.get("authorization")
    if not isinstance(authorization, dict):
        errors.append("policy.authorization must be an object")
    else:
        legacy_authorization = authorization.get("legacyRemoval")
        main_authorization = authorization.get("mainReplacement")
        _validate_authorization(
            errors,
            legacy_authorization,
            "policy.authorization.legacyRemoval",
            "legacy-removal",
        )
        _validate_authorization(
            errors,
            main_authorization,
            "policy.authorization.mainReplacement",
            "main-replacement",
        )
        if isinstance(legacy_authorization, dict) and isinstance(
            main_authorization, dict
        ):
            legacy_approved_at = _parse_timestamp(
                legacy_authorization.get("approvedAt")
            )
            main_approved_at = _parse_timestamp(
                main_authorization.get("approvedAt")
            )
            if (
                legacy_approved_at is not None
                and main_approved_at is not None
                and main_approved_at < legacy_approved_at
            ):
                errors.append(
                    "policy.authorization.mainReplacement.approvedAt must be "
                    "greater than or equal to "
                    "policy.authorization.legacyRemoval.approvedAt"
                )

    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        errors.append("capabilities must be a non-empty array")
        capabilities = []
    capability_ids: list[str] = []
    capability_gate_map: dict[str, list[str]] = {}
    for index, capability in enumerate(capabilities):
        context = f"capabilities[{index}]"
        if not isinstance(capability, dict):
            errors.append(f"{context} must be an object")
            continue
        _require_string(errors, capability, "id", context)
        _require_string(errors, capability, "name", context)
        capability_id = capability.get("id")
        if isinstance(capability_id, str):
            capability_ids.append(capability_id)
        if capability.get("required") is not True:
            errors.append(f"{context}.required must be true")
        gate_ids = capability.get("gateIds")
        if (
            not isinstance(gate_ids, list)
            or not gate_ids
            or any(not isinstance(item, str) or not item for item in gate_ids)
        ):
            errors.append(f"{context}.gateIds must be a non-empty string array")
        elif isinstance(capability_id, str):
            capability_gate_map[capability_id] = gate_ids
    if len(capability_ids) != len(set(capability_ids)):
        errors.append("capability IDs must be unique")
    if isinstance(required_capabilities, list) and set(capability_ids) != set(
        required_capabilities
    ):
        errors.append("capabilities must exactly cover policy.requiredCapabilityIds")

    gates = manifest.get("gates")
    if not isinstance(gates, list) or not gates:
        errors.append("gates must be a non-empty array")
        gates = []
    gate_ids: list[str] = []
    command_ids: set[str] = set()
    for index, gate in enumerate(gates):
        context = f"gates[{index}]"
        if not isinstance(gate, dict):
            errors.append(f"{context} must be an object")
            continue
        for key in ("id", "title", "description"):
            _require_string(errors, gate, key, context)
        gate_id = gate.get("id")
        if isinstance(gate_id, str):
            gate_ids.append(gate_id)
        if gate.get("required") is not True:
            errors.append(f"{context}.required must be true")
        if gate.get("state") not in ALLOWED_GATE_STATES:
            errors.append(
                f"{context}.state must be one of {sorted(ALLOWED_GATE_STATES)}"
            )
        refs = gate.get("capabilityIds")
        if (
            not isinstance(refs, list)
            or not refs
            or any(item not in capability_ids for item in refs)
        ):
            errors.append(f"{context}.capabilityIds must reference known capabilities")
        repo_checks = gate.get("repoChecks", [])
        if not isinstance(repo_checks, list):
            errors.append(f"{context}.repoChecks must be an array")
        else:
            for check_index, check in enumerate(repo_checks):
                _validate_repo_check(errors, check, f"{context}.repoChecks[{check_index}]")
        command_checks = gate.get("commandChecks", [])
        if not isinstance(command_checks, list):
            errors.append(f"{context}.commandChecks must be an array")
        else:
            for check_index, check in enumerate(command_checks):
                check_context = f"{context}.commandChecks[{check_index}]"
                if not isinstance(check, dict):
                    errors.append(f"{check_context} must be an object")
                    continue
                for key in ("id", "attestationPath"):
                    _require_string(errors, check, key, check_context)
                check_id = check.get("id")
                if isinstance(check_id, str):
                    if check_id in command_ids:
                        errors.append(f"duplicate command check ID: {check_id}")
                    command_ids.add(check_id)
                argv = check.get("argv")
                if (
                    not isinstance(argv, list)
                    or not argv
                    or any(not isinstance(item, str) or not item for item in argv)
                ):
                    errors.append(f"{check_context}.argv must be a non-empty string array")
                if not _is_safe_relative_path(check.get("cwd", ".")):
                    errors.append(f"{check_context}.cwd must stay inside the repository")
                if not _is_safe_relative_path(check.get("attestationPath")):
                    errors.append(f"{check_context}.attestationPath must be relative and safe")
                timeout = check.get("timeoutSeconds")
                if (
                    not isinstance(timeout, int)
                    or isinstance(timeout, bool)
                    or timeout <= 0
                ):
                    errors.append(
                        f"{check_context}.timeoutSeconds must be a positive integer"
                    )
        evidence = gate.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{context}.evidence must be an object")
        else:
            attestations = evidence.get("attestations")
            if not isinstance(attestations, list) or not attestations:
                errors.append(f"{context}.evidence.attestations must be non-empty")
            else:
                for evidence_index, spec in enumerate(attestations):
                    evidence_context = (
                        f"{context}.evidence.attestations[{evidence_index}]"
                    )
                    if not isinstance(spec, dict):
                        errors.append(f"{evidence_context} must be an object")
                        continue
                    for key in ("path", "kind"):
                        _require_string(errors, spec, key, evidence_context)
                    if not _is_safe_relative_path(spec.get("path")):
                        errors.append(f"{evidence_context}.path must be relative and safe")
                    profile = spec.get("profile")
                    if profile is not None and profile not in {
                        "realMediaAuto",
                        "realMediaManualFive",
                    }:
                        errors.append(f"{evidence_context}.profile is unsupported")
                    minimum = spec.get("minimumArtifacts", 0)
                    if (
                        not isinstance(minimum, int)
                        or isinstance(minimum, bool)
                        or minimum < 0
                    ):
                        errors.append(
                            f"{evidence_context}.minimumArtifacts must be non-negative"
                        )

    if len(gate_ids) != len(set(gate_ids)):
        errors.append("gate IDs must be unique")
    if isinstance(required_gates, list) and set(gate_ids) != set(required_gates):
        errors.append("gates must exactly cover policy.requiredGateIds")
    if isinstance(parity_gates, list) and not set(parity_gates).issubset(set(gate_ids)):
        errors.append("policy.parityGateIds contains an unknown gate")
    if isinstance(release_gates, list) and set(release_gates) != set(gate_ids):
        errors.append("policy.releaseGateIds must contain every required gate")
    for capability_id, mapped_gates in capability_gate_map.items():
        if not set(mapped_gates).issubset(set(gate_ids)):
            errors.append(f"capability {capability_id!r} maps to an unknown gate")
        for gate_id in mapped_gates:
            matching = next(
                (
                    gate
                    for gate in gates
                    if isinstance(gate, dict) and gate.get("id") == gate_id
                ),
                None,
            )
            if not matching or capability_id not in matching.get("capabilityIds", []):
                errors.append(
                    f"capability {capability_id!r} and gate {gate_id!r} "
                    "must reference each other"
                )

    _validate_real_media_policy(errors, manifest.get("realMedia"))
    _validate_pdf_policy(errors, manifest.get("javaPdf"))
    _validate_dynamic_n_policy(errors, manifest.get("dynamicN"))
    _validate_globalization_policy(errors, manifest.get("globalization"))
    return errors


def _validate_repo_check(errors: list[str], check: Any, context: str) -> None:
    if not isinstance(check, dict):
        errors.append(f"{context} must be an object")
        return
    _require_string(errors, check, "id", context)
    check_type = check.get("type")
    supported = {
        "path_exists",
        "paths_absent",
        "glob_min",
        "file_contains_all",
        "file_contains_none",
        "text_patterns_absent",
        "json_file_valid",
        "readmes_english",
    }
    if check_type not in supported:
        errors.append(f"{context}.type must be one of {sorted(supported)}")
        return
    path_keys = {
        "path_exists": ("path",),
        "glob_min": ("pattern",),
        "file_contains_all": ("path",),
        "file_contains_none": ("path",),
        "json_file_valid": ("path",),
    }
    for key in path_keys.get(check_type, ()):
        if not _is_safe_relative_path(check.get(key)):
            errors.append(f"{context}.{key} must be a safe repository-relative path")
    if check_type == "paths_absent":
        values = check.get("paths")
        if not isinstance(values, list) or not values:
            errors.append(f"{context}.paths must be non-empty")
        elif any(not _is_safe_relative_path(item) for item in values):
            errors.append(f"{context}.paths contains an unsafe path")
    if check_type in {"file_contains_all", "file_contains_none"}:
        values = check.get("strings")
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item for item in values)
        ):
            errors.append(f"{context}.strings must be a non-empty string array")
    if check_type == "glob_min":
        minimum = check.get("minimum")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
            errors.append(f"{context}.minimum must be a positive integer")
    if check_type == "text_patterns_absent":
        roots = check.get("roots")
        patterns = check.get("patterns")
        if (
            not isinstance(roots, list)
            or not roots
            or any(not _is_safe_relative_path(item) for item in roots)
        ):
            errors.append(f"{context}.roots must contain safe relative paths")
        if (
            not isinstance(patterns, list)
            or not patterns
            or any(not isinstance(item, str) or not item for item in patterns)
        ):
            errors.append(f"{context}.patterns must be a non-empty regex array")
        else:
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(f"{context}.patterns has invalid regex {pattern!r}: {exc}")


def _validate_real_media_policy(errors: list[str], value: Any) -> None:
    if not isinstance(value, dict):
        errors.append("realMedia must be an object")
        return
    _require_string(errors, value, "sourceFileName", "realMedia")
    source_name = value.get("sourceFileName")
    if isinstance(source_name, str) and Path(source_name).name != source_name:
        errors.append("realMedia.sourceFileName must be a basename, never an absolute path")
    if value.get("manualRegressionSpeakerCount") != 5:
        errors.append("realMedia.manualRegressionSpeakerCount must be 5")
    roles = value.get("requiredArtifactRoles")
    if (
        not isinstance(roles, list)
        or not roles
        or any(not isinstance(item, str) or not item for item in roles)
        or len(roles) != len(set(roles))
    ):
        errors.append("realMedia.requiredArtifactRoles must be a unique string array")
    elif set(roles) != set(REAL_MEDIA_ARTIFACT_CONTRACTS):
        errors.append(
            "realMedia.requiredArtifactRoles must exactly match the canonical "
            "typed real-media artifact roles"
        )
    role_contracts = value.get("artifactRoleContracts")
    if role_contracts != REAL_MEDIA_ARTIFACT_CONTRACTS:
        errors.append(
            "realMedia.artifactRoleContracts must exactly match the canonical "
            "extension, media type, and format contracts"
        )
    domains = value.get("requiredMetricDomains")
    expected_domains = {
        "speakerCount",
        "diarization",
        "boundary",
        "asr",
        "semantic",
        "efficiency",
        "pdf",
    }
    if not isinstance(domains, list) or set(domains) != expected_domains:
        errors.append(
            "realMedia.requiredMetricDomains must cover all seven independent domains"
        )
    metric_fields = value.get("requiredMetricFields")
    expected_metric_fields = {
        key: list(fields) for key, fields in REAL_MEDIA_METRIC_FIELDS.items()
    }
    if metric_fields != expected_metric_fields:
        errors.append(
            "realMedia.requiredMetricFields must exactly match the canonical raw "
            "metric evidence contract"
        )
    maximum_rtf = value.get("maximumRealTimeFactor")
    if (
        not isinstance(maximum_rtf, (int, float))
        or isinstance(maximum_rtf, bool)
        or not (0 < float(maximum_rtf) <= 10)
    ):
        errors.append("realMedia.maximumRealTimeFactor must be in (0, 10]")


def _validate_pdf_policy(errors: list[str], value: Any) -> None:
    if not isinstance(value, dict):
        errors.append("javaPdf must be an object")
        return
    if value.get("renderer") != "OpenHTMLtoPDF":
        errors.append("javaPdf.renderer must be 'OpenHTMLtoPDF'")
    if value.get("rendererVersion") != "1.0.10":
        errors.append("javaPdf.rendererVersion must be '1.0.10'")
    if value.get("validator") != "PDFBox":
        errors.append("javaPdf.validator must be 'PDFBox'")
    if value.get("validatorVersion") != "2.0.30":
        errors.append("javaPdf.validatorVersion must be '2.0.30'")
    hard_gates = value.get("hardGateIds")
    facets = value.get("designPackFacetIds")
    if (
        not isinstance(hard_gates, list)
        or len(hard_gates) != 13
        or len(hard_gates) != len(set(hard_gates))
    ):
        errors.append("javaPdf.hardGateIds must contain 13 unique IDs")
    if (
        not isinstance(facets, list)
        or len(facets) != 14
        or len(facets) != len(set(facets))
    ):
        errors.append("javaPdf.designPackFacetIds must contain 14 unique IDs")
    score = value.get("minimumQualityScore")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or score < 85:
        errors.append("javaPdf.minimumQualityScore must be at least 85")


def _validate_dynamic_n_policy(errors: list[str], value: Any) -> None:
    if not isinstance(value, dict):
        errors.append("dynamicN must be an object")
        return
    if value.get("modes") != ["auto", "manual", "hybrid"]:
        errors.append("dynamicN.modes must be ['auto', 'manual', 'hybrid']")
    if value.get("regressionMatrix") != [1, 2, 5, 8, 13]:
        errors.append("dynamicN.regressionMatrix must be [1, 2, 5, 8, 13]")
    if value.get("fixedProductMaximum") is not None:
        errors.append("dynamicN.fixedProductMaximum must be null")


def _validate_globalization_policy(errors: list[str], value: Any) -> None:
    if not isinstance(value, dict):
        errors.append("globalization must be an object")
        return
    if value.get("persistedAutoAllowed") is not False:
        errors.append("globalization.persistedAutoAllowed must be false")
    if value.get("unknownLanguageCode") != "und":
        errors.append("globalization.unknownLanguageCode must be 'und'")
    if value.get("multilingualCode") != "mul":
        errors.append("globalization.multilingualCode must be 'mul'")
    if value.get("publicReadmesEnglishOnly") is not True:
        errors.append("globalization.publicReadmesEnglishOnly must be true")


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise ManifestError(f"Git command failed: git {' '.join(args)}: {message}")
    return completed.stdout.strip()


def _git_index_flagged_paths(repo_root: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "ls-files", "-v", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        raise ManifestError(
            "Git command failed while checking index flags: "
            f"{stderr or stdout}"
        )
    flagged: list[dict[str, str]] = []
    for record in completed.stdout.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise ManifestError("Cannot parse git ls-files -v output")
        tag = record[:1].decode("ascii", errors="strict")
        path = record[2:].decode("utf-8", errors="surrogateescape")
        if tag.islower():
            flagged.append({"flag": "assume-unchanged", "path": path})
        elif tag == "S":
            flagged.append({"flag": "skip-worktree", "path": path})
    return flagged


def _git_state(
    repo_root: Path,
    protected_refs: Iterable[str],
    manifest_path: Path,
    manifest_decision_bytes: bytes,
) -> dict[str, Any]:
    head = _run_git(repo_root, "rev-parse", "HEAD")
    branch = _run_git(repo_root, "branch", "--show-current") or "(detached)"
    status = _run_git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    index_flagged_paths = _git_index_flagged_paths(repo_root)
    manifest_integrity = _git_bound_file_integrity(
        repo_root=repo_root,
        head_commit=head,
        path=manifest_path,
        label="release manifest",
        decision_bytes=manifest_decision_bytes,
    )
    refs: dict[str, str | None] = {}
    for ref_name in protected_refs:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", ref_name],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        refs[ref_name] = (
            completed.stdout.strip() if completed.returncode == 0 else None
        )
    synthetic_changes = [
        f"[index-{item['flag']}] {item['path']}" for item in index_flagged_paths
    ]
    synthetic_changes.extend(
        f"[manifest-integrity] {message}"
        for message in manifest_integrity["violations"]
    )
    return {
        "headCommit": head,
        "branch": branch,
        "workingTreeClean": (
            not bool(status)
            and not index_flagged_paths
            and manifest_integrity["passed"]
        ),
        "workingTreeChanges": status.splitlines() + synthetic_changes,
        "indexFlagsClean": not index_flagged_paths,
        "indexFlaggedPaths": index_flagged_paths,
        "manifestIntegrity": manifest_integrity,
        "refs": refs,
    }


def _git_ref(repo_root: Path, ref_name: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", ref_name],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    if completed.returncode == 128:
        return None
    message = completed.stderr.strip() or completed.stdout.strip()
    raise ManifestError(f"Git command failed: git rev-parse --verify {ref_name}: {message}")


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    message = completed.stderr.strip() or completed.stdout.strip()
    raise ManifestError(
        "Git command failed: "
        f"git merge-base --is-ancestor {ancestor} {descendant}: {message}"
    )


def _git_tree_entries(
    repo_root: Path,
    commit: str,
    relative: str,
    *,
    recursive: bool,
) -> dict[str, dict[str, str]]:
    normalized = relative.replace("\\", "/")
    argv = ["git", "ls-tree", "-z", "--full-tree"]
    if recursive:
        argv.extend(["-r", "-t"])
    argv.extend([commit, "--", normalized])
    completed = subprocess.run(
        argv,
        cwd=repo_root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        raise ManifestError(
            f"Git command failed: {' '.join(argv)}: {stderr or stdout}"
        )
    entries: dict[str, dict[str, str]] = {}
    for raw_record in completed.stdout.split(b"\0"):
        if not raw_record:
            continue
        try:
            raw_metadata, raw_path = raw_record.split(b"\t", 1)
            git_mode, git_type, object_id = raw_metadata.decode("ascii").split()
            entry_path = raw_path.decode("utf-8", errors="surrogateescape")
        except (ValueError, UnicodeError) as exc:
            raise ManifestError(
                f"Cannot parse git ls-tree output for {normalized!r}"
            ) from exc
        entries[entry_path] = {
            "gitMode": git_mode,
            "gitType": git_type,
            "gitObjectId": object_id,
        }
    return entries


def _git_tree_entry(
    repo_root: Path,
    commit: str,
    relative: str,
) -> dict[str, str] | None:
    normalized = relative.replace("\\", "/")
    return _git_tree_entries(
        repo_root,
        commit,
        normalized,
        recursive=False,
    ).get(normalized)


def _git_blob_bytes(repo_root: Path, object_id: str) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", object_id],
        cwd=repo_root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        raise ManifestError(
            f"Git command failed while reading baseline blob {object_id}: "
            f"{stderr or stdout}"
        )
    return completed.stdout


def _git_path_attributes(repo_root: Path, relative: str) -> dict[str, str]:
    normalized = relative.replace("\\", "/")
    argv = [
        "git",
        "check-attr",
        "-z",
        "--all",
        "--",
        normalized,
    ]
    completed = subprocess.run(
        argv,
        cwd=repo_root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        raise ManifestError(
            f"Git command failed while checking attributes for {normalized!r}: "
            f"{stderr or stdout}"
        )
    fields = completed.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3 != 0:
        raise ManifestError(
            f"Cannot parse git check-attr output for protected path {normalized!r}"
        )
    attributes: dict[str, str] = {}
    for index in range(0, len(fields), 3):
        try:
            resolved_path = fields[index].decode(
                "utf-8", errors="surrogateescape"
            )
            attribute = fields[index + 1].decode("ascii")
            value = fields[index + 2].decode("utf-8", errors="surrogateescape")
        except UnicodeError as exc:
            raise ManifestError(
                f"Cannot decode git attributes for protected path {normalized!r}"
            ) from exc
        if resolved_path != normalized:
            raise ManifestError(
                "Git returned attributes for an unexpected protected path: "
                f"expected {normalized!r}, got {resolved_path!r}"
            )
        if attribute in attributes:
            raise ManifestError(
                f"Git returned duplicate {attribute!r} attributes for "
                f"{normalized!r}"
            )
        if attribute in CONTENT_TRANSFORMING_GIT_ATTRIBUTES:
            attributes[attribute] = value
    return attributes


def _content_transform_attribute_violations(
    repo_root: Path,
    relative: str,
) -> list[str]:
    normalized = relative.replace("\\", "/")
    attributes = _git_path_attributes(repo_root, normalized)
    return [
        "protected legacy path has a content-transforming Git attribute before "
        f"authorization: {normalized} ({attribute}={value})"
        for attribute, value in attributes.items()
    ]


def _is_utf8_text_without_nul(value: bytes) -> bool:
    if b"\0" in value:
        return False
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _guard_content_equivalent(expected: bytes, actual: bytes) -> bool:
    if actual == expected:
        return True
    if not (
        _is_utf8_text_without_nul(expected)
        and _is_utf8_text_without_nul(actual)
    ):
        return False
    return expected.replace(b"\r\n", b"\n") == actual.replace(b"\r\n", b"\n")


def _git_bound_file_integrity(
    *,
    repo_root: Path,
    head_commit: str,
    path: Path,
    label: str,
    decision_bytes: bytes | None = None,
) -> dict[str, Any]:
    resolved_root = repo_root.resolve()
    requested_path = path if path.is_absolute() else resolved_root / path
    absolute_path = Path(os.path.abspath(requested_path))
    violations: list[str] = []
    try:
        relative_path = absolute_path.relative_to(resolved_root)
    except ValueError:
        return {
            "passed": False,
            "path": str(absolute_path),
            "gitObjectId": None,
            "decisionSha256": (
                hashlib.sha256(decision_bytes).hexdigest()
                if decision_bytes is not None
                else None
            ),
            "violations": [f"{label} must be a tracked file inside the repository"],
        }
    relative = relative_path.as_posix()
    kind = _filesystem_entry_kind(absolute_path)
    if kind != "blob":
        violations.append(
            f"{label} must be a regular non-reparse file: {relative} (got {kind})"
        )
    try:
        entry = _git_tree_entry(repo_root, head_commit, relative)
    except ManifestError as exc:
        entry = None
        violations.append(str(exc))
    if entry is None:
        violations.append(f"{label} is not tracked at current HEAD: {relative}")
    elif entry["gitType"] != "blob":
        violations.append(
            f"{label} is not a Git blob at current HEAD: {relative}"
        )
    if kind == "blob":
        try:
            attributes = _git_path_attributes(repo_root, relative)
            violations.extend(
                f"{label} has a content-transforming Git attribute: "
                f"{relative} ({attribute}={value})"
                for attribute, value in attributes.items()
            )
            if entry is not None and entry["gitType"] == "blob":
                expected_bytes = _git_blob_bytes(
                    repo_root,
                    entry["gitObjectId"],
                )
                actual_bytes = absolute_path.read_bytes()
                if decision_bytes is not None and not _guard_content_equivalent(
                    expected_bytes,
                    decision_bytes,
                ):
                    violations.append(
                        f"{label} decision bytes differ from current HEAD: {relative} "
                        f"(expected Git blob {entry['gitObjectId']}, decision "
                        f"SHA-256 {hashlib.sha256(decision_bytes).hexdigest()})"
                    )
                if not _guard_content_equivalent(expected_bytes, actual_bytes):
                    violations.append(
                        f"{label} raw bytes differ from current HEAD: {relative} "
                        f"(expected Git blob {entry['gitObjectId']}, raw worktree "
                        f"SHA-256 {hashlib.sha256(actual_bytes).hexdigest()})"
                    )
        except (ManifestError, OSError) as exc:
            violations.append(f"Cannot verify {label} {relative}: {exc}")
    return {
        "passed": not violations,
        "path": relative,
        "gitObjectId": entry["gitObjectId"] if entry is not None else None,
        "decisionSha256": (
            hashlib.sha256(decision_bytes).hexdigest()
            if decision_bytes is not None
            else None
        ),
        "violations": violations,
    }


def _filesystem_entry_kind(path: Path) -> str:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        raise ManifestError(f"Cannot inspect protected path {path}: {exc}") from exc
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    junction_check = getattr(path, "is_junction", None)
    try:
        is_junction = bool(junction_check()) if junction_check is not None else False
    except OSError as exc:
        raise ManifestError(f"Cannot inspect protected path {path}: {exc}") from exc
    if (
        stat_module.S_ISLNK(metadata.st_mode)
        or bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)
        or is_junction
    ):
        return "reparse"
    if stat_module.S_ISREG(metadata.st_mode):
        return "blob"
    if stat_module.S_ISDIR(metadata.st_mode):
        return "tree"
    return "unsupported"


def _filesystem_tree_entries(
    repo_root: Path,
    relative: str,
) -> dict[str, str]:
    normalized = relative.replace("\\", "/")
    root_path = repo_root / Path(normalized)
    entries: dict[str, str] = {}
    pending: list[tuple[str, Path]] = [(normalized, root_path)]
    while pending:
        entry_relative, entry_path = pending.pop()
        kind = _filesystem_entry_kind(entry_path)
        entries[entry_relative] = kind
        if kind != "tree":
            continue
        try:
            children = sorted(
                os.scandir(entry_path),
                key=lambda item: item.name.encode("utf-8", errors="surrogateescape"),
                reverse=True,
            )
        except OSError as exc:
            raise ManifestError(
                f"Cannot enumerate protected directory {entry_path}: {exc}"
            ) from exc
        for child in children:
            child_relative = f"{entry_relative}/{child.name}"
            pending.append((child_relative, Path(child.path)))
    return entries


def _describe_git_entry(entry: Mapping[str, str] | None) -> str:
    if entry is None:
        return "missing"
    return (
        f"{entry.get('gitMode', 'unknown')} "
        f"{entry.get('gitType', 'unknown')} "
        f"{entry.get('gitObjectId', 'unknown')}"
    )


def _protected_worktree_violations(
    *,
    repo_root: Path,
    source_commit: str,
    relative: str,
) -> list[str]:
    normalized = relative.replace("\\", "/")
    violations: list[str] = []
    try:
        expected_entries = _git_tree_entries(
            repo_root,
            source_commit,
            normalized,
            recursive=True,
        )
        actual_entries = _filesystem_tree_entries(repo_root, normalized)
    except ManifestError as exc:
        return [str(exc)]
    root_kind = actual_entries.get(normalized, "missing")
    if root_kind == "missing":
        return [
            f"protected legacy path was removed before authorization: {normalized}"
        ]
    if root_kind == "reparse":
        return [
            "protected legacy path uses a symlink or junction before "
            f"authorization: {normalized}"
        ]
    expected_root = expected_entries.get(normalized)
    if expected_root is None:
        return [
            f"approved legacy baseline is missing protected path: {normalized}"
        ]
    expected_root_kind = expected_root["gitType"]
    if root_kind != expected_root_kind:
        return [
            "protected legacy path type changed before authorization: "
            f"{normalized} (expected {expected_root_kind}, got {root_kind})"
        ]

    expected_paths = set(expected_entries)
    actual_paths = set(actual_entries)
    for entry_relative in sorted(expected_paths - actual_paths):
        violations.append(
            "protected legacy content was removed before authorization: "
            f"{entry_relative}"
        )
    for entry_relative in sorted(actual_paths - expected_paths):
        violations.append(
            "protected legacy path contains an unapproved filesystem entry before "
            f"authorization: {entry_relative}"
        )
    for entry_relative in sorted(expected_paths & actual_paths):
        expected = expected_entries[entry_relative]
        actual_kind = actual_entries[entry_relative]
        if actual_kind == "reparse":
            violations.append(
                "protected legacy path uses a symlink or junction before "
                f"authorization: {entry_relative}"
            )
            continue
        if actual_kind == "unsupported":
            violations.append(
                "protected legacy path has an unsupported filesystem type before "
                f"authorization: {entry_relative}"
            )
            continue
        if actual_kind != expected["gitType"]:
            violations.append(
                "protected legacy path type changed before authorization: "
                f"{entry_relative} (expected {expected['gitType']}, "
                f"got {actual_kind})"
            )
            continue
        if actual_kind != "blob":
            continue
        if expected["gitMode"] not in PROTECTED_BLOB_MODES:
            violations.append(
                "approved legacy baseline contains a non-regular blob mode: "
                f"{entry_relative} ({expected['gitMode']})"
            )
            continue
        if os.name != "nt":
            executable = bool(os.stat(repo_root / Path(entry_relative)).st_mode & 0o111)
            expected_executable = expected["gitMode"] == "100755"
            if executable != expected_executable:
                violations.append(
                    "protected legacy executable mode changed before authorization: "
                    f"{entry_relative}"
                )
        try:
            violations.extend(
                _content_transform_attribute_violations(
                    repo_root,
                    entry_relative,
                )
            )
            expected_bytes = _git_blob_bytes(
                repo_root,
                expected["gitObjectId"],
            )
            actual_bytes = (repo_root / Path(entry_relative)).read_bytes()
        except ManifestError as exc:
            violations.append(str(exc))
            continue
        except OSError as exc:
            violations.append(
                f"Cannot read protected legacy path {entry_relative}: {exc}"
            )
            continue
        if not _guard_content_equivalent(expected_bytes, actual_bytes):
            violations.append(
                "protected legacy content changed before authorization: "
                f"{entry_relative} (expected Git blob "
                f"{expected['gitObjectId']}, raw worktree SHA-256 "
                f"{hashlib.sha256(actual_bytes).hexdigest()})"
            )
    return violations


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def _repo_check(repo_root: Path, check: Mapping[str, Any]) -> dict[str, Any]:
    check_id = str(check["id"])
    check_type = str(check["type"])
    messages: list[str] = []
    details: dict[str, Any] = {}
    try:
        if check_type == "path_exists":
            path = _resolve_under(repo_root, str(check["path"]))
            passed = path.exists()
            if not passed:
                messages.append(f"required path is missing: {check['path']}")
        elif check_type == "paths_absent":
            present = [
                item
                for item in check["paths"]
                if _resolve_under(repo_root, str(item)).exists()
            ]
            passed = not present
            details["presentPaths"] = present
            if present:
                messages.append(f"paths must be absent: {', '.join(present)}")
        elif check_type == "glob_min":
            matches = [
                path
                for path in repo_root.glob(str(check["pattern"]))
                if path.exists()
            ]
            minimum = int(check["minimum"])
            passed = len(matches) >= minimum
            details["matchCount"] = len(matches)
            if not passed:
                messages.append(
                    f"glob {check['pattern']!r} matched {len(matches)}, needs {minimum}"
                )
        elif check_type in {"file_contains_all", "file_contains_none"}:
            path = _resolve_under(repo_root, str(check["path"]))
            text = _read_text(path)
            strings = [str(item) for item in check["strings"]]
            found = [item for item in strings if item in text]
            if check_type == "file_contains_all":
                missing = [item for item in strings if item not in text]
                passed = not missing
                details["missingStrings"] = missing
                if missing:
                    messages.append(f"required strings are missing from {check['path']}")
            else:
                passed = not found
                details["forbiddenStringsFound"] = found
                if found:
                    messages.append(f"forbidden strings remain in {check['path']}")
        elif check_type == "json_file_valid":
            load_json(_resolve_under(repo_root, str(check["path"])))
            passed = True
        elif check_type == "readmes_english":
            excluded = set(check.get("excludeParts", [])) | DEFAULT_EXCLUDED_PARTS
            violations: list[str] = []
            for path in sorted(repo_root.rglob("README*")):
                if not path.is_file() or any(part in excluded for part in path.parts):
                    continue
                try:
                    text = _read_text(path)
                except (OSError, UnicodeError):
                    violations.append(path.relative_to(repo_root).as_posix())
                    continue
                if CJK_RE.search(text):
                    violations.append(path.relative_to(repo_root).as_posix())
            passed = not violations
            details["violations"] = violations
            if violations:
                messages.append("public README files must be English-only")
        elif check_type == "text_patterns_absent":
            excluded = set(check.get("excludeParts", [])) | DEFAULT_EXCLUDED_PARTS
            compiled = [re.compile(str(pattern)) for pattern in check["patterns"]]
            violations: list[dict[str, Any]] = []
            for root_value in check["roots"]:
                root = _resolve_under(repo_root, str(root_value))
                candidates = [root] if root.is_file() else root.rglob("*") if root.exists() else []
                for path in candidates:
                    if (
                        not path.is_file()
                        or any(part in excluded for part in path.parts)
                        or len(violations) >= 100
                    ):
                        continue
                    try:
                        text = _read_text(path)
                    except (OSError, UnicodeError):
                        continue
                    for pattern in compiled:
                        match = pattern.search(text)
                        if match:
                            violations.append(
                                {
                                    "path": path.relative_to(repo_root).as_posix(),
                                    "pattern": pattern.pattern,
                                }
                            )
            passed = not violations
            details["violations"] = violations
            if violations:
                messages.append("forbidden production text patterns were found")
        else:
            passed = False
            messages.append(f"unsupported repository check type: {check_type}")
    except (OSError, UnicodeError, ManifestError, json.JSONDecodeError) as exc:
        passed = False
        messages.append(str(exc))
    return {
        "id": check_id,
        "type": check_type,
        "passed": passed,
        "messages": messages,
        "details": details,
    }


def command_sha256(check: Mapping[str, Any]) -> str:
    payload = {
        "argv": list(check["argv"]),
        "cwd": check.get("cwd", "."),
        "timeoutSeconds": check["timeoutSeconds"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_command_argv(check: Mapping[str, Any], repo_root: Path) -> list[str]:
    values: list[str] = []
    for item in check["argv"]:
        rendered = str(item).replace("{python}", sys.executable).replace(
            "{repo}", str(repo_root)
        )
        values.append(rendered)
    if values:
        executable = shutil.which(values[0])
        if executable:
            values[0] = executable
    return values


def _execute_command(repo_root: Path, check: Mapping[str, Any]) -> dict[str, Any]:
    cwd = _resolve_under(repo_root, str(check.get("cwd", ".")))
    argv = _resolve_command_argv(check, repo_root)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(check["timeoutSeconds"]),
            shell=False,
        )
        return {
            "passed": completed.returncode == 0,
            "exitCode": completed.returncode,
            "timedOut": False,
            "durationSeconds": round(time.monotonic() - started, 3),
            "stdoutTail": completed.stdout[-2000:],
            "stderrTail": completed.stderr[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "passed": False,
            "exitCode": None,
            "timedOut": True,
            "durationSeconds": round(time.monotonic() - started, 3),
            "stdoutTail": _coerce_tail(exc.stdout),
            "stderrTail": _coerce_tail(exc.stderr),
        }
    except OSError as exc:
        return {
            "passed": False,
            "exitCode": None,
            "timedOut": False,
            "durationSeconds": round(time.monotonic() - started, 3),
            "stdoutTail": "",
            "stderrTail": str(exc),
        }


def _coerce_tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-2000:]
    return value[-2000:]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> Any:
        raise ManifestError(f"non-finite JSON number is forbidden: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ManifestError(f"duplicate JSON key is forbidden: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"artifact is not strict UTF-8 JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"artifact JSON root must be an object: {path.name}")
    return value


def _artifact_payload(
    path: Path, role: str, contract: Mapping[str, str]
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    expected_extension = contract["extension"]
    if path.suffix.lower() != expected_extension:
        errors.append(f"role {role!r} must use {expected_extension}")
        return None, errors
    if expected_extension == ".json":
        try:
            payload = _load_strict_json_object(path)
        except ManifestError as exc:
            return None, [str(exc)]
        expected_schema = {
            "transcript": "2.0.0",
            "report-document": "1.0.0",
            "pdf-inspection": "1.0.0",
            "quality-report": "1.0.0",
            "artifact-manifest": "1.0.0",
        }.get(role)
        if expected_schema and payload.get("schemaVersion") != expected_schema:
            errors.append(
                f"role {role!r} schemaVersion must be {expected_schema!r}"
            )
        required_fields = {
            "transcript": (
                "documentId",
                "jobId",
                "source",
                "speakerPolicy",
                "speakers",
                "segments",
            ),
            "report-document": (
                "documentId",
                "source",
                "speakerPolicy",
                "speakers",
                "segments",
                "provenance",
            ),
            "pdf-inspection": (
                "validator",
                "validatorVersion",
                "documentId",
                "pdfSha256",
                "openable",
                "pageCount",
            ),
            "quality-report": (
                "documentId",
                "status",
                "minimumScore",
                "score",
                "hardGatesPassed",
                "hardGates",
                "facets",
                "evidence",
            ),
            "artifact-manifest": (
                "jobId",
                "documentId",
                "rendererVersion",
                "artifacts",
            ),
        }.get(role, ())
        for field in required_fields:
            if field not in payload:
                errors.append(f"role {role!r} is missing JSON field {field!r}")
        return payload, errors
    raw = path.read_bytes()
    if role == "pdf":
        if not raw.startswith(b"%PDF-"):
            errors.append("role 'pdf' is missing the PDF header")
        if b"%%EOF" not in raw[-2048:]:
            errors.append("role 'pdf' is missing a terminal PDF EOF marker")
    elif role == "contact-sheet":
        if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append("role 'contact-sheet' is not a PNG")
        elif raw[12:16] != b"IHDR" or int.from_bytes(raw[16:20], "big") < 1 or int.from_bytes(
            raw[20:24], "big"
        ) < 1:
            errors.append("role 'contact-sheet' has an invalid PNG IHDR")
    return None, errors


def _verify_artifacts(
    evidence_root: Path,
    artifacts: Any,
    minimum: int,
    role_contracts: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[bool, list[str], list[str], dict[str, dict[str, Any]]]:
    errors: list[str] = []
    roles: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    if not isinstance(artifacts, list) or len(artifacts) < minimum:
        return (
            False,
            [f"artifacts must contain at least {minimum} entries"],
            roles,
            details,
        )
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for index, artifact in enumerate(artifacts):
        context = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{context} must be an object")
            continue
        relative = artifact.get("path")
        role = artifact.get("role")
        expected_bytes = artifact.get("bytes")
        expected_sha = artifact.get("sha256")
        if not _is_safe_relative_path(relative):
            errors.append(f"{context}.path is unsafe")
            continue
        if relative in seen_paths:
            errors.append(f"{context}.path is duplicated")
        seen_paths.add(str(relative))
        if not isinstance(role, str) or not role:
            errors.append(f"{context}.role must be a non-empty string")
        else:
            roles.append(role)
            if role in seen_roles:
                errors.append(f"{context}.role is duplicated")
            seen_roles.add(role)
            if role_contracts is not None and role not in role_contracts:
                errors.append(f"{context}.role is not a canonical artifact role")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
        ):
            errors.append(f"{context}.bytes must be a positive integer")
        if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
            errors.append(f"{context}.sha256 must be a lowercase SHA-256")
        try:
            path = _resolve_under(evidence_root, str(relative))
        except ManifestError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"{context} file is missing: {relative}")
            continue
        actual_bytes = path.stat().st_size
        if isinstance(expected_bytes, int) and actual_bytes != expected_bytes:
            errors.append(
                f"{context}.bytes mismatch: expected {expected_bytes}, got {actual_bytes}"
            )
        if isinstance(expected_sha, str) and SHA256_RE.fullmatch(expected_sha):
            actual_sha = _sha256_file(path)
            if actual_sha != expected_sha:
                errors.append(f"{context}.sha256 mismatch")
        if role_contracts is not None and isinstance(role, str) and role in role_contracts:
            contract = role_contracts[role]
            if artifact.get("mediaType") != contract["mediaType"]:
                errors.append(
                    f"{context}.mediaType must be {contract['mediaType']!r}"
                )
            if artifact.get("format") != contract["format"]:
                errors.append(f"{context}.format must be {contract['format']!r}")
            payload, payload_errors = _artifact_payload(path, role, contract)
            errors.extend(f"{context}: {message}" for message in payload_errors)
            details[role] = {
                "path": path,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "payload": payload,
            }
    if role_contracts is not None:
        missing_roles = sorted(set(role_contracts) - seen_roles)
        if missing_roles:
            errors.append(
                "artifacts are missing canonical roles: " + ", ".join(missing_roles)
            )
    return not errors, errors, roles, details


def verify_attestation(
    *,
    evidence_root: Path | None,
    spec: Mapping[str, Any],
    gate_id: str,
    head_commit: str,
    max_age_hours: int,
    now: datetime,
    command_check: Mapping[str, Any] | None = None,
    real_media: Mapping[str, Any] | None = None,
    java_pdf: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a commit-bound attestation and every referenced artifact."""

    result: dict[str, Any] = {
        "path": spec.get("path"),
        "kind": spec.get("kind"),
        "passed": False,
        "messages": [],
    }
    messages: list[str] = result["messages"]
    if evidence_root is None:
        messages.append("evidence root is not configured")
        return result
    try:
        path = _resolve_under(evidence_root, str(spec["path"]))
        attestation = load_json(path)
    except ManifestError as exc:
        messages.append(str(exc))
        return result

    if attestation.get("schemaVersion") != SCHEMA_VERSION:
        messages.append(f"schemaVersion must be {SCHEMA_VERSION!r}")
    if attestation.get("kind") != spec.get("kind"):
        messages.append("kind does not match the manifest")
    if attestation.get("gateId") != gate_id:
        messages.append("gateId does not match the owning gate")
    if attestation.get("status") != "passed":
        messages.append("status must be 'passed'")
    if attestation.get("commitSha") != head_commit:
        messages.append("commitSha does not match the current HEAD")
    generated_at = _parse_timestamp(attestation.get("generatedAt"))
    if generated_at is None:
        messages.append("generatedAt must be an ISO-8601 timestamp with timezone")
    else:
        if generated_at > now + timedelta(minutes=5):
            messages.append("generatedAt is unacceptably in the future")
        if now - generated_at > timedelta(hours=max_age_hours):
            messages.append("attestation is stale")

    if command_check is not None:
        if attestation.get("checkId") != command_check.get("id"):
            messages.append("checkId does not match the command check")
        if attestation.get("commandSha256") != command_sha256(command_check):
            messages.append("commandSha256 does not match the manifest command")
        if attestation.get("exitCode") != 0:
            messages.append("command exitCode must be 0")

    minimum = int(spec.get("minimumArtifacts", 0))
    artifact_details: dict[str, dict[str, Any]] = {}
    if minimum or "artifacts" in attestation:
        profile = spec.get("profile")
        role_contracts = (
            real_media.get("artifactRoleContracts")
            if profile and real_media is not None
            else None
        )
        artifacts_ok, artifact_errors, roles, artifact_details = _verify_artifacts(
            evidence_root,
            attestation.get("artifacts"),
            minimum,
            role_contracts=role_contracts,
        )
        if not artifacts_ok:
            messages.extend(artifact_errors)
        result["artifactRoles"] = roles

    profile = spec.get("profile")
    if profile and real_media is not None and java_pdf is not None:
        _validate_real_media_attestation(
            messages,
            attestation,
            profile,
            real_media,
            java_pdf,
            result,
            artifact_details,
        )

    result["passed"] = not messages
    return result


def _validate_real_media_attestation(
    messages: list[str],
    attestation: Mapping[str, Any],
    profile: str,
    real_media: Mapping[str, Any],
    java_pdf: Mapping[str, Any],
    result: dict[str, Any],
    artifact_details: Mapping[str, Mapping[str, Any]],
) -> None:
    run_id = attestation.get("runId")
    source_name = attestation.get("sourceFileName")
    source_sha = attestation.get("sourceSha256")
    job_id = attestation.get("jobId")
    document_id = attestation.get("documentId")
    if not isinstance(run_id, str) or not run_id.strip():
        messages.append("runId must be a non-empty string")
    if not isinstance(job_id, str) or not job_id.strip():
        messages.append("jobId must be a non-empty string")
    if not isinstance(document_id, str) or not document_id.strip():
        messages.append("documentId must be a non-empty string")
    if source_name != real_media.get("sourceFileName"):
        messages.append("sourceFileName does not match the approved real MOV basename")
    if not isinstance(source_sha, str) or not SHA256_RE.fullmatch(source_sha):
        messages.append("sourceSha256 must be a lowercase SHA-256")
    if attestation.get("terminalState") != "completed":
        messages.append("terminalState must be 'completed'")
    language = attestation.get("persistedLanguage")
    if not isinstance(language, str) or not language or language.lower() == "auto":
        messages.append("persistedLanguage must be concrete and must not be 'auto'")

    mode = attestation.get("speakerCountMode")
    detected = attestation.get("detectedSpeakerCount")
    resolved = attestation.get("resolvedSpeakerCount")
    speaker_ids = attestation.get("speakerIds")
    if (
        not isinstance(detected, int)
        or isinstance(detected, bool)
        or detected <= 0
        or not isinstance(resolved, int)
        or isinstance(resolved, bool)
        or resolved <= 0
    ):
        messages.append("detectedSpeakerCount and resolvedSpeakerCount must be positive")
    elif detected != resolved:
        messages.append("detected and resolved speaker counts must agree")
    if isinstance(resolved, int) and not isinstance(resolved, bool) and resolved > 0:
        expected_ids = [f"speaker-{index}" for index in range(1, resolved + 1)]
        if speaker_ids != expected_ids:
            messages.append("speakerIds must be contiguous speaker-1 through speaker-N")

    if profile == "realMediaAuto":
        if mode != "auto":
            messages.append("auto evidence must use speakerCountMode='auto'")
        for key in ("manualSpeakerCount", "requestedSpeakerCount"):
            if attestation.get(key) is not None:
                messages.append(f"auto evidence must not contain a {key} override")
    elif profile == "realMediaManualFive":
        expected = real_media.get("manualRegressionSpeakerCount")
        if mode != "manual":
            messages.append("manual evidence must use speakerCountMode='manual'")
        if attestation.get("requestedSpeakerCount") != expected:
            messages.append("manual evidence requestedSpeakerCount must be 5")
        if attestation.get("manualSpeakerCount") != expected:
            messages.append("manual evidence manualSpeakerCount must be 5")
        if detected != expected or resolved != expected:
            messages.append("manual evidence must detect and resolve exactly five speakers")

    required_roles = set(real_media.get("requiredArtifactRoles", []))
    actual_roles = set(result.get("artifactRoles", []))
    missing_roles = sorted(required_roles - actual_roles)
    if missing_roles:
        messages.append(
            "real MOV evidence is missing artifact roles: " + ", ".join(missing_roles)
        )
    _validate_real_media_artifact_relationships(
        messages,
        attestation,
        real_media,
        java_pdf,
        artifact_details,
    )

    quality = attestation.get("quality")
    if not isinstance(quality, dict):
        messages.append("quality must be an object")
    else:
        if quality.get("hardGatesPassed") is not True:
            messages.append("quality.hardGatesPassed must be true")
        _validate_status_sequence(
            messages,
            quality.get("hardGates"),
            java_pdf.get("hardGateIds", []),
            "quality.hardGates",
        )
        _validate_status_sequence(
            messages,
            quality.get("facets"),
            java_pdf.get("designPackFacetIds", []),
            "quality.facets",
        )
        score = quality.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or score < java_pdf.get("minimumQualityScore", 85)
        ):
            messages.append("quality.score is below the required minimum")
        for key in (
            "missingSegments",
            "missingTimestamps",
            "missingSpeakerIds",
            "fontFailures",
            "blankPages",
            "overflowFindings",
            "remoteAssetFindings",
        ):
            if quality.get(key) != []:
                messages.append(f"quality.{key} must be an empty array")

    metrics = attestation.get("metrics")
    if not isinstance(metrics, dict):
        messages.append("metrics must be an object")
    else:
        for domain in real_media.get("requiredMetricDomains", []):
            domain_value = metrics.get(domain)
            if not isinstance(domain_value, dict) or domain_value.get("status") != "passed":
                messages.append(f"metrics.{domain}.status must be 'passed'")
        _validate_real_media_metrics(
            messages,
            metrics,
            attestation,
            real_media,
            artifact_details,
        )

    result["runId"] = run_id
    result["sourceSha256"] = source_sha
    result["speakerCountMode"] = mode
    result["resolvedSpeakerCount"] = resolved


def _payload(
    artifact_details: Mapping[str, Mapping[str, Any]], role: str
) -> Mapping[str, Any] | None:
    detail = artifact_details.get(role)
    value = detail.get("payload") if isinstance(detail, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _validate_real_media_artifact_relationships(
    messages: list[str],
    attestation: Mapping[str, Any],
    real_media: Mapping[str, Any],
    java_pdf: Mapping[str, Any],
    artifact_details: Mapping[str, Mapping[str, Any]],
) -> None:
    transcript = _payload(artifact_details, "transcript")
    report = _payload(artifact_details, "report-document")
    inspection = _payload(artifact_details, "pdf-inspection")
    quality = _payload(artifact_details, "quality-report")
    manifest = _payload(artifact_details, "artifact-manifest")
    if not all((transcript, report, inspection, quality, manifest)):
        return

    job_id = attestation.get("jobId")
    document_id = attestation.get("documentId")
    source_sha = attestation.get("sourceSha256")
    source_name = real_media.get("sourceFileName")
    if transcript.get("jobId") != job_id:
        messages.append("transcript jobId does not match the attestation")
    if transcript.get("documentId") != document_id:
        messages.append("transcript documentId does not match the attestation")
    if report.get("documentId") != document_id:
        messages.append("ReportDocument documentId does not match the attestation")
    transcript_source = transcript.get("source")
    report_source = report.get("source")
    for label, source in (
        ("transcript", transcript_source),
        ("ReportDocument", report_source),
    ):
        if not isinstance(source, Mapping):
            messages.append(f"{label} source must be an object")
            continue
        if source.get("fileName") != source_name:
            messages.append(f"{label} source fileName does not match the real MOV")
        if source.get("sha256") != source_sha:
            messages.append(f"{label} source sha256 does not match the attestation")

    expected_ids = attestation.get("speakerIds")
    resolved = attestation.get("resolvedSpeakerCount")
    mode = attestation.get("speakerCountMode")
    for label, document in (("transcript", transcript), ("ReportDocument", report)):
        policy = document.get("speakerPolicy")
        speakers = document.get("speakers")
        segments = document.get("segments")
        if not isinstance(policy, Mapping):
            messages.append(f"{label} speakerPolicy must be an object")
            continue
        if policy.get("mode") != mode:
            messages.append(f"{label} speakerPolicy mode does not match")
        if policy.get("resolvedCount") != resolved:
            messages.append(f"{label} resolved speaker count does not match")
        if policy.get("speakerIds") != expected_ids:
            messages.append(f"{label} speaker IDs do not match")
        if not isinstance(speakers, list) or [
            item.get("id") for item in speakers if isinstance(item, Mapping)
        ] != expected_ids:
            messages.append(f"{label} speaker records do not match the canonical set")
        if not isinstance(segments, list) or not segments:
            messages.append(f"{label} segments must be a non-empty array")
        elif any(
            not isinstance(item, Mapping) or item.get("speakerId") not in set(expected_ids or [])
            for item in segments
        ):
            messages.append(f"{label} contains a segment outside the speaker set")

    transcript_segments = transcript.get("segments")
    report_segments = report.get("segments")
    if isinstance(transcript_segments, list) and isinstance(report_segments, list):
        identity_fields = ("id", "startMs", "endMs", "speakerId")
        transcript_identity = [
            tuple(item.get(field) for field in identity_fields)
            for item in transcript_segments
            if isinstance(item, Mapping)
        ]
        report_identity = [
            tuple(item.get(field) for field in identity_fields)
            for item in report_segments
            if isinstance(item, Mapping)
        ]
        if transcript_identity != report_identity:
            messages.append(
                "transcript and ReportDocument segment identity/timing/speaker data differ"
            )

    pdf_detail = artifact_details.get("pdf", {})
    if inspection.get("validator") != java_pdf.get("validator"):
        messages.append("PDF inspection validator is not the approved PDFBox engine")
    if inspection.get("validatorVersion") != java_pdf.get("validatorVersion"):
        messages.append("PDF inspection validatorVersion does not match policy")
    if inspection.get("documentId") != document_id:
        messages.append("PDF inspection documentId does not match")
    if inspection.get("pdfSha256") != pdf_detail.get("sha256"):
        messages.append("PDF inspection is not bound to the attested PDF hash")
    for field in (
        "openable",
        "allPagesA4",
        "transcriptTextIntegrity",
        "segmentCountIntegrity",
        "timestampIntegrity",
        "speakerSetIntegrity",
        "allFontsEmbedded",
        "searchableText",
        "noReplacementCharacters",
    ):
        if inspection.get(field) is not True:
            messages.append(f"PDFBox inspection field {field} must be true")
    page_count = inspection.get("pageCount")
    pages = inspection.get("pages")
    if (
        not isinstance(page_count, int)
        or isinstance(page_count, bool)
        or page_count < 1
        or not isinstance(pages, list)
        or len(pages) != page_count
    ):
        messages.append("PDFBox inspection page evidence is incomplete")
    for field in (
        "missingSegmentIds",
        "duplicateSegmentIds",
        "missingTimestamps",
        "missingSpeakerIds",
    ):
        if inspection.get(field) != []:
            messages.append(f"PDFBox inspection {field} must be empty")

    if quality.get("documentId") != document_id:
        messages.append("quality report documentId does not match")
    if quality.get("status") != "passed" or quality.get("hardGatesPassed") is not True:
        messages.append("quality report must be in a terminal passed state")
    _validate_status_sequence(
        messages,
        quality.get("hardGates"),
        java_pdf.get("hardGateIds", []),
        "quality-report.hardGates",
    )
    _validate_status_sequence(
        messages,
        quality.get("facets"),
        java_pdf.get("designPackFacetIds", []),
        "quality-report.facets",
    )
    score = quality.get("score")
    minimum_score = quality.get("minimumScore")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not isinstance(minimum_score, (int, float))
        or isinstance(minimum_score, bool)
        or float(score) < float(java_pdf.get("minimumQualityScore", 85))
        or float(minimum_score) < float(java_pdf.get("minimumQualityScore", 85))
    ):
        messages.append("quality report score/minimumScore is below policy")
    evidence = quality.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(
        not isinstance(item, Mapping)
        or item.get("verified") is not True
        or not isinstance(item.get("sha256"), str)
        or not SHA256_RE.fullmatch(item["sha256"])
        for item in evidence
    ):
        messages.append("quality report must contain verified hashed evidence")
    repairs = quality.get("repairQueue")
    if not isinstance(repairs, list) or any(
        isinstance(item, Mapping) and item.get("status") in {"pending", "blocked"}
        for item in repairs
    ):
        messages.append("quality report contains unresolved repairs")

    if manifest.get("jobId") != job_id:
        messages.append("artifact manifest jobId does not match")
    if manifest.get("documentId") != document_id:
        messages.append("artifact manifest documentId does not match")
    if manifest.get("rendererVersion") != "3.0.0":
        messages.append("artifact manifest rendererVersion must be '3.0.0'")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, list):
        messages.append("artifact manifest artifacts must be an array")
    else:
        by_type = {
            item.get("type"): item
            for item in manifest_artifacts
            if isinstance(item, Mapping)
        }
        for role in ("report-document", "pdf", "quality-report", "contact-sheet"):
            item = by_type.get(role)
            detail = artifact_details.get(role)
            if (
                not isinstance(item, Mapping)
                or not isinstance(detail, Mapping)
                or item.get("verified") is not True
                or item.get("sha256") != detail.get("sha256")
                or item.get("bytes") != detail.get("bytes")
            ):
                messages.append(
                    f"artifact manifest does not bind the attested {role} artifact"
                )


def _validate_real_media_metrics(
    messages: list[str],
    metrics: Mapping[str, Any],
    attestation: Mapping[str, Any],
    real_media: Mapping[str, Any],
    artifact_details: Mapping[str, Mapping[str, Any]],
) -> None:
    for domain, fields in REAL_MEDIA_METRIC_FIELDS.items():
        value = metrics.get(domain)
        if not isinstance(value, Mapping):
            continue
        for field in fields:
            if field not in value:
                messages.append(f"metrics.{domain}.{field} is required")

    segment_count = 0
    transcript = _payload(artifact_details, "transcript")
    if transcript is not None and isinstance(transcript.get("segments"), list):
        segment_count = len(transcript["segments"])

    speaker = metrics.get("speakerCount")
    if isinstance(speaker, Mapping):
        expected = attestation.get("resolvedSpeakerCount")
        checks = (
            ("expectedCount", expected),
            ("detectedCount", attestation.get("detectedSpeakerCount")),
            ("resolvedCount", expected),
            ("absoluteError", 0),
            ("reviewRequired", False),
        )
        for field, expected_value in checks:
            if speaker.get(field) != expected_value:
                messages.append(
                    f"metrics.speakerCount.{field} does not match recomputed evidence"
                )

    diarization = metrics.get("diarization")
    if isinstance(diarization, Mapping):
        for field, expected_value in (
            ("segmentCount", segment_count),
            ("auditedSegmentCount", segment_count),
            ("speakerAssignmentErrors", 0),
            ("unresolvedSpeakerSegments", 0),
            ("humanAuditCompleted", True),
        ):
            if diarization.get(field) != expected_value:
                messages.append(
                    f"metrics.diarization.{field} does not meet acceptance"
                )

    boundary = metrics.get("boundary")
    if isinstance(boundary, Mapping):
        for field, expected_value in (
            ("segmentCount", segment_count),
            ("invalidIntervals", 0),
            ("nonMonotonicIntervals", 0),
            ("outOfBoundsIntervals", 0),
        ):
            if boundary.get(field) != expected_value:
                messages.append(f"metrics.boundary.{field} does not meet acceptance")

    asr = metrics.get("asr")
    if isinstance(asr, Mapping):
        for field, expected_value in (
            ("segmentCount", segment_count),
            ("auditedSegmentCount", segment_count),
            ("emptySegments", 0),
            ("unresolvedTextSegments", 0),
            ("sourceLanguagePreserved", True),
        ):
            if asr.get(field) != expected_value:
                messages.append(f"metrics.asr.{field} does not meet acceptance")

    semantic = metrics.get("semantic")
    if isinstance(semantic, Mapping):
        revision_count = semantic.get("reviewedRevisionCount")
        if (
            not isinstance(revision_count, int)
            or isinstance(revision_count, bool)
            or revision_count < 0
        ):
            messages.append(
                "metrics.semantic.reviewedRevisionCount must be non-negative"
            )
        for field, expected_value in (
            ("unresolvedRevisionCount", 0),
            ("rawTranscriptImmutable", True),
            ("speakerLocksPreserved", True),
            ("humanApproved", True),
        ):
            if semantic.get(field) != expected_value:
                messages.append(f"metrics.semantic.{field} does not meet acceptance")

    efficiency = metrics.get("efficiency")
    if isinstance(efficiency, Mapping):
        try:
            duration = float(efficiency.get("mediaDurationSeconds"))
            wall = float(efficiency.get("wallClockSeconds"))
            rtf = float(efficiency.get("realTimeFactor"))
            peak_ram = float(efficiency.get("peakRamMb"))
            peak_vram = float(efficiency.get("peakVramMb"))
        except (TypeError, ValueError):
            messages.append("metrics.efficiency numeric values are invalid")
        else:
            if (
                not all(map(math.isfinite, (duration, wall, rtf, peak_ram, peak_vram)))
                or duration <= 0
                or wall <= 0
                or peak_ram < 0
                or peak_vram < 0
            ):
                messages.append("metrics.efficiency values are out of range")
            else:
                computed = wall / duration
                if abs(computed - rtf) > max(0.001, computed * 0.01):
                    messages.append(
                        "metrics.efficiency.realTimeFactor does not match wall/media time"
                    )
                if rtf > float(real_media.get("maximumRealTimeFactor", 4.0)):
                    messages.append("metrics.efficiency.realTimeFactor exceeds policy")

    pdf_metrics = metrics.get("pdf")
    inspection = _payload(artifact_details, "pdf-inspection")
    quality = _payload(artifact_details, "quality-report")
    if (
        isinstance(pdf_metrics, Mapping)
        and inspection is not None
        and quality is not None
    ):
        hard_failures = sum(
            1
            for item in quality.get("hardGates", [])
            if not isinstance(item, Mapping) or item.get("status") != "passed"
        )
        facet_failures = sum(
            1
            for item in quality.get("facets", [])
            if not isinstance(item, Mapping) or item.get("status") != "passed"
        )
        for field, expected_value in (
            ("pageCount", inspection.get("pageCount")),
            ("qualityScore", quality.get("score")),
            ("hardGateFailureCount", hard_failures),
            ("facetFailureCount", facet_failures),
            ("pdfBoxValidated", True),
        ):
            if pdf_metrics.get(field) != expected_value:
                messages.append(f"metrics.pdf.{field} does not match PDF evidence")


def _validate_status_sequence(
    messages: list[str], value: Any, expected_ids: Sequence[str], context: str
) -> None:
    if not isinstance(value, list):
        messages.append(f"{context} must be an array")
        return
    actual_ids = [
        item.get("id") if isinstance(item, dict) else None for item in value
    ]
    if actual_ids != list(expected_ids):
        messages.append(f"{context} must preserve the canonical complete ordered IDs")
    if any(
        not isinstance(item, dict) or item.get("status") != "passed" for item in value
    ):
        messages.append(f"every {context} entry must have status='passed'")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _command_check_result(
    *,
    repo_root: Path,
    evidence_root: Path | None,
    gate_id: str,
    check: Mapping[str, Any],
    head_commit: str,
    max_age_hours: int,
    now: datetime,
    execute: bool,
    write_attestations: bool,
) -> dict[str, Any]:
    spec = {
        "path": check["attestationPath"],
        "kind": "command-attestation",
        "minimumArtifacts": 0,
    }
    if not execute:
        verified = verify_attestation(
            evidence_root=evidence_root,
            spec=spec,
            gate_id=gate_id,
            head_commit=head_commit,
            max_age_hours=max_age_hours,
            now=now,
            command_check=check,
        )
        return {
            "id": check["id"],
            "argv": check["argv"],
            "executed": False,
            "passed": verified["passed"],
            "messages": verified["messages"],
        }

    execution = _execute_command(repo_root, check)
    messages: list[str] = []
    if not execution["passed"]:
        messages.append(
            "command failed"
            + (" by timeout" if execution["timedOut"] else "")
            + (
                f" with exit code {execution['exitCode']}"
                if execution["exitCode"] is not None
                else ""
            )
        )
    if write_attestations and execution["passed"]:
        if evidence_root is None:
            execution["passed"] = False
            messages.append("cannot write attestation without an external evidence root")
        else:
            path = _resolve_under(evidence_root, str(check["attestationPath"]))
            _write_json_atomic(
                path,
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "command-attestation",
                    "gateId": gate_id,
                    "status": "passed",
                    "commitSha": head_commit,
                    "generatedAt": now.isoformat().replace("+00:00", "Z"),
                    "checkId": check["id"],
                    "commandSha256": command_sha256(check),
                    "exitCode": 0,
                    "durationSeconds": execution["durationSeconds"],
                },
            )
    return {
        "id": check["id"],
        "argv": check["argv"],
        "executed": True,
        "passed": execution["passed"],
        "messages": messages,
        "durationSeconds": execution["durationSeconds"],
        "exitCode": execution["exitCode"],
        "timedOut": execution["timedOut"],
        "stdoutTail": execution["stdoutTail"],
        "stderrTail": execution["stderrTail"],
    }


def _authorization_ready(value: Mapping[str, Any], now: datetime) -> tuple[bool, str]:
    normalized_now = _normalize_evaluation_time(now)
    if normalized_now is None:
        return False, "evaluation time must be a timezone-aware datetime"
    if value.get("approved") is not True:
        return False, "approval is false"
    for key in ("approvedBy", "approvedAt", "changeTicket"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            return False, f"{key} is missing"
    approved_at = _parse_timestamp(value.get("approvedAt"))
    if approved_at is None:
        return False, "approvedAt is not a valid timezone-qualified RFC3339 timestamp"
    if approved_at > normalized_now:
        return False, "approvedAt is in the future"
    return True, "approved"


def evaluate_guardrails(
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    parity_eligible: bool,
    release_gates_passed: bool,
    head_commit: str,
    refs: Mapping[str, str | None],
    now: datetime,
    repository_integrity_passed: bool = True,
    repository_integrity_violations: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate legacy and main protection independently from capability scores."""

    policy = manifest["policy"]
    authorization = policy["authorization"]
    legacy_auth, legacy_auth_message = _authorization_ready(
        authorization["legacyRemoval"], now
    )
    main_auth, main_auth_message = _authorization_ready(
        authorization["mainReplacement"], now
    )
    authorization_order: list[str] = []
    legacy_approved_at = (
        _parse_timestamp(authorization["legacyRemoval"].get("approvedAt"))
        if legacy_auth
        else None
    )
    main_approved_at = (
        _parse_timestamp(authorization["mainReplacement"].get("approvedAt"))
        if main_auth
        else None
    )
    if main_auth and not legacy_auth:
        authorization_order.append(
            "main replacement cannot be approved before legacy removal approval"
        )
    elif (
        legacy_approved_at is not None
        and main_approved_at is not None
        and main_approved_at < legacy_approved_at
    ):
        authorization_order.append(
            "main replacement approvedAt must be greater than or equal to "
            "legacy removal approvedAt"
        )
    authorization_order_passed = not authorization_order
    legacy_provisionally_allowed = (
        parity_eligible
        and legacy_auth
        and repository_integrity_passed
    )
    main_provisionally_allowed = (
        release_gates_passed
        and legacy_auth
        and main_auth
        and authorization_order_passed
        and repository_integrity_passed
    )

    head_violations: list[str] = []
    actual_head: str | None = None
    baseline = policy["protectedLegacyBaseline"]
    source_commit = baseline["sourceCommit"]
    try:
        actual_head = _run_git(repo_root, "rev-parse", "HEAD")
    except ManifestError as exc:
        head_violations.append(str(exc))
    if actual_head is not None and actual_head != head_commit:
        head_violations.append(
            "guardrail HEAD snapshot mismatch: "
            f"expected current HEAD {actual_head}, got {head_commit}"
        )
    if actual_head is not None:
        try:
            if not _git_is_ancestor(repo_root, source_commit, actual_head):
                head_violations.append(
                    "approved legacy baseline commit is not an ancestor of current "
                    f"HEAD: baseline {source_commit}, HEAD {actual_head}"
                )
        except ManifestError as exc:
            head_violations.append(str(exc))

    legacy_violations: list[str] = []
    for relative in policy["protectedLegacyPaths"]:
        expected_entry = baseline["entries"][relative]
        try:
            source_entry = _git_tree_entry(repo_root, source_commit, relative)
        except ManifestError as exc:
            legacy_violations.append(str(exc))
            source_entry = None
        if source_entry != expected_entry:
            legacy_violations.append(
                "approved legacy baseline identity does not match its source commit: "
                f"{relative} (expected {_describe_git_entry(expected_entry)}, "
                f"got {_describe_git_entry(source_entry)})"
            )
        if actual_head is None:
            legacy_violations.append(
                "cannot verify protected legacy identity without a current Git HEAD: "
                f"{relative}"
            )
            continue
        try:
            current_entry = _git_tree_entry(repo_root, actual_head, relative)
        except ManifestError as exc:
            legacy_violations.append(str(exc))
            current_entry = None
        if current_entry != expected_entry:
            legacy_violations.append(
                "protected legacy Git identity changed before authorization: "
                f"{relative} (expected {_describe_git_entry(expected_entry)}, "
                f"got {_describe_git_entry(current_entry)})"
            )
        legacy_violations.extend(
            _protected_worktree_violations(
                repo_root=repo_root,
                source_commit=source_commit,
                relative=relative,
            )
        )

    main_violations: list[str] = []
    actual_refs: dict[str, str | None] = {}
    for ref_name, expected in policy["protectedMainRefs"].items():
        try:
            actual = _git_ref(repo_root, ref_name)
        except ManifestError as exc:
            main_violations.append(str(exc))
            actual = None
        actual_refs[ref_name] = actual
        supplied = refs.get(ref_name)
        if supplied != actual:
            main_violations.append(
                f"guardrail ref snapshot mismatch for {ref_name}: "
                f"current value is {actual or 'missing'}, "
                f"supplied value is {supplied or 'missing'}"
            )
        if actual != expected:
            main_violations.append(
                f"protected ref {ref_name} changed before cutover execution: "
                f"expected {expected}, got {actual or 'missing'}"
            )

    legacy_allowed = (
        legacy_provisionally_allowed
        and not head_violations
        and not legacy_violations
    )
    main_allowed = (
        main_provisionally_allowed
        and legacy_allowed
        and not main_violations
    )

    return {
        "legacyRemovalAuthorization": {
            "passed": legacy_auth,
            "message": legacy_auth_message,
        },
        "mainReplacementAuthorization": {
            "passed": main_auth,
            "message": main_auth_message,
        },
        "headProtection": {
            "passed": not head_violations,
            "evaluatedHead": head_commit,
            "actualHead": actual_head,
            "baselineSourceCommit": source_commit,
            "violations": head_violations,
        },
        "legacyProtection": {
            "passed": not legacy_violations,
            "violations": legacy_violations,
        },
        "mainProtection": {
            "passed": not main_violations,
            "actualRefs": actual_refs,
            "violations": main_violations,
        },
        "authorizationOrder": {
            "passed": not authorization_order,
            "violations": authorization_order,
        },
        "repositoryIntegrity": {
            "passed": repository_integrity_passed,
            "violations": list(repository_integrity_violations),
        },
        "legacyRemovalAllowed": legacy_allowed,
        "mainReplacementAllowed": main_allowed,
    }


def _safe_evidence_root(
    repo_root: Path, manifest: Mapping[str, Any], requested: Path | None
) -> tuple[Path | None, list[str]]:
    if requested is None:
        env_name = manifest["policy"]["evidenceRootEnv"]
        env_value = os.environ.get(env_name)
        if not env_value:
            return None, []
        requested = Path(env_value)
    resolved = requested.expanduser().resolve()
    repo = repo_root.resolve()
    errors: list[str] = []
    if (
        manifest["policy"]["forbidEvidenceInsideRepository"]
        and (resolved == repo or repo in resolved.parents)
    ):
        errors.append("evidence root must be outside the repository")
    return resolved, errors


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    evidence_root: Path | None = None,
    execute: bool = False,
    write_attestations: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the complete audit and return a stable machine-readable result."""

    if now is None:
        now = datetime.now(timezone.utc)
    else:
        normalized_now = _normalize_evaluation_time(now)
        if normalized_now is None:
            error = "evaluation time must be a timezone-aware datetime"
            return {
                "schemaVersion": SCHEMA_VERSION,
                "manifestValid": False,
                "configurationErrors": [error],
                "releaseEligible": False,
                "parityEligible": False,
                "legacyRemovalAllowed": False,
                "mainReplacementAllowed": False,
                "blockingReasons": [error],
            }
        now = normalized_now
    manifest, manifest_decision_bytes = load_json_with_bytes(manifest_path)
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "manifestValid": False,
            "configurationErrors": manifest_errors,
            "releaseEligible": False,
            "parityEligible": False,
            "legacyRemovalAllowed": False,
            "mainReplacementAllowed": False,
            "blockingReasons": manifest_errors,
        }

    repo_root = repo_root.resolve()
    try:
        git = _git_state(
            repo_root,
            manifest["policy"]["protectedMainRefs"].keys(),
            manifest_path,
            manifest_decision_bytes,
        )
    except ManifestError as exc:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "manifestValid": True,
            "configurationErrors": [str(exc)],
            "releaseEligible": False,
            "parityEligible": False,
            "legacyRemovalAllowed": False,
            "mainReplacementAllowed": False,
            "blockingReasons": [str(exc)],
        }

    resolved_evidence, evidence_configuration_errors = _safe_evidence_root(
        repo_root, manifest, evidence_root
    )
    max_age_hours = int(manifest["policy"]["maxEvidenceAgeHours"])
    gate_results: list[dict[str, Any]] = []
    real_results: dict[str, dict[str, Any]] = {}

    for gate in manifest["gates"]:
        gate_messages: list[str] = []
        if gate["state"] != "passed":
            gate_messages.append(
                f"manifest state is {gate['state']!r}; only 'passed' can release"
            )
        repo_checks = [_repo_check(repo_root, item) for item in gate["repoChecks"]]
        command_checks = [
            _command_check_result(
                repo_root=repo_root,
                evidence_root=resolved_evidence,
                gate_id=gate["id"],
                check=item,
                head_commit=git["headCommit"],
                max_age_hours=max_age_hours,
                now=now,
                execute=execute,
                write_attestations=write_attestations,
            )
            for item in gate["commandChecks"]
        ]
        attestations = [
            verify_attestation(
                evidence_root=resolved_evidence,
                spec=spec,
                gate_id=gate["id"],
                head_commit=git["headCommit"],
                max_age_hours=max_age_hours,
                now=now,
                real_media=manifest["realMedia"],
                java_pdf=manifest["javaPdf"],
            )
            for spec in gate["evidence"]["attestations"]
        ]
        for item in attestations:
            if item.get("speakerCountMode"):
                real_results[gate["id"]] = item
        passed = (
            gate["state"] == "passed"
            and all(item["passed"] for item in repo_checks)
            and all(item["passed"] for item in command_checks)
            and all(item["passed"] for item in attestations)
        )
        gate_messages.extend(
            message
            for collection in (repo_checks, command_checks, attestations)
            for item in collection
            for message in item.get("messages", [])
        )
        gate_results.append(
            {
                "id": gate["id"],
                "title": gate["title"],
                "manifestState": gate["state"],
                "passed": passed,
                "repoChecks": repo_checks,
                "commandChecks": command_checks,
                "attestations": attestations,
                "blockingReasons": gate_messages,
            }
        )

    auto_gate = manifest["realMedia"]["autoGateId"]
    manual_gate = manifest["realMedia"]["manualFiveGateId"]
    auto_result = real_results.get(auto_gate)
    manual_result = real_results.get(manual_gate)
    if auto_result and manual_result and auto_result["passed"] and manual_result["passed"]:
        cross_errors: list[str] = []
        if auto_result.get("sourceSha256") != manual_result.get("sourceSha256"):
            cross_errors.append(
                "real MOV auto and manual=5 attestations must share sourceSha256"
            )
        if auto_result.get("runId") == manual_result.get("runId"):
            cross_errors.append(
                "real MOV auto and manual=5 evidence must come from independent runs"
            )
        if cross_errors:
            for gate_result in gate_results:
                if gate_result["id"] in {auto_gate, manual_gate}:
                    gate_result["passed"] = False
                    gate_result["blockingReasons"].extend(cross_errors)

    gate_map = {item["id"]: item for item in gate_results}
    parity_eligible = all(
        gate_map[gate_id]["passed"] for gate_id in manifest["policy"]["parityGateIds"]
    )
    release_gates_passed = all(
        gate_map[gate_id]["passed"] for gate_id in manifest["policy"]["releaseGateIds"]
    )
    guardrails = evaluate_guardrails(
        repo_root=repo_root,
        manifest=manifest,
        parity_eligible=parity_eligible,
        release_gates_passed=release_gates_passed,
        head_commit=git["headCommit"],
        refs=git["refs"],
        now=now,
        repository_integrity_passed=git["workingTreeClean"],
        repository_integrity_violations=git["workingTreeChanges"],
    )
    guardrails_passed = all(
        guardrails[key]["passed"]
        for key in (
            "headProtection",
            "legacyProtection",
            "mainProtection",
            "authorizationOrder",
            "repositoryIntegrity",
        )
    )
    release_eligible = (
        release_gates_passed
        and guardrails["legacyRemovalAllowed"]
        and guardrails["mainReplacementAllowed"]
        and git["workingTreeClean"]
        and guardrails_passed
        and not evidence_configuration_errors
    )

    blocking_reasons: list[str] = []
    blocking_reasons.extend(evidence_configuration_errors)
    for gate in gate_results:
        if not gate["passed"]:
            blocking_reasons.append(f"{gate['id']} did not pass")
    if not git["workingTreeClean"]:
        blocking_reasons.append("working tree is not clean")
    for key in (
        "headProtection",
        "legacyProtection",
        "mainProtection",
        "authorizationOrder",
        "repositoryIntegrity",
    ):
        blocking_reasons.extend(guardrails[key]["violations"])
    if not guardrails["legacyRemovalAllowed"]:
        blocking_reasons.append("legacy removal is not authorized")
    if not guardrails["mainReplacementAllowed"]:
        blocking_reasons.append("main replacement is not authorized")

    passed_count = sum(1 for item in gate_results if item["passed"])
    return {
        "schemaVersion": SCHEMA_VERSION,
        "auditId": manifest["auditId"],
        "manifestValid": True,
        "configurationErrors": evidence_configuration_errors,
        "releaseEligible": release_eligible,
        "parityEligible": parity_eligible,
        "legacyRemovalAllowed": guardrails["legacyRemovalAllowed"],
        "mainReplacementAllowed": guardrails["mainReplacementAllowed"],
        "headCommit": git["headCommit"],
        "branch": git["branch"],
        "workingTreeClean": git["workingTreeClean"],
        "workingTreeChanges": git["workingTreeChanges"],
        "indexFlagsClean": git["indexFlagsClean"],
        "indexFlaggedPaths": git["indexFlaggedPaths"],
        "manifestIntegrity": git["manifestIntegrity"],
        "evidenceRootConfigured": resolved_evidence is not None,
        "evidenceRootSafe": not evidence_configuration_errors,
        "gateSummary": {
            "required": len(gate_results),
            "passed": passed_count,
            "failed": len(gate_results) - passed_count,
        },
        "gates": gate_results,
        "guardrails": guardrails,
        "blockingReasons": list(dict.fromkeys(blocking_reasons)),
    }


def _human_output(result: Mapping[str, Any]) -> str:
    lines = [
        "Ultimate parity audit",
        f"  manifest valid:          {result.get('manifestValid')}",
        f"  parity eligible:         {result.get('parityEligible')}",
        f"  legacy removal allowed:  {result.get('legacyRemovalAllowed')}",
        f"  main replacement allowed:{result.get('mainReplacementAllowed')}",
        f"  release eligible:        {result.get('releaseEligible')}",
    ]
    summary = result.get("gateSummary")
    if isinstance(summary, dict):
        lines.append(
            "  gates: "
            f"{summary.get('passed', 0)}/{summary.get('required', 0)} passed"
        )
    reasons = result.get("blockingReasons", [])
    if reasons:
        lines.append("Blocking reasons:")
        lines.extend(f"  - {reason}" for reason in reasons)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Ultimate parity, evidence, and cutover audit."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repository_root(),
        help="Repository root; defaults to the checker repository.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path; defaults to docs/refactor/ultimate-parity.json.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help="External evidence root; otherwise use MTS_ULTIMATE_EVIDENCE_ROOT.",
    )
    parser.add_argument(
        "--validate-manifest",
        action="store_true",
        help="Validate structure and policy without evaluating release eligibility.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute command checks with argv arrays and shell=False.",
    )
    parser.add_argument(
        "--write-attestations",
        action="store_true",
        help="Write successful command attestations to the external evidence root.",
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        help="Limit displayed gate details; eligibility still evaluates every gate.",
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Output format.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else repo_root / "docs" / "refactor" / "ultimate-parity.json"
    )
    if args.write_attestations and not args.execute:
        parser.error("--write-attestations requires --execute")
    try:
        manifest = load_json(manifest_path)
    except ManifestError as exc:
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "manifestValid": False,
            "configurationErrors": [str(exc)],
            "releaseEligible": False,
            "blockingReasons": [str(exc)],
        }
        print(
            json.dumps(payload, indent=2)
            if args.format == "json"
            else _human_output(payload)
        )
        return 2

    errors = validate_manifest(manifest)
    if args.validate_manifest:
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "manifestValid": not errors,
            "configurationErrors": errors,
        }
        print(
            json.dumps(payload, indent=2)
            if args.format == "json"
            else (
                "Ultimate parity manifest is valid."
                if not errors
                else "\n".join(["Ultimate parity manifest is invalid:", *errors])
            )
        )
        return 0 if not errors else 2

    result = evaluate(
        repo_root=repo_root,
        manifest_path=manifest_path,
        evidence_root=args.evidence_root,
        execute=args.execute,
        write_attestations=args.write_attestations,
    )
    if args.gate and isinstance(result.get("gates"), list):
        selected = set(args.gate)
        result["gates"] = [
            gate for gate in result["gates"] if gate.get("id") in selected
        ]
    print(
        json.dumps(result, indent=2, ensure_ascii=False)
        if args.format == "json"
        else _human_output(result)
    )
    if not result.get("manifestValid") or result.get("configurationErrors"):
        return 2
    return 0 if result.get("releaseEligible") else 1


if __name__ == "__main__":
    raise SystemExit(main())
