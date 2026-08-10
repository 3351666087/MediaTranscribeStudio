"""Compare pinned local semantic models on frozen, human-adjudicated cases.

This benchmark is deliberately a *no-replace* diagnostic.  No-replace applies
to immutable benchmark and blind-review output targets, not to the active
production pointer: an external CAS promotion may replace production
immediately after a challenger wins blind review.  This tool never reads or
writes the production configuration and it never chooses a production winner.
Every model receives the same immutable document and candidate lattice for a
case; only the model identity changes.  The persisted benchmark report contains
hashes, counts, and exact-match booleans, never transcript text or provider
payloads.

The input manifest is intentionally small and explicit.  A case may inline its
``document``, ``lattice`` and ``baseline`` objects, or point at JSON files with
``documentPath``, ``latticePath`` and ``baselinePath``.  The baseline must carry
an explicit ``human``/``codex-manual``/``codex-agent`` adjudication source.

Example (Windows):

``python.exe tools/benchmark_real_multilingual_semantic_models.py \
  --cases D:\\mts-eval\\semantic-cases\\frozen.v1.json \
  --config-set-manifest D:\\mts-eval\\configs\\product-semantic-candidates-20260807\\config-set.manifest.json \
  --output D:\\mts-eval\\semantic-model-comparison\\run.v1.json``
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.errors import WorkerError  # noqa: E402
from backend.local_llm import (  # noqa: E402
    LocalLLMConfig,
    LocalLLMError,
    OllamaLocalProvider,
)
from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
    validate_strict_json,
)
from backend.semantic_candidate_lattice import (  # noqa: E402
    SEMANTIC_CANDIDATE_DOMAINS,
    validate_semantic_candidate_lattice,
)
from backend.semantic_composition import (  # noqa: E402
    SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION,
    SemanticCompositionError,
    SemanticJobArbitrationRunner,
    build_semantic_composition,
    validate_semantic_job_arbitration,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_TYPE = "real-multilingual-semantic-model-benchmark"
NO_REPLACE_POLICY = "no-replace"
BLIND_PACKET_ARTIFACT_TYPE = "anonymous-semantic-blind-review-packet"
BLIND_VAULT_ARTIFACT_TYPE = "anonymous-semantic-blind-review-identity-vault"
BLIND_MANIFEST_ARTIFACT_TYPE = "anonymous-semantic-blind-review-manifest"
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9._:/-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADJUDICATION_SOURCES = frozenset({"human", "codex-manual", "codex-agent"})
_CALIBRATION_DERIVATION = "manual-baseline-vs-pre-review-lattice-v1"
_CALIBRATION_ACTIONS = (
    "select",
    "request-default-challenger",
    "preservation-only",
)
_DEFAULT_CHALLENGER_REQUEST_KIND = {
    "speech-disposition": "speech-disposition-challenger",
    "speaker-cardinality-timeline": "timeline-challenger",
    "speaker-assignment": "speaker-assignment-challenger",
    "language-span": "open-set-lid",
    "asr-text": "provider-native-nbest",
}
_HIDDEN_CALIBRATION_KEYS = frozenset(
    {
        "semanticcalibrationtarget",
        "baselineprojectionsha256",
        "targetcandidateid",
        "acceptablecandidateids",
    }
)
_OLLAMA_RESPONSE_LIMIT = 16 * 1024 * 1024
_OLLAMA_PROCESS_NAMES = frozenset(
    {"ollama", "ollama.exe", "llama-server", "llama-server.exe"}
)
_FAILURE_TYPES = (
    LocalLLMError,
    SemanticCompositionError,
    WorkerError,
    ValueError,
    OSError,
)


class RealMultilingualSemanticBenchmarkError(ValueError):
    """Raised when frozen benchmark evidence is invalid."""


@dataclass(frozen=True)
class SemanticModelSpec:
    """One immutable Ollama model identity used by the benchmark."""

    model: str
    digest: str
    model_id: str | None = None
    config_path: str | None = None

    def __post_init__(self) -> None:
        model = _text(self.model, field="model", maximum=200)
        if _SAFE_MODEL_NAME.fullmatch(model) is None:
            raise RealMultilingualSemanticBenchmarkError(
                "model contains unsafe characters"
            )
        digest = _normalize_digest(self.digest, field="model digest")
        model_id = self.model_id
        if model_id is not None:
            model_id = _text(model_id, field="model id", maximum=200)
        config_path = self.config_path
        if config_path is not None:
            config_path = _text(config_path, field="model config path", maximum=500)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "config_path", config_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelId": self.model_id or self.model,
            "model": self.model,
            "expectedDigest": self.digest,
            "configPath": self.config_path,
        }


# A public alias makes the small value object convenient for callers that use
# the shorter name in scripts and tests.
ModelSpec = SemanticModelSpec


@dataclass(frozen=True)
class FrozenSemanticCase:
    """Validated case inputs; the text-bearing values stay out of reports."""

    case_id: str
    language: str
    document: dict[str, Any]
    lattice: dict[str, Any]
    baseline: dict[str, Any]
    semantic_calibration_target: dict[str, Any]
    document_sha256: str
    lattice_sha256: str
    baseline_sha256: str
    baseline_canonical_sha256: str
    baseline_audit_source: str
    case_sha256: str
    manifest_case_index: int

    def input_binding(self) -> dict[str, Any]:
        return {
            "caseId": self.case_id,
            "language": self.language,
            "documentSha256": self.document_sha256,
            "latticeSha256": self.lattice_sha256,
            "baselineSha256": self.baseline_sha256,
            "baselineCanonicalSha256": self.baseline_canonical_sha256,
            "baselineAuditSource": self.baseline_audit_source,
        }


@dataclass(frozen=True)
class FrozenSemanticCaseSet:
    """The complete frozen input set and its source manifest identity."""

    cases: tuple[FrozenSemanticCase, ...]
    manifest_sha256: str
    manifest_canonical_sha256: str
    manifest_path: str | None
    schema_version: str

    def binding(self) -> dict[str, Any]:
        rows = [case.input_binding() for case in self.cases]
        return {
            "caseCount": len(rows),
            "manifestSha256": self.manifest_sha256,
            "manifestCanonicalSha256": self.manifest_canonical_sha256,
            "documentSetSha256": canonical_json_sha256(
                [{"caseId": row["caseId"], "sha256": row["documentSha256"]} for row in rows]
            ),
            "latticeSetSha256": canonical_json_sha256(
                [{"caseId": row["caseId"], "sha256": row["latticeSha256"]} for row in rows]
            ),
            "baselineSetSha256": canonical_json_sha256(
                [{"caseId": row["caseId"], "sha256": row["baselineSha256"]} for row in rows]
            ),
            "casesSha256": canonical_json_sha256(rows),
            "languages": sorted({case.language for case in self.cases}),
        }


def _text(value: Any, *, field: str, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealMultilingualSemanticBenchmarkError(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise RealMultilingualSemanticBenchmarkError(f"{field} is too long")
    if any(ord(char) < 32 or ord(char) == 127 for char in result):
        raise RealMultilingualSemanticBenchmarkError(f"{field} contains control characters")
    return result


def _normalize_digest(value: Any, *, field: str) -> str:
    result = _text(value, field=field, maximum=80).casefold()
    if result.startswith("sha256:"):
        result = result.removeprefix("sha256:")
    if _SHA256.fullmatch(result) is None:
        raise RealMultilingualSemanticBenchmarkError(f"{field} must be SHA-256")
    return f"sha256:{result}"


def _json_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RealMultilingualSemanticBenchmarkError(f"{field} must be an object")
    try:
        validate_strict_json(value)
    except ValueError as exc:
        raise RealMultilingualSemanticBenchmarkError(f"{field} is not strict JSON") from exc
    return copy.deepcopy(dict(value))


def _load_json_file(path: Path, *, field: str) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise RealMultilingualSemanticBenchmarkError(f"{field} must be a regular file")
    try:
        value = read_json_strict(resolved)
    except (OSError, UnicodeError, ValueError, WorkerError) as exc:
        raise RealMultilingualSemanticBenchmarkError(f"{field} cannot be read") from exc
    return _json_object(value, field=field), sha256_file(resolved)


def _canonical_without_declared(value: Mapping[str, Any], key: str) -> str:
    body = dict(value)
    body.pop(key, None)
    return canonical_json_sha256(body)


def _resolve_object(
    raw_case: Mapping[str, Any],
    *,
    object_keys: Sequence[str],
    path_keys: Sequence[str],
    base_directory: Path,
    field: str,
) -> tuple[dict[str, Any], str, str | None]:
    for key in object_keys:
        if key not in raw_case:
            continue
        value = raw_case[key]
        if isinstance(value, Mapping):
            obj = _json_object(value, field=field)
            return obj, canonical_json_sha256(obj), None
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                path = base_directory / path
            obj, file_sha = _load_json_file(path, field=field)
            return obj, file_sha, str(path.expanduser().resolve(strict=True))
        raise RealMultilingualSemanticBenchmarkError(f"{field} must be an object or path")
    for key in path_keys:
        if key not in raw_case:
            continue
        value = raw_case[key]
        if not isinstance(value, str):
            raise RealMultilingualSemanticBenchmarkError(f"{field} path must be text")
        path = Path(value)
        if not path.is_absolute():
            path = base_directory / path
        obj, file_sha = _load_json_file(path, field=field)
        return obj, file_sha, str(path.expanduser().resolve(strict=True))
    raise RealMultilingualSemanticBenchmarkError(f"{field} is missing")


def _source_hash(document: Mapping[str, Any]) -> str:
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise RealMultilingualSemanticBenchmarkError("document.source is missing")
    value = source.get("sha256")
    return _normalize_digest(value, field="document.source.sha256").removeprefix("sha256:")


def _validate_document(document: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    value = _json_object(document, field="document")
    job_id = _text(value.get("jobId"), field="document.jobId", maximum=200)
    _text(value.get("documentId"), field="document.documentId", maximum=200)
    if not isinstance(value.get("segments"), list):
        raise RealMultilingualSemanticBenchmarkError("document.segments must be an array")
    source_sha = _source_hash(value)
    duration = value.get("source", {}).get("durationMs")
    if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
        raise RealMultilingualSemanticBenchmarkError("document.source.durationMs is invalid")
    document_sha = canonical_json_sha256(value)
    declared = value.get("canonicalSha256")
    if declared is not None and _normalize_digest(declared, field="document.canonicalSha256") != f"sha256:{document_sha}":
        raise RealMultilingualSemanticBenchmarkError("document canonical SHA-256 does not match")
    return value, document_sha, job_id


def _validate_lattice(
    lattice: Mapping[str, Any],
    *,
    document: Mapping[str, Any],
    document_sha256: str,
) -> dict[str, Any]:
    value = _json_object(lattice, field="candidate lattice")
    try:
        validated = validate_semantic_candidate_lattice(
            value,
            expected_source_media_sha256=_source_hash(document),
            expected_transcript_sha256=document_sha256,
        )
    except Exception as exc:
        raise RealMultilingualSemanticBenchmarkError("candidate lattice is invalid or rebound") from exc
    return copy.deepcopy(dict(validated))


def _baseline_audit_source(case: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    candidates: list[Any] = [
        case.get("baselineAuditSource"),
        case.get("adjudicationSource"),
        baseline.get("adjudicationSource"),
    ]
    for key in ("adjudication", "review", "audit"):
        value = baseline.get(key)
        if isinstance(value, Mapping):
            candidates.extend(value.get(name) for name in ("source", "auditSource", "reviewerSource"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip().casefold() in _ADJUDICATION_SOURCES:
            return candidate.strip().casefold()
    raise RealMultilingualSemanticBenchmarkError(
        "final-adjudicated baseline must declare human or Codex adjudication source"
    )


def _segment_text(segment: Mapping[str, Any]) -> str:
    for key in ("finalText", "normalizedText", "text"):
        value = segment.get(key)
        if isinstance(value, str):
            return value
    raise RealMultilingualSemanticBenchmarkError("baseline segment has no final text")


def _validate_baseline(
    baseline: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    document: Mapping[str, Any],
    document_sha256: str,
    lattice_sha256: str,
) -> tuple[dict[str, Any], str]:
    value = _json_object(baseline, field="final-adjudicated baseline")
    artifact_type = value.get("artifactType")
    if artifact_type not in {None, "final-adjudicated-transcript", "final-adjudicated-transcript-v1"}:
        raise RealMultilingualSemanticBenchmarkError("baseline artifact type is not final-adjudicated")
    status = value.get("status")
    if status is not None and status not in {"adjudication-complete", "completed", "final"}:
        raise RealMultilingualSemanticBenchmarkError("baseline is not adjudication-complete")
    if value.get("disposition") not in {None, "transcribable-speech", "no-transcribable-speech"}:
        raise RealMultilingualSemanticBenchmarkError("baseline disposition is invalid")
    audit_source = _baseline_audit_source(case, value)
    input_binding = value.get("input")
    if isinstance(input_binding, Mapping):
        transcript_hash = input_binding.get("transcriptDocumentSha256")
        if transcript_hash is not None and _normalize_digest(transcript_hash, field="baseline.input.transcriptDocumentSha256") != f"sha256:{document_sha256}":
            raise RealMultilingualSemanticBenchmarkError("baseline is rebound to another document")
        lattice_hash = input_binding.get("candidateLatticeSha256") or input_binding.get("inputLatticeSha256")
        if lattice_hash is not None and _normalize_digest(lattice_hash, field="baseline.input.latticeSha256") != f"sha256:{lattice_sha256}":
            raise RealMultilingualSemanticBenchmarkError("baseline is rebound to another lattice")
    segments = value.get("segments")
    if not isinstance(segments, list):
        raise RealMultilingualSemanticBenchmarkError("baseline.segments must be an array")
    seen: set[str] = set()
    duration = int(document["source"]["durationMs"])
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise RealMultilingualSemanticBenchmarkError("baseline segment must be an object")
        segment_id = _text(segment.get("id"), field=f"baseline.segments[{index}].id", maximum=200)
        if segment_id in seen:
            raise RealMultilingualSemanticBenchmarkError("baseline segment IDs must be unique")
        seen.add(segment_id)
        start = segment.get("startMs")
        end = segment.get("endMs")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0 or isinstance(end, bool) or not isinstance(end, int) or end <= start or end > duration:
            raise RealMultilingualSemanticBenchmarkError("baseline segment timing is invalid")
        _text(segment.get("speakerId"), field=f"baseline.segments[{index}].speakerId", maximum=200)
        _text(segment.get("language"), field=f"baseline.segments[{index}].language", maximum=100)
        _segment_text(segment)
    baseline_sha = canonical_json_sha256(value)
    declared = value.get("canonicalSha256")
    if declared is not None and _normalize_digest(declared, field="baseline.canonicalSha256") != f"sha256:{baseline_sha}":
        raise RealMultilingualSemanticBenchmarkError("baseline canonical SHA-256 does not match")
    return value, audit_source


def _semantic_lattice_groups(
    lattice: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    groups: list[tuple[str, Mapping[str, Any]]] = []
    for raw_domain in lattice.get("domains", []):
        if not isinstance(raw_domain, Mapping):
            continue
        domain = str(raw_domain.get("domain") or "")
        if domain not in SEMANTIC_CANDIDATE_DOMAINS:
            raise RealMultilingualSemanticBenchmarkError(
                "semantic calibration lattice contains an unsupported domain"
            )
        raw_groups = raw_domain.get("groups")
        if not isinstance(raw_groups, list):
            continue
        groups.extend(
            (domain, raw_group)
            for raw_group in raw_groups
            if isinstance(raw_group, Mapping)
        )
    return groups


def _semantic_candidate_projection(
    domain: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if domain == "speech-disposition":
        return {"classification": payload.get("classification")}
    if domain == "speaker-cardinality-timeline":
        raw_turns = payload.get("turns")
        turns = (
            [
                {
                    "startMs": turn.get("startMs"),
                    "endMs": turn.get("endMs"),
                    "speakerId": turn.get("speakerId"),
                    "overlap": bool(
                        turn.get("overlap", turn.get("overlapping", False))
                    ),
                }
                for turn in raw_turns
                if isinstance(turn, Mapping)
            ]
            if isinstance(raw_turns, list)
            else []
        )
        turns.sort(
            key=lambda turn: (
                int(turn["startMs"]),
                int(turn["endMs"]),
                str(turn["speakerId"]),
            )
        )
        return {
            "speakerCount": payload.get("speakerCount"),
            "speakerIds": copy.deepcopy(payload.get("speakerIds")),
            "turns": turns,
        }
    if domain == "speaker-assignment":
        return {
            "segmentId": payload.get("segmentId"),
            "speakerId": payload.get("speakerId"),
        }
    if domain == "language-span":
        return {
            "segmentId": payload.get("segmentId"),
            "language": payload.get("language"),
        }
    if domain == "asr-text":
        return {
            "segmentId": payload.get("segmentId"),
            "text": payload.get("text"),
        }
    raise RealMultilingualSemanticBenchmarkError(
        "semantic calibration target contains an unsupported domain"
    )


def _speaker_id_order(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"speaker-([1-9][0-9]*)", value)
    return (int(match.group(1)), value) if match is not None else (sys.maxsize, value)


def _baseline_timeline_projection(
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    timeline = baseline.get("timeline")
    raw_turns = timeline.get("turns") if isinstance(timeline, Mapping) else None
    if not isinstance(raw_turns, list) or not raw_turns:
        raise RealMultilingualSemanticBenchmarkError(
            "semantic calibration baseline timeline must contain turns"
        )
    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, Mapping):
            raise RealMultilingualSemanticBenchmarkError(
                f"semantic calibration baseline timeline turn {index} is invalid"
            )
        start = raw.get("startMs")
        end = raw.get("endMs")
        overlap = bool(raw.get("overlap", raw.get("overlapping", False)))
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end <= start
        ):
            raise RealMultilingualSemanticBenchmarkError(
                "semantic calibration baseline timeline turn is invalid"
            )
        turns.append(
            {
                "startMs": start,
                "endMs": end,
                "speakerId": _text(
                    raw.get("speakerId"),
                    field="semantic calibration baseline timeline speakerId",
                    maximum=200,
                ),
                "overlap": overlap,
            }
        )
    turns.sort(
        key=lambda item: (
            item["startMs"],
            item["endMs"],
            item["speakerId"],
        )
    )
    speaker_ids = sorted(
        {str(turn["speakerId"]) for turn in turns},
        key=_speaker_id_order,
    )
    return {
        "speakerCount": len(speaker_ids),
        "speakerIds": speaker_ids,
        "turns": turns,
    }


def _semantic_baseline_projection(
    domain: str,
    group: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if domain == "speech-disposition":
        return {"classification": baseline.get("disposition")}
    if domain == "speaker-cardinality-timeline":
        return _baseline_timeline_projection(baseline)

    scope_id = str(group.get("scopeId") or "")
    if not scope_id.startswith("segment:"):
        raise RealMultilingualSemanticBenchmarkError(
            "semantic calibration lattice group has an invalid segment scope"
        )
    segment_id = scope_id.removeprefix("segment:")
    raw_segments = baseline.get("segments")
    baseline_segments = {
        str(segment.get("id")): segment
        for segment in raw_segments
        if isinstance(raw_segments, list) and isinstance(segment, Mapping)
    }
    segment = baseline_segments.get(segment_id)
    if segment is None:
        raise RealMultilingualSemanticBenchmarkError(
            "semantic calibration baseline omits a lattice segment"
        )
    if domain == "speaker-assignment":
        return {"segmentId": segment_id, "speakerId": segment.get("speakerId")}
    if domain == "language-span":
        return {"segmentId": segment_id, "language": segment.get("language")}
    if domain == "asr-text":
        return {"segmentId": segment_id, "text": _segment_text(segment)}
    raise RealMultilingualSemanticBenchmarkError(
        "semantic calibration target contains an unsupported domain"
    )


def _validate_semantic_calibration_target(
    raw_target: Any,
    *,
    lattice: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    target = _json_object(raw_target, field="semanticCalibrationTarget")
    required = {
        "schemaVersion",
        "derivation",
        "groups",
        "counts",
        "canonicalSha256",
    }
    if set(target) != required:
        raise RealMultilingualSemanticBenchmarkError(
            "semanticCalibrationTarget fields do not match the contract"
        )
    if target.get("schemaVersion") != SCHEMA_VERSION:
        raise RealMultilingualSemanticBenchmarkError(
            "semanticCalibrationTarget schema version is unsupported"
        )
    if target.get("derivation") != _CALIBRATION_DERIVATION:
        raise RealMultilingualSemanticBenchmarkError(
            "semanticCalibrationTarget derivation is unsupported"
        )
    target_canonical_sha = _canonical_without_declared(
        target,
        "canonicalSha256",
    )
    if _normalize_digest(
        target.get("canonicalSha256"),
        field="semanticCalibrationTarget.canonicalSha256",
    ) != f"sha256:{target_canonical_sha}":
        raise RealMultilingualSemanticBenchmarkError(
            "semanticCalibrationTarget canonical SHA-256 does not match"
        )

    raw_groups = target.get("groups")
    if not isinstance(raw_groups, list):
        raise RealMultilingualSemanticBenchmarkError(
            "semanticCalibrationTarget.groups must be an array"
        )
    lattice_groups = _semantic_lattice_groups(lattice)
    if len(raw_groups) != len(lattice_groups):
        raise RealMultilingualSemanticBenchmarkError(
            "semanticCalibrationTarget group coverage does not match the lattice"
        )
    action_counts: Counter[str] = Counter()
    normalized_groups: list[dict[str, Any]] = []
    group_fields = {
        "domain",
        "groupId",
        "scopeId",
        "expectedAction",
        "targetCandidateId",
        "acceptableCandidateIds",
        "currentCandidateId",
        "baselineProjectionSha256",
        "eligibleAlternativeCount",
    }
    for index, (raw_group_target, (domain, lattice_group)) in enumerate(
        zip(raw_groups, lattice_groups, strict=True)
    ):
        field = f"semanticCalibrationTarget.groups[{index}]"
        if not isinstance(raw_group_target, Mapping) or set(raw_group_target) != group_fields:
            raise RealMultilingualSemanticBenchmarkError(
                f"{field} fields do not match the contract"
            )
        group_target = dict(raw_group_target)
        expected_identity = (
            domain,
            lattice_group.get("groupId"),
            lattice_group.get("scopeId"),
            lattice_group.get("currentCandidateId"),
        )
        actual_identity = (
            group_target.get("domain"),
            group_target.get("groupId"),
            group_target.get("scopeId"),
            group_target.get("currentCandidateId"),
        )
        if actual_identity != expected_identity:
            raise RealMultilingualSemanticBenchmarkError(
                f"{field} is rebound to another lattice group"
            )
        expected_action = group_target.get("expectedAction")
        if expected_action not in _CALIBRATION_ACTIONS:
            raise RealMultilingualSemanticBenchmarkError(
                f"{field}.expectedAction is unsupported"
            )
        baseline_projection_sha = canonical_json_sha256(
            _semantic_baseline_projection(domain, lattice_group, baseline)
        )
        if _normalize_digest(
            group_target.get("baselineProjectionSha256"),
            field=f"{field}.baselineProjectionSha256",
        ) != f"sha256:{baseline_projection_sha}":
            raise RealMultilingualSemanticBenchmarkError(
                f"{field} baseline projection SHA-256 does not match"
            )

        raw_candidates = lattice_group.get("candidates")
        if not isinstance(raw_candidates, list):
            raise RealMultilingualSemanticBenchmarkError(
                f"{field} lattice candidates are invalid"
            )
        candidates = {
            str(candidate.get("candidateId")): candidate
            for candidate in raw_candidates
            if isinstance(candidate, Mapping)
        }
        if len(candidates) != len(raw_candidates):
            raise RealMultilingualSemanticBenchmarkError(
                f"{field} lattice candidates are invalid"
            )
        current_candidate_id = str(lattice_group["currentCandidateId"])
        current_candidate = candidates[current_candidate_id]
        eligible_alternatives = sorted(
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate_id != current_candidate_id
            and candidate.get("selectionEligible") is True
        )
        eligible_count = group_target.get("eligibleAlternativeCount")
        if (
            isinstance(eligible_count, bool)
            or not isinstance(eligible_count, int)
            or eligible_count != len(eligible_alternatives)
        ):
            raise RealMultilingualSemanticBenchmarkError(
                f"{field}.eligibleAlternativeCount does not match the lattice"
            )
        matching_ids = sorted(
            candidate_id
            for candidate_id, candidate in candidates.items()
            if candidate.get("selectionEligible") is True
            and isinstance(candidate.get("payload"), Mapping)
            and canonical_json_sha256(
                _semantic_candidate_projection(domain, candidate["payload"])
            )
            == baseline_projection_sha
        )
        acceptable_ids = group_target.get("acceptableCandidateIds")
        if (
            not isinstance(acceptable_ids, list)
            or any(not isinstance(item, str) for item in acceptable_ids)
            or acceptable_ids != matching_ids
        ):
            raise RealMultilingualSemanticBenchmarkError(
                f"{field}.acceptableCandidateIds are not exact eligible matches"
            )
        current_projection_matches = (
            isinstance(current_candidate.get("payload"), Mapping)
            and canonical_json_sha256(
                _semantic_candidate_projection(domain, current_candidate["payload"])
            )
            == baseline_projection_sha
        )
        derived_action: str
        if matching_ids and eligible_alternatives:
            derived_action = "select"
        elif (
            matching_ids == [current_candidate_id]
            and not eligible_alternatives
        ):
            derived_action = "preservation-only"
        elif not matching_ids and not current_projection_matches:
            derived_action = "request-default-challenger"
        else:
            raise RealMultilingualSemanticBenchmarkError(
                f"{field} cannot be classified by the calibration contract"
            )
        if expected_action != derived_action:
            raise RealMultilingualSemanticBenchmarkError(
                f"{field}.expectedAction does not match its reachable outcome"
            )
        expected_target_id = (
            current_candidate_id
            if current_candidate_id in matching_ids
            else matching_ids[0]
            if matching_ids
            else None
        )
        if group_target.get("targetCandidateId") != expected_target_id:
            raise RealMultilingualSemanticBenchmarkError(
                f"{field}.targetCandidateId is not the canonical reachable target"
            )
        action_counts[expected_action] += 1
        normalized_groups.append(copy.deepcopy(group_target))

    counts = target.get("counts")
    expected_counts = {
        action: action_counts[action] for action in _CALIBRATION_ACTIONS
    }
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(_CALIBRATION_ACTIONS)
        or any(
            isinstance(counts[action], bool)
            or not isinstance(counts[action], int)
            or counts[action] < 0
            for action in _CALIBRATION_ACTIONS
        )
        or dict(counts) != expected_counts
    ):
        raise RealMultilingualSemanticBenchmarkError(
            "semanticCalibrationTarget counts do not match its groups"
        )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "derivation": _CALIBRATION_DERIVATION,
        "groups": normalized_groups,
        "counts": expected_counts,
        "canonicalSha256": target_canonical_sha,
    }


def load_frozen_semantic_cases(
    path: Path,
) -> FrozenSemanticCaseSet:
    """Load and validate one immutable multilingual case manifest."""

    manifest_path = path.expanduser().resolve(strict=True)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RealMultilingualSemanticBenchmarkError("case manifest must be a regular file")
    manifest, manifest_file_sha = _load_json_file(manifest_path, field="case manifest")
    raw_cases = manifest.get("cases")
    if isinstance(raw_cases, Mapping):
        raw_cases = [raw_cases]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RealMultilingualSemanticBenchmarkError("case manifest must contain cases")
    declared_manifest = manifest.get("canonicalSha256")
    manifest_canonical = _canonical_without_declared(manifest, "canonicalSha256")
    if declared_manifest is not None and _normalize_digest(declared_manifest, field="case manifest canonicalSha256") != f"sha256:{manifest_canonical}":
        raise RealMultilingualSemanticBenchmarkError("case manifest canonical SHA-256 does not match")
    cases: list[FrozenSemanticCase] = []
    seen_ids: set[str] = set()
    base = manifest_path.parent
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise RealMultilingualSemanticBenchmarkError(f"cases[{index}] must be an object")
        case = dict(raw)
        case_id = _text(case.get("caseId") or case.get("id"), field=f"cases[{index}].caseId", maximum=200)
        if case_id in seen_ids:
            raise RealMultilingualSemanticBenchmarkError("case IDs must be unique")
        seen_ids.add(case_id)
        document, document_source_hash, _ = _resolve_object(
            case,
            object_keys=("document", "transcript"),
            path_keys=("documentPath", "transcriptPath"),
            base_directory=base,
            field=f"cases[{index}].document",
        )
        document, document_sha, _job_id = _validate_document(document)
        declared_document_sha = case.get("documentSha256") or case.get("documentCanonicalSha256")
        if declared_document_sha is not None and _normalize_digest(declared_document_sha, field=f"cases[{index}].documentSha256") != f"sha256:{document_sha}":
            raise RealMultilingualSemanticBenchmarkError("case document hash does not match")
        lattice, lattice_source_hash, _ = _resolve_object(
            case,
            object_keys=("lattice", "candidateLattice"),
            path_keys=("latticePath", "candidateLatticePath"),
            base_directory=base,
            field=f"cases[{index}].lattice",
        )
        lattice = _validate_lattice(lattice, document=document, document_sha256=document_sha)
        lattice_sha = str(lattice["latticeSha256"])
        declared_lattice_sha = case.get("latticeSha256")
        if declared_lattice_sha is not None and _normalize_digest(declared_lattice_sha, field=f"cases[{index}].latticeSha256") != f"sha256:{lattice_sha}":
            raise RealMultilingualSemanticBenchmarkError("case lattice hash does not match")
        baseline, baseline_source_hash, baseline_path = _resolve_object(
            case,
            object_keys=("baseline", "finalAdjudicatedBaseline", "finalAdjudicated"),
            path_keys=("baselinePath", "finalAdjudicatedPath"),
            base_directory=base,
            field=f"cases[{index}].baseline",
        )
        baseline, audit_source = _validate_baseline(
            baseline,
            case=case,
            document=document,
            document_sha256=document_sha,
            lattice_sha256=lattice_sha,
        )
        semantic_calibration_target = _validate_semantic_calibration_target(
            case.get("semanticCalibrationTarget"),
            lattice=lattice,
            baseline=baseline,
        )
        baseline_canonical_sha = canonical_json_sha256(baseline)
        baseline_sha = baseline_source_hash
        if baseline_path is None:
            baseline_sha = baseline_canonical_sha
        declared_baseline_sha = case.get("baselineSha256")
        if declared_baseline_sha is not None:
            normalized_declared = _normalize_digest(
                declared_baseline_sha,
                field=f"cases[{index}].baselineSha256",
            )
            if normalized_declared not in {
                f"sha256:{baseline_sha}",
                f"sha256:{baseline_canonical_sha}",
            }:
                raise RealMultilingualSemanticBenchmarkError(
                    "case baseline hash does not match"
                )
        language = case.get("language") or document.get("language")
        if not isinstance(language, str) or not language.strip():
            languages = {str(segment.get("language")) for segment in baseline.get("segments", []) if isinstance(segment, Mapping) and isinstance(segment.get("language"), str)}
            language = sorted(languages)[0] if len(languages) == 1 else None
        language = _text(language, field=f"cases[{index}].language", maximum=100)
        case_body = dict(case)
        case_body.pop("caseSha256", None)
        case_sha = canonical_json_sha256(case_body)
        declared_case_sha = case.get("caseSha256")
        if declared_case_sha is not None and _normalize_digest(declared_case_sha, field=f"cases[{index}].caseSha256") != f"sha256:{case_sha}":
            raise RealMultilingualSemanticBenchmarkError("case canonical SHA-256 does not match")
        cases.append(
            FrozenSemanticCase(
                case_id=case_id,
                language=language,
                document=document,
                lattice=lattice,
                baseline=baseline,
                semantic_calibration_target=semantic_calibration_target,
                document_sha256=document_sha,
                lattice_sha256=lattice_sha,
                baseline_sha256=baseline_sha,
                baseline_canonical_sha256=baseline_canonical_sha,
                baseline_audit_source=audit_source,
                case_sha256=case_sha,
                manifest_case_index=index,
            )
        )
    cases.sort(key=lambda item: (item.language, item.case_id))
    return FrozenSemanticCaseSet(
        cases=tuple(cases),
        manifest_sha256=manifest_file_sha,
        manifest_canonical_sha256=manifest_canonical,
        manifest_path=str(manifest_path),
        schema_version=str(manifest.get("schemaVersion") or SCHEMA_VERSION),
    )


def load_model_specs_from_manifest(path: Path) -> tuple[SemanticModelSpec, ...]:
    """Read pinned model identities from the generated config-set manifest."""

    manifest, _ = _load_json_file(path, field="model config-set manifest")
    configs = manifest.get("configs")
    if not isinstance(configs, list) or not configs:
        raise RealMultilingualSemanticBenchmarkError("model config-set manifest has no configs")
    output: list[SemanticModelSpec] = []
    for index, raw in enumerate(configs):
        if not isinstance(raw, Mapping):
            raise RealMultilingualSemanticBenchmarkError(f"model configs[{index}] must be an object")
        model = raw.get("model")
        digest = raw.get("digest") or raw.get("expectedDigest")
        output.append(
            SemanticModelSpec(
                model=model,
                digest=digest,
                model_id=raw.get("modelId"),
                config_path=raw.get("configPath"),
            )
        )
    return _unique_model_specs(output)


def parse_model_specs(values: Sequence[str]) -> tuple[SemanticModelSpec, ...]:
    """Parse repeated ``NAME=DIGEST`` or ``NAME@DIGEST`` CLI values."""

    output: list[SemanticModelSpec] = []
    for value in values:
        text = _text(value, field="--model", maximum=400)
        if "=" in text:
            model, digest = text.split("=", 1)
        elif "@sha256:" in text:
            model, digest = text.split("@", 1)
        else:
            raise RealMultilingualSemanticBenchmarkError("--model must be NAME=SHA256")
        output.append(SemanticModelSpec(model=model, digest=digest))
    return _unique_model_specs(output)


def _unique_model_specs(specs: Sequence[SemanticModelSpec]) -> tuple[SemanticModelSpec, ...]:
    if not specs:
        raise RealMultilingualSemanticBenchmarkError("at least one semantic model is required")
    seen: set[str] = set()
    for spec in specs:
        key = spec.model.casefold()
        if key in seen:
            raise RealMultilingualSemanticBenchmarkError("semantic model names must be unique")
        seen.add(key)
    return tuple(specs)


ResourceProbe = Callable[[SemanticModelSpec, str], Mapping[str, Any]]


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            "redirects are disabled for local resource evidence",
            headers,
            fp,
        )


def _assert_loopback_endpoint(endpoint: str) -> None:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise RealMultilingualSemanticBenchmarkError(
            "resource evidence endpoint must use loopback HTTP"
        )
    hostname = parsed.hostname.casefold()
    if hostname == "localhost":
        return
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as exc:
        raise RealMultilingualSemanticBenchmarkError(
            "resource evidence endpoint must be loopback-only"
        ) from exc
    if not address.is_loopback:
        raise RealMultilingualSemanticBenchmarkError(
            "resource evidence endpoint must be loopback-only"
        )


def _ollama_process_state() -> dict[str, Any]:
    whole_gpu_state = _nvidia_whole_gpu_memory()
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return {
            "available": False,
            "failureCode": "PSUTIL_UNAVAILABLE",
            "ollamaPids": [],
            "wholeGpuMemory": whole_gpu_state,
        }
    try:
        host_memory = psutil.virtual_memory()
        benchmark_memory = psutil.Process(os.getpid()).memory_info()
        benchmark_rss = int(benchmark_memory.rss)
        benchmark_peak_working_set = _non_negative_int(
            getattr(benchmark_memory, "peak_wset", None)
        )
        pids: list[int] = []
        names: set[str] = set()
        aggregate_rss = 0
        aggregate_peak_working_set = 0
        peak_working_set_available = True
        for process in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                name = str(process.info.get("name") or "").casefold()
                if name not in _OLLAMA_PROCESS_NAMES:
                    continue
                pid = int(process.info.get("pid"))
                memory_info = process.info.get("memory_info")
                aggregate_rss += int(getattr(memory_info, "rss", 0))
                process_peak = _non_negative_int(
                    getattr(memory_info, "peak_wset", None)
                )
                if process_peak is None:
                    peak_working_set_available = False
                else:
                    aggregate_peak_working_set += process_peak
                pids.append(pid)
                names.add(name)
            except (psutil.Error, AttributeError, OSError, TypeError, ValueError):
                continue
    except (psutil.Error, AttributeError, OSError, TypeError, ValueError):
        return {
            "available": False,
            "failureCode": "PROCESS_MEMORY_UNAVAILABLE",
            "ollamaPids": [],
            "wholeGpuMemory": whole_gpu_state,
        }
    gpu_state = _nvidia_process_memory(frozenset(pids))
    return {
        "available": True,
        "benchmarkProcessRssBytes": benchmark_rss,
        "benchmarkProcessPeakWorkingSetBytes": benchmark_peak_working_set,
        "ollamaProcessCount": len(pids),
        "ollamaPids": sorted(pids),
        "ollamaAggregateRssBytes": aggregate_rss,
        "ollamaAggregatePeakWorkingSetBytes": (
            aggregate_peak_working_set
            if peak_working_set_available and pids
            else None
        ),
        "observedProcessNames": sorted(names),
        "totalPhysicalMemoryBytes": int(host_memory.total),
        "availablePhysicalMemoryBytes": int(host_memory.available),
        "usedPhysicalMemoryBytes": int(host_memory.used),
        "gpuProcessMemory": gpu_state,
        "wholeGpuMemory": whole_gpu_state,
    }


def _nvidia_process_memory(ollama_pids: frozenset[int]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=5.0,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "available": False,
            "failureCode": "NVIDIA_SMI_UNAVAILABLE",
            "ollamaAggregateVramBytes": None,
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "failureCode": "NVIDIA_SMI_FAILED",
            "ollamaAggregateVramBytes": None,
        }
    rows: list[dict[str, int]] = []
    nonempty_lines = 0
    for raw_line in completed.stdout.splitlines():
        if raw_line.strip():
            nonempty_lines += 1
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used_mib = int(parts[1])
        except ValueError:
            continue
        if pid in ollama_pids and used_mib >= 0:
            rows.append({"pid": pid, "usedVramBytes": used_mib * 1024 * 1024})
    if nonempty_lines and not rows and ollama_pids:
        return {
            "available": False,
            "failureCode": "NVIDIA_SMI_PROCESS_MEMORY_UNPARSEABLE",
            "ollamaAggregateVramBytes": None,
        }
    return {
        "available": True,
        "measurement": "nvidia-smi-compute-process-memory-v1",
        "matchedProcessCount": len(rows),
        "ollamaAggregateVramBytes": sum(row["usedVramBytes"] for row in rows),
        "processes": rows,
    }


def _nvidia_whole_gpu_memory() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=5.0,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return {
            "available": False,
            "failureCode": "NVIDIA_SMI_WHOLE_GPU_UNAVAILABLE",
            "aggregateUsedVramBytes": None,
            "aggregateFreeVramBytes": None,
            "includesOtherProcesses": True,
            "diagnosticOnly": True,
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "failureCode": "NVIDIA_SMI_WHOLE_GPU_FAILED",
            "aggregateUsedVramBytes": None,
            "aggregateFreeVramBytes": None,
            "includesOtherProcesses": True,
            "diagnosticOnly": True,
        }
    rows: list[dict[str, int]] = []
    nonempty_lines = 0
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        nonempty_lines += 1
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_index = int(parts[0])
            used_mib = int(parts[1])
            free_mib = int(parts[2])
        except ValueError:
            continue
        if gpu_index < 0 or used_mib < 0 or free_mib < 0:
            continue
        rows.append(
            {
                "gpuIndex": gpu_index,
                "usedVramBytes": used_mib * 1024 * 1024,
                "freeVramBytes": free_mib * 1024 * 1024,
            }
        )
    if not rows or len(rows) != nonempty_lines:
        return {
            "available": False,
            "failureCode": "NVIDIA_SMI_WHOLE_GPU_MEMORY_UNPARSEABLE",
            "aggregateUsedVramBytes": None,
            "aggregateFreeVramBytes": None,
            "includesOtherProcesses": True,
            "diagnosticOnly": True,
        }
    rows.sort(key=lambda row: row["gpuIndex"])
    return {
        "available": True,
        "measurement": "nvidia-smi-whole-gpu-memory-v1",
        "aggregateUsedVramBytes": sum(row["usedVramBytes"] for row in rows),
        "aggregateFreeVramBytes": sum(row["freeVramBytes"] for row in rows),
        "includesOtherProcesses": True,
        "diagnosticOnly": True,
        "devices": rows,
    }


def _ollama_residency(endpoint: str, model: str) -> dict[str, Any]:
    _assert_loopback_endpoint(endpoint)
    url = endpoint.rstrip("/") + "/api/ps"
    _assert_loopback_endpoint(url)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    opener = urllib.request.build_opener(_RejectRedirectHandler())
    try:
        with opener.open(request, timeout=10.0) as response:
            body = response.read(_OLLAMA_RESPONSE_LIMIT + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise RealMultilingualSemanticBenchmarkError(
            "Ollama residency evidence is unavailable"
        ) from exc
    if len(body) > _OLLAMA_RESPONSE_LIMIT:
        raise RealMultilingualSemanticBenchmarkError(
            "Ollama residency evidence exceeds the response limit"
        )
    try:
        value = json.loads(body.decode("utf-8", errors="strict"))
        validate_strict_json(value)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RealMultilingualSemanticBenchmarkError(
            "Ollama residency evidence is invalid"
        ) from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("models"), list):
        raise RealMultilingualSemanticBenchmarkError(
            "Ollama residency evidence has no model list"
        )
    selected: list[dict[str, Any]] = []
    for item in value["models"]:
        if not isinstance(item, Mapping) or model not in (
            item.get("name"),
            item.get("model"),
        ):
            continue
        selected.append(
            {
                "model": model,
                "digest": _resource_digest(item.get("digest")),
                "sizeBytes": _non_negative_int(item.get("size")),
                "sizeVramBytes": _non_negative_int(item.get("size_vram")),
            }
        )
    return {
        "ollamaApiAvailable": True,
        "selectedModelLoaded": bool(selected),
        "selectedModel": selected,
    }


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _resource_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if not normalized.startswith("sha256:"):
        normalized = f"sha256:{normalized}"
    return normalized if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized) else None


def _resource_snapshot(endpoint: str, model: str, *, phase: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "phase": phase,
        "capturedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "process": _ollama_process_state(),
    }
    try:
        snapshot.update(_ollama_residency(endpoint, model))
    except RealMultilingualSemanticBenchmarkError:
        snapshot.update(
            {
                "ollamaApiAvailable": False,
                "selectedModelLoaded": None,
                "selectedModel": [],
                "failureCode": "OLLAMA_RESIDENCY_UNAVAILABLE",
            }
        )
    return snapshot


def _default_resource_probe(endpoint: str, spec: SemanticModelSpec, phase: str) -> dict[str, Any]:
    snapshot = _resource_snapshot(endpoint, spec.model, phase=phase)
    if phase != "after-release":
        return snapshot
    for _ in range(20):
        if (
            snapshot.get("ollamaApiAvailable") is True
            and snapshot.get("selectedModelLoaded") is False
        ):
            break
        time.sleep(0.25)
        snapshot = _resource_snapshot(endpoint, spec.model, phase=phase)
    return snapshot


def _unavailable_resource_probe(spec: SemanticModelSpec, phase: str) -> dict[str, Any]:
    del spec
    return {
        "phase": phase,
        "capturedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ollamaApiAvailable": False,
        "selectedModelLoaded": None,
        "selectedModel": [],
        "process": {
            "available": False,
            "failureCode": "CUSTOM_PROVIDER_RESOURCE_PROBE_NOT_CONFIGURED",
            "ollamaPids": [],
        },
        "failureCode": "CUSTOM_PROVIDER_RESOURCE_PROBE_NOT_CONFIGURED",
    }


def _probe_resource(
    probe: ResourceProbe,
    spec: SemanticModelSpec,
    phase: str,
) -> dict[str, Any]:
    try:
        value = _json_object(probe(spec, phase), field="resource snapshot")
    except Exception as exc:  # pragma: no cover - probe boundary
        return {
            "phase": phase,
            "capturedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ollamaApiAvailable": False,
            "selectedModelLoaded": None,
            "selectedModel": [],
            "process": {"available": False, "failureCode": type(exc).__name__},
            "failureCode": "RESOURCE_PROBE_FAILED",
        }
    value["phase"] = phase
    return value


class _PeriodicResourceSampler:
    def __init__(
        self,
        probe: ResourceProbe,
        spec: SemanticModelSpec,
        *,
        interval_seconds: float,
    ) -> None:
        self.probe = probe
        self.spec = spec
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshots: list[dict[str, Any]] = []
        self._sample_index = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"semantic-resource-sampler-{spec.model_id or spec.model}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_index += 1
            snapshot = _probe_resource(
                self.probe,
                self.spec,
                f"periodic:{self._sample_index}",
            )
            with self._lock:
                self._snapshots.append(snapshot)

    def stop(self) -> tuple[list[dict[str, Any]], bool]:
        self._stop.set()
        self._thread.join(timeout=20.0)
        with self._lock:
            snapshots = copy.deepcopy(self._snapshots)
        return snapshots, not self._thread.is_alive()


def _resource_summary(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    expected_digest: str,
    sample_interval_seconds: float,
    sampler_stopped: bool,
) -> dict[str, Any]:
    model_bytes: list[int] = []
    model_vram: list[int] = []
    ollama_rss: list[int] = []
    benchmark_rss: list[int] = []
    process_vram: list[int] = []
    ollama_peak_working_set: list[int] = []
    benchmark_peak_working_set: list[int] = []
    available_memory: list[int] = []
    whole_gpu_used: list[int] = []
    whole_gpu_free: list[int] = []
    observed_digests: set[str] = set()
    for snapshot in snapshots:
        selected = snapshot.get("selectedModel")
        if snapshot.get("ollamaApiAvailable") is True and isinstance(selected, list):
            for item in selected:
                if not isinstance(item, Mapping):
                    continue
                for field, target in (
                    ("sizeBytes", model_bytes),
                    ("sizeVramBytes", model_vram),
                ):
                    value = _non_negative_int(item.get(field))
                    if value is not None:
                        target.append(value)
                digest = _resource_digest(item.get("digest"))
                if digest is not None:
                    observed_digests.add(digest)
        process = snapshot.get("process")
        if not isinstance(process, Mapping):
            continue
        whole_gpu = process.get("wholeGpuMemory")
        if isinstance(whole_gpu, Mapping) and whole_gpu.get("available") is True:
            used = _non_negative_int(whole_gpu.get("aggregateUsedVramBytes"))
            free = _non_negative_int(whole_gpu.get("aggregateFreeVramBytes"))
            if used is not None and free is not None:
                whole_gpu_used.append(used)
                whole_gpu_free.append(free)
        if process.get("available") is not True:
            continue
        for field, target in (
            ("ollamaAggregateRssBytes", ollama_rss),
            ("benchmarkProcessRssBytes", benchmark_rss),
            ("ollamaAggregatePeakWorkingSetBytes", ollama_peak_working_set),
            (
                "benchmarkProcessPeakWorkingSetBytes",
                benchmark_peak_working_set,
            ),
            ("availablePhysicalMemoryBytes", available_memory),
        ):
            value = _non_negative_int(process.get(field))
            if value is not None:
                target.append(value)
        gpu = process.get("gpuProcessMemory")
        if isinstance(gpu, Mapping) and gpu.get("available") is True:
            value = _non_negative_int(gpu.get("ollamaAggregateVramBytes"))
            if value is not None:
                process_vram.append(value)
    before = next(
        (snapshot for snapshot in snapshots if snapshot.get("phase") == "before-run"),
        snapshots[0] if snapshots else {},
    )
    before_release = next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if snapshot.get("phase") == "before-release"
        ),
        {},
    )
    after_release = next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if snapshot.get("phase") == "after-release"
        ),
        {},
    )
    digests = sorted(observed_digests)
    loaded_without_process = any(
        snapshot.get("ollamaApiAvailable") is True
        and snapshot.get("selectedModelLoaded") is True
        and isinstance(snapshot.get("process"), Mapping)
        and snapshot["process"].get("available") is True
        and snapshot["process"].get("ollamaProcessCount") == 0
        for snapshot in snapshots
    )
    periodic_count = sum(
        str(snapshot.get("phase") or "").startswith("periodic:")
        for snapshot in snapshots
    )
    selected_model_loaded_observed = any(
        snapshot.get("ollamaApiAvailable") is True
        and snapshot.get("selectedModelLoaded") is True
        for snapshot in snapshots
    )
    post_release_model_absent = bool(
        after_release.get("ollamaApiAvailable") is True
        and after_release.get("selectedModelLoaded") is False
    )
    return {
        "measurement": "ollama-residency-plus-process-snapshots-v1",
        "maximumSemantics": "maximum-observed-periodic-and-stage-snapshots",
        "sampleIntervalSeconds": sample_interval_seconds,
        "periodicSnapshotCount": periodic_count,
        "periodicSamplerStopped": sampler_stopped,
        "snapshotCount": len(snapshots),
        "preRunSelectedModelLoaded": before.get("selectedModelLoaded"),
        "preReleaseSelectedModelLoaded": before_release.get(
            "selectedModelLoaded"
        ),
        "preReleaseOllamaApiAvailable": before_release.get(
            "ollamaApiAvailable"
        ),
        "postReleaseSelectedModelLoaded": after_release.get(
            "selectedModelLoaded"
        ),
        "selectedModelLoadedObserved": selected_model_loaded_observed,
        "peakSelectedModelBytes": max(model_bytes, default=None),
        "peakSelectedModelVramBytes": max(model_vram, default=None),
        "selectedModelVramMeasurement": (
            "ollama-api-ps-size-vram-model-metadata"
        ),
        "peakOllamaAggregateRssBytes": (
            None if loaded_without_process else max(ollama_rss, default=None)
        ),
        "peakBenchmarkProcessRssBytes": max(benchmark_rss, default=None),
        "peakOllamaProcessVramBytes": (
            None if loaded_without_process else max(process_vram, default=None)
        ),
        "peakWholeGpuUsedVramBytes": max(whole_gpu_used, default=None),
        "minimumWholeGpuFreeVramBytes": min(whole_gpu_free, default=None),
        "wholeGpuSnapshotCount": len(whole_gpu_used),
        "wholeGpuVramEvidenceAvailable": bool(whole_gpu_used),
        "wholeGpuMemoryIncludesOtherProcesses": True,
        "wholeGpuMemoryDiagnosticOnly": True,
        "peakOllamaAggregateWorkingSetBytes": max(
            ollama_peak_working_set,
            default=None,
        ) if not loaded_without_process else None,
        "peakBenchmarkProcessWorkingSetBytes": max(
            benchmark_peak_working_set,
            default=None,
        ),
        "maximumObservedSelectedModelVramBytes": max(model_vram, default=None),
        "maximumObservedOllamaAggregateRssBytes": (
            None if loaded_without_process else max(ollama_rss, default=None)
        ),
        "maximumObservedOllamaProcessVramBytes": (
            None if loaded_without_process else max(process_vram, default=None)
        ),
        "loadedModelWithoutMatchedProcess": loaded_without_process,
        "processEvidenceFailureCode": (
            "LOADED_MODEL_WITHOUT_MATCHED_PROCESS"
            if loaded_without_process
            else None
        ),
        "processRamEvidenceComplete": bool(ollama_rss)
        and not loaded_without_process,
        "processVramEvidenceComplete": bool(process_vram)
        and not loaded_without_process,
        "minimumAvailablePhysicalMemoryBytes": min(available_memory, default=None),
        "observedRuntimeDigests": digests,
        "runtimeDigestObserved": bool(digests),
        "runtimeDigestMatchesExpected": bool(digests)
        and set(digests) == {expected_digest},
        "postReleaseModelAbsent": post_release_model_absent,
        # Compatibility alias for existing report readers. The transition from
        # loaded to absent is reported separately by the model execution block.
        "modelAbsentAfterRelease": post_release_model_absent,
        "snapshots": [copy.deepcopy(dict(item)) for item in snapshots],
    }


class _RecordingProvider:
    """Hash provider envelopes without retaining prompt or response bodies."""

    def __init__(
        self,
        delegate: Any,
        *,
        clock: Callable[[], float] = time.perf_counter,
        first_call_latency_class: str = "cold-unverified",
        call_residency_probe: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.delegate = delegate
        self.provider_id = getattr(delegate, "provider_id", type(delegate).__name__)
        self.provider_version = getattr(delegate, "provider_version", "unknown")
        self.network_policy = getattr(delegate, "network_policy", "loopback-only")
        self.calls: list[dict[str, Any]] = []
        self.release_calls = 0
        self.clock = clock
        self.first_call_latency_class = first_call_latency_class
        self.call_residency_probe = call_residency_probe

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    @property
    def generation_metrics(self) -> dict[str, int]:
        raw = getattr(self.delegate, "generation_metrics", {})
        if callable(raw):
            raw = raw()
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(key): int(value)
            for key, value in raw.items()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }

    def generate_json(self, **kwargs: Any) -> Mapping[str, Any]:
        schema = kwargs.get("response_schema")
        before_native_metrics = self.generation_metrics
        call_index = len(self.calls) + 1
        pre_call_loaded: bool | None = None
        pre_call_api_available: bool | None = None
        if self.call_residency_probe is not None:
            pre_call_snapshot = self.call_residency_probe(
                f"before-provider-call:{call_index}"
            )
            if isinstance(
                pre_call_snapshot.get("ollamaApiAvailable"),
                bool,
            ):
                pre_call_api_available = bool(
                    pre_call_snapshot["ollamaApiAvailable"]
                )
            if isinstance(pre_call_snapshot.get("selectedModelLoaded"), bool):
                pre_call_loaded = bool(
                    pre_call_snapshot["selectedModelLoaded"]
                )
        user_prompt = kwargs.get("user_prompt")
        correction: Mapping[str, Any] = {}
        if isinstance(user_prompt, str):
            try:
                prompt_value = json.loads(user_prompt)
            except json.JSONDecodeError:
                prompt_value = None
            if isinstance(prompt_value, Mapping) and isinstance(
                prompt_value.get("correction"), Mapping
            ):
                correction = prompt_value["correction"]
        started = self.clock()
        envelope = {
            "callIndex": call_index,
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "latencyClass": "pending",
            "preCallOllamaApiAvailable": pre_call_api_available,
            "preCallResidencyVerified": bool(
                pre_call_api_available is True
                and isinstance(pre_call_loaded, bool)
            ),
            "preCallSelectedModelLoaded": pre_call_loaded,
            "retryAttempt": _non_negative_int(correction.get("attempt")),
            "previousResponseRejected": bool(
                correction.get("previousResponseRejected", False)
            ),
            "validationFailureCode": (
                str(correction["validationFailureCode"])
                if isinstance(correction.get("validationFailureCode"), str)
                else None
            ),
            "systemPromptSha256": _sha_text(kwargs.get("system_prompt")),
            "userPromptSha256": _sha_text(user_prompt),
            "responseSchemaSha256": canonical_json_sha256(schema) if schema is not None else None,
            "outcome": "pending",
            "responseSha256": None,
            "errorType": None,
            "structuredFailureCodes": [],
            "wallTimeSeconds": None,
            "nativeMetrics": {},
        }
        self.calls.append(envelope)
        try:
            result = self.delegate.generate_json(**kwargs)
            envelope["responseSha256"] = canonical_json_sha256(result)
        except Exception as exc:
            envelope["outcome"] = "error"
            envelope["latencyClass"] = "failed-before-classification"
            envelope["errorType"] = type(exc).__name__
            envelope["structuredFailureCodes"] = _structured_failure_codes(exc)
            envelope["wallTimeSeconds"] = max(
                0.0,
                float(self.clock() - started),
            )
            envelope["nativeMetrics"] = _delta(
                before_native_metrics,
                self.generation_metrics,
            )
            raise
        envelope["outcome"] = "returned"
        if pre_call_api_available is True and pre_call_loaded is False:
            latency_class = "cold" if call_index == 1 else "reload"
        elif pre_call_api_available is True and pre_call_loaded is True:
            latency_class = "warm-preloaded" if call_index == 1 else "warm"
        elif pre_call_loaded is True:
            latency_class = "warm-unverified"
        elif self.call_residency_probe is None and call_index == 1:
            latency_class = self.first_call_latency_class
        else:
            latency_class = (
                "cold-unverified" if call_index == 1 else "warm-unverified"
            )
        envelope["latencyClass"] = latency_class
        envelope["wallTimeSeconds"] = max(
            0.0,
            float(self.clock() - started),
        )
        envelope["nativeMetrics"] = _delta(
            before_native_metrics,
            self.generation_metrics,
        )
        return result

    def release_resources(self) -> None:
        self.release_calls += 1
        release = getattr(self.delegate, "release_resources", None)
        if not callable(release):
            raise RealMultilingualSemanticBenchmarkError(
                "semantic provider has no resource release method"
            )
        release()


def _sha_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metrics(provider: _RecordingProvider) -> dict[str, int]:
    return {
        key: value
        for key, value in provider.generation_metrics.items()
        if key in {
            "completedCalls",
            "totalDurationNanoseconds",
            "loadDurationNanoseconds",
            "promptEvalTokens",
            "promptEvalDurationNanoseconds",
            "outputTokens",
            "outputEvalDurationNanoseconds",
        }
    }


def _delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    keys = sorted(set(before) | set(after))
    result = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}
    if any(value < 0 for value in result.values()):
        raise RealMultilingualSemanticBenchmarkError("provider metrics moved backwards")
    return result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _duration_distribution(values: Sequence[Any]) -> dict[str, Any]:
    durations = [
        float(value)
        for value in values
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    ]
    return {
        "count": len(durations),
        "sumSeconds": sum(durations),
        "minimumSeconds": min(durations, default=None),
        "maximumSeconds": max(durations, default=None),
        "meanSeconds": sum(durations) / len(durations) if durations else None,
        "p50Seconds": _percentile(durations, 0.50),
        "p95Seconds": _percentile(durations, 0.95),
    }


def _call_latency_summary(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = (
        "cold",
        "cold-unverified",
        "warm-preloaded",
        "warm",
        "warm-unverified",
        "reload",
        "failed-before-classification",
    )
    by_class: dict[str, Any] = {}
    for latency_class in classes:
        selected = [
            call for call in calls if call.get("latencyClass") == latency_class
        ]
        by_class[latency_class] = {
            "wallTime": _duration_distribution(
                [call.get("wallTimeSeconds") for call in selected]
            ),
            "providerTotalDuration": _duration_distribution(
                [
                    int(call.get("nativeMetrics", {}).get("totalDurationNanoseconds"))
                    / 1_000_000_000.0
                    for call in selected
                    if isinstance(call.get("nativeMetrics"), Mapping)
                    and not isinstance(
                        call.get("nativeMetrics", {}).get("totalDurationNanoseconds"),
                        bool,
                    )
                    and isinstance(
                        call.get("nativeMetrics", {}).get("totalDurationNanoseconds"),
                        int,
                    )
                ]
            ),
            "providerLoadDuration": _duration_distribution(
                [
                    int(call.get("nativeMetrics", {}).get("loadDurationNanoseconds"))
                    / 1_000_000_000.0
                    for call in selected
                    if isinstance(call.get("nativeMetrics"), Mapping)
                    and not isinstance(
                        call.get("nativeMetrics", {}).get("loadDurationNanoseconds"),
                        bool,
                    )
                    and isinstance(
                        call.get("nativeMetrics", {}).get("loadDurationNanoseconds"),
                        int,
                    )
                ]
            ),
        }
    return {
        "measurement": "provider-call-wall-clock-v1",
        "coldStartVerified": by_class["cold"]["wallTime"]["count"] > 0,
        "warmCallsObserved": by_class["warm"]["wallTime"]["count"] > 0,
        "reloadCallsObserved": by_class["reload"]["wallTime"]["count"] > 0,
        "failedCallsExcludedFromColdWarm": by_class[
            "failed-before-classification"
        ]["wallTime"]["count"],
        "byLatencyClass": by_class,
    }


def _failure_recovery_summary(
    calls: Sequence[Mapping[str, Any]],
    *,
    final_status: str,
    max_batch_attempts: int,
    terminal_failure: bool,
) -> dict[str, Any]:
    correction_calls = [
        call for call in calls if call.get("previousResponseRejected") is True
    ]
    provider_error_calls = [call for call in calls if call.get("outcome") == "error"]
    validation_codes = sorted(
        {
            str(call["validationFailureCode"])
            for call in correction_calls
            if isinstance(call.get("validationFailureCode"), str)
        }
    )
    retry_attempted = bool(correction_calls)
    failure_observed = bool(
        correction_calls or provider_error_calls or terminal_failure
    )
    final_succeeded = final_status in {
        "composition-complete",
        "candidate-generation-required",
    }
    return {
        "retryBudgetApplies": max_batch_attempts > 1,
        "failureObserved": failure_observed,
        "retryAttempted": retry_attempted,
        "recoveryAttempted": retry_attempted,
        "validationRetryCallCount": len(correction_calls),
        "providerErrorCallCount": len(provider_error_calls),
        "validationFailureCodes": validation_codes,
        "recovered": bool(retry_attempted and final_succeeded),
        "exhausted": bool(retry_attempted and not final_succeeded),
        "terminalFailureWithoutRetry": bool(
            failure_observed and not retry_attempted and not final_succeeded
        ),
    }


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, WorkerError):
        return str(exc.code)
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, Mapping) and isinstance(diagnostics.get("failureStage"), str):
        return str(diagnostics["failureStage"]).upper().replace("-", "_")
    return type(exc).__name__


def _structured_failure_codes(exc: BaseException) -> list[str]:
    codes = {_failure_code(exc)}
    seen: set[int] = set()

    def inspect(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Mapping):
            for field in ("failureStage", "validationFailureCode"):
                field_value = value.get(field)
                if isinstance(field_value, str) and field_value.strip():
                    codes.add(
                        field_value.strip().upper().replace("-", "_")
                    )
            for nested in value.values():
                if isinstance(nested, (Mapping, list, tuple)):
                    inspect(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                if isinstance(nested, (Mapping, list, tuple)):
                    inspect(nested)

    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        inspect(getattr(current, "details", None))
        inspect(getattr(current, "diagnostics", None))
        current = current.__cause__ or current.__context__
    return sorted(codes)


def _summary_hash(value: Any) -> str:
    return canonical_json_sha256(value)


def _segment_views(value: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    raw = value.get("segments")
    if not isinstance(raw, list):
        raise RealMultilingualSemanticBenchmarkError(f"{label}.segments is invalid")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise RealMultilingualSemanticBenchmarkError(f"{label}.segments[{index}] is invalid")
        text = None
        for key in ("finalText", "normalizedText", "text"):
            if isinstance(item.get(key), str):
                text = str(item[key])
                break
        if text is None:
            raise RealMultilingualSemanticBenchmarkError(f"{label}.segments[{index}] has no final text")
        segment_id = _text(item.get("id"), field=f"{label}.segments[{index}].id", maximum=200)
        start = item.get("startMs")
        end = item.get("endMs")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int) or start < 0 or end <= start:
            raise RealMultilingualSemanticBenchmarkError(f"{label}.segments[{index}] timing is invalid")
        speaker = _text(item.get("speakerId"), field=f"{label}.segments[{index}].speakerId", maximum=200)
        language = _text(item.get("language"), field=f"{label}.segments[{index}].language", maximum=100)
        result.append({"id": segment_id, "startMs": start, "endMs": end, "speakerId": speaker, "language": language, "text": text, "overlapping": bool(item.get("overlapping", False))})
    return result


def _timeline_views(value: Mapping[str, Any], *, fallback_segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw_timeline = value.get("timeline") or value.get("speakerTimeline")
    raw_turns = raw_timeline.get("turns") if isinstance(raw_timeline, Mapping) else None
    if raw_turns is None:
        raw_turns = fallback_segments
    if not isinstance(raw_turns, list):
        raise RealMultilingualSemanticBenchmarkError("timeline turns are invalid")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_turns):
        if not isinstance(item, Mapping):
            raise RealMultilingualSemanticBenchmarkError(f"timeline turn {index} is invalid")
        start = item.get("startMs")
        end = item.get("endMs")
        speaker = item.get("speakerId")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int) or start < 0 or end <= start or not isinstance(speaker, str) or not speaker.strip():
            raise RealMultilingualSemanticBenchmarkError(f"timeline turn {index} is invalid")
        result.append({"startMs": start, "endMs": end, "speakerId": speaker.strip(), "overlap": bool(item.get("overlap", item.get("overlapping", False)))})
    return result


def _align_segments(actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> tuple[list[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]], str]:
    actual_ids = [str(item["id"]) for item in actual]
    expected_ids = [str(item["id"]) for item in expected]
    if actual_ids == expected_ids:
        return list(zip(actual, expected)), "id"
    count = max(len(actual), len(expected))
    pairs = [(actual[i] if i < len(actual) else None, expected[i] if i < len(expected) else None) for i in range(count)]
    return pairs, "position"


def _field_comparison(
    pairs: Sequence[tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]],
    *,
    field: str,
    actual_key: str,
    expected_key: str,
    hash_values: bool = False,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for index, (actual, expected) in enumerate(pairs):
        compared += 1
        actual_value = actual.get(actual_key) if actual is not None else None
        expected_value = expected.get(expected_key) if expected is not None else None
        if actual_value != expected_value:
            mismatches.append({
                "position": index,
                "actual": _sha_text(actual_value) if hash_values and isinstance(actual_value, str) else actual_value,
                "expected": _sha_text(expected_value) if hash_values and isinstance(expected_value, str) else expected_value,
            })
    return {
        "field": field,
        "exactMatch": not mismatches and len(pairs) > 0 or (not pairs),
        "comparedCount": compared,
        "mismatchCount": len(mismatches),
        "mismatchSetSha256": _summary_hash(mismatches),
    }


def compare_composition_to_baseline(
    composition: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare only hash-safe projections of a composition and baseline."""

    actual_segments = _segment_views(composition, label="composition")
    expected_segments = _segment_views(baseline, label="baseline")
    pairs, alignment = _align_segments(actual_segments, expected_segments)
    comparisons = {
        "speaker": _field_comparison(pairs, field="speaker", actual_key="speakerId", expected_key="speakerId"),
        "language": _field_comparison(pairs, field="language", actual_key="language", expected_key="language"),
        "text": _field_comparison(pairs, field="text", actual_key="text", expected_key="text", hash_values=True),
        "segmentTimeline": {
            "field": "segmentTimeline",
            "exactMatch": all(
                actual is not None
                and expected is not None
                and actual.get("startMs") == expected.get("startMs")
                and actual.get("endMs") == expected.get("endMs")
                for actual, expected in pairs
            ),
            "comparedCount": len(pairs),
            "mismatchCount": sum(
                not (
                    actual is not None
                    and expected is not None
                    and actual.get("startMs") == expected.get("startMs")
                    and actual.get("endMs") == expected.get("endMs")
                )
                for actual, expected in pairs
            ),
            "mismatchSetSha256": _summary_hash([
                {"position": i, "actual": (a.get("startMs"), a.get("endMs")) if a else None, "expected": (e.get("startMs"), e.get("endMs")) if e else None}
                for i, (a, e) in enumerate(pairs)
                if not (a is not None and e is not None and a.get("startMs") == e.get("startMs") and a.get("endMs") == e.get("endMs"))
            ]),
        },
    }
    actual_timeline = _timeline_views(composition, fallback_segments=actual_segments)
    expected_timeline = _timeline_views(baseline, fallback_segments=expected_segments)
    timeline_equal = actual_timeline == expected_timeline
    comparisons["timeline"] = {
        "field": "timeline",
        "exactMatch": timeline_equal,
        "actualSha256": _summary_hash(actual_timeline),
        "expectedSha256": _summary_hash(expected_timeline),
        "actualCount": len(actual_timeline),
        "expectedCount": len(expected_timeline),
        "mismatchCount": 0 if timeline_equal else 1,
        "mismatchSetSha256": _summary_hash({"actual": actual_timeline, "expected": expected_timeline}) if not timeline_equal else _summary_hash([]),
    }
    comparisons["alignment"] = {"mode": alignment, "actualCount": len(actual_segments), "expectedCount": len(expected_segments)}
    comparisons["exactMatch"] = all(
        comparisons[field]["exactMatch"]
        for field in ("speaker", "language", "text", "segmentTimeline", "timeline")
    )
    comparisons["comparisonSha256"] = _summary_hash(comparisons)
    return comparisons


def _request_summary(requests: Any) -> list[dict[str, Any]]:
    if not isinstance(requests, list):
        return []
    output: list[dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, Mapping):
            continue
        refs = item.get("evidenceRefs")
        output.append({
            "domain": item.get("domain"),
            "groupId": item.get("groupId"),
            "scopeId": item.get("scopeId"),
            "requestKind": item.get("requestKind"),
            "minimumAlternativeCount": item.get("minimumAlternativeCount"),
            "reasonCodesSha256": _summary_hash(sorted(str(v) for v in item.get("reasonCodes", []) if isinstance(v, str))),
            "evidenceRefsSha256": _summary_hash(sorted(str(v) for v in refs if isinstance(v, str))) if isinstance(refs, list) else None,
        })
    output.sort(key=lambda row: (str(row.get("domain")), str(row.get("scopeId")), str(row.get("groupId"))))
    return output


def _score_semantic_calibration(
    arbitration: Mapping[str, Any] | None,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    selections: dict[str, Mapping[str, Any]] = {}
    requests: dict[str, Mapping[str, Any]] = {}
    if isinstance(arbitration, Mapping):
        raw_selections = arbitration.get("selections")
        if isinstance(raw_selections, list):
            selections = {
                str(item.get("groupId")): item
                for item in raw_selections
                if isinstance(item, Mapping) and item.get("groupId") is not None
            }
        raw_requests = arbitration.get("candidateGenerationRequests")
        if isinstance(raw_requests, list):
            requests = {
                str(item.get("groupId")): item
                for item in raw_requests
                if isinstance(item, Mapping) and item.get("groupId") is not None
            }

    group_rows: list[dict[str, Any]] = []
    expected_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    raw_target_groups = target.get("groups")
    if not isinstance(raw_target_groups, list):
        raise RealMultilingualSemanticBenchmarkError(
            "validated semantic calibration target lost its groups"
        )
    for raw_target in raw_target_groups:
        if not isinstance(raw_target, Mapping):
            raise RealMultilingualSemanticBenchmarkError(
                "validated semantic calibration target group is invalid"
            )
        domain = str(raw_target["domain"])
        group_id = str(raw_target["groupId"])
        expected_action = str(raw_target["expectedAction"])
        selection = selections.get(group_id)
        request = requests.get(group_id)
        actual_action = "unavailable"
        correct = False
        if selection is not None:
            selected_id = selection.get("selectedCandidateId")
            if (
                expected_action == "preservation-only"
                and selected_id == raw_target["currentCandidateId"]
            ):
                actual_action = "preservation-only"
                correct = True
            else:
                actual_action = "select"
                correct = bool(
                    expected_action == "select"
                    and selected_id in raw_target["acceptableCandidateIds"]
                )
        elif request is not None:
            default_request = (
                request.get("requestKind")
                == _DEFAULT_CHALLENGER_REQUEST_KIND[domain]
            )
            actual_action = (
                "request-default-challenger"
                if default_request
                else "request-other-challenger"
            )
            correct = bool(
                expected_action == "request-default-challenger"
                and default_request
            )
        binding_sha = canonical_json_sha256(
            {
                "domain": domain,
                "groupId": group_id,
                "scopeId": raw_target["scopeId"],
                "currentCandidateId": raw_target["currentCandidateId"],
            }
        )
        row = {
            "groupBindingSha256": binding_sha,
            "expectedAction": expected_action,
            "actualAction": actual_action,
            "correct": correct,
        }
        group_rows.append(row)
        expected_counts[expected_action] += 1
        actual_counts[actual_action] += 1
        if correct:
            correct_counts[expected_action] += 1

    group_count = len(group_rows)
    correct_count = sum(row["correct"] is True for row in group_rows)
    by_action = {
        action: {
            "count": expected_counts[action],
            "correctCount": correct_counts[action],
            "accuracy": (
                correct_counts[action] / expected_counts[action]
                if expected_counts[action]
                else None
            ),
        }
        for action in _CALIBRATION_ACTIONS
    }
    return {
        "calibrationSha256": target["canonicalSha256"],
        "groupCount": group_count,
        "correctCount": correct_count,
        "incorrectCount": group_count - correct_count,
        "microAccuracy": correct_count / group_count if group_count else None,
        "expectedActionCounts": {
            action: expected_counts[action] for action in _CALIBRATION_ACTIONS
        },
        "actualActionCounts": dict(sorted(actual_counts.items())),
        "byExpectedAction": by_action,
        "groups": group_rows,
        "groupsSha256": _summary_hash(group_rows),
    }


def _aggregate_semantic_calibration(
    case_calibrations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    group_rows = [
        group
        for calibration in case_calibrations
        for group in calibration.get("groups", [])
        if isinstance(group, Mapping)
    ]
    expected_counts: Counter[str] = Counter(
        str(group.get("expectedAction")) for group in group_rows
    )
    actual_counts: Counter[str] = Counter(
        str(group.get("actualAction")) for group in group_rows
    )
    correct_counts: Counter[str] = Counter(
        str(group.get("expectedAction"))
        for group in group_rows
        if group.get("correct") is True
    )
    group_count = len(group_rows)
    correct_count = sum(group.get("correct") is True for group in group_rows)
    return {
        "caseCount": len(case_calibrations),
        "groupCount": group_count,
        "correctCount": correct_count,
        "incorrectCount": group_count - correct_count,
        "microAccuracy": correct_count / group_count if group_count else None,
        "actualActionCounts": dict(sorted(actual_counts.items())),
        "byExpectedAction": {
            action: {
                "count": expected_counts[action],
                "correctCount": correct_counts[action],
                "accuracy": (
                    correct_counts[action] / expected_counts[action]
                    if expected_counts[action]
                    else None
                ),
            }
            for action in _CALIBRATION_ACTIONS
        },
        "calibrationSetSha256": _summary_hash(
            [
                {
                    "calibrationSha256": calibration.get("calibrationSha256"),
                    "groupsSha256": calibration.get("groupsSha256"),
                }
                for calibration in case_calibrations
            ]
        ),
    }


def _blind_shared_input(case: FrozenSemanticCase) -> dict[str, Any]:
    raw_segments = case.document.get("segments")
    if not isinstance(raw_segments, list):
        raise RealMultilingualSemanticBenchmarkError(
            "blind review input segments are invalid"
        )
    segments: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise RealMultilingualSemanticBenchmarkError(
                f"blind review input segment {index} is invalid"
            )
        raw_text = raw.get("rawText")
        if not isinstance(raw_text, str):
            raise RealMultilingualSemanticBenchmarkError(
                f"blind review input segment {index} has no raw ASR text"
            )
        segment = {
            "segmentId": str(raw.get("id") or ""),
            "startMs": int(raw.get("startMs")),
            "endMs": int(raw.get("endMs")),
            "speakerId": str(raw.get("speakerId") or ""),
            "language": str(raw.get("language") or "und"),
            "rawText": raw_text,
            "overlapping": bool(raw.get("overlapping", False)),
        }
        segments.append(segment)
        timeline.append(
            {
                "startMs": segment["startMs"],
                "endMs": segment["endMs"],
                "speakerId": segment["speakerId"],
                "overlapping": segment["overlapping"],
            }
        )
    return {
        "language": case.language,
        "rawAsrSegments": segments,
        "speakerTimeline": timeline,
    }


def _blind_composition_result(composition: Mapping[str, Any]) -> dict[str, Any]:
    raw_segments = composition.get("segments")
    timeline = composition.get("timeline")
    speaker_policy = composition.get("speakerPolicy")
    if (
        not isinstance(raw_segments, list)
        or not isinstance(timeline, Mapping)
        or not isinstance(speaker_policy, Mapping)
    ):
        raise RealMultilingualSemanticBenchmarkError(
            "blind review composition is invalid"
        )
    segments: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("finalText"), str):
            raise RealMultilingualSemanticBenchmarkError(
                f"blind review composition segment {index} is invalid"
            )
        segments.append(
            {
                "segmentId": str(raw.get("id") or ""),
                "startMs": int(raw.get("startMs")),
                "endMs": int(raw.get("endMs")),
                "speakerId": str(raw.get("speakerId") or ""),
                "language": str(raw.get("language") or "und"),
                "finalText": str(raw["finalText"]),
                "overlapping": bool(raw.get("overlapping", False)),
            }
        )
    raw_turns = timeline.get("turns")
    raw_timeline_speaker_ids = timeline.get("speakerIds")
    raw_policy_speaker_ids = speaker_policy.get("speakerIds")
    resolved_count = _non_negative_int(speaker_policy.get("resolvedCount"))
    if (
        not isinstance(raw_turns, list)
        or not isinstance(raw_timeline_speaker_ids, list)
        or not isinstance(raw_policy_speaker_ids, list)
        or resolved_count is None
        or any(
            not isinstance(value, str) or not value.strip()
            for value in [*raw_timeline_speaker_ids, *raw_policy_speaker_ids]
        )
    ):
        raise RealMultilingualSemanticBenchmarkError(
            "blind review composition timeline is invalid"
        )
    timeline_speaker_ids = [str(value) for value in raw_timeline_speaker_ids]
    policy_speaker_ids = [str(value) for value in raw_policy_speaker_ids]
    if (
        len(policy_speaker_ids) != resolved_count
        or len(set(policy_speaker_ids)) != len(policy_speaker_ids)
        or timeline_speaker_ids != policy_speaker_ids
    ):
        raise RealMultilingualSemanticBenchmarkError(
            "blind review composition speaker policy is inconsistent"
        )
    turns: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, Mapping):
            raise RealMultilingualSemanticBenchmarkError(
                f"blind review composition turn {index} is invalid"
            )
        turns.append(
            {
                "startMs": int(raw.get("startMs")),
                "endMs": int(raw.get("endMs")),
                "speakerId": str(raw.get("speakerId") or ""),
                "overlapping": bool(
                    raw.get("overlap", raw.get("overlapping", False))
                ),
            }
        )
    return {
        "status": "composition-complete",
        "disposition": str(composition.get("disposition") or ""),
        "speakerPolicy": {
            "resolvedCount": resolved_count,
            "speakerIds": policy_speaker_ids,
        },
        "segments": segments,
        "timeline": turns,
    }


def _blind_request_result(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    projected: list[dict[str, Any]] = []
    for raw in requests:
        if not isinstance(raw, Mapping):
            raise RealMultilingualSemanticBenchmarkError(
                "blind review candidate-generation request is invalid"
            )
        projected.append(
            {
                "domain": raw.get("domain"),
                "groupId": raw.get("groupId"),
                "scopeId": raw.get("scopeId"),
                "requestKind": raw.get("requestKind"),
                "minimumAlternativeCount": raw.get("minimumAlternativeCount"),
                "reasonCodes": sorted(
                    {
                        str(value)
                        for value in raw.get("reasonCodes", [])
                        if isinstance(value, str)
                    }
                ),
            }
        )
    projected.sort(
        key=lambda row: (
            str(row.get("domain")),
            str(row.get("scopeId")),
            str(row.get("groupId")),
            str(row.get("requestKind")),
        )
    )
    return {
        "status": "candidate-generation-required",
        "candidateGenerationRequests": projected,
    }


def _blind_failure_result(failure_codes: Sequence[str]) -> dict[str, Any]:
    del failure_codes
    return {
        "status": "failed",
        # Detailed exception/provider codes remain in the redacted benchmark
        # report. A fixed public code prevents failed candidates being
        # fingerprinted by model-specific failure behavior.
        "failureCodes": ["SEMANTIC_RESULT_UNAVAILABLE"],
    }


def _blind_not_executed_result(reason: str | None) -> dict[str, Any]:
    del reason
    return {
        "status": "not-executed",
        "failureCodes": ["MODEL_NOT_EXECUTED"],
    }


def _with_canonical_sha256(body: Mapping[str, Any]) -> dict[str, Any]:
    if "canonicalSha256" in body:
        raise RealMultilingualSemanticBenchmarkError(
            "blind review JSON body already declares canonicalSha256"
        )
    normalized = copy.deepcopy(dict(body))
    validate_strict_json(normalized)
    return {
        **normalized,
        "canonicalSha256": canonical_json_sha256(normalized),
    }


def _assert_hidden_calibration_metadata_absent(
    value: Any,
    *,
    artifact: str,
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _HIDDEN_CALIBRATION_KEYS:
                raise RealMultilingualSemanticBenchmarkError(
                    f"{artifact} contains hidden semantic calibration metadata"
                )
            _assert_hidden_calibration_metadata_absent(
                nested,
                artifact=artifact,
            )
    elif isinstance(value, list):
        for nested in value:
            _assert_hidden_calibration_metadata_absent(
                nested,
                artifact=artifact,
            )


def _blind_candidate_order_key(
    seed: str,
    case: FrozenSemanticCase,
    spec: SemanticModelSpec,
) -> tuple[bytes, str, str, str]:
    identity = "\0".join(
        (
            "anonymous-semantic-blind-review-candidate-order-v1",
            case.case_sha256,
            spec.model,
            spec.model_id or spec.model,
            spec.digest,
        )
    )
    return (
        hmac.new(
            seed.encode("utf-8"),
            identity.encode("utf-8"),
            hashlib.sha256,
        ).digest(),
        spec.model,
        spec.model_id or spec.model,
        spec.digest,
    )


def _assert_blind_reviewer_packet(
    packet: Mapping[str, Any],
    *,
    case: FrozenSemanticCase,
    specs: Sequence[SemanticModelSpec],
) -> None:
    _assert_hidden_calibration_metadata_absent(
        packet,
        artifact="reviewer packet",
    )
    serialized = json.dumps(
        packet,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    folded = serialized.casefold()
    forbidden_keys = (
        "baseline",
        "reference",
        "comparison",
        "resource",
        "automaticScore",
        "seed",
        "prompt",
        "response",
    )

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                key_folded = str(key).casefold()
                if any(token.casefold() in key_folded for token in forbidden_keys):
                    raise RealMultilingualSemanticBenchmarkError(
                        "reviewer packet contains forbidden evaluation metadata"
                    )
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(packet)
    leaf_strings: set[str] = set()

    def collect_strings(value: Any) -> None:
        if isinstance(value, str):
            leaf_strings.add(value.casefold())
        elif isinstance(value, Mapping):
            for nested in value.values():
                collect_strings(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_strings(nested)

    collect_strings(packet)
    sensitive_identities = {case.case_id}
    for spec in specs:
        sensitive_identities.update(
            value
            for value in (
                spec.model,
                spec.model_id,
                spec.digest,
                spec.config_path,
            )
            if isinstance(value, str) and value
        )
    for identity in sensitive_identities:
        normalized_identity = identity.casefold()
        if (
            normalized_identity in leaf_strings
            or (
                len(normalized_identity) >= 4
                and normalized_identity in folded
            )
        ):
            raise RealMultilingualSemanticBenchmarkError(
                "reviewer packet leaks a case or model identity"
            )

def _build_blind_review_documents(
    cases: Sequence[FrozenSemanticCase],
    specs: Sequence[SemanticModelSpec],
    review_material: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    blind_seed: str,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    packet_rows: list[tuple[str, dict[str, Any]]] = []
    vault_cases: list[dict[str, Any]] = []
    ordered_cases = sorted(
        cases,
        key=lambda case: (case.language, case.case_id, case.case_sha256),
    )
    for case_index, case in enumerate(ordered_cases, start=1):
        case_alias = f"case-{case_index:03d}"
        shared_input = _blind_shared_input(case)
        _assert_hidden_calibration_metadata_absent(
            shared_input,
            artifact="blind shared input",
        )
        reviewer_candidates: list[dict[str, Any]] = []
        vault_candidates: list[dict[str, Any]] = []
        ordered_specs = sorted(
            specs,
            key=lambda spec: _blind_candidate_order_key(blind_seed, case, spec),
        )
        for candidate_index, spec in enumerate(ordered_specs, start=1):
            candidate_alias = f"candidate-{candidate_index:02d}"
            model_material = review_material.get(spec.model)
            result = (
                model_material.get(case.case_id)
                if isinstance(model_material, Mapping)
                else None
            )
            if not isinstance(result, Mapping):
                raise RealMultilingualSemanticBenchmarkError(
                    "blind review material is incomplete"
                )
            projected_result = copy.deepcopy(dict(result))
            validate_strict_json(projected_result)
            result_sha = canonical_json_sha256(projected_result)
            reviewer_candidates.append(
                {
                    "candidateAlias": candidate_alias,
                    "resultCanonicalSha256": result_sha,
                    "result": projected_result,
                }
            )
            vault_candidates.append(
                {
                    "candidateAlias": candidate_alias,
                    "model": spec.model,
                    "modelId": spec.model_id or spec.model,
                    "digest": spec.digest,
                    "resultCanonicalSha256": result_sha,
                }
            )
        packet = _with_canonical_sha256(
            {
                "schemaVersion": SCHEMA_VERSION,
                "artifactType": BLIND_PACKET_ARTIFACT_TYPE,
                "caseAlias": case_alias,
                "sharedInputCanonicalSha256": canonical_json_sha256(shared_input),
                "sharedInput": shared_input,
                "candidates": reviewer_candidates,
            }
        )
        _assert_blind_reviewer_packet(packet, case=case, specs=specs)
        relative_path = f"reviewer/{case_alias}.json"
        packet_rows.append((relative_path, packet))
        vault_cases.append(
            {
                "caseAlias": case_alias,
                "caseId": case.case_id,
                "caseSha256": case.case_sha256,
                "candidates": vault_candidates,
            }
        )
    input_set_binding = [
        {
            "caseId": case.case_id,
            "caseSha256": case.case_sha256,
            "sharedInputCanonicalSha256": canonical_json_sha256(
                _blind_shared_input(case)
            ),
        }
        for case in ordered_cases
    ]
    model_set_binding = sorted(
        (
            {
                "model": spec.model,
                "modelId": spec.model_id or spec.model,
                "digest": spec.digest,
            }
            for spec in specs
        ),
        key=lambda row: (
            str(row["model"]),
            str(row["modelId"]),
            str(row["digest"]),
        ),
    )
    vault = _with_canonical_sha256(
        {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": BLIND_VAULT_ARTIFACT_TYPE,
            "seedSha256": hashlib.sha256(blind_seed.encode("utf-8")).hexdigest(),
            "inputSetCanonicalSha256": canonical_json_sha256(input_set_binding),
            "modelSetCanonicalSha256": canonical_json_sha256(model_set_binding),
            "cases": vault_cases,
        }
    )
    _assert_hidden_calibration_metadata_absent(
        vault,
        artifact="identity vault",
    )
    return packet_rows, vault


def _verified_blind_json_evidence(
    path: Path,
    *,
    relative_path: str,
    role: str,
    expected_canonical_sha256: str,
) -> dict[str, Any]:
    persisted = read_json_strict(path)
    declared = persisted.pop("canonicalSha256", None)
    if (
        declared != expected_canonical_sha256
        or canonical_json_sha256(persisted) != expected_canonical_sha256
    ):
        raise RealMultilingualSemanticBenchmarkError(
            "blind review JSON canonical SHA-256 verification failed"
        )
    return {
        "role": role,
        "relativePath": relative_path,
        "canonicalSha256": expected_canonical_sha256,
        "fileSha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def _publish_blind_review_package_no_replace(
    output_root: Path,
    cases: Sequence[FrozenSemanticCase],
    specs: Sequence[SemanticModelSpec],
    review_material: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    blind_seed: str,
) -> dict[str, Any]:
    target = output_root.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"blind review output already exists: {target}")
    packet_rows, vault = _build_blind_review_documents(
        cases,
        specs,
        review_material,
        blind_seed=blind_seed,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=target.parent,
        )
    )
    file_evidence: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    manifest_file_sha: str | None = None
    try:
        for relative_path, packet in packet_rows:
            path = staging / Path(relative_path)
            atomic_write_json_no_replace(path, packet)
            file_evidence.append(
                _verified_blind_json_evidence(
                    path,
                    relative_path=relative_path,
                    role="reviewer-packet",
                    expected_canonical_sha256=str(packet["canonicalSha256"]),
                )
            )
        vault_path = staging / "identity-vault.json"
        atomic_write_json_no_replace(vault_path, vault)
        file_evidence.append(
            _verified_blind_json_evidence(
                vault_path,
                relative_path="identity-vault.json",
                role="identity-vault",
                expected_canonical_sha256=str(vault["canonicalSha256"]),
            )
        )
        file_evidence.sort(key=lambda row: str(row["relativePath"]))
        expected_json_files = sorted(
            str(path.relative_to(staging).as_posix())
            for path in staging.rglob("*.json")
        )
        if expected_json_files != sorted(
            str(row["relativePath"]) for row in file_evidence
        ):
            raise RealMultilingualSemanticBenchmarkError(
                "blind review manifest evidence is incomplete"
            )
        manifest = _with_canonical_sha256(
            {
                "schemaVersion": SCHEMA_VERSION,
                "artifactType": BLIND_MANIFEST_ARTIFACT_TYPE,
                "publicationPolicy": "atomic-directory-no-replace",
                "manifestWrittenLast": True,
                "packetCount": len(packet_rows),
                "candidatesPerPacket": len(specs),
                "files": file_evidence,
            }
        )
        manifest_path = staging / "manifest.json"
        atomic_write_json_no_replace(manifest_path, manifest)
        manifest_evidence = _verified_blind_json_evidence(
            manifest_path,
            relative_path="manifest.json",
            role="manifest",
            expected_canonical_sha256=str(manifest["canonicalSha256"]),
        )
        manifest_file_sha = str(manifest_evidence["fileSha256"])
        _rename_directory_no_replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    assert manifest is not None and manifest_file_sha is not None
    return {
        "published": True,
        "publicationPolicy": "atomic-directory-no-replace",
        "manifestCanonicalSha256": manifest["canonicalSha256"],
        "manifestFileSha256": manifest_file_sha,
        "packetCount": len(packet_rows),
        "candidatesPerPacket": len(specs),
    }


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically rename a complete sibling directory without replacement."""

    if os.name == "nt":
        # MoveFile on Windows fails when the destination already exists.
        os.rename(source, target)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RealMultilingualSemanticBenchmarkError(
                "atomic no-replace directory rename is unavailable"
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(target),
            rename_noreplace,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                str(target),
            )
        raise OSError(error_number, os.strerror(error_number), str(target))
    raise RealMultilingualSemanticBenchmarkError(
        "atomic no-replace directory rename is unsupported on this platform"
    )


def _running_on_wsl_mount(path: Path) -> bool:
    """Return whether a Linux process is publishing through a WSL Windows mount."""

    if not sys.platform.startswith("linux") or not str(path).startswith("/mnt/"):
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").casefold()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def _preflight_blind_directory_publish(target: Path) -> None:
    """Fail before inference if the target filesystem cannot no-replace rename dirs."""

    if not _running_on_wsl_mount(target):
        return
    parent = target.parent
    source = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.publish-preflight-",
            dir=parent,
        )
    )
    destination = parent / f".{target.name}.publish-probe-{uuid.uuid4().hex}"
    try:
        (source / "probe").write_text("preflight\n", encoding="utf-8")
        _rename_directory_no_replace(source, destination)
    finally:
        if source.exists():
            shutil.rmtree(source, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)


def _default_provider_factory(
    spec: SemanticModelSpec,
    *,
    endpoint: str,
    timeout_seconds: float,
    context_tokens: int,
    output_tokens: int,
) -> Any:
    return OllamaLocalProvider(
        LocalLLMConfig(
            model=spec.model,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            temperature=0.0,
            top_p=0.1,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            keep_alive="10m",
            release_on_close=True,
            offline_only=True,
            expected_model_digest=spec.digest,
        )
    )


def _model_report(
    spec: SemanticModelSpec,
    cases: Sequence[FrozenSemanticCase],
    *,
    provider_factory: Callable[[SemanticModelSpec], Any],
    resource_probe: ResourceProbe,
    resource_probe_kind: str,
    default_provider_mode: bool,
    strict_isolation_required: bool,
    resource_sample_interval_seconds: float,
    context_tokens: int,
    output_tokens: int,
    batch_size: int,
    max_batch_attempts: int,
    clock: Callable[[], float],
    review_material_sink: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = clock()
    provider: Any = None
    recording: _RecordingProvider | None = None
    runner: SemanticJobArbitrationRunner | None = None
    rows: list[dict[str, Any]] = []
    model_failure: str | None = None
    model_failure_codes: list[str] = []
    release_attempted = False
    release_succeeded = False
    release_failure_code: str | None = None
    release_failure_codes: list[str] = []
    digest_verified = False
    resource_snapshots = [_probe_resource(resource_probe, spec, "before-run")]
    sampler = _PeriodicResourceSampler(
        resource_probe,
        spec,
        interval_seconds=resource_sample_interval_seconds,
    )
    sampler_started = False
    sampler_stopped = False
    initially_loaded = resource_snapshots[0].get("selectedModelLoaded")
    first_call_latency_class = (
        "cold"
        if initially_loaded is False
        else "warm-preloaded"
        if initially_loaded is True
        else "cold-unverified"
    )

    def record_call_residency(phase: str) -> Mapping[str, Any]:
        snapshot = _probe_resource(resource_probe, spec, phase)
        resource_snapshots.append(snapshot)
        return snapshot

    try:
        sampler.start()
        sampler_started = True
        provider = provider_factory(spec)
        recording = _RecordingProvider(
            provider,
            clock=clock,
            first_call_latency_class=first_call_latency_class,
            call_residency_probe=record_call_residency,
        )
        runner = SemanticJobArbitrationRunner(
            provider=recording,
            model=spec.model,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            batch_size=batch_size,
            max_batch_attempts=max_batch_attempts,
        )
        for case in cases:
            case_started = clock()
            before_metrics = _metrics(recording)
            before_call_count = len(recording.calls)
            arbitration: Mapping[str, Any] | None = None
            arbitration_validated = False
            composition: Mapping[str, Any] | None = None
            failure_codes: list[str] = []
            blind_requests: list[Mapping[str, Any]] = []
            status = "failed"
            frozen_hash_before = canonical_json_sha256(case.lattice)
            try:
                arbitration = runner.run(
                    copy.deepcopy(case.document),
                    candidate_lattice=copy.deepcopy(case.lattice),
                )
                arbitration = validate_semantic_job_arbitration(
                    arbitration,
                    expected_job_id=str(case.document["jobId"]),
                    expected_lattice=case.lattice,
                )
                arbitration_validated = True
                raw_requests = arbitration.get("candidateGenerationRequests")
                requests = _request_summary(raw_requests)
                if isinstance(raw_requests, list):
                    blind_requests = [
                        item for item in raw_requests if isinstance(item, Mapping)
                    ]
                if requests:
                    status = "candidate-generation-required"
                else:
                    composition = build_semantic_composition(
                        case.document,
                        case.lattice,
                        arbitration,
                    )
                    status = "composition-complete"
            except _FAILURE_TYPES as exc:
                failure_codes.extend(_structured_failure_codes(exc))
                raw_requests = (
                    arbitration.get("candidateGenerationRequests")
                    if isinstance(arbitration, Mapping)
                    else None
                )
                requests = _request_summary(raw_requests)
                if isinstance(raw_requests, list):
                    blind_requests = [
                        item for item in raw_requests if isinstance(item, Mapping)
                    ]
            except Exception as exc:  # pragma: no cover - defensive boundary
                failure_codes.extend(_structured_failure_codes(exc))
                requests = []
            frozen_hash_after = canonical_json_sha256(case.lattice)
            if frozen_hash_after != frozen_hash_before:
                failure_codes.append("FROZEN_LATTICE_MUTATED")
                status = "failed"
            after_metrics = _metrics(recording)
            digest_verified = digest_verified or bool(
                getattr(provider, "_model_digest_verified", False)
                or getattr(provider, "model_digest_verified", False)
            )
            call_evidence = recording.calls[before_call_count:]
            comparison = None
            if composition is not None and not failure_codes:
                try:
                    comparison = compare_composition_to_baseline(composition, case.baseline)
                except _FAILURE_TYPES as exc:
                    failure_codes.extend(_structured_failure_codes(exc))
                    status = "failed"
            calibration = _score_semantic_calibration(
                arbitration if arbitration_validated else None,
                case.semantic_calibration_target,
            )
            if review_material_sink is not None:
                try:
                    if failure_codes or status == "failed":
                        review_result = _blind_failure_result(failure_codes)
                    elif composition is not None:
                        review_result = _blind_composition_result(composition)
                    elif status == "candidate-generation-required":
                        review_result = _blind_request_result(blind_requests)
                    else:
                        review_result = _blind_failure_result(
                            ["SEMANTIC_RESULT_UNAVAILABLE"]
                        )
                except _FAILURE_TYPES:
                    failure_codes.append("BLIND_REVIEW_RESULT_PROJECTION_FAILED")
                    status = "failed"
                    review_result = _blind_failure_result(failure_codes)
                review_material_sink[case.case_id] = review_result
            case_wall_time = max(0.0, float(clock() - case_started))
            successful_call_latency_classes = {
                str(call.get("latencyClass"))
                for call in call_evidence
                if call.get("outcome") == "returned"
            }
            case_latency_class = (
                "cold-start-case"
                if "cold" in successful_call_latency_classes
                else "reload-case"
                if "reload" in successful_call_latency_classes
                else "preloaded-first-case"
                if "warm-preloaded" in successful_call_latency_classes
                else "warm-case"
                if successful_call_latency_classes
                and successful_call_latency_classes <= {"warm"}
                else "latency-unverified-case"
                if successful_call_latency_classes
                & {"cold-unverified", "warm-unverified"}
                else "failed-provider-call-case"
                if any(call.get("outcome") == "error" for call in call_evidence)
                else "no-provider-call"
            )
            recovery = _failure_recovery_summary(
                call_evidence,
                final_status=status,
                max_batch_attempts=max_batch_attempts,
                terminal_failure=bool(failure_codes),
            )
            recovery["maxBatchAttempts"] = max_batch_attempts
            row: dict[str, Any] = {
                "caseId": case.case_id,
                "language": case.language,
                "caseSha256": case.case_sha256,
                "input": {
                    "documentSha256": case.document_sha256,
                    "latticeSha256": case.lattice_sha256,
                    "baselineSha256": case.baseline_sha256,
                    "baselineAuditSource": case.baseline_audit_source,
                },
                "execution": {
                    "status": status,
                    "latencyClass": case_latency_class,
                    "wallTimeSeconds": case_wall_time,
                    "providerCallCount": len(call_evidence),
                    "providerCallEvidenceSha256": _summary_hash(call_evidence),
                    "providerCallEvidence": copy.deepcopy(call_evidence),
                    "metrics": _delta(before_metrics, after_metrics),
                    "failureCodes": sorted(set(failure_codes)),
                    "failureRecovery": recovery,
                },
                "arbitration": {
                    "artifactSha256": canonical_json_sha256(arbitration) if arbitration is not None else None,
                    "status": arbitration.get("status") if isinstance(arbitration, Mapping) else None,
                    "requestCount": len(requests),
                    "requests": requests,
                },
                "composition": {
                    "status": "composition-complete" if composition is not None else None,
                    "artifactSha256": canonical_json_sha256(composition) if composition is not None else None,
                },
                "calibration": calibration,
                "comparison": comparison or {
                    "exactMatch": False,
                    "speaker": {"exactMatch": False, "mismatchCount": None},
                    "language": {"exactMatch": False, "mismatchCount": None},
                    "text": {"exactMatch": False, "mismatchCount": None},
                    "timeline": {"exactMatch": False, "mismatchCount": None},
                    "comparisonSha256": _summary_hash({"status": "unavailable", "failureCodes": sorted(set(failure_codes))}),
                },
            }
            row["caseComparisonSha256"] = row["comparison"].get("comparisonSha256")
            if comparison is not None:
                row["comparison"]["speakerExactMatch"] = bool(
                    comparison["speaker"]["exactMatch"]
                )
                row["comparison"]["languageExactMatch"] = bool(
                    comparison["language"]["exactMatch"]
                )
                row["comparison"]["textExactMatch"] = bool(
                    comparison["text"]["exactMatch"]
                )
                row["comparison"]["timelineExactMatch"] = bool(
                    comparison["timeline"]["exactMatch"]
                )
            rows.append(row)
            resource_snapshots.append(
                _probe_resource(
                    resource_probe,
                    spec,
                    f"after-case:{case.case_id}",
                )
            )
    except _FAILURE_TYPES as exc:
        model_failure = _failure_code(exc)
        model_failure_codes = _structured_failure_codes(exc)
    except Exception as exc:  # pragma: no cover - defensive boundary
        model_failure = type(exc).__name__
        model_failure_codes = _structured_failure_codes(exc)
    finally:
        if sampler_started:
            periodic_snapshots, sampler_stopped = sampler.stop()
            resource_snapshots.extend(periodic_snapshots)
        else:
            sampler_stopped = True
        resource_snapshots.append(
            _probe_resource(resource_probe, spec, "before-release")
        )
        if runner is not None:
            release_attempted = True
            try:
                runner.release_resources()
                release_succeeded = True
            except Exception as exc:
                release_failure_codes = _structured_failure_codes(exc)
                release_failure_code = (
                    "RESOURCE_RELEASE_METHOD_MISSING"
                    if "no resource release method" in str(exc).casefold()
                    else _failure_code(exc)
                )
        elif recording is not None or provider is not None:
            release_attempted = True
            try:
                release_target = recording or provider
                release = getattr(release_target, "release_resources", None)
                if not callable(release):
                    raise RealMultilingualSemanticBenchmarkError(
                        "semantic provider has no resource release method"
                    )
                release()
                release_succeeded = True
            except Exception as exc:
                release_failure_codes = _structured_failure_codes(exc)
                release_failure_code = (
                    "RESOURCE_RELEASE_METHOD_MISSING"
                    if "no resource release method" in str(exc).casefold()
                    else _failure_code(exc)
                )
        resource_snapshots.append(
            _probe_resource(resource_probe, spec, "after-release")
        )
    if recording is None:
        recording = (
            _RecordingProvider(
                provider,
                clock=clock,
                first_call_latency_class=first_call_latency_class,
                call_residency_probe=record_call_residency,
            )
            if provider is not None
            else None
        )
    model_metrics = _metrics(recording) if recording is not None else {}
    aggregate = _aggregate_rows(rows, cases=cases)
    resources = _resource_summary(
        resource_snapshots,
        expected_digest=spec.digest,
        sample_interval_seconds=resource_sample_interval_seconds,
        sampler_stopped=sampler_stopped,
    )
    all_calls = recording.calls if recording is not None else []
    call_latency = _call_latency_summary(all_calls)
    aggregate["providerCallLatency"] = call_latency
    aggregate["caseLatency"] = _case_latency_summary(rows)
    aggregate["failureRecovery"] = _aggregate_failure_recovery(rows)
    actual_digest = None
    actual_digest_source = None
    digest_evidence_authoritative = False
    observed_runtime_digests = resources["observedRuntimeDigests"]
    if resources["runtimeDigestObserved"] is True:
        actual_digest = (
            observed_runtime_digests[0]
            if len(observed_runtime_digests) == 1
            else None
        )
        actual_digest_source = (
            "ollama-api-ps"
            if default_provider_mode and resource_probe_kind == "ollama-api-ps"
            else "custom-resource-probe"
            if resource_probe_kind == "custom-resource-probe"
            else resource_probe_kind
        )
        digest_evidence_authoritative = bool(
            default_provider_mode
            and resource_probe_kind == "ollama-api-ps"
            and resources["runtimeDigestMatchesExpected"] is True
        )
    elif (
        default_provider_mode
        and resource_probe_kind == "ollama-api-ps"
        and recording is not None
        and digest_verified
    ):
        actual_digest = spec.digest
        actual_digest_source = "provider-inventory-verification"
        digest_evidence_authoritative = True
    post_release_model_absent = bool(resources["postReleaseModelAbsent"])
    release_transition_verified = bool(
        release_attempted
        and release_succeeded
        and resources["preReleaseOllamaApiAvailable"] is True
        and resources["preReleaseSelectedModelLoaded"] is True
        and post_release_model_absent
    )
    no_loaded_model_observed = not bool(resources["selectedModelLoadedObserved"])
    model_isolation_verified = bool(
        sampler_stopped
        and post_release_model_absent
        and (
            release_succeeded
            if release_attempted
            else no_loaded_model_observed
        )
    )
    isolation_failure_code = None
    if not sampler_stopped:
        isolation_failure_code = "RESOURCE_SAMPLER_DID_NOT_STOP"
    elif release_attempted and not release_succeeded:
        isolation_failure_code = release_failure_code or "RESOURCE_RELEASE_FAILED"
    elif resources["postReleaseSelectedModelLoaded"] is True:
        isolation_failure_code = "MODEL_STILL_RESIDENT_AFTER_RELEASE"
    elif not post_release_model_absent:
        isolation_failure_code = "MODEL_RESIDENCY_UNKNOWN_AFTER_RELEASE"
    elif not release_attempted and not no_loaded_model_observed:
        isolation_failure_code = "MODEL_RELEASE_NOT_ATTEMPTED"
    unverified_custom_provider = (
        not strict_isolation_required
        and resource_probe_kind == "custom-provider-resource-probe-unavailable"
        and (not release_attempted or release_succeeded)
    )
    case_failures = any(
        bool(row.get("execution", {}).get("failureCodes")) for row in rows
    )
    if review_material_sink is not None:
        missing_failure_codes = [
            *model_failure_codes,
            *([model_failure] if model_failure is not None else []),
            "MODEL_RESULT_NOT_PRODUCED",
        ]
        for case in cases:
            review_material_sink.setdefault(
                case.case_id,
                _blind_failure_result(missing_failure_codes),
            )
    execution_status = (
        "failed-to-initialize"
        if recording is None
        else "completed-with-failures"
        if model_failure is not None or case_failures or release_failure_code is not None
        else "completed"
    )
    return {
        "modelIdentity": spec.to_dict(),
        "provider": {
            "providerId": recording.provider_id if recording is not None else None,
            "providerVersion": recording.provider_version if recording is not None else None,
            "networkPolicy": recording.network_policy if recording is not None else None,
            "expectedDigest": spec.digest,
            "actualDigest": actual_digest,
            "actualDigestSource": actual_digest_source,
            "digestVerified": digest_evidence_authoritative,
            "digestEvidenceAuthoritative": digest_evidence_authoritative,
            "resourceProbeKind": resource_probe_kind,
            "runtimeDigestObserved": resources["runtimeDigestObserved"],
            "runtimeDigestMatchesExpected": resources[
                "runtimeDigestMatchesExpected"
            ],
        },
        "execution": {
            "status": execution_status,
            "failureCode": model_failure,
            "failureCodes": model_failure_codes,
            "wallTimeSeconds": max(0.0, float(clock() - started)),
            "providerCallCount": len(recording.calls) if recording is not None else 0,
            "providerMetrics": model_metrics,
            "providerCallLatency": call_latency,
            "resources": resources,
            "release": {
                "attempted": release_attempted,
                "succeeded": release_succeeded,
                "postReleaseModelAbsent": post_release_model_absent,
                "verifiedModelAbsent": post_release_model_absent,
                "transitionVerified": release_transition_verified,
                "releaseTransitionVerified": release_transition_verified,
                "failureCode": release_failure_code,
                "failureCodes": release_failure_codes,
            },
            "isolation": {
                "requiredBeforeNextModel": strict_isolation_required,
                "evidenceSource": resource_probe_kind,
                "verified": model_isolation_verified,
                "continuationAllowed": bool(
                    sampler_stopped
                    and (model_isolation_verified or unverified_custom_provider)
                ),
                "failureCode": isolation_failure_code,
            },
        },
        "aggregate": aggregate,
        "cases": rows,
        "validation": {
            "noReplace": True,
            "productionConfigRead": False,
            "productionConfigMutated": False,
            "promotionAuthorized": False,
            "allCasesHaveFrozenLattice": all(row["input"]["latticeSha256"] == cases[index].lattice_sha256 for index, row in enumerate(rows)) if rows else False,
            "resourceReleaseAttempted": release_attempted,
            "resourceReleaseSucceeded": release_succeeded,
            "postReleaseModelAbsent": post_release_model_absent,
            "resourceReleaseVerifiedAbsent": post_release_model_absent,
            "resourceReleaseTransitionVerified": release_transition_verified,
            "modelIsolationVerified": model_isolation_verified,
            "coldStartLatencyRecorded": call_latency["coldStartVerified"],
            "warmLatencyRecorded": call_latency["warmCallsObserved"],
            "automaticMetricsDiagnosticOnly": True,
        },
    }


def _case_latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = (
        "cold-start-case",
        "cold-start-unverified-case",
        "preloaded-first-case",
        "warm-case",
        "reload-case",
        "latency-unverified-case",
        "failed-provider-call-case",
        "no-provider-call",
    )
    return {
        "measurement": "case-wall-clock-v1",
        "byLatencyClass": {
            latency_class: _duration_distribution(
                [
                    row.get("execution", {}).get("wallTimeSeconds")
                    for row in rows
                    if row.get("execution", {}).get("latencyClass")
                    == latency_class
                ]
            )
            for latency_class in classes
        },
    }


def _aggregate_failure_recovery(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recoveries = [
        row.get("execution", {}).get("failureRecovery", {}) for row in rows
    ]
    validation_codes = Counter(
        str(code)
        for recovery in recoveries
        if isinstance(recovery, Mapping)
        for code in recovery.get("validationFailureCodes", [])
        if isinstance(code, str)
    )
    return {
        "caseCount": len(rows),
        "recoveryAttemptedCaseCount": sum(
            isinstance(recovery, Mapping)
            and recovery.get("recoveryAttempted") is True
            for recovery in recoveries
        ),
        "recoveredCaseCount": sum(
            isinstance(recovery, Mapping) and recovery.get("recovered") is True
            for recovery in recoveries
        ),
        "exhaustedCaseCount": sum(
            isinstance(recovery, Mapping) and recovery.get("exhausted") is True
            for recovery in recoveries
        ),
        "validationRetryCallCount": sum(
            int(recovery.get("validationRetryCallCount") or 0)
            for recovery in recoveries
            if isinstance(recovery, Mapping)
        ),
        "providerErrorCallCount": sum(
            int(recovery.get("providerErrorCallCount") or 0)
            for recovery in recoveries
            if isinstance(recovery, Mapping)
        ),
        "validationFailureCodeCounts": dict(sorted(validation_codes.items())),
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases: Sequence[FrozenSemanticCase],
) -> dict[str, Any]:
    fields = ("speaker", "language", "text", "segmentTimeline", "timeline")
    exact_counts = {
        field: sum(bool(row.get("comparison", {}).get(field, {}).get("exactMatch")) for row in rows)
        for field in fields
    }
    failure_counts = Counter(
        str(code)
        for row in rows
        for code in row.get("execution", {}).get("failureCodes", [])
    )
    request_count = sum(int(row.get("arbitration", {}).get("requestCount") or 0) for row in rows)
    composed_count = sum(row.get("composition", {}).get("status") == "composition-complete" for row in rows)
    comparison_hashes = [
        {"caseId": row.get("caseId"), "comparisonSha256": row.get("caseComparisonSha256")}
        for row in rows
    ]
    calibration_by_case = {
        str(row.get("caseId")): row.get("calibration")
        for row in rows
        if isinstance(row.get("calibration"), Mapping)
    }
    calibrations = [
        calibration_by_case.get(case.case_id)
        or _score_semantic_calibration(None, case.semantic_calibration_target)
        for case in cases
    ]
    return {
        "caseCount": len(rows),
        "compositionCompletedCount": composed_count,
        "candidateGenerationRequestCount": request_count,
        "failureCodeCounts": dict(sorted(failure_counts.items())),
        "exactMatchCounts": exact_counts,
        "exactMatchRates": {
            field: (exact_counts[field] / len(rows) if rows else None)
            for field in fields
        },
        "speakerExactMatchRate": exact_counts["speaker"] / len(rows) if rows else None,
        "languageExactMatchRate": exact_counts["language"] / len(rows) if rows else None,
        "textExactMatchRate": exact_counts["text"] / len(rows) if rows else None,
        "timelineExactMatchRate": exact_counts["timeline"] / len(rows) if rows else None,
        "allFourExactMatch": bool(rows) and all(
            bool(row.get("comparison", {}).get("exactMatch")) for row in rows
        ),
        "comparisonSetSha256": _summary_hash(comparison_hashes),
        "calibration": _aggregate_semantic_calibration(calibrations),
    }


def _assert_redacted(report: Mapping[str, Any], cases: Sequence[FrozenSemanticCase]) -> None:
    forbidden_keys = {
        "document", "lattice", "baseline", "rawtext", "normalizedtext", "finaltext",
        "prompt", "response", "requestbody", "responsebody", "transcripttext",
        *_HIDDEN_CALIBRATION_KEYS,
    }
    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise RealMultilingualSemanticBenchmarkError("benchmark report contains a persisted payload")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    inspect(report)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sensitive: set[str] = set()
    for case in cases:
        for segment in case.document.get("segments", []):
            if isinstance(segment, Mapping):
                for key in ("rawText", "normalizedText", "displayText"):
                    value = segment.get(key)
                    if isinstance(value, str) and len(value) >= 4:
                        sensitive.add(value)
        for segment in case.baseline.get("segments", []):
            if isinstance(segment, Mapping):
                for key in ("finalText", "normalizedText", "text"):
                    value = segment.get(key)
                    if isinstance(value, str) and len(value) >= 4:
                        sensitive.add(value)
    for text in sensitive:
        if text in serialized:
            raise RealMultilingualSemanticBenchmarkError("benchmark report leaks transcript text")


def benchmark_real_multilingual_semantic_models(
    cases: FrozenSemanticCaseSet | Sequence[FrozenSemanticCase],
    model_specs: Sequence[SemanticModelSpec | Mapping[str, Any]],
    *,
    endpoint: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 600.0,
    context_tokens: int = 32_768,
    output_tokens: int = 4_096,
    batch_size: int = 8,
    max_batch_attempts: int = 2,
    provider_factory: Callable[[SemanticModelSpec], Any] | None = None,
    resource_probe: ResourceProbe | None = None,
    resource_sample_interval_seconds: float = 0.25,
    blind_review_output_root: Path | None = None,
    blind_seed: str | None = None,
    now: Callable[[], str] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run all pinned models serially and return a no-replace report."""

    if (blind_review_output_root is None) != (blind_seed is None):
        raise RealMultilingualSemanticBenchmarkError(
            "blind_review_output_root and blind_seed must be provided together"
        )

    if isinstance(cases, FrozenSemanticCaseSet):
        case_set = cases
    else:
        normalized_cases = tuple(cases)
        if not normalized_cases or not all(isinstance(item, FrozenSemanticCase) for item in normalized_cases):
            raise RealMultilingualSemanticBenchmarkError("benchmark cases must be validated frozen cases")
        case_set = FrozenSemanticCaseSet(
            cases=normalized_cases,
            manifest_sha256=_summary_hash([item.input_binding() for item in normalized_cases]),
            manifest_canonical_sha256=_summary_hash([item.input_binding() for item in normalized_cases]),
            manifest_path=None,
            schema_version=SCHEMA_VERSION,
        )
    specs: list[SemanticModelSpec] = []
    for item in model_specs:
        specs.append(item if isinstance(item, SemanticModelSpec) else SemanticModelSpec(model=item.get("model"), digest=item.get("digest") or item.get("expectedDigest"), model_id=item.get("modelId"), config_path=item.get("configPath")))
    specs_tuple = _unique_model_specs(specs)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 32:
        raise RealMultilingualSemanticBenchmarkError("batch_size must be between 1 and 32")
    if not isinstance(max_batch_attempts, int) or isinstance(max_batch_attempts, bool) or not 1 <= max_batch_attempts <= 3:
        raise RealMultilingualSemanticBenchmarkError("max_batch_attempts must be between 1 and 3")
    if (
        isinstance(resource_sample_interval_seconds, bool)
        or not isinstance(resource_sample_interval_seconds, (int, float))
        or not math.isfinite(float(resource_sample_interval_seconds))
        or float(resource_sample_interval_seconds) <= 0.0
        or float(resource_sample_interval_seconds) > 60.0
    ):
        raise RealMultilingualSemanticBenchmarkError(
            "resource_sample_interval_seconds must be finite and between 0 and 60"
        )
    resource_sample_interval_seconds = float(resource_sample_interval_seconds)
    blind_target: Path | None = None
    normalized_blind_seed: str | None = None
    if blind_review_output_root is not None and blind_seed is not None:
        normalized_blind_seed = _text(
            blind_seed,
            field="blind_seed",
            maximum=4_096,
        )
        blind_target = Path(blind_review_output_root).expanduser().absolute()
        blind_target.parent.mkdir(parents=True, exist_ok=True)
        if blind_target.exists() or blind_target.is_symlink():
            raise FileExistsError(
                f"blind review output already exists: {blind_target}"
            )
        _preflight_blind_directory_publish(blind_target)
    now_fn = now or (lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    clock_fn = clock or time.perf_counter
    custom_provider = provider_factory is not None
    default_provider_mode = not custom_provider
    default_resource_probe = resource_probe is None and default_provider_mode
    factory = provider_factory or (lambda spec: _default_provider_factory(spec, endpoint=endpoint, timeout_seconds=timeout_seconds, context_tokens=context_tokens, output_tokens=output_tokens))
    probe = resource_probe or (
        _unavailable_resource_probe
        if custom_provider
        else lambda spec, phase: _default_resource_probe(
            endpoint,
            spec,
            phase,
        )
    )
    resource_probe_kind = (
        "ollama-api-ps"
        if default_resource_probe
        else "custom-provider-resource-probe-unavailable"
        if resource_probe is None
        else "custom-resource-probe"
    )
    model_reports: list[dict[str, Any]] = []
    review_material: dict[str, dict[str, dict[str, Any]]] = {}
    stop_reason: str | None = None
    stopped_after_model: str | None = None
    for spec in specs_tuple:
        # A fresh deep copy protects the frozen source from a faulty provider or
        # future runner change while retaining the exact same lattice hash.
        isolated_cases = tuple(
            FrozenSemanticCase(
                case_id=case.case_id,
                language=case.language,
                document=copy.deepcopy(case.document),
                lattice=copy.deepcopy(case.lattice),
                baseline=copy.deepcopy(case.baseline),
                semantic_calibration_target=copy.deepcopy(
                    case.semantic_calibration_target
                ),
                document_sha256=case.document_sha256,
                lattice_sha256=case.lattice_sha256,
                baseline_sha256=case.baseline_sha256,
                baseline_canonical_sha256=case.baseline_canonical_sha256,
                baseline_audit_source=case.baseline_audit_source,
                case_sha256=case.case_sha256,
                manifest_case_index=case.manifest_case_index,
            )
            for case in case_set.cases
        )
        model_review_material: dict[str, dict[str, Any]] | None = (
            {} if blind_target is not None else None
        )
        if model_review_material is not None:
            review_material[spec.model] = model_review_material
        model_report = _model_report(
            spec,
            isolated_cases,
            provider_factory=factory,
            resource_probe=probe,
            resource_probe_kind=resource_probe_kind,
            default_provider_mode=default_provider_mode,
            strict_isolation_required=default_provider_mode,
            resource_sample_interval_seconds=resource_sample_interval_seconds,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            batch_size=batch_size,
            max_batch_attempts=max_batch_attempts,
            clock=clock_fn,
            review_material_sink=model_review_material,
        )
        model_reports.append(model_report)
        if model_report["execution"]["isolation"]["continuationAllowed"] is not True:
            stop_reason = (
                model_report["execution"]["isolation"]["failureCode"]
                or "MODEL_ISOLATION_UNVERIFIED"
            )
            stopped_after_model = spec.model
            break
    skipped_specs = specs_tuple[len(model_reports):]
    if blind_target is not None:
        for spec in skipped_specs:
            review_material[spec.model] = {
                case.case_id: _blind_not_executed_result(stop_reason)
                for case in case_set.cases
            }
    binding = case_set.binding()
    stable_models = [
        {
            "modelIdentity": report["modelIdentity"],
            "qualityAggregate": {
                key: value
                for key, value in report["aggregate"].items()
                if key not in {"providerCallLatency", "caseLatency"}
            },
            "artifactSafety": {
                key: report["validation"][key]
                for key in (
                    "noReplace",
                    "productionConfigRead",
                    "productionConfigMutated",
                    "promotionAuthorized",
                    "allCasesHaveFrozenLattice",
                )
            },
        }
        for report in model_reports
    ]
    report_body: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "benchmark": "real-multilingual-semantic-models",
        "generatedAt": _text(now_fn(), field="generatedAt", maximum=100),
        "promptVersion": SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION,
        "input": binding,
        "executionConfiguration": {
            "providerMode": (
                "default-ollama" if default_provider_mode else "custom-provider"
            ),
            "providerConfigurationSource": (
                "benchmark-default-ollama-config"
                if default_provider_mode
                else "custom-provider-unverified"
            ),
            "requestedEndpoint": endpoint,
            "endpoint": endpoint if default_provider_mode else None,
            "endpointApplied": default_provider_mode,
            "requestedTimeoutSeconds": timeout_seconds,
            "timeoutSeconds": timeout_seconds if default_provider_mode else None,
            "timeoutAppliedByDefaultProvider": default_provider_mode,
            "contextTokens": context_tokens,
            "outputTokens": output_tokens,
            "batchSize": batch_size,
            "maxBatchAttempts": max_batch_attempts,
            "temperature": 0.0,
            "temperatureAppliedByRunner": True,
            "topP": 0.1 if default_provider_mode else None,
            "topPAppliedByDefaultProvider": default_provider_mode,
            "keepAliveWithinModelBatch": (
                "10m" if default_provider_mode else None
            ),
            "keepAliveAppliedByDefaultProvider": default_provider_mode,
            "releaseOnCloseAppliedByDefaultProvider": default_provider_mode,
            "modelExecution": "serial",
            "freshProviderPerModel": True,
            "explicitReleaseRequestedAfterEveryModel": True,
            "resourceSampleIntervalSeconds": resource_sample_interval_seconds,
        },
        "measurementPolicy": {
            "coldCall": "successful-first-provider-call-when-immediate-pre-call-probe-reports-model-absent",
            "warmCall": "successful-provider-call-when-immediate-pre-call-probe-reports-model-resident",
            "reloadCall": "successful-non-first-provider-call-when-immediate-pre-call-probe-reports-model-absent",
            "failedCallClassification": "failed-before-classification",
            "preloadedFirstCall": "reported-separately-and-never-labeled-cold",
            "providerNativeDurationsRecorded": True,
            "providerWallClockDurationsRecorded": True,
            "resourceSampling": "periodic-plus-before-run-before-each-provider-call-after-each-case-before-release-after-release",
            "resourceMaximumIsObservedSnapshotMaximum": True,
            "continuousProcessPeakClaimed": False,
            "failureRecovery": "strict-runner-retry-without-validation-relaxation",
            "rejectedResponseBodiesPersisted": False,
        },
        "evaluationPolicy": {
            "frozenDocumentAndLattice": True,
            "sameLatticeForEveryModel": True,
            "baselineAuthority": sorted({case.baseline_audit_source for case in case_set.cases}),
            "automaticScoresAreDiagnosticOnly": True,
            "automaticLatencyAndResourceMetricsAreDiagnosticOnly": True,
            "replacementPolicy": NO_REPLACE_POLICY,
            "productionConfigRead": False,
            "productionConfigMutated": False,
            "winnerSelection": "codex-same-batch-blind-review-required",
            "productionPointerPolicy": "active-pointer-external-cas",
            "challengerMayReplaceProductionImmediatelyAfterBlindWin": True,
            "incumbentProtection": False,
            "promotionMarginRequired": False,
            "promotionCooldownRequired": False,
            "registryStatusGateRequired": False,
        },
        "models": model_reports,
        "execution": {
            "requestedModelCount": len(specs_tuple),
            "executedModelCount": len(model_reports),
            "skippedModelCount": len(skipped_specs),
            "requestedModelOrder": [spec.model for spec in specs_tuple],
            "executedModelOrder": [
                report["modelIdentity"]["model"] for report in model_reports
            ],
            "skippedModelOrder": [spec.model for spec in skipped_specs],
            "stoppedEarly": bool(skipped_specs),
            "stopReason": stop_reason,
            "stoppedAfterModel": stopped_after_model,
        },
        "comparison": {
            "modelCount": len(model_reports),
            "requestedModelCount": len(specs_tuple),
            "executedModelCount": len(model_reports),
            "skippedModelCount": len(skipped_specs),
            "modelOrder": [
                report["modelIdentity"]["model"] for report in model_reports
            ],
            "diagnosticColumns": [
                "speakerExactMatch",
                "languageExactMatch",
                "textExactMatch",
                "timelineExactMatch",
                "coldWarmLatency",
                "failureRecovery",
                "semanticCalibrationMicroAccuracy",
                "processRamVram",
                "resourceRelease",
                "modelDigest",
            ],
            "stableComparisonSha256": _summary_hash(stable_models),
            "winner": None,
            "promotionReceipt": None,
        },
        "validation": {
            "allModelsIndependent": bool(model_reports) and all(
                item["validation"]["modelIsolationVerified"] is True
                for item in model_reports
            ),
            "allRequestedModelsExecuted": not skipped_specs,
            "allModelsNoReplace": all(bool(item["validation"]["noReplace"]) for item in model_reports),
            "allCasesFrozen": True,
            "productionConfigUntouched": True,
            "promotionAuthorized": False,
            "passed": all(bool(item["validation"]["noReplace"]) for item in model_reports),
            "passedMeaning": "artifact-safety-and-no-production-mutation-only",
        },
    }
    _assert_redacted(report_body, case_set.cases)
    if blind_target is not None and normalized_blind_seed is not None:
        report_body["blindReviewPublication"] = (
            _publish_blind_review_package_no_replace(
                blind_target,
                case_set.cases,
                specs_tuple,
                review_material,
                blind_seed=normalized_blind_seed,
            )
        )
    _assert_redacted(report_body, case_set.cases)
    report_body["canonicalSha256"] = canonical_json_sha256(report_body)
    return report_body


run_benchmark = benchmark_real_multilingual_semantic_models


def publish_model_reports_no_replace(
    report: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, str]:
    """Publish one immutable sidecar per model without replacing a file.

    The summary report remains the source of the run identity.  These sidecars
    are useful when a large comparison is resumed or inspected per model, and
    deliberately contain no production pointer or promotion operation.
    """

    if report.get("artifactType") != ARTIFACT_TYPE:
        raise RealMultilingualSemanticBenchmarkError("report artifact type is invalid")
    models = report.get("models")
    if not isinstance(models, list) or not models:
        raise RealMultilingualSemanticBenchmarkError("report has no model results")
    root = output_directory.expanduser().resolve(strict=False)
    published: dict[str, str] = {}
    for index, model in enumerate(models):
        if not isinstance(model, Mapping):
            raise RealMultilingualSemanticBenchmarkError("model result is invalid")
        identity = model.get("modelIdentity")
        if not isinstance(identity, Mapping):
            raise RealMultilingualSemanticBenchmarkError("model identity is missing")
        model_id = _text(identity.get("modelId"), field="model result modelId", maximum=200)
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._")
        if not safe_id:
            safe_id = f"model-{index + 1:02d}"
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "real-multilingual-semantic-model-result",
            "benchmarkArtifactSha256": report.get("canonicalSha256"),
            "model": dict(model),
            "replacementPolicy": NO_REPLACE_POLICY,
            "productionConfigMutated": False,
        }
        path = root / f"{index + 1:02d}-{safe_id}.json"
        atomic_write_json_no_replace(path, payload)
        published[model_id] = str(path)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", "--manifest", dest="cases", required=True, type=Path)
    parser.add_argument("--config-set-manifest", type=Path)
    parser.add_argument("--model", action="append", default=[], help="MODEL=SHA256 (repeatable)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--context-tokens", type=int, default=32_768)
    parser.add_argument("--output-tokens", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-attempts", type=int, default=2)
    parser.add_argument(
        "--resource-sample-interval-seconds",
        type=float,
        default=0.25,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-output-directory",
        type=Path,
        help="optionally publish one immutable no-replace JSON sidecar per model",
    )
    parser.add_argument(
        "--blind-review-output-root",
        type=Path,
        help="optionally publish a complete anonymous reviewer packet and identity vault",
    )
    parser.add_argument(
        "--blind-seed",
        help="secret seed used only to deterministically randomize per-case candidate aliases",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.blind_review_output_root is None) != (args.blind_seed is None):
        parser.error(
            "--blind-review-output-root and --blind-seed must be provided together"
        )
    case_set = load_frozen_semantic_cases(args.cases)
    if args.model:
        specs = parse_model_specs(args.model)
    elif args.config_set_manifest is not None:
        specs = load_model_specs_from_manifest(args.config_set_manifest)
    else:
        raise SystemExit("--config-set-manifest or at least one --model is required")
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        specs,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        batch_size=args.batch_size,
        max_batch_attempts=args.max_batch_attempts,
        resource_sample_interval_seconds=args.resource_sample_interval_seconds,
        blind_review_output_root=args.blind_review_output_root,
        blind_seed=args.blind_seed,
    )
    atomic_write_json_no_replace(args.output.resolve(), report)
    sidecars = (
        publish_model_reports_no_replace(report, args.model_output_directory)
        if args.model_output_directory is not None
        else {}
    )
    print(json.dumps({"output": str(args.output.resolve()), "canonicalSha256": report["canonicalSha256"], "modelOutputs": sidecars, "validation": report["validation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
