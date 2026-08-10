#!/usr/bin/env python3
"""Freeze real multilingual semantic evidence without running any models.

The emitted artifact has two deliberately separate surfaces:

* ``cases`` is directly consumable by
  ``benchmark_real_multilingual_semantic_models.py`` and contains only
  hash-bound production documents, candidate lattices, and manually reviewed
  final baselines.
* ``inventory`` records real development audio with reference text that is not
  yet eligible for that benchmark.  Reference text is represented by hashes
  and a source-manifest locator; it is never promoted into a synthetic manual
  baseline.

The tool only reads existing artifacts and publishes one no-replace JSON file.
It does not contact Ollama, load a model, or read/write production config.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.review import (  # noqa: E402
    assert_raw_text_unchanged,
    merge_speakers,
    open_count,
    resolve_review_item,
    validate_review_state,
)
from backend.semantic_candidate_lattice import (  # noqa: E402
    validate_semantic_candidate_lattice,
)
from tools.run_production_smoke import (  # noqa: E402
    ReviewDecisionPlan,
    load_review_decision_plan,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_TYPE = "real-multilingual-semantic-development-freeze"
PRODUCTION_STATUS = "production-lattice-human-baseline"
REFERENCE_ONLY_STATUS = "audio-reference-only"
UNAVAILABLE_STATUS = "unavailable"

DEFAULT_PRODUCTION_CASE_IDS = (
    "minds_zh_cn_258",
    "minds_en_us_034",
    "minds_fr_fr_089",
    "fleurs_id_id_validation_003",
)
DEFAULT_REFERENCE_LANGUAGES = (
    "yue-Hant-HK",
    "es-419",
    "ja-JP",
    "ko-KR",
    "ar-EG",
    "hi-IN",
)
TARGET_LANGUAGES = (
    "zh-CN",
    "yue-Hant-HK",
    "en-US",
    "es-419",
    "fr-FR",
    "id-ID",
    "ja-JP",
    "ko-KR",
    "ar-EG",
    "hi-IN",
)
_LANGUAGE_ORDER = {language: index for index, language in enumerate(TARGET_LANGUAGES)}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANUAL_AUDIT_SOURCES = frozenset({"human", "codex-manual", "codex-agent"})
_KNOWN_RUN_LIMITATIONS = {
    "pilot-20260809-qwen3.5-9b-r22-multilingual-semantic-baseline": [
        "semantic composition and manual adjudication completed",
        "PDF/publication did not complete; that downstream failure is excluded from semantic-model scoring",
    ],
    "pilot-20260809-qwen3.5-9b-r23-multilingual-semantic-baseline": [
        "semantic composition and manual adjudication completed",
        "PDF/publication failed after semantic completion; that downstream failure is excluded from semantic-model scoring",
    ],
}


class RealMultilingualSemanticFreezeError(ValueError):
    """Raised when local evidence cannot be frozen without ambiguity."""


@dataclass(frozen=True)
class CandidateLatticeSelection:
    path: Path
    lattice_sha256: str
    selected_kind: str
    selected_round: int | None
    expanded_round_count: int
    initial_path: Path
    initial_lattice_sha256: str


@dataclass(frozen=True)
class ReviewReplayProvenance:
    decision_set_directory: Path
    decision_set_manifest_path: Path
    decision_set_manifest_file_sha256: str
    decision_set_manifest_canonical_sha256: str
    decision_path: Path
    decision_file_sha256: str
    decision_canonical_sha256: str
    source_open_queue_file_sha256: str
    source_open_queue_canonical_sha256: str
    source_run_root: Path
    source_output_directory: Path


@dataclass(frozen=True)
class DecisionSetCaseEvidence:
    case_id: str
    decision_path: Path
    decision_file_sha256: str
    decision_canonical_sha256: str
    source_run_root_name: str
    source_output_name: str
    source_input_file_sha256: str
    source_input_canonical_sha256: str
    source_queue_file_sha256: str
    source_queue_canonical_sha256: str
    source_media_sha256: str
    observed_open_item_ids: tuple[str, ...]
    replay_decision_ids: tuple[str, ...]
    replay_decision_count: int
    replay_open_count: int


@dataclass(frozen=True)
class ProductionEvidence:
    case_id: str
    language: str
    source_evaluation_split: str
    run_id: str
    final_generated_at: str
    output_directory: Path
    audio_path: Path
    benchmark_document_path: Path
    benchmark_lattice_path: Path
    reviewed_document_path: Path
    final_artifact_path: Path | None
    review_queue_path: Path
    benchmark_document_sha256: str
    benchmark_lattice_sha256: str
    reviewed_document_sha256: str
    reviewed_lattice_sha256: str | None
    lattice_selection: CandidateLatticeSelection
    manual_baseline: dict[str, Any]
    manual_baseline_canonical_sha256: str
    final_artifact_file_sha256: str | None
    final_artifact_canonical_sha256: str | None
    final_artifact_matches_manual_baseline: bool | None
    review_queue_sha256: str
    source_media_sha256: str
    audit_source: str
    review_actor_ids: tuple[str, ...]
    review_decision_count: int
    evidence_kind: str
    review_replay: ReviewReplayProvenance | None


def _default_eval_root() -> Path:
    configured = os.environ.get("MTS_EVAL_ROOT")
    if configured:
        return Path(configured)
    if os.name == "nt":
        return Path("D:/mts-eval")
    return Path("/mnt/d/mts-eval")


def _text(value: Any, *, field: str, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealMultilingualSemanticFreezeError(f"{field} must be non-empty text")
    result = value.strip()
    if len(result) > maximum:
        raise RealMultilingualSemanticFreezeError(f"{field} is too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in result):
        raise RealMultilingualSemanticFreezeError(f"{field} contains control characters")
    return result


def _sha(value: Any, *, field: str) -> str:
    result = _text(value, field=field, maximum=80).casefold()
    if result.startswith("sha256:"):
        result = result.removeprefix("sha256:")
    if _SHA256.fullmatch(result) is None:
        raise RealMultilingualSemanticFreezeError(f"{field} must be SHA-256")
    return result


def _load_object(path: Path, *, field: str) -> tuple[Path, dict[str, Any]]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise RealMultilingualSemanticFreezeError(f"{field} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RealMultilingualSemanticFreezeError(f"{field} is missing: {path}") from exc
    if not resolved.is_file():
        raise RealMultilingualSemanticFreezeError(f"{field} must be a regular file")
    try:
        value = read_json_strict(resolved)
    except Exception as exc:
        raise RealMultilingualSemanticFreezeError(f"{field} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise RealMultilingualSemanticFreezeError(f"{field} must contain an object")
    return resolved, dict(value)


def _digest_sidecar(path: Path, *, expected: str, field: str) -> None:
    if path.is_symlink():
        raise RealMultilingualSemanticFreezeError(f"{field} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RealMultilingualSemanticFreezeError(f"{field} is missing") from exc
    if not resolved.is_file():
        raise RealMultilingualSemanticFreezeError(f"{field} must be a regular file")
    tokens = resolved.read_text(encoding="utf-8").strip().split()
    if not tokens or tokens[0].casefold() != expected.casefold():
        raise RealMultilingualSemanticFreezeError(f"{field} does not match")


def _safe_decision_path(root: Path, raw: Any, *, field: str) -> Path:
    relative = _text(raw, field=field, maximum=1000)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RealMultilingualSemanticFreezeError(
            f"{field} must be a relative path within the decision set"
        )
    unresolved = root / candidate
    if unresolved.is_symlink():
        raise RealMultilingualSemanticFreezeError(
            f"{field} must resolve to a regular non-symlink file"
        )
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file():
        raise RealMultilingualSemanticFreezeError(
            f"{field} must resolve to a regular non-symlink file"
        )
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise RealMultilingualSemanticFreezeError(
            f"{field} escapes the decision set"
        ) from exc
    return resolved


def _decision_set_replay_threshold(manifest: Mapping[str, Any]) -> float:
    replay = manifest.get("replay")
    if not isinstance(replay, Mapping):
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set replay settings are missing"
        )
    if replay.get("api") != (
        "backend.review.validate_review_state then backend.review.resolve_review_item"
    ):
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set replay API is unsupported"
        )
    threshold = replay.get("highMarginThreshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set replay highMarginThreshold is invalid"
        )
    if replay.get("allCaseOpenCountAfterReplay") != 0:
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set does not declare a closed replay"
        )
    return float(threshold)


def _load_decision_set(
    directory: Path,
) -> tuple[Path, dict[str, Any], dict[str, DecisionSetCaseEvidence]]:
    root = directory.expanduser()
    if root.is_symlink():
        raise RealMultilingualSemanticFreezeError("decision set must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise RealMultilingualSemanticFreezeError(
            f"decision set is missing: {directory}"
        ) from exc
    if not root.is_dir():
        raise RealMultilingualSemanticFreezeError("decision set must be a directory")
    manifest_path, manifest = _load_object(
        root / "MANIFEST.v1.json", field="Codex decision-set manifest"
    )
    if manifest.get("schemaVersion") != "1.0.0":
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set manifest schemaVersion is unsupported"
        )
    if manifest.get("artifactType") != "fleurs-post-reference-codex-review-decision-set":
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set manifest artifactType is invalid"
        )
    if manifest.get("setId") != "fleurs18-post-reference-manual-decisions-r2":
        raise RealMultilingualSemanticFreezeError(
            "freeze requires the immutable r2 Codex decision set"
        )
    _decision_set_replay_threshold(manifest)
    _validate_declared_canonical(manifest, field="Codex decision-set manifest")
    manifest_file_sha256 = sha256_file(manifest_path)
    _digest_sidecar(
        root / "MANIFEST.v1.json.sha256",
        expected=manifest_file_sha256,
        field="Codex decision-set manifest file SHA sidecar",
    )
    canonical_sha256 = canonical_json_sha256(manifest)
    _digest_sidecar(
        root / "MANIFEST.v1.canonical.sha256",
        expected=canonical_sha256,
        field="Codex decision-set manifest canonical SHA sidecar",
    )
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set manifest cases are missing"
        )
    result: dict[str, DecisionSetCaseEvidence] = {}
    required = {
        "caseId",
        "decisionPath",
        "decisionFileSha256",
        "decisionCanonicalSha256",
        "sourceRunRoot",
        "sourceOutputDirectory",
        "sourceInputFileSha256",
        "sourceInputCanonicalSha256",
        "sourceQueueFileSha256",
        "sourceQueueCanonicalSha256",
        "sourceMediaSha256",
        "observedOpenItemIds",
        "observedQueueOpenCount",
        "replayDecisionIds",
        "replayDecisionCount",
        "replayOpenCount",
        "replayApi",
    }
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping) or not required <= set(raw):
            raise RealMultilingualSemanticFreezeError(
                f"Codex decision-set cases[{index}] fields are incomplete"
            )
        case_id = _text(
            raw.get("caseId"), field=f"decision-set cases[{index}].caseId", maximum=200
        )
        if case_id in result:
            raise RealMultilingualSemanticFreezeError(
                "Codex decision-set case IDs must be unique"
            )
        decision_path = _safe_decision_path(
            root, raw.get("decisionPath"), field=f"decision-set {case_id}.decisionPath"
        )
        decision_file_sha256 = _sha(
            raw.get("decisionFileSha256"),
            field=f"decision-set {case_id}.decisionFileSha256",
        )
        if sha256_file(decision_path) != decision_file_sha256:
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id} decision file SHA does not match"
            )
        _, decision_value = _load_object(
            decision_path, field=f"decision-set {case_id} decision file"
        )
        decision_canonical_sha256 = _sha(
            raw.get("decisionCanonicalSha256"),
            field=f"decision-set {case_id}.decisionCanonicalSha256",
        )
        if canonical_json_sha256(decision_value) != decision_canonical_sha256:
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id} decision canonical SHA does not match"
            )
        source_run_root = _text(
            raw.get("sourceRunRoot"),
            field=f"decision-set {case_id}.sourceRunRoot",
            maximum=1000,
        )
        source_output = _text(
            raw.get("sourceOutputDirectory"),
            field=f"decision-set {case_id}.sourceOutputDirectory",
            maximum=300,
        )
        open_ids = raw.get("observedOpenItemIds")
        if (
            not isinstance(open_ids, list)
            or not open_ids
            or any(not isinstance(item, str) or not item.strip() for item in open_ids)
            or len(set(open_ids)) != len(open_ids)
        ):
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id}.observedOpenItemIds is invalid"
            )
        if raw.get("observedQueueOpenCount") != len(open_ids):
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id}.observedQueueOpenCount does not match its item IDs"
            )
        replay_ids = raw.get("replayDecisionIds")
        if (
            not isinstance(replay_ids, list)
            or not replay_ids
            or any(not isinstance(item, str) or not item.strip() for item in replay_ids)
            or len(set(replay_ids)) != len(replay_ids)
        ):
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id}.replayDecisionIds is invalid"
            )
        replay_decision_count = raw.get("replayDecisionCount")
        if (
            isinstance(replay_decision_count, bool)
            or not isinstance(replay_decision_count, int)
            or replay_decision_count != len(replay_ids)
        ):
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id}.replayDecisionCount does not match its decision IDs"
            )
        replay_open_count = raw.get("replayOpenCount")
        if (
            isinstance(replay_open_count, bool)
            or not isinstance(replay_open_count, int)
            or replay_open_count != 0
        ):
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id}.replayOpenCount must be zero"
            )
        if raw.get("replayApi") != (
            "backend.review.validate_review_state+resolve_review_item"
        ):
            raise RealMultilingualSemanticFreezeError(
                f"decision-set {case_id}.replayApi is unsupported"
            )
        result[case_id] = DecisionSetCaseEvidence(
            case_id=case_id,
            decision_path=decision_path,
            decision_file_sha256=decision_file_sha256,
            decision_canonical_sha256=decision_canonical_sha256,
            source_run_root_name=Path(source_run_root.replace("\\", "/")).name,
            source_output_name=Path(source_output.replace("\\", "/")).name,
            source_input_file_sha256=_sha(
                raw.get("sourceInputFileSha256"),
                field=f"decision-set {case_id}.sourceInputFileSha256",
            ),
            source_input_canonical_sha256=_sha(
                raw.get("sourceInputCanonicalSha256"),
                field=f"decision-set {case_id}.sourceInputCanonicalSha256",
            ),
            source_queue_file_sha256=_sha(
                raw.get("sourceQueueFileSha256"),
                field=f"decision-set {case_id}.sourceQueueFileSha256",
            ),
            source_queue_canonical_sha256=_sha(
                raw.get("sourceQueueCanonicalSha256"),
                field=f"decision-set {case_id}.sourceQueueCanonicalSha256",
            ),
            source_media_sha256=_sha(
                raw.get("sourceMediaSha256"),
                field=f"decision-set {case_id}.sourceMediaSha256",
            ),
            observed_open_item_ids=tuple(str(item) for item in open_ids),
            replay_decision_ids=tuple(str(item) for item in replay_ids),
            replay_decision_count=replay_decision_count,
            replay_open_count=replay_open_count,
        )
    return manifest_path, manifest, result


def _validate_declared_canonical(value: Mapping[str, Any], *, field: str) -> None:
    declared = value.get("canonicalSha256")
    if declared is None:
        return
    body = dict(value)
    body.pop("canonicalSha256", None)
    if _sha(declared, field=f"{field}.canonicalSha256") != canonical_json_sha256(body):
        raise RealMultilingualSemanticFreezeError(
            f"{field} canonical SHA-256 does not match"
        )


def _relative(path: Path, output_directory: Path) -> str:
    try:
        return Path(os.path.relpath(path, output_directory)).as_posix()
    except ValueError:
        return path.as_posix()


def _json_artifact(path: Path, output_directory: Path) -> dict[str, Any]:
    resolved, value = _load_object(path, field=str(path))
    return {
        "path": _relative(resolved, output_directory),
        "fileSha256": sha256_file(resolved),
        "canonicalSha256": canonical_json_sha256(value),
    }


def _audio_artifact(
    path: Path,
    output_directory: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise RealMultilingualSemanticFreezeError("audio evidence must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RealMultilingualSemanticFreezeError(f"audio evidence is missing: {path}") from exc
    if not resolved.is_file():
        raise RealMultilingualSemanticFreezeError("audio evidence must be a regular file")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"audio SHA-256 does not match: {resolved}"
        )
    return {
        "path": _relative(resolved, output_directory),
        "bytes": resolved.stat().st_size,
        "sha256": actual,
    }


def _find_audio(project_root: Path, case_id: str) -> Path:
    audio_root = (
        project_root
        / ".runtime_cache"
        / "sample-library"
        / "global"
        / "audio"
    )
    matches = sorted(
        path for path in audio_root.glob(f"{case_id}.*") if path.is_file()
    )
    if len(matches) != 1:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} must resolve to exactly one local audio file"
        )
    return matches[0]


def _global_case_index(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise RealMultilingualSemanticFreezeError("global manifest cases are missing")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RealMultilingualSemanticFreezeError(
                f"global manifest cases[{index}] must be an object"
            )
        case_id = _text(row.get("id"), field=f"global cases[{index}].id", maximum=200)
        if case_id in result:
            raise RealMultilingualSemanticFreezeError("global case IDs must be unique")
        result[case_id] = dict(row)
    return result


def _reference_truth_index(
    value: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise RealMultilingualSemanticFreezeError("FLEURS reference cases are missing")
    result: dict[str, tuple[str, ...]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RealMultilingualSemanticFreezeError(
                f"FLEURS reference cases[{index}] must be an object"
            )
        case_id = _text(
            row.get("id"), field=f"FLEURS reference cases[{index}].id", maximum=200
        )
        if case_id in result:
            raise RealMultilingualSemanticFreezeError(
                "FLEURS reference case IDs must be unique"
            )
        truths = {
            " ".join(raw.strip().split()).casefold()
            for key in (
                "scoringTranscript",
                "expectedTranscript",
                "nativeTranscript",
                "rawTranscript",
            )
            if isinstance((raw := row.get(key)), str) and raw.strip()
        }
        result[case_id] = tuple(sorted(truths))
    return result


def _assert_truth_redacted(
    value: Any,
    *,
    reference_truths: Sequence[str],
    field: str,
) -> None:
    del reference_truths
    forbidden_tokens = (
        "reference",
        "truth",
        "scoringtranscript",
        "expectedtranscript",
        "nativetranscript",
        "rawtranscript",
    )

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                key_folded = str(key).casefold()
                if any(token in key_folded for token in forbidden_tokens):
                    raise RealMultilingualSemanticFreezeError(
                        f"{field} contains a forbidden reference-truth field"
                    )
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)


def _review_authority(queue: Mapping[str, Any]) -> tuple[str, tuple[str, ...], int]:
    if queue.get("openCount") != 0:
        raise RealMultilingualSemanticFreezeError("review queue is not closed")
    items = queue.get("items")
    decisions = queue.get("decisions")
    if not isinstance(items, list) or not items:
        raise RealMultilingualSemanticFreezeError("review queue has no review items")
    if not isinstance(decisions, list) or not decisions:
        raise RealMultilingualSemanticFreezeError("review queue has no decisions")
    if any(
        not isinstance(item, Mapping)
        or item.get("status") not in {"accepted", "rejected"}
        for item in items
    ):
        raise RealMultilingualSemanticFreezeError(
            "review queue contains unresolved items"
        )
    sources: set[str] = set()
    actors: set[str] = set()
    decision_ids: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise RealMultilingualSemanticFreezeError(
                f"review decisions[{index}] must be an object"
            )
        audit = decision.get("audit")
        if not isinstance(audit, Mapping):
            raise RealMultilingualSemanticFreezeError(
                "every review decision must carry manual audit evidence"
            )
        source = _text(
            audit.get("source"),
            field=f"review decisions[{index}].audit.source",
            maximum=100,
        ).casefold()
        if source not in _MANUAL_AUDIT_SOURCES:
            raise RealMultilingualSemanticFreezeError(
                "review decision source is not human, codex-manual, or codex-agent"
            )
        actor = _text(
            audit.get("actor"),
            field=f"review decisions[{index}].audit.actor",
            maximum=200,
        )
        decision_id = _text(
            decision.get("decisionId"),
            field=f"review decisions[{index}].decisionId",
            maximum=200,
        )
        if decision_id in decision_ids:
            raise RealMultilingualSemanticFreezeError(
                "review decision IDs must be unique"
            )
        decision_ids.add(decision_id)
        sources.add(source)
        actors.add(actor)
    if len(sources) != 1:
        raise RealMultilingualSemanticFreezeError(
            "review decisions have mixed adjudication sources"
        )
    return next(iter(sources)), tuple(sorted(actors)), len(decisions)


def _reviewed_document_segments(
    queue: Mapping[str, Any],
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_segments = document.get("segments")
    raw_items = queue.get("items")
    raw_decisions = queue.get("decisions")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RealMultilingualSemanticFreezeError("reviewed document has no segments")
    if not isinstance(raw_items, list) or not isinstance(raw_decisions, list):
        raise RealMultilingualSemanticFreezeError("review queue is incomplete")
    decisions_by_id = {
        str(decision.get("decisionId")): decision
        for decision in raw_decisions
        if isinstance(decision, Mapping) and decision.get("decisionId")
    }
    segments: dict[str, dict[str, Any]] = {}
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise RealMultilingualSemanticFreezeError(
                f"document segments[{index}] must be an object"
            )
        segment = dict(raw_segment)
        segment_id = _text(
            segment.get("id"), field=f"document segments[{index}].id", maximum=200
        )
        if segment_id in segments:
            raise RealMultilingualSemanticFreezeError(
                "reviewed document segment IDs must be unique"
            )
        segments[segment_id] = segment

    reviewed_segment_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise RealMultilingualSemanticFreezeError(
                f"review items[{index}] must be an object"
            )
        if raw_item.get("scope") != "segment":
            continue
        segment_id = _text(
            raw_item.get("segmentId"),
            field=f"review items[{index}].segmentId",
            maximum=200,
        )
        segment = segments.get(segment_id)
        if segment is None:
            raise RealMultilingualSemanticFreezeError(
                "review item is bound to an unknown document segment"
            )
        reviewed_segment_ids.add(segment_id)
        if raw_item.get("speakerId") != segment.get("speakerId"):
            raise RealMultilingualSemanticFreezeError(
                "review item speaker does not match reviewed document"
            )
        time_range = raw_item.get("timeRange")
        if not isinstance(time_range, Mapping) or (
            time_range.get("startMs") != segment.get("startMs")
            or time_range.get("endMs") != segment.get("endMs")
        ):
            raise RealMultilingualSemanticFreezeError(
                "review item timing does not match reviewed document"
            )
        item_text = raw_item.get("text")
        if not isinstance(item_text, Mapping):
            raise RealMultilingualSemanticFreezeError(
                "segment review item has no text snapshot"
            )
        reviewed_text = item_text.get("normalizedText") or item_text.get("displayText")
        document_text = (
            segment.get("normalizedText")
            or segment.get("displayText")
            or segment.get("rawText")
        )
        if reviewed_text != document_text:
            raise RealMultilingualSemanticFreezeError(
                "review item text does not match reviewed document"
            )
        item_decision = raw_item.get("decision")
        if not isinstance(item_decision, Mapping):
            raise RealMultilingualSemanticFreezeError(
                "accepted review item has no decision"
            )
        decision_id = _text(
            item_decision.get("decisionId"),
            field=f"review items[{index}].decision.decisionId",
            maximum=200,
        )
        if decision_id not in decisions_by_id:
            raise RealMultilingualSemanticFreezeError(
                "accepted item decision is absent from the review decision log"
            )
    if reviewed_segment_ids != set(segments):
        raise RealMultilingualSemanticFreezeError(
            "manual baseline requires every final segment to have an accepted review item"
        )
    return [segments[str(raw_segment["id"])] for raw_segment in raw_segments]


def _validate_pre_review_document(
    document: Mapping[str, Any],
    reviewed_document: Mapping[str, Any],
) -> None:
    """Reject benchmark inputs that carry review authority or changed identity."""

    for field in ("schemaVersion", "documentId", "jobId", "language"):
        if document.get(field) != reviewed_document.get(field):
            raise RealMultilingualSemanticFreezeError(
                f"pre-review {field} does not match the reviewed document"
            )
    source = document.get("source")
    reviewed_source = reviewed_document.get("source")
    if not isinstance(source, Mapping) or not isinstance(reviewed_source, Mapping):
        raise RealMultilingualSemanticFreezeError("transcript source binding is missing")
    if dict(source) != dict(reviewed_source):
        raise RealMultilingualSemanticFreezeError(
            "pre-review source identity does not match the reviewed document"
        )

    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise RealMultilingualSemanticFreezeError(
            "pre-review benchmark document has no segments"
        )
    manual_sources = {"manual", "human", "codex", "codex-manual", "codex-agent"}
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise RealMultilingualSemanticFreezeError(
                f"pre-review segments[{index}] must be an object"
            )
        if raw_segment.get("humanLocked") is not False:
            raise RealMultilingualSemanticFreezeError(
                "pre-review benchmark segments must explicitly be unlocked"
            )
        revisions = raw_segment.get("revisions")
        if not isinstance(revisions, list):
            raise RealMultilingualSemanticFreezeError(
                f"pre-review segments[{index}].revisions must be an array"
            )
        for revision_index, revision in enumerate(revisions):
            if not isinstance(revision, Mapping):
                raise RealMultilingualSemanticFreezeError(
                    f"pre-review segments[{index}].revisions[{revision_index}] must be an object"
                )
            source_value = revision.get("source")
            if (
                isinstance(source_value, str)
                and source_value.strip().casefold() in manual_sources
            ):
                raise RealMultilingualSemanticFreezeError(
                    "pre-review benchmark document contains a manual or Codex revision"
                )


def _manual_baseline(
    *,
    document: Mapping[str, Any],
    benchmark_document_sha256: str,
    benchmark_lattice_sha256: str,
    reviewed_document_sha256: str | None,
    reviewed_lattice_sha256: str | None,
    review_queue_sha256: str | None,
    review_queue: Mapping[str, Any],
    audit_source: str,
    actors: Sequence[str],
    decision_count: int,
    generated_at: str,
    source_media_sha256: str,
    additional_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reviewed = _reviewed_document_segments(review_queue, document)
    segments = []
    turns = []
    for segment in reviewed:
        final_text = (
            segment.get("normalizedText")
            or segment.get("displayText")
            or segment.get("rawText")
        )
        final_text = _text(final_text, field="manual baseline finalText", maximum=100_000)
        start_ms = segment.get("startMs")
        end_ms = segment.get("endMs")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
        ):
            raise RealMultilingualSemanticFreezeError(
                "manual baseline segment timing is invalid"
            )
        speaker_id = _text(
            segment.get("speakerId"), field="manual baseline speakerId", maximum=200
        )
        language = _text(
            segment.get("language") or document.get("language"),
            field="manual baseline language",
            maximum=100,
        )
        overlapping = bool(segment.get("overlapping", False))
        segments.append(
            {
                "id": str(segment["id"]),
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerId": speaker_id,
                "language": language,
                "finalText": final_text,
                "overlapping": overlapping,
            }
        )
        turns.append(
            {
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerId": speaker_id,
                "overlap": overlapping,
            }
        )
    return {
        "schemaVersion": "1.0.0",
        "artifactType": "final-adjudicated-transcript",
        "artifactId": f"manual-baseline-{benchmark_document_sha256[:24]}",
        "status": "final",
        "disposition": "transcribable-speech",
        "generatedAt": generated_at,
        "adjudicationSource": audit_source,
        "input": {
            "sourceMediaSha256": source_media_sha256,
            "transcriptDocumentSha256": benchmark_document_sha256,
            "candidateLatticeSha256": benchmark_lattice_sha256,
            **(
                {"reviewQueueSha256": review_queue_sha256}
                if review_queue_sha256 is not None
                else {}
            ),
        },
        "review": {
            "openCount": 0,
            "decisionCount": decision_count,
            "actorIds": list(actors),
            "source": audit_source,
        },
        "timeline": {"turns": turns},
        "segments": segments,
        "provenance": {
            "derivation": "reviewed-transcript-document-plus-accepted-review-decisions-v1",
            "modelGeneratedBaseline": False,
            **(
                {"reviewedTranscriptDocumentSha256": reviewed_document_sha256}
                if reviewed_document_sha256 is not None
                else {}
            ),
            **(
                {"reviewedCandidateLatticeSha256": reviewed_lattice_sha256}
                if reviewed_lattice_sha256 is not None
                else {}
            ),
            **dict(additional_provenance or {}),
        },
    }


def _final_artifact_matches_manual(
    final_artifact: Mapping[str, Any],
    manual_baseline: Mapping[str, Any],
) -> bool:
    def view(value: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        rows = value.get("segments")
        if not isinstance(rows, list):
            return []
        return [
            (
                row.get("id"),
                row.get("startMs"),
                row.get("endMs"),
                row.get("speakerId"),
                row.get("language"),
                row.get("finalText") or row.get("normalizedText"),
                bool(row.get("overlapping", False)),
            )
            for row in rows
            if isinstance(row, Mapping)
        ]

    return view(final_artifact) == view(manual_baseline)


def _select_candidate_lattice(
    output_directory: Path,
    *,
    document_sha256: str,
    source_media_sha256: str,
) -> CandidateLatticeSelection:
    initial_candidates: list[tuple[Path, dict[str, Any], str]] = []
    pattern = "composition-runs/*/semantic-candidate-lattice.initial.v1.json"
    for path in sorted((output_directory / "semantic").glob(pattern)):
        try:
            resolved, value = _load_object(path, field="candidate lattice")
            binding = value.get("binding")
            if not isinstance(binding, Mapping):
                continue
            if _sha(binding.get("transcriptSha256"), field="lattice transcript binding") != document_sha256:
                continue
            if _sha(binding.get("sourceMediaSha256"), field="lattice media binding") != source_media_sha256:
                raise RealMultilingualSemanticFreezeError(
                    "pre-review lattice is rebound to another media file"
                )
            validate_semantic_candidate_lattice(
                value,
                expected_source_media_sha256=source_media_sha256,
                expected_transcript_sha256=document_sha256,
            )
            lattice_sha256 = _sha(
                value.get("latticeSha256"), field="lattice.latticeSha256"
            )
        except RealMultilingualSemanticFreezeError:
            raise
        except Exception:
            continue
        initial_candidates.append((resolved, value, lattice_sha256))
    if len(initial_candidates) != 1:
        raise RealMultilingualSemanticFreezeError(
            "pre-review benchmark document must resolve to exactly one matching initial lattice"
        )
    initial_path, _initial_lattice, initial_lattice_sha256 = initial_candidates[0]
    run_directory = initial_path.parent
    round_directories: dict[int, Path] = {}
    for path in sorted(
        candidate
        for candidate in run_directory.iterdir()
        if candidate.is_dir() and candidate.name.startswith("round-")
    ):
        match = re.fullmatch(r"round-([0-9]{2})", path.name)
        if match is None or int(match.group(1)) < 1:
            raise RealMultilingualSemanticFreezeError(
                "semantic expansion round directory name is invalid"
            )
        round_number = int(match.group(1))
        if round_number in round_directories:
            raise RealMultilingualSemanticFreezeError(
                "semantic expansion round paths are not unique"
            )
        round_directories[round_number] = path

    expanded: dict[int, tuple[Path, str]] = {}
    for round_number, round_directory in sorted(round_directories.items()):
        lattice_path = round_directory / "semantic-candidate-lattice.output.v1.json"
        if not lattice_path.exists() and not lattice_path.is_symlink():
            continue
        resolved, lattice = _load_object(
            lattice_path, field="expanded candidate lattice"
        )
        try:
            validate_semantic_candidate_lattice(
                lattice,
                expected_source_media_sha256=source_media_sha256,
                expected_transcript_sha256=document_sha256,
            )
            lattice_sha256 = _sha(
                lattice.get("latticeSha256"), field="expanded lattice.latticeSha256"
            )
        except Exception as exc:
            raise RealMultilingualSemanticFreezeError(
                "expanded candidate lattice is invalid or rebound"
            ) from exc
        expanded[round_number] = (resolved, lattice_sha256)

    if expanded:
        highest_round = max(expanded)
        if sorted(expanded) != list(range(1, highest_round + 1)):
            raise RealMultilingualSemanticFreezeError(
                "expanded candidate lattice rounds must be continuous from round 1"
            )
        selected_path, selected_sha256 = expanded[highest_round]
        return CandidateLatticeSelection(
            path=selected_path,
            lattice_sha256=selected_sha256,
            selected_kind="expanded-output",
            selected_round=highest_round,
            expanded_round_count=len(expanded),
            initial_path=initial_path,
            initial_lattice_sha256=initial_lattice_sha256,
        )
    return CandidateLatticeSelection(
        path=initial_path,
        lattice_sha256=initial_lattice_sha256,
        selected_kind="initial",
        selected_round=None,
        expanded_round_count=0,
        initial_path=initial_path,
        initial_lattice_sha256=initial_lattice_sha256,
    )


def _speaker_sort_key(speaker_id: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"speaker-([1-9][0-9]*)", speaker_id)
    if match is not None:
        return 0, int(match.group(1))
    return 1, speaker_id


def _baseline_projection(
    baseline: Mapping[str, Any],
    *,
    domain: str,
    scope_id: str,
) -> dict[str, Any]:
    raw_segments = baseline.get("segments")
    if not isinstance(raw_segments, list):
        raise RealMultilingualSemanticFreezeError(
            "manual baseline segments are missing"
        )
    segments = {
        str(segment.get("id")): segment
        for segment in raw_segments
        if isinstance(segment, Mapping) and segment.get("id") is not None
    }
    if len(segments) != len(raw_segments):
        raise RealMultilingualSemanticFreezeError(
            "manual baseline segment identities are invalid"
        )
    if domain == "speech-disposition":
        return {"classification": baseline.get("disposition")}
    if domain == "speaker-cardinality-timeline":
        timeline = baseline.get("timeline")
        raw_turns = timeline.get("turns") if isinstance(timeline, Mapping) else None
        if not isinstance(raw_turns, list) or not raw_turns:
            raise RealMultilingualSemanticFreezeError(
                "manual baseline timeline is missing"
            )
        turns = [
            {
                "startMs": turn.get("startMs"),
                "endMs": turn.get("endMs"),
                "speakerId": turn.get("speakerId"),
                "overlap": bool(turn.get("overlap", turn.get("overlapping", False))),
            }
            for turn in raw_turns
            if isinstance(turn, Mapping)
        ]
        if len(turns) != len(raw_turns):
            raise RealMultilingualSemanticFreezeError(
                "manual baseline timeline turns are invalid"
            )
        turns.sort(
            key=lambda turn: (
                int(turn["startMs"]),
                int(turn["endMs"]),
                str(turn["speakerId"]),
            )
        )
        speaker_ids = sorted(
            {str(turn["speakerId"]) for turn in turns}, key=_speaker_sort_key
        )
        return {
            "speakerCount": len(speaker_ids),
            "speakerIds": speaker_ids,
            "turns": turns,
        }
    if not scope_id.startswith("segment:"):
        raise RealMultilingualSemanticFreezeError(
            f"{domain} calibration group has an invalid segment scope"
        )
    segment_id = scope_id.removeprefix("segment:")
    segment = segments.get(segment_id)
    if segment is None:
        raise RealMultilingualSemanticFreezeError(
            "manual baseline cannot be projected onto every lattice segment"
        )
    if domain == "speaker-assignment":
        return {"segmentId": segment_id, "speakerId": segment.get("speakerId")}
    if domain == "language-span":
        return {"segmentId": segment_id, "language": segment.get("language")}
    if domain == "asr-text":
        return {"segmentId": segment_id, "text": segment.get("finalText")}
    raise RealMultilingualSemanticFreezeError(
        f"unsupported semantic calibration domain: {domain}"
    )


def _candidate_projection(
    payload: Mapping[str, Any],
    *,
    domain: str,
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
            "speakerIds": payload.get("speakerIds"),
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
    raise RealMultilingualSemanticFreezeError(
        f"unsupported semantic calibration domain: {domain}"
    )


def _semantic_calibration_target(
    lattice: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    raw_domains = lattice.get("domains")
    if not isinstance(raw_domains, list):
        raise RealMultilingualSemanticFreezeError(
            "candidate lattice domains are missing"
        )
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, Mapping):
            raise RealMultilingualSemanticFreezeError(
                "candidate lattice domain is invalid"
            )
        domain = _text(
            raw_domain.get("domain"), field="calibration domain", maximum=100
        )
        raw_groups = raw_domain.get("groups")
        if not isinstance(raw_groups, list):
            raise RealMultilingualSemanticFreezeError(
                "candidate lattice groups are missing"
            )
        for raw_group in raw_groups:
            if not isinstance(raw_group, Mapping):
                raise RealMultilingualSemanticFreezeError(
                    "candidate lattice group is invalid"
                )
            group_id = _text(
                raw_group.get("groupId"), field="calibration groupId", maximum=200
            )
            scope_id = _text(
                raw_group.get("scopeId"), field="calibration scopeId", maximum=200
            )
            current_candidate_id = _text(
                raw_group.get("currentCandidateId"),
                field="calibration currentCandidateId",
                maximum=200,
            )
            projection = _baseline_projection(
                baseline, domain=domain, scope_id=scope_id
            )
            projection_sha256 = canonical_json_sha256(projection)
            raw_candidates = raw_group.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise RealMultilingualSemanticFreezeError(
                    "candidate lattice calibration group has no candidates"
                )
            eligible: list[Mapping[str, Any]] = []
            current: Mapping[str, Any] | None = None
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping):
                    raise RealMultilingualSemanticFreezeError(
                        "candidate lattice candidate is invalid"
                    )
                if raw_candidate.get("candidateId") == current_candidate_id:
                    current = raw_candidate
                if raw_candidate.get("selectionEligible") is True:
                    eligible.append(raw_candidate)
            if current is None:
                raise RealMultilingualSemanticFreezeError(
                    "candidate lattice current candidate is missing"
                )
            matching_ids = sorted(
                str(candidate["candidateId"])
                for candidate in eligible
                if isinstance(candidate.get("payload"), Mapping)
                and canonical_json_sha256(
                    _candidate_projection(candidate["payload"], domain=domain)
                )
                == projection_sha256
            )
            eligible_alternative_count = sum(
                candidate.get("candidateId") != current_candidate_id
                for candidate in eligible
            )
            if matching_ids:
                target_candidate_id = (
                    current_candidate_id
                    if current_candidate_id in matching_ids
                    else matching_ids[0]
                )
                expected_action = (
                    "preservation-only"
                    if target_candidate_id == current_candidate_id
                    and eligible_alternative_count == 0
                    else "select"
                )
            else:
                target_candidate_id = None
                expected_action = "request-default-challenger"
            counts[expected_action] += 1
            rows.append(
                {
                    "domain": domain,
                    "groupId": group_id,
                    "scopeId": scope_id,
                    "expectedAction": expected_action,
                    "targetCandidateId": target_candidate_id,
                    "acceptableCandidateIds": matching_ids,
                    "currentCandidateId": current_candidate_id,
                    "baselineProjectionSha256": projection_sha256,
                    "eligibleAlternativeCount": eligible_alternative_count,
                }
            )
    body = {
        "schemaVersion": "1.0.0",
        "derivation": "manual-baseline-vs-pre-review-lattice-v1",
        "groups": rows,
        "counts": {
            "select": counts["select"],
            "request-default-challenger": counts["request-default-challenger"],
            "preservation-only": counts["preservation-only"],
        },
    }
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def _production_artifact_id_matches(case_id: str, artifact_id: str) -> bool:
    pattern = re.compile(
        rf"{re.escape(case_id)}(?:-(?:manual|hybrid)(?:-run[1-9][0-9]*)?)?"
    )
    return pattern.fullmatch(artifact_id) is not None


def _production_evidence(
    output_directory: Path,
    *,
    case_row: Mapping[str, Any],
    project_root: Path,
    reference_truths: Sequence[str] = (),
) -> ProductionEvidence:
    case_id = _text(case_row.get("id"), field="production case id", maximum=200)
    if not _production_artifact_id_matches(case_id, output_directory.name):
        raise RealMultilingualSemanticFreezeError("production output case ID is rebound")
    language = _text(case_row.get("language"), field=f"{case_id}.language", maximum=100)
    source_evaluation_split = _text(
        case_row.get("evaluationSplit"),
        field=f"{case_id}.evaluationSplit",
        maximum=100,
    )
    run_id = output_directory.parents[1].name

    benchmark_document_path, benchmark_document = _load_object(
        output_directory / "semantic" / "input-transcript.v2.json",
        field=f"{case_id} pre-review benchmark transcript",
    )
    reviewed_document_path, reviewed_document = _load_object(
        output_directory / "transcript-document.v2.json",
        field=f"{case_id} reviewed transcript",
    )
    final_artifact_path, final_artifact = _load_object(
        output_directory / "final-adjudicated-transcript.v1.json",
        field=f"{case_id} legacy final artifact",
    )
    review_path, review_queue = _load_object(
        output_directory / "review" / "review-queue.json",
        field=f"{case_id} review queue",
    )

    _validate_pre_review_document(benchmark_document, reviewed_document)
    reviewed_job_id = _text(
        reviewed_document.get("jobId"), field="reviewed document jobId", maximum=200
    )
    reviewed_document_id = _text(
        reviewed_document.get("documentId"),
        field="reviewed document documentId",
        maximum=200,
    )
    if review_queue.get("jobId") != reviewed_job_id:
        raise RealMultilingualSemanticFreezeError(
            "review queue is bound to another job"
        )

    if final_artifact.get("artifactType") != "final-adjudicated-transcript":
        raise RealMultilingualSemanticFreezeError("final artifact type is invalid")
    if final_artifact.get("status") not in {"adjudication-complete", "final"}:
        raise RealMultilingualSemanticFreezeError("final artifact is not complete")
    if final_artifact.get("disposition") != "transcribable-speech":
        raise RealMultilingualSemanticFreezeError("final artifact is not transcribable speech")
    final_review = final_artifact.get("review")
    if not isinstance(final_review, Mapping) or final_review.get("openCount") != 0:
        raise RealMultilingualSemanticFreezeError("final artifact review is not closed")
    segments = final_artifact.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RealMultilingualSemanticFreezeError("final artifact has no segments")
    if final_artifact.get("jobId") != reviewed_job_id:
        raise RealMultilingualSemanticFreezeError(
            "final artifact is bound to another job"
        )
    if final_artifact.get("documentId") != reviewed_document_id:
        raise RealMultilingualSemanticFreezeError(
            "final artifact is bound to another document identity"
        )

    binding = final_artifact.get("input")
    if not isinstance(binding, Mapping):
        raise RealMultilingualSemanticFreezeError("final artifact input binding is missing")
    source_media_sha256 = _sha(
        binding.get("sourceMediaSha256"), field="final artifact source-media binding"
    )
    benchmark_document_sha256 = canonical_json_sha256(benchmark_document)
    reviewed_document_sha256 = canonical_json_sha256(reviewed_document)
    if _sha(
        binding.get("transcriptDocumentSha256"),
        field="final artifact transcript binding",
    ) != reviewed_document_sha256:
        raise RealMultilingualSemanticFreezeError(
            "final artifact is bound to another reviewed document"
        )
    reviewed_lattice_sha256 = _sha(
        binding.get("candidateLatticeSha256"), field="final artifact lattice binding"
    )
    review_queue_sha256 = canonical_json_sha256(review_queue)
    if _sha(
        binding.get("reviewQueueSha256"), field="final artifact review-queue binding"
    ) != review_queue_sha256:
        raise RealMultilingualSemanticFreezeError(
            "final artifact is bound to another review queue"
        )

    reviewed_source = reviewed_document.get("source")
    benchmark_source = benchmark_document.get("source")
    if not isinstance(reviewed_source, Mapping) or not isinstance(benchmark_source, Mapping):
        raise RealMultilingualSemanticFreezeError("transcript source binding is missing")
    if _sha(reviewed_source.get("sha256"), field="reviewed document media binding") != source_media_sha256:
        raise RealMultilingualSemanticFreezeError(
            "reviewed document is bound to another media file"
        )
    if _sha(benchmark_source.get("sha256"), field="benchmark transcript media binding") != source_media_sha256:
        raise RealMultilingualSemanticFreezeError(
            "pre-review benchmark transcript is bound to another media file"
        )

    audit_source, actors, decision_count = _review_authority(review_queue)
    if final_review.get("decisionCount") != decision_count:
        raise RealMultilingualSemanticFreezeError(
            "baseline review decision count does not match review queue"
        )
    lattice_selection = _select_candidate_lattice(
        output_directory,
        document_sha256=benchmark_document_sha256,
        source_media_sha256=source_media_sha256,
    )
    benchmark_lattice_path = lattice_selection.path
    benchmark_lattice_sha256 = lattice_selection.lattice_sha256
    _, benchmark_lattice = _load_object(
        benchmark_lattice_path, field=f"{case_id} benchmark candidate lattice"
    )
    _assert_truth_redacted(
        benchmark_document,
        reference_truths=reference_truths,
        field=f"{case_id} pre-review benchmark document",
    )
    _assert_truth_redacted(
        benchmark_lattice,
        reference_truths=reference_truths,
        field=f"{case_id} pre-review candidate lattice",
    )
    audio_path = _find_audio(project_root, case_id)
    if sha256_file(audio_path) != source_media_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} production media does not match local sample audio"
        )
    generated_at = _text(
        final_artifact.get("generatedAt"),
        field=f"{case_id}.generatedAt",
        maximum=100,
    )
    manual_baseline = _manual_baseline(
        document=reviewed_document,
        benchmark_document_sha256=benchmark_document_sha256,
        benchmark_lattice_sha256=benchmark_lattice_sha256,
        reviewed_document_sha256=reviewed_document_sha256,
        reviewed_lattice_sha256=reviewed_lattice_sha256,
        review_queue_sha256=review_queue_sha256,
        review_queue=review_queue,
        audit_source=audit_source,
        actors=actors,
        decision_count=decision_count,
        generated_at=generated_at,
        source_media_sha256=source_media_sha256,
    )
    return ProductionEvidence(
        case_id=case_id,
        language=language,
        source_evaluation_split=source_evaluation_split,
        run_id=run_id,
        final_generated_at=generated_at,
        output_directory=output_directory.resolve(),
        audio_path=audio_path.resolve(),
        benchmark_document_path=benchmark_document_path,
        benchmark_lattice_path=benchmark_lattice_path,
        reviewed_document_path=reviewed_document_path,
        final_artifact_path=final_artifact_path,
        review_queue_path=review_path,
        benchmark_document_sha256=benchmark_document_sha256,
        benchmark_lattice_sha256=benchmark_lattice_sha256,
        reviewed_document_sha256=reviewed_document_sha256,
        reviewed_lattice_sha256=reviewed_lattice_sha256,
        lattice_selection=lattice_selection,
        manual_baseline=manual_baseline,
        manual_baseline_canonical_sha256=canonical_json_sha256(manual_baseline),
        final_artifact_file_sha256=sha256_file(final_artifact_path),
        final_artifact_canonical_sha256=canonical_json_sha256(final_artifact),
        final_artifact_matches_manual_baseline=_final_artifact_matches_manual(
            final_artifact, manual_baseline
        ),
        review_queue_sha256=review_queue_sha256,
        source_media_sha256=source_media_sha256,
        audit_source=audit_source,
        review_actor_ids=actors,
        review_decision_count=decision_count,
        evidence_kind="completed-production-artifact",
        review_replay=None,
    )


def _assert_review_document_core_identity(
    benchmark_document: Mapping[str, Any],
    reviewed_document: Mapping[str, Any],
) -> None:
    before_segments = benchmark_document.get("segments")
    reviewed_segments = reviewed_document.get("segments")
    if not isinstance(before_segments, list) or not isinstance(reviewed_segments, list):
        raise RealMultilingualSemanticFreezeError(
            "review replay documents must contain segment arrays"
        )
    before_core = [
        (
            segment.get("id"),
            segment.get("startMs"),
            segment.get("endMs"),
            segment.get("speakerId"),
            segment.get("rawText"),
        )
        for segment in before_segments
        if isinstance(segment, Mapping)
    ]
    reviewed_core = [
        (
            segment.get("id"),
            segment.get("startMs"),
            segment.get("endMs"),
            segment.get("speakerId"),
            segment.get("rawText"),
        )
        for segment in reviewed_segments
        if isinstance(segment, Mapping)
    ]
    if before_core != reviewed_core:
        raise RealMultilingualSemanticFreezeError(
            "review replay input and review document segment identity differs"
        )


def _replay_production_evidence(
    output_directory: Path,
    *,
    source_run_root: Path,
    case_row: Mapping[str, Any],
    project_root: Path,
    decision_set_directory: Path,
    decision_set_manifest_path: Path,
    decision_set_manifest_file_sha256: str,
    decision_set_manifest_canonical_sha256: str,
    decision_case: DecisionSetCaseEvidence,
    high_margin_threshold: float,
    reference_truths: Sequence[str] = (),
) -> ProductionEvidence:
    case_id = _text(case_row.get("id"), field="replay production case id", maximum=200)
    if case_id != decision_case.case_id:
        raise RealMultilingualSemanticFreezeError(
            "replay case ID is not bound to the decision-set row"
        )
    if output_directory.name != decision_case.source_output_name:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay output directory is not bound to decision-set evidence"
        )
    if source_run_root.name != decision_case.source_run_root_name:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay run root is not bound to decision-set evidence"
        )
    benchmark_document_path, benchmark_document = _load_object(
        output_directory / "semantic" / "input-transcript.v2.json",
        field=f"{case_id} replay pre-review transcript",
    )
    reviewed_document_path, reviewed_document = _load_object(
        output_directory / "transcript-document.v2.json",
        field=f"{case_id} replay review transcript",
    )
    review_path, source_queue = _load_object(
        output_directory / "review" / "review-queue.json",
        field=f"{case_id} replay review queue",
    )
    _validate_pre_review_document(benchmark_document, reviewed_document)
    _assert_review_document_core_identity(benchmark_document, reviewed_document)
    if sha256_file(benchmark_document_path) != decision_case.source_input_file_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay input file SHA does not match decision-set evidence"
        )
    if canonical_json_sha256(benchmark_document) != decision_case.source_input_canonical_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay input canonical SHA does not match decision-set evidence"
        )
    if sha256_file(review_path) != decision_case.source_queue_file_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay queue file SHA does not match decision-set evidence"
        )
    if canonical_json_sha256(source_queue) != decision_case.source_queue_canonical_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay queue canonical SHA does not match decision-set evidence"
        )
    source_media_sha256 = _sha(
        benchmark_document.get("source", {}).get("sha256"),
        field=f"{case_id} replay media binding",
    )
    if source_media_sha256 != decision_case.source_media_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay media does not match decision-set evidence"
        )
    job_id = _text(
        benchmark_document.get("jobId"), field=f"{case_id} replay jobId", maximum=200
    )
    try:
        current_document, current_queue = validate_review_state(
            benchmark_document,
            source_queue,
            expected_job_id=job_id,
            high_margin_threshold=high_margin_threshold,
        )
    except Exception as exc:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} source review state is invalid"
        ) from exc
    if source_queue.get("openCount") != len(decision_case.observed_open_item_ids):
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay queue openCount does not match decision-set evidence"
        )
    actual_open_ids = tuple(
        str(item.get("id"))
        for item in current_queue.get("items", [])
        if isinstance(item, Mapping) and item.get("status") == "open"
    )
    if set(actual_open_ids) != set(decision_case.observed_open_item_ids):
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay queue open item set does not match decision-set evidence"
        )

    try:
        plan = load_review_decision_plan(
            decision_case.decision_path,
            expected_job_id=job_id,
        )
    except Exception as exc:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} decision plan is invalid"
        ) from exc
    if tuple(plan.item_ids) != decision_case.observed_open_item_ids:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} decision plan item IDs do not match observed queue"
        )
    if tuple(plan.all_decision_ids) != decision_case.replay_decision_ids:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} decision plan IDs do not match decision-set evidence"
        )
    if len(plan.all_decision_ids) != decision_case.replay_decision_count:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} decision plan count does not match decision-set evidence"
        )
    if open_count(current_queue) != len(plan.decisions):
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay queue count does not match decision plan"
        )
    persisted = current_queue.get("decisions")
    if not isinstance(persisted, list) or persisted:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay queue contains pre-existing decisions"
        )
    for raw_command in plan.pre_review_commands:
        command = dict(raw_command)
        command_type = command.pop("type", None)
        if command_type != "speaker.merge":
            raise RealMultilingualSemanticFreezeError(
                f"{case_id} replay contains an unsupported pre-review command"
            )
        before = current_document
        current_document, current_queue, persisted_decision = merge_speakers(
            current_document, current_queue, command
        )
        try:
            assert_raw_text_unchanged(before, current_document)
            current_document, current_queue = validate_review_state(
                current_document,
                current_queue,
                expected_job_id=job_id,
                high_margin_threshold=high_margin_threshold,
            )
        except Exception as exc:
            raise RealMultilingualSemanticFreezeError(
                f"{case_id} pre-review command produced invalid review state"
            ) from exc
        if persisted_decision.get("decisionId") != command.get("decisionId"):
            raise RealMultilingualSemanticFreezeError(
                f"{case_id} pre-review command decision ID was not persisted"
            )
    if set(
        str(item.get("id"))
        for item in current_queue.get("items", [])
        if isinstance(item, Mapping) and item.get("status") == "open"
    ) != set(plan.item_ids):
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} pre-review replay changed the open item set"
        )
    for raw_decision in plan.decisions:
        decision = dict(raw_decision)
        action = decision.pop("action")
        resolved_action = "accepted" if action == "accept" else "rejected"
        before = current_document
        current_document, current_queue, persisted_decision = resolve_review_item(
            current_document,
            current_queue,
            decision,
            command="review.submit",
            action=resolved_action,
        )
        try:
            assert_raw_text_unchanged(before, current_document)
            current_document, current_queue = validate_review_state(
                current_document,
                current_queue,
                expected_job_id=job_id,
                high_margin_threshold=high_margin_threshold,
            )
        except Exception as exc:
            raise RealMultilingualSemanticFreezeError(
                f"{case_id} review decision produced invalid review state"
            ) from exc
        if persisted_decision.get("decisionId") != decision.get("decisionId"):
            raise RealMultilingualSemanticFreezeError(
                f"{case_id} review decision ID was not persisted"
            )
    if open_count(current_queue) != 0:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay did not close the review queue"
        )
    final_decision_ids = tuple(
        str(item.get("decisionId"))
        for item in current_queue.get("decisions", [])
        if isinstance(item, Mapping) and item.get("decisionId") is not None
    )
    if set(final_decision_ids) != set(plan.all_decision_ids):
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay decision IDs do not exactly match the decision plan"
        )
    audit_source, actors, decision_count = _review_authority(current_queue)
    if decision_count != len(plan.all_decision_ids):
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay decision count is incomplete"
        )
    lattice_selection = _select_candidate_lattice(
        output_directory,
        document_sha256=canonical_json_sha256(benchmark_document),
        source_media_sha256=_sha(
            benchmark_document.get("source", {}).get("sha256"),
            field=f"{case_id} replay media binding",
        ),
    )
    benchmark_lattice_path = lattice_selection.path
    benchmark_lattice_sha256 = lattice_selection.lattice_sha256
    _, benchmark_lattice = _load_object(
        benchmark_lattice_path, field=f"{case_id} replay candidate lattice"
    )
    _assert_truth_redacted(
        benchmark_document,
        reference_truths=reference_truths,
        field=f"{case_id} replay pre-review transcript",
    )
    _assert_truth_redacted(
        benchmark_lattice,
        reference_truths=reference_truths,
        field=f"{case_id} replay candidate lattice",
    )
    audio_path = _find_audio(project_root, case_id)
    if sha256_file(audio_path) != source_media_sha256:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} replay media does not match local sample audio"
        )
    replay_provenance = ReviewReplayProvenance(
        decision_set_directory=decision_set_directory,
        decision_set_manifest_path=decision_set_manifest_path,
        decision_set_manifest_file_sha256=decision_set_manifest_file_sha256,
        decision_set_manifest_canonical_sha256=decision_set_manifest_canonical_sha256,
        decision_path=decision_case.decision_path,
        decision_file_sha256=decision_case.decision_file_sha256,
        decision_canonical_sha256=decision_case.decision_canonical_sha256,
        source_open_queue_file_sha256=decision_case.source_queue_file_sha256,
        source_open_queue_canonical_sha256=decision_case.source_queue_canonical_sha256,
        source_run_root=source_run_root.resolve(),
        source_output_directory=output_directory.resolve(),
    )
    manual_baseline = _manual_baseline(
        document=current_document,
        benchmark_document_sha256=canonical_json_sha256(benchmark_document),
        benchmark_lattice_sha256=benchmark_lattice_sha256,
        reviewed_document_sha256=None,
        reviewed_lattice_sha256=None,
        review_queue_sha256=None,
        review_queue=current_queue,
        audit_source=audit_source,
        actors=actors,
        decision_count=decision_count,
        generated_at=_text(
            benchmark_document.get("generatedAt"),
            field=f"{case_id} replay generatedAt",
            maximum=100,
        ),
        source_media_sha256=source_media_sha256,
        additional_provenance={
            "derivation": "pre-review-transcript-plus-in-memory-production-review-replay-v1",
            "decisionSetManifestFileSha256": decision_set_manifest_file_sha256,
            "decisionSetManifestCanonicalSha256": decision_set_manifest_canonical_sha256,
            "decisionFileSha256": decision_case.decision_file_sha256,
            "decisionCanonicalSha256": decision_case.decision_canonical_sha256,
            "sourceOpenReviewQueueFileSha256": decision_case.source_queue_file_sha256,
            "sourceOpenReviewQueueCanonicalSha256": decision_case.source_queue_canonical_sha256,
            "replayOpenCount": 0,
            "replayedDecisionIds": list(plan.all_decision_ids),
        },
    )
    return ProductionEvidence(
        case_id=case_id,
        language=_text(case_row.get("language"), field=f"{case_id}.language", maximum=100),
        source_evaluation_split=_text(
            case_row.get("evaluationSplit"),
            field=f"{case_id}.evaluationSplit",
            maximum=100,
        ),
        run_id=source_run_root.name,
        final_generated_at=_text(
            benchmark_document.get("generatedAt"),
            field=f"{case_id} replay generatedAt",
            maximum=100,
        ),
        output_directory=output_directory.resolve(),
        audio_path=audio_path.resolve(),
        benchmark_document_path=benchmark_document_path,
        benchmark_lattice_path=benchmark_lattice_path,
        reviewed_document_path=reviewed_document_path,
        final_artifact_path=None,
        review_queue_path=review_path,
        benchmark_document_sha256=canonical_json_sha256(benchmark_document),
        benchmark_lattice_sha256=benchmark_lattice_sha256,
        reviewed_document_sha256=canonical_json_sha256(reviewed_document),
        reviewed_lattice_sha256=None,
        lattice_selection=lattice_selection,
        manual_baseline=manual_baseline,
        manual_baseline_canonical_sha256=canonical_json_sha256(manual_baseline),
        final_artifact_file_sha256=None,
        final_artifact_canonical_sha256=None,
        final_artifact_matches_manual_baseline=None,
        review_queue_sha256=decision_case.source_queue_canonical_sha256,
        source_media_sha256=source_media_sha256,
        audit_source=audit_source,
        review_actor_ids=actors,
        review_decision_count=decision_count,
        evidence_kind="in-memory-review-replay",
        review_replay=replay_provenance,
    )


def _resolve_pre_review_run_roots(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    names: set[str] = set()
    for index, path in enumerate(paths):
        candidate = path.expanduser()
        if candidate.is_symlink():
            raise RealMultilingualSemanticFreezeError(
                f"pre-review run root[{index}] must not be a symlink"
            )
        try:
            root = candidate.resolve(strict=True)
        except OSError as exc:
            raise RealMultilingualSemanticFreezeError(
                f"pre-review run root[{index}] is missing: {path}"
            ) from exc
        if not root.is_dir() or not (root / "outputs").is_dir():
            raise RealMultilingualSemanticFreezeError(
                f"pre-review run root[{index}] must contain an outputs directory"
            )
        if root in resolved:
            raise RealMultilingualSemanticFreezeError(
                "pre-review run roots must be unique"
            )
        if root.name in names:
            raise RealMultilingualSemanticFreezeError(
                "pre-review run root names must be unique for portable evidence binding"
            )
        resolved.append(root)
        names.add(root.name)
    return tuple(resolved)


def _scan_production(
    *,
    eval_root: Path,
    project_root: Path,
    global_cases: Mapping[str, Mapping[str, Any]],
    reference_truth_by_case: Mapping[str, Sequence[str]],
    case_ids: Sequence[str],
    decision_set_directory: Path | None,
    decision_set_manifest_path: Path | None,
    decision_set_manifest_file_sha256: str | None,
    decision_set_manifest_canonical_sha256: str | None,
    decision_cases: Mapping[str, DecisionSetCaseEvidence],
    pre_review_run_roots: Sequence[Path],
    replay_high_margin_threshold: float | None,
) -> tuple[dict[str, ProductionEvidence], dict[str, int]]:
    runs_root = eval_root / "product-runs"
    selected: dict[str, ProductionEvidence] = {}
    rejected: Counter[str] = Counter()
    for case_id in case_ids:
        case_row = global_cases.get(case_id)
        if case_row is None:
            rejected[case_id] += 1
            continue
        decision_case = decision_cases.get(case_id)
        candidates: list[ProductionEvidence] = []
        if decision_case is None and runs_root.is_dir():
            for run in sorted(path for path in runs_root.iterdir() if path.is_dir()):
                outputs_root = run / "outputs"
                if not outputs_root.is_dir():
                    continue
                for output in sorted(
                    path
                    for path in outputs_root.iterdir()
                    if path.is_dir()
                    and _production_artifact_id_matches(case_id, path.name)
                ):
                    if not (
                        output / "final-adjudicated-transcript.v1.json"
                    ).is_file():
                        continue
                    try:
                        candidates.append(
                            _production_evidence(
                                output,
                                case_row=case_row,
                                project_root=project_root,
                                reference_truths=reference_truth_by_case.get(
                                    case_id, ()
                                ),
                            )
                        )
                    except RealMultilingualSemanticFreezeError:
                        rejected[case_id] += 1
        if candidates:
            selected[case_id] = max(
                candidates,
                key=lambda item: (item.final_generated_at, item.run_id),
            )
            continue
        if decision_case is None:
            continue
        if (
            decision_set_directory is None
            or decision_set_manifest_path is None
            or decision_set_manifest_file_sha256 is None
            or decision_set_manifest_canonical_sha256 is None
            or replay_high_margin_threshold is None
        ):
            raise RealMultilingualSemanticFreezeError(
                "decision-set replay configuration is incomplete"
            )
        replay_outputs: list[tuple[Path, Path]] = []
        for run_root in pre_review_run_roots:
            if run_root.name != decision_case.source_run_root_name:
                continue
            output = run_root / "outputs" / decision_case.source_output_name
            if output.is_symlink():
                rejected[case_id] += 1
                continue
            if output.is_dir():
                replay_outputs.append((run_root, output))
        if len(replay_outputs) != 1:
            rejected[case_id] += 1
            continue
        source_run_root, replay_output = replay_outputs[0]
        try:
            selected[case_id] = _replay_production_evidence(
                replay_output,
                source_run_root=source_run_root,
                case_row=case_row,
                project_root=project_root,
                decision_set_directory=decision_set_directory,
                decision_set_manifest_path=decision_set_manifest_path,
                decision_set_manifest_file_sha256=(
                    decision_set_manifest_file_sha256
                ),
                decision_set_manifest_canonical_sha256=(
                    decision_set_manifest_canonical_sha256
                ),
                decision_case=decision_case,
                high_margin_threshold=replay_high_margin_threshold,
                reference_truths=reference_truth_by_case.get(case_id, ()),
            )
        except RealMultilingualSemanticFreezeError:
            rejected[case_id] += 1
    return selected, dict(sorted(rejected.items()))


def _benchmark_case(evidence: ProductionEvidence, output_directory: Path) -> dict[str, Any]:
    _, lattice = _load_object(
        evidence.benchmark_lattice_path,
        field=f"{evidence.case_id} benchmark candidate lattice",
    )
    value = {
        "caseId": evidence.case_id,
        "language": evidence.language,
        "documentPath": _relative(evidence.benchmark_document_path, output_directory),
        "latticePath": _relative(evidence.benchmark_lattice_path, output_directory),
        "baseline": evidence.manual_baseline,
        "baselineAuditSource": evidence.audit_source,
        "documentSha256": evidence.benchmark_document_sha256,
        "latticeSha256": evidence.benchmark_lattice_sha256,
        "baselineSha256": evidence.manual_baseline_canonical_sha256,
        "latticeSelection": {
            "policy": "highest-contiguous-strictly-validated-expanded-output-else-initial-v1",
            "selectedKind": evidence.lattice_selection.selected_kind,
            "selectedRound": evidence.lattice_selection.selected_round,
            "expandedRoundCount": evidence.lattice_selection.expanded_round_count,
            "initialLatticeSha256": (
                evidence.lattice_selection.initial_lattice_sha256
            ),
        },
        "semanticCalibrationTarget": _semantic_calibration_target(
            lattice, evidence.manual_baseline
        ),
    }
    value["caseSha256"] = canonical_json_sha256(value)
    return value


def _production_inventory(
    evidence: ProductionEvidence,
    output_directory: Path,
) -> dict[str, Any]:
    replay = evidence.review_replay
    limitations = [
        "benchmark document and candidate lattice are pre-review, unlocked, and truth-redacted",
    ]
    if replay is None:
        limitations.extend(
            [
                "manual baseline text is derived from the reviewed transcript and closed review queue but is bound to the pre-review inputs",
                "reviewed transcript and legacy final artifact are provenance only and are never model inputs",
            ]
        )
    else:
        limitations.extend(
            [
                "manual baseline text is derived by deterministic in-memory replay of the immutable Codex decision set against the source open review queue",
                "dynamic replay document, queue hashes, timestamps, and generated revision IDs are deliberately excluded from the freeze",
                "the source run has no final-adjudicated artifact; absence is not classified as invalid evidence",
            ]
        )
    limitations.extend(_KNOWN_RUN_LIMITATIONS.get(evidence.run_id, []))
    if evidence.final_artifact_matches_manual_baseline is False:
        limitations.append(
            "legacy final-adjudicated artifact was overwritten by pre-fix composition and is excluded from baseline scoring"
        )

    artifacts: dict[str, Any] = {
        "benchmarkDocument": _json_artifact(
            evidence.benchmark_document_path, output_directory
        ),
        "candidateLattice": _json_artifact(
            evidence.benchmark_lattice_path, output_directory
        ),
        "initialCandidateLattice": _json_artifact(
            evidence.lattice_selection.initial_path, output_directory
        ),
        "manualBaseline": {
            "embeddedInBenchmarkCase": True,
            "canonicalSha256": evidence.manual_baseline_canonical_sha256,
            "derivation": (
                "reviewed-transcript-document-plus-accepted-review-decisions-v1"
                if replay is None
                else "pre-review-transcript-plus-in-memory-production-review-replay-v1"
            ),
        },
    }
    if replay is None:
        if evidence.final_artifact_path is None:
            raise RealMultilingualSemanticFreezeError(
                "completed production evidence has no final artifact path"
            )
        final_artifact = _json_artifact(
            evidence.final_artifact_path, output_directory
        )
        final_artifact["classification"] = (
            "consistent-supporting-evidence"
            if evidence.final_artifact_matches_manual_baseline
            else "known-invalid-pre-fix-evidence"
        )
        final_artifact["usedAsBenchmarkBaseline"] = False
        artifacts.update(
            {
                "reviewedTranscriptDocument": _json_artifact(
                    evidence.reviewed_document_path, output_directory
                ),
                "legacyFinalArtifact": final_artifact,
                "reviewQueue": _json_artifact(
                    evidence.review_queue_path, output_directory
                ),
            }
        )
        manual_adjudication = {
            "source": evidence.audit_source,
            "actorIds": list(evidence.review_actor_ids),
            "decisionCount": evidence.review_decision_count,
            "reviewQueueCanonicalSha256": evidence.review_queue_sha256,
        }
        selection_policy = "latest-valid-manually-adjudicated-production-run"
    else:
        artifacts.update(
            {
                "sourceTranscriptDocument": _json_artifact(
                    evidence.reviewed_document_path, output_directory
                ),
                "sourceOpenReviewQueue": _json_artifact(
                    evidence.review_queue_path, output_directory
                ),
                "reviewDecisionSetManifest": _json_artifact(
                    replay.decision_set_manifest_path, output_directory
                ),
                "reviewDecisionPlan": _json_artifact(
                    replay.decision_path, output_directory
                ),
                "legacyFinalArtifact": {
                    "status": "absent",
                    "classification": "not-applicable-in-memory-review-replay",
                    "usedAsBenchmarkBaseline": False,
                },
            }
        )
        manual_adjudication = {
            "source": evidence.audit_source,
            "actorIds": list(evidence.review_actor_ids),
            "decisionCount": evidence.review_decision_count,
            "derivation": "immutable-decision-set-in-memory-review-replay-v1",
            "sourceOpenReviewQueueFileSha256": (
                replay.source_open_queue_file_sha256
            ),
            "sourceOpenReviewQueueCanonicalSha256": (
                replay.source_open_queue_canonical_sha256
            ),
            "decisionSetManifestFileSha256": (
                replay.decision_set_manifest_file_sha256
            ),
            "decisionSetManifestCanonicalSha256": (
                replay.decision_set_manifest_canonical_sha256
            ),
            "decisionFileSha256": replay.decision_file_sha256,
            "decisionCanonicalSha256": replay.decision_canonical_sha256,
            "replayOpenCount": 0,
        }
        selection_policy = "strict-hash-bound-decision-set-review-replay"

    selection = {
        "runId": evidence.run_id,
        "policy": selection_policy,
        "evidenceKind": evidence.evidence_kind,
        "candidateLattice": {
            "policy": "highest-contiguous-strictly-validated-expanded-output-else-initial-v1",
            "selectedKind": evidence.lattice_selection.selected_kind,
            "selectedRound": evidence.lattice_selection.selected_round,
            "expandedRoundCount": evidence.lattice_selection.expanded_round_count,
            "transcriptSha256": evidence.benchmark_document_sha256,
        },
    }
    selection[
        "finalGeneratedAt" if replay is None else "sourceGeneratedAt"
    ] = evidence.final_generated_at

    return {
        "caseId": evidence.case_id,
        "language": evidence.language,
        "freezeSplit": "development",
        "sourceEvaluationSplit": evidence.source_evaluation_split,
        "status": PRODUCTION_STATUS,
        "benchmarkEligible": True,
        "selection": selection,
        "audio": _audio_artifact(
            evidence.audio_path,
            output_directory,
            expected_sha256=evidence.source_media_sha256,
        ),
        "artifacts": artifacts,
        "manualAdjudication": manual_adjudication,
        "limitations": limitations,
    }


def _reference_inventory(
    row: Mapping[str, Any],
    *,
    reference_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    case_id = _text(row.get("id"), field="reference case id", maximum=200)
    language = _text(row.get("language"), field=f"{case_id}.language", maximum=100)
    split = _text(
        row.get("evaluationSplit"),
        field=f"{case_id}.evaluationSplit",
        maximum=100,
    )
    if split != "development" or row.get("tuningEligible") is not True:
        raise RealMultilingualSemanticFreezeError(
            f"{case_id} is not development/tuning eligible"
        )
    raw_audio_path = _text(row.get("path"), field=f"{case_id}.path", maximum=1000)
    audio_path = (reference_path.parent / raw_audio_path).resolve()
    audio_sha256 = _sha(row.get("sha256"), field=f"{case_id}.sha256")
    reference_text = row.get("scoringTranscript") or row.get("expectedTranscript")
    native_text = row.get("nativeTranscript") or row.get("rawTranscript")
    reference_text = _text(
        reference_text, field=f"{case_id}.referenceTranscript", maximum=100_000
    )
    native_text = _text(
        native_text, field=f"{case_id}.nativeTranscript", maximum=100_000
    )
    return {
        "caseId": case_id,
        "language": language,
        "freezeSplit": "development",
        "sourceEvaluationSplit": split,
        "status": REFERENCE_ONLY_STATUS,
        "benchmarkEligible": False,
        "blockingReason": "production document, candidate lattice, and manual final baseline are not frozen",
        "audio": _audio_artifact(
            audio_path,
            output_directory,
            expected_sha256=audio_sha256,
        ),
        "reference": {
            "manifestPath": _relative(reference_path, output_directory),
            "manifestCaseId": case_id,
            "manifestCaseCanonicalSha256": canonical_json_sha256(dict(row)),
            "scoringTranscriptCanonicalSha256": canonical_json_sha256(reference_text),
            "nativeTranscriptCanonicalSha256": canonical_json_sha256(native_text),
            "textCopiedIntoFreeze": False,
        },
    }


def _inventory_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    language = str(row.get("language") or "")
    status_order = {
        PRODUCTION_STATUS: 0,
        REFERENCE_ONLY_STATUS: 1,
        UNAVAILABLE_STATUS: 2,
    }
    return (
        _LANGUAGE_ORDER.get(language, len(_LANGUAGE_ORDER)),
        status_order.get(str(row.get("status")), 3),
        str(row.get("caseId") or ""),
    )


def build_real_multilingual_semantic_development_freeze(
    *,
    project_root: Path,
    eval_root: Path,
    global_manifest_path: Path,
    fleurs_reference_path: Path,
    output_path: Path,
    production_case_ids: Sequence[str] = DEFAULT_PRODUCTION_CASE_IDS,
    reference_languages: Sequence[str] = DEFAULT_REFERENCE_LANGUAGES,
    codex_decision_set: Path | None = None,
    pre_review_run_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Build a deterministic benchmark manifest and non-eligible inventory."""

    output_directory = output_path.expanduser().resolve(strict=False).parent
    project_root = project_root.expanduser().resolve(strict=True)
    eval_root = eval_root.expanduser().resolve(strict=True)
    global_path, global_manifest = _load_object(
        global_manifest_path, field="global sample manifest"
    )
    reference_path, fleurs_reference = _load_object(
        fleurs_reference_path, field="FLEURS frozen reference"
    )
    _validate_declared_canonical(fleurs_reference, field="FLEURS frozen reference")
    global_cases = _global_case_index(global_manifest)
    reference_truth_by_case = _reference_truth_index(fleurs_reference)

    if codex_decision_set is None and pre_review_run_roots:
        raise RealMultilingualSemanticFreezeError(
            "pre-review run roots require a Codex decision set"
        )
    if codex_decision_set is not None and not pre_review_run_roots:
        raise RealMultilingualSemanticFreezeError(
            "Codex decision-set replay requires at least one pre-review run root"
        )
    decision_set_directory: Path | None = None
    decision_set_manifest_path: Path | None = None
    decision_set_manifest_file_sha256: str | None = None
    decision_set_manifest_canonical_sha256: str | None = None
    decision_cases: dict[str, DecisionSetCaseEvidence] = {}
    replay_high_margin_threshold: float | None = None
    resolved_pre_review_roots: tuple[Path, ...] = ()
    if codex_decision_set is not None:
        (
            decision_set_manifest_path,
            decision_set_manifest,
            decision_cases,
        ) = _load_decision_set(codex_decision_set)
        decision_set_directory = decision_set_manifest_path.parent
        decision_set_manifest_file_sha256 = sha256_file(
            decision_set_manifest_path
        )
        decision_set_manifest_canonical_sha256 = canonical_json_sha256(
            decision_set_manifest
        )
        replay_high_margin_threshold = _decision_set_replay_threshold(
            decision_set_manifest
        )
        resolved_pre_review_roots = _resolve_pre_review_run_roots(
            pre_review_run_roots
        )

    production_ids = tuple(dict.fromkeys(production_case_ids))
    if not production_ids:
        raise RealMultilingualSemanticFreezeError("production case list is empty")
    reference_language_set = set(reference_languages)
    unknown_languages = (
        set(reference_language_set)
        | {
            str(global_cases[case_id].get("language"))
            for case_id in production_ids
            if case_id in global_cases
        }
    ) - set(TARGET_LANGUAGES)
    if unknown_languages:
        raise RealMultilingualSemanticFreezeError(
            "freeze contains languages outside the declared target set: "
            + ", ".join(sorted(unknown_languages))
        )

    selected, rejected_counts = _scan_production(
        eval_root=eval_root,
        project_root=project_root,
        global_cases=global_cases,
        reference_truth_by_case=reference_truth_by_case,
        case_ids=production_ids,
        decision_set_directory=decision_set_directory,
        decision_set_manifest_path=decision_set_manifest_path,
        decision_set_manifest_file_sha256=decision_set_manifest_file_sha256,
        decision_set_manifest_canonical_sha256=(
            decision_set_manifest_canonical_sha256
        ),
        decision_cases=decision_cases,
        pre_review_run_roots=resolved_pre_review_roots,
        replay_high_margin_threshold=replay_high_margin_threshold,
    )
    benchmark_cases = [
        _benchmark_case(selected[case_id], output_directory)
        for case_id in production_ids
        if case_id in selected
    ]
    inventory: list[dict[str, Any]] = [
        _production_inventory(selected[case_id], output_directory)
        for case_id in production_ids
        if case_id in selected
    ]
    for case_id in production_ids:
        if case_id in selected:
            continue
        row = global_cases.get(case_id, {})
        inventory.append(
            {
                "caseId": case_id,
                "language": row.get("language") or "und",
                "freezeSplit": "development",
                "sourceEvaluationSplit": row.get("evaluationSplit"),
                "status": UNAVAILABLE_STATUS,
                "benchmarkEligible": False,
                "blockingReason": "no complete hash-bound production lattice and manual final baseline",
            }
        )

    reference_rows = fleurs_reference.get("cases")
    if not isinstance(reference_rows, list):
        raise RealMultilingualSemanticFreezeError("FLEURS reference cases are missing")
    for index, row in enumerate(reference_rows):
        if not isinstance(row, Mapping):
            raise RealMultilingualSemanticFreezeError(
                f"FLEURS reference cases[{index}] must be an object"
            )
        if row.get("evaluationSplit") != "development":
            continue
        if row.get("language") not in reference_language_set:
            continue
        reference_case_id = _text(
            row.get("id"),
            field=f"FLEURS reference cases[{index}].id",
            maximum=200,
        )
        if reference_case_id in selected:
            continue
        inventory.append(
            _reference_inventory(
                row,
                reference_path=reference_path,
                output_directory=output_directory,
            )
        )

    inventory.sort(key=_inventory_sort_key)
    benchmark_cases.sort(
        key=lambda row: (
            _LANGUAGE_ORDER.get(str(row["language"]), len(_LANGUAGE_ORDER)),
            str(row["caseId"]),
        )
    )
    status_counts = Counter(str(row["status"]) for row in inventory)
    coverage = []
    for language in TARGET_LANGUAGES:
        rows = [row for row in inventory if row.get("language") == language]
        coverage.append(
            {
                "language": language,
                "productionCaseCount": sum(
                    row.get("status") == PRODUCTION_STATUS for row in rows
                ),
                "referenceOnlyCount": sum(
                    row.get("status") == REFERENCE_ONLY_STATUS for row in rows
                ),
                "unavailableCount": sum(
                    row.get("status") == UNAVAILABLE_STATUS for row in rows
                ),
                "covered": any(
                    row.get("status") in {PRODUCTION_STATUS, REFERENCE_ONLY_STATUS}
                    for row in rows
                ),
            }
        )
    value: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "freezeId": "real-multilingual-semantic-development-v1",
        "selection": {
            "freezeSplit": "development",
            "targetLanguages": list(TARGET_LANGUAGES),
            "productionCaseIds": list(production_ids),
            "referenceLanguages": sorted(
                reference_language_set,
                key=lambda language: _LANGUAGE_ORDER.get(
                    language, len(_LANGUAGE_ORDER)
                ),
            ),
            "productionPolicy": (
                "strict-decision-set-review-replay-for-covered-cases-else-latest-valid-manually-adjudicated-production-run"
                if decision_cases
                else "latest-valid-manually-adjudicated-production-run"
            ),
            "referencePolicy": "all-locally-verified-FLEURS-development-rows",
            "heldOutIncluded": False,
        },
        "inputContract": {
            "benchmarkTool": "tools/benchmark_real_multilingual_semantic_models.py",
            "casesContainOnlyBenchmarkEligibleRows": True,
            "requiredArtifacts": [
                "hash-bound transcript document",
                "hash-bound initial candidate lattice",
                "manual final-adjudicated baseline",
            ],
            "referenceOnlyRowsNeverEnterCases": True,
            "benchmarkDocumentsArePreReview": True,
            "benchmarkSegmentsAreExplicitlyUnlocked": True,
            "manualOrCodexRevisionsInBenchmarkInputs": False,
            "semanticCalibrationTargetVisibleToProvider": False,
            "baselineVisibleToProvider": False,
        },
        "sourceEvidence": {
            "globalManifest": {
                "path": _relative(global_path, output_directory),
                "fileSha256": sha256_file(global_path),
            },
            "fleursReference": {
                "path": _relative(reference_path, output_directory),
                "fileSha256": sha256_file(reference_path),
                "canonicalSha256": canonical_json_sha256(fleurs_reference),
            },
            "productRunsRoot": _relative(
                (eval_root / "product-runs").resolve(), output_directory
            ),
            **(
                {
                    "codexDecisionSet": {
                        "manifest": _json_artifact(
                            decision_set_manifest_path, output_directory
                        ),
                        "caseCount": len(decision_cases),
                        "replayHighMarginThreshold": replay_high_margin_threshold,
                    },
                    "preReviewRunRoots": [
                        _relative(root, output_directory)
                        for root in resolved_pre_review_roots
                    ],
                }
                if decision_set_manifest_path is not None
                else {}
            ),
        },
        "truthPolicy": {
            "referenceTextCopiedIntoFreeze": False,
            "referenceTextHashAndLocatorPersisted": True,
            "referenceOnlyRowsAreNotManualBaselines": True,
            "modelIdentityPresent": False,
            "reviewedTextPresentOnlyInHiddenBaseline": True,
            "reviewedTranscriptUsedAsModelInput": False,
        },
        "counts": {
            "benchmarkCases": len(benchmark_cases),
            "inventoryRows": len(inventory),
            PRODUCTION_STATUS: status_counts[PRODUCTION_STATUS],
            REFERENCE_ONLY_STATUS: status_counts[REFERENCE_ONLY_STATUS],
            UNAVAILABLE_STATUS: status_counts[UNAVAILABLE_STATUS],
            "rejectedProductionCandidates": sum(rejected_counts.values()),
        },
        "languageCoverage": coverage,
        "cases": benchmark_cases,
        "inventory": inventory,
    }
    value["canonicalSha256"] = canonical_json_sha256(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=_default_eval_root())
    parser.add_argument("--global-manifest", type=Path)
    parser.add_argument("--fleurs-reference", type=Path)
    parser.add_argument(
        "--codex-decision-set",
        type=Path,
        help="immutable r2 Codex review decision-set directory",
    )
    parser.add_argument(
        "--pre-review-run-root",
        action="append",
        default=[],
        type=Path,
        help="source product run root eligible for strict decision-set replay",
    )
    parser.add_argument("--production-case", action="append", default=[])
    parser.add_argument("--reference-language", action="append", default=[])
    parser.add_argument(
        "--no-reference-inventory",
        action="store_true",
        help=(
            "explicitly freeze zero reference-only rows; mutually exclusive "
            "with --reference-language"
        ),
    )
    parser.add_argument("--expected-production-count", type=int)
    parser.add_argument("--expected-reference-count", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve(strict=True)
    production_case_ids = tuple(args.production_case or DEFAULT_PRODUCTION_CASE_IDS)
    if args.no_reference_inventory and args.reference_language:
        raise SystemExit(
            "--no-reference-inventory and --reference-language are mutually exclusive"
        )
    reference_languages = (
        ()
        if args.no_reference_inventory
        else tuple(args.reference_language or DEFAULT_REFERENCE_LANGUAGES)
    )
    global_manifest = args.global_manifest or (
        project_root / "sample_library" / "global-manifest.v1.json"
    )
    fleurs_reference = args.fleurs_reference or (
        project_root
        / ".runtime_cache"
        / "sample-library"
        / "global"
        / "fleurs-multilingual-frozen.v1.json"
    )
    value = build_real_multilingual_semantic_development_freeze(
        project_root=project_root,
        eval_root=args.eval_root,
        global_manifest_path=global_manifest,
        fleurs_reference_path=fleurs_reference,
        output_path=args.output,
        production_case_ids=production_case_ids,
        reference_languages=reference_languages,
        codex_decision_set=args.codex_decision_set,
        pre_review_run_roots=tuple(args.pre_review_run_root),
    )
    expected_production = (
        args.expected_production_count
        if args.expected_production_count is not None
        else len(production_case_ids)
    )
    expected_reference = (
        args.expected_reference_count
        if args.expected_reference_count is not None
        else 3 * len(reference_languages)
    )
    counts = value["counts"]
    if counts[PRODUCTION_STATUS] != expected_production:
        raise RealMultilingualSemanticFreezeError(
            "production benchmark case count does not match the requested freeze"
        )
    if counts[REFERENCE_ONLY_STATUS] != expected_reference:
        raise RealMultilingualSemanticFreezeError(
            "reference-only case count does not match the requested freeze"
        )
    atomic_write_json_no_replace(args.output.expanduser().resolve(), value)
    print(
        f"frozen {counts[PRODUCTION_STATUS]} benchmark cases and "
        f"{counts[REFERENCE_ONLY_STATUS]} reference-only rows to "
        f"{args.output.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
