#!/usr/bin/env python3
"""Build a cross-sample audit of final semantic-composition outcomes."""

from __future__ import annotations

import argparse
import copy
import math
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
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
from backend.semantic_candidate_lattice import (
    validate_semantic_candidate_lattice,
)
from backend.semantic_composition import (
    build_semantic_composition,
    validate_semantic_composition,
    validate_semantic_job_arbitration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _text(value: Any, field: str, *, maximum: int = 20_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field} exceeds {maximum} characters")
    return normalized


def _string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{field} must be an array")
    result = [_text(item, f"{field}[]", maximum=255) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _path(
    value: Any,
    field: str,
    *,
    manifest_root: Path,
    required: bool = True,
) -> Path | None:
    if value is None and not required:
        return None
    raw = _text(value, field, maximum=4096)
    candidate = Path(raw).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (manifest_root / candidate).resolve()
    )
    if not resolved.is_file():
        raise ValueError(f"{field} is not a file: {resolved}")
    return resolved


def _load_object(path: Path, field: str) -> dict[str, Any]:
    value = read_json_strict(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must contain a JSON object")
    return dict(value)


def _artifact_evidence(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "fileSha256": sha256_file(path),
        "canonicalSha256": canonical_json_sha256(value),
    }


def _baseline_text(document: Mapping[str, Any]) -> str:
    return " ".join(
        str(
            segment.get("normalizedText")
            or segment.get("rawText")
            or ""
        ).strip()
        for segment in document["segments"]
        if isinstance(segment, Mapping)
    ).strip()


def _final_text(composition: Mapping[str, Any]) -> str:
    return " ".join(
        str(segment.get("finalText") or "").strip()
        for segment in composition["segments"]
        if isinstance(segment, Mapping)
    ).strip()


_CHARACTER_SCORING_LANGUAGES = {
    "ja",
    "km",
    "ko",
    "lo",
    "my",
    "th",
    "yue",
    "zh",
}


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, reference_unit in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_unit in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + int(reference_unit != hypothesis_unit),
                )
            )
        previous = current
    return previous[-1]


def _text_error(
    reference: str,
    hypothesis: str,
    *,
    languages: Sequence[str],
) -> dict[str, Any]:
    roots = {
        str(language).split("-", 1)[0].casefold()
        for language in languages
    }
    metric = (
        "cer"
        if roots and roots.issubset(_CHARACTER_SCORING_LANGUAGES)
        else "wer"
    )

    def normalized_units(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        if metric == "cer":
            return [
                character
                for character in normalized
                if character.isalnum()
            ]
        return [
            token.replace("_", "")
            for token in re.findall(r"\w+", normalized, flags=re.UNICODE)
            if token.replace("_", "")
        ]

    reference_units = normalized_units(reference)
    hypothesis_units = normalized_units(hypothesis)
    if not reference_units:
        return {
            "metric": metric,
            "errors": len(hypothesis_units),
            "referenceUnits": 0,
            "hypothesisUnits": len(hypothesis_units),
            "errorRate": 0.0 if not hypothesis_units else 1.0,
        }
    if len(reference_units) * len(hypothesis_units) > 10_000_000:
        raise ValueError("text error comparison exceeds bounded matrix size")
    errors = _edit_distance(reference_units, hypothesis_units)
    return {
        "metric": metric,
        "errors": errors,
        "referenceUnits": len(reference_units),
        "hypothesisUnits": len(hypothesis_units),
        "errorRate": round(errors / len(reference_units), 9),
    }


def _text_change(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
) -> str:
    baseline_rate = float(baseline["errorRate"])
    final_rate = float(final["errorRate"])
    if final_rate < baseline_rate:
        return "improved"
    if final_rate > baseline_rate:
        return "regressed"
    return "unchanged"


def _candidate_indexes(
    lattice: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    for domain in lattice["domains"]:
        for raw_group in domain["groups"]:
            group = {**raw_group, "domain": domain["domain"]}
            groups[str(group["groupId"])] = group
            for raw_candidate in group["candidates"]:
                candidates[str(raw_candidate["candidateId"])] = {
                    **raw_candidate,
                    "domain": domain["domain"],
                }
    return groups, candidates


def _selected_systems(
    lattice: Mapping[str, Any],
    arbitration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    groups, candidates = _candidate_indexes(lattice)
    selected: list[dict[str, Any]] = []
    system_counts: Counter[str] = Counter()
    changed_from_current = 0
    for selection in arbitration["selections"]:
        group = groups[str(selection["groupId"])]
        candidate = candidates[str(selection["selectedCandidateId"])]
        if candidate["candidateId"] != group["currentCandidateId"]:
            changed_from_current += 1
        systems = sorted(
            {
                f"{producer['systemId']}@{producer['revision']}"
                for producer in candidate["producers"]
            }
        )
        system_counts.update(systems)
        selected.append(
            {
                "domain": group["domain"],
                "scopeId": group["scopeId"],
                "candidateId": candidate["candidateId"],
                "changedFromCurrent": (
                    candidate["candidateId"] != group["currentCandidateId"]
                ),
                "producerSystems": systems,
            }
        )
    return selected, dict(sorted(system_counts.items())), changed_from_current


def _quality_summary(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    baseline = value.get("baselineMetrics")
    final = value.get("finalMetrics")
    if not isinstance(baseline, Mapping) or not isinstance(final, Mapping):
        raise ValueError(
            "quality report must contain baselineMetrics and finalMetrics"
        )

    def metric(source: Mapping[str, Any], *path: str) -> Any:
        current: Any = source
        for field in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(field)
        return current

    return {
        "overallOutcome": value.get("overallOutcome"),
        "baseline": {
            "speakerCountAbsoluteError": metric(
                baseline,
                "speakerCount",
                "speakerCountAbsoluteError",
            ),
            "werOrCer": metric(baseline, "finalText", "werOrCer"),
            "cpWer": metric(
                baseline,
                "jointTranscription",
                "cpWer",
                "errorRate",
            ),
            "tcpWer": metric(
                baseline,
                "jointTranscription",
                "tcpWer",
                "errorRate",
            ),
            "speakerAttributedWer": metric(
                baseline,
                "jointTranscription",
                "speakerAttributedWer",
                "errorRate",
            ),
            "languageAccuracy": metric(
                baseline,
                "language",
                "lexicalTokenWeighted",
                "accuracy",
            ),
            "hallucinationRate": metric(
                baseline,
                "contentIntegrity",
                "hallucinationRate",
            ),
            "deletionRate": metric(
                baseline,
                "contentIntegrity",
                "deletionRate",
            ),
        },
        "final": {
            "speakerCountAbsoluteError": metric(
                final,
                "speakerCount",
                "speakerCountAbsoluteError",
            ),
            "werOrCer": metric(final, "finalText", "werOrCer"),
            "cpWer": metric(
                final,
                "jointTranscription",
                "cpWer",
                "errorRate",
            ),
            "tcpWer": metric(
                final,
                "jointTranscription",
                "tcpWer",
                "errorRate",
            ),
            "speakerAttributedWer": metric(
                final,
                "jointTranscription",
                "speakerAttributedWer",
                "errorRate",
            ),
            "languageAccuracy": metric(
                final,
                "language",
                "lexicalTokenWeighted",
                "accuracy",
            ),
            "hallucinationRate": metric(
                final,
                "contentIntegrity",
                "hallucinationRate",
            ),
            "deletionRate": metric(
                final,
                "contentIntegrity",
                "deletionRate",
            ),
        },
    }


def _agent_review(
    value: Any,
    *,
    case_id: str,
) -> dict[str, Any]:
    if value is None:
        return {
            "status": "pending",
            "reviewerType": "agent-semantic-audit",
            "meaningPreservation": "not-reviewed",
            "speakerCoherence": "not-reviewed",
            "languageCoherence": "not-reviewed",
            "translationCompleteness": "not-reviewed",
            "notes": [],
            "humanApproval": False,
        }
    if not isinstance(value, Mapping):
        raise ValueError(f"{case_id}.agentSemanticReview must be an object")
    required = {
        "status",
        "meaningPreservation",
        "speakerCoherence",
        "languageCoherence",
        "translationCompleteness",
        "notes",
    }
    if set(value) != required:
        raise ValueError(
            f"{case_id}.agentSemanticReview fields do not match the contract"
        )
    status = value.get("status")
    if status not in {"pending", "completed"}:
        raise ValueError(f"{case_id}.agentSemanticReview.status is invalid")
    verdicts: dict[str, str] = {}
    for field in (
        "meaningPreservation",
        "speakerCoherence",
        "languageCoherence",
        "translationCompleteness",
    ):
        verdict = value.get(field)
        if verdict not in {
            "not-reviewed",
            "pass",
            "partial",
            "fail",
            "not-applicable",
        }:
            raise ValueError(
                f"{case_id}.agentSemanticReview.{field} is invalid"
            )
        verdicts[field] = str(verdict)
    notes = _string_list(
        value.get("notes"),
        f"{case_id}.agentSemanticReview.notes",
        allow_empty=True,
    )
    if status == "completed" and all(
        verdict == "not-reviewed" for verdict in verdicts.values()
    ):
        raise ValueError(
            f"{case_id} completed agent review contains no reviewed dimension"
        )
    return {
        "status": status,
        "reviewerType": "agent-semantic-audit",
        **verdicts,
        "notes": notes,
        "humanApproval": False,
    }


def _case(
    raw: Mapping[str, Any],
    *,
    manifest_root: Path,
) -> dict[str, Any]:
    case_id = _text(raw.get("id"), "cases[].id", maximum=160)
    scenarios = _string_list(raw.get("scenarios"), f"{case_id}.scenarios")
    expected = raw.get("expected")
    artifacts = raw.get("artifacts")
    if not isinstance(expected, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError(f"{case_id} requires expected and artifacts objects")
    expected_count = expected.get("speakerCount")
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError(f"{case_id}.expected.speakerCount is invalid")
    expected_languages = _string_list(
        expected.get("languages"),
        f"{case_id}.expected.languages",
        allow_empty=True,
    )
    reference_text = expected.get("transcript")
    if not isinstance(reference_text, str):
        raise ValueError(f"{case_id}.expected.transcript must be a string")

    transcript_path = _path(
        artifacts.get("transcript"),
        f"{case_id}.artifacts.transcript",
        manifest_root=manifest_root,
    )
    lattice_path = _path(
        artifacts.get("lattice"),
        f"{case_id}.artifacts.lattice",
        manifest_root=manifest_root,
    )
    arbitration_path = _path(
        artifacts.get("arbitration"),
        f"{case_id}.artifacts.arbitration",
        manifest_root=manifest_root,
    )
    composition_path = _path(
        artifacts.get("composition"),
        f"{case_id}.artifacts.composition",
        manifest_root=manifest_root,
        required=False,
    )
    quality_path = _path(
        artifacts.get("qualityReport"),
        f"{case_id}.artifacts.qualityReport",
        manifest_root=manifest_root,
        required=False,
    )
    audit_path = _path(
        artifacts.get("arbitrationAudit"),
        f"{case_id}.artifacts.arbitrationAudit",
        manifest_root=manifest_root,
        required=False,
    )
    shapes_path = _path(
        artifacts.get("modelResponseShapes"),
        f"{case_id}.artifacts.modelResponseShapes",
        manifest_root=manifest_root,
        required=False,
    )
    pipeline_metrics_path = _path(
        artifacts.get("pipelineMetrics"),
        f"{case_id}.artifacts.pipelineMetrics",
        manifest_root=manifest_root,
        required=False,
    )
    checkpoint_path = _path(
        artifacts.get("checkpoint"),
        f"{case_id}.artifacts.checkpoint",
        manifest_root=manifest_root,
        required=False,
    )
    assert transcript_path is not None
    assert lattice_path is not None
    assert arbitration_path is not None
    document = _load_object(transcript_path, f"{case_id}.transcript")
    transcript_sha = canonical_json_sha256(document)
    lattice = validate_semantic_candidate_lattice(
        _load_object(lattice_path, f"{case_id}.lattice"),
        expected_source_media_sha256=document["source"]["sha256"],
        expected_transcript_sha256=transcript_sha,
    )
    arbitration = validate_semantic_job_arbitration(
        _load_object(arbitration_path, f"{case_id}.arbitration"),
        expected_job_id=document["jobId"],
        expected_lattice=lattice,
    )
    if arbitration["status"] != "ready-to-compose":
        raise ValueError(f"{case_id} arbitration is not ready to compose")
    if composition_path is None:
        composition = build_semantic_composition(
            document,
            lattice,
            arbitration,
        )
    else:
        composition = validate_semantic_composition(
            _load_object(composition_path, f"{case_id}.composition"),
            expected_document=document,
            expected_lattice=lattice,
            expected_arbitration=arbitration,
        )

    quality = (
        _load_object(quality_path, f"{case_id}.qualityReport")
        if quality_path is not None
        else None
    )
    audit = (
        _load_object(audit_path, f"{case_id}.arbitrationAudit")
        if audit_path is not None
        else None
    )
    shapes = (
        _load_object(shapes_path, f"{case_id}.modelResponseShapes")
        if shapes_path is not None
        else None
    )
    pipeline_metrics = (
        _load_object(
            pipeline_metrics_path,
            f"{case_id}.pipelineMetrics",
        )
        if pipeline_metrics_path is not None
        else None
    )
    checkpoint = (
        _load_object(checkpoint_path, f"{case_id}.checkpoint")
        if checkpoint_path is not None
        else None
    )
    selected, systems, changed = _selected_systems(lattice, arbitration)
    baseline_count = int(document["speakerPolicy"]["resolvedCount"])
    final_count = int(composition["speakerPolicy"]["resolvedCount"])
    final_languages = sorted(
        {
            str(segment["language"])
            for segment in composition["segments"]
        }
    )
    translation_targets = list(arbitration.get("translationTargets") or [])
    arbitration_translations = list(
        arbitration.get("translations") or []
    )
    translations_by_target = {
        target: [
            {
                "segmentId": item["segmentId"],
                "text": item["text"],
            }
            for item in arbitration_translations
            if item["targetLanguage"] == target
        ]
        for target in translation_targets
    }
    response_shapes = (
        shapes.get("responseShapes")
        if isinstance(shapes, Mapping)
        else None
    )
    provider_calls = (
        len(response_shapes) if isinstance(response_shapes, list) else None
    )
    audit_metrics = (
        audit.get("metrics")
        if isinstance(audit, Mapping)
        and isinstance(audit.get("metrics"), Mapping)
        else None
    )
    provider_generation = (
        audit_metrics.get("providerGeneration")
        if isinstance(audit_metrics, Mapping)
        and isinstance(audit_metrics.get("providerGeneration"), Mapping)
        else None
    )
    if provider_calls is None and isinstance(provider_generation, Mapping):
        completed_calls = provider_generation.get("completedCalls")
        if isinstance(completed_calls, int) and not isinstance(
            completed_calls,
            bool,
        ):
            provider_calls = completed_calls
    elapsed = (
        audit_metrics.get("elapsedSeconds")
        if isinstance(audit_metrics, Mapping)
        else None
    )
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        raise ValueError(f"{case_id} arbitration elapsedSeconds is invalid")
    source_duration_seconds = float(document["source"]["durationMs"]) / 1000.0
    semantic_rtf = (
        round(float(elapsed) / source_duration_seconds, 6)
        if elapsed is not None
        else None
    )

    issue_flags: list[str] = []
    if expected_count is not None and final_count != expected_count:
        issue_flags.append("speaker-count-mismatch")
        if final_count == baseline_count:
            issue_flags.append("speaker-count-error-survived-semantic")
    expected_roots = {item.split("-", 1)[0].casefold() for item in expected_languages}
    final_roots = {item.split("-", 1)[0].casefold() for item in final_languages}
    if expected_roots != final_roots:
        issue_flags.append("language-set-mismatch")
    final_text = _final_text(composition)
    if expected_count is not None and expected_count > 0 and not final_text:
        issue_flags.append("empty-final-text")
    if expected_count == 0 and final_text:
        issue_flags.append("non-speech-hallucinated-text")
    baseline_text_quality = (
        _text_error(
            reference_text,
            _baseline_text(document),
            languages=expected_languages,
        )
        if reference_text.strip()
        else None
    )
    final_text_quality = (
        _text_error(
            reference_text,
            final_text,
            languages=expected_languages,
        )
        if reference_text.strip()
        else None
    )
    text_change = (
        _text_change(baseline_text_quality, final_text_quality)
        if baseline_text_quality is not None
        and final_text_quality is not None
        else "not-scored"
    )
    if text_change == "regressed":
        issue_flags.append("semantic-text-regression")
    elif (
        text_change == "unchanged"
        and final_text_quality is not None
        and final_text_quality["errorRate"] > 0
    ):
        issue_flags.append("semantic-text-error-unchanged")
    timeline_selection = next(
        (
            item
            for item in selected
            if item["domain"] == "speaker-cardinality-timeline"
        ),
        None,
    )
    if (
        isinstance(timeline_selection, Mapping)
        and timeline_selection.get("changedFromCurrent") is False
    ):
        issue_flags.append("timeline-selection-unchanged")
    expected_translation_count = (
        len(composition["segments"]) * len(translation_targets)
    )
    if len(arbitration_translations) != expected_translation_count:
        issue_flags.append("translation-coverage-incomplete")
    if semantic_rtf is not None and semantic_rtf > 10.0:
        issue_flags.append("semantic-rtf-over-budget")

    evidence = {
        "transcript": _artifact_evidence(transcript_path, document),
        "lattice": _artifact_evidence(lattice_path, lattice),
        "arbitration": _artifact_evidence(arbitration_path, arbitration),
        "composition": (
            _artifact_evidence(composition_path, composition)
            if composition_path is not None
            else {
                "path": None,
                "fileSha256": None,
                "canonicalSha256": canonical_json_sha256(composition),
            }
        ),
    }
    for field, path, value in (
        ("qualityReport", quality_path, quality),
        ("arbitrationAudit", audit_path, audit),
        ("modelResponseShapes", shapes_path, shapes),
        ("pipelineMetrics", pipeline_metrics_path, pipeline_metrics),
        ("checkpoint", checkpoint_path, checkpoint),
    ):
        if path is not None and value is not None:
            evidence[field] = _artifact_evidence(path, value)

    return {
        "id": case_id,
        "scenarios": scenarios,
        "expected": {
            "speakerCount": expected_count,
            "languages": expected_languages,
            "transcript": reference_text,
        },
        "truthCoverage": {
            "speakerCount": expected_count is not None,
            "languageSet": bool(expected_languages),
            "transcript": bool(reference_text.strip()),
        },
        "baseline": {
            "speakerCount": baseline_count,
            "languages": sorted(
                {str(segment["language"]) for segment in document["segments"]}
            ),
            "text": _baseline_text(document),
            "textQuality": baseline_text_quality,
        },
        "final": {
            "speakerCount": final_count,
            "languages": final_languages,
            "text": final_text,
            "textQuality": final_text_quality,
            "textChangeFromBaseline": text_change,
            "segments": copy.deepcopy(composition["segments"]),
            "translations": translations_by_target,
        },
        "semanticExecution": {
            "model": arbitration["model"],
            "provider": copy.deepcopy(arbitration["provider"]),
            "promptVersion": arbitration["promptVersion"],
            "candidateGroupCount": arbitration["metrics"][
                "candidateGroupCount"
            ],
            "selectedGroupCount": arbitration["metrics"][
                "selectedGroupCount"
            ],
            "changedFromCurrentCount": changed,
            "selectedCandidates": selected,
            "selectedProducerSystemCounts": systems,
            "providerCallCount": provider_calls,
            "providerGeneration": (
                dict(provider_generation)
                if isinstance(provider_generation, Mapping)
                else None
            ),
            "elapsedSeconds": (
                round(float(elapsed), 6) if elapsed is not None else None
            ),
            "sourceDurationSeconds": round(source_duration_seconds, 6),
            "realtimeFactor": semantic_rtf,
            "translationTargets": translation_targets,
            "translationGeneratedInArbitration": bool(translation_targets),
            "secondTranslationLlmCall": False,
        },
        "observedToolchain": {
            "pipelineStages": sorted(
                (
                    pipeline_metrics.get("runtime", {})
                    .get("stages", {})
                    .keys()
                )
                if isinstance(pipeline_metrics, Mapping)
                and isinstance(pipeline_metrics.get("runtime"), Mapping)
                and isinstance(
                    pipeline_metrics.get("runtime", {}).get("stages"),
                    Mapping,
                )
                else []
            ),
            "pipelineStageMetrics": copy.deepcopy(
                pipeline_metrics.get("runtime", {}).get("stages", {})
                if isinstance(pipeline_metrics, Mapping)
                and isinstance(pipeline_metrics.get("runtime"), Mapping)
                and isinstance(
                    pipeline_metrics.get("runtime", {}).get("stages"),
                    Mapping,
                )
                else {}
            ),
            "routing": copy.deepcopy(
                pipeline_metrics.get("routing")
                if isinstance(pipeline_metrics, Mapping)
                and isinstance(pipeline_metrics.get("routing"), Mapping)
                else None
            ),
            "semanticCheckpoint": copy.deepcopy(
                checkpoint.get("semantic")
                if isinstance(checkpoint, Mapping)
                and isinstance(checkpoint.get("semantic"), Mapping)
                else None
            ),
            "selectedCandidateProducerSystems": sorted(systems),
            "semanticArbitrator": {
                "model": arbitration["model"],
                "providerId": arbitration["provider"]["id"],
                "providerVersion": arbitration["provider"]["version"],
                "promptVersion": arbitration["promptVersion"],
            },
        },
        "quality": _quality_summary(quality),
        "agentSemanticReview": _agent_review(
            raw.get("agentSemanticReview"),
            case_id=case_id,
        ),
        "issueFlags": sorted(set(issue_flags)),
        "evidence": evidence,
        "releaseApproved": False,
    }


def _architecture_signals(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flags = Counter(
        flag
        for case in cases
        for flag in case.get("issueFlags", [])
        if isinstance(flag, str)
    )
    signals: list[dict[str, Any]] = []
    if flags["speaker-count-error-survived-semantic"]:
        signals.append(
            {
                "signal": "timeline-first-global-arbitration-required",
                "caseCount": flags[
                    "speaker-count-error-survived-semantic"
                ],
                "reason": (
                    "semantic text selection cannot repair speaker identity "
                    "after an incorrect timeline remains authoritative"
                ),
            }
        )
    if flags["language-set-mismatch"]:
        signals.append(
            {
                "signal": "joint-language-asr-scope-required",
                "caseCount": flags["language-set-mismatch"],
                "reason": (
                    "language and text must be decided in the same segment or "
                    "code-switch scope under the selected timeline"
                ),
            }
        )
    if flags["translation-coverage-incomplete"]:
        signals.append(
            {
                "signal": "whole-transcript-translation-reconstruction-required",
                "caseCount": flags["translation-coverage-incomplete"],
                "reason": (
                    "segment translations did not cover every final selected "
                    "ASR segment"
                ),
            }
        )
    if flags["semantic-text-regression"]:
        signals.append(
            {
                "signal": "semantic-text-regression-blocker",
                "caseCount": flags["semantic-text-regression"],
                "reason": (
                    "semantic arbitration must not replace a lower-error ASR "
                    "candidate with a higher-error final transcript"
                ),
            }
        )
    if flags["semantic-text-error-unchanged"]:
        signals.append(
            {
                "signal": "candidate-diversity-or-model-upgrade-required",
                "caseCount": flags["semantic-text-error-unchanged"],
                "reason": (
                    "mandatory arbitration cannot repair text when the lattice "
                    "contains no better candidate or the arbitrator cannot "
                    "identify it"
                ),
            }
        )
    if flags["semantic-rtf-over-budget"]:
        signals.append(
            {
                "signal": "semantic-runtime-budget-routing-required",
                "caseCount": flags["semantic-rtf-over-budget"],
                "reason": (
                    "semantic arbitration exceeded RTF 10; only structurally "
                    "ambiguous scopes should escalate to the strongest model"
                ),
            }
        )
    calls = [
        case.get("semanticExecution", {}).get("providerCallCount")
        for case in cases
        if isinstance(case.get("semanticExecution"), Mapping)
    ]
    measured_calls = [
        int(value)
        for value in calls
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if measured_calls and max(measured_calls) > 4:
        signals.append(
            {
                "signal": "scope-atomic-batching-required",
                "caseCount": sum(value > 4 for value in measured_calls),
                "reason": (
                    "speaker, language and ASR groups for one segment should "
                    "not be split across unrelated provider calls"
                ),
            }
        )
    elapsed_values = [
        case.get("semanticExecution", {}).get("elapsedSeconds")
        for case in cases
        if isinstance(case.get("semanticExecution"), Mapping)
    ]
    slow_cases = sum(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) > 120.0
        for value in elapsed_values
    )
    if slow_cases:
        signals.append(
            {
                "signal": "tiered-semantic-model-routing-required",
                "caseCount": slow_cases,
                "reason": (
                    "full semantic arbitration exceeded 120 seconds; routine "
                    "text and translation work should use an economical model "
                    "while structural conflicts escalate to a stronger model"
                ),
            }
        )
    return signals


def main() -> int:
    args = _parser().parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest = _load_object(manifest_path, "manifest")
    if set(manifest) != {"schemaVersion", "matrixId", "cases"}:
        raise ValueError("matrix manifest fields do not match the contract")
    if manifest.get("schemaVersion") != "1.0.0":
        raise ValueError("matrix manifest schemaVersion is unsupported")
    matrix_id = _text(manifest.get("matrixId"), "matrixId", maximum=160)
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("matrix cases must be a non-empty array")
    cases = [
        _case(raw, manifest_root=manifest_path.parent)
        for raw in raw_cases
        if isinstance(raw, Mapping)
    ]
    if len(cases) != len(raw_cases):
        raise ValueError("matrix cases must be objects")
    case_ids = [case["id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("matrix case IDs must be unique")
    issue_counts = dict(
        sorted(
            Counter(
                flag
                for case in cases
                for flag in case["issueFlags"]
            ).items()
        )
    )
    reviewed = sum(
        case["agentSemanticReview"]["status"] == "completed"
        for case in cases
    )
    truth_coverage = {
        dimension: sum(
            bool(case["truthCoverage"][dimension]) for case in cases
        )
        for dimension in ("speakerCount", "languageSet", "transcript")
    }
    report = {
        "schemaVersion": "1.0.0",
        "artifactType": "semantic-final-state-meta-analysis",
        "matrixId": matrix_id,
        "policy": {
            "acceptanceUnit": "speaker-language-time-finalText",
            "semanticArbitrationRequired": True,
            "translationCoGeneratedWithArbitration": True,
            "modelSelfEvaluationIsGroundTruth": False,
            "agentSemanticReviewIsHumanApproval": False,
            "promotionFromAggregateAverageAllowed": False,
        },
        "summary": {
            "caseCount": len(cases),
            "agentReviewedCaseCount": reviewed,
            "pendingAgentReviewCaseCount": len(cases) - reviewed,
            "issueCounts": issue_counts,
            "truthQualifiedCaseCounts": truth_coverage,
            "releaseApprovedCaseCount": 0,
        },
        "architectureSignals": _architecture_signals(cases),
        "cases": cases,
        "evidence": {
            "manifest": _artifact_evidence(manifest_path, manifest),
        },
        "releaseApproved": False,
    }
    atomic_write_json_no_replace(output, report)
    print(
        f"{len(cases)} cases: {reviewed} agent-reviewed, "
        f"{sum(issue_counts.values())} issue flags; releaseApproved=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
