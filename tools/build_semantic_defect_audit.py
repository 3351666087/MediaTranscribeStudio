"""Build a deterministic defect audit from one sealed semantic blind review.

The audit is a diagnostic artifact.  It never opens reference transcripts or
scorer truth; it only joins the sealed review groups with the identity vault,
the unblinded comparison, and optional benchmark model sidecars.  This makes
the Codex observations useful as regression inputs without changing the
reviewed evidence or the active production pointer.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
    validate_strict_json,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_TYPE = "semantic-defect-audit"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEVERITIES = ("blocker", "major")
_CATEGORIES = (
    "speaker-continuity",
    "asr-lexical",
    "language-asr-scope",
    "semantic-scope",
)


class SemanticDefectAuditError(ValueError):
    """Raised when review evidence cannot be joined without ambiguity."""


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticDefectAuditError(f"{field} must be an object")
    return dict(value)


def _text(value: Any, *, field: str, maximum: int = 20_000) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SemanticDefectAuditError(f"{field} must be trimmed non-empty text")
    return value


def _sha(value: Any, *, field: str) -> str:
    text = _text(value, field=field, maximum=80).casefold()
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    if _SHA256.fullmatch(text) is None:
        raise SemanticDefectAuditError(f"{field} must be a SHA-256 digest")
    return text


def _load(path: Path, *, field: str) -> tuple[Path, dict[str, Any], str]:
    candidate = path.expanduser().resolve(strict=True)
    if not candidate.is_file() or candidate.is_symlink():
        raise SemanticDefectAuditError(f"{field} must be a regular file")
    try:
        value = read_json_strict(candidate)
    except Exception as exc:  # pragma: no cover - backend reports details
        raise SemanticDefectAuditError(f"{field} is not strict JSON") from exc
    return candidate, _object(value, field=field), sha256_file(candidate)


def _canonical(value: Mapping[str, Any], *, field: str) -> str:
    declared = _sha(value.get("canonicalSha256"), field=f"{field}.canonicalSha256")
    body = dict(value)
    body.pop("canonicalSha256", None)
    actual = canonical_json_sha256(body)
    if actual != declared:
        raise SemanticDefectAuditError(f"{field} canonical SHA-256 does not match")
    return declared


def _artifact(path: Path, value: Mapping[str, Any], *, artifact_type: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "fileSha256": sha256_file(path),
        "canonicalSha256": _canonical(value, field=str(path)),
        "artifactType": artifact_type,
    }


def _candidate_maps(
    vault: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    rows = vault.get("cases")
    if not isinstance(rows, list) or not rows:
        raise SemanticDefectAuditError("identity vault cases must be non-empty")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    expected_aliases: frozenset[str] | None = None
    for index, raw_case in enumerate(rows):
        case = _object(raw_case, field=f"identity vault cases[{index}]")
        alias = _text(case.get("caseAlias"), field=f"identity vault cases[{index}].caseAlias")
        candidates = case.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise SemanticDefectAuditError(f"{alias} candidates must be non-empty")
        mapped: dict[str, dict[str, Any]] = {}
        for cindex, raw_candidate in enumerate(candidates):
            candidate = _object(
                raw_candidate,
                field=f"identity vault {alias}.candidates[{cindex}]",
            )
            candidate_alias = _text(
                candidate.get("candidateAlias"),
                field=f"identity vault {alias}.candidateAlias",
            )
            if candidate_alias in mapped:
                raise SemanticDefectAuditError(f"{alias} has duplicate candidate aliases")
            mapped[candidate_alias] = {
                "candidateAlias": candidate_alias,
                "model": _text(candidate.get("model"), field=f"{alias}.{candidate_alias}.model"),
                "modelId": _text(candidate.get("modelId"), field=f"{alias}.{candidate_alias}.modelId"),
                "digest": f"sha256:{_sha(candidate.get('digest'), field=f'{alias}.{candidate_alias}.digest')}",
                "resultCanonicalSha256": _sha(
                    candidate.get("resultCanonicalSha256"),
                    field=f"{alias}.{candidate_alias}.resultCanonicalSha256",
                ),
            }
        aliases = frozenset(mapped)
        if expected_aliases is None:
            expected_aliases = aliases
        elif aliases != expected_aliases:
            raise SemanticDefectAuditError("identity vault candidate aliases differ by case")
        result[alias] = mapped
    return result


def _comparison_models(comparison: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = comparison.get("models")
    if not isinstance(rows, list) or not rows:
        raise SemanticDefectAuditError("comparison models must be non-empty")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _object(raw, field=f"comparison.models[{index}]")
        model = _text(row.get("model"), field=f"comparison.models[{index}].model")
        if model in result:
            raise SemanticDefectAuditError("comparison contains duplicate models")
        result[model] = {
            "model": model,
            "registryModelId": _text(
                row.get("registryModelId"),
                field=f"comparison.models[{index}].registryModelId",
            ),
            "digest": f"sha256:{_sha(row.get('digest'), field=f'comparison.models[{index}].digest')}",
        }
    return result


def _winner_model(
    comparison: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    raw = _text(
        comparison.get("uniqueWinnerRegistryModelId"),
        field="comparison.uniqueWinnerRegistryModelId",
    )
    matches = [
        model
        for model, row in models.items()
        if raw in {model, row["registryModelId"]}
    ]
    if len(matches) != 1:
        raise SemanticDefectAuditError("comparison winner does not map to one model")
    return matches[0], str(models[matches[0]]["registryModelId"])


def _classify(reason: str, action: str) -> str:
    text = f"{reason} {action}".casefold()
    speaker = any(
        token in text
        for token in (
            "speaker",
            "attribution",
            "continuity",
            "speaker split",
            "speaker boundary",
            "turn change",
            "cardinality",
        )
    )
    asr = any(
        token in text
        for token in ("asr", "lexical", "wording", "token", "grammar", "malformed")
    )
    language = any(
        token in text
        for token in ("language", "script", "cantonese", "code-switch")
    )
    if speaker:
        return "speaker-continuity"
    if asr and language:
        return "language-asr-scope"
    if asr:
        return "asr-lexical"
    return "semantic-scope"


def _recommendation(category: str) -> dict[str, Any]:
    if category == "speaker-continuity":
        return {
            "requiredDomains": ["speaker-cardinality-timeline", "speaker-assignment"],
            "policy": "Resolve timeline/assignment before composing otherwise coherent text; unresolved attribution is non-deliverable.",
            "regressionEntrypoints": [
                "tests/test_semantic_composition.py::test_job_runner_decides_structure_before_scope_atomic_segment_triads",
                "tests/test_semantic_composition.py::test_runner_enforces_assignments_supported_by_committed_timeline",
            ],
        }
    if category == "asr-lexical":
        return {
            "requiredDomains": ["asr-text"],
            "policy": "A visibly malformed lexical span must request provider-native N-best before publication; disposition-only work is insufficient.",
            "regressionEntrypoints": [
                "tests/test_semantic_composition.py::test_model_can_request_bounded_candidate_generation_for_risky_domains",
                "tests/test_benchmark_real_multilingual_semantic_models.py::test_calibration_scores_default_challenger_without_leaking_hidden_target",
            ],
        }
    if category == "language-asr-scope":
        return {
            "requiredDomains": ["language-span", "asr-text"],
            "policy": "Language/script conflict and lexical uncertainty on one span must be escalated as an atomic language+ASR scope.",
            "regressionEntrypoints": [
                "tests/test_semantic_language_calibration.py::test_simplified_han_against_explicit_hant_claim_is_flagged",
                "tests/test_semantic_composition.py::test_job_runner_co_generates_translation_and_business_reuses_without_llm",
            ],
        }
    return {
        "requiredDomains": ["speech-disposition", "speaker-cardinality-timeline", "language-span", "asr-text"],
        "policy": "Do not spend a bounded round on an unrelated domain while a material deliverability defect remains visible.",
        "regressionEntrypoints": [
            "tests/test_semantic_composition.py::test_orchestrator_carries_duplicate_one_shot_request_as_exhausted",
            "tests/test_semantic_composition.py::test_orchestrator_accumulates_exhausted_requests_across_all_rounds",
        ],
    }


def _sidecar_summary(
    path: Path,
    value: Mapping[str, Any],
    *,
    benchmark_canonical_sha: str,
    comparison_models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if value.get("artifactType") != "real-multilingual-semantic-model-result":
        raise SemanticDefectAuditError(f"unsupported model sidecar artifact: {path}")
    if value.get("benchmarkArtifactSha256") != benchmark_canonical_sha:
        raise SemanticDefectAuditError(f"model sidecar is bound to another benchmark: {path}")
    model = _object(value.get("model"), field=f"{path}.model")
    identity = _object(model.get("modelIdentity"), field=f"{path}.model.modelIdentity")
    aggregate = _object(model.get("aggregate"), field=f"{path}.model.aggregate")
    calibration = _object(aggregate.get("calibration"), field=f"{path}.model.aggregate.calibration")
    model_id = _text(identity.get("modelId"), field=f"{path}.modelIdentity.modelId")
    digest = f"sha256:{_sha(identity.get('expectedDigest'), field=f'{path}.modelIdentity.expectedDigest')}"
    bound_models = [
        row
        for model_name, row in comparison_models.items()
        if model_id in {model_name, row["registryModelId"]}
        and digest == row["digest"]
    ]
    if len(bound_models) != 1:
        raise SemanticDefectAuditError(
            f"model sidecar identity is not bound to comparison: {path}"
        )
    return {
        "path": str(path),
        "fileSha256": sha256_file(path),
        "model": model_id,
        "digest": digest,
        "calibration": {
            "groupCount": calibration.get("groupCount"),
            "correctCount": calibration.get("correctCount"),
            "incorrectCount": calibration.get("incorrectCount"),
            "microAccuracy": calibration.get("microAccuracy"),
            "expectedActionCounts": copy.deepcopy(calibration.get("expectedActionCounts")),
            "actualActionCounts": copy.deepcopy(calibration.get("actualActionCounts")),
        },
    }


def build_semantic_defect_audit(
    *,
    sealed_review_path: Path,
    identity_vault_path: Path,
    comparison_path: Path,
    benchmark_path: Path | None = None,
    model_report_paths: Sequence[Path] = (),
    output_path: Path,
) -> dict[str, Any]:
    review_path, review, review_file_sha = _load(sealed_review_path, field="sealed review")
    vault_path, vault, vault_file_sha = _load(identity_vault_path, field="identity vault")
    comparison_file_path, comparison, comparison_file_sha = _load(comparison_path, field="comparison")
    review_canonical = _canonical(review, field="sealed review")
    vault_canonical = _canonical(vault, field="identity vault")
    comparison_canonical = _canonical(comparison, field="comparison")
    if review.get("artifactType") != "codex-semantic-blind-review":
        raise SemanticDefectAuditError("sealed review artifact type is unsupported")
    if vault.get("artifactType") != "anonymous-semantic-blind-review-identity-vault":
        raise SemanticDefectAuditError("identity vault artifact type is unsupported")
    if comparison.get("artifactType") != "semantic-model-blind-review-comparison":
        raise SemanticDefectAuditError("comparison artifact type is unsupported")
    candidate_maps = _candidate_maps(vault)
    comparison_models = _comparison_models(comparison)
    winner_model, winner_registry_id = _winner_model(comparison, comparison_models)
    review_cases = review.get("cases")
    if not isinstance(review_cases, list):
        raise SemanticDefectAuditError("sealed review cases must be an array")

    findings: list[dict[str, Any]] = []
    for index, raw_case in enumerate(review_cases):
        case = _object(raw_case, field=f"sealed review cases[{index}]")
        case_alias = _text(case.get("caseAlias"), field=f"sealed review cases[{index}].caseAlias")
        candidates = candidate_maps.get(case_alias)
        if candidates is None:
            raise SemanticDefectAuditError(f"review case {case_alias} is absent from identity vault")
        groups = case.get("assessmentGroups")
        if not isinstance(groups, list):
            raise SemanticDefectAuditError(f"review case {case_alias} groups must be an array")
        vault_case = next(
            raw
            for raw in vault.get("cases", [])
            if isinstance(raw, Mapping) and raw.get("caseAlias") == case_alias
        )
        case_id = _text(vault_case.get("caseId"), field=f"identity vault {case_alias}.caseId")
        for group_index, raw_group in enumerate(groups):
            group = _object(raw_group, field=f"sealed review {case_alias}.assessmentGroups[{group_index}]")
            severity = _text(group.get("severity"), field=f"sealed review {case_alias}.severity")
            if severity not in _SEVERITIES:
                continue
            aliases = group.get("candidateAliases")
            if not isinstance(aliases, list) or not aliases:
                raise SemanticDefectAuditError(f"sealed review {case_alias} has invalid candidate aliases")
            mapped_candidates = []
            for alias in aliases:
                if alias not in candidates:
                    raise SemanticDefectAuditError(f"{case_alias} references unknown candidate {alias}")
                mapped = copy.deepcopy(candidates[alias])
                model_row = comparison_models.get(mapped["model"])
                if model_row is None or model_row["digest"] != mapped["digest"]:
                    raise SemanticDefectAuditError(f"{case_alias}.{alias} is not bound to comparison")
                mapped["comparisonRegistryModelId"] = model_row["registryModelId"]
                mapped_candidates.append(mapped)
            reason = _text(group.get("reason"), field=f"sealed review {case_alias}.reason")
            action = _text(group.get("recommendedAction"), field=f"sealed review {case_alias}.recommendedAction")
            category = _classify(reason, action)
            finding_body = {
                "caseAlias": case_alias,
                "caseId": case_id,
                "groupIndex": group_index,
                "severity": severity,
                "candidateAliases": [candidate["candidateAlias"] for candidate in mapped_candidates],
                "candidates": mapped_candidates,
                "winnerIncluded": any(candidate["model"] == winner_model for candidate in mapped_candidates),
                "category": category,
                "reason": reason,
                "evidence": copy.deepcopy(_object(group.get("evidence"), field=f"sealed review {case_alias}.evidence")),
                "recommendedAction": action,
            }
            finding_body["findingId"] = canonical_json_sha256(finding_body)
            finding_body["regression"] = _recommendation(category)
            findings.append(finding_body)

    findings.sort(key=lambda row: (row["caseAlias"], row["groupIndex"]))
    severity_counts = Counter(row["severity"] for row in findings)
    category_counts = Counter(row["category"] for row in findings)
    model_counts = Counter(
        candidate["model"]
        for finding in findings
        for candidate in finding["candidates"]
    )
    sources = {
        "sealedReview": _artifact(review_path, review, artifact_type=str(review["artifactType"])),
        "identityVault": _artifact(vault_path, vault, artifact_type=str(vault["artifactType"])),
        "comparison": _artifact(comparison_file_path, comparison, artifact_type=str(comparison["artifactType"])),
    }
    benchmark_canonical = None
    if benchmark_path is not None:
        benchmark_file_path, benchmark, _ = _load(benchmark_path, field="benchmark")
        benchmark_canonical = _canonical(benchmark, field="benchmark")
        sources["benchmark"] = _artifact(
            benchmark_file_path,
            benchmark,
            artifact_type=str(benchmark.get("artifactType")),
        )
    sidecars = []
    if model_report_paths:
        if benchmark_canonical is None:
            raise SemanticDefectAuditError("model reports require --benchmark")
        seen_sidecars: set[tuple[str, str]] = set()
        for path in model_report_paths:
            sidecar_path, sidecar, _ = _load(path, field="model report")
            summary = _sidecar_summary(
                sidecar_path,
                sidecar,
                benchmark_canonical_sha=benchmark_canonical,
                comparison_models=comparison_models,
            )
            identity_key = (summary["model"], summary["digest"])
            if identity_key in seen_sidecars:
                raise SemanticDefectAuditError(
                    f"duplicate model sidecar identity: {summary['model']}"
                )
            seen_sidecars.add(identity_key)
            sidecars.append(summary)
        sidecars.sort(key=lambda row: row["model"])

    result: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "auditId": f"{comparison.get('comparisonId')}.major-blocker-defects",
        "policy": {
            "referenceTruthRead": False,
            "automaticMetricsUsedForSeverity": False,
            "sourceReviewSealedBeforeUnblind": True,
            "purpose": "codex-review-defects-to-regression-inputs",
        },
        "reviewBinding": {
            "reviewId": _text(review.get("reviewId"), field="sealed review.reviewId"),
            "blindBatchId": _text(comparison.get("blindBatchId"), field="comparison.blindBatchId"),
            "caseCount": len(candidate_maps),
            "candidateCount": len(comparison_models),
            "winnerModel": winner_model,
            "winnerRegistryModelId": winner_registry_id,
        },
        "sources": sources,
        "summary": {
            "findingCount": len(findings),
            "severityCounts": dict(sorted(severity_counts.items())),
            "categoryCounts": dict(sorted(category_counts.items())),
            "candidateModelCounts": dict(sorted(model_counts.items())),
            "winnerFindingCount": sum(1 for row in findings if row["winnerIncluded"]),
            "winnerMajorCount": sum(1 for row in findings if row["winnerIncluded"] and row["severity"] == "major"),
            "winnerBlockerCount": sum(1 for row in findings if row["winnerIncluded"] and row["severity"] == "blocker"),
        },
        "calibrationSidecars": sidecars,
        "findings": findings,
    }
    result["canonicalSha256"] = canonical_json_sha256(result)
    validate_strict_json(result)
    atomic_write_json_no_replace(output_path.resolve(), result)
    return {
        "output": str(output_path.resolve()),
        "outputFileSha256": sha256_file(output_path.resolve()),
        "canonicalSha256": result["canonicalSha256"],
        "summary": result["summary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-review", required=True, type=Path)
    parser.add_argument("--identity-vault", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--model-report", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_semantic_defect_audit(
        sealed_review_path=args.sealed_review,
        identity_vault_path=args.identity_vault,
        comparison_path=args.comparison,
        benchmark_path=args.benchmark,
        model_report_paths=args.model_report,
        output_path=args.output,
    )
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
