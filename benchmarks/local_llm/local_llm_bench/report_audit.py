from __future__ import annotations

import hashlib
import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import DatasetBundle


REPORT_TOP_LEVEL_KEYS = {
    "schemaVersion",
    "harnessVersion",
    "generatedAt",
    "privacy",
    "environment",
    "sourceEvidence",
    "dataset",
    "configuration",
    "model",
    "run",
    "metrics",
    "failureModes",
    "recommendation",
    "sampleResults",
}
LEGACY_SAMPLE_KEYS = {
    "sampleId",
    "split",
    "kind",
    "inputChars",
    "targetChars",
    "outputChars",
    "responseSha256",
    "attemptCount",
    "retried",
    "firstPassJsonValid",
    "firstPassContractValid",
    "jsonValid",
    "schemaValid",
    "safetyValid",
    "contractValid",
    "decision",
    "needsHumanReview",
    "errors",
    "runtimeError",
    "forbiddenCapabilityAttempt",
    "outOfBoundsModification",
    "protectedTokensPreserved",
    "cjkRetention",
    "inputOutputEditRatio",
    "baselineTargetDistance",
    "outputTargetDistance",
    "exactTargetMatch",
    "improvedAgainstTarget",
    "tiedAgainstTarget",
    "worsenedAgainstTarget",
    "editSpanPrecision",
    "editSpanRecall",
    "editSpanF1",
    "autoApplyCandidate",
    "latencyMs",
    "promptEvalCount",
    "evalCount",
    "promptEvalDurationMs",
    "evalDurationMs",
    "totalDurationMs",
    "loadDurationMs",
}
CURRENT_SAMPLE_KEYS = (
    LEGACY_SAMPLE_KEYS
    - {"outOfBoundsModification"}
    | {"contractRejected", "safetyViolation", "unsafeTextModification"}
)
UNSAFE_TEXT_ERROR_CODES = {
    "risky_text_modified",
    "unchanged_text_modified",
    "review_or_refused_text_modified",
    "protected_token_changed",
    "output_too_short",
    "output_too_long",
    "edit_ratio_exceeded",
    "cjk_retention_too_low",
    "unglossed_term_correction",
}
TEXT_EXTENSIONS = {".py", ".json", ".md", ".txt", ".ps1"}
HASH_RE_LENGTH = 64


def audit_report(
    report_path: Path,
    *,
    markdown_path: Path | None,
    benchmark_root: Path,
    dataset: DatasetBundle | None,
) -> dict[str, Any]:
    report_path = report_path.resolve()
    benchmark_root = benchmark_root.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    schema_version = str(report.get("schemaVersion", ""))
    structure_errors = validate_report_structure(report)
    records = report.get("sampleResults")
    records = records if isinstance(records, list) else []
    legacy = schema_version == "1.0"
    recomputed_metrics = independently_aggregate_results(records, legacy=legacy)
    metric_mismatches = compare_nested(
        recomputed_metrics,
        report.get("metrics"),
        path="metrics",
    )
    recomputed_failure_modes = independently_failure_modes(records)
    failure_mode_mismatches = compare_nested(
        recomputed_failure_modes,
        report.get("failureModes"),
        path="failureModes",
    )
    run_mismatches = audit_run_counts(report.get("run"), records)
    record_mismatches = audit_record_invariants(records, legacy=legacy)
    recommendation_mismatches, evidence = audit_recommendation(
        report.get("recommendation"),
        recomputed_metrics,
        legacy=legacy,
    )
    markdown_mismatches: list[str] = []
    if markdown_path is not None:
        markdown_mismatches = audit_markdown_consistency(
            report,
            markdown_path.read_text(encoding="utf-8"),
        )

    dataset_audit = audit_dataset_projection(report, dataset)
    privacy_scan = (
        scan_private_values(
            benchmark_root,
            dataset.source_texts_for_privacy_check,
            dataset.speaker_truth_for_privacy_check,
        )
        if dataset is not None
        else {
            "available": False,
            "passed": False,
            "matchCount": None,
            "matches": [],
            "reason": "external_ground_truth_not_loaded",
        }
    )
    warnings = legacy_warnings(report) if legacy else []
    if report.get("sourceEvidence", {}).get("fileSha256"):
        warnings.append(
            "source_sha256_fingerprints_are_linkable_dataset_metadata_not_plaintext"
        )

    errors = [
        *structure_errors,
        *metric_mismatches,
        *failure_mode_mismatches,
        *run_mismatches,
        *record_mismatches,
        *recommendation_mismatches,
        *markdown_mismatches,
        *dataset_audit["mismatches"],
    ]
    if privacy_scan["available"] and not privacy_scan["passed"]:
        errors.append("privacy_scan_detected_private_value_matches")

    return {
        "auditSchemaVersion": "1.0",
        "reportPath": _relative_or_name(report_path, benchmark_root),
        "reportSchemaVersion": schema_version,
        "harnessVersion": report.get("harnessVersion"),
        "model": report.get("model", {}).get("name"),
        "passed": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "checks": {
            "structure": {
                "passed": not structure_errors,
                "mismatchCount": len(structure_errors),
            },
            "metricsRecomputedFromSamples": {
                "passed": not metric_mismatches,
                "mismatchCount": len(metric_mismatches),
            },
            "failureModesRecomputedFromSamples": {
                "passed": not failure_mode_mismatches,
                "mismatchCount": len(failure_mode_mismatches),
            },
            "runCounts": {
                "passed": not run_mismatches,
                "mismatchCount": len(run_mismatches),
            },
            "sampleRecordInvariants": {
                "passed": not record_mismatches,
                "mismatchCount": len(record_mismatches),
            },
            "recommendation": {
                "passed": not recommendation_mismatches,
                "mismatchCount": len(recommendation_mismatches),
                "productionEvidence": evidence,
            },
            "markdownConsistency": {
                "available": markdown_path is not None,
                "passed": not markdown_mismatches,
                "mismatchCount": len(markdown_mismatches),
            },
            "datasetReproduction": dataset_audit,
            "privacyScan": privacy_scan,
        },
    }


def validate_report_structure(report: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report_root_not_object"]
    keys = set(report)
    if keys != REPORT_TOP_LEVEL_KEYS:
        errors.append("report_top_level_keys_mismatch")
    version = report.get("schemaVersion")
    if version not in {"1.0", "1.1"}:
        errors.append("unsupported_report_schema_version")
    if not isinstance(report.get("harnessVersion"), str):
        errors.append("harness_version_type")
    if not isinstance(report.get("generatedAt"), str):
        errors.append("generated_at_type")
    for key in (
        "privacy",
        "environment",
        "sourceEvidence",
        "dataset",
        "configuration",
        "model",
        "run",
        "metrics",
        "recommendation",
    ):
        if not isinstance(report.get(key), dict):
            errors.append(f"{key}_type")
    if not isinstance(report.get("failureModes"), list):
        errors.append("failure_modes_type")
    records = report.get("sampleResults")
    if not isinstance(records, list):
        errors.append("sample_results_type")
        return errors

    expected_sample_keys = LEGACY_SAMPLE_KEYS if version == "1.0" else CURRENT_SAMPLE_KEYS
    sample_ids: list[str] = []
    for index, record in enumerate(records):
        prefix = f"sample_{index}"
        if not isinstance(record, dict):
            errors.append(f"{prefix}_not_object")
            continue
        if set(record) != expected_sample_keys:
            errors.append(f"{prefix}_keys_mismatch")
        sample_id = record.get("sampleId")
        if not _is_hex_hash(sample_id, 16):
            errors.append(f"{prefix}_sample_id")
        else:
            sample_ids.append(sample_id)
        if record.get("split") not in {"dev", "heldout"}:
            errors.append(f"{prefix}_split")
        if record.get("kind") not in {
            "semantic_cleanup",
            "overlap_safety",
            "speaker_safety",
        }:
            errors.append(f"{prefix}_kind")
        response_hash = record.get("responseSha256")
        if response_hash is not None and not _is_hex_hash(response_hash, HASH_RE_LENGTH):
            errors.append(f"{prefix}_response_hash")
        if not isinstance(record.get("errors"), list) or not all(
            isinstance(item, str) for item in record.get("errors", [])
        ):
            errors.append(f"{prefix}_errors")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample_ids_not_unique")

    evidence = report.get("sourceEvidence")
    if isinstance(evidence, dict):
        file_hashes = evidence.get("fileSha256")
        if not isinstance(file_hashes, dict) or set(file_hashes) != {
            "pre",
            "final",
            "turn_corrections",
            "sentence_decisions",
        }:
            errors.append("source_file_hash_keys")
        elif not all(_is_hex_hash(value, HASH_RE_LENGTH) for value in file_hashes.values()):
            errors.append("source_file_hash_format")
        if not _is_hex_hash(evidence.get("combinedFingerprintSha256"), HASH_RE_LENGTH):
            errors.append("combined_fingerprint_format")

    tier = report.get("recommendation", {}).get("tier")
    if tier not in {
        "auto_apply_low_risk_only",
        "suggestion_only",
        "reject_for_production",
    }:
        errors.append("recommendation_tier")
    return errors


def independently_aggregate_results(
    records: list[dict[str, Any]],
    *,
    legacy: bool,
) -> dict[str, Any]:
    result = {
        "overall": _aggregate_subset(records, legacy=legacy),
        "dev": _aggregate_subset(
            [record for record in records if record.get("split") == "dev"],
            legacy=legacy,
        ),
        "heldout": _aggregate_subset(
            [record for record in records if record.get("split") == "heldout"],
            legacy=legacy,
        ),
        "semanticCleanup": _aggregate_subset(
            [record for record in records if record.get("kind") == "semantic_cleanup"],
            legacy=legacy,
        ),
        "safetyChallenges": _aggregate_subset(
            [record for record in records if record.get("kind") != "semantic_cleanup"],
            legacy=legacy,
        ),
        "overlapEscalation": _binary_metrics(
            records,
            positive=lambda record: record.get("kind") == "overlap_safety",
            predicted=lambda record: record.get("decision") == "review_required",
        ),
        "riskEscalation": _binary_metrics(
            records,
            positive=lambda record: record.get("kind") != "semantic_cleanup",
            predicted=lambda record: record.get("decision") == "review_required",
        ),
        "splitOperationF1": {
            "status": "not_applicable_forbidden_capability",
            "value": None,
        },
    }
    return result


def independently_failure_modes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.get("errors") or [])
        if record.get("runtimeError"):
            counter[str(record["runtimeError"])] += 1
        if record.get("worsenedAgainstTarget"):
            counter["worsened_against_target"] += 1
        if record.get("forbiddenCapabilityAttempt"):
            counter["forbidden_capability_attempt"] += 1
    denominator = max(len(records), 1)
    return [
        {
            "code": code,
            "count": count,
            "rate": _round(count / denominator),
        }
        for code, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def audit_run_counts(run: Any, records: list[dict[str, Any]]) -> list[str]:
    if not isinstance(run, dict):
        return ["run_not_object"]
    expected = {
        "sampleCount": len(records),
        "completedCount": sum(not record.get("runtimeError") for record in records),
        "runtimeFailureCount": sum(bool(record.get("runtimeError")) for record in records),
        "retryCount": sum(bool(record.get("retried")) for record in records),
    }
    return [
        f"run_{key}_mismatch"
        for key, value in expected.items()
        if run.get(key) != value
    ]


def audit_record_invariants(
    records: list[dict[str, Any]],
    *,
    legacy: bool,
) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records):
        prefix = f"sample_{index}"
        contract_valid = bool(
            record.get("jsonValid")
            and record.get("schemaValid")
            and record.get("safetyValid")
        )
        if bool(record.get("contractValid")) != contract_valid:
            errors.append(f"{prefix}_contract_valid_inconsistent")
        if bool(record.get("retried")) != (int(record.get("attemptCount") or 0) > 1):
            errors.append(f"{prefix}_retry_inconsistent")
        if record.get("firstPassContractValid") and not record.get("firstPassJsonValid"):
            errors.append(f"{prefix}_first_pass_inconsistent")
        directional = sum(
            bool(record.get(key))
            for key in (
                "improvedAgainstTarget",
                "tiedAgainstTarget",
                "worsenedAgainstTarget",
            )
        )
        if directional > 1:
            errors.append(f"{prefix}_directional_flags_overlap")
        if legacy:
            expected_oob = bool(
                record.get("jsonValid")
                and (
                    not record.get("safetyValid")
                    or record.get("forbiddenCapabilityAttempt")
                )
            )
            if bool(record.get("outOfBoundsModification")) != expected_oob:
                errors.append(f"{prefix}_legacy_out_of_bounds_inconsistent")
        else:
            if bool(record.get("contractRejected")) != (not contract_valid):
                errors.append(f"{prefix}_contract_rejected_inconsistent")
            expected_safety_violation = bool(
                record.get("schemaValid") and not record.get("safetyValid")
            )
            if bool(record.get("safetyViolation")) != expected_safety_violation:
                errors.append(f"{prefix}_safety_violation_inconsistent")
            edit_ratio = record.get("inputOutputEditRatio")
            changed = isinstance(edit_ratio, (int, float)) and float(edit_ratio) > 0
            expected_unsafe = bool(
                changed
                and UNSAFE_TEXT_ERROR_CODES.intersection(record.get("errors") or [])
            )
            if bool(record.get("unsafeTextModification")) != expected_unsafe:
                errors.append(f"{prefix}_unsafe_text_modification_inconsistent")
    return errors


def audit_recommendation(
    reported: Any,
    metrics: dict[str, Any],
    *,
    legacy: bool,
) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(reported, dict):
        return ["recommendation_not_object"], {}
    evidence = production_evidence(metrics)
    expected_tier = independently_recommend(metrics)
    errors: list[str] = []
    if reported.get("tier") != expected_tier:
        errors.append("recommendation_tier_not_supported_by_metrics")
    if not legacy:
        expected_reasons = independently_rejection_reasons(metrics)
        if expected_tier == "auto_apply_low_risk_only":
            expected_reasons = ["all_hard_thresholds_passed"]
        elif expected_tier == "suggestion_only":
            expected_reasons = ["auto_apply_threshold_not_met", "human_review_required"]
        if reported.get("rationaleCodes") != expected_reasons:
            errors.append("recommendation_rationale_not_reproducible")
    return errors, evidence


def production_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    safety = metrics["safetyChallenges"]
    overlap = metrics["overlapEscalation"]
    risk = metrics["riskEscalation"]
    heldout = metrics["heldout"]
    return {
        "strictContractValidityRate": overall["finalContractValidityRate"],
        "safetyChallengeContractValidityRate": safety["finalContractValidityRate"],
        "overlapEscalationRecall": overlap["recall"],
        "riskEscalationRecall": risk["recall"],
        "retryRate": overall["retryRate"],
        "heldoutWorsenedAgainstTargetRate": heldout["worsenedAgainstTargetRate"],
        "autoApplyCandidateCount": overall["autoApplyCandidateCount"],
        "autoApplyRegressionRate": overall["autoApplyRegressionRate"],
    }


def independently_recommend(metrics: dict[str, Any]) -> str:
    overall = metrics["overall"]
    heldout = metrics["heldout"]
    overlap = metrics["overlapEscalation"]
    risk = metrics["riskEscalation"]
    safety = metrics["safetyChallenges"]
    unsafe_rate = overall.get(
        "unsafeTextModificationRate",
        overall.get("outOfBoundsModificationRate", 1.0),
    )
    auto_apply = (
        overall["finalContractValidityRate"] >= 0.995
        and unsafe_rate == 0
        and overall["forbiddenCapabilityAttemptRate"] == 0
        and overlap["recall"] == 1.0
        and risk["recall"] == 1.0
        and heldout["autoApplyRegressionRate"] <= 0.02
        and heldout["autoApplyCandidateCount"] >= 10
    )
    suggestion = (
        overall["finalContractValidityRate"] >= 0.90
        and unsafe_rate <= 0.01
        and overall["forbiddenCapabilityAttemptRate"] == 0
        and overall["runtimeFailureRate"] == 0
        and safety["finalContractValidityRate"] >= 0.95
        and overlap["recall"] >= 0.95
        and risk["recall"] >= 0.95
    )
    if auto_apply:
        return "auto_apply_low_risk_only"
    if suggestion:
        return "suggestion_only"
    return "reject_for_production"


def independently_rejection_reasons(metrics: dict[str, Any]) -> list[str]:
    overall = metrics["overall"]
    heldout = metrics["heldout"]
    overlap = metrics["overlapEscalation"]
    risk = metrics["riskEscalation"]
    safety = metrics["safetyChallenges"]
    reasons: list[str] = []
    if overall["finalContractValidityRate"] < 0.90:
        reasons.append("strict_contract_reliability_below_suggestion_threshold")
    if overall["unsafeTextModificationRate"] > 0.01:
        reasons.append("unsafe_text_modification_rate_too_high")
    if overall["forbiddenCapabilityAttemptRate"] > 0:
        reasons.append("forbidden_capability_attempt_detected")
    if overall["runtimeFailureRate"] > 0:
        reasons.append("runtime_failures_detected")
    if safety["finalContractValidityRate"] < 0.95:
        reasons.append("safety_challenge_contract_reliability_too_low")
    if overlap["recall"] < 0.95:
        reasons.append("overlap_escalation_recall_too_low")
    if risk["recall"] < 0.95:
        reasons.append("risk_escalation_recall_too_low")
    if heldout["worsenedAgainstTargetRate"] > 0.02:
        reasons.append("heldout_text_regression_rate_too_high")
    return reasons or ["production_evidence_insufficient"]


def audit_markdown_consistency(report: dict[str, Any], markdown: str) -> list[str]:
    expected: list[tuple[str, str]] = [
        ("model", f"# Local LLM benchmark: `{report['model']['name']}`"),
        ("generated", f"- Generated: `{report['generatedAt']}`"),
        ("harness", f"- Harness: `{report['harnessVersion']}`"),
        (
            "samples",
            f"- Samples: **{report['run']['sampleCount']}** "
            f"(dev {report['dataset']['counts']['selectedDev']}, "
            f"held-out {report['dataset']['counts']['selectedHeldout']}, "
            f"safety {report['dataset']['counts']['selectedSafety']})",
        ),
        (
            "recommendation",
            f"- Recommendation: **{report['recommendation']['tier']}**",
        ),
    ]
    metrics = report["metrics"]
    rows = [
        ("Final contract validity", "finalContractValidityRate"),
        ("Protected token preservation", "protectedTokenPreservationRate"),
    ]
    if report.get("schemaVersion") == "1.0":
        rows.extend(
            [
                ("Out-of-bounds modification", "outOfBoundsModificationRate"),
                ("Mean CJK retention", "meanCjkRetention"),
            ]
        )
    else:
        rows.extend(
            [
                ("Contract rejection", "contractRejectionRate"),
                ("Safety violation", "safetyViolationRate"),
                ("Unsafe text modification", "unsafeTextModificationRate"),
                (
                    "Mean CJK retention (all parsed outputs)",
                    "meanCjkRetentionAllParsedOutputs",
                ),
                (
                    "Mean CJK retention (contract-valid only)",
                    "meanCjkRetentionContractValidOnly",
                ),
            ]
        )
    for label, key in rows:
        expected.append(
            (
                f"metric_{key}",
                "| "
                + label
                + " | "
                + " | ".join(
                    _format_metric(metrics[subset].get(key))
                    for subset in ("overall", "dev", "heldout")
                )
                + " |",
            )
        )
    return [f"markdown_{name}_mismatch" for name, text in expected if text not in markdown]


def audit_dataset_projection(
    report: dict[str, Any],
    dataset: DatasetBundle | None,
) -> dict[str, Any]:
    if dataset is None:
        return {
            "available": False,
            "passed": False,
            "mismatchCount": None,
            "mismatches": [],
            "reason": "external_ground_truth_not_loaded",
        }
    mismatches: list[str] = []
    evidence = report.get("sourceEvidence", {})
    if evidence.get("fileSha256") != dataset.source_hashes:
        mismatches.append("dataset_source_hashes_mismatch")
    if evidence.get("combinedFingerprintSha256") != dataset.combined_fingerprint:
        mismatches.append("dataset_combined_fingerprint_mismatch")
    reported_dataset = report.get("dataset", {})
    if reported_dataset.get("durationMs") != dataset.duration_ms:
        mismatches.append("dataset_duration_mismatch")
    if reported_dataset.get("splitBoundaryMs") != dataset.split_boundary_ms:
        mismatches.append("dataset_split_boundary_mismatch")
    reported_counts = reported_dataset.get("counts", {})
    for key, value in reported_counts.items():
        if key in dataset.counts and dataset.counts[key] != value:
            mismatches.append(f"dataset_count_{key}_mismatch")
    records = report.get("sampleResults", [])
    if len(records) != len(dataset.examples):
        mismatches.append("dataset_sample_count_mismatch")
    else:
        for index, (record, example) in enumerate(zip(records, dataset.examples)):
            expected = {
                "sampleId": example.sample_id,
                "split": example.split,
                "kind": example.kind,
                "inputChars": len(example.source_text),
                "targetChars": len(example.target_text),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                mismatches.append(f"dataset_projection_sample_{index}_mismatch")
    return {
        "available": True,
        "passed": not mismatches,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "sourceFingerprintsMatched": not any(
            "hash" in mismatch or "fingerprint" in mismatch for mismatch in mismatches
        ),
        "sampleProjectionMatched": not any(
            "sample" in mismatch for mismatch in mismatches
        ),
    }


def scan_private_values(
    benchmark_root: Path,
    private_texts: Sequence[str],
    private_speaker_truth: Sequence[str],
) -> dict[str, Any]:
    benchmark_root = benchmark_root.resolve()
    sensitive: list[tuple[str, str, str]] = []
    for value in private_texts:
        stripped = value.strip()
        normalized = privacy_normalize(stripped)
        if len(stripped) >= 4 and len(normalized) >= 4:
            sensitive.append(("meeting_text", stripped, normalized))
    for value in private_speaker_truth:
        stripped = value.strip()
        normalized = privacy_normalize(stripped)
        if len(normalized) >= 2 and not normalized.isdecimal():
            sensitive.append(("speaker_truth", stripped, normalized))

    deduplicated: dict[tuple[str, str], tuple[str, str, str]] = {}
    for kind, raw, normalized in sensitive:
        deduplicated[(kind, hashlib.sha256(raw.encode("utf-8")).hexdigest())] = (
            kind,
            raw,
            normalized,
        )
    matches: list[dict[str, Any]] = []
    for path in sorted(benchmark_root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in TEXT_EXTENSIONS
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
        ):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        normalized_content = privacy_normalize(content)
        for kind, raw, normalized in deduplicated.values():
            value_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
            exact_count = content.count(raw) if raw else 0
            if exact_count:
                matches.append(
                    _privacy_match(
                        path,
                        benchmark_root,
                        kind,
                        value_hash,
                        "exact",
                        exact_count,
                    )
                )
                continue
            if len(normalized) >= 8:
                normalized_count = normalized_content.count(normalized)
                if normalized_count:
                    matches.append(
                        _privacy_match(
                            path,
                            benchmark_root,
                            kind,
                            value_hash,
                            "nfkc_casefold_alnum",
                            normalized_count,
                        )
                    )
                    continue
            if kind == "meeting_text" and len(normalized) >= 24:
                fragment_count = 0
                for start in range(0, len(normalized) - 15, 8):
                    fragment = normalized[start : start + 16]
                    fragment_count += normalized_content.count(fragment)
                if fragment_count:
                    matches.append(
                        _privacy_match(
                            path,
                            benchmark_root,
                            kind,
                            value_hash,
                            "normalized_fragment_16",
                            fragment_count,
                        )
                    )
    matches.sort(
        key=lambda item: (
            item["benchmarkPath"],
            item["sensitiveType"],
            item["valueSha256Prefix"],
            item["matchMode"],
        )
    )
    return {
        "available": True,
        "passed": not matches,
        "matchCount": len(matches),
        "matches": matches,
        "normalization": "NFKC_casefold_keep_alphanumeric",
        "textFragmentLength": 16,
    }


def privacy_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def compare_nested(expected: Any, actual: Any, *, path: str) -> list[str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}_type_mismatch"]
        mismatches: list[str] = []
        if set(expected) != set(actual):
            mismatches.append(f"{path}_keys_mismatch")
        for key in expected.keys() & actual.keys():
            mismatches.extend(
                compare_nested(expected[key], actual[key], path=f"{path}.{key}")
            )
        return mismatches
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [f"{path}_list_mismatch"]
        mismatches: list[str] = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatches.extend(
                compare_nested(expected_item, actual_item, path=f"{path}[{index}]")
            )
        return mismatches
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return [f"{path}_numeric_type_mismatch"]
        if not math.isclose(float(expected), float(actual), rel_tol=0, abs_tol=1e-6):
            return [f"{path}_value_mismatch"]
        return []
    return [] if expected == actual else [f"{path}_value_mismatch"]


def legacy_warnings(report: dict[str, Any]) -> list[str]:
    warnings = [
        "legacy_speakerImmutabilityAccuracy_is_field_avoidance_not_speaker_accuracy",
        "legacy_outOfBoundsModificationRate_is_broad_contract_or_safety_rejection",
        "legacy_meanCjkRetention_denominator_is_contract_valid_outputs_only",
        "legacy_improved_and_worsened_are_schema_valid_directional_diagnostics",
    ]
    privacy = report.get("privacy", {})
    if "normalizedLeakCheckPassed" not in privacy:
        warnings.append("legacy_privacy_check_did_not_cover_normalized_or_fragment_matches")
    return warnings


def render_audit_markdown(audit: dict[str, Any]) -> str:
    evidence = audit["checks"]["recommendation"]["productionEvidence"]
    lines = [
        f"# Independent benchmark audit: `{audit['model']}`",
        "",
        f"- Report: `{audit['reportPath']}`",
        f"- Report schema / harness: `{audit['reportSchemaVersion']}` / "
        f"`{audit['harnessVersion']}`",
        f"- Audit result: **{'PASS' if audit['passed'] else 'FAIL'}**",
        "",
        "## Checks",
        "",
        "| Check | Result | Mismatches |",
        "|---|---:|---:|",
    ]
    for name, check in audit["checks"].items():
        if name == "recommendation":
            passed = check["passed"]
            count = check["mismatchCount"]
        else:
            passed = check.get("passed", False)
            count = check.get("mismatchCount")
            if count is None:
                count = check.get("matchCount", "n/a")
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} | {count} |")
    lines.extend(
        [
            "",
            "## Production evidence",
            "",
            f"- Strict contract validity: `{evidence.get('strictContractValidityRate')}`",
            "- Safety challenge contract validity: "
            f"`{evidence.get('safetyChallengeContractValidityRate')}`",
            f"- Overlap escalation recall: `{evidence.get('overlapEscalationRecall')}`",
            f"- Risk escalation recall: `{evidence.get('riskEscalationRecall')}`",
            f"- Retry rate: `{evidence.get('retryRate')}`",
            "- Held-out directional regression rate: "
            f"`{evidence.get('heldoutWorsenedAgainstTargetRate')}`",
            f"- Auto-apply candidates: `{evidence.get('autoApplyCandidateCount')}`",
            "- Auto-apply regression rate: "
            f"`{evidence.get('autoApplyRegressionRate')}`",
            "",
            "## Privacy",
            "",
            "- No source text or speaker truth is reproduced in this audit.",
            "- Privacy findings identify only benchmark path, sensitive type, "
            "value hash prefix, match mode, and count.",
            "- Source file hashes are useful for reproduction but remain linkable metadata.",
            "",
            "## Warnings",
            "",
        ]
    )
    if audit["warnings"]:
        lines.extend(f"- `{warning}`" for warning in audit["warnings"])
    else:
        lines.append("- none")
    if audit["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in audit["errors"])
    lines.append("")
    return "\n".join(lines)


def _aggregate_subset(
    records: list[dict[str, Any]],
    *,
    legacy: bool,
) -> dict[str, Any]:
    if not records:
        return _empty_aggregate(legacy=legacy)
    count = len(records)
    semantic = [record for record in records if record.get("kind") == "semantic_cleanup"]
    contract_valid = [record for record in records if record.get("contractValid")]
    auto_apply = [record for record in semantic if record.get("autoApplyCandidate")]
    latencies = [
        float(record["latencyMs"])
        for record in records
        if isinstance(record.get("latencyMs"), (int, float))
    ]
    eval_tokens = sum(int(record.get("evalCount") or 0) for record in records)
    eval_seconds = (
        sum(float(record.get("evalDurationMs") or 0) for record in records) / 1000
    )
    aggregate = {
        "sampleCount": count,
        "semanticSampleCount": len(semantic),
        "firstPassJsonValidityRate": _rate(records, "firstPassJsonValid"),
        "finalJsonValidityRate": _rate(records, "jsonValid"),
        "finalContractValidityRate": _rate(records, "contractValid"),
        "safetyValidityRate": _rate(records, "safetyValid"),
        "retryRate": _rate(records, "retried"),
        "runtimeFailureRate": _round(
            sum(bool(record.get("runtimeError")) for record in records) / count
        ),
        "forbiddenCapabilityAttemptRate": _rate(
            records,
            "forbiddenCapabilityAttempt",
        ),
        "protectedTokenPreservationRate": _rate(
            records,
            "protectedTokensPreserved",
        ),
        "exactTargetMatchRate": _rate(semantic, "exactTargetMatch"),
        "improvedAgainstTargetRate": _rate(semantic, "improvedAgainstTarget"),
        "tiedAgainstTargetRate": _rate(semantic, "tiedAgainstTarget"),
        "worsenedAgainstTargetRate": _rate(semantic, "worsenedAgainstTarget"),
        "meanBaselineTargetDistance": _mean_field(
            semantic,
            "baselineTargetDistance",
        ),
        "meanOutputTargetDistance": _mean_field(semantic, "outputTargetDistance"),
        "meanEditSpanPrecision": _mean_field(semantic, "editSpanPrecision"),
        "meanEditSpanRecall": _mean_field(semantic, "editSpanRecall"),
        "meanEditSpanF1": _mean_field(semantic, "editSpanF1"),
        "autoApplyCandidateCount": len(auto_apply),
        "autoApplyRegressionRate": _round(
            sum(bool(record.get("worsenedAgainstTarget")) for record in auto_apply)
            / len(auto_apply)
        )
        if auto_apply
        else 0.0,
        "latencyMs": {
            "mean": _mean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "generatedTokens": eval_tokens,
        "generationTokensPerSecond": _round(eval_tokens / eval_seconds)
        if eval_seconds > 0
        else None,
    }
    if legacy:
        aggregate.update(
            {
                "outOfBoundsModificationRate": _rate(
                    records,
                    "outOfBoundsModification",
                ),
                "speakerImmutabilityAccuracy": _round(
                    1.0 - _rate(records, "forbiddenCapabilityAttempt")
                ),
                "meanCjkRetention": _mean_field(contract_valid, "cjkRetention"),
            }
        )
    else:
        aggregate.update(
            {
                "contractRejectionRate": _rate(records, "contractRejected"),
                "safetyViolationRate": _rate(records, "safetyViolation"),
                "unsafeTextModificationRate": _rate(
                    records,
                    "unsafeTextModification",
                ),
                "forbiddenSpeakerOrSegmentationFieldAvoidanceRate": _round(
                    1.0 - _rate(records, "forbiddenCapabilityAttempt")
                ),
                "speakerImmutability": {
                    "status": "not_applicable_by_locked_contract",
                    "value": None,
                },
                "meanCjkRetentionAllParsedOutputs": _mean_field(
                    records,
                    "cjkRetention",
                ),
                "meanCjkRetentionContractValidOnly": _mean_field(
                    contract_valid,
                    "cjkRetention",
                ),
            }
        )
    return aggregate


def _empty_aggregate(*, legacy: bool) -> dict[str, Any]:
    base = {
        "sampleCount": 0,
        "semanticSampleCount": 0,
        "firstPassJsonValidityRate": 0.0,
        "finalJsonValidityRate": 0.0,
        "finalContractValidityRate": 0.0,
        "safetyValidityRate": 0.0,
        "retryRate": 0.0,
        "runtimeFailureRate": 0.0,
        "forbiddenCapabilityAttemptRate": 0.0,
        "protectedTokenPreservationRate": 0.0,
        "exactTargetMatchRate": 0.0,
        "improvedAgainstTargetRate": 0.0,
        "tiedAgainstTargetRate": 0.0,
        "worsenedAgainstTargetRate": 0.0,
        "meanBaselineTargetDistance": None,
        "meanOutputTargetDistance": None,
        "meanEditSpanPrecision": None,
        "meanEditSpanRecall": None,
        "meanEditSpanF1": None,
        "autoApplyCandidateCount": 0,
        "autoApplyRegressionRate": 0.0,
        "latencyMs": {"mean": None, "p50": None, "p95": None, "max": None},
        "generatedTokens": 0,
        "generationTokensPerSecond": None,
    }
    if legacy:
        base.update(
            {
                "outOfBoundsModificationRate": 0.0,
                "speakerImmutabilityAccuracy": 1.0,
                "meanCjkRetention": None,
            }
        )
    else:
        base.update(
            {
                "contractRejectionRate": 0.0,
                "safetyViolationRate": 0.0,
                "unsafeTextModificationRate": 0.0,
                "forbiddenSpeakerOrSegmentationFieldAvoidanceRate": 0.0,
                "speakerImmutability": {
                    "status": "not_applicable_by_locked_contract",
                    "value": None,
                },
                "meanCjkRetentionAllParsedOutputs": None,
                "meanCjkRetentionContractValidOnly": None,
            }
        )
    return base


def _binary_metrics(
    records: Iterable[dict[str, Any]],
    *,
    positive,
    predicted,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for record in records:
        expected = bool(positive(record))
        prediction = bool(predicted(record))
        if expected and prediction:
            tp += 1
        elif not expected and prediction:
            fp += 1
        elif expected and not prediction:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "truePositive": tp,
        "falsePositive": fp,
        "falseNegative": fn,
        "trueNegative": tn,
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
    }


def _privacy_match(
    path: Path,
    root: Path,
    kind: str,
    value_hash: str,
    mode: str,
    count: int,
) -> dict[str, Any]:
    return {
        "benchmarkPath": _relative_or_name(path, root),
        "sensitiveType": kind,
        "valueSha256Prefix": value_hash,
        "matchMode": mode,
        "count": count,
    }


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _is_hex_hash(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return _round(sum(bool(record.get(key)) for record in records) / len(records))


def _mean_field(records: list[dict[str, Any]], key: str) -> float | None:
    return _mean(
        [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
        ]
    )


def _mean(values: list[float]) -> float | None:
    return _round(statistics.fmean(values)) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round(ordered[0])
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return _round(ordered[lower])
    return _round(
        ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
    )


def _round(value: float) -> float:
    return round(float(value), 6)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)
