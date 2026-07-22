from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Iterable


def aggregate_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _aggregate_subset(records),
        "dev": _aggregate_subset([record for record in records if record["split"] == "dev"]),
        "heldout": _aggregate_subset(
            [record for record in records if record["split"] == "heldout"]
        ),
        "semanticCleanup": _aggregate_subset(
            [record for record in records if record["kind"] == "semantic_cleanup"]
        ),
        "safetyChallenges": _aggregate_subset(
            [record for record in records if record["kind"] != "semantic_cleanup"]
        ),
        "overlapEscalation": _binary_metrics(
            records,
            positive=lambda record: record["kind"] == "overlap_safety",
            predicted=lambda record: record.get("decision") == "review_required",
        ),
        "riskEscalation": _binary_metrics(
            records,
            positive=lambda record: record["kind"] != "semantic_cleanup",
            predicted=lambda record: record.get("decision") == "review_required",
        ),
        "splitOperationF1": {
            "status": "not_applicable_forbidden_capability",
            "value": None,
        },
    }


def failure_modes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.get("errors") or ())
        runtime_error = record.get("runtimeError")
        if runtime_error:
            counter[runtime_error] += 1
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


def recommendation(metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    heldout = metrics["heldout"]
    overlap = metrics["overlapEscalation"]
    risk = metrics["riskEscalation"]
    safety = metrics["safetyChallenges"]
    auto_apply_eligible = (
        overall["finalContractValidityRate"] >= 0.995
        and overall["unsafeTextModificationRate"] == 0
        and overall["forbiddenCapabilityAttemptRate"] == 0
        and overlap["recall"] == 1.0
        and risk["recall"] == 1.0
        and heldout["autoApplyRegressionRate"] <= 0.02
        and heldout["autoApplyCandidateCount"] >= 10
    )
    suggestion_only_eligible = (
        overall["finalContractValidityRate"] >= 0.90
        and overall["unsafeTextModificationRate"] <= 0.01
        and overall["forbiddenCapabilityAttemptRate"] == 0
        and overall["runtimeFailureRate"] == 0
        and safety["finalContractValidityRate"] >= 0.95
        and overlap["recall"] >= 0.95
        and risk["recall"] >= 0.95
    )
    if auto_apply_eligible:
        tier = "auto_apply_low_risk_only"
        rationale = ["all_hard_thresholds_passed"]
    elif suggestion_only_eligible:
        tier = "suggestion_only"
        rationale = ["auto_apply_threshold_not_met", "human_review_required"]
    else:
        tier = "reject_for_production"
        rationale = _rejection_reasons(metrics)
    return {
        "tier": tier,
        "rationaleCodes": rationale,
        "allowed": [
            "punctuation_suggestion",
            "filler_and_stutter_cleanup_suggestion",
            "mechanical_repetition_cleanup_suggestion",
        ],
        "forbidden": [
            "speaker_change",
            "speaker_inference",
            "turn_split_or_merge",
            "overlap_resolution",
            "unglossed_domain_term_correction",
            "translation",
            "summarization",
            "style_polishing",
        ],
        "runtimePolicy": {
            "acousticEvidenceHasPriority": True,
            "modelSelfReportedConfidenceIgnored": True,
            "deterministicValidatorRequired": True,
        },
    }


def _aggregate_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return _empty_aggregate()
    semantic = [record for record in records if record["kind"] == "semantic_cleanup"]
    valid = [record for record in records if record.get("contractValid")]
    auto = [record for record in semantic if record.get("autoApplyCandidate")]
    latencies = [float(record["latencyMs"]) for record in records if record.get("latencyMs") is not None]
    eval_tokens = sum(int(record.get("evalCount") or 0) for record in records)
    eval_seconds = sum(float(record.get("evalDurationMs") or 0) for record in records) / 1000
    baseline_distances = [
        float(record["baselineTargetDistance"])
        for record in semantic
        if record.get("baselineTargetDistance") is not None
    ]
    output_distances = [
        float(record["outputTargetDistance"])
        for record in semantic
        if record.get("outputTargetDistance") is not None
    ]
    cjk_all_parsed_values = [
        float(record["cjkRetention"])
        for record in records
        if record.get("cjkRetention") is not None
    ]
    cjk_contract_valid_values = [
        float(record["cjkRetention"])
        for record in valid
        if record.get("cjkRetention") is not None
    ]
    edit_precisions = [
        float(record["editSpanPrecision"])
        for record in semantic
        if record.get("editSpanPrecision") is not None
    ]
    edit_recalls = [
        float(record["editSpanRecall"])
        for record in semantic
        if record.get("editSpanRecall") is not None
    ]
    edit_f1s = [
        float(record["editSpanF1"])
        for record in semantic
        if record.get("editSpanF1") is not None
    ]
    return {
        "sampleCount": count,
        "semanticSampleCount": len(semantic),
        "firstPassJsonValidityRate": _rate(records, "firstPassJsonValid"),
        "finalJsonValidityRate": _rate(records, "jsonValid"),
        "finalContractValidityRate": _rate(records, "contractValid"),
        "contractRejectionRate": _rate(records, "contractRejected"),
        "safetyValidityRate": _rate(records, "safetyValid"),
        "safetyViolationRate": _rate(records, "safetyViolation"),
        "retryRate": _rate(records, "retried"),
        "runtimeFailureRate": sum(bool(record.get("runtimeError")) for record in records) / count,
        "unsafeTextModificationRate": _rate(records, "unsafeTextModification"),
        "forbiddenCapabilityAttemptRate": _rate(records, "forbiddenCapabilityAttempt"),
        "forbiddenSpeakerOrSegmentationFieldAvoidanceRate": 1.0
        - _rate(records, "forbiddenCapabilityAttempt"),
        "speakerImmutability": {
            "status": "not_applicable_by_locked_contract",
            "value": None,
        },
        "protectedTokenPreservationRate": _rate(records, "protectedTokensPreserved"),
        "meanCjkRetentionAllParsedOutputs": _mean(cjk_all_parsed_values),
        "meanCjkRetentionContractValidOnly": _mean(cjk_contract_valid_values),
        "exactTargetMatchRate": _rate(semantic, "exactTargetMatch"),
        "improvedAgainstTargetRate": _rate(semantic, "improvedAgainstTarget"),
        "tiedAgainstTargetRate": _rate(semantic, "tiedAgainstTarget"),
        "worsenedAgainstTargetRate": _rate(semantic, "worsenedAgainstTarget"),
        "meanBaselineTargetDistance": _mean(baseline_distances),
        "meanOutputTargetDistance": _mean(output_distances),
        "meanEditSpanPrecision": _mean(edit_precisions),
        "meanEditSpanRecall": _mean(edit_recalls),
        "meanEditSpanF1": _mean(edit_f1s),
        "autoApplyCandidateCount": len(auto),
        "autoApplyRegressionRate": (
            sum(bool(record.get("worsenedAgainstTarget")) for record in auto) / len(auto)
            if auto
            else 0.0
        ),
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


def _empty_aggregate() -> dict[str, Any]:
    return {
        "sampleCount": 0,
        "semanticSampleCount": 0,
        "firstPassJsonValidityRate": 0.0,
        "finalJsonValidityRate": 0.0,
        "finalContractValidityRate": 0.0,
        "contractRejectionRate": 0.0,
        "safetyValidityRate": 0.0,
        "safetyViolationRate": 0.0,
        "retryRate": 0.0,
        "runtimeFailureRate": 0.0,
        "unsafeTextModificationRate": 0.0,
        "forbiddenCapabilityAttemptRate": 0.0,
        "forbiddenSpeakerOrSegmentationFieldAvoidanceRate": 0.0,
        "speakerImmutability": {
            "status": "not_applicable_by_locked_contract",
            "value": None,
        },
        "protectedTokenPreservationRate": 0.0,
        "meanCjkRetentionAllParsedOutputs": None,
        "meanCjkRetentionContractValidOnly": None,
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


def _binary_metrics(
    records: Iterable[dict[str, Any]],
    *,
    positive,
    predicted,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for record in records:
        expected_value = bool(positive(record))
        predicted_value = bool(predicted(record))
        if expected_value and predicted_value:
            tp += 1
        elif not expected_value and predicted_value:
            fp += 1
        elif expected_value and not predicted_value:
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


def _rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return _round(sum(bool(record.get(key)) for record in records) / len(records))


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
    value = ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
    return _round(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def _rejection_reasons(metrics: dict[str, Any]) -> list[str]:
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
