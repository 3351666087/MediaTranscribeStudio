from __future__ import annotations

from local_llm_bench.metrics import aggregate_results, recommendation
from local_llm_bench.text_utils import (
    cjk_retention,
    normalized_distance,
    protected_tokens_preserved,
    span_overlap_score,
)


def test_text_metrics_are_deterministic():
    assert normalized_distance("需要确认", "需要确认。") == 0.2
    assert cjk_retention("需要确认", "需要确认。") == 1.0
    assert protected_tokens_preserved("使用 API 2.0", "使用 API 2.0。")
    assert not protected_tokens_preserved("使用 API 2.0", "使用接口。")
    assert span_overlap_score([(1, 3)], [(2, 4)]) == (0.5, 0.5, 0.5)


def test_recommendation_does_not_promote_weak_model():
    record = {
        "split": "heldout",
        "kind": "semantic_cleanup",
        "firstPassJsonValid": True,
        "jsonValid": True,
        "contractValid": True,
        "safetyValid": True,
        "retried": False,
        "runtimeError": None,
        "outOfBoundsModification": False,
        "forbiddenCapabilityAttempt": False,
        "protectedTokensPreserved": True,
        "cjkRetention": 1.0,
        "exactTargetMatch": False,
        "improvedAgainstTarget": False,
        "tiedAgainstTarget": False,
        "worsenedAgainstTarget": True,
        "baselineTargetDistance": 0.2,
        "outputTargetDistance": 0.3,
        "editSpanPrecision": 0.2,
        "editSpanRecall": 0.2,
        "editSpanF1": 0.2,
        "autoApplyCandidate": True,
        "latencyMs": 100.0,
        "evalCount": 10,
        "evalDurationMs": 1000.0,
        "decision": "normalized",
    }
    metrics = aggregate_results([record])
    assert recommendation(metrics)["tier"] != "auto_apply_low_risk_only"
