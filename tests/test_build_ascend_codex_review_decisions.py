from __future__ import annotations

from copy import deepcopy

import pytest

from tools.build_ascend_codex_review_decisions import (
    CodexDecisionBuildError,
    _build_case,
)


def _template() -> dict:
    queue_sha = "a" * 64
    return {
        "schemaVersion": "1.0.0",
        "artifactType": "ascend-held-out-manual-review-decision-template",
        "caseId": "ascend-held-test-01283",
        "jobId": "sample-ascend-held-test-01283",
        "automaticScoring": False,
        "executable": False,
        "binding": {"reviewQueue": {"fileSha256": queue_sha}},
        "visibleOutputEvidence": {
            "semanticCompositionCompleted": True,
            "speakerIds": ["speaker-1", "speaker-2"],
        },
        "suggestedPreReviewCommands": [
            {
                "type": "speaker.merge",
                "sourceSpeakerId": "speaker-2",
                "targetSpeakerId": "speaker-1",
                "decisionId": "merge-1",
            }
        ],
        "decisions": [
            {
                "itemId": "speaker-count-confidence",
                "scope": "job",
                "decisionId": "review-1",
                "timeRange": None,
                "targetSpeakerId": None,
            },
            {
                "itemId": "segment-1:SEGMENT_LOW_CONFIDENCE",
                "scope": "segment",
                "decisionId": "review-2",
                "timeRange": {"startMs": 0, "endMs": 1840},
                "targetSpeakerId": "speaker-1",
                "visibleText": {"rawText": "sign in 的话只需要你"},
            },
        ],
    }


def test_builds_strict_executable_manual_decisions() -> None:
    result = _build_case(_template())

    assert set(result) == {
        "schemaVersion",
        "artifactType",
        "jobId",
        "automaticScoring",
        "preReviewCommands",
        "decisions",
    }
    assert result["artifactType"] == "production-review-decisions"
    assert result["automaticScoring"] is False
    assert result["preReviewCommands"][0]["sourceSpeakerId"] == "speaker-2"
    assert result["preReviewCommands"][0]["audit"]["source"] == "codex-agent"
    assert result["decisions"][0]["action"] == "accept"
    assert "targetSpeakerId" not in result["decisions"][0]
    assert result["decisions"][1]["targetSpeakerId"] == "speaker-1"
    assert "visibleText" not in result["decisions"][1]
    assert "do not invent the missing object" in result["decisions"][1]["reason"]


def test_rejects_a_template_that_is_already_executable() -> None:
    template = _template()
    template["executable"] = True

    with pytest.raises(CodexDecisionBuildError, match="already executable"):
        _build_case(template)


def test_rejects_a_merge_policy_not_exposed_by_visible_template() -> None:
    template = deepcopy(_template())
    template["suggestedPreReviewCommands"] = []

    with pytest.raises(CodexDecisionBuildError, match="does not match visible"):
        _build_case(template)
