from __future__ import annotations

import pytest

from backend.errors import WorkerError
from backend.review import resolve_review_item


def _document() -> dict:
    return {
        "segments": [
            {
                "id": "segment-0001",
                "speakerId": "speaker-1",
                "rawText": "source",
                "normalizedText": "source",
                "displayText": "source",
                "evidence": {},
            }
        ]
    }


def _queue(item_count: int) -> dict:
    return {
        "items": [
            {
                "id": f"review-{index}",
                "segmentId": "segment-0001",
                "status": "open",
            }
            for index in range(1, item_count + 1)
        ],
        "decisions": [],
    }


def _payload(
    item_id: str,
    decision_id: str,
    *,
    actor: str,
    reason: str,
    source: str = "human",
) -> dict:
    return {
        "itemId": item_id,
        "decisionId": decision_id,
        "reason": reason,
        "evidence": ["audio:0-1000"],
        "confidence": 0.99,
        "audit": {"actor": actor, "source": source},
    }


def test_segment_becomes_human_reviewed_only_after_its_last_open_item() -> None:
    document, queue, _ = resolve_review_item(
        _document(),
        _queue(2),
        _payload(
            "review-1",
            "decision-1",
            actor="reviewer-one",
            reason="first issue checked",
        ),
        command="review.submit",
        action="accepted",
    )

    assert "audioReview" not in document["segments"][0]["evidence"]

    document, queue, _ = resolve_review_item(
        document,
        queue,
        _payload(
            "review-2",
            "decision-2",
            actor="reviewer-two",
            reason="remaining issue checked",
        ),
        command="review.submit",
        action="rejected",
    )

    assert document["segments"][0]["evidence"]["audioReview"] == {
        "status": "human-reviewed",
        "reviewer": "reviewer-two",
        "notes": "remaining issue checked",
    }
    assert [item["status"] for item in queue["items"]] == [
        "accepted",
        "rejected",
    ]


@pytest.mark.parametrize("action", ["accepted", "rejected"])
def test_accept_and_reject_both_close_segment_review(action: str) -> None:
    document, queue, _ = resolve_review_item(
        _document(),
        _queue(1),
        _payload(
            "review-1",
            f"decision-{action}",
            actor="reviewer",
            reason=f"{action} after listening",
        ),
        command="review.submit",
        action=action,
    )

    assert queue["items"][0]["status"] == action
    assert document["segments"][0]["evidence"]["audioReview"]["status"] == (
        "human-reviewed"
    )


def test_manual_text_revision_preserves_actor_and_timestamp() -> None:
    payload = _payload(
        "review-1",
        "decision-text",
        actor="codex-semantic-adjudicator",
        reason="listening confirms the corrected spelling",
    )
    payload["normalizedText"] = "corrected source text"
    payload["displayText"] = "corrected source text"

    document, _, decision = resolve_review_item(
        _document(),
        _queue(1),
        payload,
        command="review.submit",
        action="accepted",
    )

    revision = document["segments"][0]["revisions"][0]
    assert revision["source"] == "manual"
    assert revision["actor"] == "codex-semantic-adjudicator"
    assert revision["occurredAt"] == decision["recordedAt"]


def test_codex_agent_is_preserved_as_manual_review_source() -> None:
    document, queue, decision = resolve_review_item(
        _document(),
        _queue(1),
        _payload(
            "review-1",
            "decision-codex-agent",
            actor="codex-semantic-adjudicator",
            reason="Codex reviewed the source and transcript evidence.",
            source="codex-agent",
        ),
        command="review.submit",
        action="accepted",
    )

    assert decision["audit"]["source"] == "codex-agent"
    assert queue["decisions"][0]["audit"]["source"] == "codex-agent"
    assert document["segments"][0]["evidence"]["audioReview"]["status"] == (
        "human-reviewed"
    )


@pytest.mark.parametrize("source", [None, "model-self-review", "codex-manual"])
def test_manual_review_rejects_unapproved_or_missing_source(
    source: str | None,
) -> None:
    payload = _payload(
        "review-1",
        "decision-invalid-source",
        actor="reviewer",
        reason="Source validation must fail closed.",
    )
    if source is None:
        del payload["audit"]["source"]
    else:
        payload["audit"]["source"] = source

    with pytest.raises(WorkerError, match="human or codex-agent"):
        resolve_review_item(
            _document(),
            _queue(1),
            payload,
            command="review.submit",
            action="accepted",
        )
