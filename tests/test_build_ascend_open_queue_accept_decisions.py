from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.build_ascend_open_queue_accept_decisions import (
    OpenQueueDecisionError,
    build_decisions,
)


def _write_queue(
    path: Path,
    *,
    speaker_id: str = "speaker-1",
    open_count: int = 2,
    decisions: list[dict[str, object]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "2.0.0",
                "jobId": "source-job",
                "speakerCountMode": "manual",
                "items": [
                    {
                        "id": "segment-1:SEGMENT_LOW_CONFIDENCE",
                        "status": "open",
                        "speakerId": speaker_id,
                        "text": {
                            "rawText": "visible raw text",
                            "normalizedText": "visible normalized text",
                            "displayText": "visible display text",
                        },
                    },
                    {
                        "id": "segment-2:SEGMENT_LOW_CONFIDENCE",
                        "status": "open",
                        "speakerId": speaker_id,
                        "text": {"rawText": "second visible item"},
                    },
                ],
                "decisions": [] if decisions is None else decisions,
                "openCount": open_count,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_decisions_binds_every_visible_open_item(tmp_path: Path) -> None:
    queue_path = tmp_path / "review-queue.json"
    output_path = tmp_path / "decisions.json"
    _write_queue(queue_path)

    receipt = build_decisions(
        queue_path=queue_path,
        output_path=output_path,
        job_id="new-job-r1",
        target_speaker_id="speaker-1",
        expected_open_count=2,
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["artifactType"] == "production-review-decisions"
    assert persisted["jobId"] == "new-job-r1"
    assert persisted["automaticScoring"] is False
    assert [item["itemId"] for item in persisted["decisions"]] == [
        "segment-1:SEGMENT_LOW_CONFIDENCE",
        "segment-2:SEGMENT_LOW_CONFIDENCE",
    ]
    assert all(item["action"] == "accept" for item in persisted["decisions"])
    assert all(
        item["targetSpeakerId"] == "speaker-1"
        for item in persisted["decisions"]
    )
    queue_binding = f"review-queue-file-sha256:{sha256_file(queue_path)}"
    assert all(
        queue_binding in item["evidence"] for item in persisted["decisions"]
    )
    assert receipt == {
        "path": str(output_path.resolve()),
        "fileSha256": sha256_file(output_path),
        "canonicalSha256": canonical_json_sha256(persisted),
        "jobId": "new-job-r1",
        "openCount": 2,
        "preReviewCommandCount": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"schemaVersion": "1.0.0"}), "schemaVersion"),
        (lambda value: value.update({"openCount": 1}), "initial open items"),
        (
            lambda value: value.update(
                {"decisions": [{"decisionId": "already-decided"}]}
            ),
            "already contains decisions",
        ),
        (
            lambda value: value["items"][0].update({"status": "accepted"}),
            "only open item objects",
        ),
    ],
)
def test_build_decisions_rejects_non_initial_open_queue(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    queue_path = tmp_path / "review-queue.json"
    _write_queue(queue_path)
    value = json.loads(queue_path.read_text(encoding="utf-8"))
    mutation(value)
    queue_path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(OpenQueueDecisionError, match=message):
        build_decisions(
            queue_path=queue_path,
            output_path=tmp_path / "decisions.json",
            job_id="new-job-r1",
            target_speaker_id="speaker-1",
        )


def test_build_decisions_rejects_speaker_or_count_rebinding(tmp_path: Path) -> None:
    queue_path = tmp_path / "review-queue.json"
    _write_queue(queue_path, speaker_id="speaker-2")

    with pytest.raises(OpenQueueDecisionError, match="not 'speaker-1'"):
        build_decisions(
            queue_path=queue_path,
            output_path=tmp_path / "speaker-decisions.json",
            job_id="new-job-r1",
            target_speaker_id="speaker-1",
        )

    with pytest.raises(OpenQueueDecisionError, match="expected 3 open items"):
        build_decisions(
            queue_path=queue_path,
            output_path=tmp_path / "count-decisions.json",
            job_id="new-job-r1",
            target_speaker_id="speaker-2",
            expected_open_count=3,
        )
