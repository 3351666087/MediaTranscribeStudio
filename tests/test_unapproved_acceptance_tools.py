from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.business_processing import BUSINESS_PROMPT_VERSION, BusinessProcessingConfig
from backend.persistence import canonical_json_sha256, sha256_file
from tools.export_sample_subtitles import export_sample_subtitles
from tools.run_unapproved_business_acceptance import validate_business_artifacts


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _transcript() -> dict[str, object]:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "doc-preview-test",
        "jobId": "job-preview-test",
        "language": "en",
        "source": {"fileName": "sample.wav", "sha256": "0" * 64, "durationMs": 4000},
        "speakers": [
            {"id": "speaker-1", "displayName": "Host"},
            {"id": "speaker-2", "displayName": "Guest"},
        ],
        "segments": [
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1500,
                "speakerId": "speaker-1",
                "rawText": "Hello world.",
            },
            {
                "id": "segment-2",
                "startMs": 1700,
                "endMs": 3200,
                "speakerId": "speaker-2",
                "rawText": "Testing subtitles.",
            },
        ],
    }


def _policy() -> dict[str, object]:
    return {
        "schemaVersion": "1.1.0",
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
    }


def test_subtitle_preview_manifest_binds_hashes_and_review_boundary(tmp_path: Path) -> None:
    transcript_path = _write_json(tmp_path / "transcript.json", _transcript())
    review_path = _write_json(
        tmp_path / "review.json",
        {
            "jobId": "job-preview-test",
            "items": [
                {"id": "review-1", "status": "open"},
                {"id": "review-2", "status": "resolved"},
            ],
        },
    )

    paths = export_sample_subtitles(
        transcript_path,
        tmp_path / "subtitles",
        include_speaker_labels=True,
        review_queue_path=review_path,
        expected_transcript_sha256=sha256_file(transcript_path),
        expected_review_queue_sha256=sha256_file(review_path),
        expected_open_review_items=1,
    )

    assert {path.suffix for path in paths[:-1]} == {".srt", ".vtt", ".ass"}
    manifest = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert manifest["applicationPolicy"] == "suggestion-only"
    assert manifest["requiresHumanApproval"] is True
    assert manifest["releaseApproved"] is False
    assert manifest["source"]["openReviewItems"] == 1
    assert manifest["cueQa"]["sourceTextPreserved"] is True
    assert manifest["cueQa"]["monotonicNonoverlapping"] is True
    assert manifest["cueQa"]["speakerLabelsIncluded"] is True
    assert len(manifest["artifacts"]) == 3
    assert all(item["utf8"] is True for item in manifest["artifacts"])


def test_subtitle_preview_refuses_silent_overwrite(tmp_path: Path) -> None:
    transcript_path = _write_json(tmp_path / "transcript.json", _transcript())
    output = tmp_path / "subtitles"
    export_sample_subtitles(transcript_path, output)

    with pytest.raises(RuntimeError, match="pass --replace"):
        export_sample_subtitles(transcript_path, output)


def test_business_acceptance_requires_policy_and_complete_variant_set(
    tmp_path: Path,
) -> None:
    document = _transcript()
    source_text = "Hello world."
    translation = {
        **_policy(),
        "variant": "translation:zh-CN",
        "inputHash": "1" * 64,
        "model": "qwen3.5:4b",
        "promptVersion": BUSINESS_PROMPT_VERSION,
        "provider": {
            "id": "ollama-loopback",
            "version": "native-json-v3",
            "networkPolicy": "loopback-only",
        },
        "temperature": 0,
        "status": "completed",
        "sourceLanguage": "en",
        "targetLanguage": "zh-CN",
        "segments": [
            {
                "id": "segment-1",
                "speakerId": "speaker-1",
                "startMs": 0,
                "endMs": 1500,
                "sourceTextHash": hashlib.sha256(source_text.encode()).hexdigest(),
                "text": "你好，世界。",
                "language": "zh-CN",
            },
            {
                "id": "segment-2",
                "speakerId": "speaker-2",
                "startMs": 1700,
                "endMs": 3200,
                "sourceTextHash": hashlib.sha256(
                    "Testing subtitles.".encode()
                ).hexdigest(),
                "text": "测试字幕。",
                "language": "zh-CN",
            },
        ],
    }
    translation_path = _write_json(
        tmp_path / "business" / "translation-zh-CN.v1.json", translation
    )
    manifest = {
        **_policy(),
        "documentId": document["documentId"],
        "sourceDocumentHash": canonical_json_sha256(document),
        "rawTranscriptImmutable": True,
        "artifacts": [str(translation_path)],
        "config": BusinessProcessingConfig(
            translation_targets=("zh-CN",)
        ).as_dict(),
        "completeness": {
            "translations": {
                "zh-CN": {
                    "total": 2,
                    "translated": 2,
                    "copied": 0,
                    "skipped": 0,
                    "failed": 0,
                    "pending": 0,
                    "completed": 2,
                    "complete": True,
                }
            },
            "allRequestedTasksCompleted": True,
        },
    }
    manifest_path = _write_json(
        tmp_path / "business" / "business-manifest.v1.json", manifest
    )

    result = validate_business_artifacts(
        document,
        (translation_path, manifest_path),
        config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
    )

    assert result["sourceDocumentCanonicalSha256"] == canonical_json_sha256(document)
    assert result["variants"]["translation:zh-CN"]["segmentCount"] == 2
    broken = json.loads(translation_path.read_text(encoding="utf-8"))
    broken["requiresHumanApproval"] = False
    _write_json(translation_path, broken)
    with pytest.raises(RuntimeError, match="human approval gate"):
        validate_business_artifacts(
            document,
            (translation_path, manifest_path),
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )
