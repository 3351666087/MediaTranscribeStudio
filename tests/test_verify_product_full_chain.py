from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.verify_product_full_chain import (
    ProductFullChainVerificationError,
    verify_product_full_chain,
)


RECIPE_SHA256 = "a" * 64
MODEL = "qwen3.5:27b-q4_K_M"
JOB_ID = "sample-held-out-case"


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _receipt(path: Path, *, subtitle_format: str, source_sha256: str) -> dict:
    return {
        "artifactType": "subtitle-sidecar",
        "path": str(path.resolve()),
        "sizeBytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "subtitleFormat": subtitle_format,
        "deliveryMode": "sidecar",
        "visualQaEvidenceSha256": None,
        "sourceIntegrity": {
            "unchanged": True,
            "sourceSha256": source_sha256,
        },
        "publication": {
            "atomic": True,
            "noReplace": True,
            "sourceMediaImmutable": True,
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    source = _write_bytes(tmp_path / "media" / "sample.wav", b"source-media")
    source_sha = sha256_file(source)
    output = tmp_path / "outputs" / "held-out-case"
    output.mkdir(parents=True)

    lattice = _write_json(output / "semantic" / "lattice.json", {"groups": []})
    arbitration = _write_json(
        output / "semantic" / "arbitration.json", {"decisions": []}
    )
    composition = _write_json(
        output / "semantic" / "composition.json",
        {
            "schemaVersion": "1.0.0",
            "artifactType": "semantic-composition",
            "jobId": JOB_ID,
            "status": "composition-complete",
        },
    )

    review_queue = _write_json(
        output / "review" / "review-queue.json",
        {
            "schemaVersion": "2.0.0",
            "jobId": JOB_ID,
            "items": [
                {
                    "id": "item-1",
                    "status": "accepted",
                    "decision": {"decisionId": "decision-1"},
                }
            ],
            "decisions": [
                {
                    "decisionId": "decision-1",
                    "item_id": "item-1",
                    "audit": {
                        "actor": "codex-semantic-adjudicator",
                        "source": "codex-agent",
                    },
                }
            ],
            "openCount": 0,
        },
    )

    transcript_json = _write_json(output / "sample-transcript.json", {"segments": []})
    transcript_txt = _write_bytes(output / "sample-transcript.txt", b"speaker-1: text\n")
    sidecars = {
        "srt": _write_bytes(
            output / "sample-subtitle.srt",
            b"1\n00:00:00,000 --> 00:00:01,000\ntext\n",
        ),
        "webvtt": _write_bytes(
            output / "sample-subtitle.vtt",
            b"WEBVTT\n\n00:00.000 --> 00:01.000\ntext\n",
        ),
        "ass": _write_bytes(output / "sample-subtitle.ass", b"[Script Info]\nTitle: sample\n"),
    }

    publication_body = {
        "schemaVersion": "1.0.0",
        "status": "published",
        "recipeSha256": RECIPE_SHA256,
        "source": {
            "path": str(source.resolve()),
            "sizeBytes": source.stat().st_size,
            "sha256": source_sha,
            "unchanged": True,
        },
        "plans": [
            {
                "deliveryMode": "sidecar",
                "customizationSha256": "b" * 64,
                "executionPlanSha256": "c" * 64,
            }
        ],
        "customerArtifacts": [
            _receipt(path, subtitle_format=subtitle_format, source_sha256=source_sha)
            for subtitle_format, path in sidecars.items()
        ],
        "internalEvidence": {},
        "transaction": {
            "allConflictsCheckedBeforeWrites": True,
            "mediaQuarantinedBeforePublication": True,
            "allMediaQaPassedBeforePublication": True,
            "rollbackSupported": True,
            "privatePathsExcluded": True,
            "sourceMediaImmutable": True,
        },
    }
    publication = _write_json(
        output / "output-publication-manifest.v1.json",
        {
            **publication_body,
            "manifestSha256": canonical_json_sha256(publication_body),
        },
    )

    pdf = _write_bytes(output / "render" / "report.pdf", b"%PDF-1.4\n%%EOF\n")
    quality = _write_json(
        output / "artifacts" / "quality-report.json",
        {
            "schemaVersion": "1.0.0",
            "status": "passed",
            "minimumScore": 85.0,
            "score": 98.0,
            "hardGatesPassed": True,
            "hardGates": [{"id": "PDF-OPENABLE", "status": "passed"}],
            "evidence": [
                {
                    "id": "evidence-pdf",
                    "type": "pdf",
                    "relativePath": "render/report.pdf",
                    "sha256": sha256_file(pdf),
                    "verified": True,
                }
            ],
            "repairQueue": [],
            "regressions": [],
        },
    )
    render_manifest = _write_json(
        output / "artifacts" / "manifest.json",
        {
            "schemaVersion": "1.0.0",
            "jobId": JOB_ID,
            "artifacts": [
                {
                    "artifactId": "pdf",
                    "type": "pdf",
                    "relativePath": "render/report.pdf",
                    "mimeType": "application/pdf",
                    "sha256": sha256_file(pdf),
                    "bytes": pdf.stat().st_size,
                    "verified": True,
                }
            ],
        },
    )
    final = _write_json(
        output / "final-adjudicated-transcript.v1.json",
        {
            "schemaVersion": "1.2.0",
            "artifactType": "final-adjudicated-transcript",
            "jobId": JOB_ID,
            "status": "adjudication-complete",
            "semantic": {"status": "composition-complete", "model": MODEL},
            "review": {"openCount": 0},
        },
    )

    checkpoint = _write_json(
        output / "checkpoint.v2.json",
        {
            "schemaVersion": "2.0.0",
            "jobId": JOB_ID,
            "status": "completed",
            "stage": "completed",
            "error": None,
            "outputCustomization": {
                "sha256": RECIPE_SHA256,
                "recipe": {
                    "schemaVersion": "1.0.0",
                    "delivery": {
                        "formats": ["pdf", "txt", "json", "srt", "webvtt", "ass"]
                    },
                },
            },
            "semantic": {
                "required": True,
                "status": "completed",
                "mode": "candidate-composition",
                "artifactPath": str(composition.resolve()),
                "artifactPaths": [
                    str(lattice.resolve()),
                    str(arbitration.resolve()),
                    str(composition.resolve()),
                ],
                "provenance": {
                    "model": MODEL,
                    "promptVersion": "semantic-v1",
                    "roundCount": 1,
                },
                "error": None,
                "autoApply": True,
            },
            "qualityStatus": "passed",
            "qualityReportPath": str(quality.resolve()),
            "renderManifestPath": str(render_manifest.resolve()),
            "reviewQueuePath": str(review_queue.resolve()),
            "reviewOpenCount": 0,
            "outputManifestPath": str(publication.resolve()),
            "outputPublication": {
                "status": "published",
                "manifestPath": str(publication.resolve()),
                "manifestSha256": publication_body
                and canonical_json_sha256(publication_body),
                "manifestFileSha256": sha256_file(publication),
                "customerArtifacts": publication_body["customerArtifacts"],
                "error": None,
            },
            "transcriptExports": [
                {
                    "format": "json",
                    "path": str(transcript_json.resolve()),
                    "sha256": sha256_file(transcript_json),
                    "size": transcript_json.stat().st_size,
                },
                {
                    "format": "txt",
                    "path": str(transcript_txt.resolve()),
                    "sha256": sha256_file(transcript_txt),
                    "size": transcript_txt.stat().st_size,
                },
            ],
        },
    )

    result = _write_json(
        tmp_path / "results" / "held-out-case-result.json",
        {
            "status": "observed",
            "terminal_type": "job.completed",
            "job_id": JOB_ID,
            "exit_code": 0,
            "shutdown_acknowledged": True,
            "forced_cleanup_pids": [],
            "terminal_event": {
                "schemaVersion": "1.0.0",
                "jobId": JOB_ID,
                "type": "job.completed",
                "payload": {"status": "completed", "operation": "resume"},
            },
            "error": None,
        },
    )
    return output, result, {
        "checkpoint": checkpoint,
        "review": review_queue,
        "subtitle": sidecars["srt"],
        "final": final,
    }


def _verify(output: Path, result: Path) -> dict[str, Any]:
    return verify_product_full_chain(
        case_output=output,
        result_json=result,
        expected_job_id=JOB_ID,
        expected_model=MODEL,
        expected_recipe_sha256=RECIPE_SHA256,
        require_review_resume=True,
    )


def test_verifies_terminal_reviewed_semantic_and_delivery_chain(tmp_path: Path) -> None:
    output, result, _ = _fixture(tmp_path)

    report = _verify(output, result)

    assert report["status"] == "passed"
    assert report["terminalType"] == "job.completed"
    assert report["review"] == {
        "itemCount": 1,
        "acceptedCount": 1,
        "rejectedCount": 0,
        "decisionCount": 1,
        "preReviewDecisionCount": 0,
    }
    assert report["semantic"]["model"] == MODEL
    assert report["transcriptExports"]["formats"] == ["json", "txt"]
    assert report["publication"]["subtitleFormats"] == ["ass", "srt", "webvtt"]
    assert report["pdf"]["status"] == "passed"
    assert report["verifiedFileCount"] >= 14


def test_rejects_customer_subtitle_tampering(tmp_path: Path) -> None:
    output, result, paths = _fixture(tmp_path)
    paths["subtitle"].write_bytes(b"tampered")

    with pytest.raises(
        ProductFullChainVerificationError,
        match="subtitle srt SHA-256 does not match",
    ):
        _verify(output, result)


def test_rejects_non_manual_review_authority(tmp_path: Path) -> None:
    output, result, paths = _fixture(tmp_path)
    queue = json.loads(paths["review"].read_text(encoding="utf-8"))
    queue["decisions"][0]["audit"]["source"] = "model-self-review"
    _write_json(paths["review"], queue)

    with pytest.raises(
        ProductFullChainVerificationError,
        match="manual authority",
    ):
        _verify(output, result)


def test_rejects_completion_without_review_resume(tmp_path: Path) -> None:
    output, result, _ = _fixture(tmp_path)
    value = json.loads(result.read_text(encoding="utf-8"))
    value["terminal_event"]["payload"]["operation"] = "start"
    _write_json(result, value)

    with pytest.raises(
        ProductFullChainVerificationError,
        match="manual review was resumed",
    ):
        _verify(output, result)
