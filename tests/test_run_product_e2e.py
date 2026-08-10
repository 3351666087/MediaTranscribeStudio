from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools import run_product_e2e as e2e


ROOT = Path(__file__).resolve().parents[1]
AUDIO_RECIPE = ROOT / "configs" / "product-audio-e2e-output-recipe.v1.json"
VIDEO_RECIPE = ROOT / "configs" / "product-e2e-output-recipe.v1.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _recipe_hash(path: Path, media_kind: str) -> str:
    _, _, digest = e2e._canonical_recipe(path, media_kind=media_kind)
    return digest


def _customer_artifact(
    output: Path,
    *,
    name: str,
    delivery_mode: str,
    subtitle_format: str,
    artifact_type: str = "subtitle-sidecar",
    visual_qa_hash: str | None = None,
    source_sha: str,
) -> dict[str, object]:
    path = output / name
    path.write_bytes((name + "\n").encode("utf-8"))
    return {
        "artifactType": artifact_type,
        "path": str(path),
        "sizeBytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "subtitleFormat": subtitle_format,
        "deliveryMode": delivery_mode,
        "visualQaEvidenceSha256": visual_qa_hash,
        "sourceIntegrity": {"unchanged": True, "sourceSha256": source_sha},
        "publication": {
            "atomic": True,
            "noReplace": True,
            "sourceMediaImmutable": True,
        },
    }


def _completed_fixture(
    tmp_path: Path,
    *,
    media_kind: str,
    job_id: str = "fixture-product-job",
) -> dict[str, Path | str | dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source-media.bin"
    source.write_bytes(b"immutable source media")
    source_sha = sha256_file(source)
    output = tmp_path / "output"
    output.mkdir()

    semantic_path = output / "semantic" / "semantic-composition.v1.json"
    _write_json(semantic_path, {"status": "composition-complete"})
    final_path = output / "final-adjudicated-transcript.v1.json"
    _write_json(final_path, {"jobId": job_id, "segments": []})
    transcript_json = output / "transcript.json"
    transcript_txt = output / "transcript.txt"
    _write_json(transcript_json, {"jobId": job_id, "segments": []})
    transcript_txt.write_text("fixture transcript\n", encoding="utf-8")
    quality_path = output / "artifacts" / "quality-report.json"
    _write_json(quality_path, {"status": "passed", "score": 100})
    (output / "render").mkdir()
    (output / "render" / "report.pdf").write_bytes(b"%PDF-1.7 fixture\n")

    customer = [
        _customer_artifact(
            output,
            name="captions.srt",
            delivery_mode="sidecar",
            subtitle_format="srt",
            source_sha=source_sha,
        ),
        _customer_artifact(
            output,
            name="captions.vtt",
            delivery_mode="sidecar",
            subtitle_format="webvtt",
            source_sha=source_sha,
        ),
        _customer_artifact(
            output,
            name="captions.ass",
            delivery_mode="sidecar",
            subtitle_format="ass",
            source_sha=source_sha,
        ),
    ]
    if media_kind == "video":
        visual_hash = "a" * 64
        customer.extend(
            [
                _customer_artifact(
                    output,
                    name="soft-mux.mp4",
                    delivery_mode="soft-mux",
                    subtitle_format="embedded",
                    artifact_type="subtitled-media",
                    visual_qa_hash=visual_hash,
                    source_sha=source_sha,
                ),
                _customer_artifact(
                    output,
                    name="burn-in.mp4",
                    delivery_mode="burn-in",
                    subtitle_format="burn-in",
                    artifact_type="subtitled-media",
                    visual_qa_hash="b" * 64,
                    source_sha=source_sha,
                ),
            ]
        )
    manifest_body: dict[str, object] = {
        "schemaVersion": "1.0.0",
        "artifactType": "output-publication-manifest",
        "source": {"path": str(source), "sha256": source_sha, "unchanged": True},
        "customerArtifacts": customer,
        "transaction": {
            "allConflictsCheckedBeforeWrites": True,
            "allMediaQaPassedBeforePublication": True,
            "rollbackSupported": True,
            "sourceMediaImmutable": True,
        },
    }
    manifest_body["manifestSha256"] = canonical_json_sha256(manifest_body)
    manifest_path = output / "output-publication-manifest.v1.json"
    _write_json(manifest_path, manifest_body)
    manifest_hash = str(manifest_body["manifestSha256"])

    exports = [
        {"format": "json", "path": str(transcript_json), "sizeBytes": transcript_json.stat().st_size, "sha256": sha256_file(transcript_json)},
        {"format": "txt", "path": str(transcript_txt), "sizeBytes": transcript_txt.stat().st_size, "sha256": sha256_file(transcript_txt)},
    ]
    checkpoint = {
        "status": "completed",
        "stage": "completed",
        "outputCustomization": {"sha256": _recipe_hash(VIDEO_RECIPE if media_kind == "video" else AUDIO_RECIPE, media_kind)},
        "semantic": {"status": "completed", "artifactPath": str(semantic_path), "artifactPaths": []},
        "transcriptExports": exports,
        "outputPublication": {"status": "published", "manifestPath": str(manifest_path), "manifestSha256": manifest_hash},
        "qualityReportPath": str(quality_path),
        "qualityStatus": "passed",
    }
    _write_json(output / "checkpoint.v2.json", checkpoint)
    result_path = tmp_path / "fixture-result.json"
    _write_json(
        result_path,
        {
            "status": "observed",
            "terminal_type": "job.completed",
            "job_id": job_id,
            "exit_code": 0,
            "shutdown_acknowledged": True,
            "forced_cleanup_pids": [],
            "error": None,
        },
    )
    return {
        "source": source,
        "sourceSha": source_sha,
        "output": output,
        "result": result_path,
        "manifest": manifest_path,
        "jobId": job_id,
        "manifestBody": manifest_body,
    }


def _rebind_manifest(fixture: dict[str, object]) -> None:
    path = Path(fixture["manifest"])
    # Mutating ``manifestBody`` is how the negative fixtures express stale or
    # invalid receipts; persist that exact object before recomputing its hash.
    body = fixture["manifestBody"]
    assert isinstance(body, dict)
    body.pop("manifestSha256", None)
    body["manifestSha256"] = canonical_json_sha256(body)
    _write_json(path, body)
    fixture["manifestBody"] = body
    checkpoint_path = Path(fixture["output"]) / "checkpoint.v2.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["outputPublication"]["manifestSha256"] = body["manifestSha256"]
    _write_json(checkpoint_path, checkpoint)


def test_classify_probe_ignores_attached_artwork() -> None:
    assert e2e.classify_probe_payload(
        {
            "streams": [
                {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 1}},
                {"index": 1, "codec_type": "audio"},
            ]
        }
    ) == {"kind": "audio", "audioStreamIndexes": [1], "videoStreamIndexes": []}
    assert e2e.classify_probe_payload(
        {
            "streams": [
                {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 0}},
                {"index": 1, "codec_type": "audio"},
            ]
        }
    )["kind"] == "video"


def test_shipped_recipe_contracts_are_exact() -> None:
    audio, _, _ = e2e._canonical_recipe(AUDIO_RECIPE, media_kind="audio")
    video, _, _ = e2e._canonical_recipe(VIDEO_RECIPE, media_kind="video")
    assert audio["delivery"]["subtitleModes"] == ["sidecar"]
    assert video["delivery"]["subtitleModes"] == ["sidecar", "soft-mux", "burn-in"]
    assert set(audio["delivery"]["formats"]) == e2e.REQUIRED_FORMATS
    assert set(video["delivery"]["formats"]) == e2e.REQUIRED_FORMATS


def test_worker_command_hybrid_argv_has_one_bound_each() -> None:
    command = e2e.build_worker_command(
        config=Path("config.json"),
        source=Path("source.wav"),
        output=Path("output"),
        recipe=Path("recipe.json"),
        job_id="job-1",
        mode="hybrid",
        speaker_count=None,
        speaker_count_min=2,
        speaker_count_max=5,
        speaker_count_prior=3,
        language="auto",
        title="title",
        local_llm_mode="suggestion-only",
        local_llm_model="qwen3.5:27b-q4_K_M",
        review_decisions=None,
        idle_timeout_seconds=10,
        hard_timeout_seconds=20,
    )
    assert list(command).count("--speaker-count-min") == 1
    assert list(command).count("--speaker-count-max") == 1
    assert command[command.index("--speaker-count-min") + 1] == "2"
    assert command[command.index("--speaker-count-max") + 1] == "5"
    assert command[command.index("--speaker-count-prior") + 1] == "3"


def test_cli_contract_rejects_invalid_timeout_and_incompatible_speakers() -> None:
    args = e2e.build_parser().parse_args(
        ["--source", "source.wav", "--output-dir", "out", "--mode", "auto", "--speaker-count", "1"]
    )
    with pytest.raises(e2e.ProductE2EError, match="auto mode"):
        e2e._validate_cli_contract(args)
    args = e2e.build_parser().parse_args(
        ["--source", "source.wav", "--output-dir", "out", "--idle-timeout-seconds", "0"]
    )
    with pytest.raises(e2e.ProductE2EError, match="greater than zero"):
        e2e._validate_cli_contract(args)


def test_validate_completed_audio_fixture_accepts_sidecars_only(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, media_kind="audio")
    value = e2e.validate_completed_run(
        output=Path(fixture["output"]),
        result_path=Path(fixture["result"]),
        expected_job_id=str(fixture["jobId"]),
        media_kind="audio",
        expected_recipe_hash=_recipe_hash(AUDIO_RECIPE, "audio"),
        source=Path(fixture["source"]),
        source_sha256_before=str(fixture["sourceSha"]),
    )
    assert len(value["customerArtifacts"]) == 3


def test_completed_run_rejects_recipe_binding_mismatch(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, media_kind="audio")
    checkpoint_path = Path(fixture["output"]) / "checkpoint.v2.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["outputCustomization"]["sha256"] = "0" * 64
    _write_json(checkpoint_path, checkpoint)

    with pytest.raises(e2e.ProductE2EError, match="recipe hash"):
        e2e.validate_completed_run(
            output=Path(fixture["output"]),
            result_path=Path(fixture["result"]),
            expected_job_id=str(fixture["jobId"]),
            media_kind="audio",
            expected_recipe_hash=_recipe_hash(AUDIO_RECIPE, "audio"),
            source=Path(fixture["source"]),
            source_sha256_before=str(fixture["sourceSha"]),
        )


def test_customer_artifact_cannot_escape_output_directory(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, media_kind="audio")
    outside = tmp_path / "outside.srt"
    outside.write_text("outside\n", encoding="utf-8")
    manifest = fixture["manifestBody"]
    artifact = manifest["customerArtifacts"][0]
    artifact["path"] = str(outside)
    artifact["sizeBytes"] = outside.stat().st_size
    artifact["sha256"] = sha256_file(outside)
    _rebind_manifest(fixture)

    with pytest.raises(e2e.ProductE2EError, match="escapes the output directory"):
        e2e.validate_completed_run(
            output=Path(fixture["output"]),
            result_path=Path(fixture["result"]),
            expected_job_id=str(fixture["jobId"]),
            media_kind="audio",
            expected_recipe_hash=_recipe_hash(AUDIO_RECIPE, "audio"),
            source=Path(fixture["source"]),
            source_sha256_before=str(fixture["sourceSha"]),
        )


def test_validate_completed_video_requires_visual_qa_and_all_modes(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, media_kind="video")
    value = e2e.validate_completed_run(
        output=Path(fixture["output"]),
        result_path=Path(fixture["result"]),
        expected_job_id=str(fixture["jobId"]),
        media_kind="video",
        expected_recipe_hash=_recipe_hash(VIDEO_RECIPE, "video"),
        source=Path(fixture["source"]),
        source_sha256_before=str(fixture["sourceSha"]),
    )
    assert {item["deliveryMode"] for item in value["customerArtifacts"]} == {"sidecar", "soft-mux", "burn-in"}

    manifest = fixture["manifestBody"]
    video = next(item for item in manifest["customerArtifacts"] if item["deliveryMode"] == "burn-in")
    video["visualQaEvidenceSha256"] = None
    _rebind_manifest(fixture)
    with pytest.raises(e2e.ProductE2EError, match="visual QA"):
        e2e.validate_completed_run(
            output=Path(fixture["output"]),
            result_path=Path(fixture["result"]),
            expected_job_id=str(fixture["jobId"]),
            media_kind="video",
            expected_recipe_hash=_recipe_hash(VIDEO_RECIPE, "video"),
            source=Path(fixture["source"]),
            source_sha256_before=str(fixture["sourceSha"]),
        )


def test_audio_rejects_video_delivery_and_stale_artifact_receipt(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, media_kind="audio")
    manifest = fixture["manifestBody"]
    extra = _customer_artifact(
        Path(fixture["output"]),
        name="unexpected.mp4",
        delivery_mode="soft-mux",
        subtitle_format="embedded",
        artifact_type="subtitled-media",
        visual_qa_hash="c" * 64,
        source_sha=str(fixture["sourceSha"]),
    )
    manifest["customerArtifacts"].append(extra)
    _rebind_manifest(fixture)
    with pytest.raises(e2e.ProductE2EError, match="delivery modes"):
        e2e.validate_completed_run(
            output=Path(fixture["output"]),
            result_path=Path(fixture["result"]),
            expected_job_id=str(fixture["jobId"]),
            media_kind="audio",
            expected_recipe_hash=_recipe_hash(AUDIO_RECIPE, "audio"),
            source=Path(fixture["source"]),
            source_sha256_before=str(fixture["sourceSha"]),
        )

    fixture = _completed_fixture(tmp_path / "stale", media_kind="audio")
    manifest = fixture["manifestBody"]
    manifest["customerArtifacts"][0]["sha256"] = "0" * 64
    _rebind_manifest(fixture)
    with pytest.raises(e2e.ProductE2EError, match="receipt hash"):
        e2e.validate_completed_run(
            output=Path(fixture["output"]),
            result_path=Path(fixture["result"]),
            expected_job_id=str(fixture["jobId"]),
            media_kind="audio",
            expected_recipe_hash=_recipe_hash(AUDIO_RECIPE, "audio"),
            source=Path(fixture["source"]),
            source_sha256_before=str(fixture["sourceSha"]),
        )


def test_receipt_only_does_not_probe_and_publishes_no_replace_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _completed_fixture(tmp_path, media_kind="audio", job_id="receipt-only-job")
    config = tmp_path / "production.config.json"
    _write_json(config, {"speaker": {"localLlmModel": "qwen3.5:27b-q4_K_M"}})
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(e2e, "probe_media_kind", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("probe must not run")))
    assert e2e.run(
        [
            "--config", str(config),
            "--source", str(fixture["source"]),
            "--output-dir", str(fixture["output"]),
            "--media-kind", "audio",
            "--job-id", "receipt-only-job",
            "--receipt-only",
            "--existing-result", str(fixture["result"]),
            "--receipt", str(receipt),
        ]
    ) == 0
    first = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        e2e.run(
            [
                "--config", str(config),
                "--source", str(fixture["source"]),
                "--output-dir", str(fixture["output"]),
                "--media-kind", "audio",
                "--job-id", "receipt-only-job",
                "--receipt-only",
                "--existing-result", str(fixture["result"]),
                "--receipt", str(receipt),
            ]
        )
    assert receipt.read_bytes() == first


def test_receipt_only_requires_explicit_media_kind() -> None:
    args = e2e.build_parser().parse_args(
        ["--source", "source.wav", "--output-dir", "out", "--receipt-only", "--existing-result", "result.json"]
    )
    with pytest.raises(e2e.ProductE2EError, match="explicit --media-kind"):
        e2e._validate_cli_contract(args)


def test_failed_receipt_does_not_claim_atomic_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    receipt = e2e.build_receipt(
        command=("worker",),
        job_id="failed-job",
        source=source,
        source_sha256_before=sha256_file(source),
        media_selection={"kind": "audio"},
        recipe_path=AUDIO_RECIPE,
        recipe_file_sha256=sha256_file(AUDIO_RECIPE),
        recipe_canonical_sha256=_recipe_hash(AUDIO_RECIPE, "audio"),
        result_path=tmp_path / "missing-result.json",
        worker_returncode=2,
        validation=None,
        error="worker failed",
    )
    assert receipt["status"] == "failed"
    assert receipt["safety"]["atomicPublication"] is False
