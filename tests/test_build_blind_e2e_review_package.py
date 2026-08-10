from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.authorize_held_out_unblind import (
    ARTIFACT_TYPE as UNBLIND_AUTHORIZATION_ARTIFACT_TYPE,
    HeldOutUnblindAuthorizationError,
    authorize_held_out_unblind,
)
import tools.build_blind_e2e_review_package as blind_builder
from tools.build_blind_e2e_review_package import (
    BlindE2EReviewPackageError,
    CandidateRun,
    _blind_safe_report_document,
    _ordering_key,
    build_review_package,
)


def _audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 1600)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _candidate_run(
    root: Path,
    *,
    label: str,
    source: Path,
    cases: tuple[str, ...] = ("secret-zh-case", "secret-en-case"),
    model_identity: str | None = None,
    no_speech: bool = False,
    include_pdf: bool = False,
) -> CandidateRun:
    run = root / label
    model = model_identity or label
    for case_index, case_id in enumerate(cases, start=1):
        output = run / "outputs" / case_id
        delivery = output / "delivery"
        declared_artifacts = [] if no_speech else [str(delivery / "report.srt")]
        if include_pdf and not no_speech:
            declared_artifacts.append(str(delivery / "report.pdf"))
        _write_json(
            output / "checkpoint.v2.json",
            {
                "schemaVersion": "2.0.0",
                "status": "completed",
                "sourcePath": str(source),
                "semantic": {"provenance": {"model": model}},
                "artifactPaths": declared_artifacts,
            },
        )
        if no_speech:
            _write_json(
                output / "final-adjudicated-transcript.v1.json",
                {
                    "schemaVersion": "1.1.0",
                    "disposition": "no-transcribable-speech",
                    "segments": [],
                },
            )
            delivery.mkdir()
            _write_json(
                run / "results" / f"{case_id}-result.json",
                {"status": "job-completed"},
            )
            continue
        segments = [
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1000,
                "speakerId": "speaker-1",
                "language": "zh-CN" if case_index == 1 else "en-US",
                "rawText": f"raw candidate {case_index}",
                "overlapping": False,
            }
        ]
        _write_json(
            output / "transcript-document.v2.json",
            {
                "schemaVersion": "2.0.0",
                "provenance": {"models": [{"name": model}]},
                "segments": segments,
            },
        )
        _write_json(
            output / "final-adjudicated-transcript.v1.json",
            {
                "schemaVersion": "1.2.0",
                "disposition": "transcribable-speech",
                "semantic": {"model": model},
                "timeline": {
                    "speakerCount": 1,
                    "speakerIds": ["speaker-1"],
                    "turns": [
                        {
                            "startMs": 0,
                            "endMs": 1000,
                            "speakerId": "speaker-1",
                            "overlap": False,
                        }
                    ],
                },
                "segments": [
                    {
                        **segments[0],
                        "finalText": f"final candidate {case_index}",
                    }
                ],
            },
        )
        delivery.mkdir()
        (delivery / "report.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nReview text\n",
            encoding="utf-8",
        )
        if include_pdf:
            (delivery / "report.pdf").write_bytes(
                b"%PDF-1.4\nfixture blind report\n%%EOF\n"
            )
        _write_json(
            run / "results" / f"{case_id}-result.json",
            {"status": "job-completed", "candidateModel": label},
        )
    return CandidateRun(label, run)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_blind_pdf_report_sanitization_changes_only_identity_labels() -> None:
    original = {
        "schemaVersion": "1.0.0",
        "documentId": "document-opaque-001",
        "title": "Sample secret-case-001",
        "source": {
            "fileName": "secret-case-001.wav",
            "mediaType": "audio/wav",
            "durationMs": 1000,
            "sha256": "a" * 64,
        },
        "segments": [
            {
                "id": "segment-001",
                "startMs": 0,
                "endMs": 1000,
                "speakerId": "speaker-1",
                "rawText": "immutable raw text",
                "normalizedText": "immutable raw text",
                "displayText": "immutable raw text",
                "evidence": {"speaker": {"locked": True}},
            }
        ],
        "speakerPolicy": {"resolvedCount": 1},
        "speakers": [{"id": "speaker-1"}],
    }
    frozen = json.loads(json.dumps(original))

    sanitized, projection_sha = _blind_safe_report_document(original)

    assert original == frozen
    assert sanitized["title"] == "Blind Review Transcript"
    assert sanitized["source"]["fileName"] == "source-media.wav"
    restored = json.loads(json.dumps(sanitized))
    restored["title"] = original["title"]
    restored["source"]["fileName"] = original["source"]["fileName"]
    assert restored == original
    assert projection_sha == canonical_json_sha256(
        {
            "documentId": "document-opaque-001",
            "segments": [
                {
                    "id": "segment-001",
                    "startMs": 0,
                    "endMs": 1000,
                    "speakerId": "speaker-1",
                    "rawText": "immutable raw text",
                    "normalizedText": "immutable raw text",
                    "displayText": "immutable raw text",
                }
            ],
        }
    )


def _package_with_completed_review(
    tmp_path: Path,
) -> tuple[Path, Path, dict]:
    source = _audio(tmp_path / "authorization-source.wav")
    candidates = (
        _candidate_run(
            tmp_path,
            label="authorization-candidate-a",
            source=source,
        ),
        _candidate_run(
            tmp_path,
            label="authorization-candidate-b",
            source=source,
        ),
    )
    built = build_review_package(
        candidates=candidates,
        output_root=tmp_path / "authorization-package",
        seed="authorization-seed",
    )
    package_root = Path(built["packageRoot"])
    template_path = (
        package_root / "reviewer-packet" / "human-review-form.v1.json"
    )
    completed = _load(template_path)
    dimension = {
        "severity": "pass",
        "reason": "Reviewed the complete candidate artifact.",
        "evidence": ["00:00.000-00:01.000"],
    }
    for case in completed["cases"]:
        case["preferredCandidateId"] = case["candidateOrder"][0]
        case["tie"] = False
        for field in (
            "speakerTimelineReview",
            "rawAsrReview",
            "finalTranscriptReview",
            "subtitlePdfReview",
        ):
            case[field] = dict(dimension)
        case["notes"] = ["No blocker or major defect observed."]
    completed_path = _write_json(
        tmp_path / "completed-review.v1.json",
        completed,
    )
    return package_root, completed_path, completed


def _authorize_completed_review(
    *,
    package_root: Path,
    completed_review: Path,
    output: Path,
    expected_package_manifest_sha256: str | None = None,
) -> dict:
    package_manifest = package_root / "package-manifest.v1.json"
    return authorize_held_out_unblind(
        package_root=package_root,
        completed_review_path=completed_review,
        held_out_freeze_manifest_sha256="e" * 64,
        expected_package_manifest_sha256=(
            expected_package_manifest_sha256 or sha256_file(package_manifest)
        ),
        reviewer_source="codex-agent",
        reviewer="Codex held-out reviewer",
        reviewed_at="2026-08-09T18:00:00+08:00",
        output_path=output,
    )


def test_build_review_package_is_blind_hash_bound_and_recomputable(
    tmp_path: Path,
) -> None:
    source = _audio(tmp_path / "source.wav")
    candidates = (
        _candidate_run(tmp_path, label="qwen-secret-model", source=source),
        _candidate_run(tmp_path, label="glm-secret-model", source=source),
    )

    first = build_review_package(
        candidates=candidates,
        output_root=tmp_path / "package-one",
        seed="sealed-seed-20260807",
    )
    second = build_review_package(
        candidates=candidates,
        output_root=tmp_path / "package-two",
        seed="sealed-seed-20260807",
    )
    reordered = build_review_package(
        candidates=tuple(reversed(candidates)),
        output_root=tmp_path / "package-reordered",
        seed="sealed-seed-20260807",
        case_ids=("secret-en-case", "secret-zh-case"),
    )

    first_review = Path(first["reviewManifest"])
    second_review = Path(second["reviewManifest"])
    reordered_review = Path(reordered["reviewManifest"])
    assert first_review.parent.name == "reviewer-packet"
    assert first["packageId"] == second["packageId"] == reordered["packageId"]
    assert first_review.read_bytes() == second_review.read_bytes()
    assert first_review.read_bytes() == reordered_review.read_bytes()
    blind_text = first_review.read_text(encoding="utf-8")
    assert "qwen-secret-model" not in blind_text
    assert "glm-secret-model" not in blind_text
    assert "secret-zh-case" not in blind_text
    assert "secret-en-case" not in blind_text
    manifest = json.loads(blind_text)
    assert manifest["blindnessPolicy"] == {
        "candidateIdentityPersisted": False,
        "originalPathPersisted": False,
        "referenceAnswerPersisted": False,
        "automaticScorePersisted": False,
        "referenceInputsAccepted": False,
    }
    assert manifest["counts"] == {"cases": 2, "candidatesPerCase": 2}
    assert manifest["candidateOrdering"]["seedSha256"] != "sealed-seed-20260807"
    assert all(
        [item["reviewCandidateId"] for item in case["candidates"]]
        == ["option-01", "option-02"]
        for case in manifest["cases"]
    )
    for case in manifest["cases"]:
        assert Path(first_review.parent / case["sourceAudio"]["path"]).is_file()
        for candidate in case["candidates"]:
            kinds = {item["kind"] for item in candidate["artifacts"]}
            assert kinds == {"candidate-evidence", "subtitle"}
            for artifact in candidate["artifacts"]:
                path = first_review.parent / artifact["path"]
                assert path.is_file()
                assert sha256_file(path) == artifact["sha256"]
    body = dict(manifest)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
    assert manifest["pdfIdentityScanning"] == {
        "requiredForDeclaredPdf": True,
        "declaredPdfCount": 0,
        "validator": "PDFBox",
        "validatorVersion": "2.0.30",
        "scannerJarSha256": None,
    }

    unblind_path = Path(first["identityMapping"])
    assert unblind_path.parent.name == "identity-vault"
    assert (
        sha256_file(unblind_path)
        == manifest["identityCommitment"]["fileSha256"]
    )
    unblind = _load(unblind_path)
    serialized_unblind = json.dumps(unblind)
    assert "qwen-secret-model" in serialized_unblind
    assert "glm-secret-model" in serialized_unblind
    assert unblind["ordering"]["seed"] == "sealed-seed-20260807"
    assert unblind["referenceInputsAccepted"] is False
    package_manifest = _load(Path(first["packageRoot"]) / "package-manifest.v1.json")
    assert package_manifest["reviewerPacket"]["relativePath"].startswith(
        "reviewer-packet/"
    )
    assert package_manifest["identityVault"]["relativePath"].startswith(
        "identity-vault/"
    )


def test_candidate_order_is_stable_and_case_specific() -> None:
    values = {
        case_id: sorted(
            ("candidate-a", "candidate-b", "candidate-c"),
            key=lambda candidate_id: _ordering_key(
                "seed",
                "candidate",
                case_id,
                candidate_id,
            ),
        )
        for case_id in ("case-a", "case-b")
    }

    assert values == {
        case_id: sorted(
            values[case_id],
            key=lambda candidate_id: _ordering_key(
                "seed",
                "candidate",
                case_id,
                candidate_id,
            ),
        )
        for case_id in values
    }
    assert values["case-a"] != values["case-b"]


def test_package_rejects_incomplete_or_source_rebound_candidates(
    tmp_path: Path,
) -> None:
    first_source = _audio(tmp_path / "first.wav")
    second_source = _audio(tmp_path / "second.wav")
    second_source.write_bytes(second_source.read_bytes() + b"different")
    first = _candidate_run(
        tmp_path,
        label="candidate-a",
        source=first_source,
        cases=("case-one",),
    )
    incomplete = _candidate_run(
        tmp_path,
        label="candidate-b",
        source=first_source,
        cases=("case-one",),
    )
    checkpoint = incomplete.run_root / "outputs/case-one/checkpoint.v2.json"
    value = _load(checkpoint)
    value["status"] = "failed"
    checkpoint.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(BlindE2EReviewPackageError, match="not completed"):
        build_review_package(
            candidates=(first, incomplete),
            output_root=tmp_path / "incomplete-package",
            seed="seed",
        )

    rebound = _candidate_run(
        tmp_path,
        label="candidate-c",
        source=second_source,
        cases=("case-one",),
    )
    with pytest.raises(BlindE2EReviewPackageError, match="different source audio"):
        build_review_package(
            candidates=(first, rebound),
            output_root=tmp_path / "rebound-package",
            seed="seed",
        )


def test_package_refuses_identity_leak_in_subtitle(tmp_path: Path) -> None:
    source = _audio(tmp_path / "source.wav")
    first = _candidate_run(
        tmp_path,
        label="candidate-alpha",
        source=source,
        cases=("case-one",),
    )
    second = _candidate_run(
        tmp_path,
        label="candidate-bravo",
        source=source,
        cases=("case-one",),
    )
    leaking = (
        first.run_root
        / "outputs/case-one/delivery/report.srt"
    )
    leaking.write_text("candidate-alpha", encoding="utf-8")

    with pytest.raises(BlindE2EReviewPackageError, match="blinded identity"):
        build_review_package(
            candidates=(first, second),
            output_root=tmp_path / "leaking-package",
            seed="seed",
        )


def test_package_refuses_detected_model_identity_in_utf16_subtitle(
    tmp_path: Path,
) -> None:
    source = _audio(tmp_path / "source.wav")
    hidden_model = "Hidden-Production-Model-42"
    first = _candidate_run(
        tmp_path,
        label="candidate-alpha",
        source=source,
        cases=("case-one",),
        model_identity=hidden_model,
    )
    second = _candidate_run(
        tmp_path,
        label="candidate-bravo",
        source=source,
        cases=("case-one",),
        model_identity="Other-Production-Model-84",
    )
    leaking = first.run_root / "outputs/case-one/delivery/report.srt"
    leaking.write_text(hidden_model, encoding="utf-16")

    with pytest.raises(BlindE2EReviewPackageError, match="blinded identity"):
        build_review_package(
            candidates=(first, second),
            output_root=tmp_path / "model-leaking-package",
            seed="seed",
        )


@pytest.mark.parametrize("reserved_role", ["production", "challenger"])
def test_package_refuses_reserved_candidate_role_in_reviewer_artifact(
    tmp_path: Path,
    reserved_role: str,
) -> None:
    source = _audio(tmp_path / "source.wav")
    first = _candidate_run(
        tmp_path,
        label="candidate-alpha",
        source=source,
        cases=("case-one",),
    )
    second = _candidate_run(
        tmp_path,
        label="candidate-bravo",
        source=source,
        cases=("case-one",),
    )
    leaking = first.run_root / "outputs/case-one/delivery/report.srt"
    leaking.write_text(reserved_role, encoding="utf-8")

    with pytest.raises(BlindE2EReviewPackageError, match="blinded identity"):
        build_review_package(
            candidates=(first, second),
            output_root=tmp_path / f"{reserved_role}-leaking-package",
            seed="seed",
        )


def test_declared_pdfs_are_scanned_through_anonymous_stdin_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _audio(tmp_path / "source.wav")
    candidates = (
        _candidate_run(
            tmp_path,
            label="candidate-alpha",
            source=source,
            cases=("case-one",),
            include_pdf=True,
        ),
        _candidate_run(
            tmp_path,
            label="candidate-bravo",
            source=source,
            cases=("case-one",),
            include_pdf=True,
        ),
    )
    scanner_jar = tmp_path / "pdfbox-scanner.jar"
    scanner_jar.write_bytes(b"frozen scanner fixture")
    calls: list[dict] = []

    def fake_scan_pdf_identity(**kwargs: object) -> dict:
        pdf_path = Path(str(kwargs["pdf_path"]))
        expected = str(kwargs["expected_pdf_sha256"])
        scanner = kwargs["scanner"]
        sensitive_values = tuple(kwargs["sensitive_values"])
        assert pdf_path.is_file()
        assert sha256_file(pdf_path) == expected
        assert "case-one" not in str(pdf_path)
        assert "candidate-alpha" not in str(pdf_path)
        assert "candidate-bravo" not in str(pdf_path)
        calls.append(
            {
                "pdfSha256": expected,
                "sensitiveValues": sensitive_values,
            }
        )
        return {
            "schemaVersion": "1.0.0",
            "artifactType": "blind-review-pdf-identity-scan",
            "validator": "PDFBox",
            "validatorVersion": "2.0.30",
            "status": "passed",
            "pdfSha256": expected,
            "scannerJarSha256": scanner.jar_sha256,
            "allRequiredSurfacesScanned": True,
            "sensitiveValueCount": len(sensitive_values),
            "pageCount": 1,
            "pageTextCount": 1,
            "documentInfoEntryCount": 0,
            "xmpPacketCount": 0,
            "attachmentCount": 0,
            "annotationCount": 0,
            "formFieldCount": 0,
            "scannedCosObjectCount": 1,
            "decodedStreamBytes": 0,
        }

    monkeypatch.setattr(
        blind_builder,
        "_scan_pdf_identity",
        fake_scan_pdf_identity,
    )
    built = build_review_package(
        candidates=candidates,
        output_root=tmp_path / "pdf-scanned-package",
        seed="seed",
        java_executable=sys.executable,
        pdf_scanner_jar=scanner_jar,
    )

    assert len(calls) == 2
    for call in calls:
        sensitive_values = set(call["sensitiveValues"])
        assert {"production", "challenger", "case-one"} <= sensitive_values
        assert {"candidate-alpha", "candidate-bravo"} <= sensitive_values
    review = _load(Path(built["reviewManifest"]))
    assert review["pdfIdentityScanning"] == {
        "requiredForDeclaredPdf": True,
        "declaredPdfCount": 2,
        "validator": "PDFBox",
        "validatorVersion": "2.0.30",
        "scannerJarSha256": sha256_file(scanner_jar),
    }
    pdf_artifacts = [
        artifact
        for candidate in review["cases"][0]["candidates"]
        for artifact in candidate["artifacts"]
        if artifact["kind"] == "pdf"
    ]
    assert len(pdf_artifacts) == 2
    for artifact in pdf_artifacts:
        scan = artifact["blindIdentityScan"]
        assert scan["status"] == "passed"
        assert scan["pdfSha256"] == artifact["sha256"]
        assert scan["scannerJarSha256"] == sha256_file(scanner_jar)
        assert "path" not in json.dumps(scan, ensure_ascii=False).casefold()


def test_standard_product_pdf_is_rerendered_before_blind_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _audio(tmp_path / "source.wav")
    candidates = (
        _candidate_run(
            tmp_path,
            label="candidate-alpha",
            source=source,
            cases=("case-one",),
            include_pdf=True,
        ),
        _candidate_run(
            tmp_path,
            label="candidate-bravo",
            source=source,
            cases=("case-one",),
            include_pdf=True,
        ),
    )
    original_pdf_shas: set[str] = set()
    for candidate in candidates:
        case_root = candidate.run_root / "outputs" / "case-one"
        old_pdf = case_root / "delivery" / "report.pdf"
        standard_pdf = case_root / "render" / "report.pdf"
        standard_pdf.parent.mkdir()
        standard_pdf.write_bytes(
            b"%PDF-1.4\ncase-one identity-bearing original\n%%EOF\n"
        )
        original_pdf_shas.add(sha256_file(standard_pdf))
        report_document = _write_json(
            case_root / "input" / "report-document.json",
            {
                "schemaVersion": "1.0.0",
                "documentId": "document-opaque-001",
                "title": "Sample case-one",
                "source": {
                    "fileName": "case-one.wav",
                    "mediaType": "audio/wav",
                    "durationMs": 1000,
                    "sha256": "a" * 64,
                },
                "segments": [],
            },
        )
        render_manifest = _write_json(
            case_root / "artifacts" / "manifest.json",
            {
                "schemaVersion": "1.0.0",
                "artifacts": [
                    {
                        "type": "report-document",
                        "relativePath": "input/report-document.json",
                        "sha256": sha256_file(report_document),
                    },
                    {
                        "type": "pdf",
                        "relativePath": "render/report.pdf",
                        "sha256": sha256_file(standard_pdf),
                    },
                ],
            },
        )
        checkpoint_path = case_root / "checkpoint.v2.json"
        checkpoint = _load(checkpoint_path)
        checkpoint["artifactPaths"] = [
            value
            for value in checkpoint["artifactPaths"]
            if Path(value) != old_pdf
        ] + [str(standard_pdf), str(report_document), str(render_manifest)]
        _write_json(checkpoint_path, checkpoint)

    scanner_jar = tmp_path / "pdfbox-scanner.jar"
    scanner_jar.write_bytes(b"frozen scanner fixture")
    rerender_calls: list[dict] = []

    def fake_rerender(**kwargs: object) -> tuple[str, dict]:
        alias_target = Path(str(kwargs["alias_target"]))
        original = kwargs["original_pdf"]
        alias_target.parent.mkdir(parents=True, exist_ok=True)
        alias_target.write_bytes(b"%PDF-1.4\nanonymous reviewer PDF\n%%EOF\n")
        alias_sha = sha256_file(alias_target)
        rerender_calls.append(
            {
                "source": Path(str(kwargs["source_pdf"])),
                "alias": alias_target,
                "originalSha256": original["sha256"],
                "aliasSha256": alias_sha,
            }
        )
        return alias_sha, {
            "schemaVersion": "1.0.0",
            "artifactType": "blind-review-pdf-rerender",
            "sanitizationProfile": "report-title-and-source-file-v1",
            "changedFields": ["$.title", "$.source.fileName"],
            "originalPdfSha256": original["sha256"],
            "originalReportDocumentSha256": "b" * 64,
            "blindReportDocumentSha256": "c" * 64,
            "preservedTranscriptCanonicalSha256": "d" * 64,
            "rendererJarSha256": sha256_file(scanner_jar),
            "rendererVersion": "3.0.0",
            "qualityScore": 97.87,
        }

    def fake_scan(**kwargs: object) -> dict:
        pdf_path = Path(str(kwargs["pdf_path"]))
        expected = str(kwargs["expected_pdf_sha256"])
        scanner = kwargs["scanner"]
        assert sha256_file(pdf_path) == expected
        assert b"case-one" not in pdf_path.read_bytes()
        return {
            "schemaVersion": "1.0.0",
            "artifactType": "blind-review-pdf-identity-scan",
            "validator": "PDFBox",
            "validatorVersion": "2.0.30",
            "status": "passed",
            "pdfSha256": expected,
            "scannerJarSha256": scanner.jar_sha256,
            "allRequiredSurfacesScanned": True,
            "sensitiveValueCount": len(kwargs["sensitive_values"]),
            "pageCount": 1,
            "pageTextCount": 1,
            "documentInfoEntryCount": 0,
            "xmpPacketCount": 0,
            "attachmentCount": 0,
            "annotationCount": 0,
            "formFieldCount": 0,
            "scannedCosObjectCount": 1,
            "decodedStreamBytes": 0,
        }

    monkeypatch.setattr(blind_builder, "_rerender_blind_safe_pdf", fake_rerender)
    monkeypatch.setattr(blind_builder, "_scan_pdf_identity", fake_scan)
    built = build_review_package(
        candidates=candidates,
        output_root=tmp_path / "rerendered-package",
        seed="seed",
        java_executable=sys.executable,
        pdf_scanner_jar=scanner_jar,
    )

    assert len(rerender_calls) == 2
    assert {call["originalSha256"] for call in rerender_calls} == original_pdf_shas
    assert all(
        call["aliasSha256"] != call["originalSha256"]
        for call in rerender_calls
    )
    review = _load(Path(built["reviewManifest"]))
    pdf_artifacts = [
        artifact
        for candidate in review["cases"][0]["candidates"]
        for artifact in candidate["artifacts"]
        if artifact["kind"] == "pdf"
    ]
    assert len(pdf_artifacts) == 2
    assert all(
        artifact["blindPdfRerender"]["originalPdfSha256"]
        in original_pdf_shas
        for artifact in pdf_artifacts
    )
    assert all(
        artifact["blindIdentityScan"]["pdfSha256"] == artifact["sha256"]
        for artifact in pdf_artifacts
    )


def test_package_supports_completed_no_speech_without_transcript(
    tmp_path: Path,
) -> None:
    source = _audio(tmp_path / "silence.wav")
    first = _candidate_run(
        tmp_path,
        label="candidate-alpha",
        source=source,
        cases=("silence-case",),
        no_speech=True,
    )
    second = _candidate_run(
        tmp_path,
        label="candidate-bravo",
        source=source,
        cases=("silence-case",),
        no_speech=True,
    )

    built = build_review_package(
        candidates=(first, second),
        output_root=tmp_path / "no-speech-package",
        seed="seed",
    )

    review_manifest = _load(Path(built["reviewManifest"]))
    for candidate in review_manifest["cases"][0]["candidates"]:
        evidence_path = (
            Path(built["reviewManifest"]).parent
            / candidate["artifacts"][0]["path"]
        )
        evidence = _load(evidence_path)
        assert evidence["disposition"] == "no-transcribable-speech"
        assert evidence["rawAsrSegments"] == []
        assert evidence["finalSegments"] == []
        assert evidence["speakerTimeline"] == {
            "speakerCount": 0,
            "speakerIds": [],
            "turns": [],
        }
    identity = _load(Path(built["identityMapping"]))
    assert all(
        candidate["originalArtifacts"]["transcriptDocument"] is None
        for candidate in identity["cases"][0]["candidates"]
    )


def test_package_ignores_undeclared_evaluation_artifacts(tmp_path: Path) -> None:
    source = _audio(tmp_path / "source.wav")
    first = _candidate_run(
        tmp_path,
        label="candidate-alpha",
        source=source,
        cases=("case-one",),
    )
    second = _candidate_run(
        tmp_path,
        label="candidate-bravo",
        source=source,
        cases=("case-one",),
    )
    for candidate in (first, second):
        evaluation = (
            candidate.run_root
            / "outputs/case-one/evaluation/reference-answer.pdf"
        )
        evaluation.parent.mkdir()
        evaluation.write_bytes(b"%PDF-1.4\nreference answer\n%%EOF\n")

    built = build_review_package(
        candidates=(first, second),
        output_root=tmp_path / "declared-only-package",
        seed="seed",
    )

    review = _load(Path(built["reviewManifest"]))
    assert all(
        {
            artifact["kind"]
            for artifact in candidate["artifacts"]
        }
        == {"candidate-evidence", "subtitle"}
        for candidate in review["cases"][0]["candidates"]
    )
    assert all(
        "reference-answer" not in path.name
        for path in Path(built["reviewManifest"]).parent.rglob("*")
    )


def test_package_refuses_declared_artifact_outside_case(tmp_path: Path) -> None:
    source = _audio(tmp_path / "source.wav")
    first = _candidate_run(
        tmp_path,
        label="candidate-alpha",
        source=source,
        cases=("case-one",),
    )
    second = _candidate_run(
        tmp_path,
        label="candidate-bravo",
        source=source,
        cases=("case-one",),
    )
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\noutside\n%%EOF\n")
    checkpoint_path = (
        first.run_root / "outputs/case-one/checkpoint.v2.json"
    )
    checkpoint = _load(checkpoint_path)
    checkpoint["artifactPaths"].append(str(outside))
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(
        BlindE2EReviewPackageError,
        match="outside its case",
    ):
        build_review_package(
            candidates=(first, second),
            output_root=tmp_path / "escaped-artifact-package",
            seed="seed",
        )


def test_package_hashes_result_without_parsing_scores(tmp_path: Path) -> None:
    source = _audio(tmp_path / "source.wav")
    first = _candidate_run(
        tmp_path,
        label="candidate-alpha",
        source=source,
        cases=("case-one",),
    )
    second = _candidate_run(
        tmp_path,
        label="candidate-bravo",
        source=source,
        cases=("case-one",),
    )
    result_path = first.run_root / "results/case-one-result.json"
    result_path.write_text(
        "opaque evaluator payload; not JSON; automaticScore=0.99",
        encoding="utf-8",
    )

    built = build_review_package(
        candidates=(first, second),
        output_root=tmp_path / "opaque-result-package",
        seed="seed",
    )

    identity = _load(Path(built["identityMapping"]))
    first_result = next(
        candidate["originalArtifacts"]["result"]
        for candidate in identity["cases"][0]["candidates"]
        if candidate["candidateId"] == "candidate-alpha"
    )
    assert first_result["sha256"] == sha256_file(result_path)
    assert "canonicalSha256" not in first_result
    assert "automaticScore=0.99" not in Path(built["reviewManifest"]).read_text(
        encoding="utf-8"
    )


def test_completed_blind_review_authorizes_unblind_with_canonical_evidence(
    tmp_path: Path,
) -> None:
    package_root, completed_review, completed = _package_with_completed_review(
        tmp_path
    )
    output = tmp_path / "unblind-authorization.v1.json"

    authorization = _authorize_completed_review(
        package_root=package_root,
        completed_review=completed_review,
        output=output,
    )

    assert _load(output) == authorization
    assert authorization["artifactType"] == UNBLIND_AUTHORIZATION_ARTIFACT_TYPE
    assert authorization["counts"] == {
        "reviewedCases": len(completed["cases"]),
        "dimensionDecisions": len(completed["cases"]) * 4,
        "severityByDimension": {
            "blocker": 0,
            "major": 0,
            "minor": 0,
            "pass": len(completed["cases"]) * 4,
        },
    }
    assert authorization["heldOutFreezeManifest"] == {
        "fileSha256": "e" * 64,
        "readByAuthorizer": False,
    }
    package = _load(package_root / "package-manifest.v1.json")
    reviewer = _load(
        package_root / package["reviewerPacket"]["relativePath"]
    )
    assert authorization["identityVaultCommitment"] == {
        "relativePath": package["identityVault"]["relativePath"],
        **reviewer["identityCommitment"],
    }
    body = dict(authorization)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)


def test_incomplete_review_dimension_fails_without_authorization(
    tmp_path: Path,
) -> None:
    package_root, completed_review, completed = _package_with_completed_review(
        tmp_path
    )
    completed["cases"][0]["rawAsrReview"] = None
    _write_json(completed_review, completed)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(HeldOutUnblindAuthorizationError, match="rawAsrReview"):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("mutation", ["candidate-order", "preference"])
def test_changed_candidate_order_or_invalid_preference_fails(
    tmp_path: Path,
    mutation: str,
) -> None:
    package_root, completed_review, completed = _package_with_completed_review(
        tmp_path
    )
    if mutation == "candidate-order":
        completed["cases"][0]["candidateOrder"].reverse()
    else:
        completed["cases"][0]["preferredCandidateId"] = "option-999"
    _write_json(completed_review, completed)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(HeldOutUnblindAuthorizationError):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=output,
        )

    assert not output.exists()


def test_review_and_authorization_must_remain_outside_immutable_package(
    tmp_path: Path,
) -> None:
    package_root, completed_review, completed = _package_with_completed_review(
        tmp_path
    )

    with pytest.raises(HeldOutUnblindAuthorizationError, match="output"):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=package_root / "authorization.json",
        )

    inside_review = _write_json(
        package_root / "reviewer-packet" / "completed-review.json",
        completed,
    )
    with pytest.raises(HeldOutUnblindAuthorizationError, match="completed review"):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=inside_review,
            output=tmp_path / "outside-authorization.json",
        )


def test_existing_authorization_is_never_replaced(tmp_path: Path) -> None:
    package_root, completed_review, _ = _package_with_completed_review(tmp_path)
    output = tmp_path / "existing-authorization.json"
    output.write_bytes(b"existing immutable evidence")

    with pytest.raises(FileExistsError):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=output,
        )

    assert output.read_bytes() == b"existing immutable evidence"


@pytest.mark.parametrize("target", ["reviewer-manifest", "review-template"])
def test_tampered_reviewer_manifest_or_template_binding_fails(
    tmp_path: Path,
    target: str,
) -> None:
    package_root, completed_review, _ = _package_with_completed_review(tmp_path)
    package = _load(package_root / "package-manifest.v1.json")
    reviewer_path = package_root / package["reviewerPacket"]["relativePath"]
    reviewer = _load(reviewer_path)
    if target == "reviewer-manifest":
        reviewer["counts"]["cases"] += 1
        _write_json(reviewer_path, reviewer)
    else:
        template_path = reviewer_path.parent / reviewer["reviewForm"]["path"]
        template = _load(template_path)
        template["packageId"] = "rebound-package"
        _write_json(template_path, template)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(HeldOutUnblindAuthorizationError, match="SHA-256"):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=output,
        )

    assert not output.exists()


def test_package_and_reviewer_identity_commitments_must_match(
    tmp_path: Path,
) -> None:
    package_root, completed_review, _ = _package_with_completed_review(tmp_path)
    package_path = package_root / "package-manifest.v1.json"
    package = _load(package_path)
    package["identityVault"]["fileSha256"] = "f" * 64
    _write_json(package_path, package)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(HeldOutUnblindAuthorizationError, match="commitments differ"):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=output,
            expected_package_manifest_sha256=sha256_file(package_path),
        )

    assert not output.exists()


def test_package_manifest_must_match_externally_frozen_digest(
    tmp_path: Path,
) -> None:
    package_root, completed_review, _ = _package_with_completed_review(tmp_path)
    package_path = package_root / "package-manifest.v1.json"
    expected_package_sha256 = sha256_file(package_path)
    package = _load(package_path)
    package["packageId"] = "tampered-package"
    _write_json(package_path, package)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(HeldOutUnblindAuthorizationError, match="externally frozen"):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=output,
            expected_package_manifest_sha256=expected_package_sha256,
        )

    assert not output.exists()


def test_authorizer_never_opens_identity_vault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, completed_review, _ = _package_with_completed_review(tmp_path)
    output = tmp_path / "authorization-with-vault-guard.json"
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if "identity-vault" in path.parts:
            raise AssertionError("authorizer attempted to open identity-vault")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    authorization = _authorize_completed_review(
        package_root=package_root,
        completed_review=completed_review,
        output=output,
    )

    assert (
        authorization["isolationPolicy"]["identityVaultReadByAuthorizer"]
        is False
    )


def test_authorizer_rejects_symlinked_reviewer_manifest(tmp_path: Path) -> None:
    package_root, completed_review, _ = _package_with_completed_review(tmp_path)
    package = _load(package_root / "package-manifest.v1.json")
    reviewer_path = package_root / package["reviewerPacket"]["relativePath"]
    moved = tmp_path / "moved-reviewer-manifest.json"
    reviewer_path.replace(moved)
    try:
        reviewer_path.symlink_to(moved)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(HeldOutUnblindAuthorizationError, match="symbolic link"):
        _authorize_completed_review(
            package_root=package_root,
            completed_review=completed_review,
            output=tmp_path / "must-not-exist.json",
        )


def test_held_out_unblind_authorization_cli_help_resolves_imports() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "authorize_held_out_unblind.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--held-out-freeze-manifest-sha256" in completed.stdout
    assert "--expected-package-manifest-sha256" in completed.stdout


def test_held_out_unblind_authorization_cli_executes_windows_path(
    tmp_path: Path,
) -> None:
    package_root, completed_review, _ = _package_with_completed_review(tmp_path)
    package_manifest = package_root / "package-manifest.v1.json"
    output = tmp_path / "cli-unblind-authorization.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "authorize_held_out_unblind.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--package-root",
            str(package_root),
            "--completed-review",
            str(completed_review),
            "--held-out-freeze-manifest-sha256",
            "e" * 64,
            "--expected-package-manifest-sha256",
            sha256_file(package_manifest),
            "--reviewer-source",
            "codex-agent",
            "--reviewer",
            "Codex held-out reviewer",
            "--reviewed-at",
            "2026-08-09T18:00:00+08:00",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    authorization = json.loads(completed.stdout)
    assert authorization == _load(output)
    body = dict(authorization)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
