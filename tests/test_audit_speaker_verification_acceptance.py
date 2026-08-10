from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256
from tools.audit_speaker_verification_acceptance import (
    SpeakerVerificationAcceptanceError,
    _assert_aggregate_only,
    _bucket_metrics,
    _calibration_metrics,
    _embedding_freeze_evidence,
    _fit_platt_calibrator,
    _metrics_at_threshold,
    _operating_metrics,
)


def _development_scores() -> tuple[tuple[tuple[str, str, str, bool], float], ...]:
    values = (
        ("d-1", "l-1", "r-1", True, 0.90),
        ("d-2", "l-2", "r-2", True, 0.80),
        ("d-3", "l-3", "r-3", True, 0.70),
        ("d-4", "l-4", "r-4", True, 0.60),
        ("d-5", "l-5", "r-5", False, 0.20),
        ("d-6", "l-6", "r-6", False, 0.10),
        ("d-7", "l-7", "r-7", False, 0.00),
        ("d-8", "l-8", "r-8", False, -0.10),
    )
    return tuple(((trial_id, left, right, label), score) for trial_id, left, right, label, score in values)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "score": 0.9,
            "label": True,
            "genreRelation": "same-genre",
            "genrePair": "interview+interview",
        },
        {
            "score": 0.7,
            "label": True,
            "genreRelation": "cross-genre",
            "genrePair": "interview+singing",
        },
        {
            "score": 0.1,
            "label": False,
            "genreRelation": "same-genre",
            "genrePair": "interview+interview",
        },
        {
            "score": -0.1,
            "label": False,
            "genreRelation": "cross-genre",
            "genrePair": "interview+singing",
        },
    ]


def test_platt_calibrator_is_frozen_from_development_only() -> None:
    scores = _development_scores()
    calibrator = _fit_platt_calibrator(scores)

    assert calibrator["kind"] == "platt-logistic-standardized-score-v1"
    assert calibrator["fitPolicy"]["sourceSplit"] == "development"
    assert calibrator["fitPolicy"]["heldOutFittingPerformed"] is False
    canonical = dict(calibrator)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)

    # A held-out score change cannot alter a calibrator whose training input is
    # the development score set only.
    held_out_a = _calibration_metrics(_rows(), calibrator)
    changed_rows = _rows()
    changed_rows[0]["score"] = -0.95
    held_out_b = _calibration_metrics(changed_rows, calibrator)
    assert held_out_a != held_out_b
    assert calibrator["canonicalSha256"] == _fit_platt_calibrator(scores)[
        "canonicalSha256"
    ]


def test_final_metrics_and_buckets_are_aggregate_only() -> None:
    rows = _rows()
    operating = _operating_metrics(rows)
    threshold_metrics = _metrics_at_threshold(rows, 0.5)
    calibrator = _fit_platt_calibrator(_development_scores())
    buckets = _bucket_metrics(
        rows,
        field="genreRelation",
        threshold=0.5,
        calibrator=calibrator,
    )

    assert operating["eer"] == 0.0
    assert "minDcfPTarget0.01" in operating
    assert threshold_metrics["far"] == 0.0
    assert threshold_metrics["frr"] == 0.0
    assert {item["bucket"] for item in buckets} == {
        "same-genre",
        "cross-genre",
    }
    for item in buckets:
        assert item["calibration"]["expectedCalibrationError10"] >= 0.0
        assert item["operating"] is not None
    _assert_aggregate_only({"buckets": buckets, "calibration": calibrator})


def test_identity_fields_are_rejected_from_aggregate_artifacts() -> None:
    with pytest.raises(
        SpeakerVerificationAcceptanceError,
        match="identity fields",
    ):
        _assert_aggregate_only({"trialId": "held-out-secret"})


def test_embedding_freeze_binds_model_manifest_split_and_algorithm(
    tmp_path: Path,
) -> None:
    model = {"modelKey": "fixture", "revision": "pinned"}
    model_identity = canonical_json_sha256(model)
    manifests = {"development": "a" * 64, "held-out": "b" * 64}
    reports: list[Path] = []
    for split, digest in manifests.items():
        document = {
            "schemaVersion": "1.0.0",
            "model": model,
            "trialManifest": {"canonicalSha256": digest},
            "partition": {"evaluationSplit": split},
            "execution": {
                "device": "cpu",
                "embeddingDimensions": 192,
                "embeddingSetHashAlgorithm": "clip-id-sorted-float32-le-v1",
                "embeddingSetSha256": ("c" if split == "development" else "d")
                * 64,
            },
        }
        document["canonicalSha256"] = canonical_json_sha256(document)
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        reports.append(path)

    evidence = _embedding_freeze_evidence(
        candidate_id="fixture",
        reports=(reports[0], reports[1]),
        development={
            "modelIdentitySha256": model_identity,
            "manifest": {"canonicalSha256": manifests["development"]},
        },
        held_out={
            "modelIdentitySha256": model_identity,
            "manifest": {"canonicalSha256": manifests["held-out"]},
        },
    )

    assert evidence is not None
    assert evidence["hashAlgorithm"] == "clip-id-sorted-float32-le-v1"
    assert evidence["splits"]["development"]["evaluationSplit"] == (
        "development"
    )
    assert evidence["splits"]["heldOut"]["evaluationSplit"] == "held-out"
    canonical = dict(evidence)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)
