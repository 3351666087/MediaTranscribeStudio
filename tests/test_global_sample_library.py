from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.global_sample_library import (
    GlobalSampleLibraryError,
    coverage_summary,
    load_global_manifest,
)
from tools.build_global_derived_matrix import overlap_intervals
from tools.build_global_real_diarization import select_diarization_window


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sample_library" / "global-manifest.v1.json"


def test_global_manifest_covers_regions_languages_and_splits() -> None:
    manifest = load_global_manifest(MANIFEST)
    coverage = coverage_summary(manifest)

    assert coverage["caseCount"] >= 20
    assert len(coverage["languages"]) >= 12
    assert len(coverage["regions"]) >= 8
    assert coverage["evaluationSplits"] == [
        "development",
        "held-out",
        "regression",
    ]
    assert {"real-recording", "single-speaker"} <= set(coverage["scenarios"])
    assert {source.license for source in manifest.sources} == {"cc-by-4.0"}


def test_global_manifest_requires_pinned_dataset_revision(
    tmp_path: Path,
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["sources"][0]["revision"] = "main"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="40-character commit"):
        load_global_manifest(path)


def test_global_manifest_rejects_unapproved_or_missing_license(
    tmp_path: Path,
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["sources"][0]["license"] = "unknown"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="license is not approved"):
        load_global_manifest(path)


def test_global_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["cases"][1]["id"] = value["cases"][0]["id"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GlobalSampleLibraryError, match="case IDs must be unique"):
        load_global_manifest(path)


def test_global_manifest_dynamic_n_matrices_cover_stress_counts() -> None:
    manifest = load_global_manifest(MANIFEST)
    matrices = {
        str(matrix["kind"]): matrix for matrix in manifest.derived_matrices
    }

    assert matrices["sequential-speaker-mixture"]["speakerCounts"] == [2, 3, 5, 8]
    assert matrices["overlap-speaker-mixture"]["speakerCounts"] == [
        2,
        3,
        5,
        8,
        13,
    ]
    assert len(matrices["overlap-speaker-mixture"]["sourceCaseIds"]) >= 13


def test_overlap_intervals_tracks_distinct_active_speakers() -> None:
    turns = [
        {"speakerId": "a", "startSeconds": 0.0, "endSeconds": 4.0},
        {"speakerId": "b", "startSeconds": 1.0, "endSeconds": 3.0},
        {"speakerId": "c", "startSeconds": 2.0, "endSeconds": 5.0},
    ]

    assert overlap_intervals(turns) == [
        {"startSeconds": 1.0, "endSeconds": 2.0, "speakerIds": ["a", "b"]},
        {
            "startSeconds": 2.0,
            "endSeconds": 3.0,
            "speakerIds": ["a", "b", "c"],
        },
        {"startSeconds": 3.0, "endSeconds": 4.0, "speakerIds": ["a", "c"]},
    ]


def test_real_diarization_window_is_exact_bounded_and_deterministic() -> None:
    turns = [
        {"speakerId": "a", "startSeconds": 0.0, "endSeconds": 12.0},
        {"speakerId": "b", "startSeconds": 4.0, "endSeconds": 10.0},
        {"speakerId": "c", "startSeconds": 30.0, "endSeconds": 35.0},
    ]

    first = select_diarization_window(turns, 2)
    second = select_diarization_window(turns, 2)

    assert first == second
    assert first["algorithm"] == "event-boundary-shortest-coverage-v2"
    assert first["speakerSet"] == ["a", "b"]
    assert first["durationSeconds"] == 10.0
    assert first["annotatedOverlapSeconds"] == 6.0
    assert {turn["transcript"] for turn in first["turns"]} == {None}
