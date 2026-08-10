from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.build_local_blind_product_matrix import (
    BlindProductMatrixError,
    build_matrix,
)


def _wav(path: Path, *, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * int(16_000 * seconds))


def _manifest(path: Path) -> Path:
    value = {
        "schemaVersion": "1.0.0",
        "libraryId": "fixture-library",
        "maxDurationSeconds": 1.0,
        "cases": [
            {
                "id": "dev-case",
                "sourceId": "fixture",
                "acquisition": {"kind": "fixture", "rowIndex": 0},
                "language": "zh-CN",
                "region": "East Asia",
                "evaluationSplit": "development",
                "scenario": ["real-recording", "single-speaker"],
                "expectedSpeakerCount": 1,
            },
            {
                "id": "held-case",
                "sourceId": "fixture",
                "acquisition": {"kind": "fixture", "rowIndex": 1},
                "language": "en-US",
                "region": "North America",
                "evaluationSplit": "held-out",
                "scenario": ["real-recording", "single-speaker"],
                "expectedSpeakerCount": 1,
            },
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_build_matrix_defaults_to_truth_redacted_development(tmp_path: Path) -> None:
    source = _manifest(tmp_path / "source.json")
    audio = tmp_path / "audio"
    _wav(audio / "dev-case.wav")
    output = tmp_path / "blind.json"

    result = build_matrix(
        source_manifest=source,
        audio_root=audio,
        output_path=output,
    )

    assert result["counts"] == {"cases": 1, "missingCases": 0}
    assert result["selection"]["splits"] == ["development"]
    assert result["truthPersistencePolicy"]["expectedAnswerPersisted"] is False
    row = result["cases"][0]
    assert row["id"] == "dev-case"
    assert row["durationSeconds"] == 0.1
    assert row["sha256"] == sha256_file(audio / "dev-case.wav")
    assert not any("transcript" in key.casefold() for key in row)
    body = dict(result)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)


def test_held_out_requires_explicit_unlock(tmp_path: Path) -> None:
    source = _manifest(tmp_path / "source.json")
    audio = tmp_path / "audio"
    _wav(audio / "held-case.wav")

    with pytest.raises(BlindProductMatrixError, match="explicit unlock"):
        build_matrix(
            source_manifest=source,
            audio_root=audio,
            output_path=tmp_path / "blind.json",
            splits=("held-out",),
        )

    result = build_matrix(
        source_manifest=source,
        audio_root=audio,
        output_path=tmp_path / "blind.json",
        splits=("held-out",),
        unlock_held_out=True,
    )
    assert result["cases"][0]["id"] == "held-case"
    assert result["selection"]["heldOutExplicitlyUnlocked"] is True


def test_missing_media_is_explicit_and_bounded(tmp_path: Path) -> None:
    source = _manifest(tmp_path / "source.json")
    audio = tmp_path / "audio"
    audio.mkdir()

    with pytest.raises(BlindProductMatrixError, match="missing"):
        build_matrix(
            source_manifest=source,
            audio_root=audio,
            output_path=tmp_path / "blind.json",
        )


def test_source_truth_field_is_rejected(tmp_path: Path) -> None:
    source = _manifest(tmp_path / "source.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["cases"][0]["expectedTranscript"] = "hidden"
    source.write_text(json.dumps(value), encoding="utf-8")
    audio = tmp_path / "audio"
    _wav(audio / "dev-case.wav")

    with pytest.raises(BlindProductMatrixError, match="truth field"):
        build_matrix(
            source_manifest=source,
            audio_root=audio,
            output_path=tmp_path / "blind.json",
        )
