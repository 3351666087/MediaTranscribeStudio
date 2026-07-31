from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.run_first_principles_asr_probe import BlindProbeError, load_blind_cases


def _manifest(tmp_path: Path) -> Path:
    media = tmp_path / "case-0123456789abcdef.wav"
    media.write_bytes(b"audio")
    path = tmp_path / "blind.json"
    path.write_text(
        json.dumps(
            {
                "artifactType": "first-principles-blind-media-batch",
                "batchId": "batch",
                "cases": [
                    {
                        "auditCaseId": media.stem,
                        "durationSeconds": 2.0,
                        "media": {
                            "path": str(media),
                            "sha256": hashlib.sha256(b"audio").hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_blind_cases_accepts_only_opaque_media(tmp_path: Path) -> None:
    manifest, cases = load_blind_cases(_manifest(tmp_path))

    assert manifest["batchId"] == "batch"
    assert cases[0]["auditCaseId"] == "case-0123456789abcdef"
    assert cases[0]["mediaSha256"] == hashlib.sha256(b"audio").hexdigest()


def test_load_blind_cases_rejects_source_case_id(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["cases"][0]["caseId"] = "source/language-n5"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(BlindProbeError, match="must not expose source case IDs"):
        load_blind_cases(path)


def test_load_blind_cases_rejects_revealing_filename(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    revealing = tmp_path / "english-speaker-5.wav"
    revealing.write_bytes(b"audio")
    value["cases"][0]["media"]["path"] = str(revealing)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(BlindProbeError, match="not an opaque local file"):
        load_blind_cases(path)
