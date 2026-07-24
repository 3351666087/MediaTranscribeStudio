from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.build_voice_activity_sample_library import (
    APPROVED_LICENSES,
    VoiceActivitySampleError,
    load_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sample_library" / "voice-activity-manifest.v1.json"


def test_manifest_has_diverse_licensed_lexical_speech_negatives() -> None:
    manifest = load_manifest(MANIFEST)

    assert len(manifest["cases"]) >= 11
    assert {source["license"] for source in manifest["sources"]} <= set(
        APPROVED_LICENSES
    )
    assert {
        "environment",
        "domestic",
        "human-nonlexical",
        "animal",
        "urban",
        "instrumental-music",
        "silence",
    } <= {case["signalClass"] for case in manifest["cases"]}
    assert {case["evaluationSplit"] for case in manifest["cases"]} == {
        "development",
        "regression",
        "held-out",
    }
    assert all(case["expectedLexicalSpeech"] is False for case in manifest["cases"])


def test_manifest_rejects_unapproved_license(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["sources"][0]["license"] = "unknown"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(VoiceActivitySampleError, match="license is not approved"):
        load_manifest(path)


def test_manifest_rejects_duplicate_case_and_invalid_hash(
    tmp_path: Path,
) -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["cases"][1]["id"] = value["cases"][0]["id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(VoiceActivitySampleError, match="duplicate case"):
        load_manifest(path)

    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    value["cases"][0]["asset"]["sha256"] = "bad"
    path = tmp_path / "hash.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(VoiceActivitySampleError, match="lowercase SHA-256"):
        load_manifest(path)
