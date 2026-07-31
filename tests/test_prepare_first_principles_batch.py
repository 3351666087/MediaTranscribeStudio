from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.prepare_first_principles_batch import (
    FirstPrinciplesBatchError,
    SourceSpec,
    prepare_batch,
    select_diverse_cases,
)


def _write_source(
    root: Path,
    *,
    collection: str,
    kind: str,
    cases: list[dict[str, object]],
) -> SourceSpec:
    source = root / collection
    audio = source / "audio"
    audio.mkdir(parents=True)
    for case in cases:
        media = f"{case['id']}.wav"
        payload = f"media:{collection}:{case['id']}".encode()
        (audio / media).write_bytes(payload)
        case["path"] = f"audio/{media}"
        case["sha256"] = hashlib.sha256(payload).hexdigest()
        case["bytes"] = len(payload)
    path = source / "resolved.json"
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return SourceSpec(collection, kind, path)


def test_prepare_batch_is_truth_isolated_and_hash_bound(tmp_path: Path) -> None:
    speech = _write_source(
        tmp_path,
        collection="speech",
        kind="global-single-speaker",
        cases=[
            {
                "id": "english",
                "evaluationSplit": "development",
                "durationSeconds": 4.0,
                "expectedSpeakerCount": 1,
                "language": "en-US",
                "scoringTranscript": "secret expected words",
                "scenario": ["clean"],
            },
            {
                "id": "held-out-spanish",
                "evaluationSplit": "held-out",
                "durationSeconds": 5.0,
                "expectedSpeakerCount": 1,
                "language": "es-ES",
                "scoringTranscript": "secreto",
                "scenario": ["telephone"],
            },
        ],
    )
    negative = _write_source(
        tmp_path,
        collection="negative",
        kind="voice-activity",
        cases=[
            {
                "id": "music",
                "evaluationSplit": "regression",
                "durationSeconds": 8.0,
                "expectedLexicalSpeech": False,
                "humanVocalization": False,
                "signalClass": "instrumental-music",
                "category": "orchestral",
                "truthEligibility": {"lexicalSpeech": True},
            }
        ],
    )

    summary = prepare_batch(
        sources=[speech, negative],
        output_root=tmp_path / "output",
        maximum=10,
    )

    blind_path = Path(summary["blindManifest"])
    oracle_path = Path(summary["sealedReference"])
    ledger_path = Path(summary["processingLedger"])
    blind_text = blind_path.read_text(encoding="utf-8")
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert "secret expected words" not in blind_text
    assert "secreto" not in blind_text
    cases_by_id = {case["caseId"]: case for case in oracle["cases"]}
    assert set(cases_by_id) == {"negative/music", "speech/english"}
    assert cases_by_id["negative/music"]["reference"]["lexicalSpeechPresent"] is False
    assert cases_by_id["speech/english"]["reference"]["transcript"] == (
        "secret expected words"
    )
    assert blind_path.parent == oracle_path.parent == ledger_path.parent
    assert "speech/english" not in blind_text
    assert "negative/music" not in blind_text
    assert str(tmp_path / "speech") not in blind_text
    assert all(
        case["auditCaseId"].startswith("case-")
        for case in json.loads(blind_text)["cases"]
    )
    assert hashlib.sha256(oracle_path.read_bytes()).hexdigest() == summary[
        "sealedReferenceSha256"
    ]
    assert ledger["truthAccessed"] is False
    assert all(case["status"] == "pending" for case in ledger["cases"])
    assert all("caseId" not in case for case in ledger["cases"])


def test_diverse_selection_uses_truth_only_for_sampling_not_blind_output() -> None:
    cases = [
        {
            "caseId": "a/clean",
            "collection": "a",
            "origin": "real-recording",
            "partition": "development",
            "durationSeconds": 5.0,
            "scenarios": ["clean"],
            "referenceClass": "partial-oracle",
            "eligibility": {"transcript": True},
            "reference": {
                "lexicalSpeechPresent": True,
                "speakerCount": 1,
                "languages": ["en"],
            },
        },
        {
            "caseId": "a/noise",
            "collection": "a",
            "origin": "real-recording",
            "partition": "development",
            "durationSeconds": 5.0,
            "scenarios": ["noise"],
            "referenceClass": "partial-oracle",
            "eligibility": {"transcript": True},
            "reference": {
                "lexicalSpeechPresent": True,
                "speakerCount": 1,
                "languages": ["en"],
            },
        },
        {
            "caseId": "b/overlap",
            "collection": "b",
            "origin": "real-recording",
            "partition": "regression",
            "durationSeconds": 40.0,
            "scenarios": ["overlap"],
            "referenceClass": "full-final-state-oracle",
            "eligibility": {"transcript": True, "speakerTimeline": True},
            "reference": {
                "lexicalSpeechPresent": True,
                "speakerCount": 5,
                "languages": ["zh"],
            },
        },
    ]

    selected = select_diverse_cases(cases, maximum=2)

    assert {case["caseId"] for case in selected} >= {"b/overlap"}
    assert len(selected) == 2


def test_media_hash_verification_rejects_rebound_file(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path,
        collection="speech",
        kind="global-single-speaker",
        cases=[
            {
                "id": "one",
                "evaluationSplit": "development",
                "durationSeconds": 2.0,
                "expectedSpeakerCount": 1,
                "language": "en-US",
                "scoringTranscript": "one",
                "scenario": ["clean"],
            }
        ],
    )
    media = source.path.parent / "audio/one.wav"
    media.write_bytes(b"rebound")

    with pytest.raises(FirstPrinciplesBatchError, match="media SHA-256 mismatch"):
        prepare_batch(
            sources=[source],
            output_root=tmp_path / "output",
            maximum=1,
            verify_media_hashes=True,
        )
