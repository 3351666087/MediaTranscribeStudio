from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.freeze_product_sample_matrix import (
    ProductMatrixFreezeError,
    build_product_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "sample_library" / "product-matrix-requirements.v1.json"


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _wav(path: Path, *, seconds: float = 0.1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = (sum(path.name.encode("utf-8")) % 30_000).to_bytes(
        2,
        byteorder="little",
        signed=True,
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(sample * round(16_000 * seconds))
    return path


def _declared_manifest(path: Path, cases: list[dict[str, object]]) -> Path:
    return _write_json(
        path,
        {
            "schemaVersion": "1.0.0",
            "libraryId": path.stem,
            "sources": [
                {
                    "id": "fixture",
                    "dataset": "fixture/data",
                    "revision": "a" * 40,
                    "license": "cc-by-4.0",
                    "recordingType": "real-recording",
                }
            ],
            "cases": cases,
        },
    )


def _resolved_manifest(
    path: Path,
    *,
    source_id: str,
    cases: list[dict[str, object]],
    recording_type: str = "real-recording",
) -> Path:
    return _write_json(
        path,
        {
            "schemaVersion": "1.0.0",
            "libraryId": path.stem,
            "sources": [
                {
                    "sourceId": source_id,
                    "dataset": f"fixture/{source_id}",
                    "revision": "b" * 40,
                    "license": "cc-by-4.0",
                    "recordingType": recording_type,
                }
            ],
            "cases": cases,
        },
    )


def _base_inputs(tmp_path: Path) -> dict[str, object]:
    global_audio = tmp_path / "global-audio"
    global_cases: list[dict[str, object]] = []
    for case_id, language, split in (
        ("en-dev", "en-US", "development"),
        ("en-held", "en-US", "held-out"),
        ("zh-held", "zh-CN", "held-out"),
    ):
        _wav(global_audio / f"{case_id}.wav")
        global_cases.append(
            {
                "id": case_id,
                "sourceId": "fixture",
                "language": language,
                "evaluationSplit": split,
                "scenario": ["real-recording", "single-speaker"],
                "expectedSpeakerCount": 1,
            }
        )
    global_manifest = _declared_manifest(
        tmp_path / "global.json",
        global_cases,
    )
    global_value = json.loads(global_manifest.read_text(encoding="utf-8"))
    global_value["plannedRealDiarizationSources"] = [
        {"sourceId": "ami", "evaluationSplit": "development"}
    ]
    global_manifest.write_text(json.dumps(global_value), encoding="utf-8")

    code_cases = [
        {
            "id": "cs-dev",
            "sourceId": "fixture",
            "evaluationSplit": "development",
        },
        {
            "id": "cs-held",
            "sourceId": "fixture",
            "evaluationSplit": "held-out",
        },
    ]
    code_manifest = _declared_manifest(tmp_path / "code.json", code_cases)
    voice_manifest = _declared_manifest(
        tmp_path / "voice.json",
        [
            {
                "id": "noise-dev",
                "sourceId": "fixture",
                "evaluationSplit": "development",
            }
        ],
    )
    return {
        "requirements_path": REQUIREMENTS,
        "global_manifest_path": global_manifest,
        "global_audio_root": global_audio,
        "code_switch_manifest_path": code_manifest,
        "code_switch_resolved_path": tmp_path / "missing-code-resolved.json",
        "voice_activity_manifest_path": voice_manifest,
        "voice_activity_resolved_path": tmp_path / "missing-voice-resolved.json",
        "real_diarization_manifest_paths": (),
        "long_media_manifest_paths": (),
        "supporting_artifacts": (),
        "output_path": tmp_path / "output" / "matrix.json",
    }


def test_matrix_redacts_truth_and_reports_real_split_coverage(tmp_path: Path) -> None:
    inputs = _base_inputs(tmp_path)

    diarization_manifests = []
    for source_id, split in (("ami", None), ("alimeeting", "held-out")):
        root = tmp_path / source_id
        audio = _wav(root / "audio" / f"{source_id}-n2.wav")
        row: dict[str, object] = {
            "id": f"{source_id}-n2",
            "sourceId": source_id,
            "path": f"audio/{audio.name}",
            "bytes": audio.stat().st_size,
            "sha256": sha256_file(audio),
            "audio": {"durationSeconds": 0.1},
            "realOrSynthetic": "real-recording",
            "expectedSpeakerCount": 2,
            "scenario": ["real-recording"],
            "speakerSet": ["secret-a", "secret-b"],
            "turns": [{"speakerId": "secret-a", "transcript": "never persist"}],
            "overlapIntervals": [{"startSeconds": 0, "endSeconds": 0.1}],
            "truthEligibility": {
                "speakerCount": True,
                "turnBoundaries": True,
                "overlap": True,
                "asr": False,
            },
        }
        if split is not None:
            row["evaluationSplit"] = split
        diarization_manifests.append(
            _resolved_manifest(
                root / "resolved.json",
                source_id=source_id,
                cases=[row],
            )
        )
    inputs["real_diarization_manifest_paths"] = diarization_manifests

    result = build_product_matrix(**inputs)

    coverage = {
        (row["dimension"], row["bucket"]): row for row in result["coverage"]
    }
    assert coverage[("language", "english")]["complete"] is True
    assert coverage[("language", "mandarin")]["missingEvaluationSplits"] == [
        "development"
    ]
    assert coverage[("speaker-count", "N=2")]["complete"] is True
    assert coverage[("scenario", "overlap")]["complete"] is True
    assert result["completion"]["taskChecklistMayBeMarkedComplete"] is False
    serialized = json.dumps(result)
    assert "never persist" not in serialized
    assert "secret-a" not in serialized
    assert '"turns"' not in serialized
    body = dict(result)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)


def test_missing_resolved_sources_list_every_declared_case(tmp_path: Path) -> None:
    inputs = _base_inputs(tmp_path)

    result = build_product_matrix(**inputs)

    inventory = {row["id"]: row for row in result["sourceInventory"]}
    assert inventory["code-switch"]["state"] == "missing"
    assert inventory["code-switch"]["missingCaseIds"] == ["cs-dev", "cs-held"]
    assert inventory["voice-activity"]["missingCaseIds"] == ["noise-dev"]
    gaps = {row["id"]: row for row in result["gaps"]}
    assert gaps["language:code-switch"]["reasonCode"] == (
        "NO_LOCAL_QUALIFYING_CASE"
    )
    assert gaps["scenario:noise"]["missingEvaluationSplits"] == [
        "development",
        "held-out",
    ]


def test_resolved_code_switch_truth_is_redacted_but_availability_is_kept(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    code_root = tmp_path / "code-resolved"
    rows: list[dict[str, object]] = []
    for case_id, split in (("cs-dev", "development"), ("cs-held", "held-out")):
        audio = _wav(code_root / "audio" / f"{case_id}.wav")
        rows.append(
            {
                "id": case_id,
                "sourceId": "fixture",
                "path": f"audio/{audio.name}",
                "bytes": audio.stat().st_size,
                "sha256": sha256_file(audio),
                "audio": {"durationSeconds": 0.1},
                "realOrSynthetic": "real-recording",
                "evaluationSplit": split,
                "expectedLanguages": ["zh", "en"],
                "scenario": ["real-recording", "single-speaker", "intra-utterance"],
                "expectedTranscript": "TOP SECRET REFERENCE",
                "truthEligibility": {"asr": True, "languageTiming": False},
            }
        )
    resolved = _resolved_manifest(
        code_root / "resolved.json",
        source_id="fixture",
        cases=rows,
    )
    inputs["code_switch_resolved_path"] = resolved

    result = build_product_matrix(**inputs)

    code_coverage = next(
        row
        for row in result["coverage"]
        if row["dimension"] == "language" and row["bucket"] == "code-switch"
    )
    assert code_coverage["complete"] is True
    assert code_coverage["referenceTruthAvailability"]["transcript"] is True
    assert "TOP SECRET REFERENCE" not in json.dumps(result)


def test_media_digest_mismatch_aborts_freeze(tmp_path: Path) -> None:
    inputs = _base_inputs(tmp_path)
    code_root = tmp_path / "code-resolved"
    audio = _wav(code_root / "audio" / "cs-dev.wav")
    resolved = _resolved_manifest(
        code_root / "resolved.json",
        source_id="fixture",
        cases=[
            {
                "id": "cs-dev",
                "sourceId": "fixture",
                "path": f"audio/{audio.name}",
                "bytes": audio.stat().st_size,
                "sha256": "0" * 64,
                "audio": {"durationSeconds": 0.1},
                "realOrSynthetic": "real-recording",
                "evaluationSplit": "development",
                "expectedLanguages": ["zh", "en"],
                "scenario": ["real-recording", "single-speaker", "intra-utterance"],
            }
        ],
    )
    inputs["code_switch_resolved_path"] = resolved

    with pytest.raises(ProductMatrixFreezeError, match="SHA-256 mismatch"):
        build_product_matrix(**inputs)


def test_synthetic_code_switch_does_not_satisfy_real_product_bucket(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    code_root = tmp_path / "code-resolved"
    rows: list[dict[str, object]] = []
    for case_id, split in (("cs-dev", "development"), ("cs-held", "held-out")):
        audio = _wav(code_root / "audio" / f"{case_id}.wav")
        rows.append(
            {
                "id": case_id,
                "sourceId": "fixture",
                "path": f"audio/{audio.name}",
                "bytes": audio.stat().st_size,
                "sha256": sha256_file(audio),
                "audio": {"durationSeconds": 0.1},
                "evaluationSplit": split,
                "expectedLanguages": ["en", "es"],
                "scenario": ["synthetic-mixture", "exact-switch-timing"],
            }
        )
    inputs["code_switch_resolved_path"] = _resolved_manifest(
        code_root / "resolved.json",
        source_id="fixture",
        cases=rows,
        recording_type="synthetic-mixture",
    )

    result = build_product_matrix(**inputs)

    code_coverage = next(
        row
        for row in result["coverage"]
        if row["dimension"] == "language" and row["bucket"] == "code-switch"
    )
    assert code_coverage["complete"] is False
    assert code_coverage["requiredSplitCaseCounts"] == {
        "development": 0,
        "held-out": 0,
    }


def test_long_media_license_override_requires_matching_source_digest(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    long_root = tmp_path / "long"
    source = long_root / "source.webm"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"pinned long source")
    source_sha256 = sha256_file(source)
    requirements = json.loads(REQUIREMENTS.read_text(encoding="utf-8"))
    requirements["sourceEvidenceOverrides"] = [
        {
            "sourceSha256": source_sha256,
            "languageTags": ["en-US"],
            "license": "public-domain",
            "provider": "fixture",
            "sourceReference": "fixture-reference.json",
        }
    ]
    inputs["requirements_path"] = _write_json(
        tmp_path / "requirements.json",
        requirements,
    )
    clip = _wav(long_root / "audio" / "long-dev.wav")
    resolved = _write_json(
        long_root / "resolved.json",
        {
            "schemaVersion": "1.0.0",
            "sources": [
                {
                    "id": "long-source",
                    "path": source.name,
                    "sha256": source_sha256,
                    "durationMs": 600_000,
                }
            ],
            "cases": [
                {
                    "id": "long-dev",
                    "sourceId": "long-source",
                    "path": f"audio/{clip.name}",
                    "bytes": clip.stat().st_size,
                    "sha256": sha256_file(clip),
                    "audio": {"durationSeconds": 0.1},
                    "evaluationSplit": "development",
                    "realOrSynthetic": "real-recording",
                    "scenario": ["real-recording", "long-media-stratified"],
                }
            ],
        },
    )
    inputs["long_media_manifest_paths"] = [resolved]

    result = build_product_matrix(**inputs)

    frozen = next(row for row in result["cases"] if row["id"] == "long-dev")
    assert frozen["languageTags"] == ["en-US"]
    assert frozen["provenance"]["license"] == "public-domain"
    assert frozen["provenance"]["licenseEvidence"]["provider"] == "fixture"
    coverage = next(
        row
        for row in result["coverage"]
        if row["dimension"] == "scenario" and row["bucket"] == "long-media"
    )
    assert coverage["requiredSplitCaseCounts"] == {
        "development": 1,
        "held-out": 0,
    }

    source.write_bytes(b"tampered")
    with pytest.raises(ProductMatrixFreezeError, match="source evidence SHA-256"):
        build_product_matrix(**inputs)


def _self_contained_long_media_source(
    root: Path,
    *,
    source: Path,
    source_id: str = "long-source",
) -> dict[str, object]:
    source_sha256 = sha256_file(source)
    reference = _write_json(
        root / "long-media-source-reference.v1.json",
        {
            "schemaVersion": "1.0.0",
            "artifactType": "long-media-source-reference",
            "source": {
                "dataset": "wikimedia-commons",
                "revision": "commons-page-revision-123+etag-fixture",
                "provider": "Wikimedia Commons",
                "sourceUrl": "https://upload.wikimedia.org/fixture.ogg",
                "descriptionUrl": "https://commons.wikimedia.org/wiki/File:Fixture.ogg",
                "sha256": source_sha256,
                "bytes": source.stat().st_size,
                "durationSeconds": 600.0,
                "languageTags": ["en-US"],
                "region": "US",
                "recordingType": "real-recording",
                "speechNature": "natural-multi-speaker",
                "immutableEvidence": {
                    "pageRevisionId": 123,
                    "etag": "fixture",
                },
                "license": {
                    "id": "cc-by-2.5",
                    "shortName": "CC BY 2.5",
                    "url": "https://creativecommons.org/licenses/by/2.5/",
                    "attributionRequired": True,
                },
            },
        },
    )
    return {
        "id": source_id,
        "path": source.name,
        "sha256": source_sha256,
        "durationMs": 600_000,
        "dataset": "wikimedia-commons",
        "revision": "commons-page-revision-123+etag-fixture",
        "license": "cc-by-2.5",
        "languageTags": ["en-US"],
        "recordingType": "real-recording",
        "licenseEvidence": {
            "provider": "Wikimedia Commons",
            "sourceUrl": "https://upload.wikimedia.org/fixture.ogg",
            "descriptionUrl": "https://commons.wikimedia.org/wiki/File:Fixture.ogg",
            "licenseUrl": "https://creativecommons.org/licenses/by/2.5/",
            "pageRevisionId": 123,
            "etag": "fixture",
            "sourceReference": reference.name,
            "sourceReferenceSha256": sha256_file(reference),
        },
    }


def test_long_media_self_contained_evidence_supports_held_out_split(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    long_root = tmp_path / "long"
    source = long_root / "source.ogg"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"pinned public long recording")
    frozen_source = _self_contained_long_media_source(long_root, source=source)
    clip = _wav(long_root / "audio" / "long-held.wav")
    resolved = _write_json(
        long_root / "resolved.json",
        {
            "schemaVersion": "1.0.0",
            "sources": [frozen_source],
            "cases": [
                {
                    "id": "long-held",
                    "sourceId": "long-source",
                    "path": f"audio/{clip.name}",
                    "bytes": clip.stat().st_size,
                    "sha256": sha256_file(clip),
                    "audio": {"durationSeconds": 0.1},
                    "evaluationSplit": "held-out",
                    "realOrSynthetic": "real-recording",
                    "scenario": ["real-recording", "long-media-stratified"],
                }
            ],
        },
    )
    inputs["long_media_manifest_paths"] = [resolved]

    result = build_product_matrix(**inputs)

    frozen = next(row for row in result["cases"] if row["id"] == "long-held")
    assert frozen["evaluationSplit"] == "held-out"
    assert frozen["provenance"]["sourceMediaSha256"] == sha256_file(source)
    assert frozen["provenance"]["license"] == "cc-by-2.5"
    assert frozen["provenance"]["licenseEvidence"]["pageRevisionId"] == 123


def test_single_source_audio_artifact_enforces_recording_level_isolation(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    original_sha256 = "c" * 64
    manifests = []
    for split in ("development", "held-out"):
        root = tmp_path / split
        clip = _wav(root / "audio" / f"far-{split}.wav")
        manifests.append(
            _resolved_manifest(
                root / "resolved.json",
                source_id=f"far-{split}",
                cases=[
                    {
                        "id": f"far-{split}",
                        "sourceId": f"far-{split}",
                        "path": f"audio/{clip.name}",
                        "bytes": clip.stat().st_size,
                        "sha256": sha256_file(clip),
                        "audio": {"durationSeconds": 0.1},
                        "evaluationSplit": split,
                        "realOrSynthetic": "real-recording",
                        "language": "zh-CN",
                        "scenario": ["real-recording", "far-field-meeting"],
                        "sourceAudioArtifacts": [{"sha256": original_sha256}],
                    }
                ],
            )
        )
    inputs["real_diarization_manifest_paths"] = manifests

    result = build_product_matrix(**inputs)

    far_field = next(
        row
        for row in result["coverage"]
        if row["dimension"] == "scenario" and row["bucket"] == "far-field"
    )
    assert far_field["complete"] is False
    assert far_field["recordingIsolation"] == {
        "method": "source-media-sha256-when-available-else-media-sha256",
        "crossSplitDisjoint": False,
        "duplicateSha256": [original_sha256],
    }
    frozen = [
        row for row in result["cases"] if str(row["id"]).startswith("far-")
    ]
    assert {row["provenance"]["sourceMediaSha256"] for row in frozen} == {
        original_sha256
    }
    assert {row["isolation"]["recordingIdentity"] for row in frozen} == {
        "source-media-sha256"
    }


def test_long_media_rejects_missing_evidence_and_same_recording_cross_split(
    tmp_path: Path,
) -> None:
    inputs = _base_inputs(tmp_path)
    long_root = tmp_path / "long"
    source = long_root / "source.ogg"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"one recording cannot satisfy both splits")
    frozen_source = _self_contained_long_media_source(long_root, source=source)
    clips = {
        split: _wav(long_root / "audio" / f"long-{split}.wav")
        for split in ("development", "held-out")
    }
    rows = [
        {
            "id": f"long-{split}",
            "sourceId": "long-source",
            "path": f"audio/{clips[split].name}",
            "bytes": clips[split].stat().st_size,
            "sha256": sha256_file(clips[split]),
            "audio": {"durationSeconds": 0.1},
            "evaluationSplit": split,
            "realOrSynthetic": "real-recording",
            "scenario": ["real-recording", "long-media-stratified"],
        }
        for split in ("development", "held-out")
    ]
    resolved = _write_json(
        long_root / "resolved.json",
        {"schemaVersion": "1.0.0", "sources": [frozen_source], "cases": rows},
    )
    inputs["long_media_manifest_paths"] = [resolved]

    result = build_product_matrix(**inputs)

    coverage = next(
        row
        for row in result["coverage"]
        if row["dimension"] == "scenario" and row["bucket"] == "long-media"
    )
    assert coverage["complete"] is False
    assert coverage["recordingIsolation"]["crossSplitDisjoint"] is False
    assert coverage["recordingIsolation"]["duplicateSha256"] == [sha256_file(source)]

    del frozen_source["licenseEvidence"]
    _write_json(
        resolved,
        {"schemaVersion": "1.0.0", "sources": [frozen_source], "cases": rows},
    )
    with pytest.raises(ProductMatrixFreezeError, match="source evidence"):
        build_product_matrix(**inputs)
