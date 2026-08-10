from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.build_ascend_heldout_product_manifest import (
    MANIFEST_NAME,
    AscendHeldOutProductStageError,
    build_stage,
)
from tools.freeze_ascend_code_switch_samples import assert_truth_redacted


def _wav(path: Path, *, frames: int) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * frames)
    return {
        "path": path.parent.name + "/" + path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "durationSeconds": round(frames / 16_000, 6),
        "sampleRateHz": 16_000,
        "channels": 1,
        "sampleWidthBytes": 2,
        "frameCount": frames,
    }


def _public_freeze(root: Path) -> Path:
    media_root = root / "media"
    cases: list[dict[str, object]] = []
    for split, source_split, prefix, speaker_base in (
        ("development", "validation", "dev", 10),
        ("held-out", "test", "held", 20),
    ):
        for index in range(6):
            case_id = f"ascend-{prefix}-{index}"
            media = _wav(
                media_root / f"{case_id}.wav",
                frames=1600 + index,
            )
            cases.append(
                {
                    "id": case_id,
                    "sourceId": "caire-ascend",
                    "evaluationSplit": split,
                    "languageTags": ["zh", "en"],
                    "sourceLanguageLabel": "mixed",
                    "region": "East Asia",
                    "scenarios": [
                        "real-recording",
                        "single-speaker",
                        "zh-en",
                        "intra-utterance-code-switching",
                    ],
                    "expectedSpeakerCount": 1,
                    "topic": ["sports", "education", "technology"][index % 3],
                    "durationBucket": ["short", "medium", "long"][index % 3],
                    "sourceKey": {
                        "revision": "737e9800ae31be9932ba8464c80366559bd28424",
                        "config": "main",
                        "split": source_split,
                        "id": str(index),
                        "rowIndex": index,
                        "speaker": speaker_base + (index % 2),
                        "session": index % 3,
                    },
                    "stableKeySha256": "0" * 64,
                    "media": media,
                    "tuningEligible": split == "development",
                    "truthAccess": (
                        "development-reference"
                        if split == "development"
                        else "isolated-scorer-vault-only"
                    ),
                }
            )
    attribution = root / "ATTRIBUTION.md"
    attribution.write_text("ASCEND fixture attribution\n", encoding="utf-8")
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "ascend-code-switch-truth-redacted-freeze",
        "libraryId": "mts-caire-ascend-code-switch-v1",
        "generatedAt": "2026-08-09T00:00:00+00:00",
        "sourceSelectionLock": {"path": "selection.json", "fileSha256": "0" * 64},
        "source": {
            "dataset": "CAiRE/ASCEND",
            "revision": "737e9800ae31be9932ba8464c80366559bd28424",
            "config": "main",
            "license": "cc-by-sa-4.0",
            "homepage": "https://huggingface.co/datasets/CAiRE/ASCEND",
            "attributionPath": "ascend-code-switch/ATTRIBUTION.md",
        },
        "viewerEvidence": {},
        "selectionPolicy": {},
        "truthPersistencePolicy": {
            "ordinaryManifestContainsTranscript": False,
            "developmentReferencePersistedSeparately": True,
            "developmentReferenceMayEnterReviewerPacket": False,
            "heldOutTruthPersistedOnlyInIsolatedScorerVault": True,
            "heldOutTruthInOrdinaryManifest": False,
            "heldOutTruthInReviewerPacket": False,
            "scorerVaultPublishedBeforeOrdinaryManifest": True,
        },
        "developmentReference": {
            "path": "reference/development-reference.v1.json",
            "fileSha256": "1" * 64,
            "canonicalSha256": "2" * 64,
            "reviewerPacketEligible": False,
        },
        "attribution": {
            "path": "ATTRIBUTION.md",
            "fileSha256": sha256_file(attribution),
        },
        "coverage": {"cases": 12},
        "cases": cases,
        "publication": {
            "policy": "atomic-directory-no-replace",
            "manifestWrittenLast": True,
        },
    }
    manifest = {**body, "canonicalSha256": canonical_json_sha256(body)}
    path = root / "ascend-code-switch-frozen.v1.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _rewrite_canonical(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value.pop("canonicalSha256", None)
    value["canonicalSha256"] = canonical_json_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_stage_contains_only_truth_redacted_held_out_cases(tmp_path: Path) -> None:
    public = _public_freeze(tmp_path / "public")
    stage = tmp_path / "allowed-input" / "ascend-heldout"

    result = build_stage(public_manifest=public, stage_root=stage)

    manifest_path = stage / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["manifest"] == str(manifest_path.resolve())
    assert result["caseCount"] == 6
    assert result["scorerVaultRead"] is False
    assert manifest["selection"] == {
        "evaluationSplit": "held-out",
        "sourceSplit": "test",
        "tuningEligible": False,
        "filterPolicy": "public-truth-redacted-manifest-only",
        "scorerVaultRead": False,
        "speakerCountMode": "auto",
    }
    assert {row["evaluationSplit"] for row in manifest["cases"]} == {"held-out"}
    assert {row["sourceKey"]["split"] for row in manifest["cases"]} == {"test"}
    assert not any(path.name.startswith("ascend-dev") for path in stage.rglob("*.wav"))
    assert len(list((stage / "media").glob("*.wav"))) == 6
    assert not (stage / "reference").exists()
    assert_truth_redacted(manifest)
    body = dict(manifest)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
    for row in manifest["cases"]:
        media = stage / row["media"]["path"]
        assert media.stat().st_size == row["media"]["bytes"]
        assert sha256_file(media) == row["media"]["sha256"]


def test_stage_rejects_recanonicalized_truth_field(tmp_path: Path) -> None:
    public = _public_freeze(tmp_path / "public")
    _rewrite_canonical(
        public,
        lambda value: value["cases"][6].update(
            {"referenceTranscript": "sealed held-out truth"}
        ),
    )

    with pytest.raises(AscendHeldOutProductStageError, match="transcript fields"):
        build_stage(public_manifest=public, stage_root=tmp_path / "stage")


def test_stage_rejects_manifest_or_media_integrity_tampering(tmp_path: Path) -> None:
    public = _public_freeze(tmp_path / "public")
    value = json.loads(public.read_text(encoding="utf-8"))
    value["libraryId"] = "tampered"
    public.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(AscendHeldOutProductStageError, match="libraryId"):
        build_stage(public_manifest=public, stage_root=tmp_path / "stage-a")

    public = _public_freeze(tmp_path / "public-b")
    media = tmp_path / "public-b" / "media" / "ascend-held-0.wav"
    media.write_bytes(media.read_bytes() + b"tamper")
    with pytest.raises(AscendHeldOutProductStageError, match="byte count"):
        build_stage(public_manifest=public, stage_root=tmp_path / "stage-b")


def test_stage_publication_is_no_replace(tmp_path: Path) -> None:
    public = _public_freeze(tmp_path / "public")
    stage = tmp_path / "stage"
    first = build_stage(public_manifest=public, stage_root=stage)
    first_bytes = (stage / MANIFEST_NAME).read_bytes()

    with pytest.raises(FileExistsError):
        build_stage(public_manifest=public, stage_root=stage)

    assert (stage / MANIFEST_NAME).read_bytes() == first_bytes
    assert sha256_file(stage / MANIFEST_NAME) == first["manifestFileSha256"]


def test_stage_rejects_public_media_path_escape(tmp_path: Path) -> None:
    public = _public_freeze(tmp_path / "public")
    _rewrite_canonical(
        public,
        lambda value: value["cases"][6]["media"].update(
            {"path": "../outside.wav"}
        ),
    )

    with pytest.raises(AscendHeldOutProductStageError, match="inside"):
        build_stage(public_manifest=public, stage_root=tmp_path / "stage")
