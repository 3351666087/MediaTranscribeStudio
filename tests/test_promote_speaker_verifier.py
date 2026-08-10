from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from backend.production_config import ProductionConfig
from tools.promote_speaker_verifier import (
    DECISION_ARTIFACT_TYPE,
    DEPLOYMENT_SLOT,
    SpeakerVerifierPromotionError,
    _promotion_lock,
    promote_speaker_verifier,
)


ADAPTER_ID = "modelscope-eres2netv2"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _model(tmp_path: Path, model_id: str, *, status: str) -> dict[str, object]:
    root = tmp_path / model_id
    root.mkdir()
    weights = root / "model.pt"
    weights.write_bytes(model_id.encode("utf-8"))
    readme = root / "README.md"
    readme.write_text("Apache-2.0\n", encoding="utf-8")
    model_key = "speakerVerifier" + model_id.replace("-", "").title()
    manifest = {
        "schemaVersion": "1.0.0",
        "provider": "modelscope",
        "modelKey": model_key,
        "repoId": f"fixture/{model_id}",
        "revision": "v1.0.0",
        "totalBytes": weights.stat().st_size + readme.stat().st_size,
        "files": [
            {
                "path": "model.pt",
                "size": weights.stat().st_size,
                "sha256": sha256_file(weights),
            },
            {
                "path": "README.md",
                "size": readme.stat().st_size,
                "sha256": sha256_file(readme),
            },
        ],
    }
    manifest_path = root / ".mts-model-manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "id": model_id,
        "displayName": model_id,
        "source": {
            "provider": "modelscope",
            "repository": f"fixture/{model_id}",
            "revision": "v1.0.0",
        },
        "license": {
            "spdx": "Apache-2.0",
            "evidencePath": "README.md",
            "evidenceSha256": sha256_file(readme),
        },
        "usage": {
            "status": status,
            "roles": ["secondary-speaker-verification"],
            "decisionPolicy": "challenger-only",
        },
        "deploymentSlots": [
            {"id": DEPLOYMENT_SLOT, "adapterIds": [ADAPTER_ID]}
        ],
        "runtime": {
            "engine": "modelscope",
            "version": "1.0.0",
            "executablePath": (tmp_path / "python.exe").as_posix(),
            "accelerator": "gpu",
            "artifactFormat": "pytorch-checkpoint",
        },
        "quantization": {
            "scheme": "not-declared",
            "evidencePath": ".mts-model-manifest.json",
        },
        "hardwareProfileIds": ["fixture-hardware"],
        "local": {
            "path": root.as_posix(),
            "manifest": {
                "format": "mts-model-manifest-v1",
                "path": ".mts-model-manifest.json",
                "sha256": sha256_file(manifest_path),
                "fileCount": 2,
                "totalBytes": manifest["totalBytes"],
            },
        },
    }


def _registry(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    (tmp_path / "python.exe").write_bytes(b"runtime")
    models = {
        "speaker-a": _model(tmp_path, "speaker-a", status="production"),
        "speaker-b": _model(tmp_path, "speaker-b", status="challenger"),
    }
    registry = {
        "schemaVersion": "1.0.0",
        "registryId": "fixture.registry",
        "hardwareProfiles": [
            {
                "id": "fixture-hardware",
                "platform": "test-x86-64",
                "cpu": {
                    "model": "fixture",
                    "physicalCores": 4,
                    "logicalProcessors": 8,
                },
                "ramBytes": 16 * 1024**3,
                "gpu": {
                    "model": "fixture",
                    "vramBytes": 8 * 1024**3,
                    "computeCapability": "8.6",
                    "driverVersion": "1.0",
                },
            }
        ],
        "models": list(models.values()),
    }
    path = tmp_path / "registry.json"
    _write_json(path, registry)
    return path, models


def _binding(model: dict[str, object]) -> dict[str, object]:
    local = model["local"]
    assert isinstance(local, dict)
    manifest_identity = local["manifest"]
    assert isinstance(manifest_identity, dict)
    manifest = json.loads(
        (Path(str(local["path"])) / str(manifest_identity["path"])).read_text(
            encoding="utf-8"
        )
    )
    return {
        "path": local["path"],
        "deploymentSlot": DEPLOYMENT_SLOT,
        "registryModelId": model["id"],
        "manifestModelKey": manifest["modelKey"],
        "manifestSha256": "sha256:" + str(manifest_identity["sha256"]),
        "adapterId": ADAPTER_ID,
    }


def _config(tmp_path: Path, model: dict[str, object]) -> Path:
    path = tmp_path / "production.json"
    value = {
        "schemaVersion": "1.0.0",
        "mode": "offline-production",
        "offline": True,
        "paths": {
            "allowedInputRoots": [str(tmp_path / "input")],
            "allowedOutputRoot": str(tmp_path / "output"),
            "cacheRoot": str(tmp_path / "cache"),
        },
        "models": {
            "funasrVad": str(tmp_path / "vad"),
            "qwen3Asr": str(tmp_path / "asr"),
            "qwen3ForcedAligner": None,
            "camPlus": str(tmp_path / "cam"),
            "secondarySpeakerVerifier": _binding(model),
            "pyannote": None,
            "mossformer2Separation": None,
        },
        "executables": {
            "ffmpeg": "ffmpeg",
            "java": "java",
            "pdfRendererJar": str(tmp_path / "renderer.jar"),
            "pyannotePython": None,
        },
        "runtime": {"strictStartupPreflight": False},
        "speaker": {
            "pyannoteMode": "disabled",
            "overlapRecoveryMode": "disabled",
            "localLlmModel": "qwen3.5:9b",
            "localLlmModelDigest": "sha256:" + "a" * 64,
        },
        "pdf": {},
    }
    _write_json(path, value)
    ProductionConfig.load(path)
    return path


def _decision(
    path: Path,
    *,
    decision_id: str,
    expected_config_sha256: str,
    incumbent: str,
    challenger: str,
    audit_source: str = "human",
    automatic_scoring: bool = False,
) -> tuple[Path, Path]:
    blind_review = path.with_suffix(".blind-review")
    comparison = path.with_suffix(".comparison")
    blind_review.write_bytes(f"blind:{decision_id}".encode("utf-8"))
    comparison.write_bytes(f"comparison:{decision_id}".encode("utf-8"))
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": DECISION_ARTIFACT_TYPE,
        "decisionId": decision_id,
        "deploymentSlot": DEPLOYMENT_SLOT,
        "automaticScoring": automatic_scoring,
        "expectedConfigSha256": expected_config_sha256,
        "incumbentRegistryModelId": incumbent,
        "challengerRegistryModelId": challenger,
        "adapterId": ADAPTER_ID,
        "outcome": "promote-challenger",
        "evidence": {
            "blindReviewArtifactSha256": sha256_file(blind_review),
            "comparisonArtifactSha256": sha256_file(comparison),
        },
        "audit": {
            "source": audit_source,
            "reviewer": "reviewer-1",
            "decidedAt": "2026-08-07T12:00:00+08:00",
        },
    }
    body["canonicalSha256"] = canonical_json_sha256(body)
    _write_json(path, body)
    return blind_review, comparison


def test_challenger_can_replace_active_and_be_replaced_back(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["speaker-a"])
    original_fingerprint = ProductionConfig.load(config_path).fingerprint()
    first_sha = sha256_file(config_path)
    first_decision = tmp_path / "a-to-b.json"
    first_blind, first_comparison = _decision(
        first_decision,
        decision_id="decision-a-to-b",
        expected_config_sha256=first_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
    )

    first = promote_speaker_verifier(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=first_decision,
        blind_review_artifact_path=first_blind,
        comparison_artifact_path=first_comparison,
        rollback_root=tmp_path / "rollback",
        expected_config_sha256=first_sha,
    )

    assert first["activeRegistryModelId"] == "speaker-b"
    assert first["auditSource"] == "human"
    assert first["promotionPolicy"] == {
        "registryStatusUsedAsGate": False,
        "incumbentMarginRequired": False,
        "protectionPeriodApplied": False,
        "immediateReplacementAllowed": True,
    }
    active = ProductionConfig.load(config_path)
    assert active.models.secondary_speaker_verifier.registry_model_id == "speaker-b"
    assert active.models.secondary_speaker_verifier.promotion_evidence is not None
    assert active.fingerprint() != original_fingerprint
    assert sha256_file(Path(first["rollbackConfigPath"])) == first_sha
    second_sha = sha256_file(config_path)
    second_decision = tmp_path / "b-to-a.json"
    second_blind, second_comparison = _decision(
        second_decision,
        decision_id="decision-b-to-a",
        expected_config_sha256=second_sha,
        incumbent="speaker-b",
        challenger="speaker-a",
    )

    second = promote_speaker_verifier(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=second_decision,
        blind_review_artifact_path=second_blind,
        comparison_artifact_path=second_comparison,
        rollback_root=tmp_path / "rollback",
        expected_config_sha256=second_sha,
    )

    assert second["activeRegistryModelId"] == "speaker-a"
    rebound = ProductionConfig.load(config_path)
    assert rebound.models.secondary_speaker_verifier.registry_model_id == "speaker-a"
    assert rebound.models.secondary_speaker_verifier.promotion_evidence is not None


def test_codex_manual_blind_review_can_promote_challenger(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "codex-manual.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-codex-manual",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
        audit_source="codex-manual",
    )

    receipt = promote_speaker_verifier(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review_artifact_path=blind_review,
        comparison_artifact_path=comparison,
        rollback_root=tmp_path / "rollback",
        expected_config_sha256=initial_sha,
    )

    assert receipt["activeRegistryModelId"] == "speaker-b"
    assert receipt["auditSource"] == "codex-manual"


@pytest.mark.parametrize(
    "registry_status",
    ["production", "challenger", "development", "research", "retired"],
)
def test_registry_status_is_not_a_promotion_gate(
    tmp_path: Path,
    registry_status: str,
) -> None:
    registry_path, models = _registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["models"][1]["usage"]["status"] = registry_status
    _write_json(registry_path, registry)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "status-independent.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id=f"decision-status-{registry_status}",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
        audit_source="codex-manual",
    )

    receipt = promote_speaker_verifier(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review_artifact_path=blind_review,
        comparison_artifact_path=comparison,
        rollback_root=tmp_path / "rollback",
        expected_config_sha256=initial_sha,
    )

    assert receipt["activeRegistryModelId"] == "speaker-b"


@pytest.mark.parametrize(
    ("audit_source", "automatic_scoring", "message"),
    [
        ("model-self-review", False, "manual reviewer"),
        ("codex-manual", True, "manual blind-review"),
    ],
)
def test_model_self_review_or_automatic_scoring_cannot_promote(
    tmp_path: Path,
    audit_source: str,
    automatic_scoring: bool,
    message: str,
) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "invalid-review.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-invalid-review",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
        audit_source=audit_source,
        automatic_scoring=automatic_scoring,
    )

    with pytest.raises(SpeakerVerifierPromotionError, match=message):
        promote_speaker_verifier(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review_artifact_path=blind_review,
            comparison_artifact_path=comparison,
            rollback_root=tmp_path / "rollback",
            expected_config_sha256=initial_sha,
        )

    assert sha256_file(config_path) == initial_sha
    assert not (tmp_path / "rollback").exists()


def test_stale_or_duplicate_decision_fails_without_mutation(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-once",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
    )

    with pytest.raises(SpeakerVerifierPromotionError, match="stale"):
        promote_speaker_verifier(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review_artifact_path=blind_review,
            comparison_artifact_path=comparison,
            rollback_root=tmp_path / "rollback",
            expected_config_sha256="0" * 64,
        )
    assert sha256_file(config_path) == initial_sha

    promote_speaker_verifier(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review_artifact_path=blind_review,
        comparison_artifact_path=comparison,
        rollback_root=tmp_path / "rollback",
        expected_config_sha256=initial_sha,
    )
    promoted_sha = sha256_file(config_path)
    with pytest.raises(SpeakerVerifierPromotionError, match="stale"):
        promote_speaker_verifier(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review_artifact_path=blind_review,
            comparison_artifact_path=comparison,
            rollback_root=tmp_path / "rollback",
            expected_config_sha256=initial_sha,
        )
    assert sha256_file(config_path) == promoted_sha


def test_interruption_before_replace_preserves_config(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-interrupted",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
    )

    def interrupt(_temporary: Path) -> None:
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        promote_speaker_verifier(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review_artifact_path=blind_review,
            comparison_artifact_path=comparison,
            rollback_root=tmp_path / "rollback",
            expected_config_sha256=initial_sha,
            _before_replace=interrupt,
        )

    assert sha256_file(config_path) == initial_sha
    assert not list(tmp_path.glob("*.promotion.tmp"))
    rollback = next((tmp_path / "rollback").glob("*.rollback.json"))
    assert sha256_file(rollback) == initial_sha

    promote_speaker_verifier(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review_artifact_path=blind_review,
        comparison_artifact_path=comparison,
        rollback_root=tmp_path / "rollback",
        expected_config_sha256=initial_sha,
    )
    assert ProductionConfig.load(
        config_path
    ).models.secondary_speaker_verifier.registry_model_id == "speaker-b"


def test_registry_slot_adapter_compatibility_is_required(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["models"][1]["deploymentSlots"][0]["adapterIds"] = [
        "other-adapter"
    ]
    _write_json(registry_path, registry)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-incompatible",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
    )

    with pytest.raises(SpeakerVerifierPromotionError, match="incompatible"):
        promote_speaker_verifier(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review_artifact_path=blind_review,
            comparison_artifact_path=comparison,
            rollback_root=tmp_path / "rollback",
            expected_config_sha256=initial_sha,
        )

    assert sha256_file(config_path) == initial_sha


def test_artifact_hash_mismatch_fails_before_snapshot(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-artifact-mismatch",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
    )
    blind_review.write_bytes(b"changed after review")

    with pytest.raises(SpeakerVerifierPromotionError, match="SHA-256"):
        promote_speaker_verifier(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review_artifact_path=blind_review,
            comparison_artifact_path=comparison,
            rollback_root=tmp_path / "rollback",
            expected_config_sha256=initial_sha,
        )

    assert sha256_file(config_path) == initial_sha
    assert not (tmp_path / "rollback").exists()


def test_live_lock_blocks_but_stale_lock_file_does_not(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["speaker-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-lock",
        expected_config_sha256=initial_sha,
        incumbent="speaker-a",
        challenger="speaker-b",
    )
    lock_path = config_path.with_name(f".{config_path.name}.promotion.lock")
    lock_path.write_text("stale pid metadata", encoding="utf-8")
    arguments = {
        "config_path": config_path,
        "registry_path": registry_path,
        "decision_path": decision_path,
        "blind_review_artifact_path": blind_review,
        "comparison_artifact_path": comparison,
        "rollback_root": tmp_path / "rollback",
        "expected_config_sha256": initial_sha,
    }

    with _promotion_lock(lock_path):
        with pytest.raises(
            SpeakerVerifierPromotionError,
            match="in progress",
        ):
            promote_speaker_verifier(**arguments)

    receipt = promote_speaker_verifier(**arguments)
    assert receipt["activeRegistryModelId"] == "speaker-b"
