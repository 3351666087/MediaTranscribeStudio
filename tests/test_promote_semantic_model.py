from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from backend.production_config import ProductionConfig
from tools.promote_semantic_model import (
    ADAPTER_ID,
    DECISION_ARTIFACT_TYPE,
    DEPLOYMENT_SLOT,
    REVIEW_KIND,
    SemanticModelPromotionError,
    promote_semantic_model,
)
from tools.rollback_semantic_model import (
    ROLLBACK_RECEIPT_ARTIFACT_TYPE,
    SemanticModelRollbackError,
    rollback_semantic_model,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _blob(root: Path, payload: bytes) -> tuple[str, int]:
    digest = hashlib.sha256(payload).hexdigest()
    path = root / "blobs" / f"sha256-{digest}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest, len(payload)


def _model(
    root: Path,
    runtime: Path,
    *,
    model_id: str,
    repository: str,
    tag: str,
    status: str = "challenger",
) -> dict[str, object]:
    config_digest, config_size = _blob(
        root,
        json.dumps(
            {"model_format": "gguf", "file_type": "Q4_K_M"},
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    model_digest, model_size = _blob(
        root,
        f"weights:{model_id}".encode("utf-8"),
    )
    license_digest, license_size = _blob(
        root,
        f"Apache-2.0:{model_id}".encode("utf-8"),
    )
    params_digest, params_size = _blob(
        root,
        f"num_ctx 32768:{model_id}".encode("utf-8"),
    )
    manifest = {
        "schemaVersion": 2,
        "mediaType": (
            "application/vnd.docker.distribution.manifest.v2+json"
        ),
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": f"sha256:{config_digest}",
            "size": config_size,
        },
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{model_digest}",
                "size": model_size,
            },
            {
                "mediaType": "application/vnd.ollama.image.license",
                "digest": f"sha256:{license_digest}",
                "size": license_size,
            },
            {
                "mediaType": "application/vnd.ollama.image.params",
                "digest": f"sha256:{params_digest}",
                "size": params_size,
            },
        ],
    }
    manifest_path = (
        root
        / "manifests"
        / "registry.ollama.ai"
        / repository
        / tag
    )
    _write_json(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    total_bytes = config_size + model_size + license_size + params_size
    return {
        "id": model_id,
        "displayName": model_id,
        "source": {
            "provider": "ollama",
            "repository": repository,
            "tag": tag,
            "digest": f"sha256:{manifest_sha256}",
        },
        "license": {
            "spdx": "Apache-2.0",
            "evidencePath": f"blobs/sha256-{license_digest}",
            "evidenceSha256": license_digest,
        },
        "usage": {
            "status": status,
            "roles": [DEPLOYMENT_SLOT],
            "decisionPolicy": "suggestion-only",
        },
        "runtime": {
            "engine": "ollama",
            "version": "1.0.0",
            "executablePath": runtime.as_posix(),
            "accelerator": "cuda",
            "artifactFormat": "gguf",
        },
        "quantization": {
            "scheme": "Q4_K_M",
            "evidencePath": f"blobs/sha256-{config_digest}",
        },
        "hardwareProfileIds": ["fixture-hardware"],
        "local": {
            "path": root.as_posix(),
            "manifest": {
                "format": "ollama-oci-manifest-v2",
                "path": (
                    "manifests/registry.ollama.ai/"
                    f"{repository}/{tag}"
                ),
                "sha256": manifest_sha256,
                "fileCount": 4,
                "totalBytes": total_bytes,
            },
        },
    }


def _registry(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    root = tmp_path / "ollama"
    runtime = tmp_path / "ollama.exe"
    runtime.write_bytes(b"fixture runtime")
    models = {
        "semantic-a": _model(
            root,
            runtime,
            model_id="semantic-a",
            repository="library/semantic-a",
            tag="9b",
            status="production",
        ),
        "semantic-b": _model(
            root,
            runtime,
            model_id="semantic-b",
            repository="library/semantic-b",
            tag="27b-q4_K_M",
        ),
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


def _binding(model: dict[str, object]) -> tuple[str, str]:
    source = model["source"]
    assert isinstance(source, dict)
    repository = str(source["repository"])
    return (
        f"{repository.removeprefix('library/')}:{source['tag']}",
        str(source["digest"]),
    )


def _config(tmp_path: Path, model: dict[str, object]) -> Path:
    model_name, model_digest = _binding(model)
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
            "secondarySpeakerVerifier": {
                "path": str(tmp_path / "speaker"),
                "deploymentSlot": "secondary-speaker-verification",
                "registryModelId": "speaker-fixture",
                "manifestModelKey": "speakerFixture",
                "manifestSha256": "sha256:" + "d" * 64,
                "adapterId": "modelscope-eres2netv2",
            },
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
            "localLlmModel": model_name,
            "localLlmModelDigest": model_digest,
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
    same_case_batch: bool = True,
    identities_hidden: bool = True,
    winner: str | None = None,
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
        "adapterId": ADAPTER_ID,
        "automaticScoring": False,
        "expectedConfigSha256": expected_config_sha256,
        "incumbentRegistryModelId": incumbent,
        "challengerRegistryModelId": challenger,
        "outcome": "promote-challenger",
        "blindReview": {
            "kind": REVIEW_KIND,
            "blindBatchId": f"batch-{decision_id}",
            "caseSetSha256": "c" * 64,
            "sameCaseBatch": same_case_batch,
            "candidateIdentitiesHidden": identities_hidden,
            "winnerRegistryModelId": winner or challenger,
        },
        "evidence": {
            "blindReviewArtifactSha256": sha256_file(blind_review),
            "comparisonArtifactSha256": sha256_file(comparison),
        },
        "audit": {
            "source": "codex-agent",
            "reviewer": "Codex semantic reviewer",
            "decidedAt": "2026-08-09T12:00:00+08:00",
        },
    }
    body["canonicalSha256"] = canonical_json_sha256(body)
    _write_json(path, body)
    return blind_review, comparison


def _promote(
    *,
    config_path: Path,
    registry_path: Path,
    decision_path: Path,
    blind_review: Path,
    comparison: Path,
    rollback_root: Path,
    expected_sha256: str,
    **kwargs: object,
) -> dict[str, object]:
    return promote_semantic_model(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review_artifact_path=blind_review,
        comparison_artifact_path=comparison,
        rollback_root=rollback_root,
        expected_config_sha256=expected_sha256,
        **kwargs,
    )


def test_challenger_can_replace_incumbent_and_be_replaced_back_immediately(
    tmp_path: Path,
) -> None:
    registry_path, models = _registry(tmp_path)
    assert models["semantic-a"]["usage"]["status"] == "production"
    assert models["semantic-b"]["usage"]["status"] == "challenger"
    config_path = _config(tmp_path, models["semantic-a"])
    original_fingerprint = ProductionConfig.load(config_path).fingerprint()
    first_sha = sha256_file(config_path)
    first_decision = tmp_path / "a-to-b.json"
    first_blind, first_comparison = _decision(
        first_decision,
        decision_id="decision-a-to-b",
        expected_config_sha256=first_sha,
        incumbent="semantic-a",
        challenger="semantic-b",
    )

    first = _promote(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=first_decision,
        blind_review=first_blind,
        comparison=first_comparison,
        rollback_root=tmp_path / "rollback",
        expected_sha256=first_sha,
    )

    assert first["activeRegistryModelId"] == "semantic-b"
    assert first["promotionPolicy"] == {
        "registryStatusUsedAsGate": False,
        "incumbentMarginRequired": False,
        "protectionPeriodApplied": False,
        "immediateReplacementAllowed": True,
    }
    assert sha256_file(Path(str(first["rollbackConfigPath"]))) == first_sha
    active = ProductionConfig.load(config_path)
    assert active.speaker.local_llm_model == "semantic-b:27b-q4_K_M"
    assert active.fingerprint() != original_fingerprint

    second_sha = sha256_file(config_path)
    second_decision = tmp_path / "b-to-a.json"
    second_blind, second_comparison = _decision(
        second_decision,
        decision_id="decision-b-to-a",
        expected_config_sha256=second_sha,
        incumbent="semantic-b",
        challenger="semantic-a",
    )
    second = _promote(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=second_decision,
        blind_review=second_blind,
        comparison=second_comparison,
        rollback_root=tmp_path / "rollback",
        expected_sha256=second_sha,
    )

    assert second["activeRegistryModelId"] == "semantic-a"
    rebound = ProductionConfig.load(config_path)
    assert rebound.speaker.local_llm_model == "semantic-a:9b"


def test_stale_or_reused_decision_fails_without_mutation(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["semantic-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-once",
        expected_config_sha256=initial_sha,
        incumbent="semantic-a",
        challenger="semantic-b",
    )

    with pytest.raises(SemanticModelPromotionError, match="stale"):
        _promote(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review=blind_review,
            comparison=comparison,
            rollback_root=tmp_path / "rollback",
            expected_sha256="0" * 64,
        )
    assert sha256_file(config_path) == initial_sha

    _promote(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review=blind_review,
        comparison=comparison,
        rollback_root=tmp_path / "rollback",
        expected_sha256=initial_sha,
    )
    promoted_sha = sha256_file(config_path)
    with pytest.raises(SemanticModelPromotionError, match="stale"):
        _promote(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review=blind_review,
            comparison=comparison,
            rollback_root=tmp_path / "rollback",
            expected_sha256=initial_sha,
        )
    assert sha256_file(config_path) == promoted_sha


def test_interruption_before_replace_preserves_active_config(tmp_path: Path) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["semantic-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-interrupted",
        expected_config_sha256=initial_sha,
        incumbent="semantic-a",
        challenger="semantic-b",
    )

    def interrupt(_temporary: Path) -> None:
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _promote(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review=blind_review,
            comparison=comparison,
            rollback_root=tmp_path / "rollback",
            expected_sha256=initial_sha,
            _before_replace=interrupt,
        )

    assert sha256_file(config_path) == initial_sha
    assert not list(tmp_path.glob("*.promotion.tmp"))
    rollback = next((tmp_path / "rollback").glob("*.rollback.json"))
    assert sha256_file(rollback) == initial_sha


@pytest.mark.parametrize(
    ("same_case_batch", "identities_hidden", "winner", "message"),
    [
        (False, True, None, "same review case batch"),
        (True, False, None, "identities must remain hidden"),
        (True, True, "semantic-a", "winner does not match"),
    ],
)
def test_decision_must_prove_same_batch_blind_challenger_win(
    tmp_path: Path,
    same_case_batch: bool,
    identities_hidden: bool,
    winner: str | None,
    message: str,
) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["semantic-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-invalid-review",
        expected_config_sha256=initial_sha,
        incumbent="semantic-a",
        challenger="semantic-b",
        same_case_batch=same_case_batch,
        identities_hidden=identities_hidden,
        winner=winner,
    )

    with pytest.raises(SemanticModelPromotionError, match=message):
        _promote(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review=blind_review,
            comparison=comparison,
            rollback_root=tmp_path / "rollback",
            expected_sha256=initial_sha,
        )
    assert sha256_file(config_path) == initial_sha


def test_evidence_hash_mismatch_fails_without_rollback_or_mutation(
    tmp_path: Path,
) -> None:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["semantic-a"])
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-evidence",
        expected_config_sha256=initial_sha,
        incumbent="semantic-a",
        challenger="semantic-b",
    )
    comparison.write_bytes(b"tampered")

    with pytest.raises(SemanticModelPromotionError, match="SHA-256"):
        _promote(
            config_path=config_path,
            registry_path=registry_path,
            decision_path=decision_path,
            blind_review=blind_review,
            comparison=comparison,
            rollback_root=tmp_path / "rollback",
            expected_sha256=initial_sha,
        )
    assert sha256_file(config_path) == initial_sha
    assert not (tmp_path / "rollback").exists()


def test_direct_cli_help_resolves_repository_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "tools"
                / "promote_semantic_model.py"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-config-sha256" in completed.stdout


def _promoted_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, dict[str, object]], dict[str, object], bytes]:
    registry_path, models = _registry(tmp_path)
    config_path = _config(tmp_path, models["semantic-a"])
    original = config_path.read_bytes()
    initial_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision-rollback.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-rollback",
        expected_config_sha256=initial_sha,
        incumbent="semantic-a",
        challenger="semantic-b",
    )
    promotion = _promote(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review=blind_review,
        comparison=comparison,
        rollback_root=tmp_path / "rollback",
        expected_sha256=initial_sha,
    )
    receipt_path = tmp_path / "promotion-receipt.json"
    _write_json(receipt_path, promotion)
    return config_path, registry_path, models, promotion, original


def test_receipt_bound_rollback_restores_incumbent_and_preserves_failed_config(
    tmp_path: Path,
) -> None:
    config_path, registry_path, _, promotion, original = _promoted_fixture(
        tmp_path
    )
    promoted = config_path.read_bytes()
    promoted_sha = sha256_file(config_path)
    receipt_path = tmp_path / "promotion-receipt.json"

    recovery = rollback_semantic_model(
        config_path=config_path,
        registry_path=registry_path,
        promotion_receipt_path=receipt_path,
        rollback_config_path=Path(str(promotion["rollbackConfigPath"])),
        recovery_root=tmp_path / "recovery",
        expected_active_config_sha256=promoted_sha,
        reason="operator-drill",
    )

    assert config_path.read_bytes() == original
    assert recovery["artifactType"] == ROLLBACK_RECEIPT_ARTIFACT_TYPE
    assert recovery["failedActiveRegistryModelId"] == "semantic-b"
    assert recovery["restoredRegistryModelId"] == "semantic-a"
    assert recovery["restoredConfigSha256"] == promotion["previousConfigSha256"]
    failed_snapshot = Path(str(recovery["failedActiveConfigSnapshotPath"]))
    assert failed_snapshot.read_bytes() == promoted
    assert sha256_file(failed_snapshot) == promoted_sha
    assert recovery["rollbackPolicy"] == {
        "blindReviewRequiredForEmergencyRecovery": False,
        "registryStatusUsedAsGate": False,
        "restoredModelProtectionApplied": False,
        "immediateChallengerReplacementAllowed": True,
    }
    canonical = dict(recovery)
    declared = canonical.pop("canonicalSha256")
    assert canonical_json_sha256(canonical) == declared

    restored_sha = sha256_file(config_path)
    decision_path = tmp_path / "decision-after-rollback.json"
    blind_review, comparison = _decision(
        decision_path,
        decision_id="decision-after-rollback",
        expected_config_sha256=restored_sha,
        incumbent="semantic-a",
        challenger="semantic-b",
    )
    _promote(
        config_path=config_path,
        registry_path=registry_path,
        decision_path=decision_path,
        blind_review=blind_review,
        comparison=comparison,
        rollback_root=tmp_path / "rollback-after-recovery",
        expected_sha256=restored_sha,
    )
    assert ProductionConfig.load(config_path).speaker.local_llm_model == (
        "semantic-b:27b-q4_K_M"
    )


def test_rollback_stale_cas_fails_without_mutation_or_recovery_snapshot(
    tmp_path: Path,
) -> None:
    config_path, registry_path, _, promotion, _ = _promoted_fixture(tmp_path)
    active = config_path.read_bytes()

    with pytest.raises(SemanticModelRollbackError, match="stale"):
        rollback_semantic_model(
            config_path=config_path,
            registry_path=registry_path,
            promotion_receipt_path=tmp_path / "promotion-receipt.json",
            rollback_config_path=Path(str(promotion["rollbackConfigPath"])),
            recovery_root=tmp_path / "recovery",
            expected_active_config_sha256="f" * 64,
            reason="startup-failure",
        )

    assert config_path.read_bytes() == active
    assert not (tmp_path / "recovery").exists()


def test_tampered_rollback_snapshot_fails_before_active_config_mutation(
    tmp_path: Path,
) -> None:
    config_path, registry_path, _, promotion, _ = _promoted_fixture(tmp_path)
    active = config_path.read_bytes()
    rollback_path = Path(str(promotion["rollbackConfigPath"]))
    rollback_path.write_bytes(b"tampered")

    with pytest.raises(SemanticModelRollbackError, match="SHA-256"):
        rollback_semantic_model(
            config_path=config_path,
            registry_path=registry_path,
            promotion_receipt_path=tmp_path / "promotion-receipt.json",
            rollback_config_path=rollback_path,
            recovery_root=tmp_path / "recovery",
            expected_active_config_sha256=sha256_file(config_path),
            reason="artifact-integrity-failure",
        )

    assert config_path.read_bytes() == active
    assert not (tmp_path / "recovery").exists()


def test_missing_local_rollback_model_fails_before_active_config_mutation(
    tmp_path: Path,
) -> None:
    config_path, registry_path, models, promotion, _ = _promoted_fixture(tmp_path)
    active = config_path.read_bytes()
    local = models["semantic-a"]["local"]
    assert isinstance(local, dict)
    manifest = local["manifest"]
    assert isinstance(manifest, dict)
    (Path(str(local["path"])) / str(manifest["path"])).unlink()

    with pytest.raises(SemanticModelRollbackError, match="locally available"):
        rollback_semantic_model(
            config_path=config_path,
            registry_path=registry_path,
            promotion_receipt_path=tmp_path / "promotion-receipt.json",
            rollback_config_path=Path(str(promotion["rollbackConfigPath"])),
            recovery_root=tmp_path / "recovery",
            expected_active_config_sha256=sha256_file(config_path),
            reason="active-model-unavailable",
        )

    assert config_path.read_bytes() == active
    assert not (tmp_path / "recovery").exists()


def test_interrupted_rollback_preserves_active_and_failed_config_snapshot(
    tmp_path: Path,
) -> None:
    config_path, registry_path, _, promotion, _ = _promoted_fixture(tmp_path)
    active = config_path.read_bytes()
    active_sha = sha256_file(config_path)

    def interrupt(_temporary: Path) -> None:
        raise RuntimeError("simulated interruption before rollback replace")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        rollback_semantic_model(
            config_path=config_path,
            registry_path=registry_path,
            promotion_receipt_path=tmp_path / "promotion-receipt.json",
            rollback_config_path=Path(str(promotion["rollbackConfigPath"])),
            recovery_root=tmp_path / "recovery",
            expected_active_config_sha256=active_sha,
            reason="operator-drill",
            _before_replace=interrupt,
        )

    assert config_path.read_bytes() == active
    snapshots = list((tmp_path / "recovery").glob("*.json"))
    assert len(snapshots) == 1
    assert sha256_file(snapshots[0]) == active_sha
    assert not list(tmp_path.glob("*.rollback.tmp"))


def test_semantic_rollback_cli_help_resolves_repository_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "tools"
                / "rollback_semantic_model.py"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-active-config-sha256" in completed.stdout
    assert "--reason" in completed.stdout


def test_semantic_rollback_cli_executes_windows_recovery_path(
    tmp_path: Path,
) -> None:
    config_path, registry_path, _, promotion, original = _promoted_fixture(
        tmp_path
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "tools"
                / "rollback_semantic_model.py"
            ),
            "--config",
            str(config_path),
            "--registry",
            str(registry_path),
            "--promotion-receipt",
            str(tmp_path / "promotion-receipt.json"),
            "--rollback-config",
            str(promotion["rollbackConfigPath"]),
            "--recovery-root",
            str(tmp_path / "recovery-cli"),
            "--expected-active-config-sha256",
            sha256_file(config_path),
            "--reason",
            "operator-drill",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["artifactType"] == ROLLBACK_RECEIPT_ARTIFACT_TYPE
    assert receipt["restoredRegistryModelId"] == "semantic-a"
    assert config_path.read_bytes() == original
