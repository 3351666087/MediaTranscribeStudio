from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from backend.business_processing import BusinessProcessingConfig
from backend.local_llm import LocalLLMConfig
from backend.production_config import ProductionSpeakerPolicy
from backend.speaker_pipeline import SpeakerPipelineConfig
from tools.model_registry import (
    DEFAULT_REGISTRY_PATH,
    RegistryValidationError,
    load_registry,
    main,
    validate_registry,
)


EXPECTED_MODEL_IDS = {
    "campplus",
    "eres2netv2",
    "eres2netv2-w24s4ep4",
    "funasr-fsmn-vad",
    "ollama-glm-4.7-flash-q4-k-m",
    "ollama-qwen3.5-27b-q4-k-m",
    "ollama-qwen3.5-35b-a3b-q4-k-m",
    "ollama-qwen3.5-9b",
    "ollama-qwen3.6-27b-q4-k-m",
    "ollama-qwen3.6-35b-a3b-q4-k-m",
    "pyannote-community-1",
    "qwen3-asr-1.7b",
    "qwen3-forced-aligner-0.6b",
    "wespeaker-redimnet2-b6-lm",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_CHAMPION = "qwen3.5:27b-q4_K_M"
SEMANTIC_CHAMPION_DIGEST = (
    "sha256:7653528ba5cba4dd8e19da24aaddc7f4d0b5ecd93571c0825dfd4137958ec06e"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _hardware_profile() -> dict[str, object]:
    return {
        "id": "fixture-hardware",
        "platform": "test-x86_64",
        "cpu": {
            "model": "fixture cpu",
            "physicalCores": 4,
            "logicalProcessors": 8,
        },
        "ramBytes": 16 * 1024**3,
        "gpu": {
            "model": "fixture gpu",
            "vramBytes": 8 * 1024**3,
            "computeCapability": "8.6",
            "driverVersion": "1.0",
        },
    }


def _mts_registry(tmp_path: Path) -> tuple[dict[str, object], Path]:
    root = tmp_path / "model"
    root.mkdir()
    runtime = tmp_path / "runtime" / "python.exe"
    runtime.parent.mkdir()
    runtime.write_bytes(b"fixture runtime")

    readme = b"license: apache-2.0\n"
    weights = b"model weights"
    (root / "README.md").write_bytes(readme)
    (root / "model.pt").write_bytes(weights)
    manifest = {
        "schemaVersion": "1.0.0",
        "provider": "modelscope",
        "modelKey": "fixtureModel",
        "repoId": "org/fixture-model",
        "revision": "v1.2.3",
        "totalBytes": len(readme) + len(weights),
        "files": [
            {
                "path": "README.md",
                "size": len(readme),
                "sha256": _sha256(readme),
            },
            {
                "path": "model.pt",
                "size": len(weights),
                "sha256": _sha256(weights),
            },
        ],
    }
    manifest_payload = _write_json(root / ".mts-model-manifest.json", manifest)
    model = {
        "id": "fixture-model",
        "displayName": "Fixture model",
        "source": {
            "provider": "modelscope",
            "repository": "org/fixture-model",
            "revision": "v1.2.3",
        },
        "license": {
            "spdx": "Apache-2.0",
            "evidencePath": "README.md",
            "evidenceSha256": _sha256(readme),
        },
        "usage": {
            "status": "production",
            "roles": ["fixture-inference"],
            "decisionPolicy": "direct",
        },
        "runtime": {
            "engine": "fixture",
            "version": "1.0.0",
            "executablePath": runtime.as_posix(),
            "accelerator": "cpu",
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
                "sha256": _sha256(manifest_payload),
                "fileCount": 2,
                "totalBytes": len(readme) + len(weights),
            },
        },
    }
    registry: dict[str, object] = {
        "schemaVersion": "1.0.0",
        "registryId": "fixture.registry",
        "hardwareProfiles": [_hardware_profile()],
        "models": [model],
    }
    return registry, root


def _ollama_registry(tmp_path: Path) -> tuple[dict[str, object], Path]:
    root = tmp_path / "ollama"
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    runtime = tmp_path / "ollama.exe"
    runtime.write_bytes(b"fixture runtime")

    config = _write_json(
        tmp_path / "config.json",
        {"model_format": "gguf", "file_type": "Q4_K_M"},
    )
    artifacts = [
        ("application/vnd.ollama.image.model", b"GGUF model"),
        ("application/vnd.ollama.image.license", b"Apache License 2.0"),
        ("application/vnd.ollama.image.params", b"num_ctx 32768"),
    ]
    descriptors: list[dict[str, object]] = []
    config_digest = _sha256(config)
    (blobs / f"sha256-{config_digest}").write_bytes(config)
    for media_type, payload in artifacts:
        digest = _sha256(payload)
        (blobs / f"sha256-{digest}").write_bytes(payload)
        descriptors.append(
            {
                "mediaType": media_type,
                "digest": f"sha256:{digest}",
                "size": len(payload),
            }
        )
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": f"sha256:{config_digest}",
            "size": len(config),
        },
        "layers": descriptors,
    }
    manifest_path = (
        root / "manifests/registry.ollama.ai/library/fixture/9b"
    )
    manifest_payload = _write_json(manifest_path, manifest)
    manifest_digest = _sha256(manifest_payload)
    license_digest = str(descriptors[1]["digest"]).removeprefix("sha256:")
    total_bytes = len(config) + sum(len(payload) for _, payload in artifacts)
    model = {
        "id": "ollama-fixture-9b",
        "displayName": "Ollama fixture 9B",
        "source": {
            "provider": "ollama",
            "repository": "library/fixture",
            "tag": "9b",
            "digest": f"sha256:{manifest_digest}",
        },
        "license": {
            "spdx": "Apache-2.0",
            "evidencePath": f"blobs/sha256-{license_digest}",
            "evidenceSha256": license_digest,
        },
        "usage": {
            "status": "production",
            "roles": ["semantic-arbitration"],
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
                "path": "manifests/registry.ollama.ai/library/fixture/9b",
                "sha256": manifest_digest,
                "fileCount": 4,
                "totalBytes": total_bytes,
            },
        },
    }
    registry: dict[str, object] = {
        "schemaVersion": "1.0.0",
        "registryId": "fixture.registry",
        "hardwareProfiles": [_hardware_profile()],
        "models": [model],
    }
    return registry, blobs / f"sha256-{str(descriptors[0]['digest']).removeprefix('sha256:')}"


def test_repository_registry_covers_the_installed_production_set() -> None:
    registry = load_registry(DEFAULT_REGISTRY_PATH)

    selected = validate_registry(registry)

    assert set(selected) == EXPECTED_MODEL_IDS
    models = {model["id"]: model for model in registry["models"]}
    assert models["ollama-qwen3.5-9b"]["source"]["digest"] == (
        "sha256:6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
    )
    assert models["eres2netv2-w24s4ep4"]["usage"]["status"] == "challenger"
    assert models["eres2netv2-w24s4ep4"]["deploymentSlots"] == [
        {
            "id": "secondary-speaker-verification",
            "adapterIds": ["modelscope-eres2netv2"],
        }
    ]
    assert models["wespeaker-redimnet2-b6-lm"]["usage"]["status"] == (
        "challenger"
    )
    assert models["ollama-qwen3.5-27b-q4-k-m"]["usage"]["status"] == (
        "production"
    )
    assert models["ollama-qwen3.5-9b"]["usage"]["status"] == "challenger"
    assert models["ollama-qwen3.5-35b-a3b-q4-k-m"]["usage"]["status"] == (
        "challenger"
    )
    assert models["ollama-qwen3.6-27b-q4-k-m"]["usage"]["status"] == (
        "challenger"
    )
    assert models["ollama-qwen3.6-35b-a3b-q4-k-m"]["usage"]["status"] == (
        "challenger"
    )
    assert models["ollama-glm-4.7-flash-q4-k-m"]["usage"]["status"] == (
        "challenger"
    )


def test_production_semantic_model_is_consistent_across_release_artifacts() -> None:
    registry = load_registry(DEFAULT_REGISTRY_PATH)
    validate_registry(registry)
    production_config = json.loads(
        (PROJECT_ROOT / "production.config.example.json").read_text(
            encoding="utf-8"
        )
    )
    lifecycle = json.loads(
        (PROJECT_ROOT / "model-lifecycle.v1.json").read_text(encoding="utf-8")
    )
    desktop_snapshot = json.loads(
        (
            PROJECT_ROOT
            / "apps"
            / "desktop"
            / "src"
            / "contracts"
            / "fixtures"
            / "rust-default-snapshot.json"
        ).read_text(encoding="utf-8")
    )

    configured_model = production_config["speaker"]["localLlmModel"]
    configured_digest = production_config["speaker"]["localLlmModelDigest"]
    production_semantic_models = [
        model
        for model in registry["models"]
        if model["source"]["provider"] == "ollama"
        and model["usage"]["status"] == "production"
        and "semantic-arbitration" in model["usage"]["roles"]
    ]
    assert len(production_semantic_models) == 1
    selected = production_semantic_models[0]
    selected_name = (
        selected["source"]["repository"].rsplit("/", 1)[-1]
        + ":"
        + selected["source"]["tag"]
    )

    assert selected_name == configured_model
    assert selected["source"]["digest"] == configured_digest
    assert (
        "sha256:" + selected["local"]["manifest"]["sha256"]
        == configured_digest
    )
    assert selected["usage"]["decisionPolicy"] == "suggestion-only"

    lifecycle_status = {
        item["name"]: item["status"] for item in lifecycle["ollamaModels"]
    }
    assert lifecycle_status[configured_model] == "active"
    assert {
        strategy["semanticModel"] for strategy in desktop_snapshot["strategies"]
    } == {configured_model}


def test_semantic_champion_is_the_runtime_default() -> None:
    assert LocalLLMConfig().model == SEMANTIC_CHAMPION
    assert BusinessProcessingConfig().model == SEMANTIC_CHAMPION
    assert SpeakerPipelineConfig().local_llm_model == SEMANTIC_CHAMPION
    speaker_policy = ProductionSpeakerPolicy()
    assert speaker_policy.local_llm_model == SEMANTIC_CHAMPION
    assert speaker_policy.local_llm_model_digest == SEMANTIC_CHAMPION_DIGEST


def test_load_registry_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text('{"schemaVersion":"1.0.0","schemaVersion":"1.0.0"}')

    with pytest.raises(RegistryValidationError, match="duplicate JSON key"):
        load_registry(path)


def test_registry_rejects_duplicate_deployment_slots(tmp_path: Path) -> None:
    registry, _ = _mts_registry(tmp_path)
    registry["models"][0]["deploymentSlots"] = [
        {"id": "secondary-speaker-verification", "adapterIds": ["adapter-a"]},
        {"id": "secondary-speaker-verification", "adapterIds": ["adapter-b"]},
    ]

    with pytest.raises(RegistryValidationError, match="duplicate slots"):
        validate_registry(registry)


def test_registry_rejects_duplicate_ids_and_sources(tmp_path: Path) -> None:
    registry, _ = _mts_registry(tmp_path)
    duplicate = copy.deepcopy(registry["models"][0])
    registry["models"].append(duplicate)

    with pytest.raises(RegistryValidationError, match="duplicate model IDs"):
        validate_registry(registry)

    duplicate["id"] = "second-model"
    with pytest.raises(RegistryValidationError, match="duplicate source identities"):
        validate_registry(registry)


def test_registry_allows_distinct_ollama_manifests_in_one_store(
    tmp_path: Path,
) -> None:
    registry, _ = _ollama_registry(tmp_path)
    second = copy.deepcopy(registry["models"][0])
    second["id"] = "ollama-fixture-27b"
    second["source"]["tag"] = "27b-q4_K_M"
    second_digest = "f" * 64
    second["source"]["digest"] = f"sha256:{second_digest}"
    second["local"]["manifest"]["path"] = (
        "manifests/registry.ollama.ai/library/fixture/27b-q4_K_M"
    )
    second["local"]["manifest"]["sha256"] = second_digest
    registry["models"].append(second)

    assert validate_registry(registry) == (
        "ollama-fixture-9b",
        "ollama-fixture-27b",
    )


def test_registry_rejects_shared_non_ollama_model_root(tmp_path: Path) -> None:
    registry, _ = _mts_registry(tmp_path)
    second = copy.deepcopy(registry["models"][0])
    second["id"] = "second-model"
    second["source"]["repository"] = "org/second-model"
    second["local"]["manifest"]["path"] = "second-manifest.json"
    registry["models"].append(second)

    with pytest.raises(
        RegistryValidationError,
        match="duplicate non-Ollama local model paths",
    ):
        validate_registry(registry)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (("usage", "status"), "accepted", "status is unsupported"),
        (("license", "spdx"), "UNKNOWN", "concrete SPDX"),
        (("license", "evidenceSha256"), "abc", "lowercase SHA-256"),
        (("local", "path"), "relative/model", "absolute local path"),
    ],
)
def test_registry_rejects_invalid_control_fields(
    tmp_path: Path,
    field: tuple[str, str],
    value: str,
    message: str,
) -> None:
    registry, _ = _mts_registry(tmp_path)
    model = registry["models"][0]
    model[field[0]][field[1]] = value

    with pytest.raises(RegistryValidationError, match=message):
        validate_registry(registry)


def test_verify_local_accepts_mts_manifest_and_detects_tampering(
    tmp_path: Path,
) -> None:
    registry, root = _mts_registry(tmp_path)
    assert validate_registry(registry, verify_local=True) == ("fixture-model",)

    (root / "model.pt").write_bytes(b"MODEL WEIGHTS")
    with pytest.raises(RegistryValidationError, match="SHA-256 mismatch"):
        validate_registry(registry, verify_local=True)


def test_verify_local_accepts_byte_preserving_reshard_manifest(
    tmp_path: Path,
) -> None:
    registry, root = _mts_registry(tmp_path)
    model = registry["models"][0]
    manifest_path = root / ".mts-model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schemaVersion": "1.1.0",
            "kind": "derived-safetensors-reshard",
            "reshard": {
                "exactTensorBytesPreserved": True,
                "modelQualityChanged": False,
                "source": {
                    "provider": model["source"]["provider"],
                    "repoId": model["source"]["repository"],
                    "revision": model["source"]["revision"],
                },
            },
        }
    )
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_path.write_bytes(payload)
    model["local"]["manifest"]["sha256"] = _sha256(payload)

    assert validate_registry(registry, verify_local=True) == (
        "fixture-model",
    )


def test_verify_local_rejects_reshard_without_byte_preservation(
    tmp_path: Path,
) -> None:
    registry, root = _mts_registry(tmp_path)
    model = registry["models"][0]
    manifest_path = root / ".mts-model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schemaVersion": "1.1.0",
            "kind": "derived-safetensors-reshard",
            "reshard": {
                "exactTensorBytesPreserved": False,
                "modelQualityChanged": False,
                "source": {
                    "provider": model["source"]["provider"],
                    "repoId": model["source"]["repository"],
                    "revision": model["source"]["revision"],
                },
            },
        }
    )
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_path.write_bytes(payload)
    model["local"]["manifest"]["sha256"] = _sha256(payload)

    with pytest.raises(
        RegistryValidationError, match="byte-preserving model quality"
    ):
        validate_registry(registry, verify_local=True)


def test_verify_local_accepts_ollama_oci_chain_and_detects_tampering(
    tmp_path: Path,
) -> None:
    registry, model_blob = _ollama_registry(tmp_path)
    assert validate_registry(registry, verify_local=True) == (
        "ollama-fixture-9b",
    )

    model_blob.write_bytes(b"gguf model")
    with pytest.raises(RegistryValidationError, match="blob SHA-256 mismatch"):
        validate_registry(registry, verify_local=True)


def test_cli_reports_a_machine_readable_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, _ = _mts_registry(tmp_path)
    registry_path = tmp_path / "registry.json"
    _write_json(registry_path, registry)

    assert main(["--registry", str(registry_path), "--verify-local"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["modelCount"] == 1
    assert summary["selectedModelCount"] == 1
    assert summary["verifyLocal"] is True
