from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.build_local_llm_product_configs import (
    ProductConfigSetError,
    build_config_set,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _registry() -> dict[str, object]:
    return {
        "schemaVersion": "1.0.0",
        "registryId": "fixture.registry",
        "hardwareProfiles": [
            {
                "id": "fixture-host",
                "platform": "windows-x86_64",
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
                    "driverVersion": "1",
                },
            }
        ],
        "models": [
            {
                "id": "fixture-llm",
                "displayName": "Fixture LLM",
                "source": {
                    "provider": "ollama",
                    "repository": "library/fixture",
                    "tag": "27b-q4_K_M",
                    "digest": "sha256:" + "a" * 64,
                },
                "license": {
                    "spdx": "Apache-2.0",
                    "evidencePath": "blobs/sha256-license",
                    "evidenceSha256": "b" * 64,
                },
                "usage": {
                    "status": "challenger",
                    "roles": ["semantic-arbitration"],
                    "decisionPolicy": "suggestion-only",
                },
                "runtime": {
                    "engine": "ollama",
                    "version": "1",
                    "executablePath": "D:/ollama.exe",
                    "accelerator": "cuda",
                    "artifactFormat": "gguf",
                },
                "quantization": {
                    "scheme": "Q4_K_M",
                    "evidencePath": "blobs/sha256-quant",
                },
                "hardwareProfileIds": ["fixture-host"],
                "local": {
                    "path": "D:/models/ollama",
                    "manifest": {
                        "format": "ollama-oci-manifest-v2",
                        "path": (
                            "manifests/registry.ollama.ai/"
                            "library/fixture/27b-q4_K_M"
                        ),
                        "sha256": "a" * 64,
                        "fileCount": 4,
                        "totalBytes": 123,
                    },
                },
            }
        ],
    }


def test_builds_pinned_config_and_hash_bound_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    registry = tmp_path / "registry.json"
    output = tmp_path / "configs"
    _write(
        source,
        {
            "schemaVersion": "1.0.0",
            "speaker": {
                "localLlmModel": "old:tag",
                "localLlmModelDigest": "sha256:" + "0" * 64,
            },
        },
    )
    _write(registry, _registry())

    manifest = build_config_set(
        source_config_path=source,
        registry_path=registry,
        output_directory=output,
    )

    generated_path = output / "fixture-llm.json"
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    assert generated["speaker"]["localLlmModel"] == "fixture:27b-q4_K_M"
    assert generated["speaker"]["localLlmModelDigest"] == "sha256:" + "a" * 64
    assert manifest["configs"][0]["configFileSha256"] == hashlib.sha256(
        generated_path.read_bytes()
    ).hexdigest()
    persisted = json.loads(
        (output / "config-set.manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == manifest


def test_rejects_model_without_semantic_role(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    registry = tmp_path / "registry.json"
    document = _registry()
    document["models"][0]["usage"]["roles"] = ["translation"]
    _write(source, {"speaker": {}})
    _write(registry, document)

    with pytest.raises(ProductConfigSetError, match="semantic arbitration"):
        build_config_set(
            source_config_path=source,
            registry_path=registry,
            output_directory=tmp_path / "output",
            model_ids=["fixture-llm"],
        )


def test_direct_cli_help_resolves_repository_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "tools"
                / "build_local_llm_product_configs.py"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--source-config" in completed.stdout
