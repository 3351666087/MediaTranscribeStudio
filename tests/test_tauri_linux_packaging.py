from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_ROOT = REPOSITORY_ROOT / "packaging" / "tauri"
BUILD_SCRIPT = PACKAGING_ROOT / "build_tauri_linux_release.py"
MANIFEST_SCHEMA = PACKAGING_ROOT / "linux-release-manifest.schema.json"
LINUX_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "tauri-linux-release.yml"
TARGET_TRIPLE = "x86_64-unknown-linux-gnu"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("mts_linux_packaging", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_elf(path: Path, *, machine: int = 62, executable: bool = True) -> None:
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", header, 18, machine)
    path.write_bytes(bytes(header))
    path.chmod(0o755 if executable else 0o644)


def _run(
    project_root: Path,
    *arguments: str,
    expect_success: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--project-root",
            str(project_root),
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if expect_success:
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
    else:
        assert completed.returncode != 0, completed.stdout
        payload = json.loads(completed.stderr.strip().splitlines()[-1])
        assert payload["ok"] is False
    return completed, payload


def _write_project(root: Path, *, tauri_version: str = "0.1.0") -> None:
    desktop = root / "apps" / "desktop"
    tauri = desktop / "src-tauri"
    tauri.mkdir(parents=True)
    (desktop / "package.json").write_text(
        json.dumps({"name": "fixture", "version": "0.1.0"}), encoding="utf-8"
    )
    (desktop / "package-lock.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "version": "0.1.0",
                "lockfileVersion": 3,
                "packages": {"": {"name": "fixture", "version": "0.1.0"}},
            }
        ),
        encoding="utf-8",
    )
    (tauri / "tauri.conf.json").write_text(
        json.dumps(
            {
                "productName": "Fixture",
                "version": tauri_version,
                "identifier": "studio.mediatranscribe.fixture",
            }
        ),
        encoding="utf-8",
    )
    (tauri / "Cargo.toml").write_text(
        '[package]\nname = "media-transcribe-studio"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    (tauri / "Cargo.lock").write_text(
        'version = 3\n\n[[package]]\nname = "media-transcribe-studio"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )


def test_linux_build_plan_is_locked_version_coherent_and_non_mutating(
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"
    target = tmp_path / "must-not-create-target"

    _, result = _run(
        REPOSITORY_ROOT,
        "--target-triple",
        TARGET_TRIPLE,
        "--target-directory",
        str(target),
        "--output-directory",
        str(output),
        "--source-date-epoch",
        "1784678400",
        "--dry-run",
    )

    assert result["dryRun"] is True
    assert result["version"] == "0.1.0"
    assert result["targetTriple"] == TARGET_TRIPLE
    assert result["bundles"] == ["appimage", "deb", "rpm"]
    assert result["commands"] == [
        "npm ci",
        (
            "npm run tauri -- build --ci --target x86_64-unknown-linux-gnu "
            "--bundles appimage,deb,rpm "
            "--config <generated-linux-runtime-overlay> -- --locked"
        ),
    ]
    assert [item["path"] for item in result["lockedInputs"]] == [
        "apps/desktop/package-lock.json",
        "apps/desktop/src-tauri/Cargo.lock",
    ]
    assert all(len(item["sha256"]) == 64 for item in result["lockedInputs"])
    runtime = result["runtimeBootstrap"]
    assert runtime["enabled"] is True
    assert runtime["embedded"] is True
    assert runtime["bundledModelArtifacts"] is False
    assert runtime["absolutePathsIncluded"] is False
    assert runtime["fileCount"] == len(runtime["files"]) > 0
    assert all(item["path"].startswith("mts-runtime/") for item in runtime["files"])
    assert any(
        item["path"] == "mts-runtime/tools/pyannote_runtime.py"
        for item in runtime["files"]
    )
    assert LINUX_WORKFLOW.is_file()
    assert not output.exists()
    assert not target.exists()


def test_linux_dry_run_rejects_incoherent_versions_without_writes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project, tauri_version="0.2.0")
    output = tmp_path / "must-not-exist"

    _, error = _run(
        project,
        "--target-triple",
        TARGET_TRIPLE,
        "--output-directory",
        str(output),
        "--dry-run",
        expect_success=False,
    )

    assert "versions must match" in str(error["error"])
    assert not output.exists()


def test_linux_architecture_alias_resolves_to_canonical_target_triple() -> None:
    _, result = _run(
        REPOSITORY_ROOT,
        "--architecture",
        "x64",
        "--dry-run",
    )

    assert result["architecture"] == "x64"
    assert result["targetTriple"] == TARGET_TRIPLE


def test_linux_workflow_uses_native_runners_and_mode_preserving_transport() -> None:
    workflow = LINUX_WORKFLOW.read_text(encoding="utf-8")

    assert "ubuntu-24.04-arm" in workflow
    assert "MTS_EXPECTED_RUNNER_ARCH" in workflow
    assert 'actual_architecture="$(uname -m)"' in workflow
    assert "gcc-aarch64-linux-gnu" not in workflow
    assert 'tar -czf "$archive"' in workflow
    assert 'tar -xzf "$archive"' in workflow
    assert 'sha256sum "$(basename "$archive")"' in workflow
    assert "os.access(artifact, os.X_OK)" in workflow
    assert "mts-release-transport/*" in workflow
    assert "mts-release/**" not in workflow


def test_linux_skip_compile_collects_hashed_artifacts_and_valid_manifest(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_project(project)
    target = tmp_path / "cargo-target"
    bundle_root = target / TARGET_TRIPLE / "release" / "bundle"
    source = bundle_root / "appimage" / "Fixture_0.1.0_amd64.AppImage"
    source.parent.mkdir(parents=True)
    _write_elf(source)
    output = tmp_path / "release"

    _, result = _run(
        project,
        "--target-triple",
        TARGET_TRIPLE,
        "--target-directory",
        str(target),
        "--output-directory",
        str(output),
        "--source-date-epoch",
        "1784678400",
        "--bundles",
        "appimage",
        "--skip-compile",
    )

    manifest_path = output / "linux-release-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(manifest)
    manifest_without_runtime = dict(manifest)
    manifest_without_runtime.pop("runtimeBootstrap")
    missing_runtime_errors = list(validator.iter_errors(manifest_without_runtime))
    assert any(
        "runtimeBootstrap" in error.message for error in missing_runtime_errors
    )
    expected_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    assert result["manifestSha256"] == expected_manifest_hash
    assert (
        output / "linux-release-manifest.json.sha256"
    ).read_text(encoding="ascii").strip() == expected_manifest_hash
    assert manifest["contract"] == "mts-tauri-linux-release/v1"
    assert manifest["build"]["targetTriple"] == TARGET_TRIPLE
    assert manifest["artifactCount"] == 1
    assert manifest["totalBytes"] == source.stat().st_size
    assert manifest["runtimeBootstrap"]["embedded"] is False
    assert manifest["runtimeBootstrap"]["artifactVerifications"] == [
        {
            "artifactPath": "artifacts/appimage/Fixture_0.1.0_amd64.AppImage",
            "bundle": "appimage",
            "reason": "runtime-bootstrap-disabled",
            "verified": False,
        }
    ]
    for artifact in manifest["artifacts"]:
        copied = output / artifact["path"]
        assert copied.is_file()
        assert copied.stat().st_size == artifact["size"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == artifact["sha256"]
        assert artifact["executable"] is True
        assert copied.stat().st_mode & 0o111


def test_linux_overlay_embeds_runtime_at_tauri_resource_root(tmp_path: Path) -> None:
    module = _load_build_module()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    tauri = tmp_path / "src-tauri"
    icon = tauri / "icons" / "icon.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"png")
    overlay = tmp_path / "overlay.json"

    module._write_linux_overlay(overlay, runtime, tauri)

    document = json.loads(overlay.read_text(encoding="utf-8"))
    assert document == {
        "bundle": {
            "icon": [str(icon)],
            "resources": {str(runtime): "mts-runtime"},
        }
    }


def test_linux_portable_config_removes_windows_paths_and_rejects_raw_secrets(
    tmp_path: Path,
) -> None:
    module = _load_build_module()
    template = tmp_path / "production.config.example.json"
    template.write_text(
        json.dumps(
            {
                "input": "D:/Downloads",
                "output": "D:/MediaTranscribeStudio/outputs",
                "cache": "D:/MediaTranscribeStudio/cache",
                "model": "D:/models/vendor/model",
                "runtime": "D:/MediaTranscribeStudio/runtime/media-asr/python.exe",
                "apiKeyEnv": "OPENAI_API_KEY",
            }
        ),
        encoding="utf-8",
    )

    portable = json.loads(module._portable_config_bytes(template))

    assert portable["input"] == "inputs"
    assert portable["output"] == "exports"
    assert portable["cache"] == "cache"
    assert portable["model"] == "models/vendor/model"
    assert portable["runtime"] == "runtime/media-asr/python.exe"
    assert portable["apiKeyEnv"] == "OPENAI_API_KEY"

    template.write_text(json.dumps({"apiKey": "sk-" + "a" * 32}), encoding="utf-8")
    try:
        module._portable_config_bytes(template)
    except module.ReleaseBuildError as exc:
        assert "credential" in str(exc).casefold()
    else:
        raise AssertionError("raw API key must be rejected")


def test_linux_embedded_runtime_verifier_requires_exact_standard_package_tree(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _load_build_module()
    files = {
        "backend/worker.py": b"print('worker')\n",
        "bootstrap/runtime-bootstrap.v1.json": b"{}\n",
        "bootstrap/Initialize-MtsRuntime.sh": b"#!/usr/bin/env bash\n",
        "bootstrap/Manage-MtsModels.sh": b"#!/usr/bin/env bash\n",
    }
    plan = {
        "enabled": True,
        "entries": [
            {
                "destination": path,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in files.items()
        ],
    }

    def extract(_bundle: str, _artifact: Path, destination: Path) -> None:
        runtime = destination / "usr" / "lib" / "Fixture" / "mts-runtime"
        for relative, content in files.items():
            target = runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o755 if target.suffix == ".sh" else 0o644)

    monkeypatch.setattr(module, "_extract_linux_artifact", extract)
    result = module._verify_embedded_runtime("appimage", tmp_path / "fixture.AppImage", plan)

    assert result == {
        "bundle": "appimage",
        "verified": True,
        "packageRuntimeRoot": "usr/lib/Fixture/mts-runtime",
        "fileCount": 4,
        "totalBytes": sum(map(len, files.values())),
    }


def test_linux_appimage_offset_and_architecture_are_read_from_elf(tmp_path: Path) -> None:
    module = _load_build_module()
    artifact = tmp_path / "Fixture_0.1.0_amd64.AppImage"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", header, 18, 62)
    struct.pack_into("<Q", header, 40, 64)
    struct.pack_into("<H", header, 58, 64)
    struct.pack_into("<H", header, 60, 1)
    artifact.write_bytes(bytes(header) + bytes(64) + b"hsqs" + bytes(32))
    artifact.chmod(0o755)

    assert module._appimage_squashfs_offset(artifact) == 128
    module._validate_artifact_identity(
        "appimage",
        artifact,
        {"architecture": "x64", "version": "0.1.0"},
    )
