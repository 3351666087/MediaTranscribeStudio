from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import zipfile

from jsonschema import Draft202012Validator
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging" / "tauri"
SCRIPT = PACKAGING / "build_tauri_macos_release.py"
VERIFIER = PACKAGING / "verify_tauri_macos_release.py"
SCHEMA = PACKAGING / "macos-release-manifest.schema.json"
WORKFLOW = ROOT / ".github" / "workflows" / "tauri-macos-release.yml"
TARGET = "x86_64-apple-darwin"
INITIALIZER = PACKAGING / "bootstrap" / "Initialize-MtsRuntime.sh"
MODEL_MANAGER = PACKAGING / "bootstrap" / "Manage-MtsModels.sh"


def _run(
    project: Path,
    *args: str,
    expect_success: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(project), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
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


def _load_build_module():
    spec = importlib.util.spec_from_file_location("mts_macos_packaging", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("mts_macos_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PACKAGING))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _verify(release: Path, *, expect_success: bool = True) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--release-directory",
            str(release),
            "--expected-target",
            TARGET,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    if expect_success:
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["ok"] is True
    else:
        assert completed.returncode != 0, completed.stdout
        result = json.loads(completed.stderr.strip().splitlines()[-1])
        assert result["ok"] is False
    return result


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
    # The macOS payload builder uses an explicit source allowlist.  Small text
    # fixtures keep this test independent of the workstation's model caches.
    for directory, filename in (
        ("backend", "worker.py"),
        ("contracts", "job.schema.json"),
        ("reporting", "report.py"),
    ):
        path = root / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    fixed = (
        "production.config.example.json",
        "production.config.remote.example.json",
        "configs/model-catalog.v1.json",
        "configs/model-catalog.v1.schema.json",
        "tools/build_portable_model_catalog.py",
        "configs/llm-provider-presets.v1.json",
        "configs/llm-provider-presets.schema.json",
        "tools/model_manager.py",
        "tools/model_registry.py",
        "tools/pyannote_runtime.py",
        "requirements-media-asr.txt",
        "requirements-pyannote.txt",
        "pdf-renderer/target/pdf-renderer.jar",
        "packaging/tauri/bootstrap/Initialize-MtsRuntime.sh",
        "packaging/tauri/bootstrap/Manage-MtsModels.sh",
        "packaging/tauri/bootstrap/README.md",
    )
    for relative in fixed:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        if path.suffix == ".sh":
            path.chmod(0o755)


def _write_thin_macho(path: Path, *, cpu_type: int = 0x01000007) -> None:
    path.write_bytes(
        struct.pack("<IiiIIIII", 0xFEEDFACF, cpu_type, 3, 2, 0, 0, 0, 0)
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_app(root: Path, *, with_runtime: bool = False) -> Path:
    app = root / "apps" / "desktop" / "src-tauri" / "target" / TARGET / "release" / "bundle" / "macos" / "Fixture.app"
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)
    (contents / "Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleExecutable": "Fixture",
                "CFBundleIdentifier": "studio.mediatranscribe.fixture",
                "CFBundleShortVersionString": "0.1.0",
            }
        )
    )
    executable = contents / "MacOS" / "Fixture"
    _write_thin_macho(executable)
    if with_runtime:
        runtime = contents / "Resources" / "mts-runtime"
        (runtime / "backend").mkdir(parents=True)
        (runtime / "bootstrap").mkdir(parents=True)
        (runtime / "backend" / "worker.py").write_text("# worker\n", encoding="utf-8")
        (runtime / "bootstrap" / "runtime-bootstrap.v1.json").write_text("{}\n", encoding="utf-8")
    return app


def test_macos_dry_run_is_cross_host_locked_and_weight_free(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    target = tmp_path / "must-not-create-target"
    _, result = _run(
        ROOT,
        "--target-triple",
        "universal-apple-darwin",
        "--target-directory",
        str(target),
        "--output-directory",
        str(output),
        "--source-date-epoch",
        "1784678400",
        "--dry-run",
    )
    assert result["platform"] == "macos"
    assert result["architecture"] == "universal2"
    assert result["targetTriple"] == "universal-apple-darwin"
    assert result["bundles"] == ["app", "dmg"]
    assert result["runtimePayload"]["bundledModelArtifacts"] is False
    assert all("weights" not in item["destination"] for item in result["runtimePayload"]["files"])
    assert not output.exists()
    assert not target.exists()


def test_macos_dry_run_records_notarization_gate_without_requiring_host_credentials() -> None:
    _, result = _run(
        ROOT,
        "--target-triple",
        "universal-apple-darwin",
        "--require-notarization",
        "--dry-run",
    )
    assert result["trust"] == {
        "requireCodeSigning": True,
        "requireNotarization": True,
    }


def test_macos_development_plan_can_explicitly_disable_signing() -> None:
    _, result = _run(
        ROOT,
        "--target-triple",
        "universal-apple-darwin",
        "--allow-unsigned-development",
        "--dry-run",
    )
    assert result["trust"] == {
        "requireCodeSigning": False,
        "requireNotarization": False,
    }


def test_macos_stable_plan_requires_notarization_and_excludes_portable_zip() -> None:
    _, unsigned_error = _run(
        ROOT,
        "--target-triple",
        "universal-apple-darwin",
        "--channel",
        "stable",
        "--allow-unsigned-development",
        "--dry-run",
        expect_success=False,
    )
    assert "stable macos channel requires signed, notarized" in str(unsigned_error["error"]).casefold()

    _, zip_error = _run(
        ROOT,
        "--target-triple",
        "universal-apple-darwin",
        "--channel",
        "stable",
        "--bundles",
        "app,dmg,zip",
        "--require-notarization",
        "--dry-run",
        expect_success=False,
    )
    assert "portable zip remains development-only" in str(zip_error["error"]).casefold()

    _, result = _run(
        ROOT,
        "--target-triple",
        "universal-apple-darwin",
        "--channel",
        "stable",
        "--bundles",
        "app,dmg",
        "--require-notarization",
        "--dry-run",
    )
    assert result["trust"] == {
        "requireCodeSigning": True,
        "requireNotarization": True,
    }


def test_macos_dry_run_rejects_version_drift_without_writes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project, tauri_version="0.2.0")
    output = tmp_path / "must-not-exist"
    _, error = _run(
        project,
        "--target-triple",
        TARGET,
        "--output-directory",
        str(output),
        "--dry-run",
        expect_success=False,
    )
    assert "versions must match" in str(error["error"])
    assert not output.exists()


def test_macos_dry_run_can_persist_an_explicit_plan_artifact(tmp_path: Path) -> None:
    plan_path = tmp_path / "macos-plan.json"
    _, result = _run(
        ROOT,
        "--target-triple",
        "universal-apple-darwin",
        "--dry-run",
        "--plan-output",
        str(plan_path),
    )
    assert result["planOutputPath"] == str(plan_path)
    assert json.loads(plan_path.read_text(encoding="utf-8"))["targetTriple"] == "universal-apple-darwin"
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert digest == (plan_path.with_name(plan_path.name + ".sha256")).read_text(encoding="ascii").strip()


def test_macos_skip_compile_repairs_unsigned_app_and_emits_schema_manifest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    app = _write_app(project)
    output = tmp_path / "release"
    _, result = _run(
        project,
        "--target-triple",
        TARGET,
        "--bundles",
        "app,zip",
        "--output-directory",
        str(output),
        "--skip-compile",
        "--allow-unsigned-development",
    )
    assert result["manifestPath"] == str(output / "macos-release-manifest.json")
    manifest_path = output / "macos-release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(manifest)
    assert manifest["trust"]["mode"] == "development-unsigned"
    assert manifest["runtime"]["bundledModelArtifacts"] is False
    runtime = output / "artifacts" / "app" / "Fixture.app" / "Contents" / "Resources" / "mts-runtime"
    assert (runtime / "backend" / "worker.py").is_file()
    for config_name in ("production.config.example.json", "production.config.remote.example.json"):
        assert "D:/" not in (runtime / config_name).read_text(encoding="utf-8")
    zip_path = next((output / "artifacts" / "zip").glob("*.zip"))
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert any(name.endswith("Contents/Resources/mts-runtime/backend/worker.py") for name in names)
    assert not list(output.rglob("*.safetensors"))
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        output / "macos-release-manifest.json.sha256"
    ).read_text().strip()
    assert app.exists()
    verification = _verify(output)
    assert verification["artifactBundles"] == ["app", "zip"]
    assert verification["appChecks"]["app"]["fileCount"] > 0
    assert verification["appChecks"]["zip"]["bundledModelArtifacts"] is False


def test_macos_signed_dmg_only_collection_uses_mounted_app_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_build_module()
    project = tmp_path / "project"
    _write_project(project)
    output = tmp_path / "release"
    context = module._load_build_context(
        project,
        TARGET,
        ("dmg",),
        output,
        tmp_path / "target",
        "candidate",
        0,
    )
    dmg = tmp_path / "Fixture_0.1.0_x64.dmg"
    dmg.write_bytes(b"signed fixture")
    verified: list[Path] = []

    def fake_verify_dmg(path: Path, *_args, **_kwargs) -> None:
        verified.append(path)

    monkeypatch.setattr(module, "_verify_dmg", fake_verify_dmg)
    manifest_path, _, artifacts = module._collect_release(
        context,
        {"dmg": [dmg]},
        signing_identity="Developer ID Application: Example Corp (TEAMID)",
        allow_unsigned_development=False,
        require_notarization=False,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verified == [dmg]
    assert [item["bundle"] for item in artifacts] == ["dmg"]
    assert manifest["trust"]["mode"] == "codesign"


def test_macos_verifier_rejects_any_post_manifest_artifact_change(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    _write_app(project)
    output = tmp_path / "release"
    _run(
        project,
        "--target-triple",
        TARGET,
        "--bundles",
        "app",
        "--output-directory",
        str(output),
        "--skip-compile",
        "--allow-unsigned-development",
    )
    worker = output / "artifacts" / "app" / "Fixture.app" / "Contents" / "Resources" / "mts-runtime" / "backend" / "worker.py"
    worker.write_text("# changed after collection\n", encoding="utf-8")
    error = _verify(output, expect_success=False)
    assert "ledger mismatch" in str(error["error"]).casefold()


def test_macos_verifier_rejects_unsigned_stable_manifest_even_with_fresh_checksum(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_project(project)
    _write_app(project)
    output = tmp_path / "release"
    _run(
        project,
        "--target-triple",
        TARGET,
        "--bundles",
        "app",
        "--output-directory",
        str(output),
        "--skip-compile",
        "--allow-unsigned-development",
    )
    manifest_path = output / "macos-release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["channel"] = "stable"
    manifest["releaseId"] = (
        f"{manifest['appId']}/{manifest['version']}/{manifest['platform']}/"
        f"{manifest['architecture']}/stable"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "macos-release-manifest.json.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    error = _verify(output, expect_success=False)
    assert "stable macos releases require developer id application" in str(error["error"]).casefold()


def test_macos_collection_rejects_wrong_macho_architecture(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    app = _write_app(project)
    _write_thin_macho(app / "Contents" / "MacOS" / "Fixture", cpu_type=0x0100000C)
    _, error = _run(
        project,
        "--target-triple",
        TARGET,
        "--bundles",
        "app",
        "--output-directory",
        str(tmp_path / "release"),
        "--skip-compile",
        "--allow-unsigned-development",
        expect_success=False,
    )
    assert "architectures" in str(error["error"]).casefold()


def test_macos_macho_parser_requires_both_universal2_slices(tmp_path: Path) -> None:
    module = _load_build_module()
    universal = tmp_path / "universal"
    first_offset = 4096
    second_offset = 8192
    slice_size = 32
    header = struct.pack(
        ">IIiiIIIiiIII",
        0xCAFEBABE,
        2,
        0x01000007,
        3,
        first_offset,
        slice_size,
        12,
        0x0100000C,
        0,
        second_offset,
        slice_size,
        12,
    )
    payload = bytearray(second_offset + slice_size)
    payload[: len(header)] = header
    universal.write_bytes(payload)
    assert module._read_macho_architectures(universal) == frozenset({"x86_64", "arm64"})


def test_macos_workflow_pins_python_and_isolates_unsigned_signing_identity() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in workflow
    assert "signed-notarized" in workflow
    assert "macos-15-intel" in workflow
    assert "'macos-15'" in workflow
    assert "macos-14" not in workflow
    assert "default: app,dmg" in workflow
    assert "default: development" in workflow
    assert "default: unsigned-development" in workflow
    assert "MTS_EXPECTED_RUNNER_ARCH" in workflow
    assert 'x86_64-apple-darwin) architecture="x64"' in workflow
    assert 'aarch64-apple-darwin) architecture="arm64"' in workflow
    assert 'universal-apple-darwin) architecture="universal2"' in workflow
    assert workflow.count("--require-native") == 3
    assert workflow.count("--launch-smoke-seconds 8") == 2
    assert workflow.count("--launch-via-open") == 2
    assert workflow.count("--expected-signing-identity") == 3
    assert "inputs.release_mode != 'unsigned-development' && secrets.MACOS_CODESIGN_IDENTITY || ''" in workflow
    assert workflow.index("Build the locked OpenHTMLtoPDF sidecar") < workflow.index(
        "Install macOS packaging verifier dependencies"
    )
    assert 'tar -czf "$archive"' in workflow
    assert 'shasum -a 256 "$(basename "$archive")"' in workflow
    assert workflow.count("--release-directory") == 1
    assert workflow.count("--release-archive") == 2
    assert workflow.count("--archive-checksum") == 2
    assert workflow.count("--expected-release-root") == 2
    assert "mts-release-transport/*" in workflow
    assert "mts-release/**" not in workflow
    job_environment = workflow.split("steps:", 1)[0]
    assert "MAC_CODESIGN_IDENTITY" not in job_environment
    cleanup = workflow.split("- name: Remove temporary Apple credentials", 1)[1]
    assert "mts-certificate.p12" in cleanup
    assert "AuthKey_*.p8" in cleanup


def test_macos_workflow_reverifies_downloaded_transport_natively() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    transport = workflow.split("\n  transport-verify:\n", 1)[1]

    assert "needs: build" in transport
    assert (
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
        in transport
    )
    assert "uses: actions/" in workflow
    assert "uses: actions/checkout@v4" not in workflow
    assert "uses: actions/setup-node@v4" not in workflow
    assert "uses: actions/setup-python@v5" not in workflow
    assert "uses: actions/setup-java@v4" not in workflow
    assert "uses: actions/upload-artifact@v4" not in workflow
    assert "uses: actions/download-artifact@v4" not in workflow
    assert "mts-release-download" in transport
    assert "MTS_EXPECTED_RUNNER_ARCH" in transport
    assert transport.count("--release-archive") == 1
    assert transport.count("--archive-checksum") == 1
    assert transport.count("--expected-release-root") == 1
    assert transport.count("--require-native") == 1
    assert transport.count("--run-bootstrap") == 1
    assert transport.count("--launch-smoke-seconds 8") == 1
    assert transport.count("--launch-via-open") == 1
    assert transport.count("--expected-signing-identity") == 1
    assert 'test ! -L "$archive"' in transport
    assert 'test ! -L "$checksum"' in transport


def test_macos_transport_archive_is_checksum_locked_and_safely_extracted(
    tmp_path: Path,
) -> None:
    verifier = _load_verify_module()
    source = tmp_path / "source" / "macos-universal2"
    source.mkdir(parents=True)
    executable = source / "bootstrap.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    archive = tmp_path / "macos-universal2.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        stream.add(source, arcname=source.name)
    checksum = tmp_path / "macos-universal2.tar.gz.sha256"
    checksum.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        encoding="ascii",
    )

    release, evidence = verifier._extract_transport_archive(
        archive,
        checksum,
        tmp_path / "roundtrip",
        "macos-universal2",
    )

    assert evidence["safeExtraction"] is True
    assert evidence["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert int(evidence["memberCount"]) >= 2
    assert (release / "bootstrap.sh").stat().st_mode & stat.S_IXUSR


def test_macos_transport_archive_rejects_traversal_before_extraction(
    tmp_path: Path,
) -> None:
    verifier = _load_verify_module()
    archive = tmp_path / "macos-universal2.tar.gz"
    with tarfile.open(archive, mode="w:gz") as stream:
        root = tarfile.TarInfo("macos-universal2")
        root.type = tarfile.DIRTYPE
        stream.addfile(root)
        traversal = tarfile.TarInfo("../escape")
        payload = b"must-not-extract"
        traversal.size = len(payload)
        stream.addfile(traversal, io.BytesIO(payload))
    checksum = tmp_path / "macos-universal2.tar.gz.sha256"
    checksum.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        encoding="ascii",
    )

    with pytest.raises(verifier.ReleaseVerificationError, match="Unsafe transport archive member"):
        verifier._extract_transport_archive(
            archive,
            checksum,
            tmp_path / "roundtrip",
            "macos-universal2",
        )

    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "roundtrip").exists()


def test_macos_launchservices_process_matching_is_exact_for_paths_with_spaces() -> None:
    verifier = _load_verify_module()
    executable = Path(
        "/Applications/MediaTranscribe Studio.app/Contents/MacOS/MediaTranscribe Studio"
    )
    process_table = "\n".join(
        [
            " 101 /usr/bin/open -n /Applications/MediaTranscribe Studio.app",
            f" 202 {executable}",
            f" 303 {executable} --fixture",
            f" 404 {executable}-helper",
            "not-a-pid malformed",
        ]
    )

    assert verifier._process_ids_for_executable(process_table, executable) == {202, 303}


def test_macos_stable_verification_requires_native_external_trust_anchor() -> None:
    verifier = _load_verify_module()
    identity = "Developer ID Application: Example Corp (TEAMID)"
    manifest = {
        "channel": "stable",
        "trust": {
            "mode": "codesign",
            "signingIdentity": identity,
            "notarization": "stapled",
        },
    }

    with pytest.raises(verifier.ReleaseVerificationError, match="require --require-native"):
        verifier._verify_external_trust_anchor(
            manifest,
            expected_signing_identity=identity,
            require_native=False,
        )
    with pytest.raises(verifier.ReleaseVerificationError, match="external trust anchor"):
        verifier._verify_external_trust_anchor(
            manifest,
            expected_signing_identity=None,
            require_native=True,
        )
    with pytest.raises(verifier.ReleaseVerificationError, match="does not match"):
        verifier._verify_external_trust_anchor(
            manifest,
            expected_signing_identity="Developer ID Application: Other Corp (OTHERID)",
            require_native=True,
        )

    verifier._verify_external_trust_anchor(
        manifest,
        expected_signing_identity=identity,
        require_native=True,
    )


def test_macos_native_trust_commands_distinguish_apps_and_disk_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_build_module()
    codesign = tmp_path / "codesign"
    spctl = tmp_path / "spctl"
    codesign.write_text("fixture\n", encoding="utf-8")
    spctl.write_text("fixture\n", encoding="utf-8")
    app = tmp_path / "Fixture.app"
    app.mkdir()
    dmg = tmp_path / "Fixture.dmg"
    dmg.write_bytes(b"fixture")
    identity = "Developer ID Application: Example Corp (TEAMID)"
    commands: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        return {"codesign": str(codesign), "spctl": str(spctl)}.get(name)

    def fake_run(command, **kwargs):
        normalized = [str(item) for item in command]
        commands.append(normalized)
        details = f"Authority={identity}\n" if "-dv" in normalized else ""
        return subprocess.CompletedProcess(normalized, 0, stdout="", stderr=details)

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.shutil, "which", fake_which)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._verify_codesign(app, identity)
    module._verify_codesign(dmg, identity)
    module._verify_gatekeeper(app, artifact_type="app")
    module._verify_gatekeeper(dmg, artifact_type="dmg")

    verify_commands = [command for command in commands if "--verify" in command]
    assert "--deep" in verify_commands[0]
    assert "--deep" not in verify_commands[1]
    gatekeeper_commands = [command for command in commands if command[0] == str(spctl)]
    assert gatekeeper_commands[0][gatekeeper_commands[0].index("--type") + 1] == "execute"
    assert gatekeeper_commands[1][gatekeeper_commands[1].index("--type") + 1] == "open"
    assert "context:primary-signature" in gatekeeper_commands[1]


def test_macos_unsigned_collection_fails_closed_without_explicit_flag(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    _write_app(project)
    _, error = _run(
        project,
        "--target-triple",
        TARGET,
        "--bundles",
        "app",
        "--output-directory",
        str(tmp_path / "release"),
        "--skip-compile",
        expect_success=False,
    )
    assert "signing identity" in str(error["error"])


def test_macos_runtime_allowlist_rejects_raw_api_key_values(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_project(project)
    (project / "production.config.remote.example.json").write_text(
        json.dumps({"llm": {"apiKey": "sk-this-value-must-never-ship-123456"}}),
        encoding="utf-8",
    )
    _, error = _run(
        project,
        "--target-triple",
        TARGET,
        "--dry-run",
        expect_success=False,
    )
    assert "credential" in str(error["error"]).casefold()


def test_macos_shell_bootstrap_binds_paths_without_overwriting_operator_config(tmp_path: Path) -> None:
    if os.name == "nt":
        return
    bash = shutil.which("bash")
    if bash is None:
        return
    data_root = tmp_path / "Application Support" / "MediaTranscribeStudio"
    model_root = tmp_path / "external models"
    config_path = data_root / "config" / "production.config.json"
    command = [
        bash,
        str(INITIALIZER),
        "--app-root",
        str(ROOT),
        "--data-root",
        str(data_root),
        "--model-root",
        str(model_root),
        "--production-config",
        str(config_path),
    ]
    bootstrap_environment = dict(os.environ)
    bootstrap_environment["MTS_PLATFORM_OVERRIDE"] = "Darwin"
    first = subprocess.run(
        command,
        cwd=ROOT,
        env=bootstrap_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    result = json.loads(first.stdout)
    assert result["configCreated"] is True
    python_hint = Path(result["workerPythonHint"])
    assert python_hint.read_text(encoding="utf-8").strip() == result["workerPython"]
    document = json.loads(config_path.read_text(encoding="utf-8"))
    assert document["paths"]["allowedInputRoots"] == [str(data_root / "inputs")]
    assert document["models"]["qwen3Asr"].startswith(str(model_root))
    expected_macos_runtime = {
        "vadDevice": "cpu",
        "asrDevice": "cpu",
        "asrDtype": "float32",
        "camPlusDevice": "cpu",
        "eres2netDevice": "cpu",
        "pyannoteDevice": "cpu",
    }
    assert {
        key: document["runtime"][key] for key in expected_macos_runtime
    } == expected_macos_runtime
    assert "pyannotePython" not in document["executables"]
    assert document["models"]["pyannote"] is None
    assert document["speaker"]["pyannoteMode"] == "disabled"
    assert document["speaker"]["overlapRecoveryMode"] == "disabled"
    assert "D:/" not in json.dumps(document)
    from backend.production_config import ProductionConfig

    parsed = ProductionConfig.load(config_path)
    assert parsed.runtime.asr_device == "cpu"
    assert parsed.speaker.pyannote_mode == "disabled"

    document["operatorSentinel"] = "preserve"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    second = subprocess.run(
        command,
        cwd=ROOT,
        env=bootstrap_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["configCreated"] is False
    assert json.loads(config_path.read_text(encoding="utf-8"))["operatorSentinel"] == "preserve"

    environment = dict(os.environ)
    environment["MTS_DATA_ROOT"] = str(data_root)
    listed = subprocess.run(
        [bash, str(MODEL_MANAGER), "list"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stderr
    assert json.loads(listed.stdout)["models"]
