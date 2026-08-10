from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import uuid
import zipfile

import pytest
from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_ROOT = REPOSITORY_ROOT / "packaging" / "tauri"
NEW_RELEASE_SCRIPT = PACKAGING_ROOT / "New-TauriRelease.ps1"
LIFECYCLE_SCRIPT = PACKAGING_ROOT / "Invoke-TauriLifecycle.ps1"
BUILD_SCRIPT = PACKAGING_ROOT / "Build-TauriRelease.ps1"
PAYLOAD_SCRIPT = PACKAGING_ROOT / "New-WindowsReleasePayload.ps1"
RUNTIME_INITIALIZER = PACKAGING_ROOT / "bootstrap" / "Initialize-MtsRuntime.ps1"
MODEL_MANAGER_ENTRYPOINT = PACKAGING_ROOT / "bootstrap" / "Manage-MtsModels.ps1"
WINDOWS_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "tauri-windows-release.yml"
TAURI_ICON = REPOSITORY_ROOT / "apps" / "desktop" / "src-tauri" / "icons" / "icon.png"
FIXTURE_V1 = PACKAGING_ROOT / "fixtures" / "payload-v1"
FIXTURE_V2 = PACKAGING_ROOT / "fixtures" / "payload-v2"
APP_ID = "studio.mediatranscribe.desktop"
ENTRYPOINT = "media-transcribe-studio.exe"


@pytest.fixture
def packaging_workspace() -> Path:
    root = REPOSITORY_ROOT / ".codex" / "tp" / uuid.uuid4().hex[:10]
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _powershell() -> str:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    executable = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    assert executable.is_file(), f"Windows PowerShell is required: {executable}"
    return str(executable)


def _run_script(
    script: Path,
    arguments: list[str],
    *,
    expect_success: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    command = [
        _powershell(),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        *arguments,
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    if expect_success:
        assert completed.returncode == 0, (
            f"command failed: {command}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        return completed, payload

    assert completed.returncode != 0, (
        f"command unexpectedly succeeded: {command}\nstdout={completed.stdout}"
    )
    payload = json.loads(completed.stderr.strip().splitlines()[-1])
    assert payload["ok"] is False
    return completed, payload


def _run_powershell(command: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    assert completed.returncode == 0, (
        f"PowerShell command failed\nstdout={completed.stdout}\nstderr={completed.stderr}"
    )
    return completed


@pytest.fixture
def trusted_test_publisher(packaging_workspace: Path) -> str:
    subject = f"CN=MediaTranscribe Studio Packaging Test {uuid.uuid4().hex}"
    created = _run_powershell(
        "$ErrorActionPreference='Stop';"
        f"$certificate=New-SelfSignedCertificate -Type CodeSigningCert -Subject '{subject}' "
        "-CertStoreLocation 'Cert:\\CurrentUser\\My' -KeyAlgorithm RSA -KeyLength 2048 "
        "-HashAlgorithm SHA256 -KeyExportPolicy Exportable "
        "-NotAfter ([DateTimeOffset]::UtcNow.AddDays(7).UtcDateTime);"
        "$certificate.Thumbprint"
    )
    thumbprint = created.stdout.strip().upper()
    assert len(thumbprint) == 40
    try:
        yield thumbprint
    finally:
        _run_powershell(
            "$ErrorActionPreference='SilentlyContinue';"
            f"Remove-Item -LiteralPath 'Cert:\\CurrentUser\\My\\{thumbprint}' -Force;"
        )


def _create_release(
    root: Path,
    version: str,
    *,
    fixture: Path,
    minimum_version: str | None = None,
    readable_min: int = 1,
    readable_max: int = 1,
    write_version: int = 1,
) -> Path:
    release = root / f"release-{version}"
    arguments = [
        "-PayloadDirectory",
        str(fixture),
        "-OutputDirectory",
        str(release),
        "-Version",
        version,
        "-EntryPoint",
        ENTRYPOINT,
        "-SourceDateEpoch",
        "1700000000",
        "-DataSchemaReadableMin",
        str(readable_min),
        "-DataSchemaReadableMax",
        str(readable_max),
        "-DataSchemaWriteVersion",
        str(write_version),
        "-AllowUnsignedDevelopment",
    ]
    if minimum_version is not None:
        arguments.extend(["-MinInstalledVersion", minimum_version])
    _run_script(NEW_RELEASE_SCRIPT, arguments)
    return release


def _create_signed_release(root: Path, version: str, thumbprint: str) -> Path:
    payload = root / f"signed-payload-{version}"
    payload.mkdir(parents=True)
    entrypoint = payload / ENTRYPOINT
    _run_powershell(
        "$ErrorActionPreference='Stop';"
        "Add-Type -TypeDefinition 'public static class Program { "
        "public static void Main() { } }' -Language CSharp "
        f"-OutputAssembly '{entrypoint}' -OutputType ConsoleApplication"
    )
    worker = payload / "backend" / "worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# signed release worker\n", encoding="utf-8")
    _run_powershell(
        "$ErrorActionPreference='Stop';"
        f"$certificate=Get-Item -LiteralPath 'Cert:\\CurrentUser\\My\\{thumbprint}';"
        f"$signature=Set-AuthenticodeSignature -LiteralPath '{entrypoint}' "
        "-Certificate $certificate -HashAlgorithm SHA256;"
        "if($signature.Status -notin @('Valid','UnknownError')){"
        "throw \"Fixture PE signature was not created: $($signature.Status)\"};"
        f"if($signature.SignerCertificate.Thumbprint -ne '{thumbprint}'){{"
        "throw \"Fixture PE signer thumbprint does not match the test publisher.\"}"
    )

    release = root / f"signed-release-{version}"
    _run_script(
        NEW_RELEASE_SCRIPT,
        [
            "-PayloadDirectory",
            str(payload),
            "-OutputDirectory",
            str(release),
            "-Version",
            version,
            "-EntryPoint",
            ENTRYPOINT,
            "-SourceDateEpoch",
            "1700000000",
            "-PublisherThumbprint",
            thumbprint,
        ],
    )
    return release


def _lifecycle(
    action: str,
    *,
    install_root: Path | None = None,
    data_root: Path | None = None,
    release: Path | None = None,
    allow_downgrade: bool = False,
    dry_run: bool = False,
    allow_unsigned: bool = True,
    expected_publisher_thumbprint: str | None = None,
    expect_success: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    arguments = ["-Action", action, "-ExpectedAppId", APP_ID]
    if release is not None:
        arguments.extend(["-ReleaseDirectory", str(release)])
    if install_root is not None:
        arguments.extend(["-InstallRoot", str(install_root)])
    if data_root is not None:
        arguments.extend(["-DataRoot", str(data_root)])
    if allow_unsigned:
        arguments.append("-AllowUnsignedDevelopment")
    if expected_publisher_thumbprint is not None:
        arguments.extend(["-ExpectedPublisherThumbprint", expected_publisher_thumbprint])
    if allow_downgrade:
        arguments.append("-AllowDowngrade")
    if dry_run:
        arguments.append("-DryRun")
    return _run_script(
        LIFECYCLE_SCRIPT,
        arguments,
        expect_success=expect_success,
    )


def _read_enveloped_record(path: Path) -> dict[str, object]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    record_json = envelope["recordJson"]
    assert hashlib.sha256(record_json.encode("utf-8")).hexdigest() == envelope["recordSha256"]
    return json.loads(record_json)


def _write_enveloped_record(path: Path, record_contract: str, record: dict[str, object]) -> None:
    record_json = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    envelope = {
        "contract": "mts-tauri-record-envelope/v1",
        "recordContract": record_contract,
        "recordSha256": hashlib.sha256(record_json.encode("utf-8")).hexdigest(),
        "recordJson": record_json,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_manifest_checksum(release: Path) -> None:
    manifest_path = release / "release-manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (release / "release-manifest.json.sha256").write_text(digest + "\n", encoding="utf-8")


def test_contract_schemas_and_packaging_scripts_are_present() -> None:
    release_schema = json.loads((PACKAGING_ROOT / "release-manifest.schema.json").read_text("utf-8"))
    state_schema = json.loads((PACKAGING_ROOT / "install-state.schema.json").read_text("utf-8"))

    Draft202012Validator.check_schema(release_schema)
    Draft202012Validator.check_schema(state_schema)
    assert release_schema["properties"]["schemaVersion"]["const"] == "1.1.0"
    assert release_schema["properties"]["contract"]["const"] == "mts-tauri-release/v1"
    assert release_schema["properties"]["trust"]["oneOf"]
    assert state_schema["properties"]["contract"]["const"] == "mts-tauri-record-envelope/v1"
    assert NEW_RELEASE_SCRIPT.is_file()
    assert LIFECYCLE_SCRIPT.is_file()
    assert BUILD_SCRIPT.is_file()
    assert PAYLOAD_SCRIPT.is_file()
    assert RUNTIME_INITIALIZER.is_file()
    assert MODEL_MANAGER_ENTRYPOINT.is_file()
    assert WINDOWS_WORKFLOW.is_file()
    assert TAURI_ICON.is_file()
    icon_bytes = TAURI_ICON.read_bytes()
    assert icon_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", icon_bytes[16:24]) >= (32, 32)
    assert FIXTURE_V1.is_dir()
    assert FIXTURE_V2.is_dir()


def test_build_plan_is_locked_version_coherent_and_non_mutating(
    packaging_workspace: Path,
) -> None:
    output = packaging_workspace / "must-not-exist"
    _, result = _run_script(
        BUILD_SCRIPT,
        [
            "-ProjectRoot",
            str(REPOSITORY_ROOT),
            "-OutputDirectory",
            str(output),
            "-AllowUnsignedDevelopment",
            "-DryRun",
        ],
    )

    assert result["dryRun"] is True
    assert result["version"] == "0.1.0"
    assert result["commands"] == [
        "npm ci",
        "npm run tauri -- build --ci --target x86_64-pc-windows-msvc --bundles nsis,msi --config <temporary-windows-tauri-overlay.json> -- --locked",
    ]
    assert len(result["lockedInputs"]) == 2
    assert all(item["size"] > 0 for item in result["lockedInputs"])
    assert all(len(item["sha256"]) == 64 for item in result["lockedInputs"])
    bootstrap = result["runtimeBootstrap"]
    assert bootstrap["bundledPythonRuntime"] is False
    assert bootstrap["bundledModelArtifacts"] is False
    destinations = {entry["destination"] for entry in bootstrap["files"]}
    assert "production.config.example.json" in destinations
    assert "production.config.remote.example.json" in destinations
    assert "configs/llm-provider-presets.v1.json" in destinations
    assert "configs/model-catalog.v1.json" in destinations
    assert "bootstrap/Manage-MtsModels.ps1" in destinations
    assert "bootstrap/runtime-bootstrap.v1.json" in destinations
    assert result["portableArchivePath"].endswith(".portable.zip")
    assert result["runtimeInjection"] == {
        "destination": "resources/mts-runtime",
        "strategy": "tauri-bundle-resources-overlay",
        "bundledPythonRuntime": False,
        "bundledModelArtifacts": False,
    }
    assert not output.exists()


def test_windows_payload_bundles_bootstrap_without_model_weights(
    packaging_workspace: Path,
) -> None:
    executable = packaging_workspace / ENTRYPOINT
    executable.write_bytes(b"signed executable fixture")
    payload = packaging_workspace / "payload"

    _, result = _run_script(
        PAYLOAD_SCRIPT,
        [
            "-ProjectRoot",
            str(REPOSITORY_ROOT),
            "-ApplicationExecutable",
            str(executable),
            "-OutputDirectory",
            str(payload),
        ],
    )

    assert result["bundledPythonRuntime"] is False
    assert result["bundledModelArtifacts"] is False
    assert (payload / ENTRYPOINT).read_bytes() == executable.read_bytes()
    assert (payload / "backend" / "worker.py").is_file()
    assert (payload / "contracts" / "semantic-job-arbitration.schema.json").is_file()
    assert (payload / "reporting" / "report_document_assembler.py").is_file()
    assert (payload / "pdf-renderer" / "target" / "pdf-renderer.jar").is_file()
    assert (payload / "production.config.example.json").is_file()
    assert (payload / "production.config.remote.example.json").is_file()
    assert (payload / "configs" / "llm-provider-presets.v1.json").is_file()
    assert (payload / "configs" / "model-catalog.v1.json").is_file()
    assert not (payload / "local-model-registry.json").exists()
    assert not (payload / "production-models.lock.json").exists()
    assert (payload / "tools" / "model_manager.py").is_file()
    assert (payload / "tools" / "pyannote_runtime.py").is_file()
    packaged_text = "\n".join(
        path.read_text("utf-8", errors="replace")
        for path in payload.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".json", ".ps1", ".md"}
    )
    assert '"apiKey"' not in packaged_text
    assert '"authorization"' not in packaged_text.casefold()
    assert "sk-" not in packaged_text
    assert "?api_key=" not in packaged_text.casefold()
    profile = json.loads(
        (payload / "bootstrap" / "runtime-bootstrap.v1.json").read_text("utf-8")
    )
    assert profile["bundledModelArtifacts"] is False
    assert profile["defaultModelRootPolicy"] == [
        "MTS_MODEL_ROOT",
        "D:/models",
        "LOCALAPPDATA",
    ]
    assert not (payload / "models").exists()
    assert not list(payload.rglob("*.safetensors"))
    assert not list(payload.rglob("*.gguf"))
    packaged_config = json.loads(
        (payload / "production.config.example.json").read_text("utf-8")
    )
    assert packaged_config["paths"] == {
        "allowedInputRoots": ["inputs"],
        "allowedOutputRoot": "exports",
        "cacheRoot": "cache",
    }
    assert "D:/" not in (payload / "production.config.example.json").read_text("utf-8")


def test_windows_runtime_only_plan_excludes_desktop_executable(
    packaging_workspace: Path,
) -> None:
    _, result = _run_script(
        PAYLOAD_SCRIPT,
        [
            "-ProjectRoot",
            str(REPOSITORY_ROOT),
            "-OutputDirectory",
            str(packaging_workspace / "runtime-only"),
            "-RuntimeOnly",
            "-DryRun",
        ],
    )
    assert result["payloadKind"] == "runtime-only"
    assert result["entrypoint"] is None
    assert all(item["destination"] != ENTRYPOINT for item in result["files"])


def test_packaged_bootstrap_entrypoints_are_non_mutating_in_dry_run(
    packaging_workspace: Path,
) -> None:
    executable = packaging_workspace / ENTRYPOINT
    executable.write_bytes(b"signed executable fixture")
    payload = packaging_workspace / "payload"
    _run_script(
        PAYLOAD_SCRIPT,
        [
            "-ProjectRoot",
            str(REPOSITORY_ROOT),
            "-ApplicationExecutable",
            str(executable),
            "-OutputDirectory",
            str(payload),
        ],
    )
    data_root = packaging_workspace / "must-not-create-data"
    model_root = packaging_workspace / "must-not-create-models"
    python = Path(sys.executable).resolve()

    _, initialized = _run_script(
        payload / "bootstrap" / "Initialize-MtsRuntime.ps1",
        [
            "-AppRoot",
            str(payload),
            "-DataRoot",
            str(data_root),
            "-ModelRoot",
            str(model_root),
            "-WorkerPython",
            str(python),
            "-DryRun",
        ],
    )
    assert initialized["bundledModelArtifacts"] is False
    assert initialized["modelRoot"] == str(model_root.resolve())
    assert not data_root.exists()
    assert not model_root.exists()

    _, manager = _run_script(
        payload / "bootstrap" / "Manage-MtsModels.ps1",
        [
            "-AppRoot",
            str(payload),
            "-WorkerPython",
            str(python),
            "-ModelRoot",
            str(model_root),
            "-DryRun",
            "list",
        ],
    )
    assert manager["dryRun"] is True
    assert manager["arguments"][-1] == "list"
    assert not model_root.exists()

    _, applied = _run_script(
        payload / "bootstrap" / "Initialize-MtsRuntime.ps1",
        [
            "-AppRoot",
            str(payload),
            "-DataRoot",
            str(data_root),
            "-ModelRoot",
            str(model_root),
            "-WorkerPython",
            str(python),
        ],
    )
    assert applied["dryRun"] is False
    bound_config = json.loads(
        (data_root / "config" / "production.config.json").read_text("utf-8")
    )
    assert bound_config["paths"]["allowedInputRoots"] == [
        str((data_root / "inputs").resolve()).replace("\\", "/")
    ]
    assert bound_config["paths"]["allowedOutputRoot"] == str(
        (data_root / "exports").resolve()
    ).replace("\\", "/")
    assert bound_config["paths"]["cacheRoot"] == str(
        (data_root / "cache").resolve()
    ).replace("\\", "/")
    assert str(model_root.resolve()).replace("\\", "/") in json.dumps(
        bound_config["models"], ensure_ascii=False
    )
    worker_hint = data_root / "config" / "worker-python.path"
    assert worker_hint.is_file()
    assert worker_hint.read_text("utf-8").strip() == applied["workerPython"]


def test_windows_payload_resources_are_hash_ledgered_in_release_manifest(
    packaging_workspace: Path,
) -> None:
    executable = packaging_workspace / ENTRYPOINT
    executable.write_bytes(b"signed executable fixture")
    payload = packaging_workspace / "payload"
    _run_script(
        PAYLOAD_SCRIPT,
        [
            "-ProjectRoot",
            str(REPOSITORY_ROOT),
            "-ApplicationExecutable",
            str(executable),
            "-OutputDirectory",
            str(payload),
        ],
    )
    release = packaging_workspace / "release"
    _run_script(
        NEW_RELEASE_SCRIPT,
        [
            "-PayloadDirectory",
            str(payload),
            "-OutputDirectory",
            str(release),
            "-Version",
            "1.0.0",
            "-EntryPoint",
            ENTRYPOINT,
            "-SourceDateEpoch",
            "1700000000",
            "-AllowUnsignedDevelopment",
        ],
    )
    manifest = json.loads((release / "release-manifest.json").read_text("utf-8"))
    paths = {entry["path"] for entry in manifest["payload"]["files"]}
    assert "bootstrap/runtime-bootstrap.v1.json" in paths
    assert "configs/llm-provider-presets.v1.json" in paths
    assert "configs/model-catalog.v1.json" in paths
    assert "tools/model_manager.py" in paths
    assert "tools/pyannote_runtime.py" in paths
    assert manifest["payload"]["fileCount"] == len(paths)
    _lifecycle("Validate", release=release)


def test_windows_skip_compile_emits_reproducible_portable_archive(
    packaging_workspace: Path,
) -> None:
    target = packaging_workspace / "tauri-target"
    release_root = target / "x86_64-pc-windows-msvc" / "release"
    (release_root / "bundle" / "nsis").mkdir(parents=True)
    (release_root / "bundle" / "msi").mkdir(parents=True)
    (release_root / ENTRYPOINT).write_bytes(b"native executable fixture")
    (release_root / "bundle" / "nsis" / "MediaTranscribeStudio-setup.exe").write_bytes(
        b"nsis installer fixture"
    )
    (release_root / "bundle" / "msi" / "MediaTranscribeStudio.msi").write_bytes(
        b"msi installer fixture"
    )

    archives: list[Path] = []
    for suffix in ("a", "b"):
        output = packaging_workspace / f"release-{suffix}"
        _, result = _run_script(
            BUILD_SCRIPT,
            [
                "-ProjectRoot",
                str(REPOSITORY_ROOT),
                "-TargetDirectory",
                str(target),
                "-OutputDirectory",
                str(output),
                "-AllowUnsignedDevelopment",
                "-SkipCompile",
                "-SourceDateEpoch",
                "1700000000",
            ],
        )
        archive = Path(result["portableArchive"]["path"])
        assert archive.is_file()
        checksum = Path(result["portableArchive"]["checksumPath"])
        assert checksum.read_text("utf-8").strip() == hashlib.sha256(
            archive.read_bytes()
        ).hexdigest()
        archives.append(archive)
        with zipfile.ZipFile(archive) as opened:
            names = set(opened.namelist())
        assert "release-manifest.json" in names
        assert "payload/backend/worker.py" in names
        assert "payload/media-transcribe-studio.exe" in names
        assert "installers/msi/MediaTranscribeStudio.msi" in names
        assert not any(name.casefold().endswith((".gguf", ".safetensors")) for name in names)

    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_release_manifest_is_reproducible_for_identical_inputs(
    packaging_workspace: Path,
) -> None:
    release_a = _create_release(packaging_workspace / "a", "1.0.0", fixture=FIXTURE_V1)
    release_b = _create_release(packaging_workspace / "b", "1.0.0", fixture=FIXTURE_V1)

    assert (release_a / "release-manifest.json").read_bytes() == (
        release_b / "release-manifest.json"
    ).read_bytes()
    assert (release_a / "release-manifest.json.sha256").read_text("utf-8") == (
        release_b / "release-manifest.json.sha256"
    ).read_text("utf-8")
    release_schema = json.loads((PACKAGING_ROOT / "release-manifest.schema.json").read_text("utf-8"))
    Draft202012Validator(release_schema).validate(
        json.loads((release_a / "release-manifest.json").read_text("utf-8"))
    )


def test_unsigned_fixture_requires_explicit_development_consent(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)

    _, error = _lifecycle(
        "Validate",
        release=release,
        allow_unsigned=False,
        expect_success=False,
    )

    assert "AllowUnsignedDevelopment" in str(error["error"])


def test_signed_release_requires_fixed_publisher_and_preserves_manifest_signature(
    packaging_workspace: Path,
    trusted_test_publisher: str,
) -> None:
    release = _create_signed_release(
        packaging_workspace,
        "1.0.0",
        trusted_test_publisher,
    )
    signature_path = release / "release-manifest.json.p7s"
    assert signature_path.is_file()
    assert signature_path.stat().st_size > 0
    manifest = json.loads((release / "release-manifest.json").read_text("utf-8"))
    assert manifest["schemaVersion"] == "1.1.0"
    assert manifest["trust"]["publisherThumbprint"] == trusted_test_publisher
    assert manifest["trust"]["manifestSignature"] == {
        "path": "release-manifest.json.p7s",
        "format": "cms-detached",
        "digestAlgorithm": "sha256",
    }

    _, missing_anchor = _lifecycle(
        "Validate",
        release=release,
        allow_unsigned=False,
        expect_success=False,
    )
    assert "fixed publisher trust anchor" in str(missing_anchor["error"])

    _, validated = _lifecycle(
        "Validate",
        release=release,
        allow_unsigned=False,
        expected_publisher_thumbprint=trusted_test_publisher,
    )
    assert validated["trustMode"] == "authenticode"

    install_root = packaging_workspace / "signed-install"
    data_root = packaging_workspace / "signed-data"
    _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
        allow_unsigned=False,
        expected_publisher_thumbprint=trusted_test_publisher,
    )
    assert (install_root / "app" / ".mts-release" / signature_path.name).is_file()
    _, status = _lifecycle(
        "Status",
        install_root=install_root,
        data_root=data_root,
        allow_unsigned=False,
        expected_publisher_thumbprint=trusted_test_publisher,
    )
    assert status["verified"] is True


def test_signed_release_rejects_missing_detached_manifest_signature(
    packaging_workspace: Path,
    trusted_test_publisher: str,
) -> None:
    release = _create_signed_release(
        packaging_workspace,
        "1.0.0",
        trusted_test_publisher,
    )
    (release / "release-manifest.json.p7s").unlink()

    _, error = _lifecycle(
        "Validate",
        release=release,
        allow_unsigned=False,
        expected_publisher_thumbprint=trusted_test_publisher,
        expect_success=False,
    )
    assert "publisher signature is missing" in str(error["error"])


def test_signed_release_rejects_payload_tamper_with_recomputed_manifest(
    packaging_workspace: Path,
    trusted_test_publisher: str,
) -> None:
    release = _create_signed_release(
        packaging_workspace,
        "1.0.0",
        trusted_test_publisher,
    )
    worker = release / "payload" / "backend" / "worker.py"
    worker.write_text("# attacker replacement\n", encoding="utf-8")
    manifest_path = release / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    worker_record = next(
        item for item in manifest["payload"]["files"] if item["path"] == "backend/worker.py"
    )
    worker_record["size"] = worker.stat().st_size
    worker_record["sha256"] = hashlib.sha256(worker.read_bytes()).hexdigest()
    manifest["payload"]["totalBytes"] = sum(
        int(item["size"]) for item in manifest["payload"]["files"]
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _rewrite_manifest_checksum(release)

    _, error = _lifecycle(
        "Validate",
        release=release,
        allow_unsigned=False,
        expected_publisher_thumbprint=trusted_test_publisher,
        expect_success=False,
    )
    assert "publisher signature is invalid" in str(error["error"])


def test_signed_release_rejects_entrypoint_hash_mismatch_before_install(
    packaging_workspace: Path,
    trusted_test_publisher: str,
) -> None:
    release = _create_signed_release(
        packaging_workspace,
        "1.0.0",
        trusted_test_publisher,
    )
    entrypoint = release / "payload" / ENTRYPOINT
    tampered = bytearray(entrypoint.read_bytes())
    tampered[-1] ^= 0x01
    entrypoint.write_bytes(tampered)
    install_root = packaging_workspace / "signed-tamper-install"
    data_root = packaging_workspace / "signed-tamper-data"

    _, error = _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
        allow_unsigned=False,
        expected_publisher_thumbprint=trusted_test_publisher,
        expect_success=False,
    )

    assert "payload sha-256 mismatch" in str(error["error"]).lower()
    assert not (install_root / ".mts-control" / "install-state.json").exists()
    assert not (install_root / "app").exists()


def test_release_creation_rejects_protected_output_anchor() -> None:
    protected_anchor = Path(os.environ["USERPROFILE"])

    _, error = _run_script(
        NEW_RELEASE_SCRIPT,
        [
            "-PayloadDirectory",
            str(FIXTURE_V1),
            "-OutputDirectory",
            str(protected_anchor),
            "-Version",
            "1.0.0",
            "-EntryPoint",
            ENTRYPOINT,
            "-SourceDateEpoch",
            "1700000000",
            "-AllowUnsignedDevelopment",
            "-DryRun",
        ],
        expect_success=False,
    )

    assert "protected anchor path" in str(error["error"])


def test_install_dry_run_validates_but_writes_nothing(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"

    _, result = _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
        dry_run=True,
    )

    assert result["dryRun"] is True
    assert result["targetVersion"] == "1.0.0"
    assert result["userDataPolicy"] == "preserve"
    assert "atomically-commit-install-state" in result["operations"]
    assert not install_root.exists()
    assert not data_root.exists()


def test_install_rejects_overlapping_managed_and_user_data_paths(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    install_root = packaging_workspace / "install"

    _, error = _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=install_root / "user-data",
        expect_success=False,
    )

    assert "must not overlap" in str(error["error"])
    assert not install_root.exists()


def test_manifest_path_traversal_is_rejected_before_install(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    manifest_path = release / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["payload"]["files"][0]["path"] = "../escape.exe"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _rewrite_manifest_checksum(release)

    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"
    _, error = _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
        expect_success=False,
    )

    assert "unsafe segment" in str(error["error"])
    assert not (packaging_workspace / "escape.exe").exists()
    assert not install_root.exists()


def test_payload_hash_mismatch_fails_closed_without_install_state(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    (release / "payload" / ENTRYPOINT).write_text("tampered", encoding="utf-8")
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"

    _, error = _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
        expect_success=False,
    )

    assert "mismatch" in str(error["error"]).lower()
    assert not (install_root / ".mts-control" / "install-state.json").exists()
    assert not (install_root / "app").exists()


def test_release_with_undeclared_root_file_is_rejected(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    (release / "unexpected-bootstrap.ps1").write_text("Write-Output unsafe", encoding="utf-8")

    _, error = _lifecycle(
        "Validate",
        release=release,
        expect_success=False,
    )

    assert "undeclared entry" in str(error["error"])


def test_upgrade_atomically_switches_to_verified_release_and_keeps_backup_and_data(
    packaging_workspace: Path,
) -> None:
    release_v1 = _create_release(
        packaging_workspace / "releases", "1.0.0", fixture=FIXTURE_V1
    )
    release_v2 = _create_release(
        packaging_workspace / "releases",
        "2.0.0",
        fixture=FIXTURE_V2,
        minimum_version="1.0.0",
    )
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"

    _lifecycle(
        "Install",
        release=release_v1,
        install_root=install_root,
        data_root=data_root,
    )
    sentinel = data_root / "profiles" / "speaker-locks.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('{"keep":true}', encoding="utf-8")

    _, upgraded = _lifecycle(
        "Upgrade",
        release=release_v2,
        install_root=install_root,
        data_root=data_root,
    )
    _, status = _lifecycle(
        "Status",
        install_root=install_root,
        data_root=data_root,
    )

    state = _read_enveloped_record(install_root / ".mts-control" / "install-state.json")
    state_envelope = json.loads(
        (install_root / ".mts-control" / "install-state.json").read_text("utf-8")
    )
    state_schema = json.loads((PACKAGING_ROOT / "install-state.schema.json").read_text("utf-8"))
    Draft202012Validator(state_schema).validate(state_envelope)
    rollback_path = Path(str(state["rollback"]["path"]))
    assert upgraded["status"] == "committed-and-verified"
    assert status["version"] == "2.0.0"
    assert status["rollbackAvailable"] is True
    assert "fixture release 2.0.0" in (install_root / "app" / ENTRYPOINT).read_text("utf-8")
    assert "fixture release 1.0.0" in (rollback_path / ENTRYPOINT).read_text("utf-8")
    assert sentinel.read_text("utf-8") == '{"keep":true}'
    assert not (install_root / ".mts-control" / "transaction.json").exists()


def test_version_downgrade_is_rejected_by_default_and_explicitly_allowed(
    packaging_workspace: Path,
) -> None:
    release_v1 = _create_release(
        packaging_workspace / "releases", "1.0.0", fixture=FIXTURE_V1
    )
    release_v2 = _create_release(
        packaging_workspace / "releases", "2.0.0", fixture=FIXTURE_V2
    )
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"

    _lifecycle(
        "Install",
        release=release_v2,
        install_root=install_root,
        data_root=data_root,
    )
    _, error = _lifecycle(
        "Upgrade",
        release=release_v1,
        install_root=install_root,
        data_root=data_root,
        expect_success=False,
    )
    assert "downgrade is rejected" in str(error["error"])
    _, status_before = _lifecycle("Status", install_root=install_root, data_root=data_root)
    assert status_before["version"] == "2.0.0"

    _, downgraded = _lifecycle(
        "Upgrade",
        release=release_v1,
        install_root=install_root,
        data_root=data_root,
        allow_downgrade=True,
    )
    assert downgraded["version"] == "1.0.0"


def test_explicit_rollback_restores_backup_and_preserves_user_data(
    packaging_workspace: Path,
) -> None:
    release_v1 = _create_release(
        packaging_workspace / "releases", "1.0.0", fixture=FIXTURE_V1
    )
    release_v2 = _create_release(
        packaging_workspace / "releases", "2.0.0", fixture=FIXTURE_V2
    )
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"

    _lifecycle(
        "Install",
        release=release_v1,
        install_root=install_root,
        data_root=data_root,
    )
    sentinel = data_root / "transcripts" / "keep.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("irreplaceable", encoding="utf-8")
    _lifecycle(
        "Upgrade",
        release=release_v2,
        install_root=install_root,
        data_root=data_root,
    )

    _, error = _lifecycle(
        "Rollback",
        install_root=install_root,
        data_root=data_root,
        expect_success=False,
    )
    assert "requires explicit -AllowDowngrade" in str(error["error"])

    _, rolled_back = _lifecycle(
        "Rollback",
        install_root=install_root,
        data_root=data_root,
        allow_downgrade=True,
    )
    assert rolled_back["version"] == "1.0.0"
    assert sentinel.read_text("utf-8") == "irreplaceable"
    assert "fixture release 1.0.0" in (install_root / "app" / ENTRYPOINT).read_text("utf-8")


def test_recovery_restores_verified_backup_after_interrupted_atomic_switch(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"
    _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
    )

    state_path = install_root / ".mts-control" / "install-state.json"
    previous_state = _read_enveloped_record(state_path)
    transaction_id = "fixtureinterruption0000000000000001"
    transaction_root = (
        install_root / ".mts-control" / "transactions" / transaction_id
    )
    stage_path = transaction_root / "staged-app"
    transaction_root.mkdir(parents=True)
    backup_path = install_root / "backups" / f"recovery--{transaction_id}"
    shutil.move(str(install_root / "app"), str(backup_path))

    journal = {
        "schemaVersion": "1.0.0",
        "contract": "mts-tauri-transaction/v1",
        "transactionId": transaction_id,
        "action": "Upgrade",
        "phase": "current-backed-up",
        "installRoot": str(install_root.resolve()),
        "dataRoot": str(data_root.resolve()),
        "stagePath": str(stage_path.resolve()),
        "backupPath": str(backup_path.resolve()),
        "targetManifestSha256": previous_state["current"]["manifestSha256"],
        "previousState": previous_state,
    }
    _write_enveloped_record(
        install_root / ".mts-control" / "transaction.json",
        "mts-tauri-transaction/v1",
        journal,
    )

    _, recovered = _lifecycle(
        "Recover",
        install_root=install_root,
        data_root=data_root,
    )
    _, status = _lifecycle("Status", install_root=install_root, data_root=data_root)

    assert recovered["status"] == "previous-release-restored"
    assert status["version"] == "1.0.0"
    assert (install_root / "app" / ENTRYPOINT).is_file()
    assert not (install_root / ".mts-control" / "transaction.json").exists()


def test_tampered_install_state_blocks_status_and_mutating_operations(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"
    _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
    )
    state_path = install_root / ".mts-control" / "install-state.json"
    state_path.write_text(state_path.read_text("utf-8").replace("1.0.0", "9.9.9", 1), "utf-8")

    _, status_error = _lifecycle(
        "Status",
        install_root=install_root,
        data_root=data_root,
        expect_success=False,
    )
    _, uninstall_error = _lifecycle(
        "Uninstall",
        install_root=install_root,
        data_root=data_root,
        expect_success=False,
    )

    assert "checksum mismatch" in str(status_error["error"])
    assert "checksum mismatch" in str(uninstall_error["error"])
    assert (install_root / "app" / ENTRYPOINT).is_file()


def test_uninstall_removes_only_managed_application_and_preserves_user_data(
    packaging_workspace: Path,
) -> None:
    release = _create_release(packaging_workspace, "1.0.0", fixture=FIXTURE_V1)
    install_root = packaging_workspace / "install"
    data_root = packaging_workspace / "data"
    _lifecycle(
        "Install",
        release=release,
        install_root=install_root,
        data_root=data_root,
    )
    user_file = data_root / "projects" / "meeting.json"
    user_file.parent.mkdir(parents=True)
    user_file.write_text('{"speakerLocks":5}', encoding="utf-8")
    operator_file = install_root / "operator-note.txt"
    operator_file.write_text("unmanaged", encoding="utf-8")

    _, uninstalled = _lifecycle(
        "Uninstall",
        install_root=install_root,
        data_root=data_root,
    )

    assert uninstalled["status"] == "application-removed-user-data-preserved"
    assert user_file.read_text("utf-8") == '{"speakerLocks":5}'
    assert operator_file.read_text("utf-8") == "unmanaged"
    assert not (install_root / "app").exists()
    assert not (install_root / ".mts-control").exists()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("1.0.0", "1.0.0", 0),
        ("2.0.0", "1.9.9", 1),
        ("1.0.0-alpha.1", "1.0.0-alpha.2", -1),
        ("1.0.0", "1.0.0-rc.1", 1),
    ],
)
def test_semver_comparison_contract(left: str, right: str, expected: int) -> None:
    command = (
        f"Import-Module '{PACKAGING_ROOT / 'Mts.TauriPackaging.psm1'}' -Force; "
        f"Compare-MtsSemVer '{left}' '{right}'"
    )
    completed = subprocess.run(
        [
            _powershell(),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert int(completed.stdout.strip()) == expected
