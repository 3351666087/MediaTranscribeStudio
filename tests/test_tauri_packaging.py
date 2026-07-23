from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest
from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_ROOT = REPOSITORY_ROOT / "packaging" / "tauri"
NEW_RELEASE_SCRIPT = PACKAGING_ROOT / "New-TauriRelease.ps1"
LIFECYCLE_SCRIPT = PACKAGING_ROOT / "Invoke-TauriLifecycle.ps1"
BUILD_SCRIPT = PACKAGING_ROOT / "Build-TauriRelease.ps1"
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


def _lifecycle(
    action: str,
    *,
    install_root: Path | None = None,
    data_root: Path | None = None,
    release: Path | None = None,
    allow_downgrade: bool = False,
    dry_run: bool = False,
    allow_unsigned: bool = True,
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
    assert release_schema["properties"]["contract"]["const"] == "mts-tauri-release/v1"
    assert release_schema["properties"]["trust"]["oneOf"]
    assert state_schema["properties"]["contract"]["const"] == "mts-tauri-record-envelope/v1"
    assert NEW_RELEASE_SCRIPT.is_file()
    assert LIFECYCLE_SCRIPT.is_file()
    assert BUILD_SCRIPT.is_file()
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
        "npm run tauri -- build --target x86_64-pc-windows-msvc --bundles nsis,msi",
    ]
    assert len(result["lockedInputs"]) == 2
    assert not output.exists()


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
