#!/usr/bin/env python3
"""Verify a collected Tauri macOS release candidate.

Ledger and Mach-O checks are cross-host. Native DMG, code-signing,
notarization, bootstrap, and launch checks are enabled by the macOS workflow
with ``--require-native`` and related flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Mapping, Sequence
import zipfile

from jsonschema import Draft202012Validator

import build_tauri_macos_release as build


class ReleaseVerificationError(RuntimeError):
    """Raised when a collected candidate does not match its release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(raw: object) -> str:
    value = str(raw or "")
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ReleaseVerificationError(f"Unsafe release-relative path: {value!r}")
    return value


def _transport_checksum(checksum_path: Path, archive_name: str) -> str:
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ReleaseVerificationError(
            f"Transport checksum is missing or unsafe: {checksum_path}"
        )
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ReleaseVerificationError(
            f"Unable to read transport checksum {checksum_path}: {exc}"
        ) from exc
    if len(lines) != 1:
        raise ReleaseVerificationError("Transport checksum must contain exactly one record.")
    match = re.fullmatch(r"([0-9A-Fa-f]{64})[ \t]+\*?([^\r\n]+)", lines[0])
    if match is None or match.group(2) != archive_name:
        raise ReleaseVerificationError(
            "Transport checksum must name the downloaded archive exactly."
        )
    return match.group(1).lower()


def _transport_member_path(raw: str, expected_root: str) -> PurePosixPath:
    value = raw.rstrip("/")
    raw_parts = value.split("/")
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ReleaseVerificationError(f"Unsafe transport archive member: {raw!r}")
    if any(
        part == ".DS_Store" or part == "__MACOSX" or part.startswith("._")
        for part in raw_parts
    ):
        raise ReleaseVerificationError(
            f"Transport archive contains non-portable macOS metadata: {raw!r}"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path.parts[0] != expected_root:
        raise ReleaseVerificationError(
            f"Transport archive member is outside {expected_root!r}: {raw!r}"
        )
    return path


def _transport_link_target(
    member: tarfile.TarInfo,
    member_path: PurePosixPath,
    expected_root: str,
) -> PurePosixPath:
    raw_target = member.linkname
    if not raw_target or "\\" in raw_target or "\x00" in raw_target:
        raise ReleaseVerificationError(
            f"Unsafe transport archive link target: {member.name!r} -> {raw_target!r}"
        )
    target = PurePosixPath(raw_target)
    if target.is_absolute():
        raise ReleaseVerificationError(
            f"Transport archive link target is absolute: {member.name!r} -> {raw_target!r}"
        )
    combined = member_path.parent / target if member.issym() else target
    normalized = PurePosixPath(posixpath.normpath(combined.as_posix()))
    if (
        not normalized.parts
        or normalized.parts[0] == ".."
        or normalized.parts[0] != expected_root
    ):
        raise ReleaseVerificationError(
            f"Transport archive link escapes {expected_root!r}: "
            f"{member.name!r} -> {raw_target!r}"
        )
    return normalized


def _validate_transport_members(archive_path: Path, expected_root: str) -> int:
    seen: set[str] = set()
    paths: list[PurePosixPath] = []
    symlinks: set[PurePosixPath] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise ReleaseVerificationError(
                    f"Transport archive is empty: {archive_path}"
                )
            for member in members:
                path = _transport_member_path(member.name, expected_root)
                folded = path.as_posix().casefold()
                if folded in seen:
                    raise ReleaseVerificationError(
                        f"Transport archive contains a duplicate path: {member.name!r}"
                    )
                seen.add(folded)
                paths.append(path)
                if member.isdev() or member.isfifo() or not (
                    member.isfile()
                    or member.isdir()
                    or member.issym()
                    or member.islnk()
                ):
                    raise ReleaseVerificationError(
                        f"Transport archive contains an unsupported file type: {member.name!r}"
                    )
                if member.issym() or member.islnk():
                    _transport_link_target(member, path, expected_root)
                if member.issym():
                    symlinks.add(path)
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseVerificationError(
            f"Unable to inspect transport archive {archive_path}: {exc}"
        ) from exc

    for path in paths:
        if any(parent in symlinks for parent in path.parents):
            raise ReleaseVerificationError(
                f"Transport archive member is nested below a symlink: {path.as_posix()}"
            )
    return len(paths)


def _extract_transport_archive(
    archive_path: Path,
    checksum_path: Path,
    destination: Path,
    expected_root: str,
) -> tuple[Path, dict[str, object]]:
    source = archive_path.expanduser().resolve()
    if archive_path.expanduser().is_symlink() or not source.is_file():
        raise ReleaseVerificationError(f"Transport archive is missing or unsafe: {archive_path}")
    root_name = _safe_relative(expected_root)
    if "/" in root_name:
        raise ReleaseVerificationError("--expected-release-root must be one directory name.")
    expected_sha256 = _transport_checksum(
        checksum_path.expanduser().resolve(),
        source.name,
    )
    if destination.exists():
        raise ReleaseVerificationError(
            f"Transport extraction destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged_archive = destination.parent / ".verified-macos-release.tar.gz"
    if staged_archive.exists():
        raise ReleaseVerificationError(f"Transport staging path already exists: {staged_archive}")
    shutil.copyfile(source, staged_archive)
    actual_sha256 = _sha256(staged_archive)
    if actual_sha256 != expected_sha256:
        raise ReleaseVerificationError(
            "Transport archive checksum mismatch "
            f"({expected_sha256!r} != {actual_sha256})."
        )
    member_count = _validate_transport_members(staged_archive, root_name)
    destination.mkdir(parents=True, exist_ok=False)
    tar = shutil.which("tar") or "/usr/bin/tar"
    if not Path(tar).is_file():
        raise ReleaseVerificationError("A native tar implementation is required for transport extraction.")
    try:
        completed = subprocess.run(
            [tar, "-xzf", str(staged_archive), "-C", str(destination)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseVerificationError(f"Unable to extract transport archive: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ReleaseVerificationError(f"Unable to extract transport archive: {detail}")
    release_root = destination / root_name
    if release_root.is_symlink() or not release_root.is_dir():
        raise ReleaseVerificationError(
            f"Transport archive did not extract the expected release root: {release_root}"
        )
    return release_root, {
        "archive": str(source),
        "checksum": str(checksum_path.expanduser().resolve()),
        "sha256": actual_sha256,
        "memberCount": member_count,
        "safeExtraction": True,
    }


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"Unable to read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"JSON document must contain an object: {path}")
    return value


def _load_manifest(release_root: Path, schema_path: Path) -> dict[str, object]:
    manifest_path = release_root / "macos-release-manifest.json"
    checksum_path = release_root / "macos-release-manifest.json.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ReleaseVerificationError(f"Release manifest is missing or unsafe: {manifest_path}")
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ReleaseVerificationError(f"Release manifest checksum is missing or unsafe: {checksum_path}")
    expected_checksum = checksum_path.read_text(encoding="ascii").strip().lower()
    actual_checksum = _sha256(manifest_path)
    if expected_checksum != actual_checksum:
        raise ReleaseVerificationError(
            f"Release manifest checksum mismatch ({expected_checksum!r} != {actual_checksum})."
        )
    schema = _read_json_object(schema_path)
    manifest = _read_json_object(manifest_path)
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
    except Exception as exc:
        raise ReleaseVerificationError(f"Release manifest does not satisfy its schema: {exc}") from exc
    return manifest


def _verify_exact_ledger(release_root: Path, manifest: Mapping[str, object]) -> None:
    allowed_root_entries = {
        "artifacts",
        "macos-release-manifest.json",
        "macos-release-manifest.json.sha256",
    }
    actual_root_entries = {path.name for path in release_root.iterdir()}
    if actual_root_entries != allowed_root_entries:
        raise ReleaseVerificationError(
            "Release root contains missing or undeclared entries "
            f"(expected={sorted(allowed_root_entries)}, actual={sorted(actual_root_entries)})."
        )
    artifact_root = release_root / "artifacts"
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ReleaseVerificationError(f"Release artifacts root is missing or unsafe: {artifact_root}")
    try:
        actual = build._hash_path(artifact_root, release_root)
    except build.ReleaseBuildError as exc:
        raise ReleaseVerificationError(str(exc)) from exc
    expected = manifest.get("files")
    if not isinstance(expected, list):
        raise ReleaseVerificationError("Manifest files ledger must be an array.")
    expected_sorted = sorted(expected, key=lambda item: str(item.get("path", "")) if isinstance(item, dict) else "")
    if actual != expected_sorted:
        actual_by_path = {str(item["path"]): item for item in actual}
        expected_by_path = {
            str(item.get("path")): item for item in expected_sorted if isinstance(item, dict)
        }
        missing = sorted(set(expected_by_path).difference(actual_by_path))
        unexpected = sorted(set(actual_by_path).difference(expected_by_path))
        changed = sorted(
            path
            for path in set(actual_by_path).intersection(expected_by_path)
            if actual_by_path[path] != expected_by_path[path]
        )
        raise ReleaseVerificationError(
            "Release artifact ledger mismatch "
            f"(missing={missing}, unexpected={unexpected}, changed={changed})."
        )
    if manifest.get("fileCount") != len(actual):
        raise ReleaseVerificationError("Manifest fileCount does not match the artifact ledger.")
    total_bytes = sum(int(item["size"]) for item in actual)
    if manifest.get("totalBytes") != total_bytes:
        raise ReleaseVerificationError("Manifest totalBytes does not match the artifact ledger.")


def _artifact_records(release_root: Path, artifact_path: Path) -> list[dict[str, object]]:
    try:
        return build._hash_path(artifact_path, release_root)
    except build.ReleaseBuildError as exc:
        raise ReleaseVerificationError(str(exc)) from exc


def _verify_artifact_declarations(
    release_root: Path, manifest: Mapping[str, object]
) -> dict[str, Path]:
    declarations = manifest.get("artifacts")
    if not isinstance(declarations, list) or not declarations:
        raise ReleaseVerificationError("Manifest artifacts must be a non-empty array.")
    found: dict[str, Path] = {}
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ReleaseVerificationError("Each artifact declaration must be an object.")
        bundle = str(declaration.get("bundle", ""))
        relative = _safe_relative(declaration.get("path"))
        if bundle in found:
            raise ReleaseVerificationError(f"Manifest declares duplicate {bundle} artifacts.")
        if not relative.startswith(f"artifacts/{bundle}/"):
            raise ReleaseVerificationError(
                f"Artifact path is outside its declared bundle directory: {relative}"
            )
        artifact = release_root / relative
        if artifact.is_symlink() or not artifact.exists():
            raise ReleaseVerificationError(f"Declared artifact is missing or unsafe: {artifact}")
        records = _artifact_records(release_root, artifact)
        if declaration.get("fileCount") != len(records):
            raise ReleaseVerificationError(f"Artifact fileCount mismatch: {relative}")
        total_bytes = sum(int(item["size"]) for item in records)
        if declaration.get("totalBytes") != total_bytes:
            raise ReleaseVerificationError(f"Artifact totalBytes mismatch: {relative}")
        found[bundle] = artifact
    return found


def _verify_release_identity_and_policy(manifest: Mapping[str, object]) -> None:
    expected_release_id = (
        f"{manifest['appId']}/{manifest['version']}/{manifest['platform']}/"
        f"{manifest['architecture']}/{manifest['channel']}"
    )
    if manifest.get("releaseId") != expected_release_id:
        raise ReleaseVerificationError(
            f"Manifest releaseId is inconsistent ({manifest.get('releaseId')!r} != {expected_release_id!r})."
        )
    if manifest.get("channel") != "stable":
        return
    trust = manifest.get("trust")
    if not isinstance(trust, dict):
        raise ReleaseVerificationError("Stable macOS releases require a trust declaration.")
    identity = str(trust.get("signingIdentity") or "")
    if (
        trust.get("mode") != "codesign"
        or trust.get("notarization") != "stapled"
        or not identity.casefold().startswith("developer id application:")
    ):
        raise ReleaseVerificationError(
            "Stable macOS releases require Developer ID Application signing and stapled notarization."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or any(
        isinstance(item, dict) and item.get("bundle") == "zip" for item in artifacts
    ):
        raise ReleaseVerificationError(
            "Stable macOS releases currently accept app and DMG artifacts only."
        )


def _verify_external_trust_anchor(
    manifest: Mapping[str, object],
    *,
    expected_signing_identity: str | None,
    require_native: bool,
) -> None:
    trust = manifest.get("trust")
    if not isinstance(trust, dict):
        raise ReleaseVerificationError("Manifest trust declaration must be an object.")
    expected = str(expected_signing_identity or "").strip()
    actual = str(trust.get("signingIdentity") or "").strip()
    if expected:
        if trust.get("mode") != "codesign":
            raise ReleaseVerificationError(
                "--expected-signing-identity cannot verify a development-unsigned artifact."
            )
        if actual.casefold() != expected.casefold():
            raise ReleaseVerificationError(
                "Manifest signing identity does not match the externally trusted identity "
                f"({actual!r} != {expected!r})."
            )
    if manifest.get("channel") != "stable":
        return
    if not require_native:
        raise ReleaseVerificationError(
            "Stable macOS releases require --require-native on a macOS host."
        )
    if not expected:
        raise ReleaseVerificationError(
            "Stable macOS releases require --expected-signing-identity as an external trust anchor."
        )


def _runtime_shape(app: Path) -> dict[str, object]:
    runtime = app / "Contents" / "Resources" / "mts-runtime"
    required = (
        "backend/worker.py",
        "bootstrap/runtime-bootstrap.v1.json",
        "bootstrap/Initialize-MtsRuntime.sh",
        "bootstrap/Manage-MtsModels.sh",
        "configs/model-catalog.v1.json",
        "configs/llm-provider-presets.v1.json",
        "tools/pyannote_runtime.py",
        "pdf-renderer/target/pdf-renderer.jar",
    )
    missing = [relative for relative in required if not (runtime / relative).is_file()]
    if missing:
        raise ReleaseVerificationError(f"App runtime is missing required files: {missing}")
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(runtime, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ReleaseVerificationError(f"App runtime contains a directory symlink: {candidate}")
        for name in file_names:
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ReleaseVerificationError(f"App runtime contains a non-regular file: {candidate}")
            if candidate.suffix.casefold() in build.FORBIDDEN_MODEL_EXTENSIONS:
                raise ReleaseVerificationError(f"App runtime contains model/media bytes: {candidate}")
            try:
                build._assert_no_secret_bytes(candidate)
            except build.ReleaseBuildError as exc:
                raise ReleaseVerificationError(str(exc)) from exc
            files.append(candidate)
    if os.name != "nt":
        for relative in (
            "bootstrap/Initialize-MtsRuntime.sh",
            "bootstrap/Manage-MtsModels.sh",
        ):
            if not (runtime / relative).stat().st_mode & 0o111:
                raise ReleaseVerificationError(f"App runtime bootstrap is not executable: {relative}")
    profile = _read_json_object(runtime / "bootstrap" / "runtime-bootstrap.v1.json")
    if profile.get("bundledPythonRuntime") is not False or profile.get("bundledModelArtifacts") is not False:
        raise ReleaseVerificationError("Runtime bootstrap profile makes an invalid bundled-runtime claim.")
    return {
        "root": str(runtime),
        "fileCount": len(files),
        "totalBytes": sum(path.stat().st_size for path in files),
        "bundledPythonRuntime": False,
        "bundledModelArtifacts": False,
    }


def _app_context(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "appId": manifest["appId"],
        "productName": manifest["productName"],
        "version": manifest["version"],
        "architecture": manifest["architecture"],
    }


def _verify_app(
    app: Path,
    manifest: Mapping[str, object],
    *,
    require_native: bool,
) -> tuple[Path, dict[str, object]]:
    context = _app_context(manifest)
    try:
        executable = build._validate_app_bundle(app, context)
    except build.ReleaseBuildError as exc:
        raise ReleaseVerificationError(str(exc)) from exc
    runtime = _runtime_shape(app)
    trust = manifest.get("trust")
    if not isinstance(trust, dict):
        raise ReleaseVerificationError("Manifest trust declaration must be an object.")
    if trust.get("mode") == "codesign":
        if require_native:
            try:
                build._verify_codesign(app, str(trust.get("signingIdentity", "")))
            except build.ReleaseBuildError as exc:
                raise ReleaseVerificationError(str(exc)) from exc
    elif trust.get("mode") != "development-unsigned":
        raise ReleaseVerificationError(f"Unsupported macOS trust mode: {trust.get('mode')!r}")
    if trust.get("notarization") == "stapled":
        if not require_native:
            raise ReleaseVerificationError("A stapled notarization claim requires native macOS verification.")
        try:
            build._verify_stapled_notarization(app)
            build._verify_gatekeeper(app, artifact_type="app")
        except build.ReleaseBuildError as exc:
            raise ReleaseVerificationError(str(exc)) from exc
    return executable, runtime


def _safe_extract_zip(archive_path: Path, destination: Path, expected_app_name: str) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if not members:
            raise ReleaseVerificationError(f"Portable app ZIP is empty: {archive_path}")
        seen: set[str] = set()
        symlinks: dict[str, str] = {}
        regular: list[tuple[zipfile.ZipInfo, str, int]] = []
        for member in members:
            raw = member.filename.rstrip("/")
            if not raw:
                continue
            relative = _safe_relative(raw)
            if relative.casefold() in seen:
                raise ReleaseVerificationError(f"Portable app ZIP has a duplicate path: {relative}")
            seen.add(relative.casefold())
            if PurePosixPath(relative).parts[0] != expected_app_name:
                raise ReleaseVerificationError(f"Portable app ZIP contains data outside {expected_app_name}: {relative}")
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode):
                try:
                    target = archive.read(member).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ReleaseVerificationError(f"ZIP symlink target is not UTF-8: {relative}") from exc
                target_path = PurePosixPath(target)
                resolved = PurePosixPath(relative).parent.joinpath(target_path)
                if target_path.is_absolute() or ".." in resolved.parts:
                    raise ReleaseVerificationError(f"ZIP symlink escapes the app: {relative} -> {target}")
                symlinks[relative] = target
            elif member.is_dir() or file_type in {0, stat.S_IFDIR, stat.S_IFREG}:
                regular.append((member, relative, mode))
            else:
                raise ReleaseVerificationError(f"Portable app ZIP contains an unsupported file type: {relative}")
        symlink_paths = {PurePosixPath(path) for path in symlinks}
        for _, relative, _ in regular:
            parents = PurePosixPath(relative).parents
            if any(parent in symlink_paths for parent in parents):
                raise ReleaseVerificationError(f"ZIP member is nested below a symlink: {relative}")
        destination.mkdir(parents=True, exist_ok=False)
        for member, relative, mode in sorted(regular, key=lambda item: (len(PurePosixPath(item[1]).parts), item[1])):
            target = destination.joinpath(*PurePosixPath(relative).parts)
            if member.is_dir() or stat.S_ISDIR(mode):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            permissions = stat.S_IMODE(mode) or 0o644
            target.chmod(permissions)
        for relative, target_value in sorted(symlinks.items()):
            target = destination.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(target_value)
    app = destination / expected_app_name
    if not app.is_dir() or app.is_symlink():
        raise ReleaseVerificationError(f"Portable app ZIP did not extract {expected_app_name}.")
    return app


def _verify_dmg(
    dmg: Path,
    manifest: Mapping[str, object],
    *,
    require_native: bool,
    retain_app: bool,
) -> tuple[Path | None, dict[str, object] | None]:
    if not require_native:
        return None, None
    if sys.platform != "darwin":
        raise ReleaseVerificationError("--require-native must run on macOS for DMG validation.")
    hdiutil = shutil.which("hdiutil") or "/usr/bin/hdiutil"
    if not Path(hdiutil).is_file():
        raise ReleaseVerificationError("hdiutil is required for native DMG validation.")
    try:
        build._run_native_check([hdiutil, "verify", str(dmg)], timeout=300)
    except build.ReleaseBuildError as exc:
        raise ReleaseVerificationError(str(exc)) from exc
    trust = manifest.get("trust")
    if isinstance(trust, dict) and trust.get("mode") == "codesign":
        try:
            build._verify_codesign(dmg, str(trust.get("signingIdentity", "")))
        except build.ReleaseBuildError as exc:
            raise ReleaseVerificationError(str(exc)) from exc
    mount_root = Path(tempfile.mkdtemp(prefix="mts-verify-dmg-"))
    attached = False
    retained_root: Path | None = None
    retained_app: Path | None = None
    try:
        try:
            build._run_native_check(
                [
                    hdiutil,
                    "attach",
                    "-readonly",
                    "-nobrowse",
                    "-noautoopen",
                    "-mountpoint",
                    str(mount_root),
                    str(dmg),
                ],
                timeout=300,
            )
        except build.ReleaseBuildError as exc:
            raise ReleaseVerificationError(str(exc)) from exc
        attached = True
        expected_name = f"{manifest['productName']}.app"
        apps = [path for path in build._app_candidates(mount_root) if path.name == expected_name]
        if len(apps) != 1:
            raise ReleaseVerificationError(
                f"Mounted DMG must contain exactly one {expected_name}; found {len(apps)}."
            )
        _, runtime = _verify_app(apps[0], manifest, require_native=True)
        if retain_app:
            retained_root = Path(tempfile.mkdtemp(prefix="mts-verify-dmg-app-"))
            retained_app = retained_root / apps[0].name
            build._copy_artifact(apps[0], retained_app)
        if isinstance(manifest.get("trust"), dict) and manifest["trust"].get("notarization") == "stapled":  # type: ignore[index]
            try:
                build._verify_stapled_notarization(dmg)
                build._verify_gatekeeper(dmg, artifact_type="dmg")
            except build.ReleaseBuildError as exc:
                raise ReleaseVerificationError(str(exc)) from exc
        return retained_app, runtime
    except BaseException:
        if retained_root is not None:
            shutil.rmtree(retained_root, ignore_errors=True)
        raise
    finally:
        if attached:
            try:
                build._run_native_check([hdiutil, "detach", str(mount_root)], timeout=120)
            except build.ReleaseBuildError:
                build._run_native_check([hdiutil, "detach", "-force", str(mount_root)], timeout=120)
        shutil.rmtree(mount_root, ignore_errors=True)


def _run_bootstrap(app: Path) -> dict[str, object]:
    initializer = app / "Contents" / "Resources" / "mts-runtime" / "bootstrap" / "Initialize-MtsRuntime.sh"
    temporary_root = Path(tempfile.mkdtemp(prefix="mts-macos-bootstrap-"))
    try:
        data_root = temporary_root / "Application Support" / "MediaTranscribeStudio"
        model_root = temporary_root / "models"
        config = data_root / "config" / "production.config.json"
        completed = subprocess.run(
            [
                str(initializer),
                "--data-root",
                str(data_root),
                "--model-root",
                str(model_root),
                "--worker-python",
                sys.executable,
                "--production-config",
                str(config),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ReleaseVerificationError(f"macOS runtime bootstrap failed: {detail}")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ReleaseVerificationError("macOS runtime bootstrap did not return JSON.") from exc
        if not isinstance(result, dict) or result.get("ok") is not True or result.get("configCreated") is not True:
            raise ReleaseVerificationError(f"macOS runtime bootstrap returned an invalid result: {result!r}")
        document = _read_json_object(config)
        serialized = json.dumps(document, ensure_ascii=True)
        if "D:/" in serialized or "D:\\" in serialized:
            raise ReleaseVerificationError("macOS runtime bootstrap left a Windows drive path in configuration.")
        if document.get("paths", {}).get("allowedInputRoots") != [str(data_root / "inputs")]:  # type: ignore[union-attr]
            raise ReleaseVerificationError("macOS runtime bootstrap did not bind the input root.")
        runtime_root = app / "Contents" / "Resources" / "mts-runtime"
        validation_code = (
            "import json,sys; "
            "sys.path.insert(0, sys.argv[1]); "
            "from backend.production_config import ProductionConfig; "
            "config=ProductionConfig.load(sys.argv[2]); "
            "print(json.dumps({'ok': True, 'mode': config.mode, "
            "'pyannoteMode': config.speaker.pyannote_mode}))"
        )
        validation = subprocess.run(
            [sys.executable, "-c", validation_code, str(runtime_root), str(config)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=60,
        )
        if validation.returncode != 0:
            detail = validation.stderr.strip() or validation.stdout.strip()
            raise ReleaseVerificationError(f"Bootstrapped production config is invalid: {detail}")
        try:
            config_validation = json.loads(validation.stdout)
        except json.JSONDecodeError as exc:
            raise ReleaseVerificationError("Production config validation did not return JSON.") from exc
        if config_validation.get("ok") is not True:
            raise ReleaseVerificationError("Production config validation returned an invalid result.")
        return {
            "ok": True,
            "configCreated": True,
            "workerPython": result.get("workerPython"),
            "modelRootBound": str(model_root),
            "configValidation": config_validation,
        }
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _copy_app_for_runtime_actions(
    app: Path,
    executable: Path,
) -> tuple[Path, Path, Path]:
    runner_temp = os.environ.get("RUNNER_TEMP")
    temporary_parent = Path(runner_temp) if runner_temp else Path(tempfile.gettempdir())
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="mts-macos-runtime-actions-", dir=temporary_parent)
    )
    copied_app = temporary_root / app.name
    try:
        executable_relative = executable.relative_to(app)
        build._copy_artifact(app, copied_app)
    except (ValueError, build.ReleaseBuildError, OSError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise ReleaseVerificationError(
            f"Unable to isolate the macOS app for runtime checks: {exc}"
        ) from exc
    copied_executable = copied_app / executable_relative
    if copied_executable.is_symlink() or not copied_executable.is_file():
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise ReleaseVerificationError(
            "Isolated macOS app is missing its declared executable."
        )
    return temporary_root, copied_app, copied_executable


def _process_ids_for_executable(process_table: str, executable: Path) -> set[int]:
    prefix = str(executable)
    matches: set[int] = set()
    for raw_line in process_table.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        command = fields[1]
        if command == prefix or command.startswith(prefix + " "):
            matches.add(int(fields[0]))
    return matches


def _running_process_ids(executable: Path) -> set[int]:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReleaseVerificationError(f"Unable to inspect macOS app processes: {detail}")
    return _process_ids_for_executable(completed.stdout, executable)


def _terminate_processes(process_ids: set[int]) -> None:
    for process_id in process_ids:
        try:
            os.kill(process_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    remaining = set(process_ids)
    while remaining and time.monotonic() < deadline:
        remaining = {
            process_id
            for process_id in remaining
            if _process_is_alive(process_id)
        }
        if remaining:
            time.sleep(0.1)
    for process_id in remaining:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _process_is_alive(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
        return True
    except ProcessLookupError:
        return False


def _launch_services_smoke(app: Path, executable: Path, seconds: int) -> dict[str, object]:
    before = _running_process_ids(executable)
    opened = subprocess.run(
        ["/usr/bin/open", "-n", str(app)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    if opened.returncode != 0:
        detail = opened.stderr.strip() or opened.stdout.strip()
        raise ReleaseVerificationError(f"LaunchServices rejected the macOS app: {detail}")
    launched: set[int] = set()
    startup_deadline = time.monotonic() + 30
    try:
        while time.monotonic() < startup_deadline and not launched:
            current = _running_process_ids(executable)
            new_processes = current - before
            if new_processes:
                launched.update(new_processes)
                break
            time.sleep(0.25)
        if not launched:
            raise ReleaseVerificationError(
                "LaunchServices accepted the app but no matching GUI process appeared."
            )
        alive_deadline = time.monotonic() + seconds
        while time.monotonic() < alive_deadline:
            if not launched.issubset(_running_process_ids(executable)):
                raise ReleaseVerificationError(
                    "macOS app exited during the LaunchServices smoke test."
                )
            time.sleep(0.25)
        return {
            "ok": True,
            "method": "launch-services",
            "secondsAlive": seconds,
            "terminatedByVerifier": True,
        }
    finally:
        _terminate_processes(launched)


def _launch_smoke(
    app: Path,
    executable: Path,
    seconds: int,
    *,
    via_launch_services: bool,
) -> dict[str, object]:
    if sys.platform != "darwin":
        raise ReleaseVerificationError("A macOS launch smoke can only run on macOS.")
    if via_launch_services:
        return _launch_services_smoke(app, executable, seconds)
    temporary_root = Path(tempfile.mkdtemp(prefix="mts-macos-launch-"))
    log_path = temporary_root / "launch.log"
    environment = dict(os.environ)
    environment["HOME"] = str(temporary_root / "home")
    environment["MTS_DATA_ROOT"] = str(temporary_root / "data")
    Path(environment["HOME"]).mkdir(parents=True)
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [str(executable)],
                cwd=app / "Contents" / "MacOS",
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                return_code = process.poll()
                if return_code is not None:
                    log.flush()
                    detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                    raise ReleaseVerificationError(
                        f"macOS app exited during launch smoke with code {return_code}: {detail}"
                    )
                time.sleep(0.25)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        return {
            "ok": True,
            "method": "direct-executable",
            "secondsAlive": seconds,
            "terminatedByVerifier": True,
        }
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def verify(
    release_directory: Path,
    schema_path: Path,
    *,
    expected_target: str | None,
    expected_signing_identity: str | None,
    require_native: bool,
    run_bootstrap: bool,
    launch_smoke_seconds: int,
    launch_via_open: bool,
) -> dict[str, object]:
    release_root = release_directory.expanduser().resolve()
    if release_root.is_symlink() or not release_root.is_dir():
        raise ReleaseVerificationError(f"Release directory is missing or unsafe: {release_root}")
    if require_native and sys.platform != "darwin":
        raise ReleaseVerificationError("--require-native must run on macOS.")
    manifest = _load_manifest(release_root, schema_path.expanduser().resolve())
    _verify_release_identity_and_policy(manifest)
    _verify_external_trust_anchor(
        manifest,
        expected_signing_identity=expected_signing_identity,
        require_native=require_native,
    )
    if expected_target is not None and manifest.get("build", {}).get("targetTriple") != expected_target:  # type: ignore[union-attr]
        raise ReleaseVerificationError(
            f"Manifest target does not match --expected-target: {manifest.get('build')!r}"
        )
    _verify_exact_ledger(release_root, manifest)
    artifacts = _verify_artifact_declarations(release_root, manifest)
    app_for_actions: Path | None = None
    executable_for_actions: Path | None = None
    app_checks: dict[str, object] = {}
    temporary_roots: list[Path] = []
    try:
        direct_app = artifacts.get("app")
        if direct_app is not None:
            executable, runtime = _verify_app(direct_app, manifest, require_native=require_native)
            app_for_actions = direct_app
            executable_for_actions = executable
            app_checks["app"] = runtime
        zip_artifact = artifacts.get("zip")
        if zip_artifact is not None:
            extract_parent = Path(tempfile.mkdtemp(prefix="mts-verify-zip-"))
            temporary_roots.append(extract_parent)
            extracted = _safe_extract_zip(
                zip_artifact,
                extract_parent / "payload",
                f"{manifest['productName']}.app",
            )
            executable, runtime = _verify_app(extracted, manifest, require_native=require_native)
            app_checks["zip"] = runtime
            if app_for_actions is None:
                app_for_actions = extracted
                executable_for_actions = executable
        dmg_artifact = artifacts.get("dmg")
        if dmg_artifact is not None:
            retained_app, runtime = _verify_dmg(
                dmg_artifact,
                manifest,
                require_native=require_native,
                retain_app=(
                    require_native
                    and app_for_actions is None
                    and (run_bootstrap or launch_smoke_seconds > 0)
                ),
            )
            app_checks["dmg"] = runtime or {"nativeValidation": "not-requested"}
            if retained_app is not None:
                temporary_roots.append(retained_app.parent)
                executable, _ = _verify_app(retained_app, manifest, require_native=require_native)
                app_for_actions = retained_app
                executable_for_actions = executable
        if require_native and dmg_artifact is None:
            # Native app/signature checks above are still complete; explicitly
            # record that no DMG was selected rather than treating it as absent evidence.
            app_checks["dmg"] = {"nativeValidation": "not-selected"}
        if (run_bootstrap or launch_smoke_seconds > 0) and (
            app_for_actions is None or executable_for_actions is None
        ):
            raise ReleaseVerificationError(
                "Bootstrap/launch verification requires an app or ZIP artifact in addition to DMG."
            )
        action_app: Path | None = None
        action_executable: Path | None = None
        if run_bootstrap or launch_smoke_seconds > 0:
            assert app_for_actions is not None and executable_for_actions is not None
            action_root, action_app, action_executable = _copy_app_for_runtime_actions(
                app_for_actions,
                executable_for_actions,
            )
            temporary_roots.append(action_root)
        bootstrap = _run_bootstrap(action_app) if run_bootstrap and action_app else None
        launch = (
            _launch_smoke(
                action_app,
                action_executable,
                launch_smoke_seconds,
                via_launch_services=launch_via_open,
            )
            if launch_smoke_seconds > 0 and action_app and action_executable
            else None
        )
        # Runtime checks may import Python modules or start the worker. Recheck
        # the source ledger after those actions so any accidental mutation of
        # the publishable candidate remains fail-closed.
        _verify_exact_ledger(release_root, manifest)
        return {
            "ok": True,
            "action": "VerifyTauriMacOSRelease",
            "releaseDirectory": str(release_root),
            "releaseId": manifest["releaseId"],
            "targetTriple": manifest["build"]["targetTriple"],  # type: ignore[index]
            "architecture": manifest["architecture"],
            "trust": manifest["trust"],
            "artifactBundles": sorted(artifacts),
            "nativeChecks": require_native,
            "appChecks": app_checks,
            "bootstrap": bootstrap,
            "launch": launch,
            "postActionLedgerVerified": True,
        }
    finally:
        for root in temporary_roots:
            shutil.rmtree(root, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a Tauri macOS release candidate.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release-directory", type=Path)
    source.add_argument(
        "--release-archive",
        type=Path,
        help="mode-preserving tar.gz downloaded from the release transport",
    )
    parser.add_argument(
        "--archive-checksum",
        type=Path,
        help="shasum-compatible SHA-256 companion for --release-archive",
    )
    parser.add_argument(
        "--expected-release-root",
        help="single top-level directory required inside --release-archive",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).with_name("macos-release-manifest.schema.json"),
    )
    parser.add_argument("--expected-target", choices=tuple(build.SUPPORTED_TARGETS))
    parser.add_argument(
        "--expected-signing-identity",
        help=(
            "externally trusted Developer ID Application identity; required for stable releases "
            "and never inferred from the manifest"
        ),
    )
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--launch-smoke-seconds", type=int, default=0)
    parser.add_argument(
        "--launch-via-open",
        action="store_true",
        help="launch the app through macOS LaunchServices instead of executing its Mach-O directly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    transport_root: Path | None = None
    try:
        if args.launch_smoke_seconds < 0 or args.launch_smoke_seconds > 60:
            raise ReleaseVerificationError("--launch-smoke-seconds must be between 0 and 60.")
        if args.launch_via_open and args.launch_smoke_seconds == 0:
            raise ReleaseVerificationError("--launch-via-open requires --launch-smoke-seconds.")
        if args.release_archive is not None:
            if args.archive_checksum is None or not args.expected_release_root:
                raise ReleaseVerificationError(
                    "--release-archive requires --archive-checksum and --expected-release-root."
                )
            transport_root = Path(tempfile.mkdtemp(prefix="mts-verify-transport-"))
            release_directory, transport = _extract_transport_archive(
                args.release_archive,
                args.archive_checksum,
                transport_root / "payload",
                args.expected_release_root,
            )
        else:
            if args.archive_checksum is not None or args.expected_release_root:
                raise ReleaseVerificationError(
                    "--archive-checksum and --expected-release-root require --release-archive."
                )
            release_directory = args.release_directory
            transport = None
        assert release_directory is not None
        result = verify(
            release_directory,
            args.schema,
            expected_target=args.expected_target,
            expected_signing_identity=args.expected_signing_identity,
            require_native=args.require_native,
            run_bootstrap=args.run_bootstrap,
            launch_smoke_seconds=args.launch_smoke_seconds,
            launch_via_open=args.launch_via_open,
        )
        if transport is not None:
            result["transport"] = transport
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except (ReleaseVerificationError, OSError) as exc:
        print(
            json.dumps(
                {"ok": False, "action": "VerifyTauriMacOSRelease", "error": str(exc)},
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        if transport_root is not None:
            shutil.rmtree(transport_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
