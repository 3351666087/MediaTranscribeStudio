#!/usr/bin/env python3
"""Build and collect a locked Tauri macOS release.

This is the active macOS packaging path for the Tauri desktop application.  It
intentionally does not call ``packaging/build_macos.py`` (the legacy
PyInstaller flow).  A build on macOS can inject the same small, weight-free
runtime payload used by the other native release paths.  A dry run is safe to
execute on any host and is useful for CI planning; compilation is fail-closed
to a macOS host because Apple SDK/signing tools cannot be reliably emulated.

The output is a self-contained release directory containing a manifest,
manifest checksum, and requested app/zip/dmg artifacts.  The app bundle is
treated as a first-class artifact and every file (including symlink targets) is
hashed in the manifest.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import Iterator, Mapping, Sequence
import zipfile

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows cannot compile macOS releases
    fcntl = None  # type: ignore[assignment]


CONTRACT = "mts-tauri-macos-release/v1"
SCHEMA_VERSION = "1.0.0"
SUPPORTED_TARGETS = {
    "x86_64-apple-darwin": "x64",
    "aarch64-apple-darwin": "arm64",
    "universal-apple-darwin": "universal2",
}
SUPPORTED_ARCHITECTURES = set(SUPPORTED_TARGETS.values())
SUPPORTED_BUNDLES = ("app", "dmg", "zip")
MACOS_PLATFORM = "macos"
EXPECTED_MACHO_ARCHITECTURES = {
    "x64": frozenset({"x86_64"}),
    "arm64": frozenset({"arm64"}),
    "universal2": frozenset({"x86_64", "arm64"}),
}
MACHO_CPU_TYPES = {
    0x01000007: "x86_64",
    0x0100000C: "arm64",
}
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
CHANNEL_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")

# Keep this list in lockstep with New-WindowsReleasePayload.ps1.  It is an
# explicit allowlist because copying a developer checkout would silently ship
# private media, caches, model weights, or an operator's production config.
FIXED_RUNTIME_FILES: tuple[tuple[str, str, str], ...] = (
    ("production.config.example.json", "production.config.example.json", "config-template"),
    (
        "production.config.remote.example.json",
        "production.config.remote.example.json",
        "remote-config-template",
    ),
    ("configs/model-catalog.v1.json", "configs/model-catalog.v1.json", "portable-model-catalog"),
    (
        "configs/model-catalog.v1.schema.json",
        "configs/model-catalog.v1.schema.json",
        "model-catalog-contract",
    ),
    (
        "tools/build_portable_model_catalog.py",
        "tools/build_portable_model_catalog.py",
        "portable-catalog-builder",
    ),
    ("configs/llm-provider-presets.v1.json", "configs/llm-provider-presets.v1.json", "provider-presets"),
    (
        "configs/llm-provider-presets.schema.json",
        "configs/llm-provider-presets.schema.json",
        "provider-contract",
    ),
    ("tools/model_manager.py", "tools/model_manager.py", "model-manager"),
    ("tools/model_registry.py", "tools/model_registry.py", "model-registry-validator"),
    ("tools/pyannote_runtime.py", "tools/pyannote_runtime.py", "pyannote-runtime"),
    ("requirements-media-asr.txt", "requirements-media-asr.txt", "runtime-requirements"),
    ("requirements-pyannote.txt", "requirements-pyannote.txt", "runtime-requirements"),
    (
        "pdf-renderer/target/pdf-renderer.jar",
        "pdf-renderer/target/pdf-renderer.jar",
        "pdf-runtime",
    ),
    (
        "packaging/tauri/bootstrap/Initialize-MtsRuntime.sh",
        "bootstrap/Initialize-MtsRuntime.sh",
        "bootstrap-entrypoint",
    ),
    (
        "packaging/tauri/bootstrap/Manage-MtsModels.sh",
        "bootstrap/Manage-MtsModels.sh",
        "model-manager-entrypoint",
    ),
    (
        "packaging/tauri/bootstrap/README.md",
        "bootstrap/README.md",
        "bootstrap-guide",
    ),
)
TREE_RUNTIME_DIRECTORIES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("backend", "backend", "worker-code", (".py",)),
    ("contracts", "contracts", "worker-contract", (".json", ".py")),
    ("reporting", "reporting", "report-runtime", (".py",)),
)

FORBIDDEN_MODEL_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".mlmodel",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".wav",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mov",
}
FORBIDDEN_DIRECTORY_NAMES = {
    ".cache",
    ".downloads",
    "_downloads",
    "models",
    "runtime",
    "__pycache__",
    ".git",
}
SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bsk-[a-z0-9]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{20,}"),
    re.compile(r"(?i)\bghp_[a-z0-9]{20,}\b"),
    re.compile(r"(?i)\bhf_[a-z0-9]{20,}\b"),
)
SENSITIVE_JSON_VALUE_KEYS = {
    "apikey",
    "api_key",
    "authorization",
    "accesstoken",
    "access_token",
    "bearertoken",
    "bearer_token",
}
PORTABLE_CONFIG_NAMES = {
    "production.config.example.json",
    "production.config.remote.example.json",
}


class ReleaseBuildError(RuntimeError):
    """Raised when a macOS release cannot be planned or collected safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"Unable to read JSON build input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseBuildError(f"JSON build input must contain an object: {path}")
    return value


def _read_cargo_metadata(path: Path) -> tuple[str, str]:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseBuildError(f"Unable to read Cargo build input {path}: {exc}") from exc
    package = value.get("package")
    if not isinstance(package, dict):
        raise ReleaseBuildError(f"Cargo.toml does not declare [package]: {path}")
    name = package.get("name")
    version = package.get("version")
    if not isinstance(name, str) or not name.strip():
        raise ReleaseBuildError(f"Cargo.toml does not declare [package].name: {path}")
    if not isinstance(version, str):
        raise ReleaseBuildError(f"Cargo.toml does not declare [package].version: {path}")
    return name, version


def _assert_cargo_lock_contains_package(path: Path, package_name: str, version: str) -> None:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseBuildError(f"Unable to read Cargo.lock: {path}: {exc}") from exc
    packages = value.get("package")
    if not isinstance(packages, list) or not any(
        isinstance(package, dict)
        and package.get("name") == package_name
        and package.get("version") == version
        for package in packages
    ):
        raise ReleaseBuildError(
            f"Cargo.lock must contain {package_name} at the application version."
        )


def _locked_input(path: Path, project_root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    value = completed.stdout.strip().lower()
    return value if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value) else "unknown"


def _default_target_triple() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64-apple-darwin"
    if machine in {"arm64", "aarch64"}:
        return "aarch64-apple-darwin"
    raise ReleaseBuildError(
        f"Unsupported macOS host architecture {machine!r}; pass --target-triple explicitly."
    )


def _parse_bundles(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one bundle must be selected")
    unknown = sorted(set(values).difference(SUPPORTED_BUNDLES))
    if unknown:
        raise argparse.ArgumentTypeError("unsupported bundle(s): " + ", ".join(unknown))
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("bundle selections must be unique")
    return values


def _safe_relative(path: str) -> str:
    value = str(path or "").replace("\\", "/")
    if (
        not value
        or value.startswith("/")
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ReleaseBuildError(f"Unsafe relative runtime path: {path!r}")
    return value


def _assert_no_secret_bytes(path: Path) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to read runtime payload input {path}: {exc}") from exc
    _assert_no_secret_data(data, path)


def _assert_no_secret_data(data: bytes, path: Path) -> None:
    # Decode lossily so a future text fixture cannot accidentally carry a
    # credential into a release. The JAR is also scanned for obvious tokens.
    text = data.decode("utf-8", errors="ignore")
    for pattern in SECRET_VALUE_PATTERNS:
        if pattern.search(text):
            raise ReleaseBuildError(f"Possible credential detected in runtime payload input: {path}")
    if path.suffix.casefold() == ".json":
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            return

        def inspect(value: object, location: str) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    key_text = str(key)
                    normalized = key_text.casefold().replace("-", "_")
                    if normalized in SENSITIVE_JSON_VALUE_KEYS and isinstance(item, str):
                        candidate = item.strip()
                        if candidate:
                            raise ReleaseBuildError(
                                f"Raw credential-like JSON value detected at {path}:{location}/{key_text}"
                            )
                    inspect(item, f"{location}/{key_text}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    inspect(item, f"{location}/{index}")

        inspect(document, "")


def _portable_config_bytes(path: Path) -> bytes:
    """Strip developer-machine paths from shipped configuration templates."""

    if path.name not in PORTABLE_CONFIG_NAMES:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ReleaseBuildError(f"Unable to read runtime payload input {path}: {exc}") from exc
        _assert_no_secret_data(data, path)
        return data
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"Unable to read runtime config template {path}: {exc}") from exc

    def rewrite(item: object) -> object:
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        if isinstance(item, dict):
            return {key: rewrite(child) for key, child in item.items()}
        if not isinstance(item, str):
            return item
        normalized = item.replace("\\", "/")
        folded = normalized.casefold()
        if folded.startswith("d:/models/"):
            return "models/" + normalized[len("D:/models/"):]
        if folded.startswith("d:/downloads"):
            return "inputs"
        if folded.startswith("d:/desktop/"):
            return "cache" if "cache" in folded else "exports"
        if folded.startswith("d:/mediatranscribestudio/outputs"):
            return "exports"
        if folded.startswith("d:/mediatranscribestudio/cache"):
            return "cache"
        if folded.startswith("d:/mediatranscribestudio/runtime/"):
            marker = re.search(r"/runtime/", normalized, flags=re.IGNORECASE)
            assert marker is not None
            return "runtime/" + normalized[marker.end():]
        return item

    data = (json.dumps(rewrite(value), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if re.search(r"(?i)(?<![A-Za-z0-9])[a-z]:[\\/]", data.decode("utf-8")):
        raise ReleaseBuildError(f"Runtime config template still contains a drive-qualified path: {path}")
    _assert_no_secret_data(data, path)
    return data


def _iter_tree_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not root.is_dir():
        raise ReleaseBuildError(f"Required runtime directory is missing: {root}")
    files: list[Path] = []
    root_resolved = root.resolve()
    # Prune forbidden directories during traversal.  Filtering a completed
    # rglob list is insufficient because pathlib may already have descended
    # into a model/cache subtree by the time it is filtered.
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            if name in FORBIDDEN_DIRECTORY_NAMES:
                continue
            if candidate.is_symlink():
                raise ReleaseBuildError(f"Symlinks are not allowed in runtime source trees: {candidate}")
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            candidate = current_path / name
            if candidate.is_symlink():
                raise ReleaseBuildError(f"Symlinks are not allowed in runtime source trees: {candidate}")
            try:
                candidate.resolve().relative_to(root_resolved)
            except ValueError as exc:
                raise ReleaseBuildError(f"Runtime source escapes its root: {candidate}") from exc
            if candidate.suffix.lower() in FORBIDDEN_MODEL_EXTENSIONS:
                raise ReleaseBuildError(f"Model/media artifact is forbidden in runtime payload: {candidate}")
            if candidate.suffix.lower() in extensions and "__pycache__" not in candidate.parts:
                files.append(candidate)
    if not files:
        raise ReleaseBuildError(f"Runtime directory has no accepted files: {root}")
    return files


def _runtime_plan(project_root: Path, *, allow_missing: bool) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    destinations: set[str] = set()

    def add(source: Path, destination: str, role: str) -> None:
        destination_value = _safe_relative(destination)
        key = destination_value.casefold()
        if key in destinations:
            raise ReleaseBuildError(f"Runtime payload has a duplicate destination: {destination_value}")
        destinations.add(key)
        if source.is_symlink():
            raise ReleaseBuildError(f"Symlinks are not allowed in fixed runtime inputs: {source}")
        present = source.is_file()
        if not present and not allow_missing:
            raise ReleaseBuildError(f"Required macOS runtime input is missing: {source}")
        size = 0
        if present:
            if source.name in PORTABLE_CONFIG_NAMES:
                size = len(_portable_config_bytes(source))
            else:
                _assert_no_secret_bytes(source)
                size = source.stat().st_size
        entries.append(
            {
                "source": source,
                "destination": destination_value,
                "role": role,
                "size": size,
                "sourcePresent": present,
            }
        )

    for source_rel, destination, role, extensions in TREE_RUNTIME_DIRECTORIES:
        source_root = project_root / source_rel
        if not source_root.is_dir():
            if allow_missing:
                # Keep a deterministic dry-run record for a minimal fixture.
                add(source_root, destination + "/.missing", role)
                continue
            raise ReleaseBuildError(f"Required runtime directory is missing: {source_root}")
        for source in _iter_tree_files(source_root, extensions):
            relative = source.relative_to(source_root).as_posix()
            add(source, f"{destination}/{relative}", role)

    for source_rel, destination, role in FIXED_RUNTIME_FILES:
        add(project_root / source_rel, destination, role)

    entries.sort(key=lambda item: str(item["destination"]))
    return entries


def _runtime_profile() -> dict[str, object]:
    return {
        "schemaVersion": "1.0.0",
        "artifactType": "mts-macos-runtime-bootstrap",
        "runtimeStrategy": "external-or-operator-provided",
        "bundledPythonRuntime": False,
        "bundledModelArtifacts": False,
        "defaultModelRootPolicy": [
            "MTS_MODEL_ROOT",
            "MTS_DATA_ROOT/models",
            "~/Library/Application Support/MediaTranscribeStudio/models",
        ],
        "productionConfigTemplate": "production.config.example.json",
        "providerPresetCatalog": "configs/llm-provider-presets.v1.json",
        "modelRegistry": "configs/model-catalog.v1.json",
        "modelManager": "bootstrap/Manage-MtsModels.sh",
        "runtimeInitializer": "bootstrap/Initialize-MtsRuntime.sh",
    }


def _runtime_profile_bytes() -> bytes:
    return (json.dumps(_runtime_profile(), ensure_ascii=True, indent=2) + "\n").encode("utf-8")


def _runtime_payload_bytes(source: Path) -> bytes:
    if source.name in PORTABLE_CONFIG_NAMES:
        return _portable_config_bytes(source)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to read runtime payload input {source}: {exc}") from exc
    _assert_no_secret_data(payload, source)
    return payload


def _expected_runtime_ledger(project_root: Path) -> dict[str, tuple[int, str]]:
    ledger: dict[str, tuple[int, str]] = {}
    for entry in _runtime_plan(project_root, allow_missing=False):
        source = entry["source"]
        assert isinstance(source, Path)
        payload = _runtime_payload_bytes(source)
        ledger[str(entry["destination"])] = (len(payload), _sha256_bytes(payload))
    profile = _runtime_profile_bytes()
    ledger["bootstrap/runtime-bootstrap.v1.json"] = (len(profile), _sha256_bytes(profile))
    return ledger


def _stage_runtime_payload(project_root: Path, staging_root: Path) -> list[dict[str, object]]:
    plan = _runtime_plan(project_root, allow_missing=False)
    if staging_root.exists():
        raise ReleaseBuildError(f"Runtime staging directory already exists: {staging_root}")
    staging_root.mkdir(parents=True, exist_ok=False)
    for entry in plan:
        source = entry["source"]
        assert isinstance(source, Path)
        destination = staging_root / str(entry["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.name in PORTABLE_CONFIG_NAMES:
            payload = _runtime_payload_bytes(source)
            destination.write_bytes(payload)
            verified = destination.read_bytes() == payload
        else:
            shutil.copy2(source, destination)
            verified = destination.stat().st_size == source.stat().st_size and _sha256(destination) == _sha256(source)
        if not verified:
            raise ReleaseBuildError(f"Runtime payload copy verification failed: {source}")
    profile_path = staging_root / "bootstrap" / "runtime-bootstrap.v1.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_bytes = _runtime_profile_bytes()
    profile_path.write_bytes(profile_bytes)
    plan.append(
        {
            "source": None,
            "destination": "bootstrap/runtime-bootstrap.v1.json",
            "role": "bootstrap-profile",
            "size": len(profile_bytes),
            "sourcePresent": True,
        }
    )
    return plan


def _runtime_plan_json(project_root: Path) -> dict[str, object]:
    plan = _runtime_plan(project_root, allow_missing=True)
    profile_bytes = _runtime_profile_bytes()
    files = [
        {
            "destination": str(entry["destination"]),
            "role": str(entry["role"]),
            "size": int(entry["size"]),
            "sourcePresent": bool(entry["sourcePresent"]),
        }
        for entry in plan
    ]
    files.append(
        {
            "destination": "bootstrap/runtime-bootstrap.v1.json",
            "role": "bootstrap-profile",
            "size": len(profile_bytes),
            "sourcePresent": True,
        }
    )
    return {
        "artifactType": "mts-macos-runtime-bootstrap",
        "bundledPythonRuntime": False,
        "bundledModelArtifacts": False,
        "fileCount": len(files),
        "totalBytes": sum(int(item["size"]) for item in files),
        "files": files,
    }


def _load_build_context(
    project_root: Path,
    target_triple: str,
    bundles: Sequence[str],
    output_directory: Path | None,
    target_directory: Path | None,
    channel: str,
    source_date_epoch: int,
) -> dict[str, object]:
    project = project_root.expanduser().resolve()
    desktop = project / "apps" / "desktop"
    tauri = desktop / "src-tauri"
    paths = {
        "packageJson": desktop / "package.json",
        "tauriConfig": tauri / "tauri.conf.json",
        "cargoToml": tauri / "Cargo.toml",
        "packageLock": desktop / "package-lock.json",
        "cargoLock": tauri / "Cargo.lock",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise ReleaseBuildError(f"Required locked build input is missing ({label}): {path}")
    package = _read_json(paths["packageJson"])
    tauri_config = _read_json(paths["tauriConfig"])
    package_lock = _read_json(paths["packageLock"])
    cargo_name, cargo_version = _read_cargo_metadata(paths["cargoToml"])
    versions = {
        "package.json": package.get("version"),
        "tauri.conf.json": tauri_config.get("version"),
        "Cargo.toml": cargo_version,
    }
    if any(not isinstance(value, str) for value in versions.values()):
        raise ReleaseBuildError("Every application build input must declare a string version.")
    if len(set(versions.values())) != 1:
        details = ", ".join(f"{name}={value!r}" for name, value in versions.items())
        raise ReleaseBuildError(f"package.json, tauri.conf.json, and Cargo.toml versions must match ({details}).")
    version = next(iter(versions.values()))
    assert isinstance(version, str)
    if SEMVER_PATTERN.fullmatch(version) is None:
        raise ReleaseBuildError(f"Application version must be SemVer 2.0.0: {version!r}")
    package_name = package.get("name")
    lock_packages = package_lock.get("packages")
    root_lock = lock_packages.get("") if isinstance(lock_packages, dict) else None
    if (
        not isinstance(root_lock, dict)
        or root_lock.get("version") != version
        or (isinstance(package_name, str) and root_lock.get("name") != package_name)
    ):
        raise ReleaseBuildError("package-lock.json root package name/version must match package.json.")
    _assert_cargo_lock_contains_package(paths["cargoLock"], cargo_name, version)
    app_id = tauri_config.get("identifier")
    product_name = tauri_config.get("productName")
    if not isinstance(app_id, str) or not app_id.strip():
        raise ReleaseBuildError("tauri.conf.json must declare a non-empty identifier.")
    if not isinstance(product_name, str) or not product_name.strip():
        raise ReleaseBuildError("tauri.conf.json must declare a non-empty productName.")
    if target_triple not in SUPPORTED_TARGETS:
        raise ReleaseBuildError(f"Unsupported macOS target triple: {target_triple}")
    if not CHANNEL_PATTERN.fullmatch(channel):
        raise ReleaseBuildError(f"Invalid release channel: {channel!r}")
    if source_date_epoch < 0:
        raise ReleaseBuildError("--source-date-epoch must be non-negative.")
    architecture = SUPPORTED_TARGETS[target_triple]
    target_root = (
        target_directory.expanduser().resolve() if target_directory is not None else (tauri / "target").resolve()
    )
    output = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else (project / "dist" / "tauri-release" / f"{version}-macos-{architecture}").resolve()
    )
    artifact_root = target_root / target_triple / "release" / "bundle"
    tauri_bundles: list[str] = []
    if "app" in bundles or "zip" in bundles:
        tauri_bundles.append("app")
    if "dmg" in bundles:
        tauri_bundles.append("dmg")
    commands = [
        "npm ci",
        (
            "npm run tauri -- build --ci "
            f"--target {target_triple} --bundles {','.join(tauri_bundles)} -- --locked"
        ),
    ]
    locked_inputs = [
        _locked_input(paths["packageLock"], project),
        _locked_input(paths["cargoLock"], project),
    ]
    return {
        "projectRoot": project,
        "desktopRoot": desktop,
        "tauriRoot": tauri,
        "version": version,
        "appId": app_id,
        "productName": product_name,
        "targetTriple": target_triple,
        "architecture": architecture,
        "platform": MACOS_PLATFORM,
        "bundles": list(bundles),
        "tauriBundles": tauri_bundles,
        "channel": channel,
        "sourceDateEpoch": source_date_epoch,
        "targetDirectory": target_root,
        "outputDirectory": output,
        "artifactSearchRoot": artifact_root,
        "buildLockFile": target_root / ".mts-tauri-macos-build.lock",
        "lockedInputs": locked_inputs,
        "commands": commands,
        "runtimePlan": _runtime_plan_json(project),
    }


def _json_plan(
    context: Mapping[str, object],
    *,
    skip_compile: bool,
    allow_unsigned_development: bool,
    require_notarization: bool,
) -> dict[str, object]:
    target = str(context["targetTriple"])
    return {
        "ok": True,
        "action": "BuildTauriMacOSRelease",
        "dryRun": True,
        "projectRoot": str(context["projectRoot"]),
        "version": context["version"],
        "platform": MACOS_PLATFORM,
        "architecture": context["architecture"],
        "targetTriple": target,
        "channel": context["channel"],
        "bundles": context["bundles"],
        "tauriBundles": context["tauriBundles"],
        "outputDirectory": str(context["outputDirectory"]),
        "targetDirectory": str(context["targetDirectory"]),
        "buildLockFile": str(context["buildLockFile"]),
        "lockedInputs": context["lockedInputs"],
        "commands": [] if skip_compile else context["commands"],
        "runtimePayload": context["runtimePlan"],
        "runtimeInjection": {
            "destination": "Contents/Resources/mts-runtime",
            "strategy": "tauri-bundle-resources-overlay",
            "bundledPythonRuntime": False,
            "bundledModelArtifacts": False,
        },
        "trust": {
            "requireCodeSigning": not allow_unsigned_development,
            "requireNotarization": require_notarization,
        },
        "postBuild": [
            "verify-locked-input-hashes-unchanged",
            "collect-requested-macos-app-and-dmg-bundles",
            "create-deterministic-app-zip-when-requested",
            "verify-code-signing-or-explicit-development-mode",
            "emit-byte-hashed-macos-release-manifest",
            "emit-release-manifest-sha256",
        ],
    }


def _write_plan_artifact(plan: Mapping[str, object], destination: Path) -> tuple[Path, str]:
    output = destination.expanduser().resolve()
    checksum = Path(str(output) + ".sha256")
    if output.exists() or checksum.exists():
        raise ReleaseBuildError(f"Plan output already exists; choose a new path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(plan, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    stage = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        stage.write_bytes(payload)
        stage.replace(output)
        digest = _sha256(output)
        checksum.write_text(digest + "\n", encoding="ascii", newline="\n")
        return output, digest
    except BaseException:
        stage.unlink(missing_ok=True)
        checksum.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_build_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise ReleaseBuildError("The macOS build lock requires fcntl; compile on macOS.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseBuildError(f"Another macOS release build holds the lock: {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]) -> None:
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    try:
        completed = subprocess.run(
            list(command), cwd=cwd, env=dict(environment), stdout=sys.stderr, stderr=sys.stderr, check=False
        )
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to execute {command[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        raise ReleaseBuildError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _assert_locked_inputs_unchanged(context: Mapping[str, object]) -> None:
    project = context["projectRoot"]
    assert isinstance(project, Path)
    for expected in context["lockedInputs"]:  # type: ignore[index]
        assert isinstance(expected, dict)
        path = project / str(expected["path"])
        if not path.is_file() or path.stat().st_size != expected["size"] or _sha256(path) != expected["sha256"]:
            raise ReleaseBuildError(f"Locked build input changed or disappeared: {path}")


def _write_macos_overlay(path: Path, runtime_stage: Path, icon_path: Path | None) -> None:
    # Tauri preserves the mapped directory under $RESOURCES.  The explicit
    # destination avoids relying on the source checkout's absolute layout.
    bundle: dict[str, object] = {
        "resources": {str(runtime_stage): "mts-runtime"},
    }
    if icon_path is not None:
        bundle["icon"] = [str(icon_path)]
    config = {"bundle": bundle}
    path.write_text(json.dumps(config, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def _build(
    context: Mapping[str, object],
    *,
    signing_identity: str | None,
    allow_unsigned_development: bool,
) -> None:
    if sys.platform != "darwin":
        raise ReleaseBuildError("macOS release compilation must run on a macOS host.")
    project = context["projectRoot"]
    desktop = context["desktopRoot"]
    tauri = context["tauriRoot"]
    target = str(context["targetTriple"])
    assert isinstance(project, Path) and isinstance(desktop, Path) and isinstance(tauri, Path)
    temporary_root = Path(tempfile.mkdtemp(prefix="mts-macos-build-"))
    runtime_stage = temporary_root / "runtime"
    overlay = temporary_root / "tauri.macos.overlay.json"
    try:
        _stage_runtime_payload(project, runtime_stage)
        icon_path = tauri / "icons" / "icon.png"
        if not icon_path.is_file():
            # The tracked ICO remains a valid Tauri fallback.  A PNG is used
            # when available because it produces a better macOS icon, but a
            # missing generated convenience asset must not block a clean
            # checkout from reaching the native builder.
            icon_path = tauri / "icons" / "icon.ico"
        _write_macos_overlay(overlay, runtime_stage, icon_path if icon_path.is_file() else None)
        environment = dict(os.environ)
        environment["CARGO_TARGET_DIR"] = str(context["targetDirectory"])
        environment["SOURCE_DATE_EPOCH"] = str(context["sourceDateEpoch"])
        if signing_identity is not None:
            # Tauri's bundler consumes APPLE_SIGNING_IDENTITY.  Keep the
            # identity out of the JSON manifest except for the public label.
            environment["APPLE_SIGNING_IDENTITY"] = signing_identity
        elif allow_unsigned_development:
            environment.setdefault("CI", "1")
        _run(["npm", "ci"], cwd=desktop, environment=environment)
        command = [
            "npm",
            "run",
            "tauri",
            "--",
            "build",
            "--ci",
            "--target",
            target,
            "--bundles",
            ",".join(context["tauriBundles"]),  # type: ignore[arg-type]
            "--config",
            str(overlay),
            "--",
            "--locked",
        ]
        if signing_identity is None and allow_unsigned_development:
            # --no-sign is a Tauri CLI option and must precede the `--` that
            # forwards Cargo arguments.
            command.insert(command.index("--", 4), "--no-sign")
        _run(
            command,
            cwd=desktop,
            environment=environment,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _app_candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*.app")
        if path.is_dir() and not path.is_symlink()
    )


def _file_candidates(root: Path, suffix: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.casefold() == suffix.casefold()
    )


def _read_macho_architectures(path: Path) -> frozenset[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(8)
            if len(header) < 8:
                raise ReleaseBuildError(f"macOS executable has a truncated Mach-O header: {path}")

            thin_formats = {
                b"\xce\xfa\xed\xfe": "<",  # 32-bit little endian
                b"\xcf\xfa\xed\xfe": "<",  # 64-bit little endian
                b"\xfe\xed\xfa\xce": ">",  # 32-bit big endian
                b"\xfe\xed\xfa\xcf": ">",  # 64-bit big endian
            }
            if header[:4] in thin_formats:
                cpu_type = struct.unpack(f"{thin_formats[header[:4]]}I", header[4:8])[0]
                architecture = MACHO_CPU_TYPES.get(cpu_type)
                if architecture is None:
                    raise ReleaseBuildError(
                        f"macOS executable uses unsupported Mach-O CPU type 0x{cpu_type:08x}: {path}"
                    )
                return frozenset({architecture})

            fat_formats = {
                b"\xca\xfe\xba\xbe": (">", 20),
                b"\xbe\xba\xfe\xca": ("<", 20),
                b"\xca\xfe\xba\xbf": (">", 32),
                b"\xbf\xba\xfe\xca": ("<", 32),
            }
            fat = fat_formats.get(header[:4])
            if fat is None:
                raise ReleaseBuildError(f"macOS app executable is not a Mach-O binary: {path}")
            byte_order, record_size = fat
            architecture_count = struct.unpack(f"{byte_order}I", header[4:8])[0]
            if architecture_count == 0 or architecture_count > 32:
                raise ReleaseBuildError(
                    f"macOS universal binary declares an invalid architecture count {architecture_count}: {path}"
                )
            table = stream.read(record_size * architecture_count)
            if len(table) != record_size * architecture_count:
                raise ReleaseBuildError(f"macOS universal binary has a truncated architecture table: {path}")
            architectures: set[str] = set()
            for index in range(architecture_count):
                record = table[index * record_size : (index + 1) * record_size]
                cpu_type = struct.unpack(f"{byte_order}I", record[:4])[0]
                architecture = MACHO_CPU_TYPES.get(cpu_type)
                if architecture is None:
                    raise ReleaseBuildError(
                        f"macOS universal binary uses unsupported CPU type 0x{cpu_type:08x}: {path}"
                    )
                if record_size == 20:
                    offset, slice_size = struct.unpack(f"{byte_order}II", record[8:16])
                else:
                    offset, slice_size = struct.unpack(f"{byte_order}QQ", record[8:24])
                if slice_size == 0 or offset > size or slice_size > size - offset:
                    raise ReleaseBuildError(
                        f"macOS universal binary has an out-of-range {architecture} slice: {path}"
                    )
                if architecture in architectures:
                    raise ReleaseBuildError(
                        f"macOS universal binary declares duplicate {architecture} slices: {path}"
                    )
                architectures.add(architecture)
            return frozenset(architectures)
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to inspect macOS executable {path}: {exc}") from exc


def _validate_app_bundle(app: Path, context: Mapping[str, object]) -> Path:
    if not app.is_dir() or app.suffix.casefold() != ".app":
        raise ReleaseBuildError(f"macOS app bundle is not a directory: {app}")
    info = app / "Contents" / "Info.plist"
    macos_dir = app / "Contents" / "MacOS"
    if not info.is_file():
        raise ReleaseBuildError(f"macOS app bundle is missing Contents/Info.plist: {app}")
    try:
        with info.open("rb") as stream:
            metadata = plistlib.load(stream)
    except (OSError, plistlib.InvalidFileException) as exc:
        raise ReleaseBuildError(f"macOS app has an invalid Contents/Info.plist: {app}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ReleaseBuildError(f"macOS app Info.plist must contain a dictionary: {app}")
    expected_metadata = {
        "CFBundleIdentifier": str(context["appId"]),
        "CFBundleShortVersionString": str(context["version"]),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ReleaseBuildError(
                f"macOS app {key} does not match the locked build context "
                f"({metadata.get(key)!r} != {expected!r}): {app}"
            )
    executable_name = metadata.get("CFBundleExecutable")
    if (
        not isinstance(executable_name, str)
        or not executable_name
        or executable_name in {".", ".."}
        or "/" in executable_name
        or "\\" in executable_name
    ):
        raise ReleaseBuildError(f"macOS app has an unsafe CFBundleExecutable value: {app}")
    executable = macos_dir / executable_name
    if executable.is_symlink() or not executable.is_file():
        raise ReleaseBuildError(f"macOS app is missing its declared executable: {executable}")
    if not os.access(executable, os.X_OK):
        raise ReleaseBuildError(f"macOS app executable is not executable: {executable}")
    actual_architectures = _read_macho_architectures(executable)
    expected_architectures = EXPECTED_MACHO_ARCHITECTURES[str(context["architecture"])]
    if actual_architectures != expected_architectures:
        raise ReleaseBuildError(
            "macOS app Mach-O architectures do not match the requested target "
            f"({sorted(actual_architectures)} != {sorted(expected_architectures)}): {executable}"
        )
    return executable


def _discover_artifacts(context: Mapping[str, object]) -> dict[str, list[Path]]:
    root = context["artifactSearchRoot"]
    assert isinstance(root, Path)
    if not root.is_dir():
        raise ReleaseBuildError(f"Tauri macOS bundle directory is missing: {root}")
    found: dict[str, list[Path]] = {bundle: [] for bundle in context["bundles"]}  # type: ignore[union-attr]
    macos_root = root / "macos"
    dmg_root = root / "dmg"
    if any(bundle in found for bundle in ("app", "zip", "dmg")):
        apps = _app_candidates(macos_root)
        if not apps:
            # Some Tauri versions place the app directly below bundle.
            apps = _app_candidates(root)
        expected_name = f"{context['productName']}.app"
        named_apps = [path for path in apps if path.name == expected_name]
        if named_apps:
            apps = named_apps
        if not apps:
            raise ReleaseBuildError(f"Requested app bundle was not produced below {macos_root}.")
        if len(apps) != 1:
            raise ReleaseBuildError(
                f"Expected exactly one macOS app bundle for {expected_name}, found {len(apps)}: "
                + ", ".join(str(path) for path in apps)
            )
        _validate_app_bundle(apps[0], context)
        found["app"] = apps
    if "dmg" in found:
        dmgs = _file_candidates(dmg_root, ".dmg") or _file_candidates(root, ".dmg")
        version = str(context["version"])
        named_dmgs = [path for path in dmgs if version in path.name]
        if named_dmgs:
            dmgs = named_dmgs
        if not dmgs:
            raise ReleaseBuildError(f"Requested DMG bundle was not produced below {dmg_root}.")
        if len(dmgs) != 1:
            raise ReleaseBuildError(
                f"Expected exactly one versioned macOS DMG, found {len(dmgs)}: "
                + ", ".join(str(path) for path in dmgs)
            )
        found["dmg"] = dmgs
    if "zip" in found:
        # The zip is generated after discovery; callers use the app source.
        found["zip"] = []
    return found


def _zip_timestamp(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    value = time.gmtime(source_date_epoch or 315532800)  # 1980-01-01 for default epoch
    year = max(1980, min(2107, value.tm_year))
    return year, max(1, value.tm_mon), max(1, value.tm_mday), value.tm_hour, value.tm_min, value.tm_sec


def _write_deterministic_app_zip(app: Path, destination: Path, source_date_epoch: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _zip_timestamp(source_date_epoch)
    root_name = app.name
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        paths = [app, *sorted(app.rglob("*"))]
        for path in paths:
            relative = Path(root_name) if path == app else Path(root_name) / path.relative_to(app)
            name = relative.as_posix()
            if path.is_symlink():
                info = zipfile.ZipInfo(name, timestamp)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, os.readlink(path).encode("utf-8"))
            elif path.is_dir():
                info = zipfile.ZipInfo(name.rstrip("/") + "/", timestamp)
                info.create_system = 3
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
                archive.writestr(info, b"")
            elif path.is_file():
                info = zipfile.ZipInfo(name, timestamp)
                info.create_system = 3
                mode = stat.S_IMODE(path.stat().st_mode) or 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                with path.open("rb") as stream:
                    archive.writestr(info, stream.read())


def _hash_path(path: Path, relative_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if path.is_symlink():
        target = os.readlink(path)
        if os.path.isabs(target):
            raise ReleaseBuildError(f"Absolute symlink targets are forbidden in macOS artifacts: {path}")
        try:
            (path.parent / target).resolve(strict=False).relative_to(relative_root.resolve())
        except ValueError as exc:
            raise ReleaseBuildError(f"Symlink target escapes the macOS artifact: {path} -> {target}") from exc
        target_bytes = target.encode("utf-8")
        records.append(
            {
                "path": path.relative_to(relative_root).as_posix(),
                "kind": "symlink",
                "target": target,
                "size": len(target_bytes),
                "sha256": _sha256_bytes(target_bytes),
            }
        )
        return records
    if path.is_file():
        records.append(
            {
                "path": path.relative_to(relative_root).as_posix(),
                "kind": "file",
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        return records
    if not path.is_dir():
        raise ReleaseBuildError(f"Unsupported artifact filesystem entry: {path}")
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        records.extend(_hash_path(child, relative_root))
    return records


def _copy_artifact(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise ReleaseBuildError(f"Artifact is not a regular file or directory: {source}")


def _app_runtime_marker(app: Path) -> Path:
    return app / "Contents" / "Resources" / "mts-runtime" / "backend" / "worker.py"


def _validate_embedded_runtime(app: Path, project_root: Path) -> None:
    runtime = app / "Contents" / "Resources" / "mts-runtime"
    marker = runtime / "backend" / "worker.py"
    profile = runtime / "bootstrap" / "runtime-bootstrap.v1.json"
    if not marker.is_file() or not profile.is_file():
        raise ReleaseBuildError(
            "macOS app bundle is missing the embedded runtime payload under "
            "Contents/Resources/mts-runtime; build with this release tool's overlay."
        )
    expected = _expected_runtime_ledger(project_root)
    actual: dict[str, tuple[int, str]] = {}
    for current, directory_names, file_names in os.walk(runtime, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ReleaseBuildError(f"Embedded macOS runtime contains a directory symlink: {candidate}")
        for name in file_names:
            candidate = current_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ReleaseBuildError(f"Embedded macOS runtime contains a non-regular file: {candidate}")
            if candidate.suffix.casefold() in FORBIDDEN_MODEL_EXTENSIONS:
                raise ReleaseBuildError(
                    f"Embedded macOS runtime contains a forbidden model/media artifact: {candidate}"
                )
            _assert_no_secret_bytes(candidate)
            relative = candidate.relative_to(runtime).as_posix()
            actual[relative] = (candidate.stat().st_size, _sha256(candidate))
    if actual != expected:
        missing = sorted(set(expected).difference(actual))
        unexpected = sorted(set(actual).difference(expected))
        changed = sorted(
            path for path in set(actual).intersection(expected) if actual[path] != expected[path]
        )
        raise ReleaseBuildError(
            "Embedded macOS runtime does not match the locked allowlist "
            f"(missing={missing}, unexpected={unexpected}, changed={changed})."
        )
    if os.name != "nt":
        for executable in (
            "bootstrap/Initialize-MtsRuntime.sh",
            "bootstrap/Manage-MtsModels.sh",
        ):
            path = runtime / executable
            if not path.is_file() or not path.stat().st_mode & 0o111:
                raise ReleaseBuildError(f"Embedded macOS bootstrap is not executable: {executable}")


def _prepare_app_for_release(
    app_source: Path,
    stage: Path,
    context: Mapping[str, object],
    *,
    signing_identity: str | None,
    allow_unsigned_development: bool,
) -> Path:
    """Return an app with the runtime payload embedded without mutating source.

    Tauri builds use the resource overlay in ``_build``.  ``--skip-compile`` is
    also useful for collecting a prebuilt app from CI; if that app predates the
    overlay, an unsigned staging copy is repaired with the same allowlist.  A
    signed app is never modified after signing.
    """

    marker = _app_runtime_marker(app_source)
    if marker.is_file():
        _validate_embedded_runtime(app_source, context["projectRoot"])  # type: ignore[arg-type]
        return app_source
    if signing_identity is not None:
        raise ReleaseBuildError(
            "Prebuilt macOS app has no embedded runtime payload; signed collection refuses post-sign mutation."
        )
    if not allow_unsigned_development:
        raise ReleaseBuildError(
            "Prebuilt macOS app has no embedded runtime payload; pass --allow-unsigned-development "
            "only for a local repair fixture."
        )
    prepared_root = stage / ".prepared-app"
    prepared_app = prepared_root / app_source.name
    _copy_artifact(app_source, prepared_app)
    runtime_root = prepared_app / "Contents" / "Resources" / "mts-runtime"
    _stage_runtime_payload(context["projectRoot"], runtime_root)  # type: ignore[arg-type]
    _validate_embedded_runtime(prepared_app, context["projectRoot"])  # type: ignore[arg-type]
    return prepared_app


def _resolve_signing_identity(explicit: str | None) -> str | None:
    value = str(
        explicit
        or os.environ.get("APPLE_SIGNING_IDENTITY", "")
        or os.environ.get("MAC_CODESIGN_IDENTITY", "")
        or os.environ.get("CODESIGN_IDENTITY", "")
    ).strip()
    return value or None


def _verify_codesign(artifact: Path, identity: str) -> None:
    if sys.platform != "darwin":
        raise ReleaseBuildError(
            "Cannot verify a macOS code signature off-host; run collection on macOS."
        )
    tool = shutil.which("codesign") or "/usr/bin/codesign"
    if not Path(tool).is_file():
        raise ReleaseBuildError("codesign is required when a signing identity is configured.")
    verify_command = [tool, "--verify", "--strict", "--verbose=2"]
    if artifact.is_dir():
        verify_command.append("--deep")
    verify_command.append(str(artifact))
    completed = subprocess.run(
        verify_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown codesign failure"
        raise ReleaseBuildError(f"macOS signature verification failed for {artifact}: {detail}")

    # Ask codesign to resolve the identity as an additional diagnostic.  The
    # exact certificate chain is validated by Gatekeeper/notarization; this
    # check only prevents an accidentally unsigned app from being published.
    details = subprocess.run(
        [tool, "-dv", "--verbose=2", str(artifact)],
        capture_output=True,
        text=True,
        check=False,
    )
    if details.returncode != 0:
        raise ReleaseBuildError(f"Unable to inspect the macOS signature for {artifact}.")
    authority_lines = [
        line
        for line in (details.stderr + "\n" + details.stdout).splitlines()
        if line.startswith("Authority=")
    ]
    if not authority_lines:
        raise ReleaseBuildError(f"macOS artifact has no certificate authority: {artifact}")
    expected = identity.casefold()
    if expected not in {line.split("=", 1)[1].strip().casefold() for line in authority_lines}:
        raise ReleaseBuildError(
            f"macOS artifact signature identity does not match the requested identity {identity!r}: "
            + "; ".join(authority_lines)
        )


def _run_native_check(command: Sequence[str], *, timeout: int = 180) -> str:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseBuildError(f"Unable to execute {' '.join(command)}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise ReleaseBuildError(
            f"Native macOS validation failed with exit code {completed.returncode}: "
            f"{' '.join(command)} ({detail})"
        )
    return completed.stdout


def _verify_stapled_notarization(path: Path) -> None:
    if sys.platform != "darwin":
        raise ReleaseBuildError("Apple notarization/stapling can only be verified on macOS.")
    xcrun = shutil.which("xcrun") or "/usr/bin/xcrun"
    if not Path(xcrun).is_file():
        raise ReleaseBuildError("xcrun is required to verify Apple notarization.")
    _run_native_check([xcrun, "stapler", "validate", str(path)])


def _verify_gatekeeper(path: Path, *, artifact_type: str) -> None:
    if sys.platform != "darwin":
        raise ReleaseBuildError("Apple Gatekeeper assessment can only run on macOS.")
    spctl = shutil.which("spctl") or "/usr/sbin/spctl"
    if not Path(spctl).is_file():
        raise ReleaseBuildError("spctl is required for a notarized release gate.")
    if artifact_type == "app":
        command = [spctl, "--assess", "--type", "execute", "--verbose=4", str(path)]
    elif artifact_type == "dmg":
        command = [
            spctl,
            "--assess",
            "--type",
            "open",
            "--context",
            "context:primary-signature",
            "--verbose=4",
            str(path),
        ]
    else:
        raise ReleaseBuildError(f"Unsupported Gatekeeper artifact type: {artifact_type!r}")
    _run_native_check(command)


def _notarize_and_staple_dmg(path: Path) -> None:
    """Submit a DMG because Tauri notarizes/staples the app, not the image."""

    if sys.platform != "darwin":
        raise ReleaseBuildError("Apple DMG notarization can only run on macOS.")
    try:
        _verify_stapled_notarization(path)
        return
    except ReleaseBuildError:
        pass
    key_id = str(os.environ.get("APPLE_API_KEY", "")).strip()
    issuer = str(os.environ.get("APPLE_API_ISSUER", "")).strip()
    key_path = str(os.environ.get("APPLE_API_KEY_PATH", "")).strip()
    if not key_id or not issuer or not key_path or not Path(key_path).is_file():
        raise ReleaseBuildError(
            "DMG notarization requires APPLE_API_KEY, APPLE_API_ISSUER, and an existing APPLE_API_KEY_PATH."
        )
    xcrun = shutil.which("xcrun") or "/usr/bin/xcrun"
    if not Path(xcrun).is_file():
        raise ReleaseBuildError("xcrun is required to notarize the macOS DMG.")
    _run_native_check(
        [
            xcrun,
            "notarytool",
            "submit",
            str(path),
            "--key",
            key_path,
            "--key-id",
            key_id,
            "--issuer",
            issuer,
            "--wait",
        ],
        timeout=3600,
    )
    _run_native_check([xcrun, "stapler", "staple", str(path)], timeout=300)
    _verify_stapled_notarization(path)


def _verify_dmg(
    dmg: Path,
    context: Mapping[str, object],
    *,
    signing_identity: str | None,
    require_notarization: bool,
) -> None:
    if sys.platform != "darwin":
        raise ReleaseBuildError(
            "A DMG must be verified and mounted on macOS; off-host collection is not accepted."
        )
    hdiutil = shutil.which("hdiutil") or "/usr/bin/hdiutil"
    if not Path(hdiutil).is_file():
        raise ReleaseBuildError("hdiutil is required to validate a macOS DMG.")
    if require_notarization:
        _notarize_and_staple_dmg(dmg)
    _run_native_check([hdiutil, "verify", str(dmg)], timeout=300)
    if signing_identity is not None:
        _verify_codesign(dmg, signing_identity)
    mount_root = Path(tempfile.mkdtemp(prefix="mts-macos-dmg-"))
    attached = False
    try:
        _run_native_check(
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
        attached = True
        expected_name = f"{context['productName']}.app"
        apps = [
            path
            for path in _app_candidates(mount_root)
            if path.name == expected_name
        ]
        if len(apps) != 1:
            raise ReleaseBuildError(
                f"Mounted DMG must contain exactly one {expected_name}; found {len(apps)}."
            )
        mounted_app = apps[0]
        _validate_app_bundle(mounted_app, context)
        _validate_embedded_runtime(mounted_app, context["projectRoot"])  # type: ignore[arg-type]
        if signing_identity is not None:
            _verify_codesign(mounted_app, signing_identity)
        if require_notarization:
            _verify_stapled_notarization(mounted_app)
            _verify_stapled_notarization(dmg)
            _verify_gatekeeper(mounted_app, artifact_type="app")
            _verify_gatekeeper(dmg, artifact_type="dmg")
    finally:
        if attached:
            try:
                _run_native_check([hdiutil, "detach", str(mount_root)], timeout=120)
            except ReleaseBuildError:
                _run_native_check([hdiutil, "detach", "-force", str(mount_root)], timeout=120)
        shutil.rmtree(mount_root, ignore_errors=True)



def _collect_release(
    context: Mapping[str, object],
    discovered: Mapping[str, Sequence[Path]],
    *,
    signing_identity: str | None,
    allow_unsigned_development: bool,
    require_notarization: bool,
) -> tuple[Path, str, list[dict[str, object]]]:
    output = context["outputDirectory"]
    assert isinstance(output, Path)
    if output.exists():
        raise ReleaseBuildError(f"Output directory already exists; choose a new release directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    artifacts: list[dict[str, object]] = []
    file_ledger: list[dict[str, object]] = []
    try:
        if signing_identity is None and not allow_unsigned_development:
            raise ReleaseBuildError(
                "A macOS release requires a signing identity. "
                "Use --allow-unsigned-development only for local fixtures."
            )
        # A requested app is copied as a directory; a zip is generated from the
        # same source so users can download one portable file.
        app_source = next(iter(discovered.get("app", ())), None)
        if app_source is not None:
            _validate_app_bundle(app_source, context)
            app_source = _prepare_app_for_release(
                app_source,
                stage,
                context,
                signing_identity=signing_identity,
                allow_unsigned_development=allow_unsigned_development,
            )
            if signing_identity is not None:
                _verify_codesign(app_source, signing_identity)
            if require_notarization:
                _verify_stapled_notarization(app_source)
                _verify_gatekeeper(app_source, artifact_type="app")
        if app_source is not None and "app" in context["bundles"]:  # type: ignore[operator]
            app_dest = stage / "artifacts" / "app" / app_source.name
            _copy_artifact(app_source, app_dest)
            records = _hash_path(app_dest, stage)
            file_ledger.extend(records)
            artifacts.append(
                {
                    "bundle": "app",
                    "format": "app-bundle",
                    "path": app_dest.relative_to(stage).as_posix(),
                    "fileCount": len(records),
                    "totalBytes": sum(int(item["size"]) for item in records),
                }
            )
        if app_source is not None and "zip" in context["bundles"]:  # type: ignore[operator]
            zip_dest = (
                stage
                / "artifacts"
                / "zip"
                / f"{app_source.stem}-{context['version']}-{context['architecture']}.app.zip"
            )
            _write_deterministic_app_zip(app_source, zip_dest, int(context["sourceDateEpoch"]))
            file_ledger.extend(_hash_path(zip_dest, stage))
            artifacts.append(
                {
                    "bundle": "zip",
                    "format": "app-zip",
                    "path": zip_dest.relative_to(stage).as_posix(),
                    "fileCount": 1,
                    "totalBytes": zip_dest.stat().st_size,
                }
            )
        if "dmg" in context["bundles"]:  # type: ignore[operator]
            dmg_source = next(iter(discovered.get("dmg", ())), None)
            if dmg_source is None:
                raise ReleaseBuildError("Requested DMG bundle was not discovered.")
            _verify_dmg(
                dmg_source,
                context,
                signing_identity=signing_identity,
                require_notarization=require_notarization,
            )
            dmg_dest = stage / "artifacts" / "dmg" / dmg_source.name
            _copy_artifact(dmg_source, dmg_dest)
            file_ledger.extend(_hash_path(dmg_dest, stage))
            artifacts.append(
                {
                    "bundle": "dmg",
                    "format": "dmg",
                    "path": dmg_dest.relative_to(stage).as_posix(),
                    "fileCount": 1,
                    "totalBytes": dmg_dest.stat().st_size,
                }
            )
        if not artifacts:
            raise ReleaseBuildError("No macOS artifacts were selected for collection.")
        # The prepared copy is an internal repair workspace, never a release
        # artifact. Remove it before the final directory is atomically moved.
        shutil.rmtree(stage / ".prepared-app", ignore_errors=True)
        file_ledger.sort(key=lambda item: str(item["path"]))
        signing_mode = "codesign" if signing_identity is not None else "development-unsigned"
        # Tauri performs signing/notarization during bundling. A stapled claim
        # is emitted only after xcrun validates the ticket on the app and DMG.
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "contract": CONTRACT,
            "appId": context["appId"],
            "productName": context["productName"],
            "releaseId": (
                f"{context['appId']}/{context['version']}/macos/"
                f"{context['architecture']}/{context['channel']}"
            ),
            "version": context["version"],
            "platform": MACOS_PLATFORM,
            "architecture": context["architecture"],
            "channel": context["channel"],
            "sourceDateEpoch": context["sourceDateEpoch"],
            "build": {
                "targetTriple": context["targetTriple"],
                "gitCommit": _git_commit(context["projectRoot"]),
                "lockedInputs": context["lockedInputs"],
                "commands": context["commands"],
                "host": platform.platform(),
            },
            "trust": {
                "mode": signing_mode,
                "signingIdentity": signing_identity,
                "notarization": "stapled" if require_notarization else "not-claimed",
            },
            "runtime": {
                "root": "Contents/Resources/mts-runtime",
                "bundledPythonRuntime": False,
                "bundledModelArtifacts": False,
                "providerCatalog": "Contents/Resources/mts-runtime/configs/llm-provider-presets.v1.json",
                "modelCatalog": "Contents/Resources/mts-runtime/configs/model-catalog.v1.json",
            },
            "artifacts": artifacts,
            "fileCount": len(file_ledger),
            "totalBytes": sum(int(item["size"]) for item in file_ledger),
            "files": file_ledger,
        }
        manifest_path = stage / "macos-release-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_sha256 = _sha256(manifest_path)
        (stage / "macos-release-manifest.json.sha256").write_text(
            manifest_sha256 + "\n", encoding="ascii", newline="\n"
        )
        stage.replace(output)
        return output / manifest_path.name, manifest_sha256, artifacts
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build locked Tauri app/DMG/portable ZIP releases on macOS.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--target-directory", type=Path)
    parser.add_argument("--target-triple", choices=tuple(SUPPORTED_TARGETS), default=None)
    parser.add_argument("--architecture", choices=tuple(sorted(SUPPORTED_ARCHITECTURES)), default=None)
    parser.add_argument("--bundles", type=_parse_bundles, default=_parse_bundles("app,dmg"), metavar="LIST")
    parser.add_argument("--channel", default="development")
    parser.add_argument("--source-date-epoch", type=int, default=0)
    parser.add_argument(
        "--codesign-identity",
        default=None,
        help=(
            "Developer ID identity; defaults to APPLE_SIGNING_IDENTITY, "
            "MAC_CODESIGN_IDENTITY, or CODESIGN_IDENTITY."
        ),
    )
    parser.add_argument(
        "--allow-unsigned-development",
        action="store_true",
        help="allow an explicitly unsigned local fixture; never use for a published release",
    )
    parser.add_argument(
        "--require-notarization",
        action="store_true",
        help="require Tauri notarization and validate the stapled Apple ticket on every native artifact",
    )
    parser.add_argument("--skip-compile", action="store_true", help="collect already-built macOS bundles")
    parser.add_argument("--dry-run", action="store_true", help="validate and print plan without writing or compiling")
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="with --dry-run, atomically persist the plan and a .sha256 companion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = args.target_triple
        if target is not None and args.architecture is not None and SUPPORTED_TARGETS[target] != args.architecture:
            raise ReleaseBuildError("--architecture does not match --target-triple.")
        if target is None and args.architecture is not None:
            target = next(triple for triple, arch in SUPPORTED_TARGETS.items() if arch == args.architecture)
        target = target or _default_target_triple()
        context = _load_build_context(
            args.project_root,
            target,
            args.bundles,
            args.output_directory,
            args.target_directory,
            args.channel,
            args.source_date_epoch,
        )
        if context["channel"] == "stable":
            if args.allow_unsigned_development or not args.require_notarization:
                raise ReleaseBuildError(
                    "The stable macOS channel requires signed, notarized artifacts; "
                    "use --require-notarization without --allow-unsigned-development."
                )
            if "zip" in context["bundles"]:
                raise ReleaseBuildError(
                    "The stable macOS channel currently supports app and DMG only. "
                    "The portable ZIP remains development-only until native metadata preservation is accepted."
                )
        if args.plan_output is not None and not args.dry_run:
            raise ReleaseBuildError("--plan-output requires --dry-run.")
        if args.dry_run:
            plan = _json_plan(
                context,
                skip_compile=args.skip_compile,
                allow_unsigned_development=args.allow_unsigned_development,
                require_notarization=args.require_notarization,
            )
            if args.plan_output is not None:
                plan_path, plan_sha256 = _write_plan_artifact(plan, args.plan_output)
                plan = dict(plan)
                plan["planOutputPath"] = str(plan_path)
                plan["planSha256"] = plan_sha256
            print(json.dumps(plan, indent=2))
            return 0
        if not args.skip_compile and sys.platform != "darwin":
            raise ReleaseBuildError("macOS release compilation must run on a macOS host.")
        signing_identity = _resolve_signing_identity(args.codesign_identity)
        if context["channel"] == "stable" and (
            signing_identity is None
            or not signing_identity.casefold().startswith("developer id application:")
        ):
            raise ReleaseBuildError(
                "The stable macOS channel requires a Developer ID Application signing identity."
            )
        if args.require_notarization:
            if signing_identity is None:
                raise ReleaseBuildError("--require-notarization requires a code-signing identity.")
            if args.allow_unsigned_development:
                raise ReleaseBuildError(
                    "--require-notarization cannot be combined with --allow-unsigned-development."
                )
            required_notary_environment = (
                "APPLE_API_KEY",
                "APPLE_API_ISSUER",
                "APPLE_API_KEY_PATH",
            )
            if not all(str(os.environ.get(name, "")).strip() for name in required_notary_environment):
                raise ReleaseBuildError(
                    "--require-notarization requires APPLE_API_KEY, APPLE_API_ISSUER, "
                    "and APPLE_API_KEY_PATH for Tauri/notarytool authentication."
                )
        if not args.skip_compile:
            lock = context["buildLockFile"]
            assert isinstance(lock, Path)
            with _exclusive_build_lock(lock):
                _build(
                    context,
                    signing_identity=signing_identity,
                    allow_unsigned_development=args.allow_unsigned_development,
                )
                _assert_locked_inputs_unchanged(context)
                discovered = _discover_artifacts(context)
                manifest_path, manifest_sha256, artifacts = _collect_release(
                    context,
                    discovered,
                    signing_identity=signing_identity,
                    allow_unsigned_development=args.allow_unsigned_development,
                    require_notarization=args.require_notarization,
                )
        else:
            _assert_locked_inputs_unchanged(context)
            discovered = _discover_artifacts(context)
            manifest_path, manifest_sha256, artifacts = _collect_release(
                context,
                discovered,
                signing_identity=signing_identity,
                allow_unsigned_development=args.allow_unsigned_development,
                require_notarization=args.require_notarization,
            )
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "BuildTauriMacOSRelease",
                    "dryRun": False,
                    "version": context["version"],
                    "targetTriple": context["targetTriple"],
                    "architecture": context["architecture"],
                    "outputDirectory": str(context["outputDirectory"]),
                    "manifestPath": str(manifest_path),
                    "manifestSha256": manifest_sha256,
                    "artifacts": artifacts,
                },
                indent=2,
            )
        )
        return 0
    except ReleaseBuildError as exc:
        print(
            json.dumps({"ok": False, "action": "BuildTauriMacOSRelease", "error": str(exc)}, separators=(",", ":")),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
