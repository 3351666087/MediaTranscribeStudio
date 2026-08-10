#!/usr/bin/env python3
"""Build and collect locked Tauri Linux release bundles.

The script deliberately keeps the Linux artifact contract separate from the
Windows install/rollback bundle contract.  A dry run validates all source
metadata and lock files without creating the output or build-lock file.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import tomllib
from typing import Iterator, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - available on Linux build hosts
    fcntl = None  # type: ignore[assignment]


CONTRACT = "mts-tauri-linux-release/v1"
SCHEMA_VERSION = "1.0.0"
SUPPORTED_TARGETS = {
    "x86_64-unknown-linux-gnu": "x64",
    "aarch64-unknown-linux-gnu": "arm64",
}
SUPPORTED_BUNDLES = ("appimage", "deb", "rpm")
BUNDLE_SUFFIXES = {
    "appimage": ".AppImage",
    "deb": ".deb",
    "rpm": ".rpm",
}
RUNTIME_BOOTSTRAP_FILES = (
    ("configs/llm-provider-presets.v1.json", "configs/llm-provider-presets.v1.json"),
    ("configs/llm-provider-presets.schema.json", "configs/llm-provider-presets.schema.json"),
    ("configs/model-catalog.v1.json", "configs/model-catalog.v1.json"),
    ("configs/model-catalog.v1.schema.json", "configs/model-catalog.v1.schema.json"),
    ("production.config.example.json", "production.config.example.json"),
    ("production.config.remote.example.json", "production.config.remote.example.json"),
    ("requirements-media-asr.txt", "requirements-media-asr.txt"),
    ("requirements-pyannote.txt", "requirements-pyannote.txt"),
    ("tools/model_manager.py", "tools/model_manager.py"),
    ("tools/model_registry.py", "tools/model_registry.py"),
    ("tools/pyannote_runtime.py", "tools/pyannote_runtime.py"),
    ("tools/build_portable_model_catalog.py", "tools/build_portable_model_catalog.py"),
    ("packaging/tauri/bootstrap/Initialize-MtsRuntime.sh", "bootstrap/Initialize-MtsRuntime.sh"),
    ("packaging/tauri/bootstrap/Manage-MtsModels.sh", "bootstrap/Manage-MtsModels.sh"),
    ("packaging/tauri/bootstrap/README.md", "bootstrap/README.md"),
    ("pdf-renderer/target/pdf-renderer.jar", "pdf-renderer/target/pdf-renderer.jar"),
)
RUNTIME_BOOTSTRAP_TREES = (
    ("backend", "backend", {".py"}),
    ("contracts", "contracts", {".json", ".py"}),
    ("reporting", "reporting", {".py"}),
)
FORBIDDEN_MODEL_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".flac",
    ".gguf",
    ".h5",
    ".m4a",
    ".mlmodel",
    ".mov",
    ".mp3",
    ".mp4",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".wav",
}
FORBIDDEN_DIRECTORY_NAMES = {
    ".cache",
    ".downloads",
    ".git",
    "__pycache__",
    "_downloads",
    "models",
    "runtime",
}
SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bsk-[a-z0-9]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{20,}"),
    re.compile(r"(?i)\bghp_[a-z0-9]{20,}\b"),
    re.compile(r"(?i)\bhf_[a-z0-9]{20,}\b"),
)
SENSITIVE_JSON_VALUE_KEYS = {
    "access_token",
    "accesstoken",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "bearertoken",
}
PORTABLE_CONFIG_NAMES = {
    "production.config.example.json",
    "production.config.remote.example.json",
}
LINUX_PACKAGE_ARCHITECTURES = {
    "x64": {"deb": "amd64", "rpm": "x86_64", "elf": 62},
    "arm64": {"deb": "arm64", "rpm": "aarch64", "elf": 183},
}
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
CHANNEL_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class ReleaseBuildError(RuntimeError):
    """Raised when a release cannot be planned or collected safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _assert_cargo_lock_contains_package(
    path: Path, package_name: str, version: str
) -> None:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseBuildError(f"Unable to read Cargo.lock: {path}: {exc}") from exc
    packages = value.get("package")
    if not isinstance(packages, list):
        raise ReleaseBuildError(f"Cargo.lock does not contain package records: {path}")
    if not any(
        isinstance(package, dict)
        and package.get("name") == package_name
        and package.get("version") == version
        for package in packages
    ):
        raise ReleaseBuildError(
            f"Cargo.lock must contain {package_name} at the application version."
        )


def _relative_to_project(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _locked_input(path: Path, project_root: Path) -> dict[str, object]:
    return {
        "path": _relative_to_project(path, project_root),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


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


def _assert_no_secret_data(data: bytes, path: Path) -> None:
    text = data.decode("utf-8", errors="ignore")
    for pattern in SECRET_VALUE_PATTERNS:
        if pattern.search(text):
            raise ReleaseBuildError(
                f"Possible credential detected in runtime payload input: {path}"
            )
    if path.suffix.casefold() != ".json":
        return
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
                    # Empty values and environment-variable *names* are safe;
                    # concrete credential values are not.
                    candidate = item.strip()
                    if candidate and not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", candidate):
                        raise ReleaseBuildError(
                            f"Raw credential-like JSON value detected at {path}:{location}/{key_text}"
                        )
                inspect(item, f"{location}/{key_text}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, f"{location}/{index}")

    inspect(document, "")


def _assert_no_secret_bytes(path: Path) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to read runtime payload input {path}: {exc}") from exc
    _assert_no_secret_data(data, path)


def _portable_config_bytes(path: Path) -> bytes:
    """Return a config template with build-machine paths and secrets removed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"Unable to read runtime config template: {path}") from exc

    def rewrite(item: object) -> object:
        if isinstance(item, str):
            normalized = item.replace("\\", "/")
            folded = normalized.casefold()
            if folded.startswith("d:/models/"):
                return "models/" + normalized[len("D:/models/") :]
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
                return "runtime/" + normalized[marker.end() :]
            return item
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        if isinstance(item, dict):
            return {key: rewrite(child) for key, child in item.items()}
        return item

    portable = rewrite(value)
    data = (json.dumps(portable, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if re.search(r"(?i)(?<![A-Za-z0-9])[a-z]:[\\/]", data.decode("utf-8")):
        raise ReleaseBuildError(
            f"Runtime config template still contains a drive-qualified path: {path}"
        )
    _assert_no_secret_data(data, path)
    return data


def _runtime_bootstrap_plan(project_root: Path) -> dict[str, object]:
    """Build a hash-bound allowlist for the weight-free runtime payload."""

    # Small synthetic projects used by unit tests intentionally contain only
    # locked build metadata. A real repository is recognized by its registry;
    # there, missing bootstrap inputs are a release error rather than a silent
    # partial package.
    registry_present = (project_root / "local-model-registry.json").is_file()
    fixed = [project_root / source for source, _ in RUNTIME_BOOTSTRAP_FILES]
    trees = [project_root / source for source, _, _ in RUNTIME_BOOTSTRAP_TREES]
    missing = [str(path.relative_to(project_root)) for path in fixed if not path.is_file()]
    missing.extend(
        str(path.relative_to(project_root)) for path in trees if not path.is_dir()
    )
    if missing and registry_present:
        raise ReleaseBuildError(
            "Runtime bootstrap inputs are incomplete: " + ", ".join(sorted(missing))
        )
    if missing:
        return {
            "enabled": False,
            "bundledModelArtifacts": False,
            "absolutePathsIncluded": False,
            "embedded": False,
            "root": "mts-runtime",
            "entries": [],
            "files": [],
            "missing": sorted(missing),
        }

    entries: list[dict[str, object]] = []
    destinations: set[str] = set()

    def add(source: Path, destination: str, role: str) -> None:
        destination = _safe_relative(destination)
        folded_destination = destination.casefold()
        if folded_destination in destinations:
            raise ReleaseBuildError(f"Runtime payload has a duplicate destination: {destination}")
        destinations.add(folded_destination)
        if source.is_symlink():
            raise ReleaseBuildError(f"Symlinks are not allowed in runtime inputs: {source}")
        if not source.is_file():
            raise ReleaseBuildError(f"Required Linux runtime input is missing: {source}")
        if source.name in PORTABLE_CONFIG_NAMES:
            content = _portable_config_bytes(source)
        else:
            _assert_no_secret_bytes(source)
            try:
                content = source.read_bytes()
            except OSError as exc:
                raise ReleaseBuildError(f"Unable to read runtime input: {source}") from exc
        entries.append(
            {
                "source": source,
                "destination": destination,
                "role": role,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    fixed_roles = {
        "configs/llm-provider-presets.v1.json": "provider-presets",
        "configs/llm-provider-presets.schema.json": "provider-contract",
        "configs/model-catalog.v1.json": "portable-model-catalog",
        "configs/model-catalog.v1.schema.json": "model-catalog-contract",
        "production.config.example.json": "config-template",
        "production.config.remote.example.json": "remote-config-template",
        "requirements-media-asr.txt": "runtime-requirements",
        "requirements-pyannote.txt": "runtime-requirements",
        "tools/model_manager.py": "model-manager",
        "tools/model_registry.py": "model-registry-validator",
        "tools/pyannote_runtime.py": "pyannote-runtime",
        "tools/build_portable_model_catalog.py": "portable-catalog-builder",
        "packaging/tauri/bootstrap/Initialize-MtsRuntime.sh": "bootstrap-entrypoint",
        "packaging/tauri/bootstrap/Manage-MtsModels.sh": "model-manager-entrypoint",
        "packaging/tauri/bootstrap/README.md": "bootstrap-guide",
        "pdf-renderer/target/pdf-renderer.jar": "pdf-runtime",
    }
    for source_relative, destination in RUNTIME_BOOTSTRAP_FILES:
        add(project_root / source_relative, destination, fixed_roles[source_relative])

    for source, destination, extensions in RUNTIME_BOOTSTRAP_TREES:
        root = project_root / source
        root_resolved = root.resolve()
        for current, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                candidate = current_path / name
                if name in FORBIDDEN_DIRECTORY_NAMES:
                    continue
                if candidate.is_symlink():
                    raise ReleaseBuildError(
                        f"Symlinks are not allowed in runtime source trees: {candidate}"
                    )
                kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                path = current_path / name
                if path.is_symlink():
                    raise ReleaseBuildError(
                        f"Symlinks are not allowed in runtime source trees: {path}"
                    )
                try:
                    path.resolve().relative_to(root_resolved)
                except ValueError as exc:
                    raise ReleaseBuildError(f"Runtime source escapes its root: {path}") from exc
                if path.suffix.casefold() in FORBIDDEN_MODEL_EXTENSIONS:
                    raise ReleaseBuildError(
                        f"Model/media artifact is forbidden in runtime payload: {path}"
                    )
                if path.suffix.casefold() in extensions and "__pycache__" not in path.parts:
                    relative = (Path(destination) / path.relative_to(root)).as_posix()
                    add(path, relative, f"{source}-runtime")

    entries.sort(key=lambda item: str(item["destination"]))
    profile = {
        "schemaVersion": "1.0.0",
        "artifactType": "mts-linux-runtime-bootstrap",
        "runtimeStrategy": "external-or-operator-provided",
        "bundledPythonRuntime": False,
        "bundledModelArtifacts": False,
        "absolutePathsIncluded": False,
        "providerPresetCatalog": "configs/llm-provider-presets.v1.json",
        "modelCatalog": "configs/model-catalog.v1.json",
        "productionConfigTemplate": "production.config.example.json",
        "modelManager": "tools/model_manager.py",
        "runtimeInitializer": "bootstrap/Initialize-MtsRuntime.sh",
        "fileCount": len(entries) + 1,
    }
    profile_bytes = (json.dumps(profile, ensure_ascii=True, indent=2) + "\n").encode("utf-8")
    entries.append(
        {
            "source": None,
            "destination": "bootstrap/runtime-bootstrap.v1.json",
            "role": "runtime-profile",
            "size": len(profile_bytes),
            "sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "content": profile_bytes,
        }
    )
    entries.sort(key=lambda item: str(item["destination"]))
    return {
        "enabled": True,
        "bundledModelArtifacts": False,
        "absolutePathsIncluded": False,
        "embedded": True,
        "root": "mts-runtime",
        "entries": entries,
        "files": [str(item["destination"]) for item in entries],
        "missing": [],
    }


def _runtime_bootstrap_public(plan: Mapping[str, object]) -> dict[str, object]:
    entries = list(plan.get("entries", []))
    file_records = [
        {
            "path": (Path("mts-runtime") / str(item["destination"])).as_posix(),
            "size": int(item["size"]),
            "sha256": str(item["sha256"]),
        }
        for item in entries
    ]
    return {
        "enabled": bool(plan["enabled"]),
        "bundledModelArtifacts": False,
        "absolutePathsIncluded": False,
        "embedded": bool(plan.get("embedded", False)),
        "root": str(plan.get("root", "mts-runtime")),
        "fileCount": len(file_records),
        "totalBytes": sum(int(item["size"]) for item in file_records),
        "files": file_records,
        "missing": list(plan.get("missing", [])),
        "artifactVerifications": list(plan.get("artifactVerifications", [])),
    }


def _default_target_triple() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64-unknown-linux-gnu"
    if machine in {"aarch64", "arm64"}:
        return "aarch64-unknown-linux-gnu"
    raise ReleaseBuildError(
        f"Unsupported Linux host architecture {machine!r}; pass --target-triple explicitly."
    )


def _parse_bundles(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one bundle must be selected")
    unknown = sorted(set(values).difference(SUPPORTED_BUNDLES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unsupported bundle(s): " + ", ".join(unknown)
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("bundle selections must be unique")
    return values


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
    desktop_root = project / "apps" / "desktop"
    tauri_root = desktop_root / "src-tauri"
    paths = {
        "packageJson": desktop_root / "package.json",
        "tauriConfig": tauri_root / "tauri.conf.json",
        "cargoToml": tauri_root / "Cargo.toml",
        "packageLock": desktop_root / "package-lock.json",
        "cargoLock": tauri_root / "Cargo.lock",
    }
    for label, required in paths.items():
        if not required.is_file():
            raise ReleaseBuildError(
                f"Required locked build input is missing ({label}): {required}"
            )

    package_json = _read_json(paths["packageJson"])
    tauri_config = _read_json(paths["tauriConfig"])
    package_lock = _read_json(paths["packageLock"])
    cargo_package_name, cargo_version = _read_cargo_metadata(paths["cargoToml"])
    versions = {
        "package.json": package_json.get("version"),
        "tauri.conf.json": tauri_config.get("version"),
        "Cargo.toml": cargo_version,
    }
    if any(not isinstance(value, str) for value in versions.values()):
        raise ReleaseBuildError("Every application build input must declare a string version.")
    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        details = ", ".join(f"{name}={value!r}" for name, value in versions.items())
        raise ReleaseBuildError(
            f"package.json, tauri.conf.json, and Cargo.toml versions must match ({details})."
        )
    version = next(iter(unique_versions))
    assert isinstance(version, str)
    if not SEMVER_PATTERN.fullmatch(version):
        raise ReleaseBuildError(f"Application version must be SemVer 2.0.0: {version!r}")

    package_name = package_json.get("name")
    lock_packages = package_lock.get("packages")
    root_lock = lock_packages.get("") if isinstance(lock_packages, dict) else None
    if (
        not isinstance(root_lock, dict)
        or root_lock.get("version") != version
        or (
            isinstance(package_name, str)
            and root_lock.get("name") != package_name
        )
    ):
        raise ReleaseBuildError(
            "package-lock.json root package name/version must match package.json."
        )
    _assert_cargo_lock_contains_package(paths["cargoLock"], cargo_package_name, version)

    app_id = tauri_config.get("identifier")
    product_name = tauri_config.get("productName")
    if not isinstance(app_id, str) or not app_id.strip():
        raise ReleaseBuildError("tauri.conf.json must declare a non-empty identifier.")
    if not isinstance(product_name, str) or not product_name.strip():
        raise ReleaseBuildError("tauri.conf.json must declare a non-empty productName.")
    if target_triple not in SUPPORTED_TARGETS:
        raise ReleaseBuildError(f"Unsupported Linux target triple: {target_triple}")
    if not CHANNEL_PATTERN.fullmatch(channel):
        raise ReleaseBuildError(f"Invalid release channel: {channel!r}")
    if source_date_epoch < 0:
        raise ReleaseBuildError("--source-date-epoch must be non-negative.")

    architecture = SUPPORTED_TARGETS[target_triple]
    resolved_target = (
        target_directory.expanduser().resolve()
        if target_directory is not None
        else (tauri_root / "target").resolve()
    )
    resolved_output = (
        output_directory.expanduser().resolve()
        if output_directory is not None
        else (project / "dist" / "tauri-release" / f"{version}-linux-{architecture}").resolve()
    )
    artifact_search_root = resolved_target / target_triple / "release" / "bundle"
    commands = [
        "npm ci",
        (
            "npm run tauri -- build "
            f"--ci --target {target_triple} --bundles {','.join(bundles)} "
            "--config <generated-linux-runtime-overlay> -- --locked"
        ),
    ]
    locked_inputs = [
        _locked_input(paths["packageLock"], project),
        _locked_input(paths["cargoLock"], project),
    ]
    runtime_bootstrap = _runtime_bootstrap_plan(project)
    return {
        "projectRoot": project,
        "desktopRoot": desktop_root,
        "tauriRoot": tauri_root,
        "version": version,
        "appId": app_id,
        "productName": product_name,
        "architecture": architecture,
        "targetTriple": target_triple,
        "bundles": list(bundles),
        "channel": channel,
        "sourceDateEpoch": source_date_epoch,
        "targetDirectory": resolved_target,
        "outputDirectory": resolved_output,
        "artifactSearchRoot": artifact_search_root,
        "buildLockFile": resolved_target / ".mts-tauri-linux-build.lock",
        "lockedInputs": locked_inputs,
        "commands": commands,
        "runtimeBootstrap": runtime_bootstrap,
    }


def _json_plan(context: dict[str, object], *, skip_compile: bool) -> dict[str, object]:
    return {
        "ok": True,
        "action": "BuildTauriLinuxRelease",
        "dryRun": True,
        "projectRoot": str(context["projectRoot"]),
        "version": context["version"],
        "appId": context["appId"],
        "productName": context["productName"],
        "platform": "linux",
        "architecture": context["architecture"],
        "targetTriple": context["targetTriple"],
        "bundles": context["bundles"],
        "channel": context["channel"],
        "sourceDateEpoch": context["sourceDateEpoch"],
        "targetDirectory": str(context["targetDirectory"]),
        "artifactSearchRoot": str(context["artifactSearchRoot"]),
        "outputDirectory": str(context["outputDirectory"]),
        "buildLockFile": str(context["buildLockFile"]),
        "lockedInputs": context["lockedInputs"],
        "commands": [] if skip_compile else context["commands"],
        "runtimeBootstrap": _runtime_bootstrap_public(context["runtimeBootstrap"]),
        "postBuild": [
            "verify-locked-input-hashes-unchanged",
            "verify-runtime-source-hashes-unchanged",
            "collect-requested-linux-bundles",
            "verify-embedded-runtime-in-every-bundle",
            "verify-artifact-paths-and-types",
            "emit-byte-hashed-linux-release-manifest",
            "emit-release-manifest-sha256",
        ],
    }


@contextmanager
def _exclusive_build_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise ReleaseBuildError(
            "The Linux build lock requires fcntl; run compilation on a Linux host."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseBuildError(f"Another Linux release build holds the lock: {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run(command: Sequence[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=False,
        )
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to execute {command[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        raise ReleaseBuildError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def _runtime_entry_content(item: Mapping[str, object]) -> bytes:
    source = item.get("source")
    content = item.get("content")
    if isinstance(content, bytes):
        value = content
    elif isinstance(source, Path):
        try:
            value = (
                _portable_config_bytes(source)
                if source.name in PORTABLE_CONFIG_NAMES
                else source.read_bytes()
            )
        except OSError as exc:
            raise ReleaseBuildError(f"Unable to read runtime input: {source}") from exc
    else:
        raise ReleaseBuildError("Runtime plan entry has neither a source nor generated content.")
    expected_size = int(item["size"])
    expected_sha = str(item["sha256"])
    if len(value) != expected_size or hashlib.sha256(value).hexdigest() != expected_sha:
        source_text = str(source) if source is not None else str(item["destination"])
        raise ReleaseBuildError(f"Runtime input changed after planning: {source_text}")
    _assert_no_secret_data(value, Path(str(item["destination"])))
    return value


def _assert_runtime_inputs_unchanged(context: Mapping[str, object]) -> None:
    plan = context["runtimeBootstrap"]
    if not isinstance(plan, Mapping) or not bool(plan.get("enabled")):
        return
    for item in plan.get("entries", []):
        if not isinstance(item, Mapping):
            raise ReleaseBuildError("Runtime plan contains a malformed entry.")
        _runtime_entry_content(item)


def _stage_runtime_payload(
    project_root: Path,
    destination_root: Path,
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    if not bool(plan.get("enabled")):
        return []
    destination_root.mkdir(parents=True, exist_ok=True)
    ledger: list[dict[str, object]] = []
    for raw_item in plan.get("entries", []):
        if not isinstance(raw_item, Mapping):
            raise ReleaseBuildError("Runtime plan contains a malformed entry.")
        relative = _safe_relative(str(raw_item["destination"]))
        destination = destination_root / Path(relative)
        try:
            destination.resolve().relative_to(destination_root.resolve())
        except ValueError as exc:
            raise ReleaseBuildError(f"Runtime destination escapes its root: {relative}") from exc
        content = _runtime_entry_content(raw_item)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        source = raw_item.get("source")
        mode = 0o644
        if isinstance(source, Path):
            try:
                mode = stat.S_IMODE(source.stat().st_mode) or mode
            except OSError:
                pass
        if destination.suffix == ".sh":
            mode |= 0o111
        destination.chmod(mode & 0o777)
        ledger.append(
            {
                "path": (Path("mts-runtime") / relative).as_posix(),
                "size": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    ledger.sort(key=lambda item: str(item["path"]))
    return ledger


def _write_linux_overlay(path: Path, runtime_stage: Path, tauri_root: Path) -> None:
    """Create a temporary Tauri config that embeds the staged runtime."""

    icon = tauri_root / "icons" / "icon.png"
    if not icon.is_file():
        icon = tauri_root / "icons" / "icon.ico"
    config: dict[str, object] = {
        "bundle": {
            "resources": {str(runtime_stage): "mts-runtime"},
            "icon": [str(icon)] if icon.is_file() else [],
        }
    }
    path.write_text(
        json.dumps(config, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _build(context: Mapping[str, object]) -> None:
    if not sys.platform.startswith("linux"):
        raise ReleaseBuildError("Linux release compilation must run on a Linux host.")
    environment = dict(os.environ)
    environment["CARGO_TARGET_DIR"] = str(context["targetDirectory"])
    environment["SOURCE_DATE_EPOCH"] = str(context["sourceDateEpoch"])
    desktop_root = context["desktopRoot"]
    project_root = context["projectRoot"]
    tauri_root = context["tauriRoot"]
    assert (
        isinstance(desktop_root, Path)
        and isinstance(project_root, Path)
        and isinstance(tauri_root, Path)
    )
    temporary_root = Path(tempfile.mkdtemp(prefix="mts-linux-build-"))
    runtime_stage = temporary_root / "runtime"
    overlay = temporary_root / "tauri.linux.overlay.json"
    try:
        plan = context["runtimeBootstrap"]
        assert isinstance(plan, Mapping)
        _stage_runtime_payload(project_root, runtime_stage, plan)
        if bool(plan.get("enabled")):
            _write_linux_overlay(overlay, runtime_stage, tauri_root)
        _run(["npm", "ci"], cwd=desktop_root, environment=environment)
        command = [
            "npm",
            "run",
            "tauri",
            "--",
            "build",
            "--ci",
            "--target",
            str(context["targetTriple"]),
            "--bundles",
            ",".join(context["bundles"]),
        ]
        if bool(plan.get("enabled")):
            command.extend(["--config", str(overlay)])
        command.extend(["--", "--locked"])
        _run(command, cwd=desktop_root, environment=environment)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _assert_locked_inputs_unchanged(context: dict[str, object]) -> None:
    project_root = context["projectRoot"]
    assert isinstance(project_root, Path)
    for expected in context["lockedInputs"]:
        path = project_root / str(expected["path"])
        if not path.is_file():
            raise ReleaseBuildError(
                f"Locked build input disappeared during the build: {path}"
            )
        if (
            path.stat().st_size != expected["size"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ReleaseBuildError(
                f"Locked build input changed during the build: {path}"
            )


def _read_elf_machine(path: Path) -> int:
    try:
        header = path.read_bytes()[:20]
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to read Linux executable header: {path}") from exc
    if len(header) < 20 or header[:4] != b"\x7fELF":
        raise ReleaseBuildError(f"Linux AppImage is not an ELF executable: {path}")
    data_encoding = header[5]
    if data_encoding == 1:
        return struct.unpack_from("<H", header, 18)[0]
    if data_encoding == 2:
        return struct.unpack_from(">H", header, 18)[0]
    raise ReleaseBuildError(f"Linux AppImage has an invalid ELF byte order: {path}")


def _run_capture(command: Sequence[str], *, input_bytes: bytes | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            input=input_bytes,
            capture_output=True,
            check=False,
            text=input_bytes is None,
            encoding="utf-8" if input_bytes is None else None,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseBuildError(f"Unable to execute {' '.join(command)}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ReleaseBuildError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
            + (f" ({detail})" if detail else "")
        )
    value = completed.stdout
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def _validate_artifact_identity(
    bundle: str, path: Path, context: Mapping[str, object]
) -> None:
    architecture = str(context["architecture"])
    expected = LINUX_PACKAGE_ARCHITECTURES[architecture]
    version = str(context["version"])
    if version not in path.name:
        raise ReleaseBuildError(
            f"{bundle} artifact filename does not contain application version {version}: {path.name}"
        )
    if bundle == "appimage":
        if not (path.stat().st_mode & 0o111):
            raise ReleaseBuildError(f"AppImage is not executable: {path}")
        if _read_elf_machine(path) != expected["elf"]:
            raise ReleaseBuildError(
                f"AppImage architecture does not match {architecture}: {path}"
            )
        return
    if bundle == "deb":
        if path.read_bytes()[:8] != b"!<arch>\n":
            raise ReleaseBuildError(f"Debian artifact has an invalid ar header: {path}")
        tool = shutil.which("dpkg-deb")
        if tool is None:
            raise ReleaseBuildError("dpkg-deb is required to verify a .deb artifact.")
        artifact_version = _run_capture([tool, "-f", str(path), "Version"]).strip()
        artifact_architecture = _run_capture(
            [tool, "-f", str(path), "Architecture"]
        ).strip()
        if artifact_version != version or artifact_architecture != expected["deb"]:
            raise ReleaseBuildError(
                f"Debian artifact metadata does not match version/architecture: {path}"
            )
        return
    if bundle == "rpm":
        if path.read_bytes()[:4] != b"\xed\xab\xee\xdb":
            raise ReleaseBuildError(f"RPM artifact has an invalid header: {path}")
        tool = shutil.which("rpm")
        if tool is None:
            raise ReleaseBuildError("rpm is required to verify an .rpm artifact.")
        metadata = _run_capture(
            [tool, "-qp", "--qf", "%{VERSION}\\n%{ARCH}\\n", str(path)]
        ).splitlines()
        if len(metadata) < 2 or metadata[0].strip() != version or metadata[1].strip() != expected["rpm"]:
            raise ReleaseBuildError(
                f"RPM artifact metadata does not match version/architecture: {path}"
            )
        return
    raise ReleaseBuildError(f"Unsupported Linux artifact type: {bundle}")


def _discover_artifacts(context: Mapping[str, object]) -> list[tuple[str, Path]]:
    root = context["artifactSearchRoot"]
    assert isinstance(root, Path)
    if not root.is_dir():
        raise ReleaseBuildError(f"Tauri Linux bundle directory is missing: {root}")
    artifacts: list[tuple[str, Path]] = []
    for bundle in context["bundles"]:
        bundle_root = root / bundle
        suffix = BUNDLE_SUFFIXES[bundle]
        candidates = sorted(
            path
            for path in bundle_root.rglob("*")
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() == suffix.lower()
            )
        ) if bundle_root.is_dir() else []
        if not candidates:
            raise ReleaseBuildError(
                f"Requested {bundle} bundle was not produced below {bundle_root}."
            )
        named = [path for path in candidates if str(context["version"]) in path.name]
        if not named:
            raise ReleaseBuildError(
                f"No {bundle} artifact for version {context['version']} was found below {bundle_root}."
            )
        compatible: list[Path] = []
        failures: list[str] = []
        for candidate in named:
            try:
                candidate.resolve().relative_to(bundle_root.resolve())
            except ValueError as exc:
                raise ReleaseBuildError(
                    f"Artifact escapes its bundle directory: {candidate}"
                ) from exc
            try:
                _validate_artifact_identity(bundle, candidate, context)
            except ReleaseBuildError as exc:
                failures.append(str(exc))
                continue
            compatible.append(candidate)
        if len(compatible) != 1:
            details = "; ".join(failures)
            if len(compatible) > 1:
                raise ReleaseBuildError(
                    f"Multiple compatible {bundle} artifacts were produced for version {context['version']}: "
                    + ", ".join(str(path) for path in compatible)
                )
            raise ReleaseBuildError(
                f"No compatible {bundle} artifact was produced for target {context['targetTriple']}."
                + (f" {details}" if details else "")
            )
        artifacts.append((bundle, compatible[0]))
    return artifacts


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
    return (
        value
        if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40,64}", value)
        else "unknown"
    )


def _extract_rpm(artifact: Path, destination: Path) -> None:
    bsdtar = shutil.which("bsdtar")
    if bsdtar is not None:
        _run_capture([bsdtar, "-xf", str(artifact), "-C", str(destination)])
        return
    rpm2cpio = shutil.which("rpm2cpio")
    cpio = shutil.which("cpio")
    if rpm2cpio is None or cpio is None:
        raise ReleaseBuildError(
            "bsdtar or rpm2cpio+cpio is required to inspect an RPM payload."
        )
    try:
        producer = subprocess.Popen(
            [rpm2cpio, str(artifact)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert producer.stdout is not None
        consumer = subprocess.Popen(
            [cpio, "-idm", "--quiet", "--no-absolute-filenames"],
            cwd=destination,
            stdin=producer.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        producer.stdout.close()
        _, consumer_stderr = consumer.communicate(timeout=120)
        producer_stderr = producer.stderr.read() if producer.stderr is not None else b""
        producer_returncode = producer.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseBuildError(f"Unable to extract RPM artifact {artifact}: {exc}") from exc
    if producer_returncode != 0 or consumer.returncode != 0:
        detail = (producer_stderr + consumer_stderr).decode("utf-8", errors="replace").strip()
        raise ReleaseBuildError(f"Unable to extract RPM artifact {artifact}: {detail}")


def _appimage_squashfs_offset(artifact: Path) -> int:
    """Read the Type-2 AppImage trailer offset from its ELF section table."""

    try:
        header = artifact.read_bytes()[:64]
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to read AppImage ELF header: {artifact}") from exc
    if len(header) < 16 or header[:4] != b"\x7fELF":
        raise ReleaseBuildError(f"AppImage is not an ELF executable: {artifact}")
    byte_order = "<" if header[5] == 1 else ">" if header[5] == 2 else ""
    if not byte_order:
        raise ReleaseBuildError(f"AppImage has an invalid ELF byte order: {artifact}")
    elf_class = header[4]
    if elf_class == 2 and len(header) >= 64:
        section_offset = struct.unpack_from(byte_order + "Q", header, 40)[0]
        section_size = struct.unpack_from(byte_order + "H", header, 58)[0]
        section_count = struct.unpack_from(byte_order + "H", header, 60)[0]
    elif elf_class == 1 and len(header) >= 52:
        section_offset = struct.unpack_from(byte_order + "I", header, 32)[0]
        section_size = struct.unpack_from(byte_order + "H", header, 46)[0]
        section_count = struct.unpack_from(byte_order + "H", header, 48)[0]
    else:
        raise ReleaseBuildError(f"AppImage has an unsupported ELF class: {artifact}")
    offset = int(section_offset + section_size * section_count)
    try:
        with artifact.open("rb") as stream:
            stream.seek(offset)
            magic = stream.read(4)
    except OSError as exc:
        raise ReleaseBuildError(f"Unable to inspect AppImage trailer: {artifact}") from exc
    if magic != b"hsqs":
        raise ReleaseBuildError(
            f"AppImage does not contain a SquashFS trailer at ELF section end ({offset}): {artifact}"
        )
    return offset


def _extract_linux_artifact(bundle: str, artifact: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if bundle == "appimage":
        unsquashfs = shutil.which("unsquashfs")
        if unsquashfs is None:
            raise ReleaseBuildError(
                "unsquashfs is required to verify the embedded AppImage runtime."
            )
        offset = _appimage_squashfs_offset(artifact)
        _run_capture(
            [
                unsquashfs,
                "-offset",
                str(offset),
                "-no-progress",
                "-force",
                "-dest",
                str(destination),
                str(artifact),
            ]
        )
        return
    if bundle == "deb":
        dpkg_deb = shutil.which("dpkg-deb")
        if dpkg_deb is None:
            raise ReleaseBuildError("dpkg-deb is required to inspect a .deb payload.")
        _run_capture([dpkg_deb, "-x", str(artifact), str(destination)])
        return
    if bundle == "rpm":
        _extract_rpm(artifact, destination)
        return
    raise ReleaseBuildError(f"Unsupported Linux bundle type: {bundle}")


def _verify_embedded_runtime(
    bundle: str,
    artifact: Path,
    plan: Mapping[str, object],
) -> dict[str, object]:
    if not bool(plan.get("enabled")):
        return {
            "bundle": bundle,
            "verified": False,
            "reason": "runtime-bootstrap-disabled",
        }
    temporary_root = Path(tempfile.mkdtemp(prefix=f"mts-linux-{bundle}-inspect-"))
    try:
        _extract_linux_artifact(bundle, artifact, temporary_root)
        markers = sorted(temporary_root.rglob("mts-runtime/backend/worker.py"))
        if len(markers) != 1:
            raise ReleaseBuildError(
                f"{bundle} must contain exactly one mts-runtime/backend/worker.py; found {len(markers)}."
            )
        runtime_root = markers[0].parents[1]
        relative_root = runtime_root.relative_to(temporary_root)
        parts = relative_root.parts
        if len(parts) < 4 or parts[0:2] != ("usr", "lib") or parts[-1] != "mts-runtime":
            raise ReleaseBuildError(
                f"{bundle} runtime is outside the standard Tauri Linux resource directory: {relative_root}"
            )

        expected = {
            str(item["destination"]): (int(item["size"]), str(item["sha256"]))
            for item in plan.get("entries", [])
            if isinstance(item, Mapping)
        }
        actual: dict[str, tuple[int, str]] = {}
        for current, directory_names, file_names in os.walk(
            runtime_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            for name in directory_names:
                candidate = current_path / name
                if candidate.is_symlink():
                    raise ReleaseBuildError(
                        f"Embedded Linux runtime contains a directory symlink: {candidate}"
                    )
            for name in file_names:
                candidate = current_path / name
                if candidate.is_symlink() or not candidate.is_file():
                    raise ReleaseBuildError(
                        f"Embedded Linux runtime contains a non-regular file: {candidate}"
                    )
                relative = candidate.relative_to(runtime_root).as_posix()
                if candidate.suffix.casefold() in FORBIDDEN_MODEL_EXTENSIONS:
                    raise ReleaseBuildError(
                        f"Embedded Linux runtime contains a model/media artifact: {candidate}"
                    )
                _assert_no_secret_bytes(candidate)
                actual[relative] = (candidate.stat().st_size, _sha256(candidate))
        if actual != expected:
            missing = sorted(set(expected).difference(actual))
            unexpected = sorted(set(actual).difference(expected))
            changed = sorted(
                path
                for path in set(actual).intersection(expected)
                if actual[path] != expected[path]
            )
            raise ReleaseBuildError(
                f"{bundle} embedded runtime does not match the staged allowlist "
                f"(missing={missing}, unexpected={unexpected}, changed={changed})."
            )
        for executable in (
            "bootstrap/Initialize-MtsRuntime.sh",
            "bootstrap/Manage-MtsModels.sh",
        ):
            if not (runtime_root / executable).stat().st_mode & 0o111:
                raise ReleaseBuildError(
                    f"Embedded Linux bootstrap is not executable: {executable}"
                )
        return {
            "bundle": bundle,
            "verified": True,
            "packageRuntimeRoot": relative_root.as_posix(),
            "fileCount": len(actual),
            "totalBytes": sum(size for size, _ in actual.values()),
        }
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _collect_release(
    context: dict[str, object], artifacts: Sequence[tuple[str, Path]]
) -> tuple[Path, str, list[dict[str, object]], list[dict[str, object]]]:
    output = context["outputDirectory"]
    project_root = context["projectRoot"]
    assert isinstance(output, Path)
    assert isinstance(project_root, Path)
    if output.exists():
        raise ReleaseBuildError(
            f"Output directory already exists; choose a new release directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    ledger: list[dict[str, object]] = []
    runtime_verifications: list[dict[str, object]] = []
    try:
        seen_destinations: set[str] = set()
        for bundle, source in artifacts:
            plan = context["runtimeBootstrap"]
            assert isinstance(plan, Mapping)
            verification = _verify_embedded_runtime(bundle, source, plan)
            relative = Path("artifacts") / bundle / source.name
            relative_text = relative.as_posix()
            if relative_text.casefold() in seen_destinations:
                raise ReleaseBuildError(f"Duplicate artifact destination: {relative_text}")
            seen_destinations.add(relative_text.casefold())
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if (
                source.stat().st_size != destination.stat().st_size
                or _sha256(source) != _sha256(destination)
            ):
                raise ReleaseBuildError(
                    f"Artifact copy verification failed: {source}"
                )
            ledger.append(
                {
                    "bundle": bundle,
                    "path": relative_text,
                    "size": destination.stat().st_size,
                    "sha256": _sha256(destination),
                    "executable": bool(destination.stat().st_mode & 0o111),
                }
            )
            runtime_verifications.append(verification | {"artifactPath": relative_text})
        ledger.sort(key=lambda item: str(item["path"]))
        total_bytes = sum(int(item["size"]) for item in ledger)
        runtime_verifications.sort(key=lambda item: str(item["artifactPath"]))
        runtime_public = _runtime_bootstrap_public(context["runtimeBootstrap"])
        runtime_public["artifactVerifications"] = runtime_verifications
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "contract": CONTRACT,
            "appId": context["appId"],
            "productName": context["productName"],
            "releaseId": (
                f"{context['appId']}/{context['version']}/linux/"
                f"{context['architecture']}/{context['channel']}"
            ),
            "version": context["version"],
            "platform": "linux",
            "architecture": context["architecture"],
            "channel": context["channel"],
            "sourceDateEpoch": context["sourceDateEpoch"],
            "build": {
                "targetTriple": context["targetTriple"],
                "gitCommit": _git_commit(project_root),
                "lockedInputs": context["lockedInputs"],
                "commands": context["commands"],
            },
            "bundles": context["bundles"],
            "artifactCount": len(ledger),
            "totalBytes": total_bytes,
            "artifacts": ledger,
            "runtimeBootstrap": runtime_public,
        }
        manifest_path = stage / "linux-release-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        manifest_sha256 = _sha256(manifest_path)
        (stage / "linux-release-manifest.json.sha256").write_text(
            manifest_sha256 + "\n", encoding="ascii", newline="\n"
        )
        stage.replace(output)
        return output / manifest_path.name, manifest_sha256, ledger, runtime_verifications
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build locked Tauri AppImage, deb, and rpm release artifacts on Linux."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--target-directory", type=Path)
    parser.add_argument(
        "--target-triple",
        choices=tuple(SUPPORTED_TARGETS),
        default=None,
    )
    parser.add_argument(
        "--architecture",
        choices=("x64", "arm64"),
        default=None,
        help="architecture alias; target triple remains the canonical selector",
    )
    parser.add_argument(
        "--bundles",
        type=_parse_bundles,
        default=_parse_bundles("appimage,deb,rpm"),
        metavar="LIST",
        help="comma-separated subset of appimage,deb,rpm",
    )
    parser.add_argument("--channel", default="stable")
    parser.add_argument("--source-date-epoch", type=int, default=0)
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="collect already-built bundles from the target directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the locked plan without writing or compiling",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        target_triple = args.target_triple
        if target_triple is not None and args.architecture is not None:
            if SUPPORTED_TARGETS[target_triple] != args.architecture:
                raise ReleaseBuildError(
                    "--architecture does not match --target-triple."
                )
        if target_triple is None and args.architecture is not None:
            target_triple = next(
                triple
                for triple, architecture in SUPPORTED_TARGETS.items()
                if architecture == args.architecture
            )
        target_triple = target_triple or _default_target_triple()
        context = _load_build_context(
            args.project_root,
            target_triple,
            args.bundles,
            args.output_directory,
            args.target_directory,
            args.channel,
            args.source_date_epoch,
        )
        if args.dry_run:
            print(json.dumps(_json_plan(context, skip_compile=args.skip_compile), indent=2))
            return 0
        if not sys.platform.startswith("linux"):
            raise ReleaseBuildError(
                "Linux release build and package verification must run on a Linux host."
            )
        output = context["outputDirectory"]
        assert isinstance(output, Path)
        if output.exists():
            raise ReleaseBuildError(
                f"Output directory already exists; choose a new release directory: {output}"
            )
        lock_path = context["buildLockFile"]
        assert isinstance(lock_path, Path)
        with _exclusive_build_lock(lock_path):
            if not args.skip_compile:
                _build(context)
            _assert_locked_inputs_unchanged(context)
            _assert_runtime_inputs_unchanged(context)
            artifacts = _discover_artifacts(context)
            manifest_path, manifest_sha256, ledger, runtime_verifications = _collect_release(
                context, artifacts
            )
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "BuildTauriLinuxRelease",
                    "dryRun": False,
                    "version": context["version"],
                    "targetTriple": context["targetTriple"],
                    "outputDirectory": str(context["outputDirectory"]),
                    "manifestPath": str(manifest_path),
                    "manifestSha256": manifest_sha256,
                    "artifacts": ledger,
                    "runtimeBootstrap": {
                        "enabled": bool(context["runtimeBootstrap"]["enabled"]),
                        "embedded": bool(context["runtimeBootstrap"].get("embedded", False)),
                        "bundledModelArtifacts": False,
                        "fileCount": len(context["runtimeBootstrap"].get("entries", [])),
                        "totalBytes": sum(
                            int(item["size"])
                            for item in context["runtimeBootstrap"].get("entries", [])
                        ),
                        "artifactVerifications": runtime_verifications,
                    },
                },
                indent=2,
            )
        )
        return 0
    except ReleaseBuildError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "action": "BuildTauriLinuxRelease",
                    "error": str(exc),
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
