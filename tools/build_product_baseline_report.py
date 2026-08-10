"""Freeze a reproducible snapshot of the current product development baseline."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    sha256_file,
)
from backend.production_config import ProductionConfig  # noqa: E402
from tools.model_registry import (  # noqa: E402
    load_registry,
    validate_registry,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_TYPE = "product-development-baseline"
_DRIVE_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<tail>.*)$")
_DEFAULT_SOURCE_ROOTS = (
    "apps",
    "backend",
    "contracts",
    "diar_fusion",
    "docs",
    "mts_ui",
    "native",
    "packaging",
    "pdf-renderer/src",
    "reporting",
    "sample_library",
    "tests",
    "tools",
)
_IGNORED_SOURCE_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "target",
    }
)
_COMMAND_OUTPUT_LIMIT = 8192


class BaselineCaptureError(RuntimeError):
    """Raised when the baseline cannot be captured without ambiguity."""


def _host_path(path: str | os.PathLike[str]) -> Path:
    value = os.fspath(path)
    match = _DRIVE_PATH_RE.fullmatch(value)
    if os.name != "nt" and match is not None:
        tail = PurePosixPath(match.group("tail").replace("\\", "/"))
        return Path("/mnt") / match.group("drive").lower() / Path(*tail.parts)
    return Path(value)


def _read_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise BaselineCaptureError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BaselineCaptureError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, Mapping):
        raise BaselineCaptureError(f"JSON artifact must be an object: {path}")
    return value


def _run_bytes(
    command: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout_seconds: float = 30.0,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BaselineCaptureError(
            f"command could not complete: {' '.join(command)}"
        ) from error
    if check and completed.returncode != 0:
        output = completed.stdout.decode("utf-8", errors="replace")
        raise BaselineCaptureError(
            f"command failed ({completed.returncode}): {' '.join(command)}: "
            f"{output[:1000]}"
        )
    return completed


def _git_capture(repository_root: Path) -> dict[str, Any]:
    head = _run_bytes(
        ("git", "rev-parse", "HEAD"), cwd=repository_root
    ).stdout.strip()
    branch = _run_bytes(
        ("git", "branch", "--show-current"), cwd=repository_root
    ).stdout.strip()
    status = _run_bytes(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ),
        cwd=repository_root,
    ).stdout
    diff = _run_bytes(
        ("git", "diff", "--binary", "HEAD", "--"),
        cwd=repository_root,
        timeout_seconds=120.0,
    ).stdout
    return {
        "headCommit": head.decode("ascii"),
        "branch": branch.decode("utf-8", errors="strict"),
        "dirty": bool(status),
        "statusPorcelainSha256": hashlib.sha256(status).hexdigest(),
        "statusEntryCount": len(status.splitlines()),
        "trackedDiffSha256": hashlib.sha256(diff).hexdigest(),
        "trackedDiffBytes": len(diff),
    }


def _tracked_paths(repository_root: Path) -> set[Path]:
    raw = _run_bytes(
        ("git", "ls-files", "-z"), cwd=repository_root
    ).stdout
    paths: set[Path] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        relative = Path(item.decode("utf-8", errors="strict"))
        candidate = repository_root / relative
        if candidate.is_file() and not candidate.is_symlink():
            paths.add(candidate)
    return paths


def _additional_source_paths(
    repository_root: Path,
    source_roots: Iterable[str],
) -> set[Path]:
    paths: set[Path] = set()
    for raw in source_roots:
        candidate = repository_root / raw
        if candidate.is_symlink():
            raise BaselineCaptureError(
                f"source snapshot root must not be a symlink: {candidate}"
            )
        if candidate.is_file():
            paths.add(candidate)
            continue
        if not candidate.is_dir():
            continue
        for path in candidate.rglob("*"):
            relative = path.relative_to(repository_root)
            if any(part in _IGNORED_SOURCE_PARTS for part in relative.parts):
                continue
            if path.is_symlink():
                raise BaselineCaptureError(
                    f"source snapshot contains a symlink: {path}"
                )
            if path.is_file() and not path.name.endswith((".pyc", ".pyo")):
                paths.add(path)
    return paths


def build_source_snapshot(
    repository_root: Path,
    *,
    source_roots: Sequence[str] = _DEFAULT_SOURCE_ROOTS,
) -> dict[str, Any]:
    paths = _tracked_paths(repository_root)
    paths.update(_additional_source_paths(repository_root, source_roots))
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(paths, key=lambda value: value.as_posix().casefold()):
        try:
            relative = path.relative_to(repository_root).as_posix()
            size = path.stat().st_size
            digest = sha256_file(path)
        except OSError as error:
            raise BaselineCaptureError(
                f"source file changed during capture: {path}"
            ) from error
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        total_bytes += size
    if not rows:
        raise BaselineCaptureError("source snapshot contains no files")
    return {
        "algorithm": "tracked-plus-source-roots-v1",
        "sourceRoots": list(source_roots),
        "fileCount": len(rows),
        "totalBytes": total_bytes,
        "filesCanonicalSha256": canonical_json_sha256(rows),
        "files": rows,
    }


def _artifact_evidence(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BaselineCaptureError(f"required artifact is missing: {path}")
    value = _read_json(path)
    counts: dict[str, int] = {}
    for field in ("cases", "models", "sources", "trials"):
        items = value.get(field)
        if isinstance(items, list):
            counts[field] = len(items)
    declared_counts = value.get("counts")
    if isinstance(declared_counts, Mapping):
        for key, item in declared_counts.items():
            if isinstance(item, int) and not isinstance(item, bool):
                counts[f"declared.{key}"] = item
    identity = {
        key: value[key]
        for key in (
            "schemaVersion",
            "artifactType",
            "libraryId",
            "matrixId",
            "trialSetId",
            "canonicalSha256",
        )
        if isinstance(value.get(key), (str, int, bool))
    }
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "fileSha256": sha256_file(path),
        "canonicalSha256": canonical_json_sha256(value),
        "identity": identity,
        "counts": dict(sorted(counts.items())),
    }


def _physical_memory_bytes() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memoryLoad", ctypes.c_ulong),
            ("totalPhysical", ctypes.c_ulonglong),
            ("availablePhysical", ctypes.c_ulonglong),
            ("totalPageFile", ctypes.c_ulonglong),
            ("availablePageFile", ctypes.c_ulonglong),
            ("totalVirtual", ctypes.c_ulonglong),
            ("availableVirtual", ctypes.c_ulonglong),
            ("availableExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.totalPhysical)


def _command_evidence(
    command_id: str,
    command: Sequence[str],
    *,
    cwd: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = _run_bytes(command, cwd=cwd, check=False)
        output = completed.stdout.decode("utf-8", errors="replace")
        return {
            "id": command_id,
            "available": True,
            "exitCode": completed.returncode,
            "elapsedMilliseconds": round((time.monotonic() - started) * 1000),
            "output": output.replace("\r\n", "\n")[:_COMMAND_OUTPUT_LIMIT],
            "outputTruncated": len(output) > _COMMAND_OUTPUT_LIMIT,
        }
    except BaselineCaptureError as error:
        return {
            "id": command_id,
            "available": False,
            "exitCode": None,
            "elapsedMilliseconds": round((time.monotonic() - started) * 1000),
            "error": str(error),
        }


def _version_commands(
    *,
    config: ProductionConfig,
    registry: Mapping[str, Any],
) -> list[tuple[str, tuple[str, ...]]]:
    commands: list[tuple[str, tuple[str, ...]]] = [
        ("git", ("git", "--version")),
        ("python", (sys.executable, "--version")),
        ("node", ("node", "--version")),
        ("npm", ("npm", "--version")),
        ("rustc", ("rustc", "--version")),
        ("cargo", ("cargo", "--version")),
        ("maven", ("mvn", "--version")),
        (
            "nvidia-smi",
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
        ),
        ("ffmpeg", (str(config.executables.ffmpeg), "-version")),
        ("java", (str(config.executables.java), "-version")),
    ]
    seen_engines: set[str] = set()
    for raw_model in registry.get("models", []):
        if not isinstance(raw_model, Mapping):
            continue
        runtime = raw_model.get("runtime")
        if not isinstance(runtime, Mapping):
            continue
        engine = runtime.get("engine")
        executable = runtime.get("executablePath")
        if (
            not isinstance(engine, str)
            or not isinstance(executable, str)
            or engine in seen_engines
        ):
            continue
        seen_engines.add(engine)
        if engine == "ollama":
            commands.append(("ollama", (executable, "--version")))
    unique: dict[str, tuple[str, ...]] = {}
    for command_id, command in commands:
        unique.setdefault(command_id, command)
    return list(unique.items())


def _model_evidence(
    registry_path: Path,
    *,
    verify_local: bool,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    registry = load_registry(registry_path)
    started = time.monotonic()
    selected = validate_registry(registry, verify_local=verify_local)
    elapsed_ms = round((time.monotonic() - started) * 1000)
    rows: list[dict[str, Any]] = []
    statuses: dict[str, int] = {}
    for raw_model in registry["models"]:
        model = dict(raw_model)
        usage = dict(model["usage"])
        local = dict(model["local"])
        manifest = dict(local["manifest"])
        status = str(usage["status"])
        statuses[status] = statuses.get(status, 0) + 1
        rows.append(
            {
                "id": model["id"],
                "status": status,
                "roles": list(usage["roles"]),
                "localPath": local["path"],
                "manifestSha256": manifest["sha256"],
                "manifestFileCount": manifest["fileCount"],
                "manifestTotalBytes": manifest["totalBytes"],
            }
        )
    return registry, {
        "registryPath": str(registry_path.resolve()),
        "registryFileSha256": sha256_file(registry_path),
        "registryCanonicalSha256": canonical_json_sha256(registry),
        "modelCount": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "localVerification": {
            "requested": verify_local,
            "passed": True,
            "verifiedModelIds": list(selected) if verify_local else [],
            "elapsedMilliseconds": elapsed_ms,
        },
        "models": rows,
    }


def build_report(
    *,
    repository_root: Path,
    registry_path: Path,
    production_config_path: Path,
    data_manifests: Sequence[Path],
    storage_root: Path,
    source_roots: Sequence[str],
    verify_local_models: bool,
) -> dict[str, Any]:
    repository_root = repository_root.resolve(strict=True)
    git_before = _git_capture(repository_root)
    source_snapshot = build_source_snapshot(
        repository_root, source_roots=source_roots
    )
    git_after = _git_capture(repository_root)
    if git_after != git_before:
        raise BaselineCaptureError(
            "repository changed while the source baseline was being captured"
        )

    registry, models = _model_evidence(
        registry_path, verify_local=verify_local_models
    )
    config = ProductionConfig.load(production_config_path)
    config_document = _read_json(production_config_path)
    usage = shutil.disk_usage(storage_root)
    command_evidence = [
        _command_evidence(command_id, command, cwd=repository_root)
        for command_id, command in _version_commands(
            config=config,
            registry=registry,
        )
    ]
    report: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "capturedAt": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "repository": {
            "root": str(repository_root),
            **git_before,
            "sourceSnapshot": source_snapshot,
        },
        "productionConfiguration": {
            "path": str(production_config_path.resolve()),
            "bytes": production_config_path.stat().st_size,
            "fileSha256": sha256_file(production_config_path),
            "canonicalSha256": canonical_json_sha256(config_document),
            "effectiveFingerprint": config.fingerprint(),
        },
        "environment": {
            "osName": os.name,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "pythonVersion": platform.python_version(),
            "pythonExecutable": sys.executable,
            "logicalProcessorCount": os.cpu_count(),
            "physicalMemoryBytes": _physical_memory_bytes(),
            "commands": command_evidence,
        },
        "hardwareDeclaration": registry["hardwareProfiles"],
        "storage": {
            "root": str(storage_root.resolve()),
            "totalBytes": usage.total,
            "usedBytes": usage.used,
            "freeBytes": usage.free,
        },
        "models": models,
        "dataArtifacts": [_artifact_evidence(path) for path in data_manifests],
    }
    report["canonicalSha256"] = canonical_json_sha256(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "local-model-registry.json"
    )
    parser.add_argument("--production-config", type=Path, required=True)
    parser.add_argument("--data-manifest", action="append", type=Path, default=[])
    parser.add_argument("--storage-root", type=Path, default=Path("D:/"))
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--verify-local-models", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _host_path(args.output)
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        print("baseline output already exists", file=sys.stderr)
        return 2
    try:
        report = build_report(
            repository_root=_host_path(args.repository_root),
            registry_path=_host_path(args.registry),
            production_config_path=_host_path(args.production_config),
            data_manifests=tuple(_host_path(path) for path in args.data_manifest),
            storage_root=_host_path(args.storage_root),
            source_roots=tuple(args.source_root or _DEFAULT_SOURCE_ROOTS),
            verify_local_models=args.verify_local_models,
        )
        atomic_write_json_no_replace(output, report)
        file_digest = sha256_file(output)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(f"{file_digest}  {output.name}\n")
    except (BaselineCaptureError, OSError, ValueError) as error:
        print(f"baseline capture failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "sidecar": str(sidecar.resolve()),
                "canonicalSha256": report["canonicalSha256"],
                "fileSha256": file_digest,
                "modelCount": report["models"]["modelCount"],
                "sourceFileCount": report["repository"]["sourceSnapshot"][
                    "fileCount"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
