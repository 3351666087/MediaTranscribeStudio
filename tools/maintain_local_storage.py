"""Audit and apply the repository's bounded local storage lifecycle policy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "model-lifecycle.v1.json"
POLICY_SCHEMA_VERSION = "1.0.0"
MINIMUM_ALLOWED_FREE_BYTES = 8 * 1024**3
ALLOWED_MODEL_STATUSES = frozenset({"active", "retired"})
ALLOWED_SCRATCH_ROOT = PurePosixPath(".runtime_cache")
ALLOWED_LOCAL_ARTIFACT_ROOTS = frozenset({"project", "applicationSupport"})
ALLOWED_PROJECT_ARTIFACT_PREFIXES = frozenset(
    {
        (".runtime_cache", "production-models"),
        (".runtime_cache", "venvs"),
    }
)
ALLOWED_APPLICATION_SUPPORT_ARTIFACT_PREFIXES = frozenset(
    {"production-models", "venvs"}
)


class StoragePolicyError(RuntimeError):
    """Raised when cleanup cannot proceed without weakening its safety rules."""


@dataclass(frozen=True, slots=True)
class OllamaModelPolicy:
    name: str
    status: str


@dataclass(frozen=True, slots=True)
class ScratchPolicy:
    path: Path
    max_age_days: int


@dataclass(frozen=True, slots=True)
class ProductionConfigPolicy:
    path: Path
    required: bool


@dataclass(frozen=True, slots=True)
class LocalArtifactPolicy:
    artifact_id: str
    root_name: str
    root: Path
    path: Path
    status: str


@dataclass(frozen=True, slots=True)
class StoragePolicy:
    path: Path
    project_root: Path
    minimum_free_bytes: int
    production_configs: tuple[ProductionConfigPolicy, ...]
    ollama_models: tuple[OllamaModelPolicy, ...]
    local_artifacts: tuple[LocalArtifactPolicy, ...]
    scratch_policies: tuple[ScratchPolicy, ...]


@dataclass(frozen=True, slots=True)
class ScratchCandidate:
    path: Path
    allocated_bytes: int
    newest_mtime_ns: int


def _safe_project_relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise StoragePolicyError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() == ".":
        raise StoragePolicyError(f"{field} is not a safe project-relative path")
    return path


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StoragePolicyError(f"{field} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StoragePolicyError(f"{field} root must be an object: {path}")
    return value


def load_policy(
    path: Path = DEFAULT_POLICY_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    application_support_root: Path | None = None,
) -> StoragePolicy:
    root = project_root.resolve(strict=True)
    app_support = (
        application_support_root
        if application_support_root is not None
        else (Path.home() / "Library" / "Application Support" / "MediaTranscribeStudio")
    ).absolute()
    resolved = path.resolve(strict=True)
    value = _load_json_object(resolved, "storage policy")
    if value.get("schemaVersion") != POLICY_SCHEMA_VERSION:
        raise StoragePolicyError(
            f"unsupported storage policy schemaVersion={value.get('schemaVersion')!r}"
        )

    minimum_free_bytes = value.get("minimumFreeBytes")
    if (
        isinstance(minimum_free_bytes, bool)
        or not isinstance(minimum_free_bytes, int)
        or minimum_free_bytes < MINIMUM_ALLOWED_FREE_BYTES
    ):
        raise StoragePolicyError(
            "minimumFreeBytes must preserve at least 8 GiB of workspace capacity"
        )

    raw_configs = value.get("productionConfigPaths")
    if not isinstance(raw_configs, list) or not raw_configs:
        raise StoragePolicyError("productionConfigPaths must be a non-empty array")
    config_policies: list[ProductionConfigPolicy] = []
    for index, item in enumerate(raw_configs):
        if not isinstance(item, Mapping) or not isinstance(
            item.get("required"),
            bool,
        ):
            raise StoragePolicyError(
                f"productionConfigPaths[{index}] must contain path and required"
            )
        relative = _safe_project_relative_path(
            item.get("path"),
            f"productionConfigPaths[{index}].path",
        )
        config_policies.append(
            ProductionConfigPolicy(
                path=root / Path(*relative.parts),
                required=bool(item["required"]),
            )
        )
    if len({item.path for item in config_policies}) != len(config_policies):
        raise StoragePolicyError("productionConfigPaths contains duplicates")
    if not any(item.required for item in config_policies):
        raise StoragePolicyError(
            "productionConfigPaths must contain at least one required config"
        )

    raw_models = value.get("ollamaModels")
    if not isinstance(raw_models, list) or not raw_models:
        raise StoragePolicyError("ollamaModels must be a non-empty array")
    models: list[OllamaModelPolicy] = []
    for index, item in enumerate(raw_models):
        if not isinstance(item, Mapping):
            raise StoragePolicyError(f"ollamaModels[{index}] must be an object")
        name = item.get("name")
        status = item.get("status")
        if not isinstance(name, str) or not name.strip():
            raise StoragePolicyError(
                f"ollamaModels[{index}].name must be non-empty text"
            )
        if status not in ALLOWED_MODEL_STATUSES:
            raise StoragePolicyError(
                f"ollamaModels[{index}].status must be active or retired"
            )
        models.append(OllamaModelPolicy(name=name.strip(), status=str(status)))
    if len({item.name for item in models}) != len(models):
        raise StoragePolicyError("ollamaModels contains duplicate names")
    if not any(item.status == "active" for item in models):
        raise StoragePolicyError("ollamaModels must retain at least one active model")

    raw_artifacts = value.get("localArtifacts", [])
    if not isinstance(raw_artifacts, list):
        raise StoragePolicyError("localArtifacts must be an array")
    local_artifacts: list[LocalArtifactPolicy] = []
    for index, item in enumerate(raw_artifacts):
        if not isinstance(item, Mapping):
            raise StoragePolicyError(f"localArtifacts[{index}] must be an object")
        artifact_id = item.get("id")
        root_name = item.get("root")
        status = item.get("status")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise StoragePolicyError(
                f"localArtifacts[{index}].id must be non-empty text"
            )
        if root_name not in ALLOWED_LOCAL_ARTIFACT_ROOTS:
            raise StoragePolicyError(f"localArtifacts[{index}].root is unsupported")
        if status not in ALLOWED_MODEL_STATUSES:
            raise StoragePolicyError(
                f"localArtifacts[{index}].status must be active or retired"
            )
        relative = _safe_project_relative_path(
            item.get("path"),
            f"localArtifacts[{index}].path",
        )
        if root_name == "project":
            if (
                len(relative.parts) < 3
                or tuple(relative.parts[:2]) not in ALLOWED_PROJECT_ARTIFACT_PREFIXES
            ):
                raise StoragePolicyError(
                    "project local artifacts must be specific children of "
                    ".runtime_cache/production-models or .runtime_cache/venvs"
                )
            artifact_root = root
        else:
            if (
                len(relative.parts) < 2
                or relative.parts[0]
                not in ALLOWED_APPLICATION_SUPPORT_ARTIFACT_PREFIXES
            ):
                raise StoragePolicyError(
                    "applicationSupport local artifacts must be specific "
                    "children of production-models or venvs"
                )
            artifact_root = app_support
        local_artifacts.append(
            LocalArtifactPolicy(
                artifact_id=artifact_id.strip(),
                root_name=str(root_name),
                root=artifact_root,
                path=artifact_root / Path(*relative.parts),
                status=str(status),
            )
        )
    if len({item.artifact_id for item in local_artifacts}) != len(local_artifacts):
        raise StoragePolicyError("localArtifacts contains duplicate IDs")
    if len({item.path for item in local_artifacts}) != len(local_artifacts):
        raise StoragePolicyError("localArtifacts contains duplicate paths")

    raw_scratch = value.get("scratchPolicies")
    if not isinstance(raw_scratch, list):
        raise StoragePolicyError("scratchPolicies must be an array")
    scratch_policies: list[ScratchPolicy] = []
    for index, item in enumerate(raw_scratch):
        if not isinstance(item, Mapping):
            raise StoragePolicyError(f"scratchPolicies[{index}] must be an object")
        relative = _safe_project_relative_path(
            item.get("path"),
            f"scratchPolicies[{index}].path",
        )
        if (
            len(relative.parts) < 2
            or PurePosixPath(relative.parts[0]) != ALLOWED_SCRATCH_ROOT
        ):
            raise StoragePolicyError(
                "scratch policy paths must be children of .runtime_cache"
            )
        max_age_days = item.get("maxAgeDays")
        if (
            isinstance(max_age_days, bool)
            or not isinstance(max_age_days, int)
            or max_age_days < 1
        ):
            raise StoragePolicyError(
                f"scratchPolicies[{index}].maxAgeDays must be a positive integer"
            )
        scratch_policies.append(
            ScratchPolicy(
                path=root / Path(*relative.parts),
                max_age_days=max_age_days,
            )
        )
    if len({item.path for item in scratch_policies}) != len(scratch_policies):
        raise StoragePolicyError("scratchPolicies contains duplicate paths")

    return StoragePolicy(
        path=resolved,
        project_root=root,
        minimum_free_bytes=minimum_free_bytes,
        production_configs=tuple(config_policies),
        ollama_models=tuple(models),
        local_artifacts=tuple(local_artifacts),
        scratch_policies=tuple(scratch_policies),
    )


def production_ollama_references(
    config_policies: Sequence[ProductionConfigPolicy],
) -> tuple[set[str], tuple[Path, ...]]:
    references: set[str] = set()
    inspected: list[Path] = []
    for policy in config_policies:
        path = policy.path
        if not path.is_file():
            if policy.required:
                raise StoragePolicyError(
                    f"required production config is missing: {path}"
                )
            continue
        value = _load_json_object(path, "production config")
        inspected.append(path)
        speaker = value.get("speaker")
        if not isinstance(speaker, Mapping):
            raise StoragePolicyError(f"production config has no speaker object: {path}")
        model = speaker.get("localLlmModel")
        if not isinstance(model, str) or not model.strip():
            raise StoragePolicyError(f"production config has no localLlmModel: {path}")
        references.add(model.strip())
    if not inspected:
        raise StoragePolicyError("no production config was available for audit")
    return references, tuple(inspected)


def parse_ollama_list(output: str) -> tuple[str, ...]:
    names: list[str] = []
    for line in output.splitlines():
        columns = line.split()
        if not columns:
            continue
        if columns[0].casefold() == "name":
            continue
        names.append(columns[0])
    return tuple(dict.fromkeys(names))


def list_ollama_models(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, ...]:
    try:
        result = runner(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise StoragePolicyError("ollama executable is unavailable") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[:240]
        raise StoragePolicyError(
            f"ollama list failed with exit code {result.returncode}: {detail}"
        )
    return parse_ollama_list(result.stdout)


def _allocated_bytes(path: Path) -> int:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return 0
    if os.name == "nt":
        own_bytes = max(0, int(getattr(stat, "st_size", 0)))
    else:
        own_bytes = max(0, int(getattr(stat, "st_blocks", 0))) * 512
    if path.is_symlink() or not path.is_dir():
        return own_bytes
    total = own_bytes
    with os.scandir(path) as entries:
        for entry in entries:
            total += _allocated_bytes(Path(entry.path))
    return total


def _newest_mtime_ns(path: Path) -> int:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return 0
    newest = stat.st_mtime_ns
    if path.is_symlink() or not path.is_dir():
        return newest
    with os.scandir(path) as entries:
        for entry in entries:
            newest = max(newest, _newest_mtime_ns(Path(entry.path)))
    return newest


def scratch_candidates(
    policies: Sequence[ScratchPolicy],
    *,
    now_ns: int,
) -> tuple[ScratchCandidate, ...]:
    candidates: list[ScratchCandidate] = []
    for policy in policies:
        if not policy.path.exists():
            continue
        cutoff_ns = now_ns - policy.max_age_days * 86_400 * 1_000_000_000
        for candidate in sorted(policy.path.iterdir(), key=lambda item: item.name):
            newest = _newest_mtime_ns(candidate)
            if newest and newest < cutoff_ns:
                candidates.append(
                    ScratchCandidate(
                        path=candidate,
                        allocated_bytes=_allocated_bytes(candidate),
                        newest_mtime_ns=newest,
                    )
                )
    return tuple(candidates)


def _remove_scratch_candidate(
    candidate: ScratchCandidate,
    policies: Sequence[ScratchPolicy],
) -> None:
    allowed_parents = {policy.path.resolve(strict=True) for policy in policies}
    parent = candidate.path.parent.resolve(strict=True)
    if parent not in allowed_parents:
        raise StoragePolicyError(
            f"scratch candidate escaped its registered root: {candidate.path}"
        )
    if candidate.path.is_symlink() or candidate.path.is_file():
        candidate.path.unlink(missing_ok=True)
    elif candidate.path.is_dir():
        shutil.rmtree(candidate.path)
    else:
        raise StoragePolicyError(
            f"scratch candidate has an unsupported file type: {candidate.path}"
        )


def _remove_ollama_model(
    name: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    try:
        result = runner(
            ["ollama", "rm", name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise StoragePolicyError("ollama executable is unavailable") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[:240]
        raise StoragePolicyError(
            f"ollama rm failed for {name!r} with exit code "
            f"{result.returncode}: {detail}"
        )


def _local_artifact_exists(policy: LocalArtifactPolicy) -> bool:
    return policy.path.exists() or policy.path.is_symlink()


def _remove_local_artifact(policy: LocalArtifactPolicy) -> None:
    path = policy.path
    try:
        path.relative_to(policy.root)
    except ValueError as exc:
        raise StoragePolicyError(
            f"local artifact escaped its registered root: {path}"
        ) from exc
    if path == policy.root:
        raise StoragePolicyError("local artifact cannot equal its root")
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    if not path.is_dir():
        raise StoragePolicyError(
            f"local artifact must be a directory or symlink: {path}"
        )
    resolved_root = policy.root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise StoragePolicyError(
            f"local artifact resolved outside its registered root: {path}"
        ) from exc
    shutil.rmtree(path)


def _local_artifact_report(
    policies: Sequence[LocalArtifactPolicy],
) -> list[dict[str, Any]]:
    return [
        {
            "id": item.artifact_id,
            "root": item.root_name,
            "path": str(item.path),
            "status": item.status,
            "allocatedBytes": _allocated_bytes(item.path),
        }
        for item in policies
        if _local_artifact_exists(item)
    ]


def maintain_storage(
    policy: StoragePolicy,
    *,
    apply: bool,
    now_ns: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    references, inspected_configs = production_ollama_references(
        policy.production_configs
    )
    by_name = {item.name: item for item in policy.ollama_models}
    unregistered_references = sorted(references - by_name.keys())
    retired_references = sorted(
        name
        for name in references
        if name in by_name and by_name[name].status == "retired"
    )
    if unregistered_references or retired_references:
        raise StoragePolicyError(
            "production model references are not active in the lifecycle policy: "
            + ", ".join([*unregistered_references, *retired_references])
        )

    installed = set(list_ollama_models(runner))
    active = {item.name for item in policy.ollama_models if item.status == "active"}
    retired = {item.name for item in policy.ollama_models if item.status == "retired"}
    retired_installed = sorted(installed & retired)
    missing_active = sorted((active & references) - installed)
    local_installed_before = [
        item for item in policy.local_artifacts if _local_artifact_exists(item)
    ]
    retired_local_before = [
        item for item in local_installed_before if item.status == "retired"
    ]
    retired_local_reclaim_bytes = sum(
        _allocated_bytes(item.path) for item in retired_local_before
    )
    missing_active_local = sorted(
        item.artifact_id
        for item in policy.local_artifacts
        if item.status == "active" and not _local_artifact_exists(item)
    )
    scratch = scratch_candidates(
        policy.scratch_policies,
        now_ns=time.time_ns() if now_ns is None else now_ns,
    )
    disk_before = shutil.disk_usage(policy.project_root)

    removed_models: list[str] = []
    removed_local_artifacts: list[str] = []
    removed_scratch: list[str] = []
    if apply:
        for name in retired_installed:
            _remove_ollama_model(name, runner)
            removed_models.append(name)
        for artifact in retired_local_before:
            _remove_local_artifact(artifact)
            removed_local_artifacts.append(artifact.artifact_id)
        for candidate in scratch:
            _remove_scratch_candidate(candidate, policy.scratch_policies)
            removed_scratch.append(str(candidate.path))

    installed_after = installed - set(removed_models)
    managed_names = set(by_name)
    retired_installed_after = sorted(installed_after & retired)
    local_installed_after = [
        item for item in policy.local_artifacts if _local_artifact_exists(item)
    ]
    retired_local_after = [
        item for item in local_installed_after if item.status == "retired"
    ]
    remaining_scratch = (
        scratch_candidates(
            policy.scratch_policies,
            now_ns=time.time_ns() if now_ns is None else now_ns,
        )
        if apply
        else scratch
    )
    disk_after = shutil.disk_usage(policy.project_root)
    return {
        "schemaVersion": "1.0.0",
        "mode": "apply" if apply else "audit",
        "policyPath": str(policy.path),
        "minimumFreeBytes": policy.minimum_free_bytes,
        "freeBytesBefore": disk_before.free,
        "freeBytesAfter": disk_after.free,
        "freeSpaceThresholdMet": disk_after.free >= policy.minimum_free_bytes,
        "inspectedProductionConfigs": [str(path) for path in inspected_configs],
        "productionOllamaReferences": sorted(references),
        "installedOllamaModelsBefore": sorted(installed),
        "installedOllamaModels": sorted(installed_after),
        "unmanagedInstalledOllamaModels": sorted(installed_after - managed_names),
        "retiredInstalledModelsBefore": retired_installed,
        "retiredInstalledModels": retired_installed_after,
        "missingActiveModels": missing_active,
        "installedLocalArtifactsBefore": _local_artifact_report(local_installed_before),
        "installedLocalArtifacts": _local_artifact_report(local_installed_after),
        "retiredLocalArtifactsBefore": [
            item.artifact_id for item in retired_local_before
        ],
        "retiredLocalArtifacts": [item.artifact_id for item in retired_local_after],
        "missingActiveLocalArtifacts": missing_active_local,
        "scratchCandidatesBefore": [
            {
                "path": str(item.path),
                "allocatedBytes": item.allocated_bytes,
                "newestMtimeNs": item.newest_mtime_ns,
            }
            for item in scratch
        ],
        "scratchCandidates": [
            {
                "path": str(item.path),
                "allocatedBytes": item.allocated_bytes,
                "newestMtimeNs": item.newest_mtime_ns,
            }
            for item in remaining_scratch
        ],
        "plannedLocalArtifactReclaimBytes": retired_local_reclaim_bytes,
        "plannedReclaimBytes": (
            retired_local_reclaim_bytes + sum(item.allocated_bytes for item in scratch)
        ),
        "removedOllamaModels": removed_models,
        "removedLocalArtifacts": removed_local_artifacts,
        "removedScratchPaths": removed_scratch,
        "actionRequired": bool(
            retired_installed_after
            or retired_local_after
            or remaining_scratch
            or missing_active
            or missing_active_local
            or disk_after.free < policy.minimum_free_bytes
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "remove only policy-retired Ollama/local artifacts and expired "
            "scratch entries"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optionally write the JSON audit report to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = maintain_storage(
            load_policy(args.policy),
            apply=args.apply,
        )
    except StoragePolicyError as exc:
        print(f"storage maintenance failed closed: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    print(payload)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
