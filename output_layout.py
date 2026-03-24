from __future__ import annotations

import hashlib
from pathlib import Path

RUNTIME_ARTIFACTS_DIRNAME = "_runtime_artifacts"


def stable_source_hash(source_file: Path) -> str:
    source_key = str(source_file.resolve()).encode("utf-8", errors="replace")
    return hashlib.sha1(source_key).hexdigest()[:8]


def resolve_output_subdir(
    output_root: Path,
    source_file: Path,
    input_dir: Path,
) -> Path:
    source_resolved = Path(source_file).resolve()
    input_resolved = Path(input_dir).resolve()
    try:
        rel_parent = source_resolved.relative_to(input_resolved).parent
    except ValueError:
        rel_parent = Path("_external")

    unique_leaf = f"{source_file.stem}_{stable_source_hash(source_resolved)}"
    file_subdir = Path(output_root) / rel_parent / unique_leaf
    file_subdir.mkdir(parents=True, exist_ok=True)
    return file_subdir


def resolve_output_temp_dir(
    output_root: Path,
    source_file: Path,
    input_dir: Path,
) -> Path:
    temp_dir = resolve_output_subdir(
        output_root=output_root,
        source_file=source_file,
        input_dir=input_dir,
    ) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def resolve_runtime_artifacts_root(output_root: Path) -> Path:
    root = Path(output_root) / RUNTIME_ARTIFACTS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_runtime_artifact_path(output_root: Path, *parts: str) -> Path:
    return resolve_runtime_artifacts_root(output_root).joinpath(*parts)
