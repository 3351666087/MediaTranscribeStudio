"""Fail-closed, resumable installer for the offline production model set.

The installer deliberately separates three concerns:

1. ``production-models.lock.json`` is the reviewed source-of-truth.
2. ModelScope downloads into a sibling ``.partial`` directory so the final
   configured path is never exposed until every locked file passes size and
   SHA-256 verification.
3. The final directory is published with one same-volume rename.

The Qwen repositories currently expose ``master`` rather than a version tag.
For those entries the installer compares the live ModelScope file inventory
with the locked inventory before downloading.  Any drift fails closed instead
of silently replacing a production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


LOCK_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.0.0"
DEFAULT_LOCK_PATH = Path(__file__).resolve().parents[1] / "production-models.lock.json"
MANIFEST_NAME = ".mts-model-manifest.json"
PARTIAL_SUFFIX = ".mts-download.partial"
MIB = 1024 * 1024
GIB = 1024 * MIB


class ModelInstallError(RuntimeError):
    """Raised when a model cannot be installed without weakening guarantees."""


@dataclass(frozen=True, slots=True)
class LockedFile:
    path: str
    size: int
    sha256: str

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "LockedFile":
        path = value.get("path")
        size = value.get("size")
        sha256 = value.get("sha256")
        if not isinstance(path, str) or not path:
            raise ModelInstallError("locked file path must be a non-empty string")
        normalized = PurePosixPath(path)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ModelInstallError(f"unsafe locked file path: {path!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ModelInstallError(f"invalid locked size for {path!r}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ModelInstallError(f"invalid locked sha256 for {path!r}")
        return cls(path=normalized.as_posix(), size=size, sha256=sha256)


@dataclass(frozen=True, slots=True)
class LockedModel:
    key: str
    repo_id: str
    revision: str
    target: Path
    total_bytes: int
    files: tuple[LockedFile, ...]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "LockedModel":
        key = value.get("key")
        repo_id = value.get("repoId")
        revision = value.get("revision")
        target = value.get("target")
        total_bytes = value.get("totalBytes")
        raw_files = value.get("files")
        for name, candidate in (
            ("key", key),
            ("repoId", repo_id),
            ("revision", revision),
            ("target", target),
        ):
            if not isinstance(candidate, str) or not candidate:
                raise ModelInstallError(f"model {name} must be a non-empty string")
        if not isinstance(raw_files, list) or not raw_files:
            raise ModelInstallError(f"model {key!r} has no locked files")
        files = tuple(LockedFile.from_json(item) for item in raw_files)
        if len({item.path.casefold() for item in files}) != len(files):
            raise ModelInstallError(f"model {key!r} contains duplicate file paths")
        computed_total = sum(item.size for item in files)
        if (
            not isinstance(total_bytes, int)
            or isinstance(total_bytes, bool)
            or total_bytes != computed_total
        ):
            raise ModelInstallError(
                f"model {key!r} totalBytes={total_bytes!r}, expected {computed_total}"
            )
        return cls(
            key=key,
            repo_id=repo_id,
            revision=revision,
            target=Path(target),
            total_bytes=total_bytes,
            files=files,
        )


@dataclass(frozen=True, slots=True)
class ModelLock:
    path: Path
    sha256: str
    models: tuple[LockedModel, ...]

    def select(self, keys: Sequence[str] | None) -> tuple[LockedModel, ...]:
        if not keys:
            return self.models
        requested = tuple(dict.fromkeys(keys))
        by_key = {model.key: model for model in self.models}
        missing = [key for key in requested if key not in by_key]
        if missing:
            raise ModelInstallError(
                "unknown model key(s): "
                + ", ".join(missing)
                + "; available: "
                + ", ".join(sorted(by_key))
            )
        return tuple(by_key[key] for key in requested)


@dataclass(frozen=True, slots=True)
class VerificationFailure:
    path: str
    reason: str


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, *, chunk_size: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path = DEFAULT_LOCK_PATH) -> ModelLock:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelInstallError(f"invalid UTF-8 JSON lock file: {resolved}") from error
    if not isinstance(document, dict):
        raise ModelInstallError("model lock root must be an object")
    if document.get("schemaVersion") != LOCK_SCHEMA_VERSION:
        raise ModelInstallError(
            f"unsupported model lock schemaVersion={document.get('schemaVersion')!r}"
        )
    if document.get("provider") != "modelscope":
        raise ModelInstallError("only the ModelScope provider is supported")
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ModelInstallError("model lock must contain at least one model")
    models = tuple(LockedModel.from_json(item) for item in raw_models)
    if len({model.key for model in models}) != len(models):
        raise ModelInstallError("model lock contains duplicate model keys")
    return ModelLock(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        models=models,
    )


def resolve_target(model: LockedModel, target_root: Path | None) -> Path:
    if target_root is None:
        return model.target.resolve()
    root = target_root.resolve()
    return (root / model.key).resolve()


def verify_model(
    model: LockedModel,
    directory: Path,
    *,
    workers: int,
) -> tuple[VerificationFailure, ...]:
    root = directory.resolve()

    def verify_one(locked: LockedFile) -> VerificationFailure | None:
        candidate = (root / Path(*PurePosixPath(locked.path).parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return VerificationFailure(locked.path, "resolved outside model directory")
        if not candidate.is_file():
            return VerificationFailure(locked.path, "missing")
        actual_size = candidate.stat().st_size
        if actual_size != locked.size:
            return VerificationFailure(
                locked.path,
                f"size mismatch: expected {locked.size}, got {actual_size}",
            )
        actual_hash = _sha256_file(candidate)
        if actual_hash != locked.sha256:
            return VerificationFailure(
                locked.path,
                f"sha256 mismatch: expected {locked.sha256}, got {actual_hash}",
            )
        return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        failures = tuple(
            failure
            for failure in executor.map(verify_one, model.files)
            if failure is not None
        )
    return failures


def _locked_inventory(model: LockedModel) -> dict[str, tuple[int, str]]:
    return {item.path: (item.size, item.sha256) for item in model.files}


def _live_inventory(model: LockedModel, api: Any) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for entry in api.get_model_files(
        model.repo_id,
        revision=model.revision,
        recursive=True,
    ):
        if entry.get("Type") != "blob":
            continue
        path = entry.get("Path")
        size = entry.get("Size")
        sha256 = entry.get("Sha256")
        if not isinstance(path, str) or not isinstance(size, int) or not isinstance(
            sha256, str
        ):
            raise ModelInstallError(
                f"ModelScope returned incomplete inventory for {model.repo_id}"
            )
        result[PurePosixPath(path).as_posix()] = (size, sha256)
    return result


def assert_source_matches_lock(model: LockedModel, api: Any) -> None:
    locked = _locked_inventory(model)
    live = _live_inventory(model, api)
    if locked == live:
        return
    missing = sorted(set(locked) - set(live))
    added = sorted(set(live) - set(locked))
    changed = sorted(
        path for path in set(locked) & set(live) if locked[path] != live[path]
    )
    raise ModelInstallError(
        f"source drift for {model.repo_id}@{model.revision}; "
        f"missing={missing}, added={added}, changed={changed}. "
        "Review the upstream changes and regenerate the lock deliberately."
    )


def required_download_bytes(
    selections: Iterable[tuple[LockedModel, Path]],
    *,
    workers: int,
) -> int:
    required = 0
    for model, target in selections:
        if target.is_dir() and not verify_model(model, target, workers=workers):
            continue
        partial = target.with_name(target.name + PARTIAL_SUFFIX)
        present = 0
        if partial.is_dir():
            for locked in model.files:
                candidate = partial / Path(*PurePosixPath(locked.path).parts)
                if candidate.is_file() and candidate.stat().st_size == locked.size:
                    present += locked.size
        required += max(0, model.total_bytes - present)
    return required


def assert_disk_capacity(
    selections: Iterable[tuple[LockedModel, Path]],
    *,
    workers: int,
    reserve_bytes: int,
) -> None:
    grouped: dict[str, list[tuple[LockedModel, Path]]] = {}
    for selection in selections:
        _, target = selection
        anchor = target
        while not anchor.exists():
            if anchor.parent == anchor:
                break
            anchor = anchor.parent
        grouped.setdefault(str(anchor.resolve()), []).append(selection)
    for anchor_string, items in grouped.items():
        anchor = Path(anchor_string)
        required = required_download_bytes(items, workers=workers)
        free = shutil.disk_usage(anchor).free
        if free < required + reserve_bytes:
            raise ModelInstallError(
                f"insufficient free space on {anchor.anchor or anchor}: "
                f"required download={required / GIB:.2f} GiB, "
                f"reserve={reserve_bytes / GIB:.2f} GiB, "
                f"free={free / GIB:.2f} GiB"
            )


def build_manifest(model: LockedModel, lock_sha256: str) -> Mapping[str, Any]:
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "provider": "modelscope",
        "modelKey": model.key,
        "repoId": model.repo_id,
        "revision": model.revision,
        "lockSha256": lock_sha256,
        "totalBytes": model.total_bytes,
        "files": [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in model.files
        ],
    }


def _write_manifest(
    model: LockedModel,
    directory: Path,
    lock_sha256: str,
) -> None:
    destination = directory / MANIFEST_NAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(build_manifest(model, lock_sha256)))
    os.replace(temporary, destination)


def install_model(
    model: LockedModel,
    target: Path,
    *,
    lock_sha256: str,
    workers: int,
    api: Any,
    downloader: Callable[..., str],
    verify_source: bool,
) -> str:
    if target.exists():
        if not target.is_dir():
            raise ModelInstallError(f"model target is not a directory: {target}")
        failures = verify_model(model, target, workers=workers)
        if failures:
            preview = "; ".join(
                f"{failure.path}: {failure.reason}" for failure in failures[:8]
            )
            raise ModelInstallError(
                f"existing target for {model.key!r} is invalid; "
                f"refusing destructive replacement: {preview}"
            )
        _write_manifest(model, target, lock_sha256)
        return "verified"

    if verify_source:
        assert_source_matches_lock(model, api)

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    partial.mkdir(parents=True, exist_ok=True)
    downloader(
        model.repo_id,
        revision=model.revision,
        local_dir=str(partial),
        max_workers=max(1, workers),
    )
    failures = verify_model(model, partial, workers=workers)
    if failures:
        preview = "; ".join(
            f"{failure.path}: {failure.reason}" for failure in failures[:8]
        )
        raise ModelInstallError(
            f"download verification failed for {model.key!r}; "
            f"partial files retained for resume: {preview}"
        )
    _write_manifest(model, partial, lock_sha256)
    try:
        partial.rename(target)
    except OSError as error:
        raise ModelInstallError(
            f"verified model could not be published atomically: {partial} -> {target}"
        ) from error
    return "installed"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="model lock JSON (default: repository production-models.lock.json)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="install/verify one model key; repeat to select multiple (default: all)",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        help="testing/portable override; models are placed under <root>/<model-key>",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform no network calls and require every selected target to validate",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="skip live inventory comparison; intended only for a trusted mirror outage",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel ModelScope downloads and local hash checks (default: 4)",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=3.0,
        help="free-space reserve that must remain during installation (default: 3)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.workers < 1:
        raise ModelInstallError("--workers must be >= 1")
    if arguments.reserve_gib < 0:
        raise ModelInstallError("--reserve-gib must be >= 0")

    lock = load_lock(arguments.lock)
    selected = lock.select(arguments.models)
    selections = tuple(
        (model, resolve_target(model, arguments.target_root)) for model in selected
    )

    if arguments.verify_only:
        failed = False
        for model, target in selections:
            failures = (
                verify_model(model, target, workers=arguments.workers)
                if target.is_dir()
                else (VerificationFailure(".", "target directory is missing"),)
            )
            if failures:
                failed = True
                print(f"[FAIL] {model.key}: {target}", file=sys.stderr)
                for failure in failures[:20]:
                    print(
                        f"  - {failure.path}: {failure.reason}",
                        file=sys.stderr,
                    )
            else:
                print(f"[OK] {model.key}: {target}")
        return 1 if failed else 0

    assert_disk_capacity(
        selections,
        workers=arguments.workers,
        reserve_bytes=int(arguments.reserve_gib * GIB),
    )
    try:
        from modelscope import snapshot_download
        from modelscope.hub.api import HubApi
    except ImportError as error:
        raise ModelInstallError(
            "ModelScope is required; run this command inside the media-asr environment"
        ) from error

    api = HubApi()
    for model, target in selections:
        print(
            f"[START] {model.key}: {model.repo_id}@{model.revision} "
            f"-> {target} ({model.total_bytes / GIB:.2f} GiB)"
        )
        outcome = install_model(
            model,
            target,
            lock_sha256=lock.sha256,
            workers=arguments.workers,
            api=api,
            downloader=snapshot_download,
            verify_source=not arguments.skip_source_check,
        )
        print(f"[{outcome.upper()}] {model.key}: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModelInstallError as error:
        print(f"model installation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
