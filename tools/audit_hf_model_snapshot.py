"""Verify a pinned Hugging Face model snapshot and persist its hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
IGNORED_DIRECTORY_NAMES = frozenset({".cache"})


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_file(path: Path) -> tuple[str, str, int]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(usedforsecurity=False)
    git_sha1.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_sha1.update(chunk)
    return sha256.hexdigest(), git_sha1.hexdigest(), size


def _load_tree(
    *,
    repo_id: str,
    revision: str,
    tree_json: Path | None,
) -> list[Mapping[str, Any]]:
    if tree_json is not None:
        raw = json.loads(tree_json.read_text(encoding="utf-8"))
    else:
        encoded_repo = urllib.parse.quote(repo_id, safe="/")
        encoded_revision = urllib.parse.quote(revision, safe="")
        url = (
            "https://huggingface.co/api/models/"
            f"{encoded_repo}/tree/{encoded_revision}"
            "?recursive=true&expand=true"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = json.load(response)
    if not isinstance(raw, list):
        raise ValueError("Hugging Face tree response must be a list")
    rows = [row for row in raw if isinstance(row, Mapping)]
    if len(rows) != len(raw):
        raise ValueError("Hugging Face tree contains a malformed entry")
    return rows


def _local_files(local_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in local_dir.rglob("*"):
        relative = path.relative_to(local_dir)
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"snapshot contains a symbolic link: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path
    return files


def audit_snapshot(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    tree: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not repo_id.strip() or not revision.strip():
        raise ValueError("repo_id and revision must be non-empty")
    root = local_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("local_dir must be a directory")

    upstream: dict[str, Mapping[str, Any]] = {}
    for row in tree:
        if row.get("type") != "file":
            continue
        path = row.get("path")
        size = row.get("size")
        oid = row.get("oid")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(oid, str)
            or len(oid) != 40
        ):
            raise ValueError("Hugging Face tree contains invalid file metadata")
        if path in upstream:
            raise ValueError(f"Hugging Face tree repeats file path: {path}")
        upstream[path] = row

    local = _local_files(root)
    missing = sorted(set(upstream) - set(local))
    extra = sorted(set(local) - set(upstream))
    if missing or extra:
        raise ValueError(
            "snapshot file set mismatch: "
            f"missing={missing or []}, extra={extra or []}"
        )

    files: list[dict[str, Any]] = []
    total_bytes = 0
    for relative in sorted(upstream):
        row = upstream[relative]
        sha256, git_blob_sha1, actual_size = _hash_file(local[relative])
        expected_size = int(row["size"])
        if actual_size != expected_size:
            raise ValueError(
                f"snapshot size mismatch for {relative}: "
                f"expected {expected_size}, got {actual_size}"
            )
        lfs = row.get("lfs")
        if isinstance(lfs, Mapping):
            expected_sha256 = lfs.get("oid")
            lfs_size = lfs.get("size")
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or lfs_size != expected_size
            ):
                raise ValueError(
                    f"Hugging Face tree contains invalid LFS metadata: {relative}"
                )
            if sha256 != expected_sha256:
                raise ValueError(
                    f"snapshot SHA-256 mismatch for {relative}: "
                    f"expected {expected_sha256}, got {sha256}"
                )
            verification = "lfs-sha256"
        else:
            expected_git_oid = str(row["oid"])
            if git_blob_sha1 != expected_git_oid:
                raise ValueError(
                    f"snapshot Git blob mismatch for {relative}: "
                    f"expected {expected_git_oid}, got {git_blob_sha1}"
                )
            expected_sha256 = None
            verification = "git-blob-sha1"
        total_bytes += actual_size
        security = row.get("securityFileStatus")
        files.append(
            {
                "path": relative,
                "size": actual_size,
                "sha256": sha256,
                "upstreamGitOid": row["oid"],
                "upstreamLfsSha256": expected_sha256,
                "verification": verification,
                "securityStatus": (
                    security.get("status")
                    if isinstance(security, Mapping)
                    else None
                ),
            }
        )

    body: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "provider": "huggingface",
        "repoId": repo_id,
        "revision": revision,
        "localDirectory": str(root),
        "fileCount": len(files),
        "totalBytes": total_bytes,
        "files": files,
    }
    body["manifestSha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tree-json",
        type=Path,
        help="use a saved Hub tree response instead of fetching the API",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tree = _load_tree(
        repo_id=args.repo_id,
        revision=args.revision,
        tree_json=args.tree_json,
    )
    report = audit_snapshot(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        tree=tree,
    )
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "repoId": report["repoId"],
                "revision": report["revision"],
                "fileCount": report["fileCount"],
                "totalBytes": report["totalBytes"],
                "manifestSha256": report["manifestSha256"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
