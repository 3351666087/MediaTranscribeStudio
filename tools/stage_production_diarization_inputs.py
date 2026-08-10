#!/usr/bin/env python3
"""Hard-link frozen diarization media into a production-allowed input root."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)


_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class DiarizationInputStageError(ValueError):
    """Raised when immutable production input staging cannot be proven."""


def _relative_media_path(value: object, *, case_id: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DiarizationInputStageError(f"case {case_id} has no media path")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise DiarizationInputStageError(
            f"case {case_id} media path must be relative and traversal-free"
        )
    return path


def _case_sha256(case: Mapping[str, Any], *, case_id: str) -> str:
    value = case.get("sha256")
    media = case.get("media")
    if isinstance(media, Mapping):
        value = media.get("sha256", value)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DiarizationInputStageError(
            f"case {case_id} has no valid lowercase SHA-256"
        )
    return value


def _case_bytes(case: Mapping[str, Any]) -> int | None:
    value = case.get("bytes")
    media = case.get("media")
    if isinstance(media, Mapping):
        value = media.get("bytes", value)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DiarizationInputStageError("case media byte count is invalid")
    return value


def _load_cases(
    manifests: Sequence[Path],
) -> tuple[dict[str, tuple[Mapping[str, Any], Path]], list[dict[str, Any]]]:
    index: dict[str, tuple[Mapping[str, Any], Path]] = {}
    evidence: list[dict[str, Any]] = []
    for path in manifests:
        resolved = path.resolve(strict=True)
        document = read_json_strict(resolved)
        cases = document.get("cases")
        if not isinstance(cases, list):
            raise DiarizationInputStageError(
                f"manifest has no cases: {resolved}"
            )
        for row in cases:
            if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
                raise DiarizationInputStageError(
                    f"manifest contains an invalid case: {resolved}"
                )
            case_id = str(row["id"])
            if not _CASE_ID.fullmatch(case_id):
                raise DiarizationInputStageError(
                    f"case ID is not path-safe: {case_id!r}"
                )
            if case_id in index:
                raise DiarizationInputStageError(
                    f"case is duplicated across manifests: {case_id}"
                )
            index[case_id] = (row, resolved)
        evidence.append(
            {
                "path": str(resolved),
                "fileSha256": sha256_file(resolved),
                "canonicalSha256": canonical_json_sha256(document),
            }
        )
    return index, evidence


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _remove_created_tree(
    staging: Path,
    *,
    created_files: Sequence[Path],
) -> None:
    for path in reversed(created_files):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    if not staging.exists():
        return
    directories = sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        staging.rmdir()
    except OSError:
        pass


def stage_inputs(
    *,
    manifests: Sequence[Path],
    case_ids: Sequence[str],
    allowed_root: Path,
    staging_directory: Path,
) -> dict[str, Any]:
    if not manifests:
        raise DiarizationInputStageError("at least one manifest is required")
    if not case_ids:
        raise DiarizationInputStageError("at least one case is required")
    if len(case_ids) != len(set(case_ids)):
        raise DiarizationInputStageError("case IDs must not be repeated")
    root = allowed_root.resolve(strict=True)
    if not root.is_dir():
        raise DiarizationInputStageError("allowed root must be a directory")
    requested_staging = staging_directory.absolute()
    try:
        staging_parent = requested_staging.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DiarizationInputStageError(
            "staging parent directory must already exist"
        ) from exc
    staging = staging_parent / requested_staging.name
    if not _within(staging, root) or staging == root:
        raise DiarizationInputStageError(
            "staging directory must be a new descendant of allowed root"
        )
    if os.path.lexists(staging):
        raise DiarizationInputStageError("staging directory already exists")

    index, manifest_evidence = _load_cases(manifests)
    prepared: list[dict[str, Any]] = []
    for case_id in case_ids:
        if not _CASE_ID.fullmatch(case_id):
            raise DiarizationInputStageError(
                f"case ID is not path-safe: {case_id!r}"
            )
        indexed = index.get(case_id)
        if indexed is None:
            raise DiarizationInputStageError(f"case is not frozen: {case_id}")
        case, manifest_path = indexed
        relative = _relative_media_path(case.get("path"), case_id=case_id)
        source = (manifest_path.parent / relative).resolve(strict=True)
        if not source.is_file():
            raise DiarizationInputStageError(
                f"case source is not a regular file: {case_id}"
            )
        expected_sha256 = _case_sha256(case, case_id=case_id)
        actual_sha256 = sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise DiarizationInputStageError(
                f"case source SHA-256 mismatch: {case_id}"
            )
        expected_bytes = _case_bytes(case)
        actual_bytes = source.stat().st_size
        if expected_bytes is not None and actual_bytes != expected_bytes:
            raise DiarizationInputStageError(
                f"case source byte count mismatch: {case_id}"
            )
        prepared.append(
            {
                "case": case,
                "caseId": case_id,
                "manifestPath": manifest_path,
                "source": source,
                "sourceSha256": actual_sha256,
                "sourceBytes": actual_bytes,
                "sourceMtimeNs": source.stat().st_mtime_ns,
            }
        )

    staging.mkdir(parents=True, exist_ok=False)
    created_files: list[Path] = []
    rows: list[dict[str, Any]] = []
    try:
        media_root = staging / "media"
        media_root.mkdir()
        for item in prepared:
            source = item["source"]
            destination = media_root / f"{item['caseId']}{source.suffix.lower()}"
            os.link(source, destination)
            created_files.append(destination)
            if not os.path.samefile(source, destination):
                raise DiarizationInputStageError(
                    f"hard-link identity check failed: {item['caseId']}"
                )
            if (
                destination.stat().st_size != item["sourceBytes"]
                or sha256_file(destination) != item["sourceSha256"]
            ):
                raise DiarizationInputStageError(
                    f"staged media integrity check failed: {item['caseId']}"
                )
            if (
                source.stat().st_size != item["sourceBytes"]
                or source.stat().st_mtime_ns != item["sourceMtimeNs"]
                or sha256_file(source) != item["sourceSha256"]
            ):
                raise DiarizationInputStageError(
                    f"source changed during staging: {item['caseId']}"
                )
            case = item["case"]
            rows.append(
                {
                    "caseId": item["caseId"],
                    "evaluationSplit": case.get(
                        "evaluationSplit", "unspecified"
                    ),
                    "language": case.get("language"),
                    "expectedSpeakerCount": case.get(
                        "expectedSpeakerCount"
                    ),
                    "source": {
                        "path": str(source),
                        "bytes": item["sourceBytes"],
                        "sha256": item["sourceSha256"],
                        "manifestPath": str(item["manifestPath"]),
                    },
                    "staged": {
                        "path": str(destination.resolve(strict=True)),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                        "hardLinkVerified": True,
                    },
                }
            )
        body = {
            "schemaVersion": "1.0.0",
            "artifactType": "production-diarization-input-staging-receipt",
            "status": "completed",
            "method": "same-volume-hard-link-no-copy-v1",
            "allowedInputRoot": str(root),
            "stagingDirectory": str(staging.resolve(strict=True)),
            "manifests": manifest_evidence,
            "cases": rows,
            "sourceBytes": sum(int(row["source"]["bytes"]) for row in rows),
            "additionalMediaBytesAllocated": 0,
        }
        receipt = {**body, "canonicalSha256": canonical_json_sha256(body)}
        receipt_path = staging / "staging-receipt.v1.json"
        atomic_write_json_no_replace(receipt_path, receipt)
        created_files.append(receipt_path)
        return receipt
    except Exception:
        _remove_created_tree(staging, created_files=created_files)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--staging-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = stage_inputs(
        manifests=args.manifest,
        case_ids=args.case,
        allowed_root=args.allowed_root,
        staging_directory=args.staging_directory,
    )
    print(f"caseCount={len(receipt['cases'])}")
    print(f"sourceBytes={receipt['sourceBytes']}")
    print(f"canonicalSha256={receipt['canonicalSha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
