"""Crash-safe JSON persistence and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import WorkerError


@dataclass(frozen=True)
class PublishedJsonEvidence:
    """Exact integrity evidence for an immutably published JSON artifact."""

    path: Path
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sizeBytes": self.size_bytes,
            "sha256": self.sha256,
        }


def validate_strict_json(value: Any, *, path: str = "$") -> None:
    """Reject values that JSON would silently coerce or serialize unsafely.

    Python's default encoder accepts tuples as arrays, non-finite floats as
    ``NaN``/``Infinity``, and some non-string dictionary keys.  Pipeline
    artifacts are durable contracts, so those implicit conversions are
    forbidden and every value must already be an exact JSON data type.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_strict_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            validate_strict_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, tuple):
        raise ValueError(f"{path} contains a tuple; use an explicit JSON array")
    raise ValueError(
        f"{path} contains unsupported JSON value type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    validate_strict_json(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    validate_strict_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json_no_replace(path: Path, value: Any) -> None:
    """Atomically publish a new JSON artifact without replacing any path.

    Durable evidence and user-facing outputs must never inherit checkpoint
    semantics.  Checkpoints intentionally replace their previous version;
    immutable evidence artifacts instead use an exclusive hard-link publish
    in the destination directory so a concurrently created file fails closed.
    """

    validate_strict_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if not published and path.exists():
            # The destination belongs to another publisher.  Never remove it.
            pass


def atomic_publish_json_evidence(
    path: Path,
    value: Any,
) -> PublishedJsonEvidence:
    """Publish JSON once and return a hash of the exact persisted bytes."""

    atomic_write_json_no_replace(path, value)
    canonical = path.resolve(strict=True)
    if not canonical.is_file():
        raise WorkerError(
            "ARTIFACT_INTEGRITY_FAILED",
            "published JSON evidence is not a regular file",
            details={"path": str(path)},
        )
    size_bytes = canonical.stat().st_size
    digest = sha256_file(canonical)
    if size_bytes <= 0 or len(digest) != 64:
        raise WorkerError(
            "ARTIFACT_INTEGRITY_FAILED",
            "published JSON evidence could not be verified",
            details={"path": str(canonical)},
        )
    return PublishedJsonEvidence(
        path=canonical,
        size_bytes=size_bytes,
        sha256=digest,
    )


def read_json_strict(path: Path) -> dict[str, Any]:
    """Read a JSON object without accepting duplicate keys or non-finite values."""

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
        validate_strict_json(value)
    except FileNotFoundError as exc:
        raise WorkerError(
            "PERSISTED_STATE_MISSING",
            "required persisted JSON state is missing",
            details={"path": str(path)},
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkerError(
            "PERSISTED_STATE_INVALID",
            "persisted JSON state is unreadable or invalid",
            details={"path": str(path), "exceptionType": type(exc).__name__},
        ) from exc
    if not isinstance(value, dict):
        raise WorkerError(
            "PERSISTED_STATE_INVALID",
            "persisted JSON root must be an object",
            details={"path": str(path)},
        )
    return value


def _transaction_root(journal_path: Path) -> Path:
    root = journal_path.parent.absolute().resolve(strict=False)
    if journal_path.absolute().resolve(strict=False).parent != root:
        raise WorkerError(
            "PERSISTENCE_TRANSACTION_INVALID",
            "transaction journal must be directly inside its transaction root",
        )
    return root


def _relative_transaction_path(path: Path, root: Path) -> str:
    candidate = path.absolute().resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise WorkerError(
            "PERSISTENCE_TRANSACTION_INVALID",
            "transaction paths must remain inside the output directory",
            details={"path": str(path), "root": str(root)},
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise WorkerError(
            "PERSISTENCE_TRANSACTION_INVALID",
            "transaction path is invalid",
            details={"path": str(path)},
        )
    return relative.as_posix()


def _journal_member(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise WorkerError(
            "PERSISTENCE_JOURNAL_INVALID",
            "transaction journal contains an invalid relative path",
        )
    candidate = (root / Path(value)).absolute().resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkerError(
            "PERSISTENCE_JOURNAL_INVALID",
            "transaction journal path escapes the output directory",
        ) from exc
    return candidate


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _replace_and_sync(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    if os.name != "nt":
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _transaction_entries(
    journal: Mapping[str, Any],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    if journal.get("schemaVersion") != "1.0.0":
        raise WorkerError(
            "PERSISTENCE_JOURNAL_INVALID",
            "transaction journal schemaVersion is unsupported",
        )
    raw_entries = journal.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise WorkerError(
            "PERSISTENCE_JOURNAL_INVALID",
            "transaction journal must contain entries",
        )
    entries: list[dict[str, Any]] = []
    targets: set[Path] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise WorkerError(
                "PERSISTENCE_JOURNAL_INVALID",
                "transaction journal entries must be objects",
            )
        target = _journal_member(root, raw.get("target"))
        staged = _journal_member(root, raw.get("staged"))
        backup = _journal_member(root, raw.get("backup"))
        expected = raw.get("sha256")
        had_original = raw.get("hadOriginal")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or not isinstance(had_original, bool)
        ):
            raise WorkerError(
                "PERSISTENCE_JOURNAL_INVALID",
                "transaction journal entry metadata is invalid",
            )
        if target in targets:
            raise WorkerError(
                "PERSISTENCE_JOURNAL_INVALID",
                "transaction journal contains duplicate targets",
            )
        targets.add(target)
        entries.append(
            {
                "target": target,
                "staged": staged,
                "backup": backup,
                "sha256": expected,
                "hadOriginal": had_original,
            }
        )
    return entries


def recover_json_transaction(journal_path: Path) -> bool:
    """Recover an interrupted journaled JSON transaction.

    Recovery first checks whether every target can be rolled forward from an
    already-installed target or its staged payload.  If not, it restores every
    available backup and removes targets that did not exist before the
    transaction.  The journal is deleted only after a complete recovery.
    """

    if not journal_path.exists():
        return False
    root = _transaction_root(journal_path)
    journal = read_json_strict(journal_path)
    entries = _transaction_entries(journal, root=root)

    can_roll_forward = all(
        (
            entry["target"].is_file()
            and sha256_file(entry["target"]) == entry["sha256"]
        )
        or (
            entry["staged"].is_file()
            and sha256_file(entry["staged"]) == entry["sha256"]
        )
        for entry in entries
    )
    if can_roll_forward:
        for entry in entries:
            target = entry["target"]
            if not target.is_file() or sha256_file(target) != entry["sha256"]:
                _replace_and_sync(entry["staged"], target)
        if not all(
            entry["target"].is_file()
            and sha256_file(entry["target"]) == entry["sha256"]
            for entry in entries
        ):
            raise WorkerError(
                "PERSISTENCE_RECOVERY_FAILED",
                "transaction roll-forward verification failed",
            )
    else:
        for entry in reversed(entries):
            target = entry["target"]
            backup = entry["backup"]
            if entry["hadOriginal"]:
                if backup.is_file():
                    _replace_and_sync(backup, target)
                elif not target.exists():
                    raise WorkerError(
                        "PERSISTENCE_RECOVERY_FAILED",
                        "transaction backup required for rollback is missing",
                        details={"path": str(target)},
                    )
            else:
                _safe_unlink(target)

    for entry in entries:
        _safe_unlink(entry["staged"])
        _safe_unlink(entry["backup"])
    _safe_unlink(journal_path)
    return True


def atomic_write_json_transaction(
    updates: Mapping[Path, Any],
    *,
    journal_path: Path,
) -> None:
    """Atomically update a set of JSON files using a recoverable journal."""

    if not isinstance(updates, Mapping) or not updates:
        raise ValueError("updates must contain at least one path")
    root = _transaction_root(journal_path)
    recover_json_transaction(journal_path)
    transaction_id = uuid.uuid4().hex
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_path, value in sorted(
        updates.items(), key=lambda item: str(item[0]).casefold()
    ):
        if not isinstance(raw_path, Path):
            raise TypeError("transaction update keys must be pathlib.Path values")
        target = raw_path.absolute().resolve(strict=False)
        target_relative = _relative_transaction_path(target, root)
        if target in seen:
            raise WorkerError(
                "PERSISTENCE_TRANSACTION_INVALID",
                "transaction contains duplicate targets",
            )
        seen.add(target)
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        validate_strict_json(value)
        staged = target.with_name(f".{target.name}.{transaction_id}.stage")
        backup = target.with_name(f".{target.name}.{transaction_id}.backup")
        entries.append(
            {
                "targetPath": target,
                "stagedPath": staged,
                "backupPath": backup,
                "payload": payload,
                "journal": {
                    "target": target_relative,
                    "staged": _relative_transaction_path(staged, root),
                    "backup": _relative_transaction_path(backup, root),
                    "hadOriginal": target.exists(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        )

    journal = {
        "schemaVersion": "1.0.0",
        "transactionId": transaction_id,
        "createdAt": transaction_id,
        "entries": [entry["journal"] for entry in entries],
    }
    journal_written = False
    try:
        for entry in entries:
            _write_bytes_durable(entry["stagedPath"], entry["payload"])
        atomic_write_json(journal_path, journal)
        journal_written = True
        for entry in entries:
            target = entry["targetPath"]
            if entry["journal"]["hadOriginal"]:
                shutil.copyfile(target, entry["backupPath"])
                # Windows' CRT rejects fsync() on a read-only descriptor even
                # though the backup itself was created successfully.
                with entry["backupPath"].open("r+b") as handle:
                    os.fsync(handle.fileno())
            _replace_and_sync(entry["stagedPath"], target)
        for entry in entries:
            if sha256_file(entry["targetPath"]) != entry["journal"]["sha256"]:
                raise OSError("transaction target hash verification failed")
    except Exception:
        if journal_written:
            try:
                journal_entries = _transaction_entries(journal, root=root)
                for entry in reversed(journal_entries):
                    if entry["hadOriginal"] and entry["backup"].is_file():
                        _replace_and_sync(entry["backup"], entry["target"])
                    elif not entry["hadOriginal"]:
                        _safe_unlink(entry["target"])
                for entry in journal_entries:
                    _safe_unlink(entry["staged"])
                    _safe_unlink(entry["backup"])
                _safe_unlink(journal_path)
            except Exception:
                # Leave the journal in place for deterministic startup recovery.
                pass
        else:
            for entry in entries:
                _safe_unlink(entry["stagedPath"])
                _safe_unlink(entry["backupPath"])
        raise
    else:
        for entry in entries:
            _safe_unlink(entry["stagedPath"])
            _safe_unlink(entry["backupPath"])
        _safe_unlink(journal_path)
