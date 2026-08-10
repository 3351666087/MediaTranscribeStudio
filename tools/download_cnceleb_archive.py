"""Download CN-Celeb1 v2 as independently verified HTTP range chunks."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file
from tools.verify_cnceleb_archive import (
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ETAG,
)


MIRRORS = (
    "https://openslr.trmal.net/resources/82/cn-celeb_v2.tar.gz",
    "https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz",
)
DEFAULT_CHUNK_BYTES = 512 * 1024 * 1024
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$", re.IGNORECASE)


class CNCelebDownloadError(RuntimeError):
    """Raised when one byte range cannot be bound to the frozen object."""


def chunk_ranges(
    total_bytes: int = EXPECTED_ARCHIVE_BYTES,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> tuple[tuple[int, int, int], ...]:
    if total_bytes < 1 or chunk_bytes < 1:
        raise ValueError("total_bytes and chunk_bytes must be positive")
    output = []
    for index, start in enumerate(range(0, total_bytes, chunk_bytes)):
        output.append((index, start, min(total_bytes - 1, start + chunk_bytes - 1)))
    return tuple(output)


def _last_response_headers(raw: str) -> tuple[int, dict[str, str]]:
    responses: list[tuple[int, dict[str, str]]] = []
    status: int | None = None
    headers: dict[str, str] = {}
    for raw_line in raw.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if line.upper().startswith("HTTP/"):
            if status is not None:
                responses.append((status, headers))
            parts = line.split()
            if len(parts) < 2 or not parts[1].isdigit():
                raise CNCelebDownloadError("range response status is malformed")
            status = int(parts[1])
            headers = {}
        elif line and status is not None and ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().casefold()] = value.strip()
    if status is not None:
        responses.append((status, headers))
    if not responses:
        raise CNCelebDownloadError("range response contains no HTTP status")
    return responses[-1]


def validate_range_response(
    raw_headers: str,
    *,
    start: int,
    end: int,
    total_bytes: int = EXPECTED_ARCHIVE_BYTES,
    expected_etag: str = EXPECTED_ETAG,
) -> dict[str, Any]:
    status, headers = _last_response_headers(raw_headers)
    expected_length = end - start + 1
    if status != 206:
        raise CNCelebDownloadError(
            f"range server returned HTTP {status}, expected 206"
        )
    match = _CONTENT_RANGE.fullmatch(headers.get("content-range", ""))
    if (
        match is None
        or tuple(int(value) for value in match.groups())
        != (start, end, total_bytes)
    ):
        raise CNCelebDownloadError("Content-Range does not match the request")
    if headers.get("etag") != expected_etag:
        raise CNCelebDownloadError("range response ETag differs")
    try:
        content_length = int(headers.get("content-length", ""))
    except ValueError as exc:
        raise CNCelebDownloadError("range response has no valid Content-Length") from exc
    if content_length != expected_length:
        raise CNCelebDownloadError("range response Content-Length differs")
    return {
        "status": status,
        "contentRange": headers["content-range"],
        "contentLength": content_length,
        "etag": headers["etag"],
        "lastModified": headers.get("last-modified"),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _chunk_paths(
    chunk_directory: Path, index: int, start: int, end: int
) -> tuple[Path, Path, Path]:
    stem = f"chunk-{index:03d}-{start:012d}-{end:012d}"
    return (
        chunk_directory / f"{stem}.bin",
        chunk_directory / f"{stem}.json",
        chunk_directory / f"{stem}.log",
    )


def _validated_existing_chunk(
    chunk_path: Path,
    sidecar_path: Path,
    *,
    index: int,
    start: int,
    end: int,
    total_bytes: int,
    expected_etag: str,
) -> dict[str, Any] | None:
    if not chunk_path.exists() and not sidecar_path.exists():
        return None
    if not chunk_path.is_file() or not sidecar_path.is_file():
        raise CNCelebDownloadError(
            f"chunk {index} is incomplete; preserved files require manual audit"
        )
    try:
        value = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CNCelebDownloadError(f"chunk {index} sidecar is invalid") from exc
    body = dict(value) if isinstance(value, dict) else {}
    declared = body.pop("canonicalSha256", None)
    expected = {
        "index": index,
        "start": start,
        "end": end,
        "bytes": end - start + 1,
        "totalBytes": total_bytes,
        "etag": expected_etag,
    }
    if (
        not isinstance(value, dict)
        or not isinstance(declared, str)
        or canonical_json_sha256(body) != declared
        or any(value.get(key) != item for key, item in expected.items())
        or chunk_path.stat().st_size != expected["bytes"]
        or value.get("sha256") != sha256_file(chunk_path)
    ):
        raise CNCelebDownloadError(f"chunk {index} failed resume verification")
    return value


def download_chunk(
    *,
    index: int,
    start: int,
    end: int,
    chunk_directory: Path,
    mirrors: Sequence[str] = MIRRORS,
    total_bytes: int = EXPECTED_ARCHIVE_BYTES,
    expected_etag: str = EXPECTED_ETAG,
    retries_per_mirror: int = 4,
) -> dict[str, Any]:
    if not mirrors:
        raise ValueError("at least one mirror is required")
    root = chunk_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    chunk_path, sidecar_path, log_path = _chunk_paths(root, index, start, end)
    existing = _validated_existing_chunk(
        chunk_path,
        sidecar_path,
        index=index,
        start=start,
        end=end,
        total_bytes=total_bytes,
        expected_etag=expected_etag,
    )
    if existing is not None:
        return {**existing, "reused": True, "path": str(chunk_path)}

    attempts: list[dict[str, Any]] = []
    ordered_mirrors = tuple(mirrors[index % len(mirrors) :]) + tuple(
        mirrors[: index % len(mirrors)]
    )
    for mirror in ordered_mirrors:
        for retry in range(1, retries_per_mirror + 1):
            token = f"{os.getpid()}-{threading.get_ident()}-{retry}"
            temporary = root / f".{chunk_path.name}.{token}.tmp"
            header_path = root / f".{chunk_path.name}.{token}.headers"
            command = [
                "curl",
                "--http1.1",
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "30",
                "--max-time",
                "1800",
                "--range",
                f"{start}-{end}",
                "--dump-header",
                str(header_path),
                "--output",
                str(temporary),
                mirror,
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=1900,
            )
            attempt: dict[str, Any] = {
                "mirror": mirror,
                "retry": retry,
                "exitCode": completed.returncode,
                "stderr": completed.stderr.strip(),
                "receivedBytes": temporary.stat().st_size if temporary.exists() else 0,
            }
            try:
                if completed.returncode != 0:
                    raise CNCelebDownloadError(
                        f"curl exited {completed.returncode}"
                    )
                headers_text = header_path.read_text(
                    encoding="iso-8859-1", errors="strict"
                )
                response = validate_range_response(
                    headers_text,
                    start=start,
                    end=end,
                    total_bytes=total_bytes,
                    expected_etag=expected_etag,
                )
                expected_length = end - start + 1
                if temporary.stat().st_size != expected_length:
                    raise CNCelebDownloadError(
                        "downloaded range length differs"
                    )
                digest = sha256_file(temporary)
                attempt["response"] = response
                attempt["sha256"] = digest
                attempts.append(attempt)
                os.replace(temporary, chunk_path)
                value: dict[str, Any] = {
                    "schemaVersion": "1.0.0",
                    "index": index,
                    "start": start,
                    "end": end,
                    "bytes": expected_length,
                    "totalBytes": total_bytes,
                    "etag": expected_etag,
                    "sha256": digest,
                    "sourceUrl": mirror,
                    "response": response,
                    "attemptCount": len(attempts),
                }
                value["canonicalSha256"] = canonical_json_sha256(value)
                _atomic_json(sidecar_path, value)
                log_path.write_text(
                    json.dumps(attempts, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return {**value, "reused": False, "path": str(chunk_path)}
            except (OSError, CNCelebDownloadError) as exc:
                attempt["validationError"] = str(exc)
                attempts.append(attempt)
            finally:
                temporary.unlink(missing_ok=True)
                header_path.unlink(missing_ok=True)
            log_path.write_text(
                json.dumps(attempts, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    raise CNCelebDownloadError(
        f"chunk {index} failed after {len(attempts)} bounded attempts"
    )


def download_chunks(
    *,
    chunk_directory: Path,
    workers: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    mirrors: Sequence[str] = MIRRORS,
    total_bytes: int = EXPECTED_ARCHIVE_BYTES,
    expected_etag: str = EXPECTED_ETAG,
) -> list[dict[str, Any]]:
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    ranges = chunk_ranges(total_bytes, chunk_bytes)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_chunk,
                index=index,
                start=start,
                end=end,
                chunk_directory=chunk_directory,
                mirrors=mirrors,
                total_bytes=total_bytes,
                expected_etag=expected_etag,
            ): index
            for index, start, end in ranges
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return sorted(results, key=lambda item: int(item["index"]))


def assemble_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    total_bytes: int = EXPECTED_ARCHIVE_BYTES,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("chunks must not be empty")
    output = output_path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_start = 0
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".assembling", dir=output.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    written = 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            for row in sorted(chunks, key=lambda item: int(item["index"])):
                start = int(row["start"])
                end = int(row["end"])
                source = Path(str(row["path"])).resolve(strict=True)
                if start != expected_start or end < start:
                    raise CNCelebDownloadError("chunk sequence is not contiguous")
                if source.stat().st_size != end - start + 1:
                    raise CNCelebDownloadError("chunk size changed before assembly")
                if row.get("sha256") != sha256_file(source):
                    raise CNCelebDownloadError("chunk digest changed before assembly")
                with source.open("rb") as handle:
                    while block := handle.read(8 * 1024 * 1024):
                        target.write(block)
                        digest.update(block)
                        written += len(block)
                expected_start = end + 1
            target.flush()
            os.fsync(target.fileno())
        if written != total_bytes or expected_start != total_bytes:
            raise CNCelebDownloadError("assembled archive length differs")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "bytes": written,
        "sha256": digest.hexdigest(),
        "chunkCount": len(chunks),
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=_positive_integer, default=8)
    parser.add_argument(
        "--chunk-bytes", type=_positive_integer, default=DEFAULT_CHUNK_BYTES
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = download_chunks(
        chunk_directory=args.chunk_directory,
        workers=args.workers,
        chunk_bytes=args.chunk_bytes,
    )
    assembled = assemble_chunks(chunks, output_path=args.output)
    print(json.dumps(assembled, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
