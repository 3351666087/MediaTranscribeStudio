from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from tools.download_cnceleb_archive import (
    CNCelebDownloadError,
    _chunk_paths,
    _validated_existing_chunk,
    assemble_chunks,
    chunk_ranges,
    validate_range_response,
)


def test_chunk_ranges_are_contiguous_and_end_exactly() -> None:
    ranges = chunk_ranges(total_bytes=11, chunk_bytes=4)

    assert ranges == ((0, 0, 3), (1, 4, 7), (2, 8, 10))


def test_range_response_requires_exact_206_identity() -> None:
    headers = (
        "HTTP/1.1 206 Partial Content\r\n"
        "ETag: \"fixed\"\r\n"
        "Content-Length: 4\r\n"
        "Content-Range: bytes 4-7/11\r\n\r\n"
    )

    evidence = validate_range_response(
        headers,
        start=4,
        end=7,
        total_bytes=11,
        expected_etag='"fixed"',
    )

    assert evidence["status"] == 206
    assert evidence["contentLength"] == 4


def test_server_ignoring_range_is_rejected() -> None:
    headers = (
        "HTTP/1.1 200 OK\r\n"
        "ETag: \"fixed\"\r\n"
        "Content-Length: 11\r\n\r\n"
    )

    with pytest.raises(CNCelebDownloadError, match="expected 206"):
        validate_range_response(
            headers,
            start=4,
            end=7,
            total_bytes=11,
            expected_etag='"fixed"',
        )


def test_resume_rehashes_every_completed_chunk(tmp_path: Path) -> None:
    chunk, sidecar, _ = _chunk_paths(tmp_path, 0, 0, 3)
    chunk.write_bytes(b"abcd")
    value = {
        "schemaVersion": "1.0.0",
        "index": 0,
        "start": 0,
        "end": 3,
        "bytes": 4,
        "totalBytes": 4,
        "etag": '"fixed"',
        "sha256": sha256_file(chunk),
        "sourceUrl": "fixture",
        "response": {},
        "attemptCount": 1,
    }
    value["canonicalSha256"] = canonical_json_sha256(value)
    sidecar.write_text(json.dumps(value), encoding="utf-8")

    reused = _validated_existing_chunk(
        chunk,
        sidecar,
        index=0,
        start=0,
        end=3,
        total_bytes=4,
        expected_etag='"fixed"',
    )
    assert reused == value

    chunk.write_bytes(b"abce")
    with pytest.raises(CNCelebDownloadError, match="resume verification"):
        _validated_existing_chunk(
            chunk,
            sidecar,
            index=0,
            start=0,
            end=3,
            total_bytes=4,
            expected_etag='"fixed"',
        )


def test_assembly_is_ordered_hashed_and_refuses_overwrite(tmp_path: Path) -> None:
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    left.write_bytes(b"abcd")
    right.write_bytes(b"efg")
    chunks = [
        {
            "index": 1,
            "start": 4,
            "end": 6,
            "path": str(right),
            "sha256": sha256_file(right),
        },
        {
            "index": 0,
            "start": 0,
            "end": 3,
            "path": str(left),
            "sha256": sha256_file(left),
        },
    ]
    output = tmp_path / "archive.bin"

    evidence = assemble_chunks(chunks, output_path=output, total_bytes=7)

    assert output.read_bytes() == b"abcdefg"
    assert evidence["sha256"] == sha256_file(output)
    with pytest.raises(FileExistsError, match="overwrite"):
        assemble_chunks(chunks, output_path=output, total_bytes=7)
