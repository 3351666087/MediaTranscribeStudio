from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256
from tools.verify_cnceleb_archive import (
    CNCelebArchiveError,
    build_evidence,
    extract_archive,
    inspect_archive,
)


def _archive(path: Path, *, unsafe_name: str | None = None) -> Path:
    members = {
        "CN-Celeb_flac/eval/lists/enroll.map": (
            b"id00800 id00800/movie/id00800-00001.flac\n"
        ),
        "CN-Celeb_flac/eval/lists/enroll.lst": (
            b"id00800-enroll id00800/movie/id00800-00001.flac\n"
        ),
        "CN-Celeb_flac/eval/lists/trials.lst": (
            b"id00800-enroll eval/id00800-a.flac 1\n"
        ),
        "CN-Celeb_flac/data/id00800/movie/id00800-00001.flac": b"fLaCfixture",
        "CN-Celeb_flac/eval/id00800-a.flac": b"fLaCfixture",
    }
    if unsafe_name is not None:
        members[unsafe_name] = b"escape"
    with tarfile.open(path, "w:gz") as handle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return path


def test_complete_archive_is_hashed_scanned_and_canonical(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "cn-celeb_v2.tar.gz")

    evidence = build_evidence(archive, expected_bytes=archive.stat().st_size)

    assert evidence["validation"]["audioFileCount"] == 2
    assert evidence["validation"]["completeTarScanPassed"] is True
    assert set(evidence["validation"]["requiredMembers"]) == {
        "eval/lists/enroll.map",
        "eval/lists/enroll.lst",
        "eval/lists/trials.lst",
    }
    canonical = dict(evidence)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)


def test_wrong_content_length_is_rejected_before_tar_scan(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "cn-celeb_v2.tar.gz")

    with pytest.raises(CNCelebArchiveError, match="Content-Length"):
        inspect_archive(archive, expected_bytes=archive.stat().st_size + 1)


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute", "a\\b"])
def test_unsafe_member_paths_are_rejected(
    tmp_path: Path, unsafe_name: str
) -> None:
    archive = _archive(tmp_path / "cn-celeb_v2.tar.gz", unsafe_name=unsafe_name)

    with pytest.raises(CNCelebArchiveError, match="member path"):
        inspect_archive(archive, expected_bytes=archive.stat().st_size)


def test_non_regular_member_is_rejected(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "cn-celeb_v2.tar.gz")
    replacement = tmp_path / "linked.tar.gz"
    with tarfile.open(replacement, "w:gz") as handle:
        info = tarfile.TarInfo("CN-Celeb/eval/lists/trials.lst")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        handle.addfile(info)

    with pytest.raises(CNCelebArchiveError, match="non-regular"):
        inspect_archive(replacement, expected_bytes=replacement.stat().st_size)


def test_safe_extraction_refuses_nonempty_destination(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "cn-celeb_v2.tar.gz")
    destination = tmp_path / "extracted"
    destination.mkdir()
    (destination / "owned.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(CNCelebArchiveError, match="must be empty"):
        extract_archive(archive, destination)

    assert (destination / "owned.txt").read_text(encoding="utf-8") == "keep"
