"""Fail-closed integrity and extraction checks for the CN-Celeb1 v2 archive."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402


EXPECTED_ARCHIVE_BYTES = 22_264_439_915
EXPECTED_ETAG = '"52f10646b-5c426857e46c0"'
SOURCE_URL = "https://openslr.trmal.net/resources/82/cn-celeb_v2.tar.gz"
CATALOG_URL = "https://www.openslr.org/82/"
REQUIRED_SUFFIXES = (
    "eval/lists/enroll.map",
    "eval/lists/enroll.lst",
    "eval/lists/trials.lst",
)
AUDIO_SUFFIXES = frozenset({".flac", ".wav"})
MAX_UNCOMPRESSED_BYTES = 256 * 1024**3


class CNCelebArchiveError(RuntimeError):
    """Raised when the archive cannot support a trustworthy evaluation gate."""


def _safe_member_name(raw_name: str) -> str:
    if not isinstance(raw_name, str) or not raw_name or "\\" in raw_name:
        raise CNCelebArchiveError("archive contains an invalid member path")
    normalized_text = raw_name
    while normalized_text.startswith("./"):
        normalized_text = normalized_text[2:]
    path = PurePosixPath(normalized_text)
    if (
        not normalized_text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CNCelebArchiveError(
            f"archive contains an unsafe member path: {raw_name!r}"
        )
    return path.as_posix()


def inspect_archive(
    archive_path: Path,
    *,
    expected_bytes: int | None = EXPECTED_ARCHIVE_BYTES,
    maximum_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    """Scan every member without extraction and return immutable evidence."""

    archive = archive_path.resolve(strict=True)
    if not archive.is_file() or archive.is_symlink():
        raise CNCelebArchiveError("archive must be a regular local file")
    archive_bytes = archive.stat().st_size
    if expected_bytes is not None and archive_bytes != expected_bytes:
        raise CNCelebArchiveError(
            "archive size differs from the frozen upstream Content-Length"
        )
    if maximum_uncompressed_bytes < 1:
        raise ValueError("maximum_uncompressed_bytes must be positive")

    seen: set[str] = set()
    matched_required: dict[str, str] = {}
    member_count = 0
    regular_file_count = 0
    audio_file_count = 0
    uncompressed_bytes = 0
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            for member in handle:
                member_count += 1
                name = _safe_member_name(member.name)
                folded = name.casefold()
                if folded in seen:
                    raise CNCelebArchiveError(
                        f"archive contains a duplicate member path: {name}"
                    )
                seen.add(folded)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise CNCelebArchiveError(
                        f"archive contains a non-regular member: {name}"
                    )
                regular_file_count += 1
                if member.size < 0:
                    raise CNCelebArchiveError(
                        f"archive member has an invalid size: {name}"
                    )
                uncompressed_bytes += member.size
                if uncompressed_bytes > maximum_uncompressed_bytes:
                    raise CNCelebArchiveError(
                        "archive exceeds the bounded uncompressed-size policy"
                    )
                if PurePosixPath(folded).suffix in AUDIO_SUFFIXES:
                    audio_file_count += 1
                for suffix in REQUIRED_SUFFIXES:
                    if folded == suffix or folded.endswith(f"/{suffix}"):
                        if suffix in matched_required:
                            raise CNCelebArchiveError(
                                f"archive contains multiple {suffix} files"
                            )
                        matched_required[suffix] = name
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise CNCelebArchiveError(
            "archive is not a complete readable gzip tar stream"
        ) from exc

    missing = sorted(set(REQUIRED_SUFFIXES) - set(matched_required))
    if missing:
        raise CNCelebArchiveError(
            "archive is missing official evaluation lists: " + ", ".join(missing)
        )
    if audio_file_count < 1:
        raise CNCelebArchiveError("archive contains no supported FLAC or WAV audio")
    return {
        "archivePath": str(archive),
        "archiveBytes": archive_bytes,
        "archiveSha256": sha256_file(archive),
        "memberCount": member_count,
        "regularFileCount": regular_file_count,
        "audioFileCount": audio_file_count,
        "uncompressedBytes": uncompressed_bytes,
        "requiredMembers": {
            key: matched_required[key] for key in sorted(matched_required)
        },
        "completeTarScanPassed": True,
        "safeMemberPolicyPassed": True,
    }


def build_evidence(
    archive_path: Path,
    *,
    expected_bytes: int | None = EXPECTED_ARCHIVE_BYTES,
    maximum_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Any]:
    validation = inspect_archive(
        archive_path,
        expected_bytes=expected_bytes,
        maximum_uncompressed_bytes=maximum_uncompressed_bytes,
    )
    evidence: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "datasetKey": "openslr-cnceleb1-v2",
        "source": {
            "catalogUrl": CATALOG_URL,
            "archiveUrl": SOURCE_URL,
            "expectedContentLengthBytes": expected_bytes,
            "expectedEtag": EXPECTED_ETAG,
        },
        "license": {
            "spdx": "CC-BY-SA-4.0",
            "catalogLabel": "Attribution-ShareAlike 4.0 International",
        },
        "integrity": {
            "publishedChecksumAvailableForV2": False,
            "localSha256IsDownloadEvidenceNotPublishedUpstreamIdentity": True,
        },
        "validation": validation,
        "promotionPolicy": {
            "speakerDisjointDevelopmentAndHeldOutRequired": True,
            "recordingDisjointTrialSidesRequired": True,
            "heldOutThresholdFittingAllowed": False,
        },
    }
    evidence["canonicalSha256"] = canonical_json_sha256(evidence)
    return evidence


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extract a previously validated archive into a new empty directory."""

    archive = archive_path.resolve(strict=True)
    target = destination.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise CNCelebArchiveError(
            "extraction destination must not exist or must be empty"
        )
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            handle.extractall(target, filter="data")
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise CNCelebArchiveError("safe archive extraction failed") from exc
    return target


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
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
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-bytes",
        type=_positive_integer,
        default=EXPECTED_ARCHIVE_BYTES,
    )
    parser.add_argument(
        "--maximum-uncompressed-bytes",
        type=_positive_integer,
        default=MAX_UNCOMPRESSED_BYTES,
    )
    parser.add_argument("--extract-to", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = build_evidence(
        args.archive,
        expected_bytes=args.expected_bytes,
        maximum_uncompressed_bytes=args.maximum_uncompressed_bytes,
    )
    _write_json(args.output, evidence)
    if args.extract_to is not None:
        extract_archive(args.archive, args.extract_to)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonicalSha256": evidence["canonicalSha256"],
                "archiveSha256": evidence["validation"]["archiveSha256"],
                "counts": {
                    key: evidence["validation"][key]
                    for key in (
                        "memberCount",
                        "regularFileCount",
                        "audioFileCount",
                        "uncompressedBytes",
                    )
                },
                "extractedTo": (
                    str(args.extract_to.resolve())
                    if args.extract_to is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
