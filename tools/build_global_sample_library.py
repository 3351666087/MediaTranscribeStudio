"""Download and normalize the versioned global speech sample matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.global_sample_library import (  # noqa: E402
    GlobalSampleCase,
    GlobalSampleLibraryError,
    GlobalSampleManifest,
    GlobalSampleSource,
    coverage_summary,
    load_global_manifest,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "sample_library" / "global-manifest.v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / ".runtime_cache" / "sample-library" / "global"
RESOLVED_NAME = "global-sample-library.resolved.v1.json"
USER_AGENT = "MediaTranscribeStudio-global-sample-library/1.0"
PARQUET_CACHE_ROOT = (
    PROJECT_ROOT / ".runtime_cache" / "sample-library" / "fleurs-parquet"
)
_PARQUET_FILE_CACHE: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_bytes(
    url: str,
    *,
    timeout: float = 120.0,
    attempts: int = 5,
) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise GlobalSampleLibraryError(
        f"remote request failed after {attempts} attempts: {url}: {last_error}"
    )


def _request_json(url: str, *, timeout: float = 120.0) -> dict[str, Any]:
    try:
        value = json.loads(_request_bytes(url, timeout=timeout))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalSampleLibraryError(f"remote JSON is invalid: {url}") from exc
    if not isinstance(value, dict):
        raise GlobalSampleLibraryError(f"remote JSON is not an object: {url}")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file: Any,
        code: int,
        msg: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def _request_redirect_headers(
    url: str,
    *,
    timeout: float = 120.0,
    attempts: int = 3,
) -> dict[str, str]:
    """Read Hub's immutable file headers without following the CDN redirect."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
    )
    opener = urllib.request.build_opener(_NoRedirect())
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                return {
                    str(key).casefold(): str(value)
                    for key, value in response.headers.items()
                }
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                return {
                    str(key).casefold(): str(value)
                    for key, value in exc.headers.items()
                }
            last_error = exc
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(float(attempt))
    raise GlobalSampleLibraryError(
        f"remote parquet headers failed after {attempts} attempts: {url}: {last_error}"
    )


def _dataset_metadata(source: GlobalSampleSource) -> dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{source.dataset}"
    value = _request_json(url)
    if value.get("sha") != source.revision:
        raise GlobalSampleLibraryError(
            f"{source.dataset} moved from pinned revision {source.revision}; "
            "review and update the manifest explicitly"
        )
    if value.get("private") is not False or value.get("gated") not in {False, None}:
        raise GlobalSampleLibraryError(
            f"{source.dataset} is no longer an ungated public dataset"
        )
    tags = value.get("tags", [])
    license_tokens = {
        str(item).split(":", 1)[1].casefold()
        for item in tags
        if isinstance(item, str) and item.startswith("license:")
    }
    card = value.get("cardData")
    card_license = card.get("license") if isinstance(card, dict) else None
    if isinstance(card_license, str):
        license_tokens.add(card_license.casefold())
    elif isinstance(card_license, list):
        license_tokens.update(
            str(item).casefold() for item in card_license if isinstance(item, str)
        )
    if source.license not in license_tokens:
        raise GlobalSampleLibraryError(
            f"{source.dataset} no longer declares pinned license {source.license}"
        )
    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "license": source.license,
        "homepage": source.homepage,
        "attribution": source.attribution,
        "lastModified": value.get("lastModified"),
    }


def _audio_source(value: Any, field: str) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise GlobalSampleLibraryError(f"{field} must contain exactly one audio asset")
    item = value[0]
    if not isinstance(item, dict):
        raise GlobalSampleLibraryError(f"{field}[0] must be an object")
    source_url = item.get("src")
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        raise GlobalSampleLibraryError(f"{field}[0].src must be HTTPS")
    return source_url


def _case_row_metadata(
    row: dict[str, Any],
    *,
    source: GlobalSampleSource,
    case: GlobalSampleCase,
) -> dict[str, Any]:
    acquisition = case.acquisition
    transcript_field = acquisition.get("transcriptField", "transcription")
    raw_transcript_field = acquisition.get(
        "rawTranscriptField",
        "raw_transcription",
    )
    path_field = acquisition.get("pathField", "path")
    transcript = row.get(transcript_field)
    if not isinstance(transcript, str) or not transcript.strip():
        raise GlobalSampleLibraryError(
            f"{case.case_id} transcript is missing from {transcript_field}"
        )
    raw_transcript = row.get(raw_transcript_field)
    if not isinstance(raw_transcript, str) or not raw_transcript.strip():
        raw_transcript = transcript

    verified_groups: dict[str, dict[str, str]] = {}
    for group_kind, field_key, id_key in (
        ("speaker", "speakerField", "speakerId"),
        ("recording", "recordingField", "recordingId"),
    ):
        field_name = acquisition.get(field_key)
        expected_id = acquisition.get(id_key)
        if not isinstance(field_name, str) or not isinstance(expected_id, str):
            continue
        actual_id = row.get(field_name)
        if str(actual_id) != expected_id:
            raise GlobalSampleLibraryError(
                f"{case.case_id} {group_kind} identity mismatch: "
                f"expected {expected_id!r}, got {actual_id!r}"
            )
        verified_groups[group_kind] = {
            "field": field_name,
            "id": expected_id,
        }

    source_row_id: dict[str, str] | None = None
    row_id_field = acquisition.get("rowIdField")
    expected_row_id = acquisition.get("rowId")
    if isinstance(row_id_field, str) and isinstance(expected_row_id, str):
        actual_row_id = row.get(row_id_field)
        if str(actual_row_id) != expected_row_id:
            raise GlobalSampleLibraryError(
                f"{case.case_id} source row identity mismatch: "
                f"expected {expected_row_id!r}, got {actual_row_id!r}"
            )
        source_row_id = {"field": row_id_field, "id": expected_row_id}

    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "config": acquisition["config"],
        "split": acquisition["split"],
        "rowIndex": acquisition["rowIndex"],
        "path": row.get(path_field),
        "transcript": transcript.strip(),
        "rawTranscript": raw_transcript.strip(),
        "englishTranscript": (
            row.get("english_transcription").strip()
            if isinstance(row.get("english_transcription"), str)
            else None
        ),
        "gender": row.get("gender"),
        "fieldMapping": {
            "transcript": transcript_field,
            "rawTranscript": raw_transcript_field,
            "path": path_field,
        },
        "verifiedSourceGroups": verified_groups,
        "sourceRowId": source_row_id,
        "sourceAssetType": "audio/wav",
    }


def _immutable_parquet_descriptor(
    source: GlobalSampleSource,
    *,
    config: str,
    split: str,
) -> dict[str, Any]:
    cache_key = (source.dataset, source.revision, config, split)
    cached = _PARQUET_FILE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    # Validate the source revision and license independently of the mutable
    # parquet conversion ref before accepting any embedded audio bytes.
    _dataset_metadata(source)
    body = _request_json(
        "https://datasets-server.huggingface.co/parquet?"
        + urllib.parse.urlencode({"dataset": source.dataset}),
        timeout=180.0,
    )
    if body.get("partial") is not False:
        raise GlobalSampleLibraryError(f"{source.dataset} parquet listing is partial")
    files = body.get("parquet_files")
    if not isinstance(files, list) or not files:
        raise GlobalSampleLibraryError(f"{source.dataset} parquet listing is invalid")
    matches = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("dataset") == source.dataset
        and item.get("config") == config
        and item.get("split") == split
    ]
    if len(matches) != 1:
        raise GlobalSampleLibraryError(
            f"{source.dataset} parquet shard is not unique for {config}/{split}"
        )
    item = matches[0]
    url = item.get("url")
    filename = item.get("filename")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise GlobalSampleLibraryError(f"{config}/{split} parquet URL must be HTTPS")
    if not isinstance(filename, str) or not filename:
        raise GlobalSampleLibraryError(f"{config}/{split} parquet filename is invalid")
    headers = _request_redirect_headers(url, timeout=180.0)
    parquet_revision = headers.get("x-repo-commit")
    content_sha = headers.get("x-linked-etag", "").strip('"')
    linked_size = headers.get("x-linked-size")
    if not isinstance(parquet_revision, str) or not _HEX40.fullmatch(parquet_revision):
        raise GlobalSampleLibraryError(
            f"{config}/{split}/{filename} parquet revision is not immutable"
        )
    if not _HEX64.fullmatch(content_sha):
        raise GlobalSampleLibraryError(
            f"{config}/{split}/{filename} parquet content hash is missing"
        )
    try:
        size = int(linked_size or item.get("size"))
    except (TypeError, ValueError) as exc:
        raise GlobalSampleLibraryError(
            f"{config}/{split}/{filename} parquet size is invalid"
        ) from exc
    if size <= 0:
        raise GlobalSampleLibraryError(
            f"{config}/{split}/{filename} parquet size is invalid"
        )
    parsed = urllib.parse.urlsplit(url)
    marker = "/resolve/"
    if marker not in parsed.path:
        raise GlobalSampleLibraryError(f"{config}/{split}/{filename} parquet URL is invalid")
    prefix, resolved_tail = parsed.path.split(marker, 1)
    _, separator, shard_tail = resolved_tail.partition("/")
    if not separator or not shard_tail:
        shard_tail = f"{config}/{split}/{filename.lstrip('/')}"
    immutable_path = f"{prefix}{marker}{parquet_revision}/{shard_tail}"
    descriptor = {
        "url": urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, immutable_path, "", "")
        ),
        "revision": parquet_revision,
        "filename": f"{config}/{split}/{filename}",
        "contentSha256": content_sha,
        "size": size,
        "sourceRevision": source.revision,
    }
    _PARQUET_FILE_CACHE[cache_key] = descriptor
    return dict(descriptor)


def _parquet_audio_bytes(value: Any, field: str) -> bytes:
    if isinstance(value, dict):
        audio_bytes = value.get("bytes")
    elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        audio_bytes = value[0].get("bytes")
    else:
        audio_bytes = None
    if isinstance(audio_bytes, memoryview):
        audio_bytes = audio_bytes.tobytes()
    if isinstance(audio_bytes, bytearray):
        audio_bytes = bytes(audio_bytes)
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise GlobalSampleLibraryError(
            f"{field} must contain exactly one non-empty audio asset"
        )
    return audio_bytes


def _validate_parquet_access(
    access: Any,
    *,
    source: GlobalSampleSource,
    case: GlobalSampleCase,
    audio_bytes: bytes,
) -> dict[str, Any]:
    if not isinstance(access, dict):
        raise GlobalSampleLibraryError(
            f"{case.case_id} parquet provenance is invalid"
        )
    revision = access.get("revision")
    content_sha = access.get("contentSha256")
    audio_sha = access.get("audioSha256")
    url = access.get("url")
    if access.get("sourceRevision") != source.revision:
        raise GlobalSampleLibraryError(
            f"{case.case_id} parquet source revision is not pinned"
        )
    if not isinstance(revision, str) or not _HEX40.fullmatch(revision):
        raise GlobalSampleLibraryError(f"{case.case_id} parquet revision is invalid")
    if not isinstance(content_sha, str) or not _HEX64.fullmatch(content_sha):
        raise GlobalSampleLibraryError(
            f"{case.case_id} parquet content hash is invalid"
        )
    if not isinstance(audio_sha, str) or not _HEX64.fullmatch(audio_sha):
        raise GlobalSampleLibraryError(f"{case.case_id} parquet audio hash is invalid")
    if hashlib.sha256(audio_bytes).hexdigest() != audio_sha:
        raise GlobalSampleLibraryError(
            f"{case.case_id} parquet audio content hash mismatch"
        )
    if not isinstance(url, str) or not url.startswith(
        f"https://huggingface.co/datasets/{source.dataset}/resolve/{revision}/"
    ):
        raise GlobalSampleLibraryError(
            f"{case.case_id} parquet URL is not pinned to its revision"
        )
    if not isinstance(access.get("filename"), str) or not access["filename"]:
        raise GlobalSampleLibraryError(f"{case.case_id} parquet filename is invalid")
    if not isinstance(access.get("size"), int) or access["size"] <= 0:
        raise GlobalSampleLibraryError(f"{case.case_id} parquet size is invalid")
    if access.get("rowIndex") != case.acquisition["rowIndex"]:
        raise GlobalSampleLibraryError(
            f"{case.case_id} parquet row identity is invalid"
        )
    return access


def _parquet_row(
    source: GlobalSampleSource,
    case: GlobalSampleCase,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Resolve one pinned row from an immutable Dataset Hub parquet shard."""
    acquisition = case.acquisition
    descriptor = _immutable_parquet_descriptor(
        source,
        config=acquisition["config"],
        split=acquisition["split"],
    )
    if descriptor.get("sourceRevision") != source.revision:
        raise GlobalSampleLibraryError(
            f"{case.case_id} parquet source revision is not pinned"
        )
    try:
        import fsspec
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise GlobalSampleLibraryError(
            "pyarrow and fsspec are required for immutable parquet fallback"
        ) from exc
    cache_root = PARQUET_CACHE_ROOT
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        filesystem = fsspec.filesystem(
            "blockcache",
            target_protocol="https",
            cache_storage=str(cache_root),
            block_size=8 * 1024 * 1024,
            check_files=True,
        )
        with filesystem.open(descriptor["url"], "rb") as handle:
            parquet_file = parquet.ParquetFile(handle)
            target_index = acquisition["rowIndex"]
            if target_index < 0 or target_index >= parquet_file.metadata.num_rows:
                raise GlobalSampleLibraryError(
                    f"{case.case_id} parquet rowIndex is out of bounds"
                )
            row_group = 0
            row_offset = target_index
            for index in range(parquet_file.num_row_groups):
                group_rows = parquet_file.metadata.row_group(index).num_rows
                if row_offset < group_rows:
                    row_group = index
                    break
                row_offset -= group_rows
            field_names = set(parquet_file.schema_arrow.names)
            fields = {
                "audio",
                acquisition.get("transcriptField", "transcription"),
                acquisition.get("rawTranscriptField", "raw_transcription"),
                acquisition.get("pathField", "path"),
                "english_transcription",
                "gender",
                acquisition.get("rowIdField"),
                acquisition.get("speakerField"),
                acquisition.get("recordingField"),
            }
            columns = sorted(field for field in fields if field in field_names)
            row = parquet_file.read_row_group(row_group, columns=columns).slice(
                row_offset,
                1,
            ).to_pylist()[0]
    except GlobalSampleLibraryError:
        raise
    except Exception as exc:
        raise GlobalSampleLibraryError(
            f"{case.case_id} immutable parquet read failed: {type(exc).__name__}"
        ) from exc
    if not isinstance(row, dict):
        raise GlobalSampleLibraryError(f"{case.case_id} parquet row is invalid")
    audio_bytes = _parquet_audio_bytes(row.get("audio"), f"{case.case_id}.audio")
    access = {
        **descriptor,
        "rowIndex": acquisition["rowIndex"],
        "audioSha256": hashlib.sha256(audio_bytes).hexdigest(),
    }
    _validate_parquet_access(access, source=source, case=case, audio_bytes=audio_bytes)
    return audio_bytes, row, access


def _viewer_row(
    source: GlobalSampleSource,
    case: GlobalSampleCase,
) -> tuple[bytes, dict[str, Any]]:
    acquisition = case.acquisition
    params = urllib.parse.urlencode(
        {
            "dataset": source.dataset,
            "config": acquisition["config"],
            "split": acquisition["split"],
            "offset": acquisition["rowIndex"],
            "length": 1,
        }
    )
    try:
        value = _request_json(
            f"https://datasets-server.huggingface.co/rows?{params}",
            timeout=180.0,
        )
    except GlobalSampleLibraryError as rows_error:
        if acquisition["kind"] != "hf-viewer-search-row":
            raise
        try:
            audio_bytes, row, parquet_access = _parquet_row(source, case)
            _validate_parquet_access(
                parquet_access,
                source=source,
                case=case,
                audio_bytes=audio_bytes,
            )
            metadata = _case_row_metadata(row, source=source, case=case)
            metadata["viewerAccess"] = {
                "kind": acquisition["kind"],
                "searchQuery": acquisition.get("searchQuery"),
                "selectionEndpoint": "/search",
                "fetchEndpoint": "/parquet",
                "fetchRowIndex": acquisition["rowIndex"],
                "fallbackFrom": "/rows",
                "parquet": parquet_access,
            }
            return audio_bytes, metadata
        except GlobalSampleLibraryError as parquet_error:
            raise GlobalSampleLibraryError(
                f"{case.case_id} pinned /rows failed ({rows_error}); immutable "
                f"parquet fallback failed: {parquet_error}"
            ) from parquet_error
        except (OSError, ValueError, TypeError) as parquet_error:
            raise GlobalSampleLibraryError(
                f"{case.case_id} pinned /rows failed ({rows_error}); immutable "
                f"parquet fallback failed: {type(parquet_error).__name__}"
            ) from parquet_error
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise GlobalSampleLibraryError(f"{case.case_id} did not resolve one row")
    wrapped = rows[0]
    if (
        not isinstance(wrapped, dict)
        or wrapped.get("row_idx") != acquisition["rowIndex"]
        or not isinstance(wrapped.get("row"), dict)
    ):
        raise GlobalSampleLibraryError(f"{case.case_id} row identity is invalid")
    row = wrapped["row"]
    audio_url = _audio_source(row.get("audio"), f"{case.case_id}.audio")
    if source.revision not in urllib.parse.unquote(audio_url):
        raise GlobalSampleLibraryError(
            f"{case.case_id} audio asset is not bound to the pinned dataset revision"
        )
    metadata = _case_row_metadata(
        row,
        source=source,
        case=case,
    )
    viewer_access = {
        "kind": acquisition["kind"],
        "searchQuery": acquisition.get("searchQuery"),
        "fetchEndpoint": "/rows",
        "fetchRowIndex": acquisition["rowIndex"],
    }
    if acquisition["kind"] == "hf-viewer-search-row":
        viewer_access["selectionEndpoint"] = "/search"
    metadata["viewerAccess"] = viewer_access
    return _request_bytes(audio_url, timeout=180.0), metadata


def _streaming_row(
    source: GlobalSampleSource,
    case: GlobalSampleCase,
) -> tuple[bytes, dict[str, Any]]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise GlobalSampleLibraryError(
            "datasets is required for hf-streaming-row cases"
        ) from exc
    acquisition = case.acquisition
    try:
        stream = load_dataset(
            source.dataset,
            acquisition["config"],
            split=acquisition["split"],
            revision=source.revision,
            streaming=True,
        ).cast_column("audio", Audio(decode=False))
        row: dict[str, Any] | None = None
        for index, candidate in enumerate(stream):
            if index == acquisition["rowIndex"]:
                row = candidate
                break
    except Exception as exc:
        raise GlobalSampleLibraryError(
            f"{case.case_id} Hugging Face streaming failed: {type(exc).__name__}"
        ) from exc
    if row is None:
        raise GlobalSampleLibraryError(f"{case.case_id} row is unavailable")
    audio = row.get("audio")
    if not isinstance(audio, dict):
        raise GlobalSampleLibraryError(f"{case.case_id}.audio must be an object")
    audio_bytes = audio.get("bytes")
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise GlobalSampleLibraryError(f"{case.case_id}.audio bytes are missing")
    metadata = _case_row_metadata(row, source=source, case=case)
    del stream
    gc.collect()
    return audio_bytes, metadata


def _ffmpeg_normalize(source: Path, output: Path) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise GlobalSampleLibraryError(
            f"ffmpeg failed to normalize {source.name}: {completed.stderr.strip()}"
        )


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return float(completed.stdout.strip())


def _build_case(
    *,
    manifest: GlobalSampleManifest,
    source: GlobalSampleSource,
    case: GlobalSampleCase,
    output_root: Path,
) -> dict[str, Any]:
    if case.acquisition["kind"] in {
        "hf-viewer-row",
        "hf-viewer-search-row",
    }:
        audio_bytes, row = _viewer_row(source, case)
    else:
        audio_bytes, row = _streaming_row(source, case)
    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    output = audio_root / f"{case.case_id}.wav"
    with tempfile.TemporaryDirectory(prefix="mts-global-sample-") as temp_root:
        original = Path(temp_root) / "source-audio"
        original.write_bytes(audio_bytes)
        source_sha = _sha256(original)
        _ffmpeg_normalize(original, output)
    duration = _probe_duration(output)
    if not 0 < duration <= manifest.max_duration_seconds:
        output.unlink(missing_ok=True)
        raise GlobalSampleLibraryError(
            f"{case.case_id} duration {duration:.3f}s exceeds "
            f"{manifest.max_duration_seconds:.3f}s"
        )
    return {
        "id": case.case_id,
        "sourceId": case.source_id,
        "language": case.language,
        "region": case.region,
        "evaluationSplit": case.evaluation_split,
        "scenario": list(case.scenarios),
        "expectedSpeakerCount": case.expected_speaker_count,
        "expectedTranscript": row["transcript"],
        "scoringTranscript": row["transcript"],
        "rawTranscript": row["rawTranscript"],
        "nativeTranscript": row["rawTranscript"],
        "transcriptNormalization": (
            "dataset-provided-pair"
            if row["rawTranscript"] != row["transcript"]
            else "identity"
        ),
        "englishTranscript": row["englishTranscript"],
        "gender": row["gender"],
        "path": str(output.relative_to(output_root)),
        "durationSeconds": round(duration, 6),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "sourceArtifactSha256": source_sha,
        "downloadedAt": datetime.fromtimestamp(
            output.stat().st_mtime,
            UTC,
        ).isoformat(),
        "sourceLocator": {
            "url": (
                f"https://huggingface.co/datasets/{source.dataset}/tree/"
                f"{source.revision}"
            ),
            "dataset": source.dataset,
            "revision": source.revision,
            "config": case.acquisition["config"],
            "split": case.acquisition["split"],
            "rowIndex": case.acquisition["rowIndex"],
        },
        "sourceRow": {
            key: value
            for key, value in row.items()
            if key
            not in {
                "transcript",
                "rawTranscript",
                "englishTranscript",
                "gender",
            }
        },
    }


def _refresh_existing_case(
    row: dict[str, Any],
    *,
    source: GlobalSampleSource,
    case: GlobalSampleCase,
    output_path: Path,
) -> None:
    transcript = row.get("expectedTranscript")
    raw_transcript = row.get("rawTranscript")
    if not isinstance(transcript, str) or not transcript:
        raise GlobalSampleLibraryError(
            f"{case.case_id} cached expectedTranscript is missing"
        )
    if not isinstance(raw_transcript, str) or not raw_transcript:
        raw_transcript = transcript
    source_row = row.get("sourceRow")
    if not isinstance(source_row, dict):
        raise GlobalSampleLibraryError(
            f"{case.case_id} cached sourceRow is missing"
        )
    source_row["fieldMapping"] = {
        "transcript": case.acquisition.get("transcriptField", "transcription"),
        "rawTranscript": case.acquisition.get(
            "rawTranscriptField",
            "raw_transcription",
        ),
        "path": case.acquisition.get("pathField", "path"),
    }
    source_row.setdefault("verifiedSourceGroups", {})
    row.update(
        {
            "sourceId": case.source_id,
            "language": case.language,
            "region": case.region,
            "evaluationSplit": case.evaluation_split,
            "scenario": list(case.scenarios),
            "expectedSpeakerCount": case.expected_speaker_count,
            "scoringTranscript": transcript,
            "nativeTranscript": raw_transcript,
            "transcriptNormalization": (
                "dataset-provided-pair"
                if raw_transcript != transcript
                else "identity"
            ),
            "downloadedAt": datetime.fromtimestamp(
                output_path.stat().st_mtime,
                UTC,
            ).isoformat(),
            "sourceLocator": {
                "url": (
                    f"https://huggingface.co/datasets/{source.dataset}/tree/"
                    f"{source.revision}"
                ),
                "dataset": source.dataset,
                "revision": source.revision,
                "config": case.acquisition["config"],
                "split": case.acquisition["split"],
                "rowIndex": case.acquisition["rowIndex"],
            },
        }
    )


def _cached_case_matches_acquisition(
    row: dict[str, Any],
    *,
    source: GlobalSampleSource,
    case: GlobalSampleCase,
) -> bool:
    locator = row.get("sourceLocator")
    if not isinstance(locator, dict) or locator != {
        "url": (
            f"https://huggingface.co/datasets/{source.dataset}/tree/"
            f"{source.revision}"
        ),
        "dataset": source.dataset,
        "revision": source.revision,
        "config": case.acquisition["config"],
        "split": case.acquisition["split"],
        "rowIndex": case.acquisition["rowIndex"],
    }:
        return False
    source_row = row.get("sourceRow")
    if not isinstance(source_row, dict):
        return False
    expected_groups: dict[str, dict[str, str]] = {}
    for group_kind, field_key, id_key in (
        ("speaker", "speakerField", "speakerId"),
        ("recording", "recordingField", "recordingId"),
    ):
        field_name = case.acquisition.get(field_key)
        group_id = case.acquisition.get(id_key)
        if isinstance(field_name, str) and isinstance(group_id, str):
            expected_groups[group_kind] = {
                "field": field_name,
                "id": group_id,
            }
    if source_row.get("verifiedSourceGroups", {}) != expected_groups:
        return False
    row_id_field = case.acquisition.get("rowIdField")
    row_id = case.acquisition.get("rowId")
    if isinstance(row_id_field, str) and isinstance(row_id, str):
        if source_row.get("sourceRowId") != {
            "field": row_id_field,
            "id": row_id,
        }:
            return False
    if case.acquisition["kind"] == "hf-viewer-search-row":
        viewer_access = source_row.get("viewerAccess")
        if not isinstance(viewer_access, dict) or viewer_access.get(
            "kind"
        ) != "hf-viewer-search-row" or viewer_access.get(
            "searchQuery"
        ) != case.acquisition["searchQuery"]:
            return False
        if viewer_access.get("selectionEndpoint") != "/search":
            return False
        if viewer_access.get("fetchRowIndex") != case.acquisition["rowIndex"]:
            return False
        if viewer_access.get("fetchEndpoint") == "/rows":
            if set(viewer_access) != {
                "kind",
                "searchQuery",
                "selectionEndpoint",
                "fetchEndpoint",
                "fetchRowIndex",
            }:
                return False
        elif viewer_access.get("fetchEndpoint") == "/parquet":
            parquet_access = viewer_access.get("parquet")
            if not isinstance(parquet_access, dict):
                return False
            if parquet_access.get("sourceRevision") != source.revision:
                return False
            if not isinstance(parquet_access.get("revision"), str) or not _HEX40.fullmatch(
                parquet_access["revision"]
            ):
                return False
            if not isinstance(parquet_access.get("contentSha256"), str) or not _HEX64.fullmatch(
                parquet_access["contentSha256"]
            ):
                return False
            parquet_revision = parquet_access["revision"]
            parquet_url = parquet_access.get("url")
            if not isinstance(parquet_url, str) or not parquet_url.startswith(
                f"https://huggingface.co/datasets/{source.dataset}/resolve/"
                f"{parquet_revision}/"
            ):
                return False
            if not isinstance(parquet_access.get("filename"), str) or not parquet_access[
                "filename"
            ].startswith(
                f"{case.acquisition['config']}/{case.acquisition['split']}/"
            ):
                return False
            if not isinstance(parquet_access.get("size"), int) or parquet_access[
                "size"
            ] <= 0:
                return False
            if parquet_access.get("rowIndex") != case.acquisition["rowIndex"]:
                return False
            if not isinstance(parquet_access.get("audioSha256"), str) or not _HEX64.fullmatch(
                parquet_access["audioSha256"]
            ):
                return False
            if viewer_access.get("fallbackFrom") != "/rows":
                return False
        else:
            return False
    expected_mapping = {
        "transcript": case.acquisition.get("transcriptField", "transcription"),
        "rawTranscript": case.acquisition.get(
            "rawTranscriptField",
            "raw_transcription",
        ),
        "path": case.acquisition.get("pathField", "path"),
    }
    cached_mapping = source_row.get("fieldMapping")
    if cached_mapping is None:
        return not any(
            key in case.acquisition
            for key in ("transcriptField", "rawTranscriptField", "pathField")
        )
    return cached_mapping == expected_mapping


def _existing_cases(output_root: Path) -> dict[str, dict[str, Any]]:
    path = output_root / RESOLVED_NAME
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    rows = value.get("cases") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        return {}
    return {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def _write_attribution(
    output_root: Path,
    sources: Sequence[dict[str, Any]],
) -> Path:
    path = output_root / "ATTRIBUTION.md"
    lines = [
        "# Global Sample Library Attribution",
        "",
        "Generated media is local evaluation evidence and is not committed to Git.",
        "",
    ]
    for source in sources:
        lines.extend(
            [
                f"## {source['dataset']}",
                "",
                f"- Revision: `{source['revision']}`",
                f"- License: `{source['license']}`",
                f"- Homepage: {source['homepage']}",
                f"- Attribution: {source['attribution']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_global_manifest(args.manifest)
    source_by_id = {source.source_id: source for source in manifest.sources}
    selected_sources = set(args.source_id)
    selected_cases = set(args.case)
    unknown_sources = sorted(selected_sources - set(source_by_id))
    known_cases = {case.case_id for case in manifest.cases}
    unknown_cases = sorted(selected_cases - known_cases)
    if unknown_sources or unknown_cases:
        messages = []
        if unknown_sources:
            messages.append("unknown source IDs: " + ", ".join(unknown_sources))
        if unknown_cases:
            messages.append("unknown case IDs: " + ", ".join(unknown_cases))
        raise SystemExit("; ".join(messages))
    cases = [
        case
        for case in manifest.cases
        if (not selected_sources or case.source_id in selected_sources)
        and (not selected_cases or case.case_id in selected_cases)
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    resolved_by_id = _existing_cases(args.output_root)
    source_ids = sorted({case.source_id for case in cases})
    source_evidence = [
        _dataset_metadata(source_by_id[source_id]) for source_id in source_ids
    ]
    failures: list[dict[str, str]] = []
    for case in cases:
        existing = resolved_by_id.get(case.case_id)
        existing_path = (
            args.output_root / str(existing.get("path"))
            if isinstance(existing, dict)
            else None
        )
        if (
            not args.force
            and isinstance(existing, dict)
            and existing_path is not None
            and existing_path.is_file()
            and existing.get("sha256") == _sha256(existing_path)
            and _cached_case_matches_acquisition(
                existing,
                source=source_by_id[case.source_id],
                case=case,
            )
        ):
            _refresh_existing_case(
                existing,
                source=source_by_id[case.source_id],
                case=case,
                output_path=existing_path,
            )
            print(f"reuse {case.case_id}: {existing_path}")
            continue
        print(f"build {case.case_id}", flush=True)
        try:
            resolved_by_id[case.case_id] = _build_case(
                manifest=manifest,
                source=source_by_id[case.source_id],
                case=case,
                output_root=args.output_root,
            )
        except (GlobalSampleLibraryError, OSError, subprocess.SubprocessError) as exc:
            failures.append({"id": case.case_id, "error": str(exc)})
            print(f"failed {case.case_id}: {exc}", file=sys.stderr, flush=True)
    all_source_evidence = [
        {
            "dataset": source.dataset,
            "revision": source.revision,
            "license": source.license,
            "homepage": source.homepage,
            "attribution": source.attribution,
        }
        for source in manifest.sources
        if source.source_id in {row.get("sourceId") for row in resolved_by_id.values()}
    ]
    attribution_path = _write_attribution(
        args.output_root,
        all_source_evidence or source_evidence,
    )
    resolved = {
        "schemaVersion": "1.0.0",
        "libraryId": manifest.library_id,
        "sourceManifest": str(args.manifest.resolve()),
        "maxDurationSeconds": manifest.max_duration_seconds,
        "randomSeed": manifest.random_seed,
        "coveragePlan": coverage_summary(manifest),
        "sources": all_source_evidence or source_evidence,
        "cases": [
            resolved_by_id[case_id] for case_id in sorted(resolved_by_id)
        ],
        "failedCases": failures,
        "attributionPath": str(attribution_path.relative_to(args.output_root)),
    }
    resolved_path = args.output_root / RESOLVED_NAME
    resolved_path.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "resolvedManifest": str(resolved_path),
                "builtCases": len(resolved_by_id),
                "failedCases": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
