"""Download and normalize the versioned global speech sample matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
        "sourceAssetType": "audio/wav",
    }


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
    value = _request_json(
        f"https://datasets-server.huggingface.co/rows?{params}",
        timeout=180.0,
    )
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
    return _request_bytes(audio_url, timeout=180.0), _case_row_metadata(
        row,
        source=source,
        case=case,
    )


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
    if case.acquisition["kind"] == "hf-viewer-row":
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
