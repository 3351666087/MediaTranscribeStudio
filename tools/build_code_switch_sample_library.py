"""Build short, provenance-bound real and synthetic code-switching samples."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_global_real_diarization import select_diarization_window
from tools.global_sample_library import GlobalSampleLibraryError


DEFAULT_MANIFEST = ROOT / "sample_library" / "code-switch-manifest.v1.json"
DEFAULT_OUTPUT = ROOT / ".runtime_cache" / "sample-library" / "code-switch"
RESOLVED_NAME = "code-switch-sample-library.resolved.v1.json"
RESOLVED_SCHEMA_VERSION = "1.1.0"
SOURCE_EVIDENCE_SCHEMA_VERSION = "1.0.0"
USER_AGENT = "MediaTranscribeStudio-code-switch-samples/1.0"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")
_LIVA_TURN = re.compile(
    r"^Speaker (?P<speaker>[0-9]+) "
    r"\[(?P<start>[0-9:.]+) - (?P<end>[0-9:.]+)\]: "
    r"(?P<text>.*?)(?=^Speaker [0-9]+ \[|\Z)",
    re.MULTILINE | re.DOTALL,
)
_TAGGED_TRANSCRIPT = re.compile(
    r"<(?P<language>[a-z]{2,3})><start:(?P<start>[0-9.]+)>"
    r"(?P<text>.*?)<end:(?P<end>[0-9.]+)>",
    re.DOTALL,
)
_ALLOWED_LICENSES = frozenset({"cc-by-4.0", "cc-by-sa-4.0", "apache-2.0"})
_ALLOWED_KINDS = frozenset({"whole-utterance", "speaker-window", "switch-window"})
_ALLOWED_SPLITS = frozenset({"development", "regression", "held-out"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_bytes(url: str, *, timeout: float = 240.0, attempts: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
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


def _request_json(url: str, *, timeout: float = 240.0) -> dict[str, Any]:
    try:
        value = json.loads(_request_bytes(url, timeout=timeout))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalSampleLibraryError(f"remote JSON is invalid: {url}") from exc
    if not isinstance(value, dict):
        raise GlobalSampleLibraryError(f"remote JSON is not an object: {url}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GlobalSampleLibraryError(f"{field} must be non-empty text")
    return value.strip()


def _identifier(value: Any, field: str) -> str:
    text = _text(value, field)
    if not _ID.fullmatch(text):
        raise GlobalSampleLibraryError(f"{field} is not a safe identifier")
    return text


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalSampleLibraryError(f"cannot read code-switch manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise GlobalSampleLibraryError("code-switch manifest must be an object")
    required = {
        "schemaVersion",
        "libraryId",
        "maxDurationSeconds",
        "generatedRoot",
        "sources",
        "cases",
    }
    if set(value) != required or value.get("schemaVersion") != "1.0.0":
        raise GlobalSampleLibraryError("code-switch manifest fields or version are invalid")
    _identifier(value["libraryId"], "libraryId")
    maximum = value["maxDurationSeconds"]
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise GlobalSampleLibraryError("maxDurationSeconds must be numeric")
    if not 10 <= float(maximum) <= 90:
        raise GlobalSampleLibraryError("maxDurationSeconds must be between 10 and 90")
    generated = Path(_text(value["generatedRoot"], "generatedRoot"))
    if generated.is_absolute() or ".." in generated.parts:
        raise GlobalSampleLibraryError("generatedRoot must be a safe relative path")

    sources = value["sources"]
    cases = value["cases"]
    if not isinstance(sources, list) or not sources:
        raise GlobalSampleLibraryError("sources must be a non-empty array")
    if not isinstance(cases, list) or not cases:
        raise GlobalSampleLibraryError("cases must be a non-empty array")
    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise GlobalSampleLibraryError(f"sources[{index}] must be an object")
        expected = {
            "id",
            "dataset",
            "revision",
            "license",
            "homepage",
            "attribution",
            "recordingType",
        }
        if set(source) != expected:
            raise GlobalSampleLibraryError(f"sources[{index}] fields are invalid")
        source_id = _identifier(source["id"], f"sources[{index}].id")
        if source_id in source_ids:
            raise GlobalSampleLibraryError("source IDs must be unique")
        source_ids.add(source_id)
        dataset = _text(source["dataset"], f"sources[{index}].dataset")
        if dataset.count("/") != 1:
            raise GlobalSampleLibraryError(f"sources[{index}].dataset is invalid")
        if source["homepage"] != f"https://huggingface.co/datasets/{dataset}":
            raise GlobalSampleLibraryError(f"sources[{index}].homepage is invalid")
        if not _REVISION.fullmatch(_text(source["revision"], "revision")):
            raise GlobalSampleLibraryError(f"sources[{index}].revision is invalid")
        if source["license"] not in _ALLOWED_LICENSES:
            raise GlobalSampleLibraryError(f"sources[{index}].license is not approved")
        if source["recordingType"] not in {"real-recording", "synthetic-mixture"}:
            raise GlobalSampleLibraryError(
                f"sources[{index}].recordingType is invalid"
            )
        _text(source["attribution"], f"sources[{index}].attribution")

    case_ids: set[str] = set()
    covered_splits: set[str] = set()
    recording_kinds: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise GlobalSampleLibraryError(f"cases[{index}] must be an object")
        case_id = _identifier(case.get("id"), f"cases[{index}].id")
        if case_id in case_ids:
            raise GlobalSampleLibraryError("case IDs must be unique")
        case_ids.add(case_id)
        if case.get("sourceId") not in source_ids:
            raise GlobalSampleLibraryError(f"cases[{index}].sourceId is not declared")
        case_maximum = case.get("maxDurationSeconds", maximum)
        if (
            isinstance(case_maximum, bool)
            or not isinstance(case_maximum, (int, float))
            or not 10 <= float(case_maximum) <= float(maximum)
        ):
            raise GlobalSampleLibraryError(
                f"cases[{index}].maxDurationSeconds must be between 10 and "
                "the manifest maximum"
            )
        acquisition = case.get("acquisition")
        if not isinstance(acquisition, dict) or set(acquisition) != {
            "kind",
            "config",
            "split",
            "rowIndex",
        }:
            raise GlobalSampleLibraryError(f"cases[{index}].acquisition is invalid")
        kind = acquisition.get("kind")
        if kind not in _ALLOWED_KINDS:
            raise GlobalSampleLibraryError(f"cases[{index}] acquisition kind is invalid")
        recording_kinds.add(str(kind))
        _text(acquisition.get("config"), f"cases[{index}].acquisition.config")
        _text(acquisition.get("split"), f"cases[{index}].acquisition.split")
        row_index = acquisition.get("rowIndex")
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
            raise GlobalSampleLibraryError(f"cases[{index}].rowIndex is invalid")
        evaluation_split = case.get("evaluationSplit")
        if evaluation_split not in _ALLOWED_SPLITS:
            raise GlobalSampleLibraryError(f"cases[{index}].evaluationSplit is invalid")
        covered_splits.add(str(evaluation_split))
        languages = case.get("expectedLanguages")
        if (
            not isinstance(languages, list)
            or len(languages) < 2
            or len(set(languages)) != len(languages)
            or any(not isinstance(item, str) or not _LANGUAGE.fullmatch(item) for item in languages)
        ):
            raise GlobalSampleLibraryError(
                f"cases[{index}].expectedLanguages must contain unique primary tags"
            )
        scenarios = case.get("scenario")
        if (
            not isinstance(scenarios, list)
            or not scenarios
            or any(not isinstance(item, str) or not _ID.fullmatch(item) for item in scenarios)
        ):
            raise GlobalSampleLibraryError(f"cases[{index}].scenario is invalid")
        _text(case.get("region"), f"cases[{index}].region")
        _text(case.get("transcriptField"), f"cases[{index}].transcriptField")
        if kind == "speaker-window":
            target = case.get("targetSpeakerCount")
            if isinstance(target, bool) or not isinstance(target, int) or target < 2:
                raise GlobalSampleLibraryError(
                    f"cases[{index}].targetSpeakerCount is invalid"
                )
        if kind == "switch-window":
            switch_index = case.get("switchIndex")
            if (
                isinstance(switch_index, bool)
                or not isinstance(switch_index, int)
                or switch_index < 0
            ):
                raise GlobalSampleLibraryError(f"cases[{index}].switchIndex is invalid")
    if covered_splits != _ALLOWED_SPLITS:
        raise GlobalSampleLibraryError("all evaluation splits must be covered")
    if recording_kinds != _ALLOWED_KINDS:
        raise GlobalSampleLibraryError("all acquisition kinds must be covered")
    return value


def _dataset_evidence(source: Mapping[str, Any]) -> dict[str, Any]:
    value = _request_json(f"https://huggingface.co/api/datasets/{source['dataset']}")
    if value.get("sha") != source["revision"]:
        raise GlobalSampleLibraryError(
            f"{source['dataset']} moved from pinned revision {source['revision']}"
        )
    if value.get("private") is not False or value.get("gated") not in {False, None}:
        raise GlobalSampleLibraryError(f"{source['dataset']} is no longer public")
    tags = value.get("tags", [])
    licenses = {
        item.split(":", 1)[1].casefold()
        for item in tags
        if isinstance(item, str) and item.startswith("license:")
    }
    card = value.get("cardData")
    if isinstance(card, dict) and isinstance(card.get("license"), str):
        licenses.add(card["license"].casefold())
    if source["license"] not in licenses:
        raise GlobalSampleLibraryError(
            f"{source['dataset']} no longer declares {source['license']}"
        )
    last_modified = value.get("lastModified")
    if not isinstance(last_modified, str) or not last_modified.strip():
        raise GlobalSampleLibraryError(
            f"{source['dataset']} does not expose a last-modified timestamp"
        )
    return {
        key: source[key]
        for key in (
            "id",
            "dataset",
            "revision",
            "license",
            "homepage",
            "attribution",
            "recordingType",
        )
    } | {"lastModified": last_modified}


def _source_evidence_path(output_root: Path, source_id: str) -> Path:
    return output_root / "source-evidence" / f"{source_id}.json"


def _validate_dataset_evidence(
    value: Any,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        key: source[key]
        for key in (
            "id",
            "dataset",
            "revision",
            "license",
            "homepage",
            "attribution",
            "recordingType",
        )
    }
    if (
        not isinstance(value, dict)
        or set(value) != set(expected) | {"lastModified"}
        or any(value.get(key) != item for key, item in expected.items())
        or not isinstance(value.get("lastModified"), str)
        or not value["lastModified"].strip()
    ):
        raise GlobalSampleLibraryError(
            f"{source['id']} cached dataset evidence does not match the pinned source"
        )
    return dict(value)


def _load_cached_dataset_evidence(
    output_root: Path,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    path = _source_evidence_path(output_root, str(source["id"]))
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalSampleLibraryError(
            f"{source['id']} cached dataset evidence is unreadable"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"schemaVersion", "verifiedAt", "source"}
        or value.get("schemaVersion") != SOURCE_EVIDENCE_SCHEMA_VERSION
        or not isinstance(value.get("verifiedAt"), str)
        or not value["verifiedAt"].strip()
    ):
        raise GlobalSampleLibraryError(
            f"{source['id']} cached dataset evidence is invalid"
        )
    return _validate_dataset_evidence(value.get("source"), source=source), path


def _write_dataset_evidence(
    output_root: Path,
    *,
    source: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> Path:
    path = _source_evidence_path(output_root, str(source["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": SOURCE_EVIDENCE_SCHEMA_VERSION,
        "verifiedAt": datetime.now(UTC).isoformat(),
        "source": _validate_dataset_evidence(evidence, source=source),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _resolve_dataset_evidence(
    output_root: Path,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    cached = _load_cached_dataset_evidence(output_root, source=source)
    if cached is None:
        evidence = _dataset_evidence(source)
        path = _write_dataset_evidence(
            output_root,
            source=source,
            evidence=evidence,
        )
    else:
        evidence, path = cached
    return evidence | {
        "evidencePath": str(path.relative_to(output_root)),
        "evidenceSha256": _sha256(path),
    }


def _viewer_row(source: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    acquisition = case["acquisition"]
    params = urllib.parse.urlencode(
        {
            "dataset": source["dataset"],
            "config": acquisition["config"],
            "split": acquisition["split"],
            "offset": acquisition["rowIndex"],
            "length": 1,
        }
    )
    response = _request_json(
        f"https://datasets-server.huggingface.co/rows?{params}",
        timeout=300.0,
    )
    rows = response.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise GlobalSampleLibraryError(f"{case['id']} did not resolve exactly one row")
    wrapped = rows[0]
    if (
        not isinstance(wrapped, dict)
        or wrapped.get("row_idx") != acquisition["rowIndex"]
        or not isinstance(wrapped.get("row"), dict)
    ):
        raise GlobalSampleLibraryError(f"{case['id']} row identity is invalid")
    return wrapped["row"]


def _audio_url(
    row: Mapping[str, Any],
    *,
    case_id: str,
    revision: str,
) -> str:
    audio = row.get("audio")
    if (
        not isinstance(audio, list)
        or len(audio) != 1
        or not isinstance(audio[0], dict)
        or not isinstance(audio[0].get("src"), str)
    ):
        raise GlobalSampleLibraryError(f"{case_id}.audio is missing")
    url = audio[0]["src"]
    if revision not in urllib.parse.unquote(url):
        raise GlobalSampleLibraryError(f"{case_id}.audio is not revision-bound")
    return url


def _timestamp_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise GlobalSampleLibraryError(f"invalid timestamp: {value}")
    hours, minutes, seconds = parts
    result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if result < 0:
        raise GlobalSampleLibraryError(f"negative timestamp: {value}")
    return result


def parse_liva_turns_with_issues(
    transcript: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    turns: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for match in _LIVA_TURN.finditer(transcript.strip()):
        start = _timestamp_seconds(match.group("start"))
        end = _timestamp_seconds(match.group("end"))
        text = " ".join(match.group("text").strip().split())
        if end < start or not text:
            raise GlobalSampleLibraryError("Liva transcript contains an invalid turn")
        if end == start:
            issues.append(
                {
                    "reasonCode": "ZERO_DURATION_SOURCE_ANNOTATION",
                    "speakerId": f"source-speaker-{int(match.group('speaker'))}",
                    "sourceTimestampSeconds": round(start, 6),
                    "transcript": text,
                }
            )
            continue
        turns.append(
            {
                "speakerId": f"source-speaker-{int(match.group('speaker'))}",
                "startSeconds": round(start, 6),
                "endSeconds": round(end, 6),
                "transcript": text,
            }
        )
    if not turns:
        raise GlobalSampleLibraryError("Liva transcript contains no parseable turns")
    return turns, issues


def parse_liva_turns(transcript: str) -> list[dict[str, Any]]:
    turns, _ = parse_liva_turns_with_issues(transcript)
    return turns


def parse_tagged_transcript(transcript: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for match in _TAGGED_TRANSCRIPT.finditer(transcript):
        start = float(match.group("start"))
        end = float(match.group("end"))
        text = " ".join(match.group("text").strip().split())
        if end <= start or not text:
            raise GlobalSampleLibraryError("tagged transcript contains an invalid chunk")
        chunks.append(
            {
                "language": match.group("language"),
                "startSeconds": round(start, 6),
                "endSeconds": round(end, 6),
                "transcript": text,
            }
        )
    if not chunks:
        raise GlobalSampleLibraryError("tagged transcript contains no chunks")
    for previous, current in zip(chunks, chunks[1:]):
        if abs(previous["endSeconds"] - current["startSeconds"]) > 0.02:
            raise GlobalSampleLibraryError("tagged transcript is not contiguous")
    return chunks


def select_switch_window(
    chunks: Sequence[Mapping[str, Any]],
    *,
    switch_index: int,
    maximum_seconds: float,
) -> dict[str, Any]:
    switches = [
        index
        for index in range(len(chunks) - 1)
        if chunks[index]["language"] != chunks[index + 1]["language"]
    ]
    if switch_index >= len(switches):
        raise GlobalSampleLibraryError("requested switch index is unavailable")
    boundary_index = switches[switch_index]
    best: tuple[tuple[float, float, float], tuple[int, int]] | None = None
    for left in range(boundary_index + 1):
        for right in range(boundary_index + 1, len(chunks)):
            start = float(chunks[left]["startSeconds"])
            end = float(chunks[right]["endSeconds"])
            duration = end - start
            if duration <= 0 or duration > maximum_seconds + 1e-6:
                continue
            switch_time = float(chunks[boundary_index]["endSeconds"])
            balance = min(switch_time - start, end - switch_time)
            score = (balance, duration, -start)
            if best is None or score > best[0]:
                best = (score, (left, right))
    if best is None:
        raise GlobalSampleLibraryError("no bounded full-chunk switch window is available")
    left, right = best[1]
    selected = [dict(item) for item in chunks[left : right + 1]]
    start = float(selected[0]["startSeconds"])
    end = float(selected[-1]["endSeconds"])
    relative = [
        {
            **item,
            "sourceStartSeconds": item["startSeconds"],
            "sourceEndSeconds": item["endSeconds"],
            "startSeconds": round(float(item["startSeconds"]) - start, 6),
            "endSeconds": round(float(item["endSeconds"]) - start, 6),
        }
        for item in selected
    ]
    switch_points = [
        round(float(first["endSeconds"]), 6)
        for first, second in zip(relative, relative[1:])
        if first["language"] != second["language"]
    ]
    return {
        "algorithm": "full-chunk-balanced-around-reference-switch-v1",
        "sourceStartSeconds": round(start, 6),
        "sourceEndSeconds": round(end, 6),
        "durationSeconds": round(end - start, 6),
        "referenceSwitchIndex": switch_index,
        "chunks": relative,
        "switchPointsSeconds": switch_points,
    }


def _ffmpeg_audio(
    source: Path,
    output: Path,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds is not None:
        command.extend(["-ss", f"{start_seconds:.6f}"])
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.6f}"])
    command.extend(
        [
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
        ]
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=360,
    )
    if completed.returncode != 0:
        raise GlobalSampleLibraryError(
            f"ffmpeg failed for {output.name}: {completed.stderr.strip()}"
        )


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    value = json.loads(completed.stdout)
    stream = value["streams"][0]
    return {
        "codec": stream["codec_name"],
        "sampleRate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "durationSeconds": round(float(value["format"]["duration"]), 6),
    }


def _source_row_path(output_root: Path, case_id: str) -> Path:
    return output_root / "source-metadata" / f"{case_id}.json"


def _load_cached_source_row(
    output_root: Path,
    case_id: str,
    *,
    source: Mapping[str, Any],
    case: Mapping[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    path = _source_row_path(output_root, case_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalSampleLibraryError(
            f"{case_id} cached source metadata is unreadable"
        ) from exc
    expected = {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "config": case["acquisition"]["config"],
        "split": case["acquisition"]["split"],
        "rowIndex": case["acquisition"]["rowIndex"],
    }
    if (
        not isinstance(value, dict)
        or any(value.get(key) != item for key, item in expected.items())
        or not isinstance(value.get("row"), dict)
    ):
        raise GlobalSampleLibraryError(
            f"{case_id} cached source metadata does not match the pinned row"
        )
    row = dict(value["row"])
    audio_asset = row.pop("audioAsset", None)
    if isinstance(audio_asset, str) and audio_asset:
        row["audio"] = [{"src": audio_asset}]
    return row, path


def _write_source_row(
    output_root: Path,
    case_id: str,
    *,
    source: Mapping[str, Any],
    case: Mapping[str, Any],
    row: Mapping[str, Any],
) -> Path:
    path = _source_row_path(output_root, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = row.get("audio")
    serializable = {
        key: value
        for key, value in row.items()
        if key != "audio"
    }
    serializable["audioAsset"] = (
        audio[0].get("src", "").split("?", 1)[0]
        if isinstance(audio, list) and audio and isinstance(audio[0], dict)
        else None
    )
    payload = {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "config": case["acquisition"]["config"],
        "split": case["acquisition"]["split"],
        "rowIndex": case["acquisition"]["rowIndex"],
        "row": serializable,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _base_case(
    case: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    output_root: Path,
    output: Path,
    source_audio: Path,
    source_row_path: Path,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "sourceId": case["sourceId"],
        "sourceDataset": source["dataset"],
        "sourceRevision": source["revision"],
        "sourceRowIndex": case["acquisition"]["rowIndex"],
        "sourceLocator": {
            "url": (
                f"https://huggingface.co/datasets/{source['dataset']}/tree/"
                f"{source['revision']}"
            ),
            "config": case["acquisition"]["config"],
            "split": case["acquisition"]["split"],
            "rowIndex": case["acquisition"]["rowIndex"],
        },
        "sourceArtifactPath": str(source_audio.relative_to(output_root)),
        "sourceArtifactSha256": _sha256(source_audio),
        "sourceMetadataPath": str(source_row_path.relative_to(output_root)),
        "sourceMetadataSha256": _sha256(source_row_path),
        "path": str(output.relative_to(output_root)),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "audio": dict(probe),
        "realOrSynthetic": source["recordingType"],
        "language": "mul",
        "expectedLanguages": list(case["expectedLanguages"]),
        "region": case["region"],
        "evaluationSplit": case["evaluationSplit"],
        "scenario": list(case["scenario"]),
    }


def _build_case(
    case: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    output_root: Path,
    maximum_seconds: float,
) -> dict[str, Any]:
    case_id = str(case["id"])
    cached = _load_cached_source_row(
        output_root,
        case_id,
        source=source,
        case=case,
    )
    if cached is None:
        row = _viewer_row(source, case)
        source_row_path = _write_source_row(
            output_root,
            case_id,
            source=source,
            case=case,
            row=row,
        )
    else:
        row, source_row_path = cached
    source_audio = output_root / "sources" / f"{case_id}.source"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    if not source_audio.is_file():
        source_audio.write_bytes(
            _request_bytes(
                _audio_url(row, case_id=case_id, revision=str(source["revision"])),
                timeout=600.0,
            )
        )
    output = output_root / "audio" / f"{case_id}.wav"
    kind = case["acquisition"]["kind"]
    case_maximum_seconds = float(
        case.get("maxDurationSeconds", maximum_seconds)
    )
    transcript_field = str(case["transcriptField"])
    transcript = row.get(transcript_field)
    if not isinstance(transcript, str) or not transcript.strip():
        raise GlobalSampleLibraryError(f"{case_id}.{transcript_field} is missing")

    if kind == "whole-utterance":
        _ffmpeg_audio(source_audio, output)
        expected_transcript = transcript.strip()
        language_truth: dict[str, Any] = {
            "qualification": "document-language-pair-without-time-alignment",
            "expectedLanguages": list(case["expectedLanguages"]),
            "timeScoringEligible": False,
            "wordScoringEligible": False,
        }
        switch_count_field = case.get("switchCountField")
        if isinstance(switch_count_field, str):
            raw_count = row.get(switch_count_field)
            try:
                language_truth["expectedSwitchCount"] = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise GlobalSampleLibraryError(
                    f"{case_id}.{switch_count_field} is invalid"
                ) from exc
        main_field = case.get("mainLanguageField")
        level_field = case.get("switchLevelField")
        if isinstance(main_field, str):
            language_truth["mainLanguage"] = row.get(main_field)
        if isinstance(level_field, str):
            language_truth["switchLevel"] = row.get(level_field)
        truth_eligibility = {
            "speakerCount": False,
            "turnBoundaries": False,
            "overlap": False,
            "derJer": False,
            "asr": True,
            "languageDocumentPair": True,
            "languageTiming": False,
            "languageWords": False,
        }
        extra: dict[str, Any] = {
            "expectedSpeakerCount": None,
            "expectedTranscript": expected_transcript,
            "scoringTranscript": expected_transcript,
            "languageTruth": language_truth,
            "truthEligibility": truth_eligibility,
        }
    elif kind == "speaker-window":
        turns, annotation_issues = parse_liva_turns_with_issues(transcript)
        window = select_diarization_window(turns, int(case["targetSpeakerCount"]))
        _ffmpeg_audio(
            source_audio,
            output,
            start_seconds=float(window["sourceStartSeconds"]),
            duration_seconds=float(window["durationSeconds"]),
        )
        extra = {
            "expectedSpeakerCount": len(window["speakerSet"]),
            "speakerSet": window["speakerSet"],
            "turns": window["turns"],
            "overlapIntervals": window["overlapIntervals"],
            "windowSelection": {
                key: value
                for key, value in window.items()
                if key not in {"turns", "overlapIntervals"}
            },
            "sourceAnnotationIssues": annotation_issues,
            "languageTruth": {
                "qualification": (
                    "document-language-pair-with-speaker-turns-without-"
                    "per-turn-language-labels"
                ),
                "expectedLanguages": list(case["expectedLanguages"]),
                "timeScoringEligible": False,
                "wordScoringEligible": False,
            },
            "truthEligibility": {
                "speakerCount": True,
                "turnBoundaries": True,
                "overlap": True,
                "derJer": True,
                "asr": False,
                "languageDocumentPair": True,
                "languageTiming": False,
                "languageWords": False,
            },
        }
    else:
        chunks = parse_tagged_transcript(transcript)
        window = select_switch_window(
            chunks,
            switch_index=int(case["switchIndex"]),
            maximum_seconds=case_maximum_seconds,
        )
        _ffmpeg_audio(
            source_audio,
            output,
            start_seconds=float(window["sourceStartSeconds"]),
            duration_seconds=float(window["durationSeconds"]),
        )
        expected_transcript = " ".join(
            str(chunk["transcript"]) for chunk in window["chunks"]
        )
        extra = {
            "expectedSpeakerCount": None,
            "expectedTranscript": expected_transcript,
            "scoringTranscript": expected_transcript,
            "windowSelection": {
                key: value
                for key, value in window.items()
                if key != "chunks"
            },
            "languageTruth": {
                "qualification": "synthetic-concatenation-exact-chunk-timestamps",
                "expectedLanguages": list(case["expectedLanguages"]),
                "timeScoringEligible": True,
                "wordScoringEligible": True,
                "intervals": [
                    {
                        "language": chunk["language"],
                        "startSeconds": chunk["startSeconds"],
                        "endSeconds": chunk["endSeconds"],
                        "transcript": chunk["transcript"],
                    }
                    for chunk in window["chunks"]
                ],
                "switchPointsSeconds": window["switchPointsSeconds"],
            },
            "truthEligibility": {
                "speakerCount": False,
                "turnBoundaries": False,
                "overlap": False,
                "derJer": False,
                "asr": True,
                "languageDocumentPair": True,
                "languageTiming": True,
                "languageWords": True,
            },
        }
    probe = _probe(output)
    if (
        probe["codec"] != "pcm_s16le"
        or probe["sampleRate"] != 16_000
        or probe["channels"] != 1
        or not 0 < probe["durationSeconds"] <= case_maximum_seconds + 0.05
    ):
        raise GlobalSampleLibraryError(f"{case_id} normalized audio is invalid")
    return _base_case(
        case,
        source,
        output_root=output_root,
        output=output,
        source_audio=source_audio,
        source_row_path=source_row_path,
        probe=probe,
    ) | extra


def build_library(manifest_path: Path, output_root: Path) -> Path:
    manifest = load_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    source_by_id = {source["id"]: source for source in manifest["sources"]}
    sources = [
        _resolve_dataset_evidence(output_root, source=source)
        for source in manifest["sources"]
    ]
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in manifest["cases"]:
        print(f"build {case['id']}", flush=True)
        try:
            cases.append(
                _build_case(
                    case,
                    source_by_id[case["sourceId"]],
                    output_root=output_root,
                    maximum_seconds=float(manifest["maxDurationSeconds"]),
                )
            )
        except (GlobalSampleLibraryError, OSError, subprocess.SubprocessError) as exc:
            failures.append({"id": str(case["id"]), "error": str(exc)})
            print(f"failed {case['id']}: {exc}", flush=True)
    attribution = output_root / "ATTRIBUTION.md"
    lines = [
        "# Code-Switching Sample Attribution",
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
                f"- Recording type: `{source['recordingType']}`",
                f"- Attribution: {source['attribution']}",
                "",
            ]
        )
    attribution.write_text("\n".join(lines), encoding="utf-8")
    resolved = {
        "schemaVersion": RESOLVED_SCHEMA_VERSION,
        "libraryId": manifest["libraryId"],
        "generatedAt": datetime.now(UTC).isoformat(),
        "sourceManifest": str(manifest_path.resolve()),
        "sourceManifestSha256": _sha256(manifest_path),
        "maximumDurationSeconds": manifest["maxDurationSeconds"],
        "sources": sources,
        "cases": cases,
        "failedCases": failures,
        "coverage": {
            "caseCount": len(cases),
            "realRecordingCount": sum(
                case["realOrSynthetic"] == "real-recording" for case in cases
            ),
            "syntheticMixtureCount": sum(
                case["realOrSynthetic"] == "synthetic-mixture" for case in cases
            ),
            "languages": sorted(
                {
                    language
                    for case in cases
                    for language in case["expectedLanguages"]
                }
            ),
            "regions": sorted({case["region"] for case in cases}),
            "speakerCounts": sorted(
                {
                    case["expectedSpeakerCount"]
                    for case in cases
                    if isinstance(case.get("expectedSpeakerCount"), int)
                }
            ),
            "timeScoredCases": sum(
                case["truthEligibility"]["languageTiming"] is True for case in cases
            ),
            "wordScoredCases": sum(
                case["truthEligibility"]["languageWords"] is True for case in cases
            ),
        },
        "attributionPath": str(attribution.relative_to(output_root)),
    }
    destination = output_root / RESOLVED_NAME
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_root,
        prefix=f".{RESOLVED_NAME}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(resolved, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = build_library(args.manifest, args.output_root)
    resolved = json.loads(destination.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "resolvedManifest": str(destination),
                "coverage": resolved["coverage"],
                "failedCases": resolved["failedCases"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if resolved["failedCases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
