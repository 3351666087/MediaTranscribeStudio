"""Run a pinned MOSS transcription challenge inside a bounded local worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / ".runtime_cache"
    / "production-models"
    / "moss-transcribe-diarize-0.9b"
)
DEFAULT_MODEL_SOURCE = (
    PROJECT_ROOT / ".runtime_cache" / "model-sources" / "moss-transcribe-diarize"
)
DEFAULT_MODEL_AUDIT = (
    PROJECT_ROOT
    / ".runtime_cache"
    / "model-audits"
    / "moss-transcribe-diarize-0.9b-e5118b4.audit.v1.json"
)
DEFAULT_WORKER_PYTHON = (
    Path.home()
    / "Library"
    / "Application Support"
    / "MediaTranscribeStudio"
    / "venvs"
    / "moss-transcribe-diarize-0.9b"
    / "bin"
    / "python"
)
EXPECTED_REPO_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
EXPECTED_REVISION = "e5118b411bf5a77d7a90c4941066bec93c967312"
EXPECTED_MANIFEST_SHA256 = (
    "89533dc712a143a1b0219daed6202485ee1020fab9172f449de21a2a702243fc"
)
EXPECTED_WEIGHT_SHA256 = (
    "9a0ceb4ab7330357db3ff583dba8d83625d5b733b00e1d55d6970e11b07026c4"
)
EXPECTED_RUNTIME_VERSIONS = {
    "av": "18.0.0",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
}
SCHEMA_VERSION = "1.0.0"
MAX_EVENT_BYTES = 1024 * 1024
_SPEAKER_PATTERN = re.compile(r"^S[0-9]{2,}$")
_MEMORY_PRESSURE_PATTERN = re.compile(
    r"System-wide memory free percentage:\s*([0-9]+)%"
)
_FOOTPRINT_PATTERN = re.compile(
    r"phys_footprint:\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B)",
    re.IGNORECASE,
)


class ChallengeError(RuntimeError):
    """Raised when a challenge cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class MediaIdentity:
    path: Path
    sha256: str
    bytes: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class SupervisionLimits:
    idle_timeout_seconds: float
    hard_deadline_seconds: float
    sample_interval_seconds: float
    footprint_interval_seconds: float
    max_rss_mb: float
    max_footprint_mb: float
    minimum_system_free_percent: int


@dataclass(frozen=True, slots=True)
class OutputLimits:
    max_new_tokens: int
    max_segments: int
    max_text_characters: int
    timestamp_tolerance_seconds: float


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    elapsed_seconds: float
    process_count: int | None
    rss_mb: float | None
    footprint_mb: float | None
    system_free_percent: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsedSeconds": round(self.elapsed_seconds, 6),
            "processCount": self.process_count,
            "rssMb": (round(self.rss_mb, 6) if self.rss_mb is not None else None),
            "footprintMb": (
                round(self.footprint_mb, 6) if self.footprint_mb is not None else None
            ),
            "systemFreePercent": self.system_free_percent,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChallengeError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ChallengeError(f"{label} root must be an object: {path}")
    return value


def verify_model_snapshot(
    *,
    model_path: Path,
    audit_path: Path,
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    model_root = model_path.resolve(strict=True)
    if not model_root.is_dir():
        raise ChallengeError("MOSS model path must be a directory")
    audit = _load_json_object(audit_path.resolve(strict=True), "model audit")
    if (
        audit.get("schemaVersion") != SCHEMA_VERSION
        or audit.get("repoId") != EXPECTED_REPO_ID
        or audit.get("revision") != EXPECTED_REVISION
        or audit.get("manifestSha256") != EXPECTED_MANIFEST_SHA256
    ):
        raise ChallengeError("MOSS audit does not match the pinned model identity")
    declared_directory = audit.get("localDirectory")
    if (
        not isinstance(declared_directory, str)
        or Path(declared_directory).resolve(strict=True) != model_root
    ):
        raise ChallengeError("MOSS audit localDirectory does not match model path")
    body = dict(audit)
    declared_manifest = body.pop("manifestSha256", None)
    if _sha256_bytes(_canonical_json(body)) != declared_manifest:
        raise ChallengeError("MOSS audit manifest hash is invalid")
    files = audit.get("files")
    if not isinstance(files, list) or audit.get("fileCount") != len(files) or not files:
        raise ChallengeError("MOSS audit file inventory is invalid")
    observed_total = 0
    observed_weight = None
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            raise ChallengeError(f"MOSS audit file {index} is invalid")
        relative = item.get("path")
        size = item.get("size")
        expected_sha256 = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ChallengeError(f"MOSS audit file {index} metadata is invalid")
        candidate = model_root.joinpath(*Path(relative).parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise ChallengeError(f"MOSS snapshot file is missing or unsafe: {relative}")
        actual_size = candidate.stat().st_size
        if actual_size != size:
            raise ChallengeError(f"MOSS snapshot size mismatch: {relative}")
        if verify_file_hashes and _sha256_file(candidate) != expected_sha256:
            raise ChallengeError(f"MOSS snapshot hash mismatch: {relative}")
        observed_total += actual_size
        if relative == "model-00000-of-00001.safetensors":
            observed_weight = expected_sha256
    if observed_total != audit.get("totalBytes"):
        raise ChallengeError("MOSS snapshot total byte count is invalid")
    if observed_weight != EXPECTED_WEIGHT_SHA256:
        raise ChallengeError("MOSS weight identity is not the audited checkpoint")
    return {
        "repoId": EXPECTED_REPO_ID,
        "revision": EXPECTED_REVISION,
        "manifestSha256": EXPECTED_MANIFEST_SHA256,
        "weightSha256": EXPECTED_WEIGHT_SHA256,
        "fileCount": len(files),
        "totalBytes": observed_total,
        "fullHashVerified": verify_file_hashes,
    }


def probe_media(path: Path) -> MediaIdentity:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ChallengeError("challenge input must be a regular non-symlink file")
    before = resolved.stat()
    before_sha256 = _sha256_file(resolved)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ChallengeError("ffprobe is required to establish media duration")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(resolved),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise ChallengeError("ffprobe could not decode challenge media")
    try:
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
        streams = payload["streams"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ChallengeError("ffprobe returned incomplete media metadata") from exc
    if (
        not math.isfinite(duration)
        or duration <= 0
        or not isinstance(streams, list)
        or not any(
            isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
            for stream in streams
        )
    ):
        raise ChallengeError("challenge media has no bounded audio stream")
    after = resolved.stat()
    after_sha256 = _sha256_file(resolved)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or before_sha256 != after_sha256:
        raise ChallengeError("challenge media changed during identity probing")
    return MediaIdentity(
        path=resolved,
        sha256=after_sha256,
        bytes=after.st_size,
        duration_seconds=duration,
    )


def worker_python_entry(path: Path) -> Path:
    entry = Path(os.path.abspath(os.path.expanduser(str(path))))
    try:
        target = entry.resolve(strict=True)
    except OSError as exc:
        raise ChallengeError("MOSS worker Python is missing") from exc
    if not target.is_file() or not entry.is_file() or not os.access(entry, os.X_OK):
        raise ChallengeError("MOSS worker Python is not executable")
    return entry


def validate_worker_result(
    payload: Mapping[str, Any],
    *,
    duration_seconds: float,
    limits: OutputLimits,
) -> dict[str, Any]:
    raw_text = payload.get("rawText")
    generated_tokens = payload.get("generatedTokens")
    raw_segments = payload.get("segments")
    runtime_versions = payload.get("runtimeVersions")
    if runtime_versions != EXPECTED_RUNTIME_VERSIONS:
        raise ChallengeError("MOSS worker runtime versions are not pinned")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ChallengeError("MOSS worker returned an empty raw transcript")
    if len(raw_text) > limits.max_text_characters:
        raise ChallengeError("MOSS raw transcript exceeded the text boundary")
    if (
        isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, int)
        or generated_tokens < 1
        or generated_tokens > limits.max_new_tokens
    ):
        raise ChallengeError("MOSS generated token count exceeded its boundary")
    if (
        not isinstance(raw_segments, list)
        or not raw_segments
        or len(raw_segments) > limits.max_segments
    ):
        raise ChallengeError("MOSS segment count exceeded its boundary")
    normalized: list[dict[str, Any]] = []
    previous_start = -1.0
    total_text_characters = 0
    maximum_end_overrun = 0.0
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise ChallengeError(f"MOSS segment {index} is not an object")
        start = segment.get("start")
        end = segment.get("end")
        speaker = segment.get("speaker")
        text = segment.get("text")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
        ):
            raise ChallengeError(f"MOSS segment {index} has invalid timestamps")
        start_value = float(start)
        end_value = float(end)
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0
            or end_value <= start_value
            or start_value < previous_start
        ):
            raise ChallengeError(f"MOSS segment {index} has unsafe timestamps")
        overrun = max(0.0, end_value - duration_seconds)
        if overrun > limits.timestamp_tolerance_seconds:
            raise ChallengeError(
                f"MOSS segment {index} exceeds the media time boundary"
            )
        if not isinstance(speaker, str) or _SPEAKER_PATTERN.fullmatch(speaker) is None:
            raise ChallengeError(f"MOSS segment {index} has invalid speaker syntax")
        if not isinstance(text, str) or not text.strip():
            raise ChallengeError(f"MOSS segment {index} has empty text")
        total_text_characters += len(text)
        if total_text_characters > limits.max_text_characters:
            raise ChallengeError("MOSS segment text exceeded the text boundary")
        normalized.append(
            {
                "segmentId": f"moss-segment-{index + 1:04d}",
                "startMs": round(start_value * 1000),
                "endMs": round(end_value * 1000),
                "speakerId": speaker,
                "rawText": text,
            }
        )
        previous_start = start_value
        maximum_end_overrun = max(maximum_end_overrun, overrun)
    speakers = sorted({segment["speakerId"] for segment in normalized})
    return {
        "segments": normalized,
        "checks": {
            "generatedTokens": generated_tokens,
            "maxNewTokens": limits.max_new_tokens,
            "segmentCount": len(normalized),
            "maxSegments": limits.max_segments,
            "textCharacters": total_text_characters,
            "maxTextCharacters": limits.max_text_characters,
            "speakerCount": len(speakers),
            "speakers": speakers,
            "mediaDurationSeconds": round(duration_seconds, 6),
            "timestampToleranceSeconds": limits.timestamp_tolerance_seconds,
            "maximumEndOverrunSeconds": round(maximum_end_overrun, 6),
            "runtimeVersions": dict(EXPECTED_RUNTIME_VERSIONS),
        },
    }


def _parse_size_mb(value: float, unit: str) -> float:
    factors = {
        "B": 1 / (1024 * 1024),
        "KB": 1 / 1024,
        "MB": 1.0,
        "GB": 1024.0,
        "TB": 1024.0 * 1024.0,
    }
    try:
        return value * factors[unit.upper()]
    except KeyError as exc:
        raise ChallengeError(f"unsupported footprint unit: {unit}") from exc


def parse_footprint_mb(output: str) -> float | None:
    matches = _FOOTPRINT_PATTERN.findall(output)
    if not matches:
        return None
    value, unit = matches[-1]
    return _parse_size_mb(float(value), unit)


def parse_system_free_percent(output: str) -> int | None:
    match = _MEMORY_PRESSURE_PATTERN.search(output)
    return int(match.group(1)) if match is not None else None


def _process_tree_rss(root_pid: int) -> tuple[int | None, float | None]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if result.returncode != 0:
        return None, None
    rows: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) != 3:
            continue
        try:
            pid, ppid, rss_kb = map(int, columns)
        except ValueError:
            continue
        rows[pid] = (ppid, rss_kb)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _rss) in rows.items():
            if pid not in descendants and ppid in descendants:
                descendants.add(pid)
                changed = True
    observed = [rows[pid][1] for pid in descendants if pid in rows]
    if not observed:
        return None, None
    return len(observed), sum(observed) / 1024.0


def sample_resources(
    root_pid: int,
    *,
    elapsed_seconds: float,
    include_footprint: bool,
) -> ResourceSnapshot:
    process_count, rss_mb = _process_tree_rss(root_pid)
    footprint_mb = None
    if include_footprint and Path("/usr/bin/footprint").is_file():
        try:
            result = subprocess.run(
                ["/usr/bin/footprint", "-p", str(root_pid)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                footprint_mb = parse_footprint_mb(result.stdout)
        except (OSError, subprocess.TimeoutExpired):
            footprint_mb = None
    system_free_percent = None
    memory_pressure = shutil.which("memory_pressure")
    if include_footprint and memory_pressure is not None:
        try:
            result = subprocess.run(
                [memory_pressure, "-Q"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                system_free_percent = parse_system_free_percent(result.stdout)
        except (OSError, subprocess.TimeoutExpired):
            system_free_percent = None
    return ResourceSnapshot(
        elapsed_seconds=elapsed_seconds,
        process_count=process_count,
        rss_mb=rss_mb,
        footprint_mb=footprint_mb,
        system_free_percent=system_free_percent,
    )


def _resource_failure(
    snapshot: ResourceSnapshot,
    limits: SupervisionLimits,
) -> str | None:
    if snapshot.rss_mb is not None and snapshot.rss_mb > limits.max_rss_mb:
        return "WORKER_RSS_LIMIT_EXCEEDED"
    if (
        snapshot.footprint_mb is not None
        and snapshot.footprint_mb > limits.max_footprint_mb
    ):
        return "WORKER_FOOTPRINT_LIMIT_EXCEEDED"
    if (
        snapshot.system_free_percent is not None
        and snapshot.system_free_percent < limits.minimum_system_free_percent
    ):
        return "SYSTEM_MEMORY_PRESSURE_LIMIT_EXCEEDED"
    return None


def _terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return False
    terminated = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return terminated
    try:
        process.wait(timeout=5)
        return terminated
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return terminated
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return terminated


def _resource_peaks(samples: Sequence[ResourceSnapshot]) -> dict[str, Any]:
    def maximum(field: str) -> float | None:
        values = [
            value for sample in samples if (value := getattr(sample, field)) is not None
        ]
        return round(max(values), 6) if values else None

    free_values = [
        sample.system_free_percent
        for sample in samples
        if sample.system_free_percent is not None
    ]
    return {
        "sampleCount": len(samples),
        "peakRssMb": maximum("rss_mb"),
        "peakFootprintMb": maximum("footprint_mb"),
        "minimumSystemFreePercent": min(free_values) if free_values else None,
    }


def supervise_command(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    output_directory: Path,
    run_id: str,
    limits: SupervisionLimits,
    sampler: Callable[..., ResourceSnapshot] = sample_resources,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=False)
    events_path = output_directory / "events.jsonl"
    stderr_path = output_directory / "stderr.log"
    resources_path = output_directory / "resources.jsonl"
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(environment),
        cwd=PROJECT_ROOT,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_queue: queue.Queue[bytes | None] = queue.Queue()

    def read_stdout() -> None:
        for line in iter(process.stdout.readline, b""):
            stdout_queue.put(line)
        stdout_queue.put(None)

    def read_stderr() -> None:
        with stderr_path.open("wb") as handle:
            for chunk in iter(lambda: process.stderr.read(64 * 1024), b""):
                handle.write(chunk)
                handle.flush()

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    started = time.monotonic()
    last_progress = started
    next_sample = started
    next_footprint = started
    samples: list[ResourceSnapshot] = []
    terminal_event: dict[str, Any] | None = None
    failure_code: str | None = None
    invalid_event_count = 0
    stdout_closed = False
    terminated = False
    with events_path.open("wb") as event_log, resources_path.open("wb") as resource_log:
        while True:
            now = time.monotonic()
            try:
                raw_line = stdout_queue.get(timeout=0.1)
            except queue.Empty:
                raw_line = b""
            if raw_line is None:
                stdout_closed = True
            elif raw_line:
                event_log.write(raw_line)
                event_log.flush()
                if len(raw_line) > MAX_EVENT_BYTES:
                    failure_code = "WORKER_EVENT_LINE_LIMIT_EXCEEDED"
                else:
                    try:
                        event = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        invalid_event_count += 1
                    else:
                        if (
                            isinstance(event, dict)
                            and event.get("schemaVersion") == SCHEMA_VERSION
                            and event.get("runId") == run_id
                            and event.get("type")
                            in {
                                "worker.started",
                                "worker.progress",
                                "worker.completed",
                                "worker.failed",
                            }
                        ):
                            last_progress = now
                            if event["type"] in {
                                "worker.completed",
                                "worker.failed",
                            }:
                                terminal_event = event
                        else:
                            invalid_event_count += 1
                if invalid_event_count:
                    failure_code = "WORKER_PROTOCOL_INVALID"

            now = time.monotonic()
            elapsed = now - started
            if now >= next_sample and process.poll() is None:
                include_footprint = now >= next_footprint
                snapshot = sampler(
                    process.pid,
                    elapsed_seconds=elapsed,
                    include_footprint=include_footprint,
                )
                samples.append(snapshot)
                resource_log.write(_canonical_json(snapshot.to_dict()) + b"\n")
                resource_log.flush()
                failure_code = _resource_failure(snapshot, limits) or failure_code
                next_sample = now + limits.sample_interval_seconds
                if include_footprint:
                    next_footprint = now + limits.footprint_interval_seconds

            if failure_code is None and elapsed > limits.hard_deadline_seconds:
                failure_code = "WORKER_HARD_DEADLINE_EXCEEDED"
            if (
                failure_code is None
                and now - last_progress > limits.idle_timeout_seconds
            ):
                failure_code = "WORKER_IDLE_TIMEOUT"
            if failure_code is not None and process.poll() is None:
                terminated = _terminate_process_group(process) or terminated
            if stdout_closed and process.poll() is not None and stdout_queue.empty():
                break

    if process.poll() is None:
        terminated = _terminate_process_group(process) or terminated
    exit_code = process.wait(timeout=5)
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)
    elapsed = time.monotonic() - started
    if failure_code is None:
        if terminal_event is None:
            failure_code = "WORKER_TERMINAL_EVENT_MISSING"
        elif terminal_event["type"] == "worker.failed":
            payload = terminal_event.get("payload")
            failure_code = (
                str(payload.get("code"))
                if isinstance(payload, Mapping) and isinstance(payload.get("code"), str)
                else "WORKER_INFERENCE_FAILED"
            )
        elif exit_code != 0:
            failure_code = "WORKER_EXIT_NONZERO"
    return {
        "exitCode": exit_code,
        "elapsedSeconds": round(elapsed, 6),
        "failureCode": failure_code,
        "invalidEventCount": invalid_event_count,
        "terminated": terminated,
        "terminalEvent": terminal_event,
        "resourcePeaks": _resource_peaks(samples),
        "artifactPaths": {
            "events": str(events_path),
            "resources": str(resources_path),
            "stderr": str(stderr_path),
        },
    }


def score_against_case(
    *,
    manifest_path: Path,
    case_id: str,
    media: MediaIdentity,
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path.resolve(strict=True), "truth manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ChallengeError("truth manifest cases must be an array")
    matches = [
        item
        for item in cases
        if isinstance(item, Mapping) and item.get("id") == case_id
    ]
    if len(matches) != 1:
        raise ChallengeError("truth case must resolve exactly once")
    case = dict(matches[0])
    if (
        case.get("sha256") != media.sha256
        or case.get("bytes") != media.bytes
        or case.get("expectedSpeakerCount") is None
    ):
        raise ChallengeError("truth case does not match challenge media identity")

    from backend.pipeline_metrics import ReferenceTurn, evaluate_reference_quality
    from tools.evaluate_sample_library import (
        _boundary_quality,
        _joint_transcription_quality,
    )
    from tools.sample_library import word_error_rate

    reference_turns = [
        ReferenceTurn(
            start_ms=round(float(turn["startSeconds"]) * 1000),
            end_ms=round(float(turn["endSeconds"]) * 1000),
            speaker_ids=(str(turn["speakerId"]),),
        )
        for turn in case["turns"]
        if isinstance(turn, Mapping)
    ]
    predicted = [
        SimpleNamespace(
            start_ms=int(segment["startMs"]),
            end_ms=int(segment["endMs"]),
            speaker_id=str(segment["speakerId"]),
            overlapping=any(
                other is not segment
                and int(other["startMs"]) < int(segment["endMs"])
                and int(other["endMs"]) > int(segment["startMs"])
                for other in segments
            ),
            evidence={},
        )
        for segment in segments
    ]
    diarization = evaluate_reference_quality(predicted, reference_turns)
    hypothesis_text = " ".join(str(segment["rawText"]) for segment in segments)
    duration_ms = round(media.duration_seconds * 1000)
    joint = _joint_transcription_quality(
        case=case,
        transcript={"source": {"durationMs": duration_ms}},
        segments=segments,
        speaker_timeline=None,
    )
    expected_count = int(case["expectedSpeakerCount"])
    observed_speakers = sorted({str(segment["speakerId"]) for segment in segments})
    return {
        "schemaVersion": SCHEMA_VERSION,
        "caseId": case_id,
        "truthManifest": str(manifest_path.resolve()),
        "truthManifestSha256": _sha256_file(manifest_path.resolve()),
        "expectedSpeakerCount": expected_count,
        "observedSpeakerCount": len(observed_speakers),
        "speakerCountMatch": len(observed_speakers) == expected_count,
        "referenceTurnCount": len(reference_turns),
        "hypothesisTurnCount": len(segments),
        "diarizationQuality": {
            key: round(float(value), 9) for key, value in diarization.items()
        },
        "boundaryQuality": _boundary_quality(
            segments,
            reference_turns,
            speaker_timeline=None,
        ),
        "serializedTextErrorRate": word_error_rate(
            str(case["scoringTranscript"]),
            hypothesis_text,
        ),
        "jointTranscriptionQuality": joint,
        "languageQuality": {
            "scored": False,
            "reason": "moss-transformers-output-has-no-independent-language-field",
            "expectedLanguage": case.get("language"),
        },
    }


class _WorkerProgress:
    def __init__(self, *, run_id: str, heartbeat_interval_seconds: float):
        self.run_id = run_id
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.started = time.monotonic()
        self.stage = "worker_bootstrap"
        self.status = "starting"
        self.progress = 0.0
        self.generated_tokens: int | None = None
        self._last_emitted_tokens = 0
        self._last_token_event = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": self.run_id,
            "type": event_type,
            "elapsedSeconds": round(time.monotonic() - self.started, 6),
            "payload": dict(payload),
        }
        line = _canonical_json(event)
        if len(line) + 1 > MAX_EVENT_BYTES:
            raise ChallengeError("worker event exceeded the JSONL line boundary")
        with self._lock:
            sys.stdout.buffer.write(line + b"\n")
            sys.stdout.buffer.flush()

    def start(self) -> None:
        self.emit(
            "worker.started",
            {
                "pid": os.getpid(),
                "stage": self.stage,
                "status": self.status,
            },
        )
        self._thread.start()

    def update(
        self,
        stage: str,
        progress: float | None,
        generated_tokens: int | None,
    ) -> None:
        self.stage = stage
        self.status = "running"
        if progress is not None and math.isfinite(progress):
            self.progress = max(0.0, min(1.0, float(progress)))
        if generated_tokens is not None:
            self.generated_tokens = int(generated_tokens)
            now = time.monotonic()
            if (
                self.generated_tokens - self._last_emitted_tokens >= 8
                or now - self._last_token_event >= self.heartbeat_interval_seconds
            ):
                self._last_emitted_tokens = self.generated_tokens
                self._last_token_event = now
                self._emit_progress("token")

    def _emit_progress(self, kind: str) -> None:
        self.emit(
            "worker.progress",
            {
                "kind": kind,
                "stage": self.stage,
                "status": self.status,
                "progress": round(self.progress, 6),
                "generatedTokens": self.generated_tokens,
            },
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            self._emit_progress("heartbeat")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.heartbeat_interval_seconds + 1)


def worker_main(args: argparse.Namespace) -> int:
    progress = _WorkerProgress(
        run_id=args.run_id,
        heartbeat_interval_seconds=args.worker_heartbeat_interval_seconds,
    )
    progress.start()
    try:
        source = args.model_source.resolve(strict=True)
        model = args.model_path.resolve(strict=True)
        audio = args.audio.resolve(strict=True)
        if not source.is_dir() or not model.is_dir() or not audio.is_file():
            raise ChallengeError("worker input paths are incomplete")
        from importlib import metadata

        runtime_versions = {
            package: metadata.version(package) for package in EXPECTED_RUNTIME_VERSIONS
        }
        if runtime_versions != EXPECTED_RUNTIME_VERSIONS:
            raise ChallengeError("worker runtime does not match pinned versions")
        sys.path.insert(0, str(source))
        progress.update("runtime_import", 0.01, None)
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.app.model_runner import ModelRunner

        progress.update("loading_model", 0.05, None)
        runner = ModelRunner(
            model,
            device=args.device,
            dtype=args.dtype,
        )
        result = runner.transcribe(
            audio,
            max_new_tokens=args.max_new_tokens,
            decoding="greedy",
            status_callback=progress.update,
        )
        progress.update("validating_output", 0.9, result.generated_tokens)
        parsed = list(parse_transcript(result.text))
        progress.emit(
            "worker.completed",
            {
                "rawText": result.text,
                "promptTokens": result.prompt_len,
                "generatedTokens": result.generated_tokens,
                "inferenceSeconds": round(result.elapsed_sec, 6),
                "runtimeVersions": runtime_versions,
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": segment.speaker,
                        "text": segment.text,
                    }
                    for segment in parsed
                ],
            },
        )
        return 0
    # The worker boundary must turn every inference failure into a terminal event.
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        try:
            progress.emit(
                "worker.failed",
                {
                    "code": "MOSS_INFERENCE_FAILED",
                    "errorType": type(exc).__name__,
                },
            )
        except Exception:  # noqa: BLE001 - stderr remains the final diagnostic sink
            traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        progress.close()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--model-audit", type=Path, default=DEFAULT_MODEL_AUDIT)
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=DEFAULT_WORKER_PYTHON,
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=1024)
    parser.add_argument("--max-segments", type=_positive_int, default=512)
    parser.add_argument(
        "--max-text-characters",
        type=_positive_int,
        default=200_000,
    )
    parser.add_argument(
        "--timestamp-tolerance-seconds",
        type=_positive_float,
        default=0.05,
    )
    parser.add_argument(
        "--worker-heartbeat-interval-seconds",
        type=_positive_float,
        default=2.0,
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument(
        "--hard-deadline-seconds",
        type=_positive_float,
        default=360.0,
    )
    parser.add_argument(
        "--resource-sample-interval-seconds",
        type=_positive_float,
        default=1.0,
    )
    parser.add_argument(
        "--footprint-sample-interval-seconds",
        type=_positive_float,
        default=5.0,
    )
    parser.add_argument("--max-rss-mb", type=_positive_float, default=12_288.0)
    parser.add_argument(
        "--max-footprint-mb",
        type=_positive_float,
        default=14_336.0,
    )
    parser.add_argument(
        "--minimum-system-free-percent",
        type=int,
        default=5,
    )
    parser.add_argument("--truth-manifest", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--skip-full-snapshot-hash",
        action="store_true",
        help="development only; production evidence records this weaker check",
    )
    return parser


def run_challenge(args: argparse.Namespace) -> int:
    if args.output_directory is None:
        raise ChallengeError("--output-directory is required outside worker mode")
    if (args.truth_manifest is None) != (args.case_id is None):
        raise ChallengeError("--truth-manifest and --case-id must be used together")
    if not 0 <= args.minimum_system_free_percent <= 100:
        raise ChallengeError("minimum system free percent must be in [0, 100]")
    output_directory = args.output_directory.resolve()
    if output_directory.exists():
        raise ChallengeError("output directory already exists; evidence is immutable")
    media = probe_media(args.audio)
    model_identity = verify_model_snapshot(
        model_path=args.model_path,
        audit_path=args.model_audit,
        verify_file_hashes=not args.skip_full_snapshot_hash,
    )
    worker_python = worker_python_entry(args.worker_python)
    limits = SupervisionLimits(
        idle_timeout_seconds=args.idle_timeout_seconds,
        hard_deadline_seconds=args.hard_deadline_seconds,
        sample_interval_seconds=args.resource_sample_interval_seconds,
        footprint_interval_seconds=args.footprint_sample_interval_seconds,
        max_rss_mb=args.max_rss_mb,
        max_footprint_mb=args.max_footprint_mb,
        minimum_system_free_percent=args.minimum_system_free_percent,
    )
    output_limits = OutputLimits(
        max_new_tokens=args.max_new_tokens,
        max_segments=args.max_segments,
        max_text_characters=args.max_text_characters,
        timestamp_tolerance_seconds=args.timestamp_tolerance_seconds,
    )
    command = [
        str(worker_python),
        str(Path(__file__).resolve()),
        "--worker",
        "--run-id",
        args.run_id,
        "--audio",
        str(media.path),
        "--model-path",
        str(args.model_path.resolve()),
        "--model-source",
        str(args.model_source.resolve()),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--worker-heartbeat-interval-seconds",
        str(args.worker_heartbeat_interval_seconds),
    ]
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONNOUSERSITE": "1",
    }
    supervision = supervise_command(
        command=command,
        environment=environment,
        output_directory=output_directory,
        run_id=args.run_id,
        limits=limits,
    )
    failure_code = supervision["failureCode"]
    validation = None
    quality = None
    terminal = supervision.get("terminalEvent")
    if failure_code is None and isinstance(terminal, Mapping):
        payload = terminal.get("payload")
        if not isinstance(payload, Mapping):
            failure_code = "WORKER_TERMINAL_PAYLOAD_INVALID"
        else:
            try:
                validation = validate_worker_result(
                    payload,
                    duration_seconds=media.duration_seconds,
                    limits=output_limits,
                )
            except ChallengeError:
                traceback.print_exc(file=sys.stderr)
                failure_code = "WORKER_OUTPUT_BOUNDARY_INVALID"
            else:
                raw_text = str(payload["rawText"])
                _write_bytes_atomic(
                    output_directory / "raw-transcript.txt",
                    raw_text.encode("utf-8") + b"\n",
                )
                _write_json_atomic(
                    output_directory / "segments.v1.json",
                    validation["segments"],
                )
                if args.truth_manifest is not None:
                    try:
                        quality = score_against_case(
                            manifest_path=args.truth_manifest,
                            case_id=args.case_id,
                            media=media,
                            segments=validation["segments"],
                        )
                    except (
                        ChallengeError,
                        ImportError,
                        RuntimeError,
                        ValueError,
                    ):
                        traceback.print_exc(file=sys.stderr)
                        failure_code = "QUALITY_SCORING_FAILED"
                    else:
                        _write_json_atomic(
                            output_directory / "quality.v1.json",
                            quality,
                        )
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "runId": args.run_id,
        "status": "completed" if failure_code is None else "failed",
        "failureCode": failure_code,
        "input": {
            "path": str(media.path),
            "sha256": media.sha256,
            "bytes": media.bytes,
            "durationSeconds": round(media.duration_seconds, 6),
        },
        "model": model_identity,
        "runtime": {
            "workerPython": str(worker_python),
            "device": args.device,
            "dtype": args.dtype,
            "decoding": "greedy",
            "networkPolicy": "forced-offline",
        },
        "limits": {
            "idleTimeoutSeconds": limits.idle_timeout_seconds,
            "hardDeadlineSeconds": limits.hard_deadline_seconds,
            "maxRssMb": limits.max_rss_mb,
            "maxFootprintMb": limits.max_footprint_mb,
            "minimumSystemFreePercent": limits.minimum_system_free_percent,
            "maxNewTokens": output_limits.max_new_tokens,
            "maxSegments": output_limits.max_segments,
            "maxTextCharacters": output_limits.max_text_characters,
            "timestampToleranceSeconds": (output_limits.timestamp_tolerance_seconds),
        },
        "supervision": supervision,
        "outputValidation": validation["checks"] if validation else None,
        "qualityPath": (
            str(output_directory / "quality.v1.json") if quality is not None else None
        ),
    }
    report_path = output_directory / "challenge-report.v1.json"
    _write_json_atomic(report_path, report)
    summary = {
        "status": report["status"],
        "failureCode": failure_code,
        "runId": args.run_id,
        "report": str(report_path),
        "reportSha256": _sha256_file(report_path),
        "quality": quality,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failure_code is None else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        return worker_main(args)
    try:
        return run_challenge(args)
    except ChallengeError as exc:
        print(f"bounded MOSS challenge failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
