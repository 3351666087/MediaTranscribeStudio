"""Deterministic, model-independent sampling for long local media."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import stat
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "1.0.0"
ANALYSIS_VERSION = "long-media-stratified-v1"
_SAFE_ID = re.compile(r"[^a-z0-9]+")


class LongMediaSampleError(RuntimeError):
    """Raised when long-media evidence cannot be produced safely."""


@dataclass(frozen=True)
class AudioFrameFeature:
    start_ms: int
    end_ms: int
    rms_db: float
    spectral_centroid_hz: float
    zero_crossing_rate: float
    active: bool = False
    acoustic_change_score: float = 0.0

    @property
    def midpoint_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2


@dataclass(frozen=True)
class ReferenceTurn:
    recording_id: str
    speaker_id: str
    start_ms: int
    end_ms: int


def _path_is_dataless(path: Path) -> bool:
    """Detect a macOS cloud placeholder without materializing it."""

    try:
        flags = path.stat(follow_symlinks=False).st_flags
    except (AttributeError, OSError):
        return False
    return bool(flags & getattr(stat, "SF_DATALESS", 0))


def _materialized_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise LongMediaSampleError(f"{label} is unavailable: {path}") from exc
    if not resolved.is_file():
        raise LongMediaSampleError(f"{label} is not a file: {resolved}")
    if _path_is_dataless(resolved):
        raise LongMediaSampleError(
            f"{label} is a dataless cloud placeholder: {resolved}"
        )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise LongMediaSampleError(f"{label} requires {key}")
    return raw.strip()


def _source_reference_metadata(
    reference_path: Path,
    *,
    source: Path,
    source_sha256: str,
    duration_seconds: float,
) -> dict[str, Any]:
    reference = _materialized_file(reference_path, label="source reference")
    try:
        value = json.loads(reference.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongMediaSampleError("source reference is not valid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("schemaVersion") != SCHEMA_VERSION
        or value.get("artifactType") != "long-media-source-reference"
    ):
        raise LongMediaSampleError("source reference schema is invalid")
    raw_source = value.get("source")
    if not isinstance(raw_source, dict):
        raise LongMediaSampleError("source reference has no source object")
    if raw_source.get("sha256") != source_sha256:
        raise LongMediaSampleError("source reference SHA-256 does not match media")
    if raw_source.get("bytes") != source.stat().st_size:
        raise LongMediaSampleError("source reference byte count does not match media")
    declared_duration = raw_source.get("durationSeconds")
    if (
        not isinstance(declared_duration, (int, float))
        or isinstance(declared_duration, bool)
        or abs(float(declared_duration) - duration_seconds) > 0.25
    ):
        raise LongMediaSampleError("source reference duration does not match media")
    declared_sha1 = raw_source.get("sha1")
    if isinstance(declared_sha1, str) and declared_sha1 != _sha1(source):
        raise LongMediaSampleError("source reference SHA-1 does not match media")

    dataset = _required_text(raw_source, "dataset", label="source reference")
    revision = _required_text(raw_source, "revision", label="source reference")
    provider = _required_text(raw_source, "provider", label="source reference")
    source_url = _required_text(raw_source, "sourceUrl", label="source reference")
    description_url = _required_text(
        raw_source,
        "descriptionUrl",
        label="source reference",
    )
    if not source_url.startswith("https://") or not description_url.startswith(
        "https://"
    ):
        raise LongMediaSampleError("source reference URLs must use HTTPS")
    if raw_source.get("recordingType") != "real-recording":
        raise LongMediaSampleError("source reference must describe a real recording")
    speech_nature = _required_text(
        raw_source,
        "speechNature",
        label="source reference",
    )
    language_tags = raw_source.get("languageTags")
    if not isinstance(language_tags, list) or not language_tags or any(
        not isinstance(item, str) or not item for item in language_tags
    ):
        raise LongMediaSampleError("source reference languageTags are invalid")
    region = _required_text(raw_source, "region", label="source reference")

    immutable = raw_source.get("immutableEvidence")
    if not isinstance(immutable, dict):
        raise LongMediaSampleError("source reference immutable evidence is missing")
    page_revision = immutable.get("pageRevisionId")
    etag = immutable.get("etag")
    if not (
        (isinstance(page_revision, int) and not isinstance(page_revision, bool))
        or (isinstance(etag, str) and bool(etag.strip()))
    ):
        raise LongMediaSampleError(
            "source reference requires an immutable revision or ETag"
        )

    license_value = raw_source.get("license")
    if not isinstance(license_value, dict):
        raise LongMediaSampleError("source reference license evidence is missing")
    license_id = _required_text(license_value, "id", label="source license")
    license_url = _required_text(license_value, "url", label="source license")
    if not license_url.startswith("https://"):
        raise LongMediaSampleError("source license URL must use HTTPS")
    short_name = _required_text(license_value, "shortName", label="source license")
    attribution_required = license_value.get("attributionRequired")
    if not isinstance(attribution_required, bool):
        raise LongMediaSampleError(
            "source license attributionRequired must be boolean"
        )

    license_evidence: dict[str, Any] = {
        "provider": provider,
        "sourceUrl": source_url,
        "descriptionUrl": description_url,
        "licenseUrl": license_url,
        "licenseShortName": short_name,
        "attributionRequired": attribution_required,
        "sourceReference": str(reference),
        "sourceReferenceSha256": _sha256(reference),
        **immutable,
    }
    for key in (
        "pageId",
        "fileRevisionTimestamp",
        "httpLastModified",
        "artist",
        "description",
        "sha1",
    ):
        if key in raw_source:
            license_evidence[key] = raw_source[key]
    return {
        "dataset": dataset,
        "revision": revision,
        "license": license_id,
        "languageTags": list(dict.fromkeys(language_tags)),
        "region": region,
        "recordingType": "real-recording",
        "speechNature": speech_nature,
        "licenseEvidence": license_evidence,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise LongMediaSampleError("cannot calculate a percentile without values")
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mark_activity_and_changes(
    frames: Sequence[AudioFrameFeature],
) -> tuple[tuple[AudioFrameFeature, ...], float]:
    """Apply an adaptive energy gate and adjacent acoustic-change score."""

    if not frames:
        raise LongMediaSampleError("audio analysis returned no frames")
    levels = [frame.rms_db for frame in frames]
    floor = _percentile(levels, 0.05)
    median = _percentile(levels, 0.50)
    threshold = min(median, floor + 12.0)
    marked: list[AudioFrameFeature] = []
    for index, frame in enumerate(frames):
        if index == 0:
            change = 0.0
        else:
            previous = frames[index - 1]
            change = (
                abs(frame.rms_db - previous.rms_db) / 12.0
                + abs(
                    frame.spectral_centroid_hz
                    - previous.spectral_centroid_hz
                )
                / 4000.0
                + abs(
                    frame.zero_crossing_rate - previous.zero_crossing_rate
                )
                / 0.20
            )
        marked.append(
            replace(
                frame,
                active=frame.rms_db >= threshold,
                acoustic_change_score=round(change, 9),
            )
        )
    if not any(frame.active for frame in marked):
        fallback = _percentile(levels, 0.90)
        threshold = fallback
        marked = [
            replace(frame, active=frame.rms_db >= fallback)
            for frame in marked
        ]
    return tuple(marked), round(threshold, 6)


def _frame_feature(
    samples: np.ndarray,
    *,
    start_ms: int,
    sample_rate: int,
) -> AudioFrameFeature:
    if samples.size == 0:
        raise LongMediaSampleError("cannot analyze an empty audio frame")
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    rms_db = 20.0 * math.log10(max(rms, 1e-6))
    stride = max(1, int(math.ceil(samples.size / 2048)))
    compact = samples[::stride][:2048].astype(np.float64, copy=False)
    if compact.size < 2:
        centroid = 0.0
        zcr = 0.0
    else:
        spectrum = np.abs(np.fft.rfft(compact * np.hanning(compact.size)))
        frequencies = np.fft.rfftfreq(
            compact.size,
            d=stride / sample_rate,
        )
        magnitude = float(spectrum.sum())
        centroid = (
            float(np.dot(spectrum, frequencies) / magnitude)
            if magnitude > 1e-12
            else 0.0
        )
        signs = np.signbit(compact)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / (
            compact.size - 1
        )
    duration_ms = round(samples.size * 1000 / sample_rate)
    return AudioFrameFeature(
        start_ms=start_ms,
        end_ms=start_ms + duration_ms,
        rms_db=round(rms_db, 6),
        spectral_centroid_hz=round(centroid, 6),
        zero_crossing_rate=round(zcr, 9),
    )


def analyze_audio(
    source: Path,
    *,
    ffmpeg: str = "ffmpeg",
    frame_seconds: float = 1.0,
) -> tuple[tuple[AudioFrameFeature, ...], float]:
    """Stream mono PCM from FFmpeg and retain only bounded frame features."""

    if not 0.25 <= frame_seconds <= 5.0:
        raise LongMediaSampleError("frame_seconds must be between 0.25 and 5")
    sample_rate = 16_000
    samples_per_frame = round(sample_rate * frame_seconds)
    bytes_per_frame = samples_per_frame * 4
    process = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise LongMediaSampleError("FFmpeg analysis pipes are unavailable")
    frames: list[AudioFrameFeature] = []
    pending = bytearray()
    next_start_ms = 0
    while True:
        chunk = process.stdout.read(bytes_per_frame - len(pending))
        if not chunk:
            break
        pending.extend(chunk)
        if len(pending) < bytes_per_frame:
            continue
        samples = np.frombuffer(pending, dtype="<f4").copy()
        frame = _frame_feature(
            samples,
            start_ms=next_start_ms,
            sample_rate=sample_rate,
        )
        frames.append(frame)
        next_start_ms = frame.end_ms
        pending.clear()
    if len(pending) >= sample_rate:
        samples = np.frombuffer(pending, dtype="<f4").copy()
        frames.append(
            _frame_feature(
                samples,
                start_ms=next_start_ms,
                sample_rate=sample_rate,
            )
        )
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise LongMediaSampleError(
            f"FFmpeg audio analysis failed: {stderr[-1000:].strip()}"
        )
    return mark_activity_and_changes(frames)


def _window_start(center_ms: int, duration_ms: int, window_ms: int) -> int:
    return max(0, min(duration_ms - window_ms, center_ms - window_ms // 2))


def _window_statistics(
    frames: Sequence[AudioFrameFeature],
    *,
    start_ms: int,
    end_ms: int,
) -> dict[str, float]:
    selected = [
        frame
        for frame in frames
        if start_ms <= frame.midpoint_ms < end_ms
    ]
    if not selected:
        return {
            "audioActivityRatio": 0.0,
            "meanAcousticChangeScore": 0.0,
            "maxAcousticChangeScore": 0.0,
        }
    return {
        "audioActivityRatio": round(
            sum(frame.active for frame in selected) / len(selected),
            9,
        ),
        "meanAcousticChangeScore": round(
            sum(frame.acoustic_change_score for frame in selected)
            / len(selected),
            9,
        ),
        "maxAcousticChangeScore": round(
            max(frame.acoustic_change_score for frame in selected),
            9,
        ),
    }


def select_stratified_windows(
    frames: Sequence[AudioFrameFeature],
    *,
    duration_ms: int,
    window_ms: int,
    random_seed: int,
    random_window_count: int = 2,
    change_window_count: int = 3,
) -> list[dict[str, Any]]:
    """Select full-timeline strata without consulting diarization output."""

    if duration_ms < window_ms or window_ms < 10_000:
        raise LongMediaSampleError(
            "media duration must cover a window of at least 10 seconds"
        )
    if random_window_count < 0 or change_window_count < 0:
        raise LongMediaSampleError("window counts must be non-negative")
    active = [frame for frame in frames if frame.active]
    if not active:
        raise LongMediaSampleError("no active audio frames were detected")

    chosen: list[dict[str, Any]] = []
    starts: set[int] = set()
    minimum_spacing = window_ms // 2

    def add(reason: str, desired_start: int, *, strict_spacing: bool) -> bool:
        start_ms = max(0, min(duration_ms - window_ms, desired_start))
        if start_ms in starts:
            return False
        if strict_spacing and any(
            abs(start_ms - existing) < minimum_spacing for existing in starts
        ):
            return False
        starts.add(start_ms)
        end_ms = start_ms + window_ms
        chosen.append(
            {
                "reason": reason,
                "startMs": start_ms,
                "endMs": end_ms,
                "durationMs": window_ms,
                **_window_statistics(
                    frames,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "selectionUsesModelScores": False,
            }
        )
        return True

    add(
        "stratum-start",
        max(0, active[0].start_ms - window_ms // 10),
        strict_spacing=False,
    )
    middle = min(
        active,
        key=lambda frame: (
            abs(frame.midpoint_ms - duration_ms // 2),
            frame.start_ms,
        ),
    )
    add(
        "stratum-middle",
        _window_start(middle.midpoint_ms, duration_ms, window_ms),
        strict_spacing=False,
    )
    add(
        "stratum-end",
        min(
            duration_ms - window_ms,
            max(0, active[-1].end_ms - window_ms * 9 // 10),
        ),
        strict_spacing=False,
    )

    random_candidates = list(active)
    random.Random(random_seed).shuffle(random_candidates)
    random_index = 1
    for frame in random_candidates:
        if random_index > random_window_count:
            break
        if add(
            f"seeded-active-random-{random_index}",
            _window_start(frame.midpoint_ms, duration_ms, window_ms),
            strict_spacing=True,
        ):
            random_index += 1

    change_candidates = sorted(
        active,
        key=lambda frame: (
            -frame.acoustic_change_score,
            frame.start_ms,
        ),
    )
    change_index = 1
    for frame in change_candidates:
        if change_index > change_window_count:
            break
        if add(
            f"acoustic-change-{change_index}",
            _window_start(frame.midpoint_ms, duration_ms, window_ms),
            strict_spacing=True,
        ):
            change_index += 1
    return chosen


def parse_rttm(
    path: Path,
    *,
    recording_id: str | None = None,
) -> tuple[ReferenceTurn, ...]:
    """Parse SPEAKER RTTM lines and optionally select one recording."""

    turns: list[ReferenceTurn] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            raise LongMediaSampleError(
                f"invalid RTTM line {line_number}"
            )
        current_recording = fields[1]
        if recording_id is not None and current_recording != recording_id:
            continue
        try:
            start_ms = round(float(fields[3]) * 1000)
            duration_ms = round(float(fields[4]) * 1000)
        except ValueError as exc:
            raise LongMediaSampleError(
                f"invalid RTTM timing at line {line_number}"
            ) from exc
        if start_ms < 0 or duration_ms <= 0 or not fields[7].strip():
            raise LongMediaSampleError(
                f"invalid RTTM turn at line {line_number}"
            )
        turns.append(
            ReferenceTurn(
                recording_id=current_recording,
                speaker_id=fields[7].strip(),
                start_ms=start_ms,
                end_ms=start_ms + duration_ms,
            )
        )
    recording_ids = {turn.recording_id for turn in turns}
    if not turns:
        raise LongMediaSampleError("RTTM contains no selected SPEAKER turns")
    if recording_id is None and len(recording_ids) != 1:
        raise LongMediaSampleError(
            "RTTM contains multiple recordings; recording_id is required"
        )
    return tuple(turns)


def _interval_union_duration(
    intervals: Iterable[tuple[int, int]],
) -> int:
    ordered = sorted(
        (start, end) for start, end in intervals if end > start
    )
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def _reference_window_evidence(
    turns: Sequence[ReferenceTurn],
    *,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    clipped = [
        (
            turn.speaker_id,
            max(start_ms, turn.start_ms),
            min(end_ms, turn.end_ms),
        )
        for turn in turns
        if turn.end_ms > start_ms and turn.start_ms < end_ms
    ]
    per_speaker: dict[str, int] = {}
    for speaker, turn_start, turn_end in clipped:
        per_speaker[speaker] = (
            per_speaker.get(speaker, 0) + turn_end - turn_start
        )
    events: list[tuple[int, int]] = []
    for _speaker, turn_start, turn_end in clipped:
        events.append((turn_start, 1))
        events.append((turn_end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    overlap_ms = 0
    active_count = 0
    previous: int | None = None
    for position, delta in events:
        if previous is not None and active_count >= 2:
            overlap_ms += position - previous
        active_count += delta
        previous = position
    speech_ms = _interval_union_duration(
        (turn_start, turn_end)
        for _speaker, turn_start, turn_end in clipped
    )
    return {
        "speakerSet": sorted(per_speaker),
        "perSpeakerAnnotatedMs": {
            speaker: per_speaker[speaker] for speaker in sorted(per_speaker)
        },
        "annotatedSpeechMs": speech_ms,
        "annotatedOverlapMs": overlap_ms,
    }


def select_reference_speaker_window(
    turns: Sequence[ReferenceTurn],
    *,
    media_duration_ms: int,
    window_ms: int,
    target_speaker_count: int,
    random_seed: int,
) -> dict[str, Any]:
    """Select an exact-N window from reference turns, never model output."""

    if target_speaker_count < 1:
        raise LongMediaSampleError("target_speaker_count must be positive")
    max_start = media_duration_ms - window_ms
    if max_start < 0:
        raise LongMediaSampleError("reference media is shorter than the window")
    candidate_starts = {0, max_start}
    for turn in turns:
        for value in (
            turn.start_ms,
            turn.end_ms,
            turn.start_ms - window_ms,
            turn.end_ms - window_ms,
        ):
            candidate_starts.add(max(0, min(max_start, value)))

    ranked: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for start_ms in sorted(candidate_starts):
        end_ms = start_ms + window_ms
        evidence = _reference_window_evidence(
            turns,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if len(evidence["speakerSet"]) != target_speaker_count:
            continue
        durations = list(evidence["perSpeakerAnnotatedMs"].values())
        tie_break = int.from_bytes(
            hashlib.sha256(
                f"{random_seed}:{start_ms}".encode("ascii")
            ).digest()[:8],
            "big",
        )
        score = (
            min(durations),
            evidence["annotatedOverlapMs"],
            evidence["annotatedSpeechMs"],
            tie_break,
        )
        ranked.append(
            (
                score,
                {
                    "reason": f"reference-exact-n{target_speaker_count}",
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "durationMs": window_ms,
                    "selectionUsesModelScores": False,
                    "selectionTruth": "reference-rttm",
                    **evidence,
                },
            )
        )
    if not ranked:
        raise LongMediaSampleError(
            f"RTTM has no {window_ms / 1000:g}s window with exactly "
            f"{target_speaker_count} speakers"
        )
    return max(ranked, key=lambda item: item[0])[1]


def probe_media(source: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            (
                "format=format_name,duration,size:"
                "stream=index,codec_type,codec_name,sample_rate,channels,"
                "width,height,duration"
            ),
            "-of",
            "json",
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise LongMediaSampleError(
            "FFprobe failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    try:
        value = json.loads(completed.stdout)
        duration_ms = round(float(value["format"]["duration"]) * 1000)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LongMediaSampleError("FFprobe returned invalid media metadata") from exc
    if duration_ms <= 0 or not any(
        stream.get("codec_type") == "audio"
        for stream in value.get("streams", [])
        if isinstance(stream, dict)
    ):
        raise LongMediaSampleError("media must contain a positive audio stream")
    value["durationMs"] = duration_ms
    return value


def _safe_source_id(source: Path, source_sha256: str) -> str:
    slug = _SAFE_ID.sub("-", source.stem.casefold()).strip("-") or "media"
    return f"{slug[:48]}-{source_sha256[:10]}"


def _extract_audio_window(
    source: Path,
    output: Path,
    *,
    start_ms: int,
    duration_ms: int,
    ffmpeg: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration_ms / 1000:.3f}",
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise LongMediaSampleError(
            "FFmpeg window extraction failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )


def _coverage_ratio(windows: Sequence[dict[str, Any]], duration_ms: int) -> float:
    return round(
        _interval_union_duration(
            (int(item["startMs"]), int(item["endMs"]))
            for item in windows
        )
        / duration_ms,
        9,
    )


def build_long_media_matrix(
    sources: Sequence[Path],
    *,
    output_root: Path,
    window_seconds: float = 60.0,
    random_seed: int = 20260724,
    random_window_count: int = 2,
    change_window_count: int = 3,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    reference_rttm: Path | None = None,
    reference_recording_id: str | None = None,
    target_speaker_count: int | None = None,
    evaluation_split: str = "development",
    source_reference_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    sources = tuple(sources)
    source_reference_paths = tuple(source_reference_paths)
    if not sources:
        raise LongMediaSampleError("at least one source is required")
    if not 10.0 <= window_seconds <= 90.0:
        raise LongMediaSampleError("window_seconds must be between 10 and 90")
    if reference_rttm is not None and len(sources) != 1:
        raise LongMediaSampleError("RTTM selection accepts exactly one source")
    if (reference_rttm is None) != (target_speaker_count is None):
        raise LongMediaSampleError(
            "reference_rttm and target_speaker_count must be used together"
        )
    if evaluation_split not in {"development", "held-out"}:
        raise LongMediaSampleError("evaluation_split must be development or held-out")
    if source_reference_paths and len(source_reference_paths) != len(sources):
        raise LongMediaSampleError(
            "source reference count must match the source count"
        )
    if evaluation_split == "held-out" and len(source_reference_paths) != len(sources):
        raise LongMediaSampleError(
            "each held-out source requires a source reference"
        )

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    window_ms = round(window_seconds * 1000)
    source_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for source_index, raw_source in enumerate(sources):
        source = _materialized_file(raw_source, label="source")
        source_hash = _sha256(source)
        source_id = _safe_source_id(source, source_hash)
        probe = probe_media(source, ffprobe=ffprobe)
        duration_ms = int(probe.pop("durationMs"))
        if duration_ms < 300_000:
            raise LongMediaSampleError(
                "long-media source must be at least 300 seconds"
            )
        source_metadata = (
            _source_reference_metadata(
                source_reference_paths[source_index],
                source=source,
                source_sha256=source_hash,
                duration_seconds=duration_ms / 1000,
            )
            if source_reference_paths
            else {}
        )
        frames, threshold = analyze_audio(source, ffmpeg=ffmpeg)
        windows = select_stratified_windows(
            frames,
            duration_ms=duration_ms,
            window_ms=window_ms,
            random_seed=random_seed,
            random_window_count=random_window_count,
            change_window_count=change_window_count,
        )
        if reference_rttm is not None:
            turns = parse_rttm(
                _materialized_file(reference_rttm, label="reference RTTM"),
                recording_id=reference_recording_id,
            )
            windows.append(
                select_reference_speaker_window(
                    turns,
                    media_duration_ms=duration_ms,
                    window_ms=window_ms,
                    target_speaker_count=int(target_speaker_count),
                    random_seed=random_seed,
                )
            )
        source_row = {
            "id": source_id,
            "path": str(source),
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": source_hash,
            "mediaProbe": probe,
            "durationMs": duration_ms,
            "analysis": {
                "algorithm": ANALYSIS_VERSION,
                "frameDurationMs": 1000,
                "frameCount": len(frames),
                "activityThresholdDb": threshold,
                "rmsDbPercentiles": {
                    "p05": round(
                        _percentile(
                            [frame.rms_db for frame in frames],
                            0.05,
                        ),
                        6,
                    ),
                    "p50": round(
                        _percentile(
                            [frame.rms_db for frame in frames],
                            0.50,
                        ),
                        6,
                    ),
                    "p95": round(
                        _percentile(
                            [frame.rms_db for frame in frames],
                            0.95,
                        ),
                        6,
                    ),
                },
                "activeFrameRatio": round(
                    sum(frame.active for frame in frames) / len(frames),
                    9,
                ),
                "randomSeed": random_seed,
                "selectionUsesModelScores": False,
            },
            "windowCoverageRatio": _coverage_ratio(windows, duration_ms),
            "windowCount": len(windows),
            **source_metadata,
        }
        source_rows.append(source_row)
        for index, window in enumerate(windows, start=1):
            case_id = f"{source_id}-w{index:02d}"
            relative_audio = Path("audio") / f"{case_id}.wav"
            output = output_root / relative_audio
            _extract_audio_window(
                source,
                output,
                start_ms=int(window["startMs"]),
                duration_ms=int(window["durationMs"]),
                ffmpeg=ffmpeg,
            )
            clip_probe = probe_media(output, ffprobe=ffprobe)
            actual_duration_ms = int(clip_probe.pop("durationMs"))
            audio_stream = next(
                (
                    stream
                    for stream in clip_probe.get("streams", [])
                    if isinstance(stream, dict)
                    and stream.get("codec_type") == "audio"
                ),
                None,
            )
            if (
                audio_stream is None
                or audio_stream.get("codec_name") != "pcm_s16le"
                or audio_stream.get("sample_rate") != "16000"
                or audio_stream.get("channels") != 1
            ):
                raise LongMediaSampleError(
                    "extracted window is not PCM s16le/16 kHz/mono"
                )
            if abs(actual_duration_ms - int(window["durationMs"])) > 250:
                raise LongMediaSampleError(
                    "extracted window duration exceeds the 250 ms tolerance"
                )
            case_rows.append(
                {
                    "id": case_id,
                    "sourceId": source_id,
                    "language": "auto",
                    "region": source_metadata.get(
                        "region",
                        "user-provided-unknown",
                    ),
                    "evaluationSplit": evaluation_split,
                    "scenario": [
                        "real-recording",
                        "long-media-stratified",
                        (
                            "video"
                            if any(
                                stream.get("codec_type") == "video"
                                for stream in probe.get("streams", [])
                                if isinstance(stream, dict)
                            )
                            else "audio"
                        ),
                    ],
                    "realOrSynthetic": "real-recording",
                    "expectedSpeakerCount": None,
                    "path": relative_audio.as_posix(),
                    "bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                    "audio": {
                        "codec": "pcm_s16le",
                        "sampleRate": 16000,
                        "channels": 1,
                        "durationSeconds": round(
                            actual_duration_ms / 1000,
                            6,
                        ),
                        "requestedDurationSeconds": window_seconds,
                        "durationToleranceMs": 250,
                    },
                    "mediaProbe": clip_probe,
                    "windowSelection": window,
                    "truthEligibility": {
                        "speakerCount": (
                            window.get("selectionTruth") == "reference-rttm"
                        ),
                        "turnBoundaries": (
                            window.get("selectionTruth") == "reference-rttm"
                        ),
                        "overlap": (
                            window.get("selectionTruth") == "reference-rttm"
                        ),
                        "derJer": (
                            window.get("selectionTruth") == "reference-rttm"
                        ),
                        "asr": False,
                    },
                }
            )
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "libraryId": "mts-long-media-stratified-v1",
        "generatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "evaluationSplit": evaluation_split,
        "windowDurationSeconds": window_seconds,
        "selectionPolicy": {
            "algorithm": ANALYSIS_VERSION,
            "requiredStrata": [
                "stratum-start",
                "stratum-middle",
                "stratum-end",
                "seeded-active-random",
                "acoustic-change",
            ],
            "randomSeed": random_seed,
            "randomWindowCount": random_window_count,
            "changeWindowCount": change_window_count,
            "modelScoresUsed": False,
            "minimumSourceDurationSeconds": 300,
            "referenceTargetSelection": (
                "exact-speaker-count-from-rttm-before-model-evaluation"
            ),
        },
        "sources": source_rows,
        "cases": case_rows,
    }
    manifest_path = output_root / "long-media-samples.resolved.v1.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "ANALYSIS_VERSION",
    "AudioFrameFeature",
    "LongMediaSampleError",
    "ReferenceTurn",
    "analyze_audio",
    "build_long_media_matrix",
    "mark_activity_and_changes",
    "parse_rttm",
    "probe_media",
    "select_reference_speaker_window",
    "select_stratified_windows",
]
