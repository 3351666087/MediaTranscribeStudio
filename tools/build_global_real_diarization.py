"""Build short real diarization windows from pinned public datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
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

from tools.build_global_derived_matrix import overlap_intervals  # noqa: E402
from tools.global_sample_library import (  # noqa: E402
    GlobalSampleLibraryError,
    load_global_manifest,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "sample_library" / "global-manifest.v1.json"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / ".runtime_cache" / "sample-library" / "global" / "real"
)
RESOLVED_NAME = "global-real-diarization.resolved.v1.json"
USER_AGENT = "MediaTranscribeStudio-real-diarization-library/1.0"
WINDOW_DURATIONS = (10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)
MIN_SPEAKER_SECONDS = 0.5
AISHELL4_MAX_DURATION = 30.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str, *, attempts: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise GlobalSampleLibraryError(f"remote JSON is not an object: {url}")
            return value
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise GlobalSampleLibraryError(
        f"remote JSON failed after {attempts} attempts: {url}: {last_error}"
    )


def _request_json_array(url: str, *, attempts: int = 5) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                value = json.load(response)
            if not isinstance(value, list) or any(
                not isinstance(item, dict) for item in value
            ):
                raise GlobalSampleLibraryError(
                    f"remote JSON is not an object array: {url}"
                )
            return value
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
            GlobalSampleLibraryError,
        ) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise GlobalSampleLibraryError(
        f"remote JSON failed after {attempts} attempts: {url}: {last_error}"
    )


def _download_resumable(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
    attempts: int = 5,
) -> None:
    if destination.is_file() and (
        expected_bytes is None or destination.stat().st_size == expected_bytes
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", None)
                append = offset > 0 and status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            actual_size = partial.stat().st_size
            if expected_bytes is not None and actual_size != expected_bytes:
                raise GlobalSampleLibraryError(
                    f"{destination.name} size is {actual_size}, expected {expected_bytes}"
                )
            partial.replace(destination)
            return
        except (
            OSError,
            urllib.error.URLError,
            GlobalSampleLibraryError,
        ) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise GlobalSampleLibraryError(
        f"download failed after {attempts} attempts: {url}: {last_error}"
    )


def _union_duration(intervals: Sequence[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    starts = row.get("timestamps_start")
    ends = row.get("timestamps_end")
    speakers = row.get("speakers")
    if (
        not isinstance(starts, list)
        or not isinstance(ends, list)
        or not isinstance(speakers, list)
        or not starts
        or len(starts) != len(ends)
        or len(starts) != len(speakers)
    ):
        raise GlobalSampleLibraryError("diarization row annotations are invalid")
    turns: list[dict[str, Any]] = []
    for index, (start, end, speaker) in enumerate(zip(starts, ends, speakers)):
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or float(start) < 0
            or float(end) <= float(start)
            or not isinstance(speaker, str)
            or not speaker
        ):
            raise GlobalSampleLibraryError(
                f"diarization row turn {index} is invalid"
            )
        turns.append(
            {
                "speakerId": speaker,
                "startSeconds": float(start),
                "endSeconds": float(end),
            }
        )
    return turns


def parse_aishell4_rttm(
    value: str,
    recording_id: str,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if (
            len(fields) != 10
            or fields[0] != "SPEAKER"
            or fields[1] != recording_id
            or fields[2] != "1"
        ):
            raise GlobalSampleLibraryError(
                f"AISHELL-4 RTTM line {line_number} is invalid"
            )
        try:
            start = float(fields[3])
            duration = float(fields[4])
        except ValueError as exc:
            raise GlobalSampleLibraryError(
                f"AISHELL-4 RTTM line {line_number} has invalid time"
            ) from exc
        speaker = fields[7]
        if start < 0 or duration <= 0 or not speaker:
            raise GlobalSampleLibraryError(
                f"AISHELL-4 RTTM line {line_number} has invalid turn"
            )
        turns.append(
            {
                "speakerId": speaker,
                "startSeconds": start,
                "endSeconds": start + duration,
            }
        )
    if not turns:
        raise GlobalSampleLibraryError("AISHELL-4 RTTM has no turns")
    return sorted(
        turns,
        key=lambda turn: (
            float(turn["startSeconds"]),
            float(turn["endSeconds"]),
            str(turn["speakerId"]),
        ),
    )


def parse_aishell4_stm(
    value: str,
    recording_id: str,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[0] != recording_id:
            raise GlobalSampleLibraryError(
                f"AISHELL-4 STM line {line_number} is invalid"
            )
        try:
            start = float(fields[3])
            end = float(fields[4])
        except ValueError as exc:
            raise GlobalSampleLibraryError(
                f"AISHELL-4 STM line {line_number} has invalid time"
            ) from exc
        transcript = "".join(fields[5].split())
        if start < 0 or end <= start or not fields[2] or not transcript:
            raise GlobalSampleLibraryError(
                f"AISHELL-4 STM line {line_number} has invalid turn"
            )
        turns.append(
            {
                "speakerId": fields[2],
                "startSeconds": start,
                "endSeconds": end,
                "transcript": transcript,
            }
        )
    if not turns:
        raise GlobalSampleLibraryError("AISHELL-4 STM has no turns")
    return sorted(
        turns,
        key=lambda turn: (
            float(turn["startSeconds"]),
            float(turn["endSeconds"]),
            str(turn["speakerId"]),
        ),
    )


def parse_aishell4_textgrid_audio_tier(value: str) -> list[dict[str, Any]]:
    lines = value.splitlines()
    in_first_item = False
    intervals: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    expected_count: int | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if line == "item [1]:":
            in_first_item = True
            continue
        if in_first_item and line.startswith("item [") and line != "item [1]:":
            break
        if not in_first_item:
            continue
        if line.startswith("intervals: size = "):
            try:
                expected_count = int(line.rsplit("=", 1)[1].strip())
            except ValueError as exc:
                raise GlobalSampleLibraryError(
                    "AISHELL-4 TextGrid interval count is invalid"
                ) from exc
            continue
        if line.startswith("intervals [") and line.endswith("]:"):
            if current is not None:
                intervals.append(current)
            try:
                index = int(line.removeprefix("intervals [").removesuffix("]:"))
            except ValueError as exc:
                raise GlobalSampleLibraryError(
                    "AISHELL-4 TextGrid interval index is invalid"
                ) from exc
            current = {"index": index}
            continue
        if current is None:
            continue
        if line.startswith("xmin = "):
            current["startSeconds"] = float(line.rsplit("=", 1)[1].strip())
        elif line.startswith("xmax = "):
            current["endSeconds"] = float(line.rsplit("=", 1)[1].strip())
        elif line.startswith('text = "') and line.endswith('"'):
            current["text"] = line[len('text = "') : -1].replace('""', '"')
    if current is not None:
        intervals.append(current)
    if expected_count is None or len(intervals) != expected_count:
        raise GlobalSampleLibraryError(
            "AISHELL-4 TextGrid first tier interval count does not match"
        )
    previous_end: float | None = None
    for expected_index, interval in enumerate(intervals, start=1):
        if set(interval) != {"index", "startSeconds", "endSeconds", "text"}:
            raise GlobalSampleLibraryError(
                "AISHELL-4 TextGrid interval is incomplete"
            )
        start = float(interval["startSeconds"])
        end = float(interval["endSeconds"])
        if (
            interval["index"] != expected_index
            or start < 0
            or end <= start
            or (
                previous_end is not None
                and abs(start - previous_end) > 0.000_001
            )
        ):
            raise GlobalSampleLibraryError(
                "AISHELL-4 TextGrid first tier is not contiguous"
            )
        previous_end = end
    return intervals


def _clip_reference_transcript(
    turns: Sequence[dict[str, Any]],
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    start = float(window["sourceStartSeconds"])
    end = float(window["sourceEndSeconds"])
    clipped = [
        {
            "speakerId": str(turn["speakerId"]),
            "sourceStartSeconds": round(
                max(float(turn["startSeconds"]), start),
                6,
            ),
            "sourceEndSeconds": round(
                min(float(turn["endSeconds"]), end),
                6,
            ),
            "startSeconds": round(
                max(float(turn["startSeconds"]), start) - start,
                6,
            ),
            "endSeconds": round(
                min(float(turn["endSeconds"]), end) - start,
                6,
            ),
            "transcript": str(turn["transcript"]),
        }
        for turn in turns
        if float(turn["startSeconds"]) < end
        and float(turn["endSeconds"]) > start
    ]
    if not clipped:
        raise GlobalSampleLibraryError(
            "AISHELL-4 selected window has no STM transcript"
        )
    return clipped


def select_diarization_window(
    turns: Sequence[dict[str, Any]],
    target_speaker_count: int,
) -> dict[str, Any]:
    """Choose a deterministic short window with exactly the target speaker set."""

    if target_speaker_count < 1 or not turns:
        raise GlobalSampleLibraryError("window target and turns must be non-empty")
    total_duration = max(float(turn["endSeconds"]) for turn in turns)
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for duration in WINDOW_DURATIONS:
        candidate_starts = {0.0, max(0.0, total_duration - duration)}
        for turn in turns:
            start = float(turn["startSeconds"])
            end = float(turn["endSeconds"])
            candidate_starts.add(
                max(0.0, min(start, total_duration - duration))
            )
            candidate_starts.add(
                max(0.0, min(end - duration, total_duration - duration))
            )
        for start in sorted(candidate_starts):
            end = min(total_duration, start + duration)
            clipped = [
                {
                    "speakerId": str(turn["speakerId"]),
                    "startSeconds": max(float(turn["startSeconds"]), start),
                    "endSeconds": min(float(turn["endSeconds"]), end),
                }
                for turn in turns
                if float(turn["startSeconds"]) < end
                and float(turn["endSeconds"]) > start
            ]
            speaker_set = sorted({turn["speakerId"] for turn in clipped})
            if len(speaker_set) != target_speaker_count:
                continue
            per_speaker = {
                speaker: sum(
                    float(turn["endSeconds"]) - float(turn["startSeconds"])
                    for turn in clipped
                    if turn["speakerId"] == speaker
                )
                for speaker in speaker_set
            }
            if min(per_speaker.values()) < MIN_SPEAKER_SECONDS:
                continue
            relative_turns = [
                {
                    "speakerId": turn["speakerId"],
                    "sourceStartSeconds": round(
                        float(turn["startSeconds"]),
                        6,
                    ),
                    "sourceEndSeconds": round(float(turn["endSeconds"]), 6),
                    "startSeconds": round(
                        float(turn["startSeconds"]) - start,
                        6,
                    ),
                    "endSeconds": round(float(turn["endSeconds"]) - start, 6),
                    "transcript": None,
                }
                for turn in clipped
            ]
            overlaps = overlap_intervals(relative_turns)
            overlap_duration = sum(
                float(interval["endSeconds"])
                - float(interval["startSeconds"])
                for interval in overlaps
            )
            speech_duration = _union_duration(
                [
                    (
                        float(turn["startSeconds"]),
                        float(turn["endSeconds"]),
                    )
                    for turn in clipped
                ]
            )
            # Keep real multi-speaker cases short enough for bounded local runs.
            # Once a window is short enough and covers every target speaker,
            # prefer meaningful per-speaker coverage and overlap within that
            # duration rather than stretching the sample to maximize overlap.
            score = (
                -duration,
                min(per_speaker.values()),
                bool(overlaps),
                overlap_duration,
                speech_duration,
                -start,
            )
            candidate = {
                "algorithm": "event-boundary-shortest-coverage-v2",
                "sourceStartSeconds": round(start, 6),
                "sourceEndSeconds": round(end, 6),
                "durationSeconds": round(end - start, 6),
                "speakerSet": speaker_set,
                "perSpeakerAnnotatedSeconds": {
                    speaker: round(value, 6)
                    for speaker, value in sorted(per_speaker.items())
                },
                "annotatedSpeechSeconds": round(speech_duration, 6),
                "annotatedOverlapSeconds": round(overlap_duration, 6),
                "turns": relative_turns,
                "overlapIntervals": overlaps,
            }
            if best is None or score > best[0]:
                best = (score, candidate)
    if best is None:
        raise GlobalSampleLibraryError(
            f"no <=90s window contains exactly {target_speaker_count} speakers"
        )
    return best[1]


def align_diarization_window_to_stm(
    window: dict[str, Any],
    diarization_turns: Sequence[dict[str, Any]],
    transcript_turns: Sequence[dict[str, Any]],
    target_speaker_count: int,
) -> dict[str, Any]:
    """Expand a selected window until no official STM utterance is clipped."""

    start = float(window["sourceStartSeconds"])
    end = float(window["sourceEndSeconds"])
    for _ in range(len(transcript_turns) + 1):
        overlapping = [
            turn
            for turn in transcript_turns
            if float(turn["startSeconds"]) < end
            and float(turn["endSeconds"]) > start
        ]
        if not overlapping:
            raise GlobalSampleLibraryError(
                "AISHELL-4 selected window has no overlapping STM turns"
            )
        aligned_start = min(
            [start] + [float(turn["startSeconds"]) for turn in overlapping]
        )
        aligned_end = max(
            [end] + [float(turn["endSeconds"]) for turn in overlapping]
        )
        if aligned_start == start and aligned_end == end:
            break
        start, end = aligned_start, aligned_end
    else:
        raise GlobalSampleLibraryError(
            "AISHELL-4 STM boundary expansion did not converge"
        )
    if end - start > AISHELL4_MAX_DURATION:
        raise GlobalSampleLibraryError(
            "AISHELL-4 STM-aligned window exceeds 30 seconds"
        )
    clipped = [
        {
            "speakerId": str(turn["speakerId"]),
            "startSeconds": max(float(turn["startSeconds"]), start),
            "endSeconds": min(float(turn["endSeconds"]), end),
        }
        for turn in diarization_turns
        if float(turn["startSeconds"]) < end
        and float(turn["endSeconds"]) > start
    ]
    speaker_set = sorted({str(turn["speakerId"]) for turn in clipped})
    if len(speaker_set) != target_speaker_count:
        raise GlobalSampleLibraryError(
            "AISHELL-4 STM expansion changed the target speaker count"
        )
    per_speaker = {
        speaker: sum(
            float(turn["endSeconds"]) - float(turn["startSeconds"])
            for turn in clipped
            if turn["speakerId"] == speaker
        )
        for speaker in speaker_set
    }
    if min(per_speaker.values()) < MIN_SPEAKER_SECONDS:
        raise GlobalSampleLibraryError(
            "AISHELL-4 STM expansion lacks per-speaker coverage"
        )
    relative_turns = [
        {
            "speakerId": turn["speakerId"],
            "sourceStartSeconds": round(float(turn["startSeconds"]), 6),
            "sourceEndSeconds": round(float(turn["endSeconds"]), 6),
            "startSeconds": round(float(turn["startSeconds"]) - start, 6),
            "endSeconds": round(float(turn["endSeconds"]) - start, 6),
            "transcript": None,
        }
        for turn in clipped
    ]
    overlaps = overlap_intervals(relative_turns)
    overlap_duration = sum(
        float(interval["endSeconds"]) - float(interval["startSeconds"])
        for interval in overlaps
    )
    speech_duration = _union_duration(
        [
            (
                float(turn["startSeconds"]),
                float(turn["endSeconds"]),
            )
            for turn in clipped
        ]
    )
    return {
        "algorithm": (
            "event-boundary-shortest-coverage-v2"
            "+stm-whole-utterance-expansion-v1"
        ),
        "baseSourceStartSeconds": window["sourceStartSeconds"],
        "baseSourceEndSeconds": window["sourceEndSeconds"],
        "baseDurationSeconds": window["durationSeconds"],
        "sourceStartSeconds": round(start, 6),
        "sourceEndSeconds": round(end, 6),
        "durationSeconds": round(end - start, 6),
        "speakerSet": speaker_set,
        "perSpeakerAnnotatedSeconds": {
            speaker: round(value, 6)
            for speaker, value in sorted(per_speaker.items())
        },
        "annotatedSpeechSeconds": round(speech_duration, 6),
        "annotatedOverlapSeconds": round(overlap_duration, 6),
        "turns": relative_turns,
        "overlapIntervals": overlaps,
    }


def _clip_audio(source: Path, output: Path, window: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.wav")
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{float(window['sourceStartSeconds']):.6f}",
            "-t",
            f"{float(window['durationSeconds']):.6f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        raise GlobalSampleLibraryError(
            f"ffmpeg failed for {output.name}: {completed.stderr.strip()}"
        )
    temporary.replace(output)


def _probe_audio(path: Path) -> dict[str, Any]:
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


def _case_record(
    *,
    case_id: str,
    source_id: str,
    source_dataset: str,
    source_revision: str,
    source_row_index: int,
    source_path: Path,
    source_artifact_path: Path,
    output_root: Path,
    window: dict[str, Any],
) -> dict[str, Any]:
    output = output_root / "audio" / f"{case_id}.wav"
    _clip_audio(source_path, output, window)
    probe = _probe_audio(output)
    if (
        probe["codec"] != "pcm_s16le"
        or probe["sampleRate"] != 16_000
        or probe["channels"] != 1
        or probe["durationSeconds"] > 90.05
    ):
        raise GlobalSampleLibraryError(f"{case_id} normalized audio is invalid")
    return {
        "id": case_id,
        "sourceId": source_id,
        "sourceDataset": source_dataset,
        "sourceRevision": source_revision,
        "sourceRowIndex": source_row_index,
        "sourceArtifactPath": str(source_artifact_path.relative_to(output_root)),
        "sourceArtifactSha256": _sha256(source_artifact_path),
        "sourceAudioPath": str(source_path.relative_to(output_root)),
        "sourceAudioSha256": _sha256(source_path),
        "path": str(output.relative_to(output_root)),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "audio": probe,
        "realOrSynthetic": "real-recording",
        "expectedSpeakerCount": len(window["speakerSet"]),
        "speakerSet": window["speakerSet"],
        "windowSelection": {
            key: value
            for key, value in window.items()
            if key not in {"turns", "overlapIntervals"}
        },
        "turns": window["turns"],
        "overlapIntervals": window["overlapIntervals"],
        "transcript": None,
        "truthEligibility": {
            "speakerCount": True,
            "turnBoundaries": True,
            "overlap": True,
            "derJer": True,
            "asr": False,
        },
    }


def _clip_audio_shards(
    shards: Sequence[Path],
    output: Path,
    *,
    source_span_start: float,
    window: dict[str, Any],
) -> dict[str, Any]:
    if not shards:
        raise GlobalSampleLibraryError("AISHELL-4 audio shard list is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.wav")
    concat_path = output.with_suffix(output.suffix + ".concat.txt")
    resolved_shards = [path.resolve() for path in shards]
    if any("'" in str(path) for path in resolved_shards):
        raise GlobalSampleLibraryError(
            "AISHELL-4 audio shard path contains an unsupported quote"
        )
    concat_path.write_text(
        "".join(f"file '{path}'\n" for path in resolved_shards),
        encoding="utf-8",
    )
    relative_start = float(window["sourceStartSeconds"]) - source_span_start
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-ss",
                f"{relative_start:.9f}",
                "-t",
                f"{float(window['durationSeconds']):.9f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=240,
        )
    finally:
        concat_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise GlobalSampleLibraryError(
            f"ffmpeg failed for {output.name}: {completed.stderr.strip()}"
        )
    temporary.replace(output)
    probe = _probe_audio(output)
    if (
        probe["codec"] != "pcm_s16le"
        or probe["sampleRate"] != 16_000
        or probe["channels"] != 1
        or abs(
            probe["durationSeconds"] - float(window["durationSeconds"])
        )
        > 0.05
    ):
        raise GlobalSampleLibraryError(
            f"AISHELL-4 normalized audio is invalid: {output.name}"
        )
    return probe


def _download_aishell4_annotations(
    *,
    source: Any,
    plan: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    split = plan.get("split")
    session_id = plan.get("sessionId")
    official_repository = plan.get("officialEvaluationRepository")
    official_revision = plan.get("officialEvaluationRevision")
    if (
        split != "test"
        or not isinstance(session_id, str)
        or not session_id
        or official_repository != "https://github.com/felixfuyihui/AISHELL-4"
        or not isinstance(official_revision, str)
        or len(official_revision) != 40
    ):
        raise GlobalSampleLibraryError("AISHELL-4 source plan is invalid")
    source_root = output_root / "sources" / "aishell4"
    paths = {
        "rttm": source_root / f"{session_id}.rttm",
        "textgrid": source_root / f"{session_id}.TextGrid",
        "stm": source_root / f"{session_id}.stm",
    }
    hf_prefix = (
        f"https://huggingface.co/datasets/{source.dataset}/resolve/"
        f"{source.revision}/{split}/TextGrid/{session_id}"
    )
    _download_resumable(f"{hf_prefix}.rttm", paths["rttm"])
    _download_resumable(f"{hf_prefix}.TextGrid", paths["textgrid"])
    _download_resumable(
        "https://raw.githubusercontent.com/felixfuyihui/AISHELL-4/"
        f"{official_revision}/eval/stm/{session_id}.stm",
        paths["stm"],
    )
    evidence = {
        key: {
            "path": str(path.relative_to(output_root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for key, path in paths.items()
    }
    return paths, evidence


def _aishell4_audio_tree(
    *,
    source: Any,
    split: str,
    session_id: str,
) -> list[dict[str, Any]]:
    tree_url = (
        f"https://huggingface.co/api/datasets/{source.dataset}/tree/"
        f"{source.revision}/{split}/wav/{session_id}"
        "?recursive=false&expand=false&limit=1000"
    )
    entries = _request_json_array(tree_url)
    expected_prefix = f"{split}/wav/{session_id}/"
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        lfs = entry.get("lfs")
        path = entry.get("path")
        size = entry.get("size")
        if (
            entry.get("type") != "file"
            or not isinstance(path, str)
            or not path.startswith(expected_prefix)
            or not path.endswith(".wav")
            or not isinstance(size, int)
            or size <= 44
            or not isinstance(lfs, dict)
            or not isinstance(lfs.get("oid"), str)
            or len(lfs["oid"]) != 64
            or lfs.get("size") != size
        ):
            raise GlobalSampleLibraryError(
                "AISHELL-4 audio tree contains an invalid entry"
            )
        normalized.append(
            {
                "path": path,
                "bytes": size,
                "sha256": lfs["oid"],
            }
        )
    return sorted(normalized, key=lambda entry: str(entry["path"]))


def _build_aishell4(
    manifest: Any,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _source_by_id(manifest, "aishell4")
    plan = _planned_by_id(manifest, "aishell4")
    session_id = plan.get("sessionId")
    split = plan.get("split")
    targets = plan.get("targetSpeakerCounts")
    if (
        not isinstance(session_id, str)
        or split != "test"
        or not isinstance(targets, list)
        or not targets
        or any(
            isinstance(target, bool)
            or not isinstance(target, int)
            or target < 2
            for target in targets
        )
    ):
        raise GlobalSampleLibraryError("AISHELL-4 target plan is invalid")
    paths, annotation_evidence = _download_aishell4_annotations(
        source=source,
        plan=plan,
        output_root=output_root,
    )
    rttm_turns = parse_aishell4_rttm(
        paths["rttm"].read_text(encoding="utf-8"),
        session_id,
    )
    stm_turns = parse_aishell4_stm(
        paths["stm"].read_text(encoding="utf-8"),
        session_id,
    )
    audio_tier = parse_aishell4_textgrid_audio_tier(
        paths["textgrid"].read_text(encoding="utf-8")
    )
    tree = _aishell4_audio_tree(
        source=source,
        split=split,
        session_id=session_id,
    )
    if len(tree) != len(audio_tier):
        raise GlobalSampleLibraryError(
            "AISHELL-4 audio shard count does not match TextGrid first tier"
        )
    source_root = output_root / "sources" / "aishell4" / session_id
    downloaded_shards: dict[str, dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []
    for target in targets:
        base_window = select_diarization_window(rttm_turns, target)
        window = align_diarization_window_to_stm(
            base_window,
            rttm_turns,
            stm_turns,
            target,
        )
        selected_intervals = [
            interval
            for interval in audio_tier
            if float(interval["startSeconds"])
            < float(window["sourceEndSeconds"])
            and float(interval["endSeconds"])
            > float(window["sourceStartSeconds"])
        ]
        if not selected_intervals:
            raise GlobalSampleLibraryError(
                "AISHELL-4 window does not intersect source audio shards"
            )
        local_shards: list[Path] = []
        shard_evidence: list[dict[str, Any]] = []
        for interval in selected_intervals:
            shard_index = int(interval["index"]) - 1
            entry = tree[shard_index]
            expected_name = f"{shard_index:09d}.wav"
            if Path(str(entry["path"])).name != expected_name:
                raise GlobalSampleLibraryError(
                    "AISHELL-4 audio shard order is not canonical"
                )
            local_path = source_root / expected_name
            source_url = (
                f"https://huggingface.co/datasets/{source.dataset}/resolve/"
                f"{source.revision}/{entry['path']}"
            )
            _download_resumable(
                source_url,
                local_path,
                expected_bytes=int(entry["bytes"]),
            )
            actual_sha256 = _sha256(local_path)
            if actual_sha256 != entry["sha256"]:
                raise GlobalSampleLibraryError(
                    f"AISHELL-4 audio shard hash mismatch: {expected_name}"
                )
            evidence = {
                "path": str(local_path.relative_to(output_root)),
                "bytes": local_path.stat().st_size,
                "sha256": actual_sha256,
                "sourceStartSeconds": round(
                    float(interval["startSeconds"]),
                    9,
                ),
                "sourceEndSeconds": round(
                    float(interval["endSeconds"]),
                    9,
                ),
            }
            downloaded_shards[expected_name] = evidence
            shard_evidence.append(evidence)
            local_shards.append(local_path)
        case_id = f"aishell4-test-{session_id.lower()}-n{target}"
        output = output_root / "audio" / f"{case_id}.wav"
        probe = _clip_audio_shards(
            local_shards,
            output,
            source_span_start=float(selected_intervals[0]["startSeconds"]),
            window=window,
        )
        reference_turns = _clip_reference_transcript(stm_turns, window)
        scoring_transcript = "".join(
            str(turn["transcript"]) for turn in reference_turns
        )
        cases.append(
            {
                "id": case_id,
                "sourceId": source.source_id,
                "sourceDataset": source.dataset,
                "sourceRevision": source.revision,
                "sourceSessionId": session_id,
                "sourceArtifactPath": annotation_evidence["rttm"]["path"],
                "sourceArtifactSha256": annotation_evidence["rttm"]["sha256"],
                "sourceAudioShards": shard_evidence,
                "path": str(output.relative_to(output_root)),
                "bytes": output.stat().st_size,
                "sha256": _sha256(output),
                "audio": probe,
                "realOrSynthetic": "real-recording",
                "language": "zh-CN",
                "region": "East Asia",
                "evaluationSplit": "held-out",
                "scenario": [
                    "real-recording",
                    "far-field-meeting",
                    "multichannel-source",
                    "overlap",
                    "rapid-turns",
                ],
                "expectedSpeakerCount": len(window["speakerSet"]),
                "speakerSet": window["speakerSet"],
                "windowSelection": {
                    **{
                        key: value
                        for key, value in window.items()
                        if key not in {"turns", "overlapIntervals"}
                    },
                    "selectionUsesModelScores": False,
                },
                "turns": window["turns"],
                "overlapIntervals": window["overlapIntervals"],
                "referenceTranscriptTurns": reference_turns,
                "scoringTranscript": scoring_transcript,
                "asrReferenceMode": (
                    "official-stm-whole-utterances-serialized-by-"
                    "start-end-speaker-v1"
                ),
                "transcript": scoring_transcript,
                "truthEligibility": {
                    "speakerCount": True,
                    "turnBoundaries": True,
                    "overlap": True,
                    "derJer": True,
                    "asr": True,
                    "language": True,
                },
            }
        )
    return cases, {
        "sourceId": source.source_id,
        "dataset": source.dataset,
        "revision": source.revision,
        "license": source.license,
        "licenseDecision": (
            "CC BY-SA 4.0 is applied from the upstream OpenSLR 111 data "
            "notice because it is stricter than the conflicting Apache-2.0 "
            "Hugging Face card metadata."
        ),
        "attribution": source.attribution,
        "split": split,
        "sessionId": session_id,
        "audioLayout": plan["audioLayout"],
        "annotations": annotation_evidence,
        "officialEvaluationRepository": plan["officialEvaluationRepository"],
        "officialEvaluationRevision": plan["officialEvaluationRevision"],
        "audioShards": [
            downloaded_shards[key] for key in sorted(downloaded_shards)
        ],
    }


def _source_by_id(manifest: Any, source_id: str) -> Any:
    for source in manifest.sources:
        if source.source_id == source_id:
            return source
    raise GlobalSampleLibraryError(f"source is not declared: {source_id}")


def _planned_by_id(manifest: Any, source_id: str) -> dict[str, Any]:
    for plan in manifest.planned_real_diarization_sources:
        if isinstance(plan, dict) and plan.get("sourceId") == source_id:
            return plan
    raise GlobalSampleLibraryError(f"real diarization plan is missing: {source_id}")


def _build_ami(
    manifest: Any,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _source_by_id(manifest, "ami")
    plan = _planned_by_id(manifest, "ami")
    params = urllib.parse.urlencode(
        {
            "dataset": source.dataset,
            "config": plan["config"],
            "split": plan["split"],
            "offset": plan["rowIndex"],
            "length": 1,
        }
    )
    response = _request_json(
        f"https://datasets-server.huggingface.co/rows?{params}"
    )
    rows = response.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise GlobalSampleLibraryError("AMI viewer did not return one row")
    wrapped = rows[0]
    if (
        not isinstance(wrapped, dict)
        or wrapped.get("row_idx") != plan["rowIndex"]
        or not isinstance(wrapped.get("row"), dict)
    ):
        raise GlobalSampleLibraryError("AMI row identity is invalid")
    row = wrapped["row"]
    audio = row.get("audio")
    if (
        not isinstance(audio, list)
        or len(audio) != 1
        or not isinstance(audio[0], dict)
        or not isinstance(audio[0].get("src"), str)
    ):
        raise GlobalSampleLibraryError("AMI audio asset is missing")
    audio_url = audio[0]["src"]
    if source.revision not in urllib.parse.unquote(audio_url):
        raise GlobalSampleLibraryError("AMI asset is not bound to pinned revision")
    source_audio = output_root / "sources" / "ami_ihm_test_row_000.wav"
    _download_resumable(audio_url, source_audio)
    metadata_path = output_root / "sources" / "ami_ihm_test_row_000.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": source.dataset,
                "revision": source.revision,
                "config": plan["config"],
                "split": plan["split"],
                "rowIndex": plan["rowIndex"],
                "timestamps_start": row["timestamps_start"],
                "timestamps_end": row["timestamps_end"],
                "speakers": row["speakers"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    turns = _turns(row)
    cases = [
        _case_record(
            case_id=f"ami-ihm-test-row000-n{target}",
            source_id=source.source_id,
            source_dataset=source.dataset,
            source_revision=source.revision,
            source_row_index=plan["rowIndex"],
            source_path=source_audio,
            source_artifact_path=metadata_path,
            output_root=output_root,
            window=select_diarization_window(turns, target),
        )
        for target in plan["targetSpeakerCounts"]
    ]
    return cases, {
        "sourceId": source.source_id,
        "dataset": source.dataset,
        "revision": source.revision,
        "license": source.license,
        "attribution": source.attribution,
        "sourceAudioPath": str(source_audio.relative_to(output_root)),
        "sourceAudioBytes": source_audio.stat().st_size,
        "sourceAudioSha256": _sha256(source_audio),
        "annotationPath": str(metadata_path.relative_to(output_root)),
        "annotationSha256": _sha256(metadata_path),
    }


def _build_voxconverse(
    manifest: Any,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    source = _source_by_id(manifest, "voxconverse")
    plan = _planned_by_id(manifest, "voxconverse")
    parquet_path = plan.get("parquetPath")
    parquet_bytes = plan.get("parquetBytes")
    row_targets = plan.get("rowTargets")
    if (
        not isinstance(parquet_path, str)
        or not isinstance(parquet_bytes, int)
        or not isinstance(row_targets, list)
    ):
        raise GlobalSampleLibraryError("VoxConverse parquet plan is invalid")
    source_url = (
        f"https://huggingface.co/datasets/{source.dataset}/resolve/"
        f"{source.revision}/{parquet_path}"
    )
    local_parquet = output_root / "sources" / Path(parquet_path).name
    _download_resumable(
        source_url,
        local_parquet,
        expected_bytes=parquet_bytes,
    )
    table = pq.read_table(
        local_parquet,
        columns=["audio", "timestamps_start", "timestamps_end", "speakers"],
    )
    cases: list[dict[str, Any]] = []
    for target in row_targets:
        if not isinstance(target, dict):
            raise GlobalSampleLibraryError("VoxConverse row target is invalid")
        row_index = target.get("rowIndex")
        speaker_count = target.get("targetSpeakerCount")
        if (
            not isinstance(row_index, int)
            or not isinstance(speaker_count, int)
            or row_index < 0
            or row_index >= table.num_rows
        ):
            raise GlobalSampleLibraryError("VoxConverse row target is out of range")
        row = table.slice(row_index, 1).to_pylist()[0]
        audio = row.get("audio")
        if (
            not isinstance(audio, dict)
            or not isinstance(audio.get("bytes"), bytes)
            or not audio["bytes"]
        ):
            raise GlobalSampleLibraryError(
                f"VoxConverse row {row_index} audio bytes are missing"
            )
        source_audio = (
            output_root
            / "sources"
            / f"voxconverse_dev_row_{row_index:03d}.wav"
        )
        source_audio.write_bytes(audio["bytes"])
        metadata_path = (
            output_root
            / "sources"
            / f"voxconverse_dev_row_{row_index:03d}.json"
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "dataset": source.dataset,
                    "revision": source.revision,
                    "config": plan["config"],
                    "split": plan["split"],
                    "rowIndex": row_index,
                    "path": audio.get("path"),
                    "timestamps_start": row["timestamps_start"],
                    "timestamps_end": row["timestamps_end"],
                    "speakers": row["speakers"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        turns = _turns(row)
        window = select_diarization_window(turns, speaker_count)
        cases.append(
            _case_record(
                case_id=f"voxconverse-dev-row{row_index:03d}-n{speaker_count}",
                source_id=source.source_id,
                source_dataset=source.dataset,
                source_revision=source.revision,
                source_row_index=row_index,
                source_path=source_audio,
                source_artifact_path=metadata_path,
                output_root=output_root,
                window=window,
            )
        )
    return cases, {
        "sourceId": source.source_id,
        "dataset": source.dataset,
        "revision": source.revision,
        "license": source.license,
        "attribution": source.attribution,
        "parquetPath": str(local_parquet.relative_to(output_root)),
        "parquetBytes": local_parquet.stat().st_size,
        "parquetSha256": _sha256(local_parquet),
    }


def build_real_diarization_library(
    manifest_path: Path,
    output_root: Path,
    source_ids: Sequence[str] | None = None,
) -> Path:
    manifest = load_global_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    builders = {
        "ami": _build_ami,
        "voxconverse": _build_voxconverse,
        "aishell4": _build_aishell4,
    }
    selected = list(source_ids) if source_ids else list(builders)
    unknown = sorted(set(selected) - set(builders))
    if unknown:
        raise GlobalSampleLibraryError(
            f"unknown real diarization source(s): {', '.join(unknown)}"
        )
    if len(selected) != len(set(selected)):
        raise GlobalSampleLibraryError(
            "real diarization sources must not be repeated"
        )
    cases: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for source_id in selected:
        source_cases, source_record = builders[source_id](
            manifest,
            output_root,
        )
        cases.extend(source_cases)
        sources.append(source_record)
    attribution = output_root / "ATTRIBUTION.md"
    attribution_sections = "".join(
        f"## {source['dataset']}\n\n"
        f"- Revision: `{source['revision']}`\n"
        f"- License: `{source['license']}`\n"
        f"- Attribution: {source['attribution']}\n\n"
        for source in sources
    )
    attribution.write_text(
        "# Real Diarization Sample Attribution\n\n" + attribution_sections,
        encoding="utf-8",
    )
    resolved = {
        "schemaVersion": "1.0.0",
        "libraryId": f"{manifest.library_id}-real-diarization",
        "generatedAt": datetime.now(UTC).isoformat(),
        "sourceManifest": str(manifest_path.resolve()),
        "sourceManifestSha256": _sha256(manifest_path),
        "windowSelection": {
            "algorithm": "event-boundary-max-overlap-v1",
            "candidateDurationsSeconds": list(WINDOW_DURATIONS),
            "minimumPerSpeakerAnnotatedSeconds": MIN_SPEAKER_SECONDS,
            "maximumDurationSeconds": manifest.max_duration_seconds,
        },
        "sources": sources,
        "cases": cases,
        "failedCases": [],
        "attributionPath": str(attribution.relative_to(output_root)),
    }
    destination = output_root / RESOLVED_NAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source",
        action="append",
        choices=("ami", "voxconverse", "aishell4"),
        default=[],
        help="build only the selected source; repeat to select multiple",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = build_real_diarization_library(
        args.manifest,
        args.output_root,
        source_ids=args.source,
    )
    resolved = json.loads(destination.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "resolvedManifest": str(destination),
                "caseCount": len(resolved["cases"]),
                "speakerCounts": sorted(
                    {case["expectedSpeakerCount"] for case in resolved["cases"]}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
