"""Build short real diarization windows from pinned public datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_global_derived_matrix import overlap_intervals  # noqa: E402
from backend.persistence import canonical_json_sha256  # noqa: E402
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
MAX_WINDOW_DURATION_SECONDS = 300.0
MIN_SPEAKER_SECONDS = 0.5
AISHELL4_MAX_DURATION = 30.0
EVALUATION_SPLITS = frozenset({"development", "regression", "held-out"})
ALIMEETING_SESSION_ID = re.compile(r"^R[0-9]{4}_M[0-9]{4}$")
ALIMEETING_SPEAKER_ID = re.compile(r"^N_SPK[0-9]{4}$")


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


def _praat_quoted_value(line: str, prefix: str) -> str:
    raw = line.removeprefix(prefix).strip()
    if len(raw) < 2 or not raw.startswith('"') or not raw.endswith('"'):
        raise GlobalSampleLibraryError("AliMeeting TextGrid string is invalid")
    return raw[1:-1].replace('""', '"')


def parse_alimeeting_textgrid(value: str) -> dict[str, Any]:
    """Parse the deterministic long TextGrid form shipped in AliMeeting."""

    grid: dict[str, Any] = {"tiers": []}
    expected_tier_count: int | None = None
    current_tier: dict[str, Any] | None = None
    current_interval: dict[str, Any] | None = None

    def finish_interval() -> None:
        nonlocal current_interval
        if current_interval is None:
            return
        if current_tier is None:
            raise GlobalSampleLibraryError(
                "AliMeeting TextGrid interval has no tier"
            )
        current_tier["intervals"].append(current_interval)
        current_interval = None

    def finish_tier() -> None:
        nonlocal current_tier
        finish_interval()
        if current_tier is not None:
            grid["tiers"].append(current_tier)
            current_tier = None

    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        item_match = re.fullmatch(r"item \[([0-9]+)\]:", line)
        if item_match:
            finish_tier()
            current_tier = {
                "index": int(item_match.group(1)),
                "intervals": [],
            }
            continue
        interval_match = re.fullmatch(r"intervals \[([0-9]+)\]:", line)
        if interval_match:
            if current_tier is None:
                raise GlobalSampleLibraryError(
                    "AliMeeting TextGrid interval precedes its tier"
                )
            finish_interval()
            current_interval = {"index": int(interval_match.group(1))}
            continue
        try:
            if current_tier is None:
                if line.startswith("xmin = "):
                    grid["startSeconds"] = float(line.rsplit("=", 1)[1])
                elif line.startswith("xmax = "):
                    grid["endSeconds"] = float(line.rsplit("=", 1)[1])
                elif line.startswith("size = "):
                    expected_tier_count = int(line.rsplit("=", 1)[1])
                continue
            if current_interval is not None:
                if line.startswith("xmin = "):
                    current_interval["startSeconds"] = float(
                        line.rsplit("=", 1)[1]
                    )
                elif line.startswith("xmax = "):
                    current_interval["endSeconds"] = float(
                        line.rsplit("=", 1)[1]
                    )
                elif line.startswith("text = "):
                    current_interval["text"] = _praat_quoted_value(
                        line,
                        "text = ",
                    )
                continue
            if line.startswith("class = "):
                current_tier["class"] = _praat_quoted_value(line, "class = ")
            elif line.startswith("name = "):
                current_tier["name"] = _praat_quoted_value(line, "name = ")
            elif line.startswith("xmin = "):
                current_tier["startSeconds"] = float(line.rsplit("=", 1)[1])
            elif line.startswith("xmax = "):
                current_tier["endSeconds"] = float(line.rsplit("=", 1)[1])
            elif line.startswith("intervals: size = "):
                current_tier["expectedIntervalCount"] = int(
                    line.rsplit("=", 1)[1]
                )
        except ValueError as exc:
            raise GlobalSampleLibraryError(
                f"AliMeeting TextGrid line {line_number} has invalid numeric data"
            ) from exc
    finish_tier()

    tiers = grid["tiers"]
    if (
        set(grid) != {"startSeconds", "endSeconds", "tiers"}
        or expected_tier_count is None
        or expected_tier_count < 1
        or len(tiers) != expected_tier_count
        or float(grid["startSeconds"]) != 0.0
        or float(grid["endSeconds"]) <= 0.0
    ):
        raise GlobalSampleLibraryError(
            "AliMeeting TextGrid header or tier count is invalid"
        )
    for expected_tier_index, tier in enumerate(tiers, start=1):
        required_tier_fields = {
            "index",
            "class",
            "name",
            "startSeconds",
            "endSeconds",
            "expectedIntervalCount",
            "intervals",
        }
        if (
            set(tier) != required_tier_fields
            or tier["index"] != expected_tier_index
            or tier["class"] != "IntervalTier"
            or not isinstance(tier["name"], str)
            or not tier["name"]
            or float(tier["startSeconds"]) != 0.0
            or float(tier["endSeconds"]) != float(grid["endSeconds"])
            or tier["expectedIntervalCount"] != len(tier["intervals"])
            or not tier["intervals"]
        ):
            raise GlobalSampleLibraryError(
                "AliMeeting TextGrid tier is invalid"
            )
        previous_end = 0.0
        for expected_interval_index, interval in enumerate(
            tier["intervals"],
            start=1,
        ):
            if set(interval) != {
                "index",
                "startSeconds",
                "endSeconds",
                "text",
            }:
                raise GlobalSampleLibraryError(
                    "AliMeeting TextGrid interval is incomplete"
                )
            start = float(interval["startSeconds"])
            end = float(interval["endSeconds"])
            if (
                interval["index"] != expected_interval_index
                or start < previous_end
                or end <= start
                or end > float(grid["endSeconds"])
                or not isinstance(interval["text"], str)
            ):
                raise GlobalSampleLibraryError(
                    "AliMeeting TextGrid interval is invalid"
                )
            previous_end = end
        tier.pop("expectedIntervalCount")
    return grid


def alimeeting_textgrid_turns(grid: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for tier in grid["tiers"]:
        speaker_id = str(tier["name"])
        if not ALIMEETING_SPEAKER_ID.fullmatch(speaker_id):
            raise GlobalSampleLibraryError(
                "AliMeeting far TextGrid speaker ID is invalid"
            )
        for interval in tier["intervals"]:
            transcript = str(interval["text"]).strip()
            if not transcript:
                raise GlobalSampleLibraryError(
                    "AliMeeting far TextGrid transcript is empty"
                )
            turns.append(
                {
                    "speakerId": speaker_id,
                    "startSeconds": float(interval["startSeconds"]),
                    "endSeconds": float(interval["endSeconds"]),
                    "transcript": transcript,
                }
            )
    return sorted(
        turns,
        key=lambda turn: (
            float(turn["startSeconds"]),
            float(turn["endSeconds"]),
            str(turn["speakerId"]),
        ),
    )


def align_alimeeting_window_to_textgrid(
    window: dict[str, Any],
    transcript_turns: Sequence[dict[str, Any]],
    target_speaker_count: int,
    *,
    maximum_duration_seconds: float = 90.0,
) -> dict[str, Any]:
    """Expand a selected window to whole official TextGrid utterances."""

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
                "AliMeeting selected window has no TextGrid transcript"
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
            "AliMeeting TextGrid boundary expansion did not converge"
        )
    if end - start > maximum_duration_seconds:
        raise GlobalSampleLibraryError(
            "AliMeeting TextGrid-aligned window exceeds the duration limit"
        )
    selected = [
        turn
        for turn in transcript_turns
        if float(turn["startSeconds"]) < end
        and float(turn["endSeconds"]) > start
    ]
    if any(
        float(turn["startSeconds"]) < start
        or float(turn["endSeconds"]) > end
        for turn in selected
    ):
        raise GlobalSampleLibraryError(
            "AliMeeting TextGrid alignment clipped an utterance"
        )
    speaker_set = sorted({str(turn["speakerId"]) for turn in selected})
    if len(speaker_set) != target_speaker_count:
        raise GlobalSampleLibraryError(
            "AliMeeting TextGrid expansion changed the target speaker count"
        )
    per_speaker = {
        speaker: sum(
            float(turn["endSeconds"]) - float(turn["startSeconds"])
            for turn in selected
            if turn["speakerId"] == speaker
        )
        for speaker in speaker_set
    }
    if min(per_speaker.values()) < MIN_SPEAKER_SECONDS:
        raise GlobalSampleLibraryError(
            "AliMeeting TextGrid window lacks per-speaker coverage"
        )
    relative_turns = [
        {
            "speakerId": str(turn["speakerId"]),
            "sourceStartSeconds": round(float(turn["startSeconds"]), 6),
            "sourceEndSeconds": round(float(turn["endSeconds"]), 6),
            "startSeconds": round(float(turn["startSeconds"]) - start, 6),
            "endSeconds": round(float(turn["endSeconds"]) - start, 6),
            "transcript": None,
        }
        for turn in selected
    ]
    overlaps = overlap_intervals(relative_turns)
    overlap_duration = sum(
        float(interval["endSeconds"]) - float(interval["startSeconds"])
        for interval in overlaps
    )
    speech_duration = _union_duration(
        [
            (float(turn["startSeconds"]), float(turn["endSeconds"]))
            for turn in selected
        ]
    )
    return {
        "algorithm": (
            "event-boundary-shortest-coverage-v2"
            "+textgrid-whole-utterance-expansion-v1"
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


def select_alimeeting_textgrid_window(
    transcript_turns: Sequence[dict[str, Any]],
    target_speaker_count: int,
    *,
    minimum_duration_seconds: float = 10.0,
    maximum_duration_seconds: float = 90.0,
    minimum_overlap_seconds: float = 0.5,
) -> dict[str, Any]:
    """Select whole TextGrid overlap-components without consulting a model."""

    if (
        target_speaker_count < 2
        or not transcript_turns
        or minimum_duration_seconds <= 0
        or maximum_duration_seconds < minimum_duration_seconds
        or minimum_overlap_seconds < 0
    ):
        raise GlobalSampleLibraryError(
            "AliMeeting TextGrid window constraints are invalid"
        )
    ordered = sorted(
        transcript_turns,
        key=lambda turn: (
            float(turn["startSeconds"]),
            float(turn["endSeconds"]),
            str(turn["speakerId"]),
        ),
    )
    components: list[dict[str, Any]] = []
    for turn in ordered:
        start = float(turn["startSeconds"])
        end = float(turn["endSeconds"])
        if not components or start >= float(components[-1]["endSeconds"]):
            components.append(
                {"startSeconds": start, "endSeconds": end, "turns": [turn]}
            )
        else:
            components[-1]["endSeconds"] = max(
                float(components[-1]["endSeconds"]),
                end,
            )
            components[-1]["turns"].append(turn)
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for first_index, first in enumerate(components):
        selected: list[dict[str, Any]] = []
        start = float(first["startSeconds"])
        for component in components[first_index:]:
            selected.extend(component["turns"])
            end = float(component["endSeconds"])
            duration = end - start
            if duration < minimum_duration_seconds:
                continue
            if duration > maximum_duration_seconds:
                break
            speaker_set = sorted(
                {str(turn["speakerId"]) for turn in selected}
            )
            if len(speaker_set) != target_speaker_count:
                continue
            per_speaker = {
                speaker: sum(
                    float(turn["endSeconds"])
                    - float(turn["startSeconds"])
                    for turn in selected
                    if turn["speakerId"] == speaker
                )
                for speaker in speaker_set
            }
            if min(per_speaker.values()) < MIN_SPEAKER_SECONDS:
                continue
            relative_turns = [
                {
                    "speakerId": str(turn["speakerId"]),
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
                for turn in selected
            ]
            overlaps = overlap_intervals(relative_turns)
            overlap_duration = sum(
                float(interval["endSeconds"])
                - float(interval["startSeconds"])
                for interval in overlaps
            )
            if overlap_duration < minimum_overlap_seconds:
                continue
            speech_duration = _union_duration(
                [
                    (
                        float(turn["startSeconds"]),
                        float(turn["endSeconds"]),
                    )
                    for turn in selected
                ]
            )
            score = (
                -duration,
                min(per_speaker.values()),
                bool(overlaps),
                overlap_duration,
                speech_duration,
                -start,
            )
            candidate = {
                "algorithm": "textgrid-component-shortest-coverage-v1",
                "sourceStartSeconds": round(start, 6),
                "sourceEndSeconds": round(end, 6),
                "durationSeconds": round(duration, 6),
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
            "no whole-utterance AliMeeting window satisfies the constraints"
        )
    return best[1]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_alimeeting_plan(plan: dict[str, Any]) -> None:
    required = {
        "sourceId",
        "provider",
        "dataset",
        "revision",
        "split",
        "evaluationSplit",
        "targetSpeakerCounts",
        "unsupportedTargetSpeakerCounts",
        "modalities",
        "sessionTargets",
        "archiveUrl",
        "archiveBytes",
        "archiveSha256",
        "archiveCrc64",
        "extractedRootName",
        "extractedFileCount",
        "extractedBytes",
        "extractedTreeSha256",
        "officialHomepage",
        "license",
        "licenseDecision",
        "attribution",
        "officialBaselineRepository",
        "officialBaselineRevision",
        "windowSelection",
        "minimumAnnotatedOverlapSeconds",
        "selectionUsesModelScores",
    }
    if not required <= set(plan):
        raise GlobalSampleLibraryError("AliMeeting source plan is incomplete")
    archive_sha256 = plan.get("archiveSha256")
    root_name = plan.get("extractedRootName")
    targets = plan.get("targetSpeakerCounts")
    sessions = plan.get("sessionTargets")
    if (
        plan.get("sourceId") != "alimeeting"
        or plan.get("provider") != "openslr"
        or plan.get("dataset") != "SLR119/AliMeeting"
        or plan.get("revision") != f"sha256:{archive_sha256}"
        or plan.get("split") != "Eval"
        or plan.get("evaluationSplit") not in EVALUATION_SPLITS
        or targets != [2, 3, 4]
        or plan.get("unsupportedTargetSpeakerCounts") != [5]
        or plan.get("modalities")
        != ["far-field-array", "synchronized-near-field-mixture"]
        or not isinstance(sessions, list)
        or len(sessions) != 8
        or plan.get("archiveUrl")
        != (
            "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
            "AliMeeting/openlr/Eval_Ali.tar.gz"
        )
        or isinstance(plan.get("archiveBytes"), bool)
        or not isinstance(plan.get("archiveBytes"), int)
        or int(plan["archiveBytes"]) <= 0
        or not _is_sha256(archive_sha256)
        or not isinstance(plan.get("archiveCrc64"), str)
        or re.fullmatch(r"[0-9A-F]{16}", plan["archiveCrc64"]) is None
        or not isinstance(root_name, str)
        or PurePosixPath(root_name).parts != (root_name,)
        or root_name != "Eval_Ali"
        or isinstance(plan.get("extractedFileCount"), bool)
        or not isinstance(plan.get("extractedFileCount"), int)
        or int(plan["extractedFileCount"]) <= 0
        or isinstance(plan.get("extractedBytes"), bool)
        or not isinstance(plan.get("extractedBytes"), int)
        or int(plan["extractedBytes"]) <= 0
        or not _is_sha256(plan.get("extractedTreeSha256"))
        or plan.get("officialHomepage") != "https://www.openslr.org/119/"
        or plan.get("license") != "cc-by-sa-4.0"
        or not isinstance(plan.get("licenseDecision"), str)
        or not plan["licenseDecision"].strip()
        or not isinstance(plan.get("attribution"), str)
        or not plan["attribution"].strip()
        or plan.get("officialBaselineRepository")
        != "https://github.com/yufan-aslp/AliMeeting"
        or not isinstance(plan.get("officialBaselineRevision"), str)
        or re.fullmatch(
            r"[0-9a-f]{40}",
            plan["officialBaselineRevision"],
        )
        is None
        or plan.get("windowSelection")
        != "textgrid-component-shortest-coverage-v1"
        or plan.get("minimumAnnotatedOverlapSeconds") != 0.5
        or plan.get("selectionUsesModelScores") is not False
    ):
        raise GlobalSampleLibraryError("AliMeeting source plan is invalid")
    session_ids: list[str] = []
    session_counts: list[int] = []
    for target in sessions:
        if (
            not isinstance(target, dict)
            or set(target) != {"sessionId", "speakerCount"}
            or not isinstance(target.get("sessionId"), str)
            or ALIMEETING_SESSION_ID.fullmatch(target["sessionId"]) is None
            or isinstance(target.get("speakerCount"), bool)
            or target.get("speakerCount") not in targets
        ):
            raise GlobalSampleLibraryError(
                "AliMeeting session target is invalid"
            )
        session_ids.append(target["sessionId"])
        session_counts.append(target["speakerCount"])
    if (
        session_ids != sorted(session_ids)
        or len(session_ids) != len(set(session_ids))
        or sorted(set(session_counts)) != targets
    ):
        raise GlobalSampleLibraryError(
            "AliMeeting session targets are not canonical"
        )


def _alimeeting_tree_evidence(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise GlobalSampleLibraryError(
            "AliMeeting extracted root is not a regular directory"
        )
    try:
        entries = sorted(root.rglob("*"), key=lambda path: path.as_posix())
    except OSError as exc:
        raise GlobalSampleLibraryError(
            f"cannot enumerate AliMeeting extracted root: {exc}"
        ) from exc
    files: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    total_bytes = 0
    for path in entries:
        if path.is_symlink():
            raise GlobalSampleLibraryError(
                "AliMeeting extracted root contains a symbolic link"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise GlobalSampleLibraryError(
                "AliMeeting extracted root contains a non-regular entry"
            )
        relative = path.relative_to(root).as_posix()
        file_bytes = path.stat().st_size
        file_sha256 = _sha256(path)
        digest.update(
            f"{relative}\t{file_bytes}\t{file_sha256}\n".encode("utf-8")
        )
        files[relative] = {
            "path": path,
            "bytes": file_bytes,
            "sha256": file_sha256,
        }
        total_bytes += file_bytes
    return {
        "fileCount": len(files),
        "bytes": total_bytes,
        "treeSha256": digest.hexdigest(),
        "files": files,
    }


def _validate_alimeeting_tree(
    root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    evidence = _alimeeting_tree_evidence(root)
    if (
        evidence["fileCount"] != plan["extractedFileCount"]
        or evidence["bytes"] != plan["extractedBytes"]
        or evidence["treeSha256"] != plan["extractedTreeSha256"]
    ):
        raise GlobalSampleLibraryError(
            "AliMeeting extracted tree does not match the pinned source"
        )
    return evidence


def _validated_alimeeting_archive_members(
    archive: Path,
    plan: dict[str, Any],
) -> list[tarfile.TarInfo]:
    if not archive.is_file() or archive.stat().st_size != plan["archiveBytes"]:
        raise GlobalSampleLibraryError(
            "AliMeeting archive size does not match the pinned source"
        )
    if _sha256(archive) != plan["archiveSha256"]:
        raise GlobalSampleLibraryError(
            "AliMeeting archive SHA-256 does not match the pinned source"
        )
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            members = handle.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise GlobalSampleLibraryError(
            f"cannot inspect AliMeeting archive: {exc}"
        ) from exc
    seen: set[str] = set()
    regular_count = 0
    regular_bytes = 0
    for member in members:
        raw_name = member.name
        name = raw_name.rstrip("/") if member.isdir() else raw_name
        raw_parts = name.split("/")
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or any(not part or part in {".", ".."} for part in raw_parts)
            or path.is_absolute()
            or not path.parts
            or path.parts[0] != plan["extractedRootName"]
            or any(ord(character) < 32 for character in name)
            or name in seen
            or not (member.isdir() or member.isreg())
        ):
            raise GlobalSampleLibraryError(
                "AliMeeting archive contains an unsafe path or entry"
            )
        seen.add(name)
        if member.isreg():
            if member.size < 0:
                raise GlobalSampleLibraryError(
                    "AliMeeting archive contains an invalid file size"
                )
            regular_count += 1
            regular_bytes += member.size
    if (
        regular_count != plan["extractedFileCount"]
        or regular_bytes != plan["extractedBytes"]
    ):
        raise GlobalSampleLibraryError(
            "AliMeeting archive inventory does not match the pinned source"
        )
    return members


def _extract_alimeeting_archive(
    archive: Path,
    output_root: Path,
    plan: dict[str, Any],
    members: Sequence[tarfile.TarInfo],
) -> Path:
    corpus_parent = output_root / "sources" / "alimeeting" / "corpus"
    destination = corpus_parent / plan["extractedRootName"]
    if destination.exists():
        _validate_alimeeting_tree(destination, plan)
        return destination
    staging = corpus_parent / f".{plan['extractedRootName']}.part"
    if staging.exists():
        raise GlobalSampleLibraryError(
            "AliMeeting extraction staging directory already exists"
        )
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            for member in members:
                parts = PurePosixPath(member.name.rstrip("/")).parts[1:]
                if not parts:
                    continue
                target = staging.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise GlobalSampleLibraryError(
                        "AliMeeting archive file cannot be read"
                    )
                with source, target.open("xb") as destination_handle:
                    shutil.copyfileobj(source, destination_handle, 1024 * 1024)
        _validate_alimeeting_tree(staging, plan)
        staging.replace(destination)
    except (OSError, tarfile.TarError) as exc:
        raise GlobalSampleLibraryError(
            f"AliMeeting safe extraction failed: {exc}"
        ) from exc
    return destination


def _resolve_alimeeting_corpus(
    *,
    output_root: Path,
    plan: dict[str, Any],
    archive_path: Path | None,
    corpus_root: Path | None,
) -> tuple[Path, dict[str, Any], bool]:
    archive_verified = False
    members: list[tarfile.TarInfo] | None = None
    if archive_path is not None:
        archive_path = archive_path.resolve()
        members = _validated_alimeeting_archive_members(archive_path, plan)
        archive_verified = True
    if corpus_root is not None:
        corpus_root = corpus_root.resolve()
        if (
            corpus_root.name != plan["extractedRootName"]
            and (corpus_root / plan["extractedRootName"]).is_dir()
        ):
            corpus_root = corpus_root / plan["extractedRootName"]
        evidence = _validate_alimeeting_tree(corpus_root, plan)
        return corpus_root, evidence, archive_verified
    if archive_path is None:
        archive_path = (
            output_root
            / "sources"
            / "alimeeting"
            / "Eval_Ali.tar.gz"
        )
        _download_resumable(
            plan["archiveUrl"],
            archive_path,
            expected_bytes=plan["archiveBytes"],
        )
        members = _validated_alimeeting_archive_members(archive_path, plan)
        archive_verified = True
    if members is None:
        raise GlobalSampleLibraryError("AliMeeting archive was not inspected")
    corpus_root = _extract_alimeeting_archive(
        archive_path,
        output_root,
        plan,
        members,
    )
    return (
        corpus_root,
        _validate_alimeeting_tree(corpus_root, plan),
        archive_verified,
    )


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
    *,
    maximum_duration_seconds: float = 90.0,
) -> dict[str, Any]:
    """Choose a deterministic short window with exactly the target speaker set."""

    if (
        target_speaker_count < 1
        or not turns
        or isinstance(maximum_duration_seconds, bool)
        or not isinstance(maximum_duration_seconds, (int, float))
        or float(maximum_duration_seconds) <= 0
        or float(maximum_duration_seconds) > MAX_WINDOW_DURATION_SECONDS
    ):
        raise GlobalSampleLibraryError("window target and turns must be non-empty")
    maximum_duration_seconds = float(maximum_duration_seconds)
    durations = sorted(
        {
            float(duration)
            for duration in WINDOW_DURATIONS
            if float(duration) <= maximum_duration_seconds
        }
        | {maximum_duration_seconds}
    )
    total_duration = max(float(turn["endSeconds"]) for turn in turns)
    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    for duration in durations:
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
            if maximum_duration_seconds != 90.0:
                candidate["selectionMaximumDurationSeconds"] = round(
                    maximum_duration_seconds,
                    6,
                )
            if best is None or score > best[0]:
                best = (score, candidate)
    if best is None:
        raise GlobalSampleLibraryError(
            "no <="
            f"{maximum_duration_seconds:g}s window contains exactly "
            f"{target_speaker_count} speakers"
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


def _clip_synchronized_near_audio(
    sources: Sequence[Path],
    output: Path,
    window: dict[str, Any],
) -> None:
    if len(sources) < 2:
        raise GlobalSampleLibraryError(
            "AliMeeting near-field mixture requires multiple speakers"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.wav")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for source in sources:
        command.extend(
            [
                "-ss",
                f"{float(window['sourceStartSeconds']):.6f}",
                "-t",
                f"{float(window['durationSeconds']):.6f}",
                "-i",
                str(source),
            ]
        )
    labels = "".join(f"[{index}:a]" for index in range(len(sources)))
    command.extend(
        [
            "-filter_complex",
            (
                f"{labels}amix=inputs={len(sources)}:duration=longest:"
                "dropout_transition=0:normalize=1,"
                f"atrim=duration={float(window['durationSeconds']):.6f}[mix]"
            ),
            "-map",
            "[mix]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=240,
    )
    if completed.returncode != 0:
        raise GlobalSampleLibraryError(
            "ffmpeg failed for AliMeeting synchronized near-field mixture: "
            + completed.stderr.strip()
        )
    temporary.replace(output)


def _probe_alimeeting_source_wave(path: Path, expected_channels: int) -> float:
    try:
        probe = _probe_audio(path)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError) as exc:
        raise GlobalSampleLibraryError(
            f"AliMeeting source WAV is invalid: {path.name}: {exc}"
        ) from exc
    if (
        probe["codec"] != "pcm_s16le"
        or probe["channels"] != expected_channels
        or probe["sampleRate"] != 16_000
        or probe["durationSeconds"] <= 0
    ):
        raise GlobalSampleLibraryError(
            f"AliMeeting source WAV format is invalid: {path.name}"
        )
    return float(probe["durationSeconds"])


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
    evaluation_split: str,
    maximum_duration_seconds: float = 90.0,
) -> dict[str, Any]:
    if evaluation_split not in EVALUATION_SPLITS:
        raise GlobalSampleLibraryError(
            f"{case_id} evaluation split is invalid"
        )
    if (
        isinstance(maximum_duration_seconds, bool)
        or not isinstance(maximum_duration_seconds, (int, float))
        or float(maximum_duration_seconds) <= 0
        or float(maximum_duration_seconds) > MAX_WINDOW_DURATION_SECONDS
    ):
        raise GlobalSampleLibraryError(
            f"{case_id} maximum duration is invalid"
        )
    output = output_root / "audio" / f"{case_id}.wav"
    _clip_audio(source_path, output, window)
    probe = _probe_audio(output)
    if (
        probe["codec"] != "pcm_s16le"
        or probe["sampleRate"] != 16_000
        or probe["channels"] != 1
        or probe["durationSeconds"] > float(maximum_duration_seconds) + 0.05
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
        "evaluationSplit": evaluation_split,
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
    evaluation_split = plan.get("evaluationSplit")
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
        or evaluation_split not in EVALUATION_SPLITS
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
                "evaluationSplit": evaluation_split,
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


def _alimeeting_file_evidence(
    root: Path,
    tree_evidence: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    entry = tree_evidence["files"].get(relative)
    if not isinstance(entry, dict) or entry.get("path") != path:
        raise GlobalSampleLibraryError(
            f"AliMeeting file is absent from pinned tree: {relative}"
        )
    return {
        "path": relative,
        "bytes": entry["bytes"],
        "sha256": entry["sha256"],
    }


def _discover_alimeeting_sessions(
    root: Path,
    plan: dict[str, Any],
    tree_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    far_audio_dir = root / "Eval_Ali_far" / "audio_dir"
    far_textgrid_dir = root / "Eval_Ali_far" / "textgrid_dir"
    near_audio_dir = root / "Eval_Ali_near" / "audio_dir"
    near_textgrid_dir = root / "Eval_Ali_near" / "textgrid_dir"
    if not all(
        path.is_dir()
        for path in (
            far_audio_dir,
            far_textgrid_dir,
            near_audio_dir,
            near_textgrid_dir,
        )
    ):
        raise GlobalSampleLibraryError(
            "AliMeeting Eval near/far directory layout is invalid"
        )
    sessions: list[dict[str, Any]] = []
    for target in plan["sessionTargets"]:
        session_id = target["sessionId"]
        speaker_count = target["speakerCount"]
        far_audio_candidates = sorted(
            far_audio_dir.glob(f"{session_id}_MS[0-9][0-9][0-9].wav")
        )
        far_textgrid = far_textgrid_dir / f"{session_id}.TextGrid"
        if len(far_audio_candidates) != 1 or not far_textgrid.is_file():
            raise GlobalSampleLibraryError(
                f"AliMeeting far source pair is invalid: {session_id}"
            )
        far_audio = far_audio_candidates[0]
        far_grid = parse_alimeeting_textgrid(
            far_textgrid.read_text(encoding="utf-8")
        )
        turns = alimeeting_textgrid_turns(far_grid)
        speaker_set = sorted({str(turn["speakerId"]) for turn in turns})
        if len(speaker_set) != speaker_count:
            raise GlobalSampleLibraryError(
                f"AliMeeting speaker count does not match plan: {session_id}"
            )
        far_duration = _probe_alimeeting_source_wave(far_audio, 8)
        if far_duration + 0.01 < float(far_grid["endSeconds"]):
            raise GlobalSampleLibraryError(
                f"AliMeeting far audio is shorter than truth: {session_id}"
            )
        far_tiers = {str(tier["name"]): tier for tier in far_grid["tiers"]}
        near_audio: list[Path] = []
        near_audio_evidence: list[dict[str, Any]] = []
        near_textgrid_evidence: list[dict[str, Any]] = []
        for speaker_id in speaker_set:
            near_stem = f"{session_id}_{speaker_id}"
            speaker_audio = near_audio_dir / f"{near_stem}.wav"
            speaker_textgrid = near_textgrid_dir / f"{near_stem}.TextGrid"
            if not speaker_audio.is_file() or not speaker_textgrid.is_file():
                raise GlobalSampleLibraryError(
                    f"AliMeeting near source pair is missing: {near_stem}"
                )
            near_grid = parse_alimeeting_textgrid(
                speaker_textgrid.read_text(encoding="utf-8")
            )
            if (
                len(near_grid["tiers"]) != 1
                or near_grid["tiers"][0]["intervals"]
                != far_tiers[speaker_id]["intervals"]
            ):
                raise GlobalSampleLibraryError(
                    "AliMeeting synchronized near/far TextGrid truth differs: "
                    + near_stem
                )
            near_duration = _probe_alimeeting_source_wave(speaker_audio, 1)
            if near_duration + 0.01 < float(near_grid["endSeconds"]):
                raise GlobalSampleLibraryError(
                    f"AliMeeting near audio is shorter than truth: {near_stem}"
                )
            near_audio.append(speaker_audio)
            near_audio_evidence.append(
                _alimeeting_file_evidence(root, tree_evidence, speaker_audio)
            )
            near_textgrid_evidence.append(
                _alimeeting_file_evidence(root, tree_evidence, speaker_textgrid)
            )
        sessions.append(
            {
                "sessionId": session_id,
                "speakerCount": speaker_count,
                "speakerSet": speaker_set,
                "turns": turns,
                "farAudio": far_audio,
                "farAudioEvidence": _alimeeting_file_evidence(
                    root,
                    tree_evidence,
                    far_audio,
                ),
                "farTextGridEvidence": _alimeeting_file_evidence(
                    root,
                    tree_evidence,
                    far_textgrid,
                ),
                "nearAudio": near_audio,
                "nearAudioEvidence": near_audio_evidence,
                "nearTextGridEvidence": near_textgrid_evidence,
            }
        )
    return sessions


def _alimeeting_case_record(
    *,
    output_root: Path,
    plan: dict[str, Any],
    session: dict[str, Any],
    window: dict[str, Any],
    modality: str,
) -> dict[str, Any]:
    session_id = str(session["sessionId"])
    speaker_count = int(session["speakerCount"])
    if modality == "far-field-array":
        suffix = "far"
        source_audio = [session["farAudio"]]
        source_evidence = [session["farAudioEvidence"]]
        real_or_synthetic = "real-recording"
        scenarios = [
            "real-recording",
            "far-field-meeting",
            "multichannel-source",
            "overlap",
            "rapid-turns",
        ]
    elif modality == "synchronized-near-field-mixture":
        suffix = "near-mix"
        source_audio = list(session["nearAudio"])
        source_evidence = list(session["nearAudioEvidence"])
        real_or_synthetic = "synthetic-mixture"
        scenarios = [
            "synthetic-mixture",
            "synchronized-near-field",
            "close-talk",
            "overlap",
            "rapid-turns",
        ]
    else:
        raise GlobalSampleLibraryError("AliMeeting modality is unsupported")
    case_id = f"alimeeting-eval-{session_id.lower()}-{suffix}-n{speaker_count}"
    output = output_root / "audio" / f"{case_id}.wav"
    if modality == "far-field-array":
        _clip_audio(source_audio[0], output, window)
    else:
        _clip_synchronized_near_audio(source_audio, output, window)
    probe = _probe_audio(output)
    if (
        probe["codec"] != "pcm_s16le"
        or probe["sampleRate"] != 16_000
        or probe["channels"] != 1
        or probe["durationSeconds"] > 90.05
    ):
        raise GlobalSampleLibraryError(
            f"AliMeeting normalized audio is invalid: {case_id}"
        )
    references = _clip_reference_transcript(session["turns"], window)
    scoring_transcript = "".join(
        str(reference["transcript"]) for reference in references
    )
    if not scoring_transcript:
        raise GlobalSampleLibraryError(
            f"AliMeeting ASR truth is empty: {case_id}"
        )
    return {
        "id": case_id,
        "sourceId": "alimeeting",
        "sourceDataset": plan["dataset"],
        "sourceRevision": plan["revision"],
        "sourceSessionId": session_id,
        "sourceModality": modality,
        "sourceArtifactPath": session["farTextGridEvidence"]["path"],
        "sourceArtifactSha256": session["farTextGridEvidence"]["sha256"],
        "sourceAudioArtifacts": source_evidence,
        "path": str(output.relative_to(output_root)),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "audio": probe,
        "realOrSynthetic": real_or_synthetic,
        "language": "zh-CN",
        "region": "East Asia",
        "evaluationSplit": plan["evaluationSplit"],
        "scenario": scenarios,
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
        "referenceTranscriptTurns": references,
        "scoringTranscript": scoring_transcript,
        "asrReferenceMode": (
            "official-textgrid-whole-utterances-serialized-by-"
            "start-end-speaker-v1"
        ),
        "derJerReferenceMode": "official-far-textgrid-speaker-intervals-v1",
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


def _build_alimeeting(
    manifest: Any,
    output_root: Path,
    *,
    archive_path: Path | None = None,
    corpus_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = _planned_by_id(manifest, "alimeeting")
    _validate_alimeeting_plan(plan)
    root, tree_evidence, archive_verified = _resolve_alimeeting_corpus(
        output_root=output_root,
        plan=plan,
        archive_path=archive_path,
        corpus_root=corpus_root,
    )
    sessions = _discover_alimeeting_sessions(root, plan, tree_evidence)
    cases: list[dict[str, Any]] = []
    for session in sessions:
        window = select_alimeeting_textgrid_window(
            session["turns"],
            session["speakerCount"],
            minimum_duration_seconds=10.0,
            maximum_duration_seconds=90.0,
            minimum_overlap_seconds=plan["minimumAnnotatedOverlapSeconds"],
        )
        for modality in plan["modalities"]:
            cases.append(
                _alimeeting_case_record(
                    output_root=output_root,
                    plan=plan,
                    session=session,
                    window=window,
                    modality=modality,
                )
            )
    session_inventory = [
        {
            "sessionId": session["sessionId"],
            "speakerCount": session["speakerCount"],
            "speakerSet": session["speakerSet"],
            "farAudio": session["farAudioEvidence"],
            "farTextGrid": session["farTextGridEvidence"],
            "nearAudio": session["nearAudioEvidence"],
            "nearTextGrid": session["nearTextGridEvidence"],
        }
        for session in sessions
    ]
    return cases, {
        "sourceId": "alimeeting",
        "provider": plan["provider"],
        "dataset": plan["dataset"],
        "revision": plan["revision"],
        "license": plan["license"],
        "licenseDecision": plan["licenseDecision"],
        "attribution": plan["attribution"],
        "split": plan["split"],
        "archive": {
            "url": plan["archiveUrl"],
            "bytes": plan["archiveBytes"],
            "sha256": plan["archiveSha256"],
            "crc64": plan["archiveCrc64"],
            "verifiedThisRun": archive_verified,
        },
        "extractedTree": {
            "rootName": plan["extractedRootName"],
            "fileCount": tree_evidence["fileCount"],
            "bytes": tree_evidence["bytes"],
            "sha256": tree_evidence["treeSha256"],
        },
        "targetSpeakerCounts": plan["targetSpeakerCounts"],
        "unsupportedTargetSpeakerCounts": plan[
            "unsupportedTargetSpeakerCounts"
        ],
        "modalities": plan["modalities"],
        "selectionUsesModelScores": False,
        "officialHomepage": plan["officialHomepage"],
        "officialBaselineRepository": plan["officialBaselineRepository"],
        "officialBaselineRevision": plan["officialBaselineRevision"],
        "sessions": session_inventory,
    }


def _build_ami(
    manifest: Any,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _source_by_id(manifest, "ami")
    plan = _planned_by_id(manifest, "ami")
    evaluation_split = plan.get("evaluationSplit")
    if evaluation_split not in EVALUATION_SPLITS:
        raise GlobalSampleLibraryError("AMI evaluation split is invalid")
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
            evaluation_split=evaluation_split,
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


def _voxconverse_shard_plans(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw_shards = plan.get("parquetShards")
    legacy = raw_shards is None
    shards = (
        [
            {
                "config": plan.get("config"),
                "split": plan.get("split"),
                "parquetPath": plan.get("parquetPath"),
                "parquetBytes": plan.get("parquetBytes"),
                "parquetSha256": plan.get("parquetSha256"),
                "rowTargets": plan.get("rowTargets"),
            }
        ]
        if legacy
        else raw_shards
    )
    if not isinstance(shards, list) or not shards:
        raise GlobalSampleLibraryError("VoxConverse parquet shards are invalid")
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(shards):
        if not isinstance(raw, dict):
            raise GlobalSampleLibraryError("VoxConverse parquet shard is invalid")
        parquet_path = raw.get("parquetPath")
        parquet_bytes = raw.get("parquetBytes")
        parquet_sha256 = raw.get("parquetSha256")
        config = raw.get("config")
        split = raw.get("split")
        row_targets = raw.get("rowTargets")
        safe_path = PurePosixPath(parquet_path) if isinstance(parquet_path, str) else None
        if (
            safe_path is None
            or safe_path.is_absolute()
            or ".." in safe_path.parts
            or safe_path.suffix != ".parquet"
            or parquet_path in seen_paths
            or isinstance(parquet_bytes, bool)
            or not isinstance(parquet_bytes, int)
            or parquet_bytes <= 0
            or not _is_sha256(parquet_sha256)
            or not isinstance(config, str)
            or not config
            or split not in {"dev", "test"}
            or not isinstance(row_targets, list)
            or not row_targets
        ):
            raise GlobalSampleLibraryError("VoxConverse parquet shard is invalid")
        seen_paths.add(parquet_path)
        normalized.append(
            {
                "index": index,
                "legacyNames": legacy,
                "config": config,
                "split": split,
                "parquetPath": parquet_path,
                "parquetBytes": parquet_bytes,
                "parquetSha256": parquet_sha256,
                "rowTargets": row_targets,
            }
        )
    return normalized


def _voxconverse_parquet_row(parquet_file: Any, row_index: int) -> dict[str, Any]:
    if row_index < 0 or row_index >= parquet_file.metadata.num_rows:
        raise GlobalSampleLibraryError("VoxConverse row target is out of range")
    offset = 0
    for group_index in range(parquet_file.num_row_groups):
        group_rows = parquet_file.metadata.row_group(group_index).num_rows
        if offset <= row_index < offset + group_rows:
            rows = parquet_file.read_row_group(
                group_index,
                columns=["audio", "timestamps_start", "timestamps_end", "speakers"],
            ).slice(row_index - offset, 1)
            values = rows.to_pylist()
            if len(values) != 1 or not isinstance(values[0], dict):
                break
            return values[0]
        offset += group_rows
    raise GlobalSampleLibraryError("VoxConverse parquet row is missing")


def _voxconverse_shard_label(parquet_path: str) -> str:
    stem = PurePosixPath(parquet_path).stem
    label = re.sub(r"[^a-z0-9]+", "-", stem.casefold()).strip("-")
    if not label:
        raise GlobalSampleLibraryError("VoxConverse parquet label is invalid")
    return label


def _build_voxconverse(
    manifest: Any,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    source = _source_by_id(manifest, "voxconverse")
    plan = _planned_by_id(manifest, "voxconverse")
    shard_plans = _voxconverse_shard_plans(plan)
    cases: list[dict[str, Any]] = []
    source_shards: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_targets: set[tuple[str, int, int, str]] = set()
    for shard in shard_plans:
        parquet_path = str(shard["parquetPath"])
        source_url = (
            f"https://huggingface.co/datasets/{source.dataset}/resolve/"
            f"{source.revision}/{parquet_path}"
        )
        local_parquet = output_root / "sources" / PurePosixPath(parquet_path).name
        _download_resumable(
            source_url,
            local_parquet,
            expected_bytes=int(shard["parquetBytes"]),
        )
        actual_parquet_sha256 = _sha256(local_parquet)
        if actual_parquet_sha256 != shard["parquetSha256"]:
            raise GlobalSampleLibraryError(
                "VoxConverse parquet SHA-256 does not match the pinned source"
            )
        parquet_file = pq.ParquetFile(local_parquet)
        shard_label = _voxconverse_shard_label(parquet_path)
        source_shards.append(
            {
                "config": shard["config"],
                "split": shard["split"],
                "sourcePath": parquet_path,
                "path": str(local_parquet.relative_to(output_root)),
                "bytes": local_parquet.stat().st_size,
                "sha256": actual_parquet_sha256,
            }
        )
        for target in shard["rowTargets"]:
            if not isinstance(target, dict):
                raise GlobalSampleLibraryError("VoxConverse row target is invalid")
            row_index = target.get("rowIndex")
            speaker_count = target.get("targetSpeakerCount")
            evaluation_split = target.get("evaluationSplit")
            maximum_duration = target.get("maximumDurationSeconds", 90.0)
            if (
                isinstance(row_index, bool)
                or not isinstance(row_index, int)
                or isinstance(speaker_count, bool)
                or not isinstance(speaker_count, int)
                or speaker_count < 1
                or evaluation_split not in EVALUATION_SPLITS
                or isinstance(maximum_duration, bool)
                or not isinstance(maximum_duration, (int, float))
                or float(maximum_duration) <= 0
                or float(maximum_duration) > MAX_WINDOW_DURATION_SECONDS
            ):
                raise GlobalSampleLibraryError("VoxConverse row target is invalid")
            target_identity = (
                parquet_path,
                row_index,
                speaker_count,
                str(evaluation_split),
            )
            if target_identity in seen_targets:
                raise GlobalSampleLibraryError("VoxConverse row target is repeated")
            seen_targets.add(target_identity)
            row = _voxconverse_parquet_row(parquet_file, row_index)
            audio = row.get("audio")
            if (
                not isinstance(audio, dict)
                or not isinstance(audio.get("bytes"), bytes)
                or not audio["bytes"]
            ):
                raise GlobalSampleLibraryError(
                    f"VoxConverse row {row_index} audio bytes are missing"
                )
            if shard["legacyNames"]:
                source_stem = f"voxconverse_dev_row_{row_index:03d}"
                case_id = (
                    f"voxconverse-dev-row{row_index:03d}-n{speaker_count}"
                )
            else:
                source_stem = (
                    "voxconverse_"
                    f"{shard_label.replace('-', '_')}_row_{row_index:03d}"
                )
                case_id = (
                    f"voxconverse-{shard_label}-row{row_index:03d}-"
                    f"n{speaker_count}"
                )
            if case_id in seen_case_ids:
                raise GlobalSampleLibraryError("VoxConverse case ID is repeated")
            seen_case_ids.add(case_id)
            source_audio = output_root / "sources" / f"{source_stem}.wav"
            source_audio.write_bytes(audio["bytes"])
            metadata_path = output_root / "sources" / f"{source_stem}.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "dataset": source.dataset,
                        "revision": source.revision,
                        "config": shard["config"],
                        "split": shard["split"],
                        "parquetPath": parquet_path,
                        "parquetSha256": actual_parquet_sha256,
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
            window = select_diarization_window(
                turns,
                speaker_count,
                maximum_duration_seconds=float(maximum_duration),
            )
            case = _case_record(
                case_id=case_id,
                source_id=source.source_id,
                source_dataset=source.dataset,
                source_revision=source.revision,
                source_row_index=row_index,
                source_path=source_audio,
                source_artifact_path=metadata_path,
                output_root=output_root,
                window=window,
                evaluation_split=evaluation_split,
                maximum_duration_seconds=float(maximum_duration),
            )
            case["acquisition"] = {
                "kind": "hf-pinned-parquet-row",
                "config": shard["config"],
                "split": shard["split"],
                "rowIndex": row_index,
                "parquetPath": parquet_path,
            }
            case["language"] = "en-US"
            case["region"] = "Global"
            case["scenario"] = [
                "real-recording",
                "meeting-speech",
                "high-speaker-count",
            ]
            cases.append(case)
    source_record = {
        "sourceId": source.source_id,
        "dataset": source.dataset,
        "revision": source.revision,
        "license": source.license,
        "attribution": source.attribution,
        "languageTags": ["en-US"],
    }
    if len(source_shards) == 1 and shard_plans[0]["legacyNames"]:
        source_record.update(
            {
                "parquetPath": source_shards[0]["path"],
                "parquetBytes": source_shards[0]["bytes"],
                "parquetSha256": source_shards[0]["sha256"],
            }
        )
    else:
        source_record["parquetShards"] = source_shards
    return cases, source_record


def build_real_diarization_library(
    manifest_path: Path,
    output_root: Path,
    source_ids: Sequence[str] | None = None,
    *,
    alimeeting_archive: Path | None = None,
    alimeeting_root: Path | None = None,
) -> Path:
    manifest = load_global_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    builders = {
        "ami": _build_ami,
        "voxconverse": _build_voxconverse,
        "aishell4": _build_aishell4,
        "alimeeting": _build_alimeeting,
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
        if source_id == "alimeeting":
            source_cases, source_record = _build_alimeeting(
                manifest,
                output_root,
                archive_path=alimeeting_archive,
                corpus_root=alimeeting_root,
            )
        else:
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
            "algorithms": sorted(
                {
                    str(case["windowSelection"]["algorithm"])
                    for case in cases
                }
            ),
            "minimumPerSpeakerAnnotatedSeconds": MIN_SPEAKER_SECONDS,
            "maximumDurationSeconds": manifest.max_duration_seconds,
            "selectionUsesModelScores": any(
                case["windowSelection"].get("selectionUsesModelScores", False)
                is True
                for case in cases
            ),
        },
        "sources": sources,
        "cases": cases,
        "failedCases": [],
        "attributionPath": str(attribution.relative_to(output_root)),
    }
    resolved["canonicalSha256"] = canonical_json_sha256(resolved)
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
        choices=("ami", "voxconverse", "aishell4", "alimeeting"),
        default=[],
        help="build only the selected source; repeat to select multiple",
    )
    parser.add_argument(
        "--alimeeting-archive",
        type=Path,
        help=(
            "local pinned Eval_Ali.tar.gz; omitted only when using an "
            "already verified --alimeeting-root or allowing download"
        ),
    )
    parser.add_argument(
        "--alimeeting-root",
        type=Path,
        help="local extracted Eval_Ali root validated by its pinned tree hash",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = build_real_diarization_library(
        args.manifest,
        args.output_root,
        source_ids=args.source,
        alimeeting_archive=args.alimeeting_archive,
        alimeeting_root=args.alimeeting_root,
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
