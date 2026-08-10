"""Build deterministic speaker-verification trials from diarization truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import wave
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402


SCHEMA_VERSION = "1.0.0"
EVALUATION_SPLITS = frozenset({"development", "regression", "held-out"})
_SAFE_ID = re.compile(r"[^a-z0-9]+")


class SpeakerTrialError(ValueError):
    """Raised when source truth cannot produce a valid frozen trial set."""


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _milliseconds(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeakerTrialError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise SpeakerTrialError(f"{field} must be finite and non-negative")
    return round(parsed * 1000.0)


def _safe_id(value: str) -> str:
    normalized = _SAFE_ID.sub("-", value.casefold()).strip("-")
    if not normalized:
        raise SpeakerTrialError("speaker and recording ids must be printable")
    return normalized


def _load_annotations(path: Path) -> tuple[dict[str, Any], list[tuple[int, int, str]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpeakerTrialError("annotation JSON is unreadable") from exc
    if not isinstance(document, dict):
        raise SpeakerTrialError("annotation JSON must be an object")
    starts = document.get("timestamps_start")
    ends = document.get("timestamps_end")
    speakers = document.get("speakers")
    if (
        not isinstance(starts, list)
        or not isinstance(ends, list)
        or not isinstance(speakers, list)
        or not starts
        or len(starts) != len(ends)
        or len(starts) != len(speakers)
    ):
        raise SpeakerTrialError("annotation turn arrays are inconsistent")
    turns: list[tuple[int, int, str]] = []
    for index, (raw_start, raw_end, raw_speaker) in enumerate(
        zip(starts, ends, speakers)
    ):
        start_ms = _milliseconds(raw_start, field=f"turn {index} start")
        end_ms = _milliseconds(raw_end, field=f"turn {index} end")
        if end_ms <= start_ms:
            raise SpeakerTrialError(f"turn {index} has no positive duration")
        if not isinstance(raw_speaker, str) or not raw_speaker.strip():
            raise SpeakerTrialError(f"turn {index} has no speaker id")
        turns.append((start_ms, end_ms, raw_speaker.strip()))
    return document, sorted(turns)


def solo_speaker_spans(
    turns: Sequence[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Return maximal spans where exactly one annotated speaker is active."""

    if not turns:
        raise SpeakerTrialError("at least one annotated turn is required")
    events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for start_ms, end_ms, speaker_id in turns:
        if start_ms < 0 or end_ms <= start_ms or not speaker_id:
            raise SpeakerTrialError("annotated turn is invalid")
        events[start_ms].append((speaker_id, 1))
        events[end_ms].append((speaker_id, -1))

    active: Counter[str] = Counter()
    spans: list[tuple[int, int, str]] = []
    previous_ms: int | None = None
    for current_ms in sorted(events):
        active_speakers = sorted(
            speaker for speaker, count in active.items() if count > 0
        )
        if (
            previous_ms is not None
            and current_ms > previous_ms
            and len(active_speakers) == 1
        ):
            speaker_id = active_speakers[0]
            if (
                spans
                and spans[-1][1] == previous_ms
                and spans[-1][2] == speaker_id
            ):
                spans[-1] = (spans[-1][0], current_ms, speaker_id)
            else:
                spans.append((previous_ms, current_ms, speaker_id))
        for speaker_id, delta in events[current_ms]:
            active[speaker_id] += delta
            if active[speaker_id] < 0:
                raise SpeakerTrialError("annotation events are unbalanced")
            if active[speaker_id] == 0:
                del active[speaker_id]
        previous_ms = current_ms
    if active:
        raise SpeakerTrialError("annotation events do not close all turns")
    return spans


def _candidate_windows(
    spans: Sequence[tuple[int, int, str]],
    *,
    clip_duration_ms: int,
) -> dict[str, list[tuple[int, int]]]:
    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    step_ms = max(250, clip_duration_ms // 2)
    for span_start, span_end, speaker_id in spans:
        latest_start = span_end - clip_duration_ms
        if latest_start < span_start:
            continue
        start_ms = span_start
        while start_ms <= latest_start:
            candidates[speaker_id].append(
                (start_ms, start_ms + clip_duration_ms)
            )
            start_ms += step_ms
        if candidates[speaker_id][-1][0] != latest_start:
            candidates[speaker_id].append(
                (latest_start, latest_start + clip_duration_ms)
            )
    return candidates


def _spread_windows(
    candidates: Sequence[tuple[int, int]],
    *,
    minimum_spacing_ms: int,
    maximum_count: int,
) -> list[tuple[int, int]]:
    spaced: list[tuple[int, int]] = []
    previous_center: int | None = None
    for start_ms, end_ms in sorted(set(candidates)):
        center = (start_ms + end_ms) // 2
        if (
            previous_center is None
            or center - previous_center >= minimum_spacing_ms
        ):
            spaced.append((start_ms, end_ms))
            previous_center = center
    if len(spaced) <= maximum_count:
        return spaced
    if maximum_count == 1:
        return [spaced[len(spaced) // 2]]
    indices = [
        round(index * (len(spaced) - 1) / (maximum_count - 1))
        for index in range(maximum_count)
    ]
    return [spaced[index] for index in indices]


def _stable_order(seed: int, left: str, right: str) -> str:
    return hashlib.sha256(f"{seed}:{left}:{right}".encode("utf-8")).hexdigest()


def _wav_metadata(path: Path) -> dict[str, int | str]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise SpeakerTrialError("source audio must be a readable PCM WAV") from exc
    if (
        channels < 1
        or sample_width not in {1, 2, 3, 4}
        or sample_rate < 1
        or frame_count < 1
        or compression != "NONE"
    ):
        raise SpeakerTrialError("source audio WAV metadata is unsupported")
    return {
        "codec": f"pcm_s{sample_width * 8}le",
        "sampleRateHz": sample_rate,
        "channels": channels,
        "durationMs": round(frame_count * 1000.0 / sample_rate),
    }


def build_manifest(
    *,
    audio_path: Path,
    annotation_path: Path,
    source_recording_id: str,
    evaluation_split: str,
    clip_duration_ms: int = 2_500,
    maximum_clips_per_speaker: int = 12,
    minimum_spacing_ms: int = 15_000,
    minimum_pair_separation_ms: int = 30_000,
    maximum_pairs_per_class: int = 256,
    random_seed: int = 20_260_807,
) -> dict[str, Any]:
    if evaluation_split not in EVALUATION_SPLITS:
        raise SpeakerTrialError("evaluation split is invalid")
    numeric_options = (
        clip_duration_ms,
        maximum_clips_per_speaker,
        minimum_spacing_ms,
        minimum_pair_separation_ms,
        maximum_pairs_per_class,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in numeric_options
    ):
        raise SpeakerTrialError("trial selection options must be positive integers")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise SpeakerTrialError("random seed must be an integer")

    audio = audio_path.resolve(strict=True)
    annotations = annotation_path.resolve(strict=True)
    if not audio.is_file() or not annotations.is_file():
        raise SpeakerTrialError("source audio and annotations must be files")
    audio_metadata = _wav_metadata(audio)
    annotation_document, turns = _load_annotations(annotations)
    if max(end_ms for _, end_ms, _ in turns) > int(audio_metadata["durationMs"]) + 1:
        raise SpeakerTrialError("annotation extends beyond the source audio")

    candidates = _candidate_windows(
        solo_speaker_spans(turns),
        clip_duration_ms=clip_duration_ms,
    )
    selected_by_speaker = {
        speaker_id: _spread_windows(
            windows,
            minimum_spacing_ms=minimum_spacing_ms,
            maximum_count=maximum_clips_per_speaker,
        )
        for speaker_id, windows in sorted(candidates.items())
    }
    selected_by_speaker = {
        speaker_id: windows
        for speaker_id, windows in selected_by_speaker.items()
        if len(windows) >= 2
    }
    if len(selected_by_speaker) < 2:
        raise SpeakerTrialError(
            "truth does not contain two speakers with two isolated clips each"
        )
    balanced_count = min(
        maximum_clips_per_speaker,
        min(len(windows) for windows in selected_by_speaker.values()),
    )
    if balanced_count < 2:
        raise SpeakerTrialError("truth cannot produce balanced speaker clips")

    audio_sha256 = sha256_file(audio)
    annotation_sha256 = sha256_file(annotations)
    recording_key = _safe_id(source_recording_id)
    clips: list[dict[str, Any]] = []
    speaker_clip_ids: dict[str, list[str]] = {}
    clip_times: dict[str, tuple[int, int]] = {}
    for speaker_id, windows in selected_by_speaker.items():
        windows = _spread_windows(
            windows,
            minimum_spacing_ms=1,
            maximum_count=balanced_count,
        )
        clip_ids: list[str] = []
        for index, (start_ms, end_ms) in enumerate(windows, start=1):
            clip_id = (
                f"{recording_key}-{_safe_id(speaker_id)}-{index:03d}"
            )
            clip_ids.append(clip_id)
            clip_times[clip_id] = (start_ms, end_ms)
            clips.append(
                {
                    "clipId": clip_id,
                    "speakerId": speaker_id,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "audio": str(audio),
                    "sourceRecordingId": source_recording_id,
                    "evaluationSplit": evaluation_split,
                }
            )
        speaker_clip_ids[speaker_id] = clip_ids

    same_pairs: list[tuple[str, str, bool]] = []
    for clip_ids in speaker_clip_ids.values():
        for left, right in combinations(clip_ids, 2):
            left_start = clip_times[left][0]
            right_start = clip_times[right][0]
            if abs(right_start - left_start) >= minimum_pair_separation_ms:
                same_pairs.append((left, right, True))
    different_pairs: list[tuple[str, str, bool]] = []
    for left_speaker, right_speaker in combinations(sorted(speaker_clip_ids), 2):
        for left in speaker_clip_ids[left_speaker]:
            for right in speaker_clip_ids[right_speaker]:
                different_pairs.append((left, right, False))
    if not same_pairs or not different_pairs:
        raise SpeakerTrialError("truth cannot produce both trial classes")
    same_pairs.sort(key=lambda pair: _stable_order(random_seed, pair[0], pair[1]))
    different_pairs.sort(
        key=lambda pair: _stable_order(random_seed, pair[0], pair[1])
    )
    per_class = min(
        maximum_pairs_per_class,
        len(same_pairs),
        len(different_pairs),
    )
    paired = sorted(
        same_pairs[:per_class] + different_pairs[:per_class],
        key=lambda pair: _stable_order(random_seed + 1, pair[0], pair[1]),
    )
    trials = [
        {
            "trialId": f"sv-{index:04d}",
            "enrollmentClipId": left,
            "testClipId": right,
            "sameSpeaker": same_speaker,
            "evaluationSplit": evaluation_split,
        }
        for index, (left, right, same_speaker) in enumerate(paired, start=1)
    ]

    source_metadata = {
        key: annotation_document.get(key)
        for key in ("dataset", "revision", "config", "split", "rowIndex")
        if key in annotation_document
    }
    manifest: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "libraryId": f"{recording_key}-speaker-verification-v1",
        "randomSeed": random_seed,
        "source": {
            "recordingId": source_recording_id,
            "audioPath": str(audio),
            "audioBytes": audio.stat().st_size,
            "audioSha256": audio_sha256,
            "annotationPath": str(annotations),
            "annotationBytes": annotations.stat().st_size,
            "annotationSha256": annotation_sha256,
            "audio": audio_metadata,
            **source_metadata,
        },
        "selection": {
            "algorithm": "single-active-speaker-balanced-v1",
            "clipDurationMs": clip_duration_ms,
            "maximumClipsPerSpeaker": maximum_clips_per_speaker,
            "minimumSpacingMs": minimum_spacing_ms,
            "minimumPairSeparationMs": minimum_pair_separation_ms,
            "maximumPairsPerClass": maximum_pairs_per_class,
        },
        "counts": {
            "speakers": len(speaker_clip_ids),
            "clips": len(clips),
            "clipsPerSpeaker": balanced_count,
            "sameSpeakerTrials": per_class,
            "differentSpeakerTrials": per_class,
            "totalTrials": len(trials),
        },
        "clips": sorted(clips, key=lambda clip: str(clip["clipId"])),
        "trials": trials,
    }
    manifest["canonicalSha256"] = canonical_json_sha256(manifest)
    return manifest


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                manifest,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-recording-id", required=True)
    parser.add_argument(
        "--evaluation-split", choices=sorted(EVALUATION_SPLITS), required=True
    )
    parser.add_argument(
        "--clip-duration-ms", type=_positive_integer, default=2_500
    )
    parser.add_argument(
        "--maximum-clips-per-speaker", type=_positive_integer, default=12
    )
    parser.add_argument(
        "--minimum-spacing-ms", type=_positive_integer, default=15_000
    )
    parser.add_argument(
        "--minimum-pair-separation-ms",
        type=_positive_integer,
        default=30_000,
    )
    parser.add_argument(
        "--maximum-pairs-per-class", type=_positive_integer, default=256
    )
    parser.add_argument("--random-seed", type=int, default=20_260_807)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_manifest(
        audio_path=args.audio,
        annotation_path=args.annotations,
        source_recording_id=args.source_recording_id,
        evaluation_split=args.evaluation_split,
        clip_duration_ms=args.clip_duration_ms,
        maximum_clips_per_speaker=args.maximum_clips_per_speaker,
        minimum_spacing_ms=args.minimum_spacing_ms,
        minimum_pair_separation_ms=args.minimum_pair_separation_ms,
        maximum_pairs_per_class=args.maximum_pairs_per_class,
        random_seed=args.random_seed,
    )
    _write_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonicalSha256": manifest["canonicalSha256"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
