"""Build speaker-disjoint CN-Celeb gates for the shared model evaluators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import wave
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402


SAMPLE_RATE_HZ = 16_000
SAMPLE_WIDTH_BYTES = 2
EVALUATION_SPLITS = ("development", "held-out")
_CLIP_NAME = re.compile(
    r"^(?P<speaker>id[0-9]{5})-"
    r"(?P<genre>[a-z][a-z0-9_]*)-"
    r"(?P<recording>[0-9]+)-"
    r"(?P<segment>[0-9]+)\.flac$"
)


class CNCelebTrialError(RuntimeError):
    """Raised when CN-Celeb cannot produce a trustworthy frozen gate."""


PcmLoader = Callable[[Path, int, int], bytes]


def parse_clip_identity(path: Path) -> dict[str, str]:
    match = _CLIP_NAME.fullmatch(path.name.casefold())
    if match is None:
        raise CNCelebTrialError(f"unsupported CN-Celeb clip name: {path.name}")
    fields = match.groupdict()
    return {
        **fields,
        "originalRecordingId": (
            f"{fields['speaker']}-{fields['genre']}-{fields['recording']}"
        ),
    }


def load_clip_pcm(
    path: Path,
    target_frames: int,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> bytes:
    """Decode one centered mono PCM window from an official FLAC clip."""

    if target_frames < 1 or sample_rate_hz < 1:
        raise ValueError("target_frames and sample_rate_hz must be positive")
    try:
        import soundfile
    except ImportError as exc:
        raise CNCelebTrialError(
            "soundfile is required; use the pinned media-asr runtime"
        ) from exc
    try:
        with soundfile.SoundFile(str(path.resolve(strict=True))) as handle:
            if handle.samplerate != sample_rate_hz or handle.channels != 1:
                raise CNCelebTrialError(
                    f"clip is not mono {sample_rate_hz} Hz audio: {path.name}"
                )
            if handle.frames < target_frames:
                raise CNCelebTrialError(f"clip is shorter than the gate: {path.name}")
            handle.seek((handle.frames - target_frames) // 2)
            samples = handle.read(
                frames=target_frames,
                dtype="int16",
                always_2d=True,
            )
    except CNCelebTrialError:
        raise
    except (OSError, RuntimeError) as exc:
        raise CNCelebTrialError(f"clip cannot be decoded: {path.name}") from exc
    if samples.shape != (target_frames, 1):
        raise CNCelebTrialError(f"clip decode length differs: {path.name}")
    return samples[:, 0].astype("<i2", copy=False).tobytes()


def _stable_key(seed: int, *values: str) -> str:
    payload = ":".join((str(seed), *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CNCelebTrialError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CNCelebTrialError(f"{label} must be an object")
    return value


def _archive_evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    evidence = _strict_json(resolved, "archive evidence")
    declared = evidence.get("canonicalSha256")
    body = dict(evidence)
    body.pop("canonicalSha256", None)
    validation = evidence.get("validation")
    if (
        evidence.get("datasetKey") != "openslr-cnceleb1-v2"
        or not isinstance(declared, str)
        or canonical_json_sha256(body) != declared
        or not isinstance(validation, dict)
        or validation.get("completeTarScanPassed") is not True
        or validation.get("safeMemberPolicyPassed") is not True
        or not isinstance(validation.get("archiveSha256"), str)
    ):
        raise CNCelebTrialError("archive evidence did not pass the complete gate")
    archive_path = Path(str(validation.get("archivePath", ""))).resolve(strict=True)
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != validation.get("archiveBytes")
    ):
        raise CNCelebTrialError("archive no longer matches its gated byte length")
    return {
        "path": str(resolved),
        "fileSha256": sha256_file(resolved),
        "canonicalSha256": declared,
        "archivePath": str(archive_path),
        "archiveBytes": validation["archiveBytes"],
        "archiveSha256": validation["archiveSha256"],
    }


def _discover(
    dataset_root: Path,
) -> dict[str, dict[str, list[tuple[Path, dict[str, str]]]]]:
    root = dataset_root.resolve(strict=True)
    test_root = root / "eval" / "test"
    if not test_root.is_dir():
        raise CNCelebTrialError("dataset root has no eval/test directory")
    grouped: dict[
        str, dict[str, list[tuple[Path, dict[str, str]]]]
    ] = defaultdict(lambda: defaultdict(list))
    for path in sorted(test_root.glob("*.flac"), key=lambda item: item.name.casefold()):
        identity = parse_clip_identity(path)
        grouped[identity["speaker"]][identity["originalRecordingId"]].append(
            (path.resolve(strict=True), identity)
        )
    if not grouped:
        raise CNCelebTrialError("dataset contains no supported eval FLAC clips")
    return {
        speaker: dict(recordings)
        for speaker, recordings in grouped.items()
    }


def _select_speakers(
    grouped: Mapping[str, Mapping[str, Sequence[tuple[Path, dict[str, str]]]]],
    *,
    speakers_per_split: int,
    clips_per_speaker: int,
    target_frames: int,
    random_seed: int,
    pcm_loader: PcmLoader,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    required_speakers = speakers_per_split * len(EVALUATION_SPLITS)
    selected: dict[str, list[dict[str, Any]]] = {}
    rejected_audio_files = 0
    speaker_order = sorted(
        grouped,
        key=lambda speaker: (_stable_key(random_seed, speaker), speaker),
    )
    for speaker in speaker_order:
        recordings = grouped[speaker]
        if len(recordings) < clips_per_speaker:
            continue
        chosen: list[dict[str, Any]] = []
        recording_order = sorted(
            recordings,
            key=lambda recording: (
                _stable_key(random_seed, speaker, recording),
                recording,
            ),
        )
        for recording_id in recording_order:
            candidates = sorted(
                recordings[recording_id],
                key=lambda item: (
                    _stable_key(random_seed, speaker, recording_id, item[0].name),
                    item[0].name,
                ),
            )
            for path, identity in candidates:
                try:
                    pcm = pcm_loader(path, target_frames, SAMPLE_RATE_HZ)
                except CNCelebTrialError:
                    rejected_audio_files += 1
                    continue
                if len(pcm) != target_frames * SAMPLE_WIDTH_BYTES:
                    raise CNCelebTrialError(
                        f"PCM loader returned an invalid byte count: {path.name}"
                    )
                chosen.append(
                    {
                        "path": path,
                        "identity": identity,
                        "pcm": pcm,
                    }
                )
                break
            if len(chosen) == clips_per_speaker:
                break
        if len(chosen) == clips_per_speaker:
            selected[speaker] = chosen
        if len(selected) == required_speakers:
            break
    if len(selected) != required_speakers:
        raise CNCelebTrialError(
            "not enough speakers have the required distinct readable recordings"
        )
    return selected, rejected_audio_files


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {resolved}")
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


def _write_composite(
    path: Path,
    *,
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    clip_duration_ms: int,
    padding_ms: int,
    evaluation_split: str,
    composite_recording_id: str,
) -> tuple[list[dict[str, Any]], dict[str, list[Any]], int]:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite composite audio: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp.wav", dir=resolved.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    padding_frames = round(padding_ms * SAMPLE_RATE_HZ / 1000)
    silence = b"\0" * padding_frames * SAMPLE_WIDTH_BYTES
    ordered = [
        item
        for speaker in sorted(selected)
        for item in selected[speaker]
    ]
    clips: list[dict[str, Any]] = []
    annotation: dict[str, list[Any]] = {
        "timestamps_start": [],
        "timestamps_end": [],
        "speakers": [],
        "original_recording_ids": [],
    }
    frame_cursor = 0
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(SAMPLE_WIDTH_BYTES)
            handle.setframerate(SAMPLE_RATE_HZ)
            for index, item in enumerate(ordered):
                identity = item["identity"]
                source = Path(item["path"])
                pcm = item["pcm"]
                start_ms = round(frame_cursor * 1000 / SAMPLE_RATE_HZ)
                handle.writeframesraw(pcm)
                frame_cursor += len(pcm) // SAMPLE_WIDTH_BYTES
                end_ms = round(frame_cursor * 1000 / SAMPLE_RATE_HZ)
                if end_ms - start_ms != clip_duration_ms:
                    raise CNCelebTrialError("composite clip duration drifted")
                clip_id = f"cnc-{evaluation_split}-{source.stem.casefold()}"
                clips.append(
                    {
                        "clipId": clip_id,
                        "speakerId": identity["speaker"],
                        "startMs": start_ms,
                        "endMs": end_ms,
                        "audio": str(resolved),
                        "sourceRecordingId": composite_recording_id,
                        "originalRecordingId": identity["originalRecordingId"],
                        "evaluationSplit": evaluation_split,
                        "genre": identity["genre"],
                        "sourceFilePath": str(source),
                        "sourceFileBytes": source.stat().st_size,
                        "sourceFileSha256": sha256_file(source),
                    }
                )
                annotation["timestamps_start"].append(start_ms / 1000.0)
                annotation["timestamps_end"].append(end_ms / 1000.0)
                annotation["speakers"].append(identity["speaker"])
                annotation["original_recording_ids"].append(
                    identity["originalRecordingId"]
                )
                if index + 1 < len(ordered) and padding_frames:
                    handle.writeframesraw(silence)
                    frame_cursor += padding_frames
            handle.writeframes(b"")
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return clips, annotation, frame_cursor


def _trials(
    clips: Sequence[Mapping[str, Any]],
    *,
    evaluation_split: str,
    maximum_trials_per_class: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for clip in clips:
        by_speaker[str(clip["speakerId"])].append(clip)
    genuine: list[tuple[str, str, bool]] = []
    for speaker_clips in by_speaker.values():
        for left, right in combinations(speaker_clips, 2):
            if left["originalRecordingId"] != right["originalRecordingId"]:
                genuine.append((str(left["clipId"]), str(right["clipId"]), True))
    impostor = [
        (str(left["clipId"]), str(right["clipId"]), False)
        for left, right in combinations(clips, 2)
        if left["speakerId"] != right["speakerId"]
        and left["originalRecordingId"] != right["originalRecordingId"]
    ]
    genuine.sort(
        key=lambda item: _stable_key(random_seed, evaluation_split, item[0], item[1])
    )
    impostor.sort(
        key=lambda item: _stable_key(random_seed, evaluation_split, item[0], item[1])
    )
    per_class = min(maximum_trials_per_class, len(genuine), len(impostor))
    if per_class < 1:
        raise CNCelebTrialError("partition cannot produce both trial classes")
    paired = genuine[:per_class] + impostor[:per_class]
    paired.sort(
        key=lambda item: _stable_key(
            random_seed + 1, evaluation_split, item[0], item[1]
        )
    )
    return [
        {
            "trialId": f"cnc-{evaluation_split}-{index:05d}",
            "enrollmentClipId": left,
            "testClipId": right,
            "sameSpeaker": same_speaker,
            "evaluationSplit": evaluation_split,
        }
        for index, (left, right, same_speaker) in enumerate(paired, start=1)
    ]


def _build_partition(
    *,
    output_directory: Path,
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    archive: Mapping[str, Any],
    dataset_root: Path,
    evaluation_split: str,
    clip_duration_ms: int,
    padding_ms: int,
    maximum_trials_per_class: int,
    random_seed: int,
    rejected_audio_files: int,
) -> dict[str, Any]:
    composite_path = output_directory / f"{evaluation_split}.composite.wav"
    annotation_path = output_directory / f"{evaluation_split}.annotations.json"
    manifest_path = output_directory / f"{evaluation_split}.trials.json"
    composite_recording_id = (
        f"cnceleb-v2-{evaluation_split}-composite-{random_seed}"
    )
    clips, annotation, frame_count = _write_composite(
        composite_path,
        selected=selected,
        clip_duration_ms=clip_duration_ms,
        padding_ms=padding_ms,
        evaluation_split=evaluation_split,
        composite_recording_id=composite_recording_id,
    )
    annotation_document: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "dataset": "openslr/82/cn-celeb-v2",
        "revision": "v2",
        "config": "official-eval-test-speaker-disjoint-composite",
        "split": evaluation_split,
        **annotation,
    }
    annotation_document["canonicalSha256"] = canonical_json_sha256(
        annotation_document
    )
    _atomic_json(annotation_path, annotation_document)

    trials = _trials(
        clips,
        evaluation_split=evaluation_split,
        maximum_trials_per_class=maximum_trials_per_class,
        random_seed=random_seed,
    )
    same_count = sum(bool(item["sameSpeaker"]) for item in trials)
    duration_ms = round(frame_count * 1000 / SAMPLE_RATE_HZ)
    manifest: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "libraryId": (
            f"openslr-cnceleb1-v2-{evaluation_split}-speaker-verification-v1"
        ),
        "randomSeed": random_seed,
        "source": {
            "recordingId": composite_recording_id,
            "audioPath": str(composite_path.resolve()),
            "audioBytes": composite_path.stat().st_size,
            "audioSha256": sha256_file(composite_path),
            "annotationPath": str(annotation_path.resolve()),
            "annotationBytes": annotation_path.stat().st_size,
            "annotationSha256": sha256_file(annotation_path),
            "audio": {
                "codec": "pcm_s16le",
                "sampleRateHz": SAMPLE_RATE_HZ,
                "channels": 1,
                "durationMs": duration_ms,
            },
            "dataset": "openslr/82/cn-celeb-v2",
            "revision": "v2",
            "config": "official-eval-test-speaker-disjoint-composite",
            "split": evaluation_split,
            "datasetRoot": str(dataset_root.resolve()),
            "archiveEvidence": dict(archive),
        },
        "selection": {
            "algorithm": "speaker-disjoint-cross-original-recording-v1",
            "clipDurationMs": clip_duration_ms,
            "paddingMs": padding_ms,
            "speakers": len(selected),
            "clipsPerSpeaker": len(next(iter(selected.values()))),
            "maximumTrialsPerClass": maximum_trials_per_class,
            "rejectedUnreadableOrShortAudioFiles": rejected_audio_files,
            "originalRecordingIdParser": (
                "drop-final-segment-from-id-genre-recording-segment"
            ),
        },
        "counts": {
            "speakers": len(selected),
            "clips": len(clips),
            "clipsPerSpeaker": len(next(iter(selected.values()))),
            "sameSpeakerTrials": same_count,
            "differentSpeakerTrials": len(trials) - same_count,
            "totalTrials": len(trials),
        },
        "clips": clips,
        "trials": trials,
        "promotionPolicy": {
            "speakerDisjointDevelopmentAndHeldOut": True,
            "sameSpeakerTrialsCrossOriginalRecordings": True,
            "heldOutThresholdFittingAllowed": False,
        },
    }
    manifest["canonicalSha256"] = canonical_json_sha256(manifest)
    _atomic_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifestPath": manifest_path.resolve(),
        "audioPath": composite_path.resolve(),
        "annotationPath": annotation_path.resolve(),
    }


def build_gate(
    *,
    dataset_root: Path,
    archive_evidence_path: Path,
    output_directory: Path,
    speakers_per_split: int = 64,
    clips_per_speaker: int = 5,
    clip_duration_ms: int = 2_500,
    padding_ms: int = 100,
    maximum_trials_per_class: int = 1_024,
    random_seed: int = 20_260_807,
    pcm_loader: PcmLoader = load_clip_pcm,
) -> dict[str, Any]:
    numeric = (
        speakers_per_split,
        clips_per_speaker,
        clip_duration_ms,
        maximum_trials_per_class,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in numeric
    ):
        raise ValueError("gate sizes must be positive integers")
    if isinstance(padding_ms, bool) or not isinstance(padding_ms, int) or padding_ms < 0:
        raise ValueError("padding_ms must be a non-negative integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    root = dataset_root.resolve(strict=True)
    output = output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output / f"{split}.{suffix}"
        for split in EVALUATION_SPLITS
        for suffix in ("composite.wav", "annotations.json", "trials.json")
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite gate artifacts: "
            + ", ".join(path.name for path in existing)
        )
    archive = _archive_evidence(archive_evidence_path)
    target_frames = round(clip_duration_ms * SAMPLE_RATE_HZ / 1000)
    if target_frames * 1000 != clip_duration_ms * SAMPLE_RATE_HZ:
        raise ValueError("clip_duration_ms must map to an exact 16 kHz frame count")
    selected, rejected_audio_files = _select_speakers(
        _discover(root),
        speakers_per_split=speakers_per_split,
        clips_per_speaker=clips_per_speaker,
        target_frames=target_frames,
        random_seed=random_seed,
        pcm_loader=pcm_loader,
    )
    ordered_speakers = list(selected)
    development_speakers = ordered_speakers[:speakers_per_split]
    held_out_speakers = ordered_speakers[speakers_per_split:]
    if set(development_speakers) & set(held_out_speakers):
        raise CNCelebTrialError("speaker-disjoint partitioning failed")
    partitions = {}
    for split, speakers in zip(
        EVALUATION_SPLITS,
        (development_speakers, held_out_speakers),
    ):
        partitions[split] = _build_partition(
            output_directory=output,
            selected={speaker: selected[speaker] for speaker in speakers},
            archive=archive,
            dataset_root=root,
            evaluation_split=split,
            clip_duration_ms=clip_duration_ms,
            padding_ms=padding_ms,
            maximum_trials_per_class=maximum_trials_per_class,
            random_seed=random_seed,
            rejected_audio_files=rejected_audio_files,
        )
    return {
        "schemaVersion": "1.0.0",
        "datasetKey": "openslr-cnceleb1-v2",
        "archiveEvidence": archive,
        "speakerDisjoint": True,
        "developmentSpeakers": development_speakers,
        "heldOutSpeakers": held_out_speakers,
        "partitions": partitions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive-evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--speakers-per-split", type=_positive_integer, default=64
    )
    parser.add_argument(
        "--clips-per-speaker", type=_positive_integer, default=5
    )
    parser.add_argument(
        "--clip-duration-ms", type=_positive_integer, default=2_500
    )
    parser.add_argument(
        "--padding-ms", type=_nonnegative_integer, default=100
    )
    parser.add_argument(
        "--maximum-trials-per-class",
        type=_positive_integer,
        default=1_024,
    )
    parser.add_argument("--random-seed", type=int, default=20_260_807)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_gate(
        dataset_root=args.dataset_root,
        archive_evidence_path=args.archive_evidence,
        output_directory=args.output_directory,
        speakers_per_split=args.speakers_per_split,
        clips_per_speaker=args.clips_per_speaker,
        clip_duration_ms=args.clip_duration_ms,
        padding_ms=args.padding_ms,
        maximum_trials_per_class=args.maximum_trials_per_class,
        random_seed=args.random_seed,
    )
    print(
        json.dumps(
            {
                "datasetKey": result["datasetKey"],
                "speakerDisjoint": result["speakerDisjoint"],
                "partitions": {
                    split: {
                        "manifestPath": str(value["manifestPath"]),
                        "canonicalSha256": value["manifest"]["canonicalSha256"],
                        "counts": value["manifest"]["counts"],
                    }
                    for split, value in result["partitions"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
