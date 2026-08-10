"""Build held-out, within-recording AliMeeting speaker diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402
from tools.build_global_real_diarization import (  # noqa: E402
    _discover_alimeeting_sessions,
    _planned_by_id,
    _resolve_alimeeting_corpus,
    _validate_alimeeting_plan,
)
from tools.build_speaker_verification_trials import (  # noqa: E402
    _write_manifest,
    build_manifest as build_single_recording_manifest,
)
from tools.global_sample_library import load_global_manifest  # noqa: E402


DEFAULT_GLOBAL_MANIFEST = (
    PROJECT_ROOT / "sample_library" / "global-manifest.v1.json"
)
SCHEMA_VERSION = "1.0.0"
MODALITIES = ("far-channel-0", "synchronized-near-field-mixture")
USAGE_POLICY = {
    "diagnosticOnly": True,
    "heldOutOnly": True,
    "thresholdFittingAllowed": False,
    "modelPromotionAllowed": False,
    "productionDefaultSelectionAllowed": False,
    "crossRecordingGeneralizationClaimAllowed": False,
    "crossSessionSameSpeakerEvaluationAvailable": False,
}


class AliMeetingDiagnosticError(ValueError):
    """Raised when AliMeeting cannot support a legal diagnostic trial set."""


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def audit_identity_scope(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    speaker_ids_by_session: dict[str, list[str]] = {}
    for session in sessions:
        session_id = session.get("sessionId")
        speakers = session.get("speakerSet")
        if (
            not isinstance(session_id, str)
            or not session_id
            or not isinstance(speakers, list)
            or not speakers
            or any(not isinstance(speaker, str) or not speaker for speaker in speakers)
            or len(speakers) != len(set(speakers))
        ):
            raise AliMeetingDiagnosticError("session speaker identity is invalid")
        canonical_speakers = sorted(speakers)
        speaker_ids_by_session[session_id] = canonical_speakers
        for speaker_id in canonical_speakers:
            occurrences[speaker_id].append(session_id)
    repeated = {
        speaker_id: sorted(recordings)
        for speaker_id, recordings in sorted(occurrences.items())
        if len(set(recordings)) > 1
    }
    if repeated:
        raise AliMeetingDiagnosticError(
            "AliMeeting Eval unexpectedly contains cross-session speaker reuse"
        )
    return {
        "sourceSessionCount": len(speaker_ids_by_session),
        "speakerOccurrenceCount": sum(
            len(speakers) for speakers in speaker_ids_by_session.values()
        ),
        "uniqueSpeakerCount": len(occurrences),
        "speakerIdsBySession": dict(sorted(speaker_ids_by_session.items())),
        "crossSessionRepeatedSpeakerIds": [],
        "crossSessionSameSpeakerTrialsAvailable": False,
        "sameSpeakerIdentityScope": "within-source-recording-only",
        "nearFarCaptureRelationship": "synchronized-same-source-recording",
        "nearFarMayCountAsIndependentRecordings": False,
    }


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AliMeetingDiagnosticError(f"{field} is unreadable") from exc
    if not isinstance(value, dict):
        raise AliMeetingDiagnosticError(f"{field} must be an object")
    return value


def validate_related_diarization_cases(
    cases: Any,
    *,
    expected_sessions: Sequence[str],
    source_revision: str,
) -> dict[str, Any]:
    if not isinstance(cases, list) or not cases:
        raise AliMeetingDiagnosticError("related diarization cases are missing")
    expected_pairs = {
        (session_id, modality)
        for session_id in expected_sessions
        for modality in ("far-field-array", "synchronized-near-field-mixture")
    }
    actual_pairs: set[tuple[str, str]] = set()
    recording_splits: dict[str, set[str]] = defaultdict(set)
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise AliMeetingDiagnosticError(
                f"related diarization case {index} is invalid"
            )
        session_id = case.get("sourceSessionId")
        modality = case.get("sourceModality")
        evaluation_split = case.get("evaluationSplit")
        pair = (session_id, modality)
        if (
            case.get("sourceId") != "alimeeting"
            or case.get("sourceRevision") != source_revision
            or not isinstance(session_id, str)
            or not isinstance(modality, str)
            or pair not in expected_pairs
            or pair in actual_pairs
            or evaluation_split != "held-out"
        ):
            raise AliMeetingDiagnosticError(
                f"related diarization case {index} breaks source isolation"
            )
        actual_pairs.add(pair)
        recording_splits[session_id].add(evaluation_split)
    if actual_pairs != expected_pairs or any(
        splits != {"held-out"} for splits in recording_splits.values()
    ):
        raise AliMeetingDiagnosticError(
            "related diarization library is incomplete or split-leaking"
        )
    return {
        "caseCount": len(cases),
        "sourceSessions": sorted(recording_splits),
        "modalities": sorted({modality for _, modality in actual_pairs}),
        "evaluationSplit": "held-out",
        "recordingIsolationVerified": True,
        "caseSetCanonicalSha256": canonical_json_sha256({"cases": cases}),
    }


def _related_diarization_evidence(
    path: Path,
    *,
    plan: Mapping[str, Any],
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    document = _load_json_object(resolved, "related diarization manifest")
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise AliMeetingDiagnosticError("related source evidence is missing")
    source = next(
        (
            item
            for item in sources
            if isinstance(item, dict) and item.get("sourceId") == "alimeeting"
        ),
        None,
    )
    if (
        not isinstance(source, dict)
        or source.get("revision") != plan["revision"]
        or source.get("license") != plan["license"]
        or source.get("archive", {}).get("sha256") != plan["archiveSha256"]
        or source.get("extractedTree", {}).get("sha256")
        != plan["extractedTreeSha256"]
    ):
        raise AliMeetingDiagnosticError(
            "related diarization source evidence does not match AliMeeting lock"
        )
    case_evidence = validate_related_diarization_cases(
        document.get("cases"),
        expected_sessions=[str(session["sessionId"]) for session in sessions],
        source_revision=str(plan["revision"]),
    )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "libraryId": document.get("libraryId"),
        "role": "source-lineage-and-session-modality-sanity-only",
        "usedForClipSelection": False,
        **case_evidence,
    }


def _run_ffmpeg(command: Sequence[str], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite normalized audio: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.wav")
    if temporary.exists():
        raise FileExistsError(f"normalization staging file exists: {temporary}")
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *command, str(temporary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1_800,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise AliMeetingDiagnosticError(
            "ffmpeg normalization failed: " + completed.stderr.strip()
        )
    temporary.replace(output)


def _normalize_far_channel_zero(source: Path, output: Path) -> None:
    _run_ffmpeg(
        [
            "-i",
            str(source),
            "-vn",
            "-filter:a",
            "pan=mono|c0=c0",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
        ],
        output,
    )


def _normalize_near_mix(sources: Sequence[Path], output: Path) -> None:
    if len(sources) < 2:
        raise AliMeetingDiagnosticError("near mixture needs multiple speakers")
    inputs: list[str] = []
    for source in sources:
        inputs.extend(["-i", str(source)])
    labels = "".join(f"[{index}:a]" for index in range(len(sources)))
    _run_ffmpeg(
        [
            *inputs,
            "-filter_complex",
            (
                f"{labels}amix=inputs={len(sources)}:duration=longest:"
                "dropout_transition=0:normalize=1[mix]"
            ),
            "-map",
            "[mix]",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
        ],
        output,
    )


def _mono_wave_evidence(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            compression = handle.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise AliMeetingDiagnosticError("normalized WAV is unreadable") from exc
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != 16_000
        or frames < 1
        or compression != "NONE"
    ):
        raise AliMeetingDiagnosticError("normalized WAV format is invalid")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "codec": "pcm_s16le",
        "sampleRateHz": sample_rate,
        "channels": channels,
        "durationMs": round(frames * 1000.0 / sample_rate),
    }


def _annotation_document(
    session: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    modality: str,
) -> dict[str, Any]:
    turns = session["turns"]
    return {
        "dataset": plan["dataset"],
        "revision": plan["revision"],
        "config": modality,
        "split": plan["split"],
        "sessionId": session["sessionId"],
        "timestamps_start": [turn["startSeconds"] for turn in turns],
        "timestamps_end": [turn["endSeconds"] for turn in turns],
        "speakers": [turn["speakerId"] for turn in turns],
        "transcripts": [turn["transcript"] for turn in turns],
        "truthSource": session["farTextGridEvidence"],
    }


def _scope_partition_manifest(
    manifest: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    modality: str,
    related_diarization: Mapping[str, Any],
) -> dict[str, Any]:
    session_id = str(session["sessionId"])
    partition_key = f"{session_id.lower()}-{modality}"
    scoped = dict(manifest)
    scoped.pop("canonicalSha256", None)
    clips = [dict(clip) for clip in manifest["clips"]]
    trials = [dict(trial) for trial in manifest["trials"]]
    clip_ids: dict[str, str] = {}
    for clip in clips:
        old_id = str(clip["clipId"])
        new_id = f"{partition_key}-{old_id}"
        clip_ids[old_id] = new_id
        clip["clipId"] = new_id
        clip["sourceSessionId"] = session_id
        clip["sourceModality"] = modality
        clip["diagnosticOnly"] = True
    for trial in trials:
        old_trial_id = str(trial["trialId"])
        trial["trialId"] = f"{partition_key}-{old_trial_id}"
        trial["enrollmentClipId"] = clip_ids[str(trial["enrollmentClipId"])]
        trial["testClipId"] = clip_ids[str(trial["testClipId"])]
        trial["sourceSessionId"] = session_id
        trial["recordingRelation"] = "within-recording"
        trial["diagnosticOnly"] = True
    source = dict(manifest["source"])
    source_speakers = sorted(str(value) for value in session["speakerSet"])
    trial_speakers = sorted({str(clip["speakerId"]) for clip in clips})
    excluded_speakers = sorted(set(source_speakers) - set(trial_speakers))
    if excluded_speakers:
        raise AliMeetingDiagnosticError(
            "diagnostic clip constraints exclude official session speakers: "
            + ", ".join(excluded_speakers)
        )
    source.update(
        {
            "modality": modality,
            "sourceSessionId": session_id,
            "recordingIdentityUnit": session_id,
            "nearFarAreIndependentRecordings": False,
            "realOrSynthetic": (
                "real-recording"
                if modality == "far-channel-0"
                else "synthetic-mixture"
            ),
            "originalTruth": session["farTextGridEvidence"],
            "originalAudio": (
                [session["farAudioEvidence"]]
                if modality == "far-channel-0"
                else session["nearAudioEvidence"]
            ),
        }
    )
    selection = dict(manifest["selection"])
    selection["selectionUsesModelScores"] = False
    scoped.update(
        {
            "libraryId": f"alimeeting-eval-{partition_key}-diagnostic-v1",
            "source": source,
            "selection": selection,
            "clips": clips,
            "trials": trials,
            "identityScope": {
                "sameSpeakerPersistence": "within-source-recording-only",
                "crossSessionSpeakerReuseAvailable": False,
                "nearFarCaptureRelationship": "synchronized-same-source-recording",
            },
            "speakerCoverage": {
                "sourceSpeakerIds": source_speakers,
                "trialSpeakerIds": trial_speakers,
                "excludedSpeakerIds": [],
                "allOfficialSessionSpeakersCovered": True,
            },
            "usagePolicy": dict(USAGE_POLICY),
            "relatedDiarization": {
                "path": related_diarization["path"],
                "sha256": related_diarization["sha256"],
                "usedForClipSelection": False,
            },
        }
    )
    validate_partition_manifest(scoped)
    scoped["canonicalSha256"] = canonical_json_sha256(scoped)
    return scoped


def validate_partition_manifest(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source")
    clips = manifest.get("clips")
    trials = manifest.get("trials")
    policy = manifest.get("usagePolicy")
    coverage = manifest.get("speakerCoverage")
    if (
        not isinstance(source, dict)
        or not isinstance(clips, list)
        or not clips
        or not isinstance(trials, list)
        or not trials
        or policy != USAGE_POLICY
        or not isinstance(coverage, dict)
        or coverage.get("excludedSpeakerIds") != []
        or coverage.get("allOfficialSessionSpeakersCovered") is not True
        or coverage.get("sourceSpeakerIds") != coverage.get("trialSpeakerIds")
    ):
        raise AliMeetingDiagnosticError("diagnostic partition is incomplete")
    recording_id = source.get("recordingId")
    session_id = source.get("sourceSessionId")
    if recording_id != session_id or source.get("nearFarAreIndependentRecordings"):
        raise AliMeetingDiagnosticError("near/far recording identity is invalid")
    clip_by_id: dict[str, Mapping[str, Any]] = {}
    for clip in clips:
        if (
            not isinstance(clip, dict)
            or clip.get("sourceRecordingId") != session_id
            or clip.get("sourceSessionId") != session_id
            or clip.get("evaluationSplit") != "held-out"
            or clip.get("diagnosticOnly") is not True
            or clip.get("clipId") in clip_by_id
        ):
            raise AliMeetingDiagnosticError("diagnostic clip crosses its session")
        clip_by_id[str(clip["clipId"])] = clip
    labels: set[bool] = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise AliMeetingDiagnosticError("diagnostic trial is invalid")
        left = clip_by_id.get(str(trial.get("enrollmentClipId")))
        right = clip_by_id.get(str(trial.get("testClipId")))
        same_speaker = trial.get("sameSpeaker")
        if (
            left is None
            or right is None
            or not isinstance(same_speaker, bool)
            or same_speaker != (left["speakerId"] == right["speakerId"])
            or trial.get("sourceSessionId") != session_id
            or trial.get("evaluationSplit") != "held-out"
            or trial.get("recordingRelation") != "within-recording"
            or trial.get("diagnosticOnly") is not True
        ):
            raise AliMeetingDiagnosticError("diagnostic trial crosses its session")
        labels.add(same_speaker)
    if labels != {False, True}:
        raise AliMeetingDiagnosticError("diagnostic trial classes are incomplete")


def _source_audio(
    session: Mapping[str, Any],
    modality: str,
    output: Path,
    *,
    cached_audio: Path | None = None,
) -> dict[str, Any]:
    if cached_audio is not None:
        cached_audio = cached_audio.resolve(strict=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite normalized audio: {output}")
        try:
            os.link(cached_audio, output)
        except OSError as exc:
            raise AliMeetingDiagnosticError(
                "cannot hard-link cached normalized audio"
            ) from exc
    elif modality == "far-channel-0":
        _normalize_far_channel_zero(Path(session["farAudio"]), output)
    elif modality == "synchronized-near-field-mixture":
        _normalize_near_mix(
            [Path(path) for path in session["nearAudio"]],
            output,
        )
    else:
        raise AliMeetingDiagnosticError("diagnostic modality is unsupported")
    evidence = _mono_wave_evidence(output)
    evidence["transform"] = (
        "ffmpeg-pan-first-array-channel-to-mono-v1"
        if modality == "far-channel-0"
        else "ffmpeg-equal-weight-synchronized-amix-v1"
    )
    evidence["reusedByHardLink"] = cached_audio is not None
    return evidence


def build_diagnostics(
    *,
    global_manifest_path: Path,
    related_diarization_path: Path,
    output_root: Path,
    alimeeting_archive: Path | None = None,
    alimeeting_root: Path | None = None,
    normalized_audio_cache_root: Path | None = None,
    clip_duration_ms: int = 2_000,
    maximum_clips_per_speaker: int = 10,
    minimum_spacing_ms: int = 15_000,
    minimum_pair_separation_ms: int = 30_000,
    maximum_pairs_per_class: int = 128,
    random_seed: int = 20_260_807,
) -> Path:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output root: {output_root}")
    manifest = load_global_manifest(global_manifest_path)
    plan = _planned_by_id(manifest, "alimeeting")
    _validate_alimeeting_plan(plan)
    if plan["evaluationSplit"] != "held-out":
        raise AliMeetingDiagnosticError("AliMeeting diagnostics must be held-out")
    root, tree_evidence, archive_verified = _resolve_alimeeting_corpus(
        output_root=output_root,
        plan=plan,
        archive_path=alimeeting_archive,
        corpus_root=alimeeting_root,
    )
    sessions = _discover_alimeeting_sessions(root, plan, tree_evidence)
    identity_audit = audit_identity_scope(sessions)
    related = _related_diarization_evidence(
        related_diarization_path,
        plan=plan,
        sessions=sessions,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    partition_records: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session["sessionId"])
        for modality_index, modality in enumerate(MODALITIES):
            partition_key = f"{session_id.lower()}-{modality}"
            audio_path = output_root / "audio" / f"{partition_key}.wav"
            annotation_path = (
                output_root / "annotations" / f"{partition_key}.json"
            )
            cached_audio = (
                normalized_audio_cache_root.resolve(strict=True)
                / f"{partition_key}.wav"
                if normalized_audio_cache_root is not None
                else None
            )
            audio_evidence = _source_audio(
                session,
                modality,
                audio_path,
                cached_audio=cached_audio,
            )
            annotation = _annotation_document(
                session,
                plan=plan,
                modality=modality,
            )
            _write_manifest(annotation_path, annotation)
            base = build_single_recording_manifest(
                audio_path=audio_path,
                annotation_path=annotation_path,
                source_recording_id=session_id,
                evaluation_split="held-out",
                clip_duration_ms=clip_duration_ms,
                maximum_clips_per_speaker=maximum_clips_per_speaker,
                minimum_spacing_ms=minimum_spacing_ms,
                minimum_pair_separation_ms=minimum_pair_separation_ms,
                maximum_pairs_per_class=maximum_pairs_per_class,
                random_seed=random_seed + modality_index,
            )
            partition = _scope_partition_manifest(
                base,
                session=session,
                modality=modality,
                related_diarization=related,
            )
            partition_path = (
                output_root / "partitions" / f"{partition_key}.json"
            )
            _write_manifest(partition_path, partition)
            partition_records.append(
                {
                    "sourceSessionId": session_id,
                    "sourceRecordingId": session_id,
                    "modality": modality,
                    "evaluationSplit": "held-out",
                    "path": str(partition_path.relative_to(output_root)),
                    "bytes": partition_path.stat().st_size,
                    "sha256": sha256_file(partition_path),
                    "canonicalSha256": partition["canonicalSha256"],
                    "audio": audio_evidence,
                    "counts": partition["counts"],
                    "speakerCoverage": partition["speakerCoverage"],
                    "diagnosticOnly": True,
                }
            )
    recording_splits = {
        session_id: ["held-out"]
        for session_id in sorted(identity_audit["speakerIdsBySession"])
    }
    index: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "libraryId": "alimeeting-eval-within-recording-speaker-diagnostics-v1",
        "source": {
            "dataset": plan["dataset"],
            "revision": plan["revision"],
            "license": plan["license"],
            "licenseDecision": plan["licenseDecision"],
            "archiveSha256": plan["archiveSha256"],
            "archiveVerifiedThisRun": archive_verified,
            "extractedTreeSha256": tree_evidence["treeSha256"],
            "globalManifestPath": str(global_manifest_path.resolve()),
            "globalManifestSha256": sha256_file(global_manifest_path),
            "relatedDiarization": related,
        },
        "identityAudit": identity_audit,
        "splitIsolation": {
            "groupingUnit": "sourceSessionId",
            "recordingIdsBySplit": {
                "development": [],
                "regression": [],
                "held-out": sorted(recording_splits),
            },
            "splitsByRecordingId": recording_splits,
            "recordingLeakageCheckPassed": True,
        },
        "selection": {
            "algorithm": "official-textgrid-solo-speaker-balanced-v1",
            "clipDurationMs": clip_duration_ms,
            "maximumClipsPerSpeaker": maximum_clips_per_speaker,
            "minimumSpacingMs": minimum_spacing_ms,
            "minimumPairSeparationMs": minimum_pair_separation_ms,
            "maximumPairsPerClassPerPartition": maximum_pairs_per_class,
            "randomSeed": random_seed,
            "selectionUsesModelScores": False,
        },
        "usagePolicy": dict(USAGE_POLICY),
        "counts": {
            "sourceSessions": len(sessions),
            "partitions": len(partition_records),
            "sourceSpeakerIdentities": identity_audit["uniqueSpeakerCount"],
            "trialSpeakerIdentities": len(
                {
                    speaker_id
                    for row in partition_records
                    for speaker_id in row["speakerCoverage"]["trialSpeakerIds"]
                }
            ),
            "clips": sum(row["counts"]["clips"] for row in partition_records),
            "sameSpeakerTrials": sum(
                row["counts"]["sameSpeakerTrials"] for row in partition_records
            ),
            "differentSpeakerTrials": sum(
                row["counts"]["differentSpeakerTrials"]
                for row in partition_records
            ),
            "totalTrials": sum(
                row["counts"]["totalTrials"] for row in partition_records
            ),
        },
        "partitions": partition_records,
    }
    if len(partition_records) != len(sessions) * len(MODALITIES):
        raise AliMeetingDiagnosticError("diagnostic partition matrix is incomplete")
    if (
        index["counts"]["trialSpeakerIdentities"]
        != index["counts"]["sourceSpeakerIdentities"]
    ):
        raise AliMeetingDiagnosticError(
            "diagnostic partitions do not cover every official speaker identity"
        )
    index["canonicalSha256"] = canonical_json_sha256(index)
    destination = output_root / "alimeeting-speaker-diagnostics.v1.json"
    _write_manifest(destination, index)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--global-manifest",
        type=Path,
        default=DEFAULT_GLOBAL_MANIFEST,
    )
    parser.add_argument(
        "--real-diarization-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--alimeeting-archive", type=Path)
    parser.add_argument("--alimeeting-root", type=Path)
    parser.add_argument("--normalized-audio-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-duration-ms", type=_positive_integer, default=2_000)
    parser.add_argument(
        "--maximum-clips-per-speaker",
        type=_positive_integer,
        default=10,
    )
    parser.add_argument(
        "--minimum-spacing-ms",
        type=_positive_integer,
        default=15_000,
    )
    parser.add_argument(
        "--minimum-pair-separation-ms",
        type=_positive_integer,
        default=30_000,
    )
    parser.add_argument(
        "--maximum-pairs-per-class",
        type=_positive_integer,
        default=128,
    )
    parser.add_argument("--random-seed", type=int, default=20_260_807)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = build_diagnostics(
        global_manifest_path=args.global_manifest,
        related_diarization_path=args.real_diarization_manifest,
        output_root=args.output_root,
        alimeeting_archive=args.alimeeting_archive,
        alimeeting_root=args.alimeeting_root,
        normalized_audio_cache_root=args.normalized_audio_cache_root,
        clip_duration_ms=args.clip_duration_ms,
        maximum_clips_per_speaker=args.maximum_clips_per_speaker,
        minimum_spacing_ms=args.minimum_spacing_ms,
        minimum_pair_separation_ms=args.minimum_pair_separation_ms,
        maximum_pairs_per_class=args.maximum_pairs_per_class,
        random_seed=args.random_seed,
    )
    document = _load_json_object(destination, "diagnostic index")
    print(
        json.dumps(
            {
                "output": str(destination),
                "canonicalSha256": document["canonicalSha256"],
                "identityAudit": document["identityAudit"],
                "usagePolicy": document["usagePolicy"],
                "counts": document["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
