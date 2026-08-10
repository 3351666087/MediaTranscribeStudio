"""Freeze one official AliMeeting Train far-field session for development."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.persistence import canonical_json_sha256
from tools.build_global_real_diarization import (
    GlobalSampleLibraryError,
    _clip_audio,
    _clip_reference_transcript,
    _probe_alimeeting_source_wave,
    _probe_audio,
    alimeeting_textgrid_turns,
    parse_alimeeting_textgrid,
    select_alimeeting_textgrid_window,
)


DEFAULT_ARCHIVE_URL = (
    "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
    "AliMeeting/openlr/Train_Ali_far.tar.gz"
)
DEFAULT_ARCHIVE_BYTES = 78_639_309_701
DEFAULT_ARCHIVE_ETAG = '"B5C1C3F463D4D0393241F7A11C3E303F-7500"'
DEFAULT_ARCHIVE_CRC64 = "7416259116226311466"
DEFAULT_SESSION_ID = "R0003_M0046"
SESSION_ID = re.compile(r"^R\d{4}_M\d{4}$")
ARCHIVE_ROOT = "Train_Ali_far"


class TrainFreezeError(ValueError):
    """Raised when the official Train evidence cannot be frozen safely."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_held_out_isolation(
    held_out_manifest: Path,
    *,
    archive_sha256: str,
    session_id: str,
    source_audio_sha256: str,
    media_sha256: str,
) -> dict[str, Any]:
    held_out_manifest = held_out_manifest.resolve(strict=True)
    try:
        value = json.loads(held_out_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainFreezeError("held-out manifest is unreadable") from exc
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise TrainFreezeError("held-out manifest has no cases")

    sessions: set[str] = set()
    source_audio_sha256s: set[str] = set()
    media_sha256s: set[str] = set()
    for row in value["cases"]:
        if not isinstance(row, dict):
            raise TrainFreezeError("held-out manifest contains an invalid case")
        held_out_session = row.get("sourceSessionId")
        if isinstance(held_out_session, str) and held_out_session:
            sessions.add(held_out_session)
        held_out_media_sha256 = row.get("sha256")
        if _is_sha256(held_out_media_sha256):
            media_sha256s.add(str(held_out_media_sha256))
        artifacts = row.get("sourceAudioArtifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if isinstance(artifact, dict) and _is_sha256(artifact.get("sha256")):
                    source_audio_sha256s.add(str(artifact["sha256"]))

    archive_sha256s: set[str] = set()
    sources = value.get("sources")
    if isinstance(sources, list):
        for source in sources:
            archive = source.get("archive") if isinstance(source, dict) else None
            if isinstance(archive, dict) and _is_sha256(archive.get("sha256")):
                archive_sha256s.add(str(archive["sha256"]))
    if (
        not archive_sha256s
        or not sessions
        or not source_audio_sha256s
        or not media_sha256s
    ):
        raise TrainFreezeError(
            "held-out manifest lacks archive, session, source-audio, or media identities"
        )

    conflicts = {
        "archiveSha256Disjoint": archive_sha256 not in archive_sha256s,
        "sessionIdDisjoint": session_id not in sessions,
        "sourceAudioSha256Disjoint": source_audio_sha256 not in source_audio_sha256s,
        "mediaSha256Disjoint": media_sha256 not in media_sha256s,
    }
    failed = sorted(key for key, passed in conflicts.items() if not passed)
    if failed:
        raise TrainFreezeError(
            "development/held-out recording isolation failed: " + ", ".join(failed)
        )
    return {
        "policy": "archive-session-source-audio-derived-media-sha256-v1",
        "heldOutManifest": {
            "path": str(held_out_manifest),
            "sha256": _sha256_file(held_out_manifest),
        },
        "heldOutInventory": {
            "archiveSha256Count": len(archive_sha256s),
            "sessionIdCount": len(sessions),
            "sourceAudioSha256Count": len(source_audio_sha256s),
            "mediaSha256Count": len(media_sha256s),
        },
        "checks": conflicts,
    }


def _normalize_train_textgrid_speaker_ids(
    grid: dict[str, Any],
    *,
    session_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Map official Train tier names to the shared normalized speaker IDs."""

    tier_pattern = re.compile(
        rf"^{re.escape(session_id)}_[FM]_SPK([0-9]{{4}})$"
    )
    normalized_tiers: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for tier in grid.get("tiers", []):
        raw_name = str(tier.get("name", ""))
        match = tier_pattern.fullmatch(raw_name)
        if match is None:
            raise TrainFreezeError(
                f"Train TextGrid tier speaker ID is invalid: {raw_name!r}"
            )
        normalized_name = f"N_SPK{match.group(1)}"
        previous = mapping.get(raw_name)
        if previous is not None or normalized_name in mapping.values():
            raise TrainFreezeError(
                f"Train TextGrid speaker ID is duplicated: {raw_name!r}"
            )
        mapping[raw_name] = normalized_name
        normalized_tiers.append({**tier, "name": normalized_name})
    if not normalized_tiers:
        raise TrainFreezeError("Train TextGrid has no speaker tiers")
    return {**grid, "tiers": normalized_tiers}, mapping


def _safe_member_name(member: tarfile.TarInfo) -> str:
    raw = member.name
    name = raw.rstrip("/") if member.isdir() else raw
    path = PurePosixPath(name)
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or not path.parts
        or path.parts[0] != ARCHIVE_ROOT
        or any(not part or part in {".", ".."} for part in parts)
        or any(ord(character) < 32 for character in name)
        or not (member.isdir() or member.isreg())
    ):
        raise TrainFreezeError(f"unsafe archive member: {raw!r}")
    if member.isreg() and member.size < 0:
        raise TrainFreezeError(f"negative archive member size: {raw!r}")
    return name


def _extract_selected_members(
    archive: Path,
    *,
    session_id: str,
    destination: Path,
    reuse_existing: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    if not SESSION_ID.fullmatch(session_id):
        raise TrainFreezeError(f"invalid session id: {session_id}")
    audio_pattern = re.compile(
        rf"^{re.escape(ARCHIVE_ROOT)}/audio_dir/"
        rf"{re.escape(session_id)}_MS\d{{3}}\.wav$"
    )
    textgrid_name = f"{ARCHIVE_ROOT}/textgrid_dir/{session_id}.TextGrid"
    staging = destination / f".{session_id}.part"
    if staging.exists():
        raise TrainFreezeError(f"staging directory already exists: {staging}")
    final = destination / session_id
    if final.exists():
        if not reuse_existing:
            raise TrainFreezeError(f"refusing to overwrite extracted session: {final}")
        audio_candidates = sorted(
            (final / "audio_dir").glob(f"{session_id}_MS[0-9][0-9][0-9].wav")
        )
        textgrid_path = final / "textgrid_dir" / f"{session_id}.TextGrid"
        expected_paths = {
            *audio_candidates,
            textgrid_path,
        }
        if len(audio_candidates) != 1 or not textgrid_path.is_file():
            raise TrainFreezeError(
                f"existing extraction is incomplete: {final}"
            )
        for path in final.rglob("*"):
            if path.is_symlink() or (path.is_file() and path not in expected_paths):
                raise TrainFreezeError(
                    f"existing extraction contains unexpected member: {path}"
                )
        return (
            audio_candidates[0],
            textgrid_path,
            {
                "membersScanned": None,
                "audioMember": f"{ARCHIVE_ROOT}/audio_dir/{audio_candidates[0].name}",
                "textgridMember": textgrid_name,
                "reusedExistingExtraction": True,
            },
        )
    staging.mkdir(parents=True, exist_ok=False)
    audio_path: Path | None = None
    textgrid_path: Path | None = None
    audio_member_name: str | None = None
    textgrid_member_name: str | None = None
    seen = 0
    try:
        with tarfile.open(archive, mode="r|gz") as handle:
            for member in handle:
                name = _safe_member_name(member)
                seen += 1
                selected = audio_pattern.fullmatch(name) or name == textgrid_name
                if not selected:
                    continue
                relative = PurePosixPath(name).relative_to(ARCHIVE_ROOT)
                target = staging.joinpath(*relative.parts)
                if target.exists():
                    raise TrainFreezeError(f"duplicate selected member: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise TrainFreezeError(f"selected member is unreadable: {name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                if name == textgrid_name:
                    textgrid_path = target
                    textgrid_member_name = name
                else:
                    if audio_path is not None:
                        raise TrainFreezeError(
                            "session has multiple far audio members: "
                            f"{audio_member_name}, {name}"
                        )
                    audio_path = target
                    audio_member_name = name
        if audio_path is None or textgrid_path is None:
            raise TrainFreezeError(
                f"session pair not found: audio={audio_path is not None}, "
                f"textgrid={textgrid_path is not None}, membersScanned={seen}"
            )
        staging.replace(final)
        return (
            final / audio_path.relative_to(staging),
            final / textgrid_path.relative_to(staging),
            {
                "membersScanned": seen,
                "audioMember": audio_member_name,
                "textgridMember": textgrid_member_name,
            },
        )
    except (OSError, tarfile.TarError) as exc:
        raise TrainFreezeError(f"safe selective extraction failed: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _select_window(turns: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    speakers = sorted({str(turn["speakerId"]) for turn in turns})
    if len(speakers) < 2:
        raise TrainFreezeError("Train session has fewer than two speakers")
    errors: list[str] = []
    for count in range(len(speakers), 1, -1):
        try:
            return (
                select_alimeeting_textgrid_window(
                    turns,
                    count,
                    minimum_duration_seconds=10.0,
                    maximum_duration_seconds=90.0,
                    minimum_overlap_seconds=0.5,
                ),
                count,
            )
        except GlobalSampleLibraryError as exc:
            errors.append(f"N={count}: {exc}")
    raise TrainFreezeError("no qualifying Train window: " + "; ".join(errors))


def build_train_development(
    *,
    archive: Path,
    output_root: Path,
    session_id: str = DEFAULT_SESSION_ID,
    archive_url: str = DEFAULT_ARCHIVE_URL,
    archive_etag: str = DEFAULT_ARCHIVE_ETAG,
    archive_crc64: str = DEFAULT_ARCHIVE_CRC64,
    expected_archive_sha256: str | None = None,
    held_out_manifest: Path | None = None,
    reuse_existing_extraction: bool = False,
) -> Path:
    archive = archive.resolve(strict=True)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    archive_bytes = archive.stat().st_size
    if archive_bytes != DEFAULT_ARCHIVE_BYTES:
        raise TrainFreezeError(
            f"archive byte count mismatch: {archive_bytes} != {DEFAULT_ARCHIVE_BYTES}"
        )
    archive_sha256 = _sha256_file(archive)
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise TrainFreezeError(
            f"archive SHA-256 mismatch: {archive_sha256} != {expected_archive_sha256}"
        )
    extracted_root = output_root / "sources" / "alimeeting-train-far"
    audio_source, textgrid_source, extraction = _extract_selected_members(
        archive,
        session_id=session_id,
        destination=extracted_root,
        reuse_existing=reuse_existing_extraction,
    )
    source_audio_sha256 = _sha256_file(audio_source)
    source_textgrid_sha256 = _sha256_file(textgrid_source)
    source_duration = _probe_alimeeting_source_wave(audio_source, 8)
    grid = parse_alimeeting_textgrid(textgrid_source.read_text(encoding="utf-8"))
    normalized_grid, speaker_id_mapping = _normalize_train_textgrid_speaker_ids(
        grid,
        session_id=session_id,
    )
    turns = alimeeting_textgrid_turns(normalized_grid)
    window, target_count = _select_window(turns)
    case_id = f"alimeeting-train-{session_id.lower()}-far-n{target_count}"
    output_audio = output_root / "audio" / f"{case_id}.wav"
    _clip_audio(audio_source, output_audio, window)
    probe = _probe_audio(output_audio)
    if (
        probe.get("codec") != "pcm_s16le"
        or probe.get("sampleRate") != 16_000
        or probe.get("channels") != 1
        or float(probe.get("durationSeconds", 0)) > 90.05
    ):
        raise TrainFreezeError(f"normalized audio is invalid: {case_id}")
    references = _clip_reference_transcript(turns, window)
    scoring_transcript = "".join(str(item["transcript"]) for item in references)
    if not scoring_transcript:
        raise TrainFreezeError("selected TextGrid window has no transcript")
    source_manifest = {
        "sourceId": "alimeeting-train-far",
        "provider": "openslr",
        "dataset": "SLR119/AliMeeting",
        "revision": f"sha256:{archive_sha256}",
        "license": "cc-by-sa-4.0",
        "recordingType": "real-recording",
        "languageTags": ["zh-CN"],
        "split": "Train_Ali_far",
        "evaluationSplit": "development",
        "archive": {
            "url": archive_url,
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "etag": archive_etag,
            "crc64": archive_crc64,
            "verifiedThisRun": True,
        },
        "sourceAudio": {
            "member": extraction["audioMember"],
            "path": str(audio_source.relative_to(output_root)).replace("\\", "/"),
            "bytes": audio_source.stat().st_size,
            "sha256": source_audio_sha256,
            "durationSeconds": source_duration,
        },
        "sourceTextGrid": {
            "member": extraction["textgridMember"],
            "path": str(textgrid_source.relative_to(output_root)).replace("\\", "/"),
            "bytes": textgrid_source.stat().st_size,
            "sha256": source_textgrid_sha256,
        },
        "officialHomepage": "https://www.openslr.org/119/",
        "officialBaselineRepository": "https://github.com/yufan-aslp/AliMeeting",
        "officialBaselineRevision": "692a034fd510f1720f547c2c91e0fbadc9c24bf2",
        "speakerIdNormalization": (
            "official-tier-{session}_gender_SPK####-to-N_SPK####-v1"
        ),
        "licenseDecision": (
            "CC BY-SA 4.0 is applied from the official OpenSLR 119 data page; "
            "the Train archive is kept separate from the Eval held-out archive."
        ),
        "attribution": (
            "AliMeeting Mandarin multi-channel meeting corpus, Alibaba Group "
            "and AISHELL Foundation; OpenSLR SLR119; CC BY-SA 4.0."
        ),
        "sessionId": session_id,
        "speakerCount": len(window["speakerSet"]),
        "speakerIdMapping": speaker_id_mapping,
    }
    case = {
        "id": case_id,
        "sourceId": "alimeeting-train-far",
        "sourceDataset": "SLR119/AliMeeting",
        "sourceRevision": f"sha256:{archive_sha256}",
        "sourceSessionId": session_id,
        "sourceModality": "far-field-array",
        "sourceArtifactPath": source_manifest["sourceTextGrid"]["path"],
        "sourceArtifactSha256": source_textgrid_sha256,
        "sourceAudioArtifacts": [source_manifest["sourceAudio"]],
        "path": str(output_audio.relative_to(output_root)).replace("\\", "/"),
        "bytes": output_audio.stat().st_size,
        "sha256": _sha256_file(output_audio),
        "audio": probe,
        "realOrSynthetic": "real-recording",
        "language": "zh-CN",
        "region": "East Asia",
        "evaluationSplit": "development",
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
        "referenceTranscriptTurns": references,
        "scoringTranscript": scoring_transcript,
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
    held_out_isolation = (
        _validate_held_out_isolation(
            held_out_manifest,
            archive_sha256=archive_sha256,
            session_id=session_id,
            source_audio_sha256=source_audio_sha256,
            media_sha256=str(case["sha256"]),
        )
        if held_out_manifest is not None
        else None
    )
    attribution = output_root / "ATTRIBUTION.md"
    attribution.write_text(
        "# AliMeeting Train Development Attribution\n\n"
        f"- Dataset: `SLR119/AliMeeting`\n"
        f"- Split: `Train_Ali_far` (development only)\n"
        f"- Archive URL: {archive_url}\n"
        f"- Archive SHA-256: `{archive_sha256}`\n"
        "- License: `cc-by-sa-4.0`\n"
        "- The selected Train session is separate from the Eval held-out archive.\n",
        encoding="utf-8",
    )
    resolved = {
        "schemaVersion": "1.0.0",
        "libraryId": "mts-alimeeting-train-development-real-diarization-v1",
        "generatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "windowSelection": {
            "algorithms": ["textgrid-component-shortest-coverage-v1"],
            "minimumPerSpeakerAnnotatedSeconds": 0.5,
            "maximumDurationSeconds": 90.0,
            "selectionUsesModelScores": False,
        },
        "sources": [source_manifest],
        "cases": [case],
        "failedCases": [],
        "attributionPath": "ATTRIBUTION.md",
    }
    if held_out_isolation is not None:
        resolved["heldOutIsolation"] = held_out_isolation
    resolved["canonicalSha256"] = canonical_json_sha256(resolved)
    destination = output_root / "global-real-diarization.resolved.v1.json"
    if destination.exists():
        raise TrainFreezeError(f"refusing to overwrite resolved manifest: {destination}")
    destination.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--held-out-manifest", type=Path)
    parser.add_argument(
        "--reuse-existing-extraction",
        action="store_true",
        help="reuse a previously safe extracted session after a post-extraction failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = build_train_development(
        archive=args.archive,
        output_root=args.output_root,
        session_id=args.session_id,
        expected_archive_sha256=args.expected_archive_sha256,
        held_out_manifest=args.held_out_manifest,
        reuse_existing_extraction=args.reuse_existing_extraction,
    )
    value = json.loads(destination.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "resolvedManifest": str(destination),
                "caseCount": len(value["cases"]),
                "canonicalSha256": value["canonicalSha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
