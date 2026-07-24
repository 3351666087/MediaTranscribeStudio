"""Build deterministic Dynamic-N mixtures with exact diarization truth.

The source recordings stay under ``.runtime_cache``. Generated WAV files and
their resolved truth manifest are evaluation evidence and are never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.global_sample_library import (  # noqa: E402
    GlobalSampleLibraryError,
    load_global_manifest,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "sample_library" / "global-manifest.v1.json"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / ".runtime_cache" / "sample-library" / "global"
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT / "derived"
SOURCE_RESOLVED_NAME = "global-sample-library.resolved.v1.json"
RESOLVED_NAME = "global-derived-diarization.resolved.v1.json"
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2
SEQUENTIAL_GAP_FRAMES = round(0.18 * SAMPLE_RATE)
OVERLAP_OFFSET_FRAMES = round(0.4 * SAMPLE_RATE)
OVERLAP_CYCLE_GAP_FRAMES = round(0.25 * SAMPLE_RATE)


@dataclass(frozen=True)
class PcmSource:
    case_id: str
    language: str
    transcript: str
    path: Path
    sha256: str
    samples: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalSampleLibraryError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GlobalSampleLibraryError(f"JSON evidence must be an object: {path}")
    return value


def _load_pcm_source(root: Path, row: dict[str, Any]) -> PcmSource:
    case_id = row.get("id")
    relative_path = row.get("path")
    expected_hash = row.get("sha256")
    transcript = row.get("expectedTranscript")
    language = row.get("language")
    if not all(
        isinstance(value, str) and value
        for value in (case_id, relative_path, expected_hash, transcript, language)
    ):
        raise GlobalSampleLibraryError("resolved source row is incomplete")
    path = root / relative_path
    if not path.is_file():
        raise GlobalSampleLibraryError(f"source audio is missing: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise GlobalSampleLibraryError(f"source hash mismatch: {case_id}")
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != CHANNELS
            or handle.getsampwidth() != SAMPLE_WIDTH
            or handle.getframerate() != SAMPLE_RATE
            or handle.getcomptype() != "NONE"
        ):
            raise GlobalSampleLibraryError(
                f"{case_id} must be mono PCM s16le at {SAMPLE_RATE} Hz"
            )
        samples = np.frombuffer(
            handle.readframes(handle.getnframes()),
            dtype="<i2",
        ).astype(np.int32)
    if samples.size == 0:
        raise GlobalSampleLibraryError(f"{case_id} audio is empty")
    return PcmSource(
        case_id=case_id,
        language=language,
        transcript=transcript,
        path=path,
        sha256=actual_hash,
        samples=samples,
    )


def _write_pcm(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pcm = np.asarray(samples, dtype="<i2")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    temporary.replace(path)


def _turn(
    source: PcmSource,
    *,
    start_frame: int,
    end_frame: int,
    cycle: int,
) -> dict[str, Any]:
    return {
        "speakerId": source.case_id,
        "sourceCaseId": source.case_id,
        "language": source.language,
        "cycle": cycle,
        "startSeconds": round(start_frame / SAMPLE_RATE, 6),
        "endSeconds": round(end_frame / SAMPLE_RATE, 6),
        "transcript": source.transcript,
    }


def overlap_intervals(turns: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exact regions where at least two distinct speakers are active."""

    boundaries = sorted(
        {
            float(turn[key])
            for turn in turns
            for key in ("startSeconds", "endSeconds")
        }
    )
    intervals: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = sorted(
            {
                str(turn["speakerId"])
                for turn in turns
                if float(turn["startSeconds"]) < end
                and float(turn["endSeconds"]) > start
            }
        )
        if len(active) < 2:
            continue
        if (
            intervals
            and intervals[-1]["speakerIds"] == active
            and abs(float(intervals[-1]["endSeconds"]) - start) < 1e-6
        ):
            intervals[-1]["endSeconds"] = round(end, 6)
        else:
            intervals.append(
                {
                    "startSeconds": round(start, 6),
                    "endSeconds": round(end, 6),
                    "speakerIds": active,
                }
            )
    return intervals


def _sequential_mix(sources: Sequence[PcmSource]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    chunks: list[np.ndarray] = []
    turns: list[dict[str, Any]] = []
    cursor = 0
    for index, source in enumerate(sources):
        start = cursor
        end = start + source.samples.size
        chunks.append(source.samples)
        turns.append(_turn(source, start_frame=start, end_frame=end, cycle=0))
        cursor = end
        if index < len(sources) - 1:
            chunks.append(np.zeros(SEQUENTIAL_GAP_FRAMES, dtype=np.int32))
            cursor += SEQUENTIAL_GAP_FRAMES
    return np.concatenate(chunks), turns


def _overlap_mix(
    sources: Sequence[PcmSource],
    *,
    cycles: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    placements: list[tuple[PcmSource, int, int]] = []
    turns: list[dict[str, Any]] = []
    cycle_start = 0
    for cycle in range(cycles):
        ordered = list(sources if cycle % 2 == 0 else reversed(sources))
        cycle_end = cycle_start
        for index, source in enumerate(ordered):
            start = cycle_start + index * OVERLAP_OFFSET_FRAMES
            end = start + source.samples.size
            placements.append((source, start, end))
            turns.append(
                _turn(source, start_frame=start, end_frame=end, cycle=cycle)
            )
            cycle_end = max(cycle_end, end)
        cycle_start = cycle_end + OVERLAP_CYCLE_GAP_FRAMES
    output_frames = max(end for _, _, end in placements)
    mixed = np.zeros(output_frames, dtype=np.int64)
    for source, start, end in placements:
        mixed[start:end] += source.samples
    peak = int(np.max(np.abs(mixed))) if mixed.size else 0
    gain = min(1.0, 32767.0 / peak) if peak else 1.0
    if gain < 1.0:
        mixed = np.rint(mixed * gain)
    return np.clip(mixed, -32768, 32767).astype(np.int32), turns, gain


def _resolved_case(
    *,
    matrix_id: str,
    kind: str,
    sources: Sequence[PcmSource],
    samples: np.ndarray,
    turns: list[dict[str, Any]],
    overlap: list[dict[str, Any]],
    gain: float,
    output_root: Path,
    max_duration: float,
) -> dict[str, Any]:
    speaker_count = len(sources)
    case_id = f"{matrix_id}-n{speaker_count}"
    output = output_root / "audio" / f"{case_id}.wav"
    duration = samples.size / SAMPLE_RATE
    if duration > max_duration:
        raise GlobalSampleLibraryError(
            f"{case_id} is {duration:.3f}s and exceeds {max_duration:.3f}s"
        )
    _write_pcm(output, samples)
    return {
        "id": case_id,
        "matrixId": matrix_id,
        "kind": kind,
        "evaluationRole": "development-stress",
        "realOrSynthetic": "synthetic-mixture-of-real-recordings",
        "expectedSpeakerCount": speaker_count,
        "speakerSet": [source.case_id for source in sources],
        "sourceCaseIds": [source.case_id for source in sources],
        "sourceHashes": {
            source.case_id: source.sha256 for source in sources
        },
        "durationSeconds": round(duration, 6),
        "path": str(output.relative_to(output_root)),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "mixGain": round(gain, 9),
        "turns": turns,
        "overlapIntervals": overlap,
        "truthEligibility": {
            "speakerCount": True,
            "turnBoundaries": True,
            "overlap": True,
            "derJer": True,
            "asr": kind == "sequential-speaker-mixture",
        },
    }


def build_derived_matrix(
    manifest_path: Path,
    source_root: Path,
    output_root: Path,
) -> Path:
    manifest = load_global_manifest(manifest_path)
    resolved_path = source_root / SOURCE_RESOLVED_NAME
    source_resolved = _load_json(resolved_path)
    rows = source_resolved.get("cases")
    if not isinstance(rows, list):
        raise GlobalSampleLibraryError("resolved source cases are missing")
    row_by_id = {
        str(row["id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    source_cache: dict[str, PcmSource] = {}

    def sources_for(ids: Sequence[str]) -> list[PcmSource]:
        selected: list[PcmSource] = []
        for case_id in ids:
            if case_id not in row_by_id:
                raise GlobalSampleLibraryError(
                    f"derived source case is not downloaded: {case_id}"
                )
            if case_id not in source_cache:
                source_cache[case_id] = _load_pcm_source(
                    source_root,
                    row_by_id[case_id],
                )
            selected.append(source_cache[case_id])
        return selected

    cases: list[dict[str, Any]] = []
    for definition in manifest.derived_matrices:
        if not isinstance(definition, dict):
            raise GlobalSampleLibraryError("derived matrix definition must be an object")
        matrix_id = definition.get("id")
        kind = definition.get("kind")
        counts = definition.get("speakerCounts")
        source_ids = definition.get("sourceCaseIds")
        if (
            not isinstance(matrix_id, str)
            or kind not in {
                "sequential-speaker-mixture",
                "overlap-speaker-mixture",
            }
            or not isinstance(counts, list)
            or not isinstance(source_ids, list)
        ):
            raise GlobalSampleLibraryError(
                f"derived matrix definition is invalid: {matrix_id}"
            )
        for raw_count in counts:
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 1
                or raw_count > len(source_ids)
            ):
                raise GlobalSampleLibraryError(
                    f"{matrix_id} has invalid speaker count {raw_count}"
                )
            selected = sources_for([str(value) for value in source_ids[:raw_count]])
            if kind == "sequential-speaker-mixture":
                audio, turns = _sequential_mix(selected)
                gain = 1.0
            else:
                audio, turns, gain = _overlap_mix(selected)
            overlaps = overlap_intervals(turns)
            if kind == "overlap-speaker-mixture" and not overlaps:
                raise GlobalSampleLibraryError(f"{matrix_id} produced no overlap")
            cases.append(
                _resolved_case(
                    matrix_id=matrix_id,
                    kind=kind,
                    sources=selected,
                    samples=audio,
                    turns=turns,
                    overlap=overlaps,
                    gain=gain,
                    output_root=output_root,
                    max_duration=manifest.max_duration_seconds,
                )
            )
    resolved = {
        "schemaVersion": "1.0.0",
        "libraryId": f"{manifest.library_id}-derived-diarization",
        "generatedAt": datetime.now(UTC).isoformat(),
        "sourceManifest": str(manifest_path.resolve()),
        "sourceManifestSha256": _sha256(manifest_path),
        "sourceResolvedManifest": str(resolved_path.resolve()),
        "sourceResolvedManifestSha256": _sha256(resolved_path),
        "normalization": {
            "codec": "pcm_s16le",
            "sampleRate": SAMPLE_RATE,
            "channels": CHANNELS,
            "maxDurationSeconds": manifest.max_duration_seconds,
        },
        "cases": cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = build_derived_matrix(
        args.manifest,
        args.source_root,
        args.output_root,
    )
    resolved = _load_json(destination)
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
