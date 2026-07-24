"""Build the short multilingual sample library from its checked-in manifest.

Generated media is intentionally written under .runtime_cache and never
committed. macOS ``say`` supplies deterministic local voices; one public
upstream reference clip is downloaded with a pinned SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sample_library import SampleCase, SampleLibraryError, load_manifest

DEFAULT_SPEC = ROOT / "sample_library" / "manifest.v1.json"
DEFAULT_OUTPUT = ROOT / ".runtime_cache" / "sample-library"
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2


def _run(command: Sequence[str], *, timeout: float = 120.0) -> None:
    subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return float(completed.stdout.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _say_to_wav(voice: str, text: str, output: Path, work: Path) -> None:
    aiff = work / f"{output.stem}.aiff"
    _run(["say", "-v", voice, "-o", str(aiff), text])
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(aiff),
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def _concat_wav(
    paths: Sequence[Path],
    output: Path,
    work: Path,
    *,
    gap_ms: int = 0,
) -> None:
    concat_paths: list[Path] = []
    for index, path in enumerate(paths):
        concat_paths.append(path)
        if gap_ms > 0 and index < len(paths) - 1:
            silence = work / f"silence-{index}.wav"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"anullsrc=r={SAMPLE_RATE}:cl=mono:d={gap_ms / 1000:.3f}",
                    "-ac",
                    str(CHANNELS),
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    str(silence),
                ]
            )
            concat_paths.append(silence)
    concat_list = work / f"{output.stem}.concat.txt"
    concat_list.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in concat_paths),
        encoding="utf-8",
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def _mix_overlap(paths: Sequence[Path], offsets_ms: Sequence[int], output: Path) -> None:
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, (path, offset_ms) in enumerate(zip(paths, offsets_ms)):
        inputs.extend(["-i", str(path)])
        label = f"a{index}"
        filters.append(
            f"[{index}:a]adelay={offset_ms}|{offset_ms}[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(paths)}:duration=longest:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.95[out]"
    )
    _run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def _apply_effects(source: Path, output: Path, effects: Sequence[str]) -> None:
    filters: list[str] = []
    if "phone-band" in effects:
        filters.extend(["highpass=f=250", "lowpass=f=3600"])
    if "room-reverb" in effects:
        filters.append("aecho=0.8:0.88:70:0.22")
    if filters:
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-af",
                ",".join(filters),
                "-ac",
                str(CHANNELS),
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(output),
            ]
        )
        source = output
    if "white-noise" in effects:
        intermediate = output.with_suffix(".noise.wav")
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=white:amplitude=0.018:sample_rate=16000",
                "-filter_complex",
                "[1:a]atrim=duration=30[n];[0:a][n]amix=inputs=2:duration=first:weights='1 0.12':normalize=0,alimiter=limit=0.95[out]",
                "-map",
                "[out]",
                "-ac",
                str(CHANNELS),
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(intermediate),
            ]
        )
        intermediate.replace(output)
    if not output.exists():
        shutil.copyfile(source, output)


def _build_video(audio: Path, output: Path) -> None:
    duration = _probe_duration(audio)
    half = max(0.5, duration / 2)
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x20364f:s=640x360:r=24:d={half:.3f}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x6b2e35:s=640x360:r=24:d={half:.3f}",
            "-i",
            str(audio),
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(output),
        ]
    )


def _download_remote(case: SampleCase, destination: Path) -> None:
    if case.remote is None:
        raise SampleLibraryError(f"{case.case_id} has no remote source")
    request = urllib.request.Request(
        str(case.remote["url"]),
        headers={"User-Agent": "MediaTranscribeStudio-sample-library/1.0"},
    )
    expected = str(case.remote["sha256"])
    expected_size: int | None = None
    for attempt in range(1, 4):
        temporary = destination.with_suffix(destination.suffix + ".download")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_size = response.headers.get("Content-Length")
                expected_size = int(raw_size) if raw_size else None
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            actual_size = temporary.stat().st_size
            actual = _sha256(temporary)
            if (expected_size is None or actual_size == expected_size) and actual == expected:
                temporary.replace(destination)
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        if temporary.exists():
            temporary.unlink()
    raise SampleLibraryError(
        f"{case.case_id} remote download did not match pinned size/hash "
        f"(expected {expected_size or 'unknown'} bytes, {expected})"
    )


def build_case(case: SampleCase, output_root: Path, max_duration: float) -> dict[str, Any]:
    destination = output_root / case.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{case.case_id}-", dir=output_root / ".work") as temp:
        work = Path(temp)
        if case.remote is not None:
            _download_remote(case, destination)
            if case.kind == "video":
                raise SampleLibraryError("remote video cases are not supported")
        else:
            clips: list[Path] = []
            for index, utterance in enumerate(case.utterances):
                clip = work / f"utterance-{index}.wav"
                voice = utterance.get("voice")
                if not voice:
                    raise SampleLibraryError(
                        f"{case.case_id} utterance {index} needs a local voice"
                    )
                _say_to_wav(str(voice), str(utterance["text"]), clip, work)
                clips.append(clip)
            assembled = work / "assembled.wav"
            if case.scenario == "overlap":
                _mix_overlap(
                    clips,
                    [int(item["offsetMs"]) for item in case.utterances],
                    assembled,
                )
            elif len(clips) == 1:
                shutil.copyfile(clips[0], assembled)
            else:
                _concat_wav(
                    clips,
                    assembled,
                    work,
                    gap_ms=180 if "short-gaps" in case.effects else 0,
                )
            if case.effects:
                _apply_effects(assembled, work / "effect.wav", case.effects)
                assembled = work / "effect.wav"
            if case.kind == "video":
                _build_video(assembled, destination)
            else:
                shutil.copyfile(assembled, destination)
        duration = _probe_duration(destination)
        if duration > max_duration + 0.05:
            raise SampleLibraryError(
                f"{case.case_id} exceeds {max_duration:.2f}s: {duration:.3f}s"
            )
        return {
            "id": case.case_id,
            "language": case.language,
            "scenario": case.scenario,
            "kind": case.kind,
            "path": str(destination.relative_to(output_root)),
            "durationSeconds": round(duration, 3),
            "sizeBytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "expectedSpeakerCount": len(case.speaker_ids),
            "expectedTurnCount": len(case.utterances),
            "expectedSpeakerSequence": [
                str(item["speaker"]) for item in case.utterances
            ],
            "expectedOverlap": case.scenario == "overlap",
            "expectedTranscript": case.expected_transcript
            or " ".join(str(item["text"]) for item in case.utterances),
            "remote": case.remote,
        }


def build_library(spec_path: Path, output_root: Path, *, force: bool = False) -> Path:
    manifest = load_manifest(spec_path)
    if force and output_root.exists():
        generated_names = {
            ".work",
            "audio",
            "video",
            "sample-library.resolved.v1.json",
        }
        for child in output_root.iterdir():
            if child.name not in generated_names:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ".work").mkdir(parents=True, exist_ok=True)
    resolved_cases = [
        build_case(case, output_root, manifest.max_duration_seconds)
        for case in manifest.cases
    ]
    resolved = {
        "schemaVersion": manifest.schema_version,
        "libraryId": manifest.library_id,
        "maxDurationSeconds": manifest.max_duration_seconds,
        "sourceManifest": str(spec_path.resolve()),
        "cases": resolved_cases,
    }
    destination = output_root / "sample-library.resolved.v1.json"
    destination.write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = build_library(
            args.spec.resolve(),
            args.output_root.resolve(),
            force=args.force,
        )
    except (OSError, SampleLibraryError, subprocess.SubprocessError) as exc:
        print(f"sample library build failed: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
