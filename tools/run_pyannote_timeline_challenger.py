#!/usr/bin/env python3
"""Run one bounded offline Pyannote timeline challenger."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
    validate_strict_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser


def _duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16_000:
            raise ValueError("Pyannote challenger audio must be mono 16 kHz WAV")
        return round(handle.getnframes() * 1000 / handle.getframerate())


def main() -> int:
    args = _parser().parse_args()
    audio = args.audio.resolve(strict=True)
    model = args.model.resolve(strict=True)
    executable = args.python.resolve(strict=True)
    manifest_path = model / ".mts-model-manifest.json"
    manifest = read_json_strict(manifest_path)
    if (
        manifest.get("modelKey") != "pyannoteCommunity1"
        or manifest.get("repoId")
        != "pyannote/speaker-diarization-community-1"
    ):
        raise ValueError("Pyannote model manifest identity is invalid")
    duration_ms = _duration_ms(audio)
    request = {
        "schemaVersion": "1.1.0",
        "modelPath": str(model),
        "audioPath": str(audio),
        "device": args.device,
        "startMs": 0,
        "endMs": duration_ms,
        "speakerCountConstraints": None,
    }
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYANNOTE_METRICS_ENABLED": "0",
            "DO_NOT_TRACK": "1",
        }
    )
    started = time.monotonic()
    completed = subprocess.run(
        [str(executable), str(ROOT / "tools" / "pyannote_runtime.py")],
        input=json.dumps(request, separators=(",", ":")),
        text=True,
        capture_output=True,
        timeout=args.timeout_seconds,
        env=environment,
        check=False,
    )
    elapsed = time.monotonic() - started
    response = json.loads(completed.stdout)
    validate_strict_json(response)
    if (
        completed.returncode != 0
        or not isinstance(response, dict)
        or response.get("status") != "ok"
    ):
        raise RuntimeError(
            "isolated Pyannote timeline challenger failed "
            f"(exit={completed.returncode}, status={response.get('status')})"
        )
    regular = response.get("speakerTurns")
    exclusive = response.get("exclusiveSpeakerTurns")
    if not isinstance(regular, list) or not regular:
        raise ValueError("Pyannote challenger returned no regular speaker turns")
    local_speakers = sorted(
        {
            str(turn["localSpeaker"])
            for turn in regular
            if isinstance(turn, dict)
        }
    )
    if not local_speakers:
        raise ValueError("Pyannote challenger returned no speaker labels")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json_no_replace(
        output / "pyannote-timeline-result.v1.json",
        response,
    )
    report = {
        "schemaVersion": "1.0.0",
        "artifactType": "pyannote-timeline-challenger-audit",
        "status": "completed",
        "source": {
            "sha256": sha256_file(audio),
            "durationMs": duration_ms,
        },
        "model": {
            "repoId": manifest["repoId"],
            "revision": manifest["revision"],
            "manifestSha256": canonical_json_sha256(manifest),
        },
        "runtime": {
            "python": str(executable),
            "device": args.device,
            "networkPolicy": "forced-offline",
            "elapsedSeconds": round(elapsed, 6),
        },
        "output": {
            "regularTurnCount": len(regular),
            "exclusiveTurnCount": (
                len(exclusive) if isinstance(exclusive, list) else None
            ),
            "speakerCount": len(local_speakers),
            "localSpeakers": local_speakers,
            "resultSha256": canonical_json_sha256(response),
        },
        "claimsFinalQualityImprovement": False,
        "conclusion": "candidate-ready-requires-lattice-binding",
    }
    atomic_write_json_no_replace(
        output / "audit-report.v1.json",
        report,
    )
    print(
        f"completed: {len(local_speakers)} speakers, "
        f"{len(regular)} regular turns, {elapsed:.3f} seconds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
