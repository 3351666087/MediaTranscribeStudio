#!/usr/bin/env python3
"""Run a blind MLX Whisper cross-check without production backend code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_first_principles_asr_probe import (
    BlindProbeError,
    _atomic_write,
    _sha256,
    load_blind_cases,
)


SCHEMA_VERSION = "1.0.0"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _initial_report(
    *,
    blind_path: Path,
    blind_manifest: Mapping[str, Any],
    model_path: Path,
    revision: str,
    maximum: int | None,
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-whisper-probe",
        "batchId": blind_manifest.get("batchId"),
        "blindManifest": {
            "path": str(blind_path.resolve()),
            "sha256": _sha256(blind_path),
        },
        "truthAccessed": False,
        "productionBackendImported": False,
        "probe": {
            "id": "mlx-whisper-large-v3-turbo-direct-full-audio-v1",
            "package": "mlx-whisper",
            "packageVersion": importlib.metadata.version("mlx-whisper"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "modelPath": str(model_path.resolve()),
            "modelRevision": revision,
            "maximumCases": maximum,
            "conditionOnPreviousText": False,
            "wordTimestamps": False,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "cases": [],
    }


def run_probe(
    *,
    blind_path: Path,
    model_path: Path,
    output_path: Path,
    revision: str,
    maximum: int | None = None,
) -> dict[str, Any]:
    blind_manifest, cases = load_blind_cases(blind_path)
    selected = cases[:maximum] if maximum is not None else cases
    if maximum is not None and maximum < 1:
        raise BlindProbeError("maximum must be positive")
    if not (model_path / "config.json").is_file():
        raise BlindProbeError(f"Whisper model config is missing: {model_path}")

    if output_path.is_file():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            report.get("artifactType") != "first-principles-whisper-probe"
            or report.get("blindManifest", {}).get("sha256") != _sha256(blind_path)
            or report.get("truthAccessed") is not False
        ):
            raise BlindProbeError("existing Whisper report cannot be safely resumed")
    else:
        report = _initial_report(
            blind_path=blind_path,
            blind_manifest=blind_manifest,
            model_path=model_path,
            revision=revision,
            maximum=maximum,
        )
        _atomic_write(output_path, report)

    completed = {
        row.get("auditCaseId")
        for row in report.get("cases", [])
        if isinstance(row, Mapping)
    }
    pending = [case for case in selected if case["auditCaseId"] not in completed]
    if not pending:
        return report

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import mlx_whisper

    for case in pending:
        started = time.perf_counter()
        actual_sha = _sha256(case["mediaPath"])
        row: dict[str, Any] = {
            "auditCaseId": case["auditCaseId"],
            "mediaSha256": actual_sha,
            "durationSeconds": case["durationSeconds"],
            "status": "failed",
            "language": None,
            "text": None,
            "segments": [],
            "failure": None,
        }
        if actual_sha != case["mediaSha256"]:
            row["failure"] = "media-sha256-mismatch"
        else:
            try:
                result = mlx_whisper.transcribe(
                    str(case["mediaPath"]),
                    path_or_hf_repo=str(model_path),
                    verbose=False,
                    word_timestamps=False,
                    condition_on_previous_text=False,
                )
                segments = result.get("segments", [])
                normalized_segments = []
                if isinstance(segments, list):
                    for segment in segments:
                        if not isinstance(segment, Mapping):
                            continue
                        normalized_segments.append(
                            {
                                "start": segment.get("start"),
                                "end": segment.get("end"),
                                "text": str(segment.get("text") or "").strip(),
                                "avgLogprob": segment.get("avg_logprob"),
                                "noSpeechProb": segment.get("no_speech_prob"),
                                "compressionRatio": segment.get("compression_ratio"),
                            }
                        )
                row.update(
                    {
                        "status": "completed",
                        "language": str(result.get("language") or "").strip() or None,
                        "text": str(result.get("text") or "").strip(),
                        "segments": normalized_segments,
                    }
                )
            except Exception as exc:  # keep later cases observable
                row["failure"] = f"{type(exc).__name__}: {exc}"
        row["wallSeconds"] = round(time.perf_counter() - started, 6)
        report["cases"].append(row)
        _atomic_write(output_path, report)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blind_manifest", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or (
        args.blind_manifest.parent / "evidence/whisper-large-v3-turbo.v1.json"
    )
    report = run_probe(
        blind_path=args.blind_manifest,
        model_path=args.model,
        output_path=output,
        revision=args.revision,
        maximum=args.maximum,
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "observedCases": len(report["cases"]),
                "truthAccessed": report["truthAccessed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
