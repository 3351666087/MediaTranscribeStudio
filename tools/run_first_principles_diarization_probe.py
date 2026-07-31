#!/usr/bin/env python3
"""Run blind Community-1 diarization without production backend code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys

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


def _initial_report(
    *,
    blind_path: Path,
    blind_manifest: Mapping[str, Any],
    model_path: Path,
    device: str,
    maximum: int | None,
) -> dict[str, Any]:
    model_manifest = model_path / ".mts-model-manifest.json"
    return {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-diarization-probe",
        "batchId": blind_manifest.get("batchId"),
        "blindManifest": {
            "path": str(blind_path.resolve()),
            "sha256": _sha256(blind_path),
        },
        "truthAccessed": False,
        "productionBackendImported": False,
        "probe": {
            "id": "pyannote-community-1-direct-auto-v1",
            "package": "pyannote.audio",
            "packageVersion": importlib.metadata.version("pyannote-audio"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "modelPath": str(model_path.resolve()),
            "modelManifestSha256": (
                _sha256(model_manifest) if model_manifest.is_file() else None
            ),
            "device": device,
            "speakerCountConstraints": None,
            "maximumCases": maximum,
        },
        "cases": [],
    }


def _turns(annotation: Any) -> list[dict[str, Any]]:
    if annotation is None or not callable(getattr(annotation, "itertracks", None)):
        return []
    turns: list[dict[str, Any]] = []
    for item in annotation.itertracks(yield_label=True):
        if not isinstance(item, Sequence) or len(item) != 3:
            continue
        interval, _, label = item
        start = float(getattr(interval, "start"))
        end = float(getattr(interval, "end"))
        label_text = str(label or "").strip()
        if end <= start or not label_text:
            continue
        turns.append(
            {
                "startSeconds": round(start, 6),
                "endSeconds": round(end, 6),
                "localSpeaker": label_text,
            }
        )
    return sorted(
        turns,
        key=lambda item: (
            item["startSeconds"],
            item["endSeconds"],
            item["localSpeaker"],
        ),
    )


def _annotation(result: Any, name: str) -> Any:
    value = getattr(result, name, None)
    if value is None and isinstance(result, Mapping):
        value = result.get(name)
    return value


def run_probe(
    *,
    blind_path: Path,
    model_path: Path,
    output_path: Path,
    device: str = "cpu",
    maximum: int | None = None,
) -> dict[str, Any]:
    blind_manifest, cases = load_blind_cases(blind_path)
    selected = cases[:maximum] if maximum is not None else cases
    if maximum is not None and maximum < 1:
        raise BlindProbeError("maximum must be positive")
    if not model_path.is_dir():
        raise BlindProbeError(f"diarization model directory is missing: {model_path}")

    if output_path.is_file():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            report.get("artifactType") != "first-principles-diarization-probe"
            or report.get("blindManifest", {}).get("sha256") != _sha256(blind_path)
            or report.get("truthAccessed") is not False
        ):
            raise BlindProbeError(
                "existing diarization report cannot be safely resumed"
            )
    else:
        report = _initial_report(
            blind_path=blind_path,
            blind_manifest=blind_manifest,
            model_path=model_path,
            device=device,
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

    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    import soundfile
    import torch
    from pyannote.audio import Pipeline

    load_started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(str(model_path))
    pipeline.to(torch.device(device))
    report["probe"]["modelLoadSeconds"] = round(
        time.perf_counter() - load_started, 6
    )
    _atomic_write(output_path, report)

    for case in pending:
        started = time.perf_counter()
        actual_sha = _sha256(case["mediaPath"])
        row: dict[str, Any] = {
            "auditCaseId": case["auditCaseId"],
            "mediaSha256": actual_sha,
            "durationSeconds": case["durationSeconds"],
            "status": "failed",
            "speakerCount": None,
            "regularTurns": [],
            "exclusiveTurns": [],
            "failure": None,
        }
        if actual_sha != case["mediaSha256"]:
            row["failure"] = "media-sha256-mismatch"
        else:
            try:
                samples, sample_rate = soundfile.read(
                    str(case["mediaPath"]), dtype="float32", always_2d=True
                )
                if sample_rate != 16_000 or samples.shape[1] != 1:
                    raise ValueError("audio must be mono 16 kHz")
                waveform = torch.from_numpy(samples[:, 0]).unsqueeze(0)
                result = pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate}
                )
                regular = _turns(_annotation(result, "speaker_diarization"))
                exclusive = _turns(
                    _annotation(result, "exclusive_speaker_diarization")
                )
                labels = {
                    item["localSpeaker"]
                    for item in regular
                    if isinstance(item.get("localSpeaker"), str)
                }
                row.update(
                    {
                        "status": "completed",
                        "speakerCount": len(labels),
                        "regularTurns": regular,
                        "exclusiveTurns": exclusive,
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or (
        args.blind_manifest.parent / "evidence/pyannote-community-1.v1.json"
    )
    # --python is accepted for parity with the isolated command contract; the
    # caller must invoke this script with that interpreter.
    del args.python
    report = run_probe(
        blind_path=args.blind_manifest,
        model_path=args.model,
        output_path=output,
        device=args.device,
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
