#!/usr/bin/env python3
"""Run a blind FunASR FSMN-VAD probe without production backend code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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


def _initial_report(
    *,
    blind_path: Path,
    blind_manifest: Mapping[str, Any],
    model_path: Path,
    maximum: int | None,
) -> dict[str, Any]:
    model_manifest = model_path / ".mts-model-manifest.json"
    return {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-vad-probe",
        "batchId": blind_manifest.get("batchId"),
        "blindManifest": {
            "path": str(blind_path.resolve()),
            "sha256": _sha256(blind_path),
        },
        "truthAccessed": False,
        "productionBackendImported": False,
        "probe": {
            "id": "funasr-fsmn-vad-direct-v1",
            "package": "funasr",
            "packageVersion": importlib.metadata.version("funasr"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "modelPath": str(model_path.resolve()),
            "modelManifestSha256": (
                _sha256(model_manifest) if model_manifest.is_file() else None
            ),
            "device": "cpu",
            "maximumCases": maximum,
            "parameters": {"cache": {}, "inputFormat": "wav-path"},
        },
        "cases": [],
    }


def _normalize_intervals(raw: Any) -> list[dict[str, int]]:
    if not isinstance(raw, list):
        return []
    intervals: list[dict[str, int]] = []
    for value in raw:
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            )
            and value[0] >= 0
            and value[1] > value[0]
        ):
            intervals.append({"startMs": value[0], "endMs": value[1]})
    return intervals


def run_probe(
    *,
    blind_path: Path,
    model_path: Path,
    output_path: Path,
    maximum: int | None = None,
) -> dict[str, Any]:
    blind_manifest, cases = load_blind_cases(blind_path)
    selected = cases[:maximum] if maximum is not None else cases
    if maximum is not None and maximum < 1:
        raise BlindProbeError("maximum must be positive")
    if not model_path.is_dir():
        raise BlindProbeError(f"VAD model directory is missing: {model_path}")

    if output_path.is_file():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            report.get("artifactType") != "first-principles-vad-probe"
            or report.get("blindManifest", {}).get("sha256") != _sha256(blind_path)
            or report.get("truthAccessed") is not False
        ):
            raise BlindProbeError("existing VAD report cannot be safely resumed")
    else:
        report = _initial_report(
            blind_path=blind_path,
            blind_manifest=blind_manifest,
            model_path=model_path,
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
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    from funasr import AutoModel

    model_started = time.perf_counter()
    model = AutoModel(
        model=str(model_path),
        device="cpu",
        disable_update=True,
        disable_pbar=True,
    )
    report["probe"]["modelLoadSeconds"] = round(
        time.perf_counter() - model_started, 6
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
            "intervals": [],
            "failure": None,
        }
        if actual_sha != case["mediaSha256"]:
            row["failure"] = "media-sha256-mismatch"
        else:
            try:
                raw = model.generate(input=str(case["mediaPath"]), cache={})
                values = raw[0].get("value") if isinstance(raw, list) and raw else None
                row.update(
                    {
                        "status": "completed",
                        "intervals": _normalize_intervals(values),
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
    parser.add_argument("--maximum", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or (
        args.blind_manifest.parent / "evidence/funasr-fsmn-vad.v1.json"
    )
    report = run_probe(
        blind_path=args.blind_manifest,
        model_path=args.model,
        output_path=output,
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
