#!/usr/bin/env python3
"""Run a blind, resumable Qwen3-ASR probe without production backend code."""

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


SCHEMA_VERSION = "1.0.0"


class BlindProbeError(ValueError):
    """Raised when a supposedly blind input is invalid or leaks reference data."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlindProbeError(f"cannot read blind manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BlindProbeError("blind manifest must be an object")
    return value


def load_blind_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(path)
    if manifest.get("artifactType") != "first-principles-blind-media-batch":
        raise BlindProbeError("input is not a first-principles blind media batch")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BlindProbeError("blind manifest cases must be a non-empty array")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, Mapping):
            raise BlindProbeError(f"cases[{index}] must be an object")
        if "caseId" in raw:
            raise BlindProbeError("blind cases must not expose source case IDs")
        audit_id = raw.get("auditCaseId")
        media = raw.get("media")
        if (
            not isinstance(audit_id, str)
            or not audit_id.startswith("case-")
            or not isinstance(media, Mapping)
        ):
            raise BlindProbeError(f"cases[{index}] lacks an opaque ID or media")
        if audit_id in seen:
            raise BlindProbeError(f"duplicate audit case ID {audit_id}")
        seen.add(audit_id)
        media_path = Path(str(media.get("path", "")))
        expected_sha = media.get("sha256")
        duration = raw.get("durationSeconds")
        if media_path.stem != audit_id or not media_path.is_file():
            raise BlindProbeError(f"{audit_id} media path is not an opaque local file")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise BlindProbeError(f"{audit_id} media SHA-256 is invalid")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise BlindProbeError(f"{audit_id} duration is invalid")
        cases.append(
            {
                "auditCaseId": audit_id,
                "mediaPath": media_path,
                "mediaSha256": expected_sha,
                "durationSeconds": float(duration),
            }
        )
    return manifest, cases


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _initial_report(
    *,
    blind_path: Path,
    blind_manifest: Mapping[str, Any],
    model_path: Path,
    device: str,
    dtype: str,
    maximum: int | None,
) -> dict[str, Any]:
    model_manifest = model_path / ".mts-model-manifest.json"
    return {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "first-principles-asr-probe",
        "batchId": blind_manifest.get("batchId"),
        "blindManifest": {
            "path": str(blind_path.resolve()),
            "sha256": _sha256(blind_path),
        },
        "truthAccessed": False,
        "productionBackendImported": False,
        "probe": {
            "id": "qwen3-asr-direct-full-audio-v1",
            "package": "qwen-asr",
            "packageVersion": importlib.metadata.version("qwen-asr"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "dtype": dtype,
            "languagePrompt": None,
            "contextPrompt": "",
            "maxNewTokens": 512,
            "maximumCases": maximum,
            "modelPath": str(model_path.resolve()),
            "modelManifestSha256": (
                _sha256(model_manifest) if model_manifest.is_file() else None
            ),
        },
        "cases": [],
    }


def run_probe(
    *,
    blind_path: Path,
    model_path: Path,
    output_path: Path,
    device: str,
    dtype_name: str,
    maximum: int | None = None,
) -> dict[str, Any]:
    blind_manifest, cases = load_blind_cases(blind_path)
    selected = cases[:maximum] if maximum is not None else cases
    if maximum is not None and maximum < 1:
        raise BlindProbeError("maximum must be positive")
    if not model_path.is_dir():
        raise BlindProbeError(f"ASR model directory is missing: {model_path}")

    if output_path.is_file():
        report = _read_json(output_path)
        if (
            report.get("artifactType") != "first-principles-asr-probe"
            or report.get("blindManifest", {}).get("sha256") != _sha256(blind_path)
            or report.get("truthAccessed") is not False
        ):
            raise BlindProbeError("existing probe report cannot be safely resumed")
    else:
        report = _initial_report(
            blind_path=blind_path,
            blind_manifest=blind_manifest,
            model_path=model_path,
            device=device,
            dtype=dtype_name,
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
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        raise BlindProbeError(f"unsupported dtype {dtype_name}")
    load_started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        dtype=dtype,
        device_map=device,
        local_files_only=True,
        low_cpu_mem_usage=True,
        max_inference_batch_size=1,
        max_new_tokens=512,
    )
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
            "language": None,
            "text": None,
            "failure": None,
        }
        if actual_sha != case["mediaSha256"]:
            row["failure"] = "media-sha256-mismatch"
        else:
            try:
                result = model.transcribe(
                    audio=str(case["mediaPath"]),
                    language=None,
                    context="",
                    return_time_stamps=False,
                )[0]
                row.update(
                    {
                        "status": "completed",
                        "language": str(result.language or "").strip() or None,
                        "text": str(result.text or "").strip(),
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
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--maximum", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or (
        args.blind_manifest.parent / "evidence/qwen3-asr-full.v1.json"
    )
    report = run_probe(
        blind_path=args.blind_manifest,
        model_path=args.model,
        output_path=output,
        device=args.device,
        dtype_name=args.dtype,
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
    sys.exit(main())
