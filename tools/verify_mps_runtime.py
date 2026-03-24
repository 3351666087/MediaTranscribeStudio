from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]


def _record(
    name: str,
    fn: Callable[[], dict[str, Any]],
    results: list[dict[str, Any]],
) -> bool:
    try:
        payload = fn()
        results.append({"name": name, "ok": True, "details": payload})
        return True
    except Exception as exc:
        results.append(
            {
                "name": name,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        )
        return False


def _torch_runtime_check() -> dict[str, Any]:
    import torch

    from utils import mps_status, preferred_torch_device

    available, reason = mps_status()
    if not available:
        raise RuntimeError(reason or "MPS unavailable")

    tensor = torch.ones((2, 2), device="mps")
    roundtrip = (tensor + 1).cpu().tolist()
    return {
        "torch": str(getattr(torch, "__version__", "")),
        "preferred_device": preferred_torch_device(),
        "bridge_active": str(os.environ.get("MTS_TORCH_MPS_BRIDGE_ACTIVE", "")),
        "tensor_roundtrip": roundtrip,
    }


def _audio_extractor_check() -> dict[str, Any]:
    from audio_extractor import AudioExtractor
    from config import Config

    cfg = Config(config_path=None)
    extractor = AudioExtractor(cfg)
    if extractor._torch_accel_device != "mps":
        raise RuntimeError(f"AudioExtractor selected {extractor._torch_accel_device!r} instead of 'mps'")
    return {
        "torch_accel_device": extractor._torch_accel_device,
        "audio_ops_accelerated": bool(extractor.use_accelerated_audio_ops),
    }


def _transcriber_check() -> dict[str, Any]:
    from config import Config
    from transcriber import Transcriber
    from utils import mlx_status, mlx_whisper_status

    cfg = Config(config_path=None)
    original_init_engine = Transcriber._init_engine
    Transcriber._init_engine = lambda self: None
    try:
        transcriber = Transcriber(cfg)
    finally:
        Transcriber._init_engine = original_init_engine

    if not transcriber.has_mps:
        raise RuntimeError("Transcriber.has_mps is False")
    if transcriber.device_str != "mps":
        raise RuntimeError(f"Transcriber.device_str={transcriber.device_str!r}")
    return {
        "has_mps": bool(transcriber.has_mps),
        "device_str": transcriber.device_str,
        "has_mlx_whisper": bool(transcriber.has_mlx_whisper),
        "mlx_core_status": mlx_status(),
        "mlx_whisper_status": mlx_whisper_status(),
    }


def _mlx_detection_check() -> dict[str, Any]:
    from utils import mlx_status, mlx_whisper_status

    mlx_core = mlx_status()
    mlx_whisper = mlx_whisper_status()
    return {
        "mlx": {"available": bool(mlx_core[0]), "detail": str(mlx_core[1] or "")},
        "mlx_whisper": {
            "available": bool(mlx_whisper[0]),
            "detail": str(mlx_whisper[1] or ""),
        },
    }


def _pipeline_check() -> dict[str, Any]:
    from config import Config
    from pipeline import TranscriptionPipeline
    from transcriber import Transcriber

    cfg = Config(config_path=None)
    cfg.data.setdefault("llm", {})["enabled"] = False
    cfg.data.setdefault("translation", {})["enabled"] = False
    cfg.data.setdefault("report", {})["generate_html"] = False
    cfg.data.setdefault("report", {})["generate_pdf"] = False
    cfg.data.setdefault("video_text_overlay", {})["enabled"] = False
    cfg.data.setdefault("logging", {})["show_progress"] = False
    cfg.data.setdefault("resume", {})["enabled"] = False

    original_init_engine = Transcriber._init_engine
    Transcriber._init_engine = lambda self: None
    try:
        pipeline = TranscriptionPipeline(cfg, enable_signal_handlers=False)
    finally:
        Transcriber._init_engine = original_init_engine

    audio_device = str(getattr(pipeline.audio_extractor, "_torch_accel_device", ""))
    transcriber_device = str(getattr(pipeline.transcriber, "device_str", ""))
    if audio_device != "mps":
        raise RuntimeError(f"Pipeline AudioExtractor device={audio_device!r}")
    if transcriber_device != "mps":
        raise RuntimeError(f"Pipeline Transcriber device={transcriber_device!r}")
    return {
        "audio_extractor_device": audio_device,
        "transcriber_device": transcriber_device,
    }


def _taichi_check() -> dict[str, Any]:
    from taichi_kernels import init_taichi

    if not init_taichi("metal"):
        raise RuntimeError("init_taichi('metal') returned False")
    return {"metal_initialized": True}


def _speaker_refine_import_check() -> dict[str, Any]:
    import speaker_refine_longform  # noqa: F401

    return {"imported": True}


def main() -> int:
    payload: dict[str, Any] = {
        "platform": {
            "system": sys.platform,
            "machine": platform.machine(),
            "mac_ver": platform.mac_ver()[0],
            "python": sys.version.split()[0],
        },
        "results": [],
    }

    results: list[dict[str, Any]] = payload["results"]
    if sys.platform != "darwin":
        payload["skipped"] = "macOS-only verification script."
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    checks = [
        ("mlx_detection", _mlx_detection_check),
    ]
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        payload["skipped"] = "MPS-specific checks only run on Apple Silicon macOS."
    else:
        checks.extend(
            [
                ("torch_runtime", _torch_runtime_check),
                ("audio_extractor", _audio_extractor_check),
                ("transcriber", _transcriber_check),
                ("pipeline", _pipeline_check),
                ("taichi", _taichi_check),
                ("speaker_refine_import", _speaker_refine_import_check),
            ]
        )
    ok = True
    for name, fn in checks:
        ok = _record(name, fn, results) and ok

    payload["ok"] = ok
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
