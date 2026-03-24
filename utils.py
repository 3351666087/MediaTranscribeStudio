from __future__ import annotations

import importlib.util
import importlib.metadata
import logging
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]

from runtime_paths import APP_ROOT, INTERNAL_ROOT, find_tool_executable, resolve_app_writable_path


LOGGER = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False
_CACHE_CLEANUP_COUNTER = 0
_MPS_STATUS_CACHE: tuple[bool, str] | None = None
_MPS_STATUS_CACHE_CONTEXT: tuple[str, str, str] | None = None
_MLX_STATUS_CACHE: tuple[bool, str] | None = None
_MLX_WHISPER_STATUS_CACHE: tuple[bool, str] | None = None


def _run_mps_tensor_smoke_test(torch_module: Any) -> tuple[bool, str]:
    try:
        with torch_module.no_grad():
            tensor = torch_module.ones((1,), device="mps")
            _ = (tensor + 1).cpu()
        return True, ""
    except Exception as exc:
        return False, str(exc)[:240]


def is_macos() -> bool:
    return sys.platform == "darwin"


def _macos_version_tuple() -> tuple[int, int, int]:
    if not is_macos():
        return (0, 0, 0)

    raw = str(platform.mac_ver()[0] or "").strip()
    if not raw:
        try:
            proc = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            raw = str(proc.stdout or "").strip()
        except Exception:
            raw = ""
    if not raw:
        return (0, 0, 0)

    parts = [part for part in raw.split(".") if part.strip()]
    numbers: list[int] = []
    for part in parts[:3]:
        try:
            numbers.append(int(part))
        except Exception:
            numbers.append(0)
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])  # type: ignore[return-value]


def _mps_status_context() -> tuple[str, str, str]:
    return (
        str(os.environ.get("MTS_TORCH_MPS_BRIDGE_ACTIVE", "") or ""),
        str(os.environ.get("MTS_TORCH_MPS_BRIDGE_TARGET_EFFECTIVE", "") or ""),
        str(os.environ.get("MTS_APPLE_RUNTIME_SHIM_PATH", "") or ""),
    )


def reset_mps_status_cache() -> None:
    global _MPS_STATUS_CACHE, _MPS_STATUS_CACHE_CONTEXT
    _MPS_STATUS_CACHE = None
    _MPS_STATUS_CACHE_CONTEXT = None


def mps_status() -> tuple[bool, str]:
    global _MPS_STATUS_CACHE, _MPS_STATUS_CACHE_CONTEXT
    context = _mps_status_context()
    if _MPS_STATUS_CACHE is not None and _MPS_STATUS_CACHE_CONTEXT == context:
        return _MPS_STATUS_CACHE

    try:
        import torch
    except Exception:
        _MPS_STATUS_CACHE = (False, "PyTorch unavailable")
        _MPS_STATUS_CACHE_CONTEXT = context
        return _MPS_STATUS_CACHE

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        _MPS_STATUS_CACHE = (False, "PyTorch missing MPS backend")
        _MPS_STATUS_CACHE_CONTEXT = context
        return _MPS_STATUS_CACHE

    try:
        built = bool(backend.is_built())
    except Exception:
        built = bool(getattr(torch._C, "_has_mps", False))
    if not built:
        _MPS_STATUS_CACHE = (False, "PyTorch not built with MPS support")
        _MPS_STATUS_CACHE_CONTEXT = context
        return _MPS_STATUS_CACHE

    try:
        if bool(backend.is_available()):
            ok, reason = _run_mps_tensor_smoke_test(torch)
            if ok:
                _MPS_STATUS_CACHE = (True, "")
            else:
                _MPS_STATUS_CACHE = (
                    False,
                    f"MPS backend reported available but tensor smoke test failed: {reason}",
                )
            _MPS_STATUS_CACHE_CONTEXT = context
            return _MPS_STATUS_CACHE
    except Exception as exc:
        _MPS_STATUS_CACHE = (False, str(exc)[:240])
        _MPS_STATUS_CACHE_CONTEXT = context
        return _MPS_STATUS_CACHE

    if is_macos():
        major, minor, patch = _macos_version_tuple()
        bridge_active = os.environ.get("MTS_TORCH_MPS_BRIDGE_ACTIVE") == "1"
        bridge_target = str(
            os.environ.get("MTS_TORCH_MPS_BRIDGE_TARGET_EFFECTIVE")
            or os.environ.get("MTS_TORCH_MPS_BRIDGE_TARGET")
            or "15.0.0"
        ).strip()
        if major >= 16:
            if bridge_active:
                _MPS_STATUS_CACHE = (
                    False,
                    f"PyTorch {torch.__version__} MPS bridge is active "
                    f"({major}.{minor}.{patch} -> {bridge_target}) but the backend still reports unavailable",
                )
                _MPS_STATUS_CACHE_CONTEXT = context
                return _MPS_STATUS_CACHE
            _MPS_STATUS_CACHE = (
                False,
                f"PyTorch {torch.__version__} rejects future macOS {major}.{minor}.{patch} for MPS; "
                f"native bridge is not active (target {bridge_target})",
            )
            _MPS_STATUS_CACHE_CONTEXT = context
            return _MPS_STATUS_CACHE

    _MPS_STATUS_CACHE = (False, "MPS runtime unavailable")
    _MPS_STATUS_CACHE_CONTEXT = context
    return _MPS_STATUS_CACHE


def mps_is_available() -> bool:
    return bool(mps_status()[0])


def mps_unavailable_reason() -> str:
    available, reason = mps_status()
    if available:
        return ""
    return str(reason or "").strip()


def _safe_find_spec(module_name: str):
    try:
        return importlib.util.find_spec(str(module_name or "").strip())
    except Exception:
        return None


def _distribution_version(dist_name: str) -> str:
    try:
        return str(importlib.metadata.version(dist_name))
    except Exception:
        return ""


def _module_candidate_paths(module_name: str) -> list[Path]:
    dotted = str(module_name or "").strip().replace("-", "_")
    if not dotted:
        return []
    relative = Path(*[part for part in dotted.split(".") if part])
    candidates: list[Path] = []
    roots: list[Path] = []
    for root in (APP_ROOT, INTERNAL_ROOT):
        if root not in roots:
            roots.append(root)
    for raw in sys.path:
        try:
            root = Path(str(raw or "").strip()).resolve()
        except Exception:
            continue
        if not str(root):
            continue
        if root not in roots:
            roots.append(root)
    for root in roots:
        candidates.append(root / relative)
        candidates.append((root / relative).with_suffix(".py"))
    return candidates


def _module_install_status(
    module_name: str,
    *,
    dist_names: Iterable[str] = (),
) -> tuple[bool, str]:
    spec = _safe_find_spec(module_name)
    if spec is not None:
        origin = str(getattr(spec, "origin", "") or "")
        return True, f"spec:{origin}" if origin else "spec"

    for dist_name in dist_names:
        version = _distribution_version(dist_name)
        if version:
            return True, f"dist:{dist_name}=={version}"

    for candidate in _module_candidate_paths(module_name):
        try:
            if candidate.exists():
                return True, f"path:{candidate}"
        except Exception:
            continue
    return False, ""


def mlx_status() -> tuple[bool, str]:
    global _MLX_STATUS_CACHE
    if _MLX_STATUS_CACHE is not None:
        return _MLX_STATUS_CACHE

    if not is_macos():
        _MLX_STATUS_CACHE = (False, "macOS-only")
        return _MLX_STATUS_CACHE

    ok, detail = _module_install_status("mlx", dist_names=("mlx",))
    _MLX_STATUS_CACHE = (ok, detail or ("mlx package not found" if not ok else ""))
    return _MLX_STATUS_CACHE


def mlx_is_available() -> bool:
    return bool(mlx_status()[0])


def mlx_whisper_status() -> tuple[bool, str]:
    global _MLX_WHISPER_STATUS_CACHE
    if _MLX_WHISPER_STATUS_CACHE is not None:
        return _MLX_WHISPER_STATUS_CACHE

    if not is_macos():
        _MLX_WHISPER_STATUS_CACHE = (False, "macOS-only")
        return _MLX_WHISPER_STATUS_CACHE

    ok, detail = _module_install_status(
        "mlx_whisper",
        dist_names=("mlx-whisper", "mlx_whisper"),
    )
    _MLX_WHISPER_STATUS_CACHE = (
        ok,
        detail or ("mlx-whisper package not found" if not ok else ""),
    )
    return _MLX_WHISPER_STATUS_CACHE


def mlx_whisper_is_available() -> bool:
    return bool(mlx_whisper_status()[0])


def accelerator_backend() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if mps_is_available():
        return "mps"
    return "cpu"


def accelerator_display_name() -> str:
    backend = accelerator_backend()
    if backend == "cuda":
        return "CUDA"
    if backend == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


def preferred_torch_device(*, indexed_cuda: bool = False) -> str:
    if accelerator_backend() == "cuda":
        return "cuda:0" if indexed_cuda else "cuda"
    if mps_is_available():
        return "mps"
    return "cpu"


def windows_hidden_subprocess_kwargs() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    kwargs: dict[str, Any] = {"creationflags": 0x08000000}
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    except Exception:
        pass
    return kwargs


def setup_logging(level: str = "INFO", log_file: str = "pipeline.log") -> None:
    global _LOGGING_CONFIGURED

    level_value = getattr(logging, str(level or "INFO").upper(), logging.INFO)
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = resolve_app_writable_path(log_path, kind="log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level_value)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level_value)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level_value)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    _LOGGING_CONFIGURED = True


def format_timestamp(seconds: Any, fmt: str = "HH:MM:SS.mmm") -> str:
    try:
        total_ms = max(0, int(round(float(seconds) * 1000.0)))
    except (TypeError, ValueError):
        total_ms = 0

    total_seconds, millis = divmod(total_ms, 1000)
    minutes, sec = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)

    normalized = str(fmt or "HH:MM:SS.mmm").strip().upper()
    if normalized == "MM:SS":
        total_minutes = hours * 60 + minute
        return f"{total_minutes:02d}:{sec:02d}"
    if normalized == "HH:MM:SS":
        return f"{hours:02d}:{minute:02d}:{sec:02d}"
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"


def human_readable_duration(seconds: Any) -> str:
    try:
        total_seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        total_seconds = 0

    hours, rem = divmod(total_seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes > 0:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def human_readable_size(num_bytes: Any) -> str:
    try:
        value = float(num_bytes)
    except (TypeError, ValueError):
        value = 0.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return "0.0 B"


def is_video_file(path: Path | str) -> bool:
    from config import VIDEO_EXTENSIONS

    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def scan_media_files(root: str | Path) -> list[Path]:
    from config import ALL_MEDIA_EXTENSIONS

    base = Path(root)
    if not base.exists():
        return []

    files = [
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in ALL_MEDIA_EXTENSIONS
    ]
    files.sort(key=lambda item: str(item).lower())
    return files


def _run_ffprobe_duration(path: Path) -> float | None:
    ffprobe = find_tool_executable("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:
        return None
    try:
        return float((proc.stdout or "").strip())
    except (TypeError, ValueError):
        return None


def get_media_duration(path: Path | str) -> float:
    target = Path(path)
    duration = _run_ffprobe_duration(target)
    if duration is not None and duration >= 0:
        return duration

    try:
        import torchaudio

        info = torchaudio.info(str(target))
        if getattr(info, "sample_rate", 0) and getattr(info, "num_frames", 0):
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    try:
        import soundfile as sf

        info = sf.info(str(target))
        if getattr(info, "samplerate", 0) and getattr(info, "frames", 0):
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    return 0.0


def get_torch_dtype(name: str):
    import torch

    normalized = str(name or "").strip().lower()
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "int8": torch.int8,
    }
    return mapping.get(normalized, torch.float32)


def setup_torch_performance(config: Any) -> None:
    try:
        import torch
    except Exception:
        return

    perf_cfg = config.get("performance", {}) if isinstance(config, dict) else config.get("performance", {})
    if not isinstance(perf_cfg, dict):
        perf_cfg = {}

    try:
        cpu_threads = int(
            (
                (config.get("asr", {}) or {})
                .get("faster_whisper", {})
                .get("cpu_threads", os.cpu_count() or 4)
            )
        )
    except Exception:
        cpu_threads = os.cpu_count() or 4

    try:
        torch.set_num_threads(max(1, cpu_threads))
    except Exception:
        pass

    if is_macos():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cudnn.benchmark = bool(perf_cfg.get("use_cudnn_benchmark", True))
    except Exception:
        pass

    try:
        fraction = float(perf_cfg.get("gpu_memory_fraction", 0.98))
        if 0.1 <= fraction <= 1.0:
            torch.cuda.set_per_process_memory_fraction(fraction)
    except Exception:
        pass


def log_accelerator_status(tag: str = "") -> None:
    try:
        import torch
    except Exception:
        return

    backend = accelerator_backend()
    if backend == "cpu":
        reason = mps_unavailable_reason() if is_macos() else ""
        if reason:
            LOGGER.info("%sAccelerator unavailable; running on CPU (%s).", tag, reason)
        else:
            LOGGER.info("%sAccelerator unavailable; running on CPU.", tag)
        return

    if backend == "mps":
        LOGGER.info("%sApple Metal (MPS) available; using macOS-optimized runtime.", tag)
        return

    try:
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
        LOGGER.info(
            "%sGPU %s | total=%.1fGB alloc=%.2fGB reserved=%.2fGB",
            tag,
            props.name,
            total_gb,
            alloc_gb,
            reserved_gb,
        )
    except Exception as exc:
        LOGGER.debug("Failed to query GPU status: %s", exc)


def smart_empty_cache(*, force: bool = False, threshold_pct: float = 96.0) -> None:
    global _CACHE_CLEANUP_COUNTER

    try:
        import torch
    except Exception:
        return

    if mps_is_available():
        if force:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return

    if not torch.cuda.is_available():
        return

    _CACHE_CLEANUP_COUNTER += 1
    should_cleanup = bool(force)

    if not should_cleanup:
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            usage_pct = (float(reserved) / float(total)) * 100.0 if total else 0.0
            should_cleanup = usage_pct >= float(threshold_pct)
        except Exception:
            should_cleanup = False

    if not should_cleanup:
        return

    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


class GracefulShutdown:
    def __init__(self, enable_signal_handlers: bool = True):
        self._requested = False
        self._previous_handlers: dict[int, Any] = {}
        if enable_signal_handlers:
            self._install()

    @property
    def should_stop(self) -> bool:
        return self._requested

    def request_shutdown(self) -> None:
        self._requested = True

    def restore(self) -> None:
        for sig, handler in self._previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                continue
        self._previous_handlers.clear()

    def _install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except Exception:
                continue

    def _handle_signal(self, signum, frame) -> None:  # noqa: ARG002
        LOGGER.warning("Received signal %s, shutdown requested.", signum)
        self._requested = True


def iter_paths(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(p) for p in paths]
