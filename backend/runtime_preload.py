"""Main-thread preload for native production runtime dependencies.

Windows can deadlock when NumPy/OpenBLAS or other native extension DLLs are
loaded for the first time from a ``ThreadPoolExecutor`` worker while the JSONL
protocol thread is blocked in a console read.  Production jobs run in that
executor, so isolated preflight subprocesses are not enough: the actual worker
process must import native runtimes on its main thread before accepting jobs.

This module imports code only.  It never loads model weights, opens media, or
enables a network-capable model identifier.
"""

from __future__ import annotations

import importlib
import io
import threading
import time
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .errors import WorkerError

_BASE_MODULES = (
    "numpy",
    "scipy",
    "soundfile",
    "torch",
    "funasr",
    "qwen_asr",
    "modelscope",
    "modelscope.pipelines",
    "simplejson",
)
_PYANNOTE_MODULE = "pyannote.audio"


@dataclass(frozen=True)
class RuntimePreloadReport:
    """Path-free evidence that native imports completed in this process."""

    modules: tuple[str, ...]
    elapsed_ms: float
    thread_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "modules": list(self.modules),
            "elapsedMs": round(self.elapsed_ms, 3),
            "threadName": self.thread_name,
        }


def production_runtime_modules(*, include_pyannote: bool) -> tuple[str, ...]:
    """Return the deterministic native import order for production startup."""

    if include_pyannote:
        return (*_BASE_MODULES, _PYANNOTE_MODULE)
    return _BASE_MODULES


def preload_production_runtime(
    *,
    include_pyannote: bool,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> RuntimePreloadReport:
    """Import native runtimes on the process main thread before job dispatch.

    Third-party packages may print version banners during import.  The worker's
    stdout is a strict JSONL protocol, so those banners are captured while no
    background job or protocol writer exists.
    """

    current = threading.current_thread()
    if current is not threading.main_thread():
        raise WorkerError(
            "RUNTIME_PRELOAD_THREAD_INVALID",
            "production native runtimes must be preloaded on the main thread",
            details={"threadName": current.name},
        )

    modules = production_runtime_modules(include_pyannote=include_pyannote)
    started = time.perf_counter()
    for module_name in modules:
        try:
            with redirect_stdout(io.StringIO()):
                importer(module_name)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise WorkerError(
                "RUNTIME_PRELOAD_FAILED",
                "production native runtime could not be preloaded safely",
                details={
                    "module": module_name,
                    "exceptionType": type(exc).__name__,
                },
            ) from exc
    return RuntimePreloadReport(
        modules=modules,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        thread_name=current.name,
    )


__all__ = [
    "RuntimePreloadReport",
    "preload_production_runtime",
    "production_runtime_modules",
]
