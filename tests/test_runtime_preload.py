from __future__ import annotations

import threading

import pytest

from backend.errors import WorkerError
from backend.runtime_preload import (
    preload_production_runtime,
    production_runtime_modules,
)


def test_preload_imports_native_modules_in_deterministic_order() -> None:
    imported: list[str] = []

    def importer(module_name: str):
        imported.append(module_name)
        return object()

    report = preload_production_runtime(
        include_pyannote=False,
        importer=importer,
    )

    assert tuple(imported) == production_runtime_modules(
        include_pyannote=False
    )
    assert report.modules == tuple(imported)
    assert report.thread_name == threading.main_thread().name
    assert report.elapsed_ms >= 0.0


def test_preload_appends_pyannote_only_when_enabled() -> None:
    modules = production_runtime_modules(include_pyannote=True)

    assert modules[-1] == "pyannote.audio"
    assert modules[:-1] == production_runtime_modules(include_pyannote=False)


def test_preload_fails_closed_without_disclosing_exception_text() -> None:
    def importer(module_name: str):
        if module_name == "soundfile":
            raise RuntimeError(r"secret path D:\private\runtime.dll")
        return object()

    with pytest.raises(WorkerError) as captured:
        preload_production_runtime(
            include_pyannote=False,
            importer=importer,
        )

    error = captured.value
    assert error.code == "RUNTIME_PRELOAD_FAILED"
    assert error.details == {
        "module": "soundfile",
        "exceptionType": "RuntimeError",
    }
    assert "secret path" not in error.message


def test_preload_rejects_background_thread_execution() -> None:
    observed: list[WorkerError] = []

    def invoke() -> None:
        try:
            preload_production_runtime(
                include_pyannote=False,
                importer=lambda _name: object(),
            )
        except WorkerError as exc:
            observed.append(exc)

    thread = threading.Thread(target=invoke, name="background-loader")
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(observed) == 1
    assert observed[0].code == "RUNTIME_PRELOAD_THREAD_INVALID"
    assert observed[0].details == {"threadName": "background-loader"}
