from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tools import run_real_semantic_acceptance


ROOT = Path(__file__).resolve().parents[1]


def test_windows_peak_working_set_is_reported_as_real_rss_evidence(
    monkeypatch,
) -> None:
    class PsutilError(Exception):
        pass

    fake_psutil = SimpleNamespace(
        Error=PsutilError,
        Process=lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(
                peak_wset=12 * 1024 * 1024,
            )
        ),
    )
    monkeypatch.setattr(run_real_semantic_acceptance, "_resource", None)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert run_real_semantic_acceptance._max_rss_evidence() == {
        "available": True,
        "source": "psutil.Process.memory_info.peak_wset",
        "valueMb": 12.0,
    }
    assert run_real_semantic_acceptance._max_rss_mb() == 12.0


def test_peak_rss_is_explicitly_unavailable_without_a_peak_metric(
    monkeypatch,
) -> None:
    class PsutilError(Exception):
        pass

    fake_psutil = SimpleNamespace(
        Error=PsutilError,
        Process=lambda: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=8 * 1024 * 1024)
        ),
    )
    monkeypatch.setattr(run_real_semantic_acceptance, "_resource", None)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert run_real_semantic_acceptance._max_rss_evidence() == {
        "available": False,
        "failureCode": "PROCESS_PEAK_RSS_UNAVAILABLE",
    }
    assert run_real_semantic_acceptance._max_rss_mb() is None


def test_cli_help_runs_in_the_active_runtime() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_real_semantic_acceptance.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--transcript" in completed.stdout
    assert "--output" in completed.stdout
