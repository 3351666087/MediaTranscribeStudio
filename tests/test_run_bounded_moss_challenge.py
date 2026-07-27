from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from tools.run_bounded_moss_challenge import (
    EXPECTED_RUNTIME_VERSIONS,
    ChallengeError,
    OutputLimits,
    ResourceSnapshot,
    SupervisionLimits,
    parse_footprint_mb,
    parse_system_free_percent,
    supervise_command,
    validate_worker_result,
    worker_python_entry,
)


def _limits(**overrides: float) -> SupervisionLimits:
    values = {
        "idle_timeout_seconds": 0.2,
        "hard_deadline_seconds": 1.0,
        "sample_interval_seconds": 0.02,
        "footprint_interval_seconds": 0.05,
        "max_rss_mb": 1024.0,
        "max_footprint_mb": 1024.0,
        "minimum_system_free_percent": 1,
    }
    values.update(overrides)
    return SupervisionLimits(**values)


def _snapshot(
    _pid: int,
    *,
    elapsed_seconds: float,
    include_footprint: bool,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        elapsed_seconds=elapsed_seconds,
        process_count=1,
        rss_mb=12.0,
        footprint_mb=16.0 if include_footprint else None,
        system_free_percent=80 if include_footprint else None,
    )


def _fake_worker(path: Path, body: str) -> Path:
    path.write_text(
        textwrap.dedent("""
            import json
            import time

            run_id = "test-run"

            def emit(event_type, payload):
                print(json.dumps({
                    "schemaVersion": "1.0.0",
                    "runId": run_id,
                    "type": event_type,
                    "payload": payload,
                }), flush=True)
            """) + "\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    return path


def test_output_validator_accepts_overlap_and_small_timestamp_rounding() -> None:
    payload = {
        "rawText": "[0.0][S01]hello[1.0]",
        "generatedTokens": 8,
        "runtimeVersions": dict(EXPECTED_RUNTIME_VERSIONS),
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"},
            {"start": 0.8, "end": 2.03, "speaker": "S02", "text": "world"},
        ],
    }
    result = validate_worker_result(
        payload,
        duration_seconds=2.0,
        limits=OutputLimits(
            max_new_tokens=16,
            max_segments=4,
            max_text_characters=32,
            timestamp_tolerance_seconds=0.05,
        ),
    )
    assert result["checks"]["speakerCount"] == 2
    assert result["checks"]["maximumEndOverrunSeconds"] == 0.03
    assert result["segments"][1]["endMs"] == 2030


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"generatedTokens": 17}, "token count"),
        (
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.06,
                        "speaker": "S01",
                        "text": "hello",
                    }
                ]
            },
            "time boundary",
        ),
        (
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "speaker-1",
                        "text": "hello",
                    }
                ]
            },
            "speaker syntax",
        ),
    ],
)
def test_output_validator_fails_closed(
    mutation: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "rawText": "hello",
        "generatedTokens": 8,
        "runtimeVersions": dict(EXPECTED_RUNTIME_VERSIONS),
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "S01", "text": "hello"}],
    }
    payload.update(mutation)
    with pytest.raises(ChallengeError, match=message):
        validate_worker_result(
            payload,
            duration_seconds=2.0,
            limits=OutputLimits(
                max_new_tokens=16,
                max_segments=4,
                max_text_characters=32,
                timestamp_tolerance_seconds=0.05,
            ),
        )


def test_resource_output_parsers() -> None:
    assert parse_footprint_mb("phys_footprint: 3.5 GB\n") == 3584.0
    assert parse_footprint_mb("no total") is None
    assert parse_system_free_percent("System-wide memory free percentage: 42%\n") == 42
    assert parse_system_free_percent("unknown") is None


def test_output_validator_rejects_runtime_drift() -> None:
    with pytest.raises(ChallengeError, match="runtime versions"):
        validate_worker_result(
            {
                "rawText": "hello",
                "generatedTokens": 1,
                "runtimeVersions": {
                    **EXPECTED_RUNTIME_VERSIONS,
                    "torch": "unexpected",
                },
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "S01",
                        "text": "hello",
                    }
                ],
            },
            duration_seconds=1.0,
            limits=OutputLimits(
                max_new_tokens=2,
                max_segments=2,
                max_text_characters=20,
                timestamp_tolerance_seconds=0.05,
            ),
        )


def test_worker_python_entry_preserves_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "venv" / "bin" / "python"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(Path(sys.executable))
    assert worker_python_entry(entry) == entry
    assert worker_python_entry(entry).is_symlink()


def test_heartbeat_refreshes_idle_timeout_but_not_hard_deadline(
    tmp_path: Path,
) -> None:
    worker = _fake_worker(
        tmp_path / "heartbeat.py",
        """
        emit("worker.started", {})
        for _ in range(20):
            time.sleep(0.04)
            emit("worker.progress", {"kind": "heartbeat"})
        """,
    )
    result = supervise_command(
        command=[sys.executable, str(worker)],
        environment=os.environ,
        output_directory=tmp_path / "output",
        run_id="test-run",
        limits=_limits(hard_deadline_seconds=0.3),
        sampler=_snapshot,
    )
    assert result["failureCode"] == "WORKER_HARD_DEADLINE_EXCEEDED"
    assert result["terminated"] is True
    assert result["resourcePeaks"]["peakFootprintMb"] == 16.0


def test_missing_heartbeat_triggers_idle_timeout(tmp_path: Path) -> None:
    worker = _fake_worker(
        tmp_path / "idle.py",
        """
        emit("worker.started", {})
        time.sleep(2)
        """,
    )
    result = supervise_command(
        command=[sys.executable, str(worker)],
        environment=os.environ,
        output_directory=tmp_path / "output",
        run_id="test-run",
        limits=_limits(),
        sampler=_snapshot,
    )
    assert result["failureCode"] == "WORKER_IDLE_TIMEOUT"
    assert result["terminated"] is True


def test_success_requires_matching_terminal_event(tmp_path: Path) -> None:
    worker = _fake_worker(
        tmp_path / "success.py",
        """
        emit("worker.started", {})
        emit("worker.completed", {"generatedTokens": 1})
        """,
    )
    result = supervise_command(
        command=[sys.executable, str(worker)],
        environment=os.environ,
        output_directory=tmp_path / "output",
        run_id="test-run",
        limits=_limits(),
        sampler=_snapshot,
    )
    assert result["failureCode"] is None
    assert result["exitCode"] == 0
    events = [
        json.loads(line)
        for line in Path(result["artifactPaths"]["events"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["type"] for event in events] == [
        "worker.started",
        "worker.completed",
    ]
