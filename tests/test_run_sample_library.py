from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_sample_library


def _manifest(root: Path) -> Path:
    source = root / "sample.wav"
    source.write_bytes(b"not-used-by-the-mocked-runner")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "libraryId": "unknown-speaker-count",
                "cases": [
                    {
                        "id": "unknown",
                        "path": source.name,
                        "language": "auto",
                        "expectedSpeakerCount": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_auto_mode_accepts_case_without_reference_speaker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_case(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(run_sample_library, "_run_case", fake_run_case)
    exit_code = run_sample_library.main(
        [
            "--manifest",
            str(_manifest(tmp_path)),
            "--results-root",
            str(tmp_path / "results"),
            "--worker-output-root",
            str(tmp_path / "outputs"),
            "--case",
            "unknown",
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["expected_speaker_count"] is None


def test_auto_language_mode_does_not_pass_reference_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["cases"][0]["language"] = "fr-FR"
    manifest.write_text(json.dumps(value), encoding="utf-8")

    def fake_run_case(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(run_sample_library, "_run_case", fake_run_case)
    exit_code = run_sample_library.main(
        [
            "--manifest",
            str(manifest),
            "--results-root",
            str(tmp_path / "results"),
            "--worker-output-root",
            str(tmp_path / "outputs"),
            "--language-mode",
            "auto",
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["language"] == "auto"


@pytest.mark.parametrize("mode", ["manual", "hybrid"])
def test_reference_modes_reject_unknown_speaker_count(
    tmp_path: Path,
    mode: str,
) -> None:
    with pytest.raises(SystemExit, match="no reference expectedSpeakerCount"):
        run_sample_library.main(
            [
                "--manifest",
                str(_manifest(tmp_path)),
                "--results-root",
                str(tmp_path / "results"),
                "--worker-output-root",
                str(tmp_path / "outputs"),
                "--case",
                "unknown",
                "--speaker-count-mode",
                mode,
                "--no-reuse-worker",
            ]
        )


def test_shared_worker_recovers_unstarted_jobs_in_original_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = []
    for case_id in ("first", "second", "third"):
        source = tmp_path / f"{case_id}.wav"
        source.write_bytes(b"fixture")
        cases.append(
            {
                "id": case_id,
                "path": source.name,
                "language": "auto",
                "expectedSpeakerCount": 1,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"libraryId": "recovery-fixture", "cases": cases}),
        encoding="utf-8",
    )

    calls: list[tuple[str, ...]] = []

    def result(
        job: object,
        *,
        status: str,
        session_id: str,
        error: dict[str, object] | None = None,
        event_count: int = 1,
    ) -> run_sample_library.SmokeResult:
        paths = job.paths
        return run_sample_library.SmokeResult(
            status=status,
            terminal_type="job.completed" if status == "observed" else None,
            job_id=str(job.start_payload["jobId"]),
            worker_pid=100 + len(calls),
            exit_code=0 if status == "observed" else 7,
            elapsed_seconds=0.1,
            event_count=event_count,
            shutdown_acknowledged=status == "observed",
            forced_cleanup_pids=(),
            event_log=str(paths.event_log),
            stderr_log=str(paths.stderr_log),
            result_json=str(paths.result_json),
            error=error,
            worker_session_id=session_id,
            worker_reused=True,
        )

    class FakeHarness:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, jobs: object) -> tuple[run_sample_library.SmokeResult, ...]:
            current = tuple(jobs)
            calls.append(
                tuple(str(job.start_payload["jobId"]) for job in current)
            )
            if len(calls) == 1:
                session_error = {
                    "code": "WORKER_EXITED_EARLY",
                    "message": "synthetic crash",
                    "details": {},
                }
                return (
                    result(
                        current[0],
                        status="observed",
                        session_id="session-1",
                    ),
                    result(
                        current[1],
                        status="harness-failed",
                        session_id="session-1",
                        error=session_error,
                        event_count=2,
                    ),
                    result(
                        current[2],
                        status="harness-failed",
                        session_id="session-1",
                        error={
                            "code": "BATCH_ABORTED",
                            "message": "not started",
                            "details": {"sessionError": session_error},
                        },
                        event_count=0,
                    ),
                )
            return (
                result(
                    current[0],
                    status="observed",
                    session_id="session-2",
                ),
            )

    monkeypatch.setattr(
        run_sample_library,
        "ProductionBatchSmokeHarness",
        FakeHarness,
    )

    exit_code = run_sample_library.main(
        [
            "--manifest",
            str(manifest),
            "--config",
            str(tmp_path / "production.json"),
            "--results-root",
            str(tmp_path / "results"),
            "--worker-output-root",
            str(tmp_path / "outputs"),
        ]
    )

    assert exit_code == 1
    assert calls == [
        ("sample-first", "sample-second", "sample-third"),
        ("sample-third",),
    ]
    summary_text = capsys.readouterr().out
    summary = json.loads(summary_text[summary_text.rfind("\n{") + 1 :])
    assert summary["workerLifecycle"] == "recovering-shared-sessions"
    assert summary["workerSessionIds"] == ["session-1", "session-2"]
    assert summary["failedCases"] == 1
