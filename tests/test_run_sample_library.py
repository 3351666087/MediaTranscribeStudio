from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_sample_library


@pytest.fixture(autouse=True)
def _mock_configured_local_llm_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unit tests use intentionally absent config paths; production stays strict.
    monkeypatch.setattr(
        run_sample_library,
        "_configured_local_llm_model",
        lambda _: "qwen3.5:9b",
    )


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
                        "durationSeconds": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _review_decisions(root: Path, *, job_id: str) -> Path:
    path = root / f"{job_id}-review-decisions.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "artifactType": "production-review-decisions",
                "jobId": job_id,
                "automaticScoring": False,
                "decisions": [
                    {
                        "itemId": "item-a",
                        "action": "accept",
                        "decisionId": f"{job_id}-decision-1",
                        "reason": "Independent review of the source audio.",
                        "evidence": ["audio:full"],
                        "confidence": 0.63,
                        "audit": {
                            "actor": "codex-reviewer",
                            "source": "human",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_per_case_runner_uses_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0

    def fake_run(command: object, *, cwd: object) -> Completed:
        captured["command"] = command
        captured["cwd"] = cwd
        return Completed()

    monkeypatch.setattr(run_sample_library.subprocess, "run", fake_run)
    return_code = run_sample_library._run_case(
        case_id="case-a",
        artifact_id="case-a",
        source=tmp_path / "source.wav",
        config=tmp_path / "production.json",
        language="en-US",
        worker_output=tmp_path / "output",
        logs_root=tmp_path / "logs",
        speaker_count_mode="auto",
        expected_speaker_count=1,
        idle_timeout_seconds=600.0,
        hard_timeout_seconds=1800.0,
        render_pdf=False,
        output_recipe_path=None,
        review_decisions_path=None,
        local_llm_mode="suggestion-only",
        local_llm_model="qwen3.6:27b-q4_K_M",
        translation_targets=(),
        summary=False,
    )

    assert return_code == 0
    assert captured["command"][1:3] == ["-m", "tools.run_production_smoke"]
    command = captured["command"]
    assert command[command.index("--local-llm-model") + 1] == (
        "qwen3.6:27b-q4_K_M"
    )
    assert captured["cwd"] == run_sample_library.ROOT


def test_auto_mode_accepts_case_without_reference_speaker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_case(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        run_sample_library,
        "_configured_local_llm_model",
        lambda _: "qwen3.6:27b-q4_K_M",
    )
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
    assert captured["idle_timeout_seconds"] == 120.0
    assert captured["hard_timeout_seconds"] == 550.0
    assert captured["local_llm_model"] == "qwen3.6:27b-q4_K_M"


def test_review_decisions_are_forwarded_to_the_per_case_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    decisions = _review_decisions(tmp_path, job_id="sample-unknown")

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
            "--review-decisions",
            f"unknown={decisions}",
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["review_decisions_path"] == decisions


def test_review_decisions_are_bound_to_the_reused_batch_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    decisions = _review_decisions(tmp_path, job_id="sample-unknown")

    class FakeHarness:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, jobs: object) -> tuple[run_sample_library.SmokeResult, ...]:
            current = tuple(jobs)
            assert len(current) == 1
            job = current[0]
            captured["plan"] = job.review_decisions
            captured["local_llm_model"] = job.start_payload["localLlmModel"]
            return (
                run_sample_library.SmokeResult(
                    status="observed",
                    terminal_type="job.completed",
                    job_id=str(job.start_payload["jobId"]),
                    worker_pid=123,
                    exit_code=0,
                    elapsed_seconds=0.1,
                    event_count=1,
                    shutdown_acknowledged=True,
                    forced_cleanup_pids=(),
                    event_log=str(job.paths.event_log),
                    stderr_log=str(job.paths.stderr_log),
                    result_json=str(job.paths.result_json),
                    worker_session_id="review-session",
                    worker_reused=False,
                ),
            )

    monkeypatch.setattr(
        run_sample_library,
        "ProductionBatchSmokeHarness",
        FakeHarness,
    )
    monkeypatch.setattr(
        run_sample_library,
        "_configured_local_llm_model",
        lambda _: "qwen3.6:35b-a3b-q4_K_M",
    )
    exit_code = run_sample_library.main(
        [
            "--manifest",
            str(_manifest(tmp_path)),
            "--config",
            str(tmp_path / "production.json"),
            "--results-root",
            str(tmp_path / "results"),
            "--worker-output-root",
            str(tmp_path / "outputs"),
            "--review-decisions",
            f"unknown={decisions}",
        ]
    )

    assert exit_code == 0
    plan = captured["plan"]
    assert plan.job_id == "sample-unknown"
    assert plan.decisions[0]["confidence"] == 0.63
    assert captured["local_llm_model"] == "qwen3.6:35b-a3b-q4_K_M"


def test_review_decision_case_bindings_fail_closed() -> None:
    with pytest.raises(ValueError, match="CASE_ID=PATH"):
        run_sample_library._review_decision_paths(["missing-separator"])
    with pytest.raises(ValueError, match="duplicate"):
        run_sample_library._review_decision_paths(
            ["case-a=one.json", "case-a=two.json"]
        )


def test_review_decisions_bind_to_the_actual_run_suffixed_job_id(
    tmp_path: Path,
) -> None:
    decisions = _review_decisions(tmp_path, job_id="sample-unknown")
    worker_outputs = tmp_path / "outputs"
    (worker_outputs / "unknown").mkdir(parents=True)

    with pytest.raises(SystemExit, match="jobId does not match"):
        run_sample_library.main(
            [
                "--manifest",
                str(_manifest(tmp_path)),
                "--results-root",
                str(tmp_path / "results"),
                "--worker-output-root",
                str(worker_outputs),
                "--review-decisions",
                f"unknown={decisions}",
                "--no-reuse-worker",
            ]
        )


def test_single_command_recipe_is_forwarded_to_per_case_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    recipe = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "product-e2e-output-recipe.v1.json"
    )

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
            "--output-recipe",
            str(recipe),
            "--local-llm-mode",
            "suggestion-only",
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["output_recipe_path"] == recipe
    assert captured["render_pdf"] is False


def test_nested_audio_duration_drives_hard_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    case = value["cases"][0]
    del case["durationSeconds"]
    case["audio"] = {"durationSeconds": 60.0}
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
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["hard_timeout_seconds"] == 1800.0


def test_frozen_media_shape_resolves_path_duration_and_mixed_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "media" / "frozen.wav"
    source.parent.mkdir()
    source.write_bytes(b"fixture")
    manifest = tmp_path / "frozen-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "libraryId": "frozen-fixture",
                "cases": [
                    {
                        "id": "mixed-case",
                        "languageTags": ["zh", "en"],
                        "sourceLanguageLabel": "mixed",
                        "media": {
                            "path": "media/frozen.wav",
                            "durationSeconds": 6.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

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
            "--case",
            "mixed-case",
            "--no-reuse-worker",
        ]
    )

    assert exit_code == 0
    assert captured["source"] == source
    assert captured["language"] == "auto"
    assert captured["hard_timeout_seconds"] == 450.0


def test_single_language_tag_is_used_for_frozen_media_shape() -> None:
    assert run_sample_library._case_language(
        {"languageTags": ["ja"], "sourceLanguageLabel": "mixed"}
    ) == "ja"


def test_conflicting_duration_shapes_fail_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["cases"][0]["audio"] = {"durationSeconds": 60.0}
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(SystemExit, match="inconsistent durationSeconds"):
        run_sample_library.main(
            [
                "--manifest",
                str(manifest),
                "--results-root",
                str(tmp_path / "results"),
                "--worker-output-root",
                str(tmp_path / "outputs"),
                "--no-reuse-worker",
            ]
        )


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


def test_semantic_composition_gate_rejects_legacy_suggestions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "checkpoint.v2.json").write_text(
        json.dumps(
            {
                "semantic": {
                    "mode": "legacy-suggestions",
                    "status": "completed",
                    "autoApply": False,
                }
            }
        ),
        encoding="utf-8",
    )

    assert run_sample_library._semantic_composition_failure(
        output,
        translation_targets=["zh"],
    ) == "semantic-mode-is-not-candidate-composition"


def test_semantic_composition_gate_checks_translation_binding(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    semantic_root = output / "semantic"
    semantic_root.mkdir()
    composition = semantic_root / "composition.json"
    lattice = semantic_root / "lattice.json"
    arbitration = semantic_root / "arbitration.json"
    for path in (composition, lattice):
        path.write_text("{}", encoding="utf-8")
    arbitration.write_text(
        json.dumps({"translationTargets": ["zh"]}),
        encoding="utf-8",
    )
    (output / "checkpoint.v2.json").write_text(
        json.dumps(
            {
                "semantic": {
                    "mode": "candidate-composition",
                    "status": "completed",
                    "autoApply": True,
                    "artifactPath": str(composition),
                    "inputLatticePath": str(lattice),
                    "arbitrationPath": str(arbitration),
                }
            }
        ),
        encoding="utf-8",
    )

    assert run_sample_library._semantic_composition_failure(
        output,
        translation_targets=["zh"],
    ) is None
    assert run_sample_library._semantic_composition_failure(
        output,
        translation_targets=["en"],
    ) == "semantic-translation-targets-do-not-match"


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
            assert all(job.idle_timeout_seconds == 120.0 for job in current)
            assert all(job.hard_timeout_seconds == 300.0 for job in current)
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


def test_shared_worker_is_proactively_recycled_after_bounded_job_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = []
    for index in range(5):
        source = tmp_path / f"case-{index}.wav"
        source.write_bytes(b"fixture")
        cases.append(
            {
                "id": f"case-{index}",
                "path": source.name,
                "language": "auto",
                "expectedSpeakerCount": 1,
                "durationSeconds": 1.0,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"libraryId": "bounded-session-fixture", "cases": cases}),
        encoding="utf-8",
    )

    calls: list[tuple[str, ...]] = []

    class FakeHarness:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, jobs: object) -> tuple[run_sample_library.SmokeResult, ...]:
            current = tuple(jobs)
            calls.append(
                tuple(str(job.start_payload["jobId"]) for job in current)
            )
            session_id = f"session-{len(calls)}"
            return tuple(
                run_sample_library.SmokeResult(
                    status="observed",
                    terminal_type="job.completed",
                    job_id=str(job.start_payload["jobId"]),
                    worker_pid=100 + len(calls),
                    exit_code=0,
                    elapsed_seconds=0.1,
                    event_count=1,
                    shutdown_acknowledged=True,
                    forced_cleanup_pids=(),
                    event_log=str(job.paths.event_log),
                    stderr_log=str(job.paths.stderr_log),
                    result_json=str(job.paths.result_json),
                    worker_session_id=session_id,
                    worker_reused=True,
                )
                for job in current
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

    assert exit_code == 0
    assert calls == [
        (
            "sample-case-0",
            "sample-case-1",
            "sample-case-2",
            "sample-case-3",
        ),
        ("sample-case-4",),
    ]
    summary_text = capsys.readouterr().out
    summary = json.loads(summary_text[summary_text.rfind("\n{") + 1 :])
    assert summary["workerLifecycle"] == "bounded-shared-sessions"
    assert summary["workerSessionIds"] == ["session-1", "session-2"]
    assert summary["maxJobsPerWorkerSession"] == 4
    assert summary["plannedWorkerSessionCount"] == 2
    assert summary["recoverySessionCount"] == 0
