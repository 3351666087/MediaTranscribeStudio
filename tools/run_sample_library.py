"""Run short sample-library cases sequentially through the production worker."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / ".runtime_cache" / "sample-library" / "sample-library.resolved.v1.json"
DEFAULT_CONFIG = ROOT / "production.config.json"
DEFAULT_RESULTS = ROOT / ".runtime_cache" / "sample-library" / "results"
DEFAULT_WORKER_OUTPUTS = ROOT / ".runtime_cache" / "outputs" / "sample-library"

from tools.run_production_smoke import (
    BatchSmokeJob,
    HarnessSettings,
    ProductionBatchSmokeHarness,
    SmokePaths,
    SmokeResult,
    build_start_payload,
)


_RECOVERABLE_SESSION_FAILURES = frozenset(
    {
        "JOB_TIMEOUT",
        "WORKER_CAPACITY_RELEASE_TIMEOUT",
        "WORKER_EXITED_EARLY",
        "WORKER_STDOUT_CLOSED",
        "INVALID_WORKER_JSONL",
        "WORKER_PIPE_FAILED",
        "WORKER_EXIT_TIMEOUT",
        "PIPE_INITIALIZATION_FAILED",
    }
)


def _run_case(
    *,
    case_id: str,
    artifact_id: str,
    source: Path,
    config: Path,
    language: str,
    worker_output: Path,
    logs_root: Path,
    speaker_count_mode: str,
    expected_speaker_count: int | None,
    timeout_seconds: float,
    render_pdf: bool,
    local_llm_mode: str,
    translation_targets: Sequence[str],
    polish: bool,
    summary: bool,
) -> int:
    command = [
        sys.executable,
        str(ROOT / "tools" / "run_production_smoke.py"),
        "--config",
        str(config),
        "--source",
        str(source),
        "--output-dir",
        str(worker_output),
        "--job-id",
        f"sample-{artifact_id}",
        "--mode",
        speaker_count_mode,
        "--title",
        f"Sample {case_id}",
        "--language",
        language,
        "--local-llm-mode",
        local_llm_mode,
        "--timeout-seconds",
        str(timeout_seconds),
        "--event-log",
        str(logs_root / f"{artifact_id}-events.jsonl"),
        "--stderr-log",
        str(logs_root / f"{artifact_id}-stderr.log"),
        "--result-json",
        str(logs_root / f"{artifact_id}-result.json"),
    ]
    if speaker_count_mode == "manual":
        if expected_speaker_count is None:
            raise ValueError(
                "manual mode requires reference expectedSpeakerCount"
            )
        command.extend(["--speaker-count", str(expected_speaker_count)])
    elif speaker_count_mode == "hybrid":
        if expected_speaker_count is None:
            raise ValueError(
                "hybrid mode requires reference expectedSpeakerCount"
            )
        command.extend(
            [
                "--speaker-count-min",
                str(max(1, expected_speaker_count - 1)),
                "--speaker-count-max",
                str(expected_speaker_count + 1),
                "--speaker-count-prior",
                str(expected_speaker_count),
            ]
        )
    if render_pdf:
        command.append("--render-pdf")
    for target in translation_targets:
        command.extend(["--translation-target", target])
    if polish:
        command.append("--polish")
    if summary:
        command.append("--summary")
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


def _batch_job(
    *,
    case_id: str,
    artifact_id: str,
    source: Path,
    language: str,
    worker_output: Path,
    logs_root: Path,
    speaker_count_mode: str,
    expected_speaker_count: int | None,
    render_pdf: bool,
    local_llm_mode: str,
    translation_targets: Sequence[str],
    polish: bool,
    summary: bool,
) -> BatchSmokeJob:
    speaker_count = None
    speaker_count_min = None
    speaker_count_max = None
    speaker_count_prior = None
    if speaker_count_mode == "manual":
        if expected_speaker_count is None:
            raise ValueError("manual mode requires reference expectedSpeakerCount")
        speaker_count = expected_speaker_count
    elif speaker_count_mode == "hybrid":
        if expected_speaker_count is None:
            raise ValueError("hybrid mode requires reference expectedSpeakerCount")
        speaker_count_min = max(1, expected_speaker_count - 1)
        speaker_count_max = expected_speaker_count + 1
        speaker_count_prior = expected_speaker_count
    payload = build_start_payload(
        job_id=f"sample-{artifact_id}",
        source_path=source,
        output_directory=worker_output,
        speaker_count_mode=speaker_count_mode,
        speaker_count=speaker_count,
        speaker_count_min=speaker_count_min,
        speaker_count_max=speaker_count_max,
        speaker_count_prior=speaker_count_prior,
        render_pdf=render_pdf,
        title=f"Sample {case_id}",
        language=language,
        local_llm_mode=local_llm_mode,
        translation_targets=translation_targets,
        polish=polish,
        summary=summary,
    )
    return BatchSmokeJob(
        start_payload=payload,
        paths=SmokePaths(
            event_log=logs_root / f"{artifact_id}-events.jsonl",
            stderr_log=logs_root / f"{artifact_id}-stderr.log",
            result_json=logs_root / f"{artifact_id}-result.json",
        ),
    )


def _aborted_session_error(result: SmokeResult) -> Mapping[str, object] | None:
    error = result.error
    if not isinstance(error, Mapping) or error.get("code") != "BATCH_ABORTED":
        return None
    details = error.get("details")
    if not isinstance(details, Mapping):
        return None
    session_error = details.get("sessionError")
    return session_error if isinstance(session_error, Mapping) else None


def _run_recovering_batch(
    jobs: Sequence[BatchSmokeJob],
    *,
    harness_factory: Callable[[int], ProductionBatchSmokeHarness],
) -> tuple[tuple[SmokeResult, ...], tuple[str, ...]]:
    """Continue only jobs that a failed shared session never started."""

    original = tuple(jobs)
    if not original:
        return (), ()
    pending = original
    resolved: dict[str, SmokeResult] = {}
    session_ids: list[str] = []
    maximum_sessions = len(original)

    for session_index in range(1, maximum_sessions + 1):
        session_results = harness_factory(session_index).run(pending)
        if not session_results:
            raise RuntimeError("batch harness returned no results")
        session_id = next(
            (
                result.worker_session_id
                for result in session_results
                if result.worker_session_id
            ),
            None,
        )
        if session_id is not None and session_id not in session_ids:
            session_ids.append(session_id)

        by_job_id = {result.job_id: result for result in session_results}
        aborted_jobs: list[BatchSmokeJob] = []
        aborted_results: list[SmokeResult] = []
        active_results: list[SmokeResult] = []
        for job in pending:
            job_id = str(job.start_payload["jobId"])
            result = by_job_id.get(job_id)
            if result is None:
                raise RuntimeError(f"batch harness omitted result for {job_id}")
            if _aborted_session_error(result) is not None:
                aborted_jobs.append(job)
                aborted_results.append(result)
            else:
                resolved[job_id] = result
                active_results.append(result)

        if not aborted_jobs:
            break

        session_error = _aborted_session_error(aborted_results[0])
        error_code = (
            str(session_error.get("code"))
            if isinstance(session_error, Mapping)
            and session_error.get("code") is not None
            else ""
        )
        failed_before_protocol_progress = (
            not any(result.status == "observed" for result in active_results)
            and all(
                result.event_count == 0 and result.terminal_event is None
                for result in active_results
            )
        )
        if (
            error_code not in _RECOVERABLE_SESSION_FAILURES
            or failed_before_protocol_progress
        ):
            for result in aborted_results:
                resolved[result.job_id] = result
            break
        pending = tuple(aborted_jobs)
    else:
        for result in aborted_results:
            resolved[result.job_id] = result

    missing = [
        str(job.start_payload["jobId"])
        for job in original
        if str(job.start_payload["jobId"]) not in resolved
    ]
    if missing:
        raise RuntimeError(
            "recovering batch exhausted sessions without results for "
            + ", ".join(missing)
        )
    return (
        tuple(resolved[str(job.start_payload["jobId"])] for job in original),
        tuple(session_ids),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--worker-output-root",
        type=Path,
        default=DEFAULT_WORKER_OUTPUTS,
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--speaker-count-mode",
        choices=("auto", "manual", "hybrid"),
        default="auto",
    )
    parser.add_argument(
        "--language-mode",
        choices=("reference", "auto"),
        default="reference",
        help=(
            "use each case's reference language or require production "
            "language detection"
        ),
    )
    parser.add_argument("--render-pdf", action="store_true")
    parser.add_argument(
        "--local-llm-mode",
        choices=("disabled", "suggestion-only", "business"),
        default="disabled",
    )
    parser.add_argument("--translation-target", action="append", default=[])
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--reuse-worker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse one preloaded production worker for the selected sequential "
            "cases"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = resolved.get("cases", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("resolved manifest is missing non-empty cases")
    available = {str(row["id"]): row for row in rows if isinstance(row, dict)}
    selected = args.case or list(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"unknown sample case(s): {', '.join(unknown)}")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.worker_output_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    batch_jobs: list[BatchSmokeJob] = []
    for case_id in selected:
        row = available[case_id]
        source = args.manifest.parent / str(row["path"])
        base_artifact_id = (
            case_id
            if args.speaker_count_mode == "auto"
            else f"{case_id}-{args.speaker_count_mode}"
        )
        artifact_id = base_artifact_id
        run_number = 2
        while (
            (args.worker_output_root / artifact_id).exists()
            or (args.results_root / f"{artifact_id}-result.json").exists()
        ):
            artifact_id = f"{base_artifact_id}-run{run_number}"
            run_number += 1
        output = args.worker_output_root / artifact_id
        logs_root = args.results_root
        raw_expected = row.get("expectedSpeakerCount")
        expected_speaker_count = (
            raw_expected
            if isinstance(raw_expected, int)
            and not isinstance(raw_expected, bool)
            and raw_expected > 0
            else None
        )
        if (
            args.speaker_count_mode != "auto"
            and expected_speaker_count is None
        ):
            raise SystemExit(
                f"{case_id} has no reference expectedSpeakerCount for "
                f"{args.speaker_count_mode} mode"
            )
        print(f"== {case_id} ({source.name}) ==", flush=True)
        language = (
            "auto"
            if args.language_mode == "auto"
            else str(row.get("language") or "auto")
        )
        if args.reuse_worker:
            batch_jobs.append(
                _batch_job(
                    case_id=case_id,
                    artifact_id=artifact_id,
                    source=source,
                    language=language,
                    worker_output=output,
                    logs_root=logs_root,
                    speaker_count_mode=args.speaker_count_mode,
                    expected_speaker_count=expected_speaker_count,
                    render_pdf=args.render_pdf,
                    local_llm_mode=args.local_llm_mode,
                    translation_targets=args.translation_target,
                    polish=args.polish,
                    summary=args.summary,
                )
            )
        else:
            return_code = _run_case(
                case_id=case_id,
                artifact_id=artifact_id,
                source=source,
                config=args.config.resolve(),
                language=language,
                worker_output=output,
                logs_root=logs_root,
                speaker_count_mode=args.speaker_count_mode,
                expected_speaker_count=expected_speaker_count,
                timeout_seconds=args.timeout_seconds,
                render_pdf=args.render_pdf,
                local_llm_mode=args.local_llm_mode,
                translation_targets=args.translation_target,
                polish=args.polish,
                summary=args.summary,
            )
            if return_code != 0:
                failures += 1
    session_ids: tuple[str, ...] = ()
    if batch_jobs:
        session_token = uuid.uuid4().hex

        def harness_factory(session_index: int) -> ProductionBatchSmokeHarness:
            session_label = f"{session_token}-s{session_index}"
            return ProductionBatchSmokeHarness(
                worker_command=(
                    sys.executable,
                    "-m",
                    "backend.worker",
                    "--config",
                    str(args.config.resolve()),
                ),
                cwd=ROOT,
                session_event_log=(
                    args.results_root
                    / f"worker-session-{session_label}-events.jsonl"
                ),
                session_stderr_log=(
                    args.results_root
                    / f"worker-session-{session_label}-stderr.log"
                ),
                settings=HarnessSettings(timeout_seconds=args.timeout_seconds),
            )

        results, session_ids = _run_recovering_batch(
            batch_jobs,
            harness_factory=harness_factory,
        )
        failures += sum(result.status != "observed" for result in results)
    print(
        json.dumps(
            {
                "libraryId": resolved.get("libraryId"),
                "selected": selected,
                "failedCases": failures,
                "resultsRoot": str(args.results_root.resolve()),
                "workerLifecycle": (
                    (
                        "recovering-shared-sessions"
                        if len(session_ids) > 1
                        else "shared-session"
                    )
                    if args.reuse_worker
                    else "per-case"
                ),
                "workerSessionId": session_ids[0] if session_ids else None,
                "workerSessionIds": list(session_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
