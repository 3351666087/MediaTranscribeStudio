"""Run short sample-library cases sequentially through the production worker."""

from __future__ import annotations

import argparse
import json
import math
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
    ReviewDecisionPlan,
    SmokePaths,
    SmokeResult,
    build_start_payload,
    calculate_job_hard_timeout_seconds,
    load_output_recipe,
    load_review_decision_plan,
)
from backend.persistence import read_json_strict
from backend.production_config import ProductionConfig, ProductionConfigError


_RECOVERABLE_SESSION_FAILURES = frozenset(
    {
        "JOB_TIMEOUT",
        "JOB_IDLE_TIMEOUT",
        "JOB_HARD_DEADLINE_EXCEEDED",
        "WORKER_CAPACITY_RELEASE_TIMEOUT",
        "WORKER_EXITED_EARLY",
        "WORKER_STDOUT_CLOSED",
        "INVALID_WORKER_JSONL",
        "WORKER_PIPE_FAILED",
        "WORKER_EXIT_TIMEOUT",
        "PIPE_INITIALIZATION_FAILED",
    }
)


def _configured_local_llm_model(config: Path) -> str:
    """Load the digest-pinned production model from a strictly valid config."""

    return ProductionConfig.load(config).speaker.local_llm_model


def _case_duration_seconds(row: Mapping[str, object], *, case_id: str) -> float:
    """Read either supported manifest duration shape without losing budgets."""

    top_level = row.get("durationSeconds")
    raw_audio = row.get("audio")
    nested = (
        raw_audio.get("durationSeconds")
        if isinstance(raw_audio, Mapping)
        else None
    )
    raw_media = row.get("media")
    frozen_media = (
        raw_media.get("durationSeconds")
        if isinstance(raw_media, Mapping)
        else None
    )
    values = [
        (field, value)
        for field, value in (
            ("durationSeconds", top_level),
            ("audio.durationSeconds", nested),
            ("media.durationSeconds", frozen_media),
        )
        if value is not None
    ]
    if not values:
        return 0.0
    normalized: list[tuple[str, float]] = []
    for field, value in values:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f"{case_id} has invalid {field} for timeout budgeting"
            )
        normalized.append((field, float(value)))
    if (
        len(normalized) == 2
        and not math.isclose(
            normalized[0][1],
            normalized[1][1],
            rel_tol=0.0,
            abs_tol=0.001,
        )
    ):
        raise ValueError(
            f"{case_id} has inconsistent durationSeconds and "
            "audio.durationSeconds"
        )
    return normalized[0][1]


def _case_source_path(
    row: Mapping[str, object],
    *,
    manifest_parent: Path,
    case_id: str,
) -> Path:
    """Resolve both product manifests and frozen media-library manifests."""

    raw_path = row.get("path")
    if raw_path is None:
        raw_media = row.get("media")
        if isinstance(raw_media, Mapping):
            raw_path = raw_media.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{case_id} is missing a non-empty media path")
    return manifest_parent / raw_path.strip()


def _case_language(row: Mapping[str, object]) -> str:
    """Return a BCP-47 hint, falling back to auto for mixed-language rows."""

    raw_language = row.get("language")
    if isinstance(raw_language, str) and raw_language.strip():
        language = raw_language.strip()
        if language.casefold() != "mixed":
            return language
    raw_tags = row.get("languageTags")
    if isinstance(raw_tags, list):
        tags = [
            tag.strip()
            for tag in raw_tags
            if isinstance(tag, str) and tag.strip()
        ]
        if len(tags) == 1:
            return tags[0]
    return "auto"


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
    idle_timeout_seconds: float,
    hard_timeout_seconds: float,
    render_pdf: bool,
    output_recipe_path: Path | None,
    review_decisions_path: Path | None,
    local_llm_mode: str,
    local_llm_model: str,
    translation_targets: Sequence[str],
    summary: bool,
) -> int:
    command = [
        sys.executable,
        "-m",
        "tools.run_production_smoke",
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
        "--local-llm-model",
        local_llm_model,
        "--idle-timeout-seconds",
        str(idle_timeout_seconds),
        "--hard-timeout-seconds",
        str(hard_timeout_seconds),
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
    if output_recipe_path is not None:
        command.extend(["--output-recipe", str(output_recipe_path.resolve())])
    if review_decisions_path is not None:
        command.extend(
            ["--review-decisions", str(review_decisions_path.resolve())]
        )
    for target in translation_targets:
        command.extend(["--translation-target", target])
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
    idle_timeout_seconds: float,
    hard_timeout_seconds: float,
    render_pdf: bool,
    output_recipe: Mapping[str, object] | None,
    review_decisions: ReviewDecisionPlan | None,
    local_llm_mode: str,
    local_llm_model: str,
    translation_targets: Sequence[str],
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
        output_customization=output_recipe,
        title=f"Sample {case_id}",
        language=language,
        local_llm_mode=local_llm_mode,
        local_llm_model=local_llm_model,
        translation_targets=translation_targets,
        summary=summary,
    )
    return BatchSmokeJob(
        start_payload=payload,
        paths=SmokePaths(
            event_log=logs_root / f"{artifact_id}-events.jsonl",
            stderr_log=logs_root / f"{artifact_id}-stderr.log",
            result_json=logs_root / f"{artifact_id}-result.json",
        ),
        idle_timeout_seconds=idle_timeout_seconds,
        hard_timeout_seconds=hard_timeout_seconds,
        review_decisions=review_decisions,
    )


def _review_decision_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        case_id, separator, raw_path = value.partition("=")
        case_id = case_id.strip()
        raw_path = raw_path.strip()
        if not separator or not case_id or not raw_path:
            raise ValueError(
                "--review-decisions must use non-empty CASE_ID=PATH values"
            )
        if case_id in result:
            raise ValueError(
                f"duplicate --review-decisions case binding: {case_id}"
            )
        result[case_id] = Path(raw_path)
    return result


def _aborted_session_error(result: SmokeResult) -> Mapping[str, object] | None:
    error = result.error
    if not isinstance(error, Mapping) or error.get("code") != "BATCH_ABORTED":
        return None
    details = error.get("details")
    if not isinstance(details, Mapping):
        return None
    session_error = details.get("sessionError")
    return session_error if isinstance(session_error, Mapping) else None


def _semantic_composition_failure(
    output: Path,
    *,
    translation_targets: Sequence[str],
) -> str | None:
    checkpoint_path = output / "checkpoint.v2.json"
    if not checkpoint_path.is_file():
        return "missing-checkpoint"
    checkpoint = read_json_strict(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        return "invalid-checkpoint"
    semantic = checkpoint.get("semantic")
    if not isinstance(semantic, Mapping):
        return "missing-semantic-checkpoint"
    if semantic.get("mode") != "candidate-composition":
        return "semantic-mode-is-not-candidate-composition"
    if (
        semantic.get("status") != "completed"
        or semantic.get("autoApply") is not True
    ):
        return "semantic-composition-is-not-completed-and-auto-applied"
    for field in ("artifactPath", "inputLatticePath", "arbitrationPath"):
        value = semantic.get(field)
        if not isinstance(value, str) or not Path(value).is_file():
            return f"semantic-{field}-is-missing"
    arbitration = read_json_strict(Path(str(semantic["arbitrationPath"])))
    if not isinstance(arbitration, Mapping):
        return "invalid-semantic-arbitration"
    expected_targets = sorted(
        {
            target.strip().casefold()
            for target in translation_targets
            if target.strip()
        }
    )
    actual_targets = sorted(
        {
            str(target).strip().casefold()
            for target in arbitration.get("translationTargets", [])
            if isinstance(target, str) and target.strip()
        }
    )
    if actual_targets != expected_targets:
        return "semantic-translation-targets-do-not-match"
    return None


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
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help=(
            "fixed per-job hard deadline override; otherwise derive it from "
            "cold-start p95, duration, RTF p95, and safety margin"
        ),
    )
    parser.add_argument("--idle-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--cold-start-p95-seconds",
        type=float,
        default=240.0,
    )
    parser.add_argument("--rtf-p95", type=float, default=25.0)
    parser.add_argument(
        "--deadline-safety-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--minimum-hard-timeout-seconds",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--maximum-hard-timeout-seconds",
        type=float,
        default=7200.0,
    )
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
        "--output-recipe",
        type=Path,
        help="strict versioned recipe for full product artifact generation",
    )
    parser.add_argument(
        "--review-decisions",
        action="append",
        default=[],
        metavar="CASE_ID=PATH",
        help=(
            "strict manually authored review plan for one selected case; "
            "repeat for multiple cases"
        ),
    )
    parser.add_argument(
        "--local-llm-mode",
        choices=("disabled", "suggestion-only", "business"),
        default="disabled",
    )
    parser.add_argument("--translation-target", action="append", default=[])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--require-semantic-composition",
        action="store_true",
        help=(
            "fail a technically completed case unless its checkpoint proves "
            "candidate-composition completed and auto-applied"
        ),
    )
    parser.add_argument(
        "--reuse-worker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse one preloaded production worker for the selected sequential "
            "cases"
        ),
    )
    parser.add_argument(
        "--max-jobs-per-worker-session",
        type=int,
        default=4,
        help=(
            "proactively recycle a shared worker after this many jobs; increase "
            "only after the target device passes memory-pressure validation"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.render_pdf and args.output_recipe is not None:
        raise SystemExit("--render-pdf and --output-recipe are mutually exclusive")
    try:
        output_recipe = (
            load_output_recipe(args.output_recipe)
            if args.output_recipe is not None
            else None
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        local_llm_model = _configured_local_llm_model(args.config.resolve())
    except (OSError, ProductionConfigError) as exc:
        raise SystemExit(str(exc)) from exc
    resolved = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = resolved.get("cases", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("resolved manifest is missing non-empty cases")
    available = {str(row["id"]): row for row in rows if isinstance(row, dict)}
    selected = args.case or list(available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"unknown sample case(s): {', '.join(unknown)}")
    try:
        review_decision_paths = _review_decision_paths(
            args.review_decisions
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown_review_cases = sorted(set(review_decision_paths) - set(available))
    if unknown_review_cases:
        raise SystemExit(
            "review decisions reference unknown sample case(s): "
            + ", ".join(unknown_review_cases)
        )
    unselected_review_cases = sorted(
        set(review_decision_paths) - set(selected)
    )
    if unselected_review_cases:
        raise SystemExit(
            "review decisions reference unselected sample case(s): "
            + ", ".join(unselected_review_cases)
        )
    repeated_review_cases = sorted(
        case_id
        for case_id in review_decision_paths
        if selected.count(case_id) != 1
    )
    if repeated_review_cases:
        raise SystemExit(
            "review decisions require each bound case to be selected once: "
            + ", ".join(repeated_review_cases)
        )
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.max_jobs_per_worker_session <= 0:
        raise SystemExit("--max-jobs-per-worker-session must be positive")
    try:
        HarnessSettings(
            timeout_seconds=args.maximum_hard_timeout_seconds,
            idle_timeout_seconds=args.idle_timeout_seconds,
            hard_timeout_seconds=args.maximum_hard_timeout_seconds,
        )
        calculate_job_hard_timeout_seconds(
            duration_seconds=0.0,
            cold_start_p95_seconds=args.cold_start_p95_seconds,
            rtf_p95=args.rtf_p95,
            safety_margin_seconds=args.deadline_safety_seconds,
            minimum_seconds=args.minimum_hard_timeout_seconds,
            maximum_seconds=args.maximum_hard_timeout_seconds,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.worker_output_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    semantic_gate_failures: dict[str, str] = {}
    semantic_gate_candidates: set[str] = set()
    batch_jobs: list[BatchSmokeJob] = []
    selected_outputs: dict[str, Path] = {}
    for case_id in selected:
        row = available[case_id]
        try:
            source = _case_source_path(
                row,
                manifest_parent=args.manifest.parent,
                case_id=case_id,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
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
        selected_outputs[artifact_id] = output
        logs_root = args.results_root
        raw_expected = row.get("expectedSpeakerCount")
        expected_speaker_count = (
            raw_expected
            if isinstance(raw_expected, int)
            and not isinstance(raw_expected, bool)
            and raw_expected > 0
            else None
        )
        try:
            duration_seconds = _case_duration_seconds(row, case_id=case_id)
            hard_timeout_seconds = (
                args.timeout_seconds
                if args.timeout_seconds is not None
                else calculate_job_hard_timeout_seconds(
                    duration_seconds=duration_seconds,
                    cold_start_p95_seconds=args.cold_start_p95_seconds,
                    rtf_p95=args.rtf_p95,
                    safety_margin_seconds=args.deadline_safety_seconds,
                    minimum_seconds=args.minimum_hard_timeout_seconds,
                    maximum_seconds=args.maximum_hard_timeout_seconds,
                )
            )
        except ValueError as exc:
            raise SystemExit(f"{case_id}: {exc}") from exc
        if (
            args.speaker_count_mode != "auto"
            and expected_speaker_count is None
        ):
            raise SystemExit(
                f"{case_id} has no reference expectedSpeakerCount for "
                f"{args.speaker_count_mode} mode"
            )
        print(f"== {case_id} ({source.name}) ==", flush=True)
        review_decisions_path = review_decision_paths.get(case_id)
        try:
            review_decisions = (
                load_review_decision_plan(
                    review_decisions_path,
                    expected_job_id=f"sample-{artifact_id}",
                )
                if review_decisions_path is not None
                else None
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{case_id}: {exc}") from exc
        language = "auto" if args.language_mode == "auto" else _case_language(row)
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
                    idle_timeout_seconds=args.idle_timeout_seconds,
                    hard_timeout_seconds=hard_timeout_seconds,
                    render_pdf=args.render_pdf,
                    output_recipe=output_recipe,
                    review_decisions=review_decisions,
                    local_llm_mode=args.local_llm_mode,
                    local_llm_model=local_llm_model,
                    translation_targets=args.translation_target,
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
                idle_timeout_seconds=args.idle_timeout_seconds,
                hard_timeout_seconds=hard_timeout_seconds,
                render_pdf=args.render_pdf,
                output_recipe_path=args.output_recipe,
                review_decisions_path=review_decisions_path,
                local_llm_mode=args.local_llm_mode,
                local_llm_model=local_llm_model,
                translation_targets=args.translation_target,
                summary=args.summary,
            )
            if return_code != 0:
                failures += 1
            elif args.require_semantic_composition:
                semantic_gate_candidates.add(artifact_id)
                failure = _semantic_composition_failure(
                    output,
                    translation_targets=args.translation_target,
                )
                if failure is not None:
                    failures += 1
                    semantic_gate_failures[case_id] = failure
    session_ids: tuple[str, ...] = ()
    planned_session_count = 0
    if batch_jobs:
        session_token = uuid.uuid4().hex
        maximum_jobs = args.max_jobs_per_worker_session
        batches = tuple(
            tuple(batch_jobs[offset : offset + maximum_jobs])
            for offset in range(0, len(batch_jobs), maximum_jobs)
        )
        planned_session_count = len(batches)
        observed_session_ids: list[str] = []
        for batch_index, batch in enumerate(batches, start=1):

            def harness_factory(
                recovery_index: int,
                *,
                batch_index: int = batch_index,
            ) -> ProductionBatchSmokeHarness:
                session_label = (
                    f"{session_token}-b{batch_index}-r{recovery_index}"
                )
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
                    settings=HarnessSettings(
                        timeout_seconds=args.maximum_hard_timeout_seconds,
                        idle_timeout_seconds=args.idle_timeout_seconds,
                        hard_timeout_seconds=args.maximum_hard_timeout_seconds,
                    ),
                )

            results, batch_session_ids = _run_recovering_batch(
                batch,
                harness_factory=harness_factory,
            )
            observed_session_ids.extend(batch_session_ids)
            failures += sum(
                result.status != "observed" for result in results
            )
            semantic_gate_candidates.update(
                str(result.job_id).removeprefix("sample-")
                for result in results
                if result.status == "observed"
            )
        session_ids = tuple(observed_session_ids)
        if args.require_semantic_composition:
            for artifact_id in sorted(semantic_gate_candidates):
                output = selected_outputs[artifact_id]
                failure = _semantic_composition_failure(
                    output,
                    translation_targets=args.translation_target,
                )
                if failure is not None:
                    failures += 1
                    semantic_gate_failures[artifact_id] = failure
    recovery_session_count = max(
        0,
        len(session_ids) - planned_session_count,
    )
    if not args.reuse_worker:
        worker_lifecycle = "per-case"
    elif recovery_session_count:
        worker_lifecycle = (
            "recovering-bounded-shared-sessions"
            if planned_session_count > 1
            else "recovering-shared-sessions"
        )
    elif planned_session_count > 1:
        worker_lifecycle = "bounded-shared-sessions"
    else:
        worker_lifecycle = "shared-session"
    print(
        json.dumps(
            {
                "libraryId": resolved.get("libraryId"),
                "selected": selected,
                "failedCases": failures,
                "semanticCompositionGate": {
                    "required": args.require_semantic_composition,
                    "failures": semantic_gate_failures,
                },
                "resultsRoot": str(args.results_root.resolve()),
                "workerLifecycle": worker_lifecycle,
                "workerSessionId": session_ids[0] if session_ids else None,
                "workerSessionIds": list(session_ids),
                "maxJobsPerWorkerSession": (
                    args.max_jobs_per_worker_session
                    if args.reuse_worker
                    else 1
                ),
                "plannedWorkerSessionCount": planned_session_count,
                "recoverySessionCount": recovery_session_count,
                "timeoutPolicy": {
                    "mode": (
                        "fixed"
                        if args.timeout_seconds is not None
                        else "duration-rtf-p95"
                    ),
                    "idleTimeoutSeconds": args.idle_timeout_seconds,
                    "fixedHardTimeoutSeconds": args.timeout_seconds,
                    "coldStartP95Seconds": args.cold_start_p95_seconds,
                    "rtfP95": args.rtf_p95,
                    "safetyMarginSeconds": (
                        args.deadline_safety_seconds
                    ),
                    "minimumHardTimeoutSeconds": (
                        args.minimum_hard_timeout_seconds
                    ),
                    "maximumHardTimeoutSeconds": (
                        args.maximum_hard_timeout_seconds
                    ),
                    "heartbeatExtendsHardDeadline": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
