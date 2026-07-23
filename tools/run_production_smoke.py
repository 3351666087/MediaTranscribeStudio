"""Run one production worker smoke job through the JSONL protocol.

The harness deliberately keeps the worker stdin open while the asynchronous
job is running.  It waits for a job terminal event, requests an orderly worker
shutdown, and only then closes stdin.  If the worker is blocked, cleanup is
limited to the process tree rooted at the exact worker PID started here.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

try:
    import psutil
except ImportError:  # pragma: no cover - production requirements include psutil.
    psutil = None


PROTOCOL_VERSION = "1.0.0"
TARGET_JOB_EVENTS = frozenset(
    {
        "review.required",
        "job.completed",
        "job.failed",
    }
)
START_REQUEST_ID = "production-smoke-start"
SHUTDOWN_REQUEST_ID = "production-smoke-shutdown"
_EOF = object()


class SmokeHarnessError(RuntimeError):
    """Raised when the harness cannot observe a valid worker outcome."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class SmokePaths:
    event_log: Path
    stderr_log: Path
    result_json: Path


@dataclass(frozen=True)
class SmokeResult:
    status: str
    terminal_type: str | None
    job_id: str
    worker_pid: int
    exit_code: int | None
    elapsed_seconds: float
    event_count: int
    shutdown_acknowledged: bool
    forced_cleanup_pids: tuple[int, ...]
    event_log: str
    stderr_log: str
    result_json: str
    terminal_event: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["forced_cleanup_pids"] = list(self.forced_cleanup_pids)
        value["elapsed_seconds"] = round(self.elapsed_seconds, 3)
        return value


@dataclass(frozen=True)
class HarnessSettings:
    timeout_seconds: float = 7200.0
    shutdown_timeout_seconds: float = 30.0
    cleanup_timeout_seconds: float = 5.0
    stderr_tail_lines: int = 80

    def __post_init__(self) -> None:
        for name in (
            "timeout_seconds",
            "shutdown_timeout_seconds",
            "cleanup_timeout_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.stderr_tail_lines < 1:
            raise ValueError("stderr_tail_lines must be positive")


def _safe_artifact_path(output_directory: Path, suffix: str) -> Path:
    return output_directory.parent / f"{output_directory.name}{suffix}"


def default_smoke_paths(output_directory: Path) -> SmokePaths:
    """Keep harness artifacts outside the worker-owned output directory."""

    return SmokePaths(
        event_log=_safe_artifact_path(output_directory, "-events.jsonl"),
        stderr_log=_safe_artifact_path(output_directory, "-stderr.log"),
        result_json=_safe_artifact_path(output_directory, "-result.json"),
    )


def build_start_payload(
    *,
    job_id: str,
    source_path: Path,
    output_directory: Path,
    speaker_count_mode: str,
    speaker_count: int | None = None,
    speaker_roles: Sequence[str] = (),
    speaker_count_min: int | None = None,
    speaker_count_max: int | None = None,
    speaker_count_prior: int | None = None,
    render_pdf: bool = False,
    title: str = "中文说话人分离生产烟雾测试",
    language: str = "auto",
    local_llm_mode: str = "disabled",
    local_llm_model: str = "qwen3.5:4b",
    local_llm_endpoint: str = "http://127.0.0.1:11434",
    local_llm_endpoint_policy: str = "loopback-only",
    translation_targets: Sequence[str] = (),
    polish: bool = False,
    summary: bool = False,
    output_locale: str = "en",
    business_prompt_version: str = "business-v2",
) -> dict[str, Any]:
    """Build a worker payload while enforcing mode-specific cardinality fields."""

    mode = speaker_count_mode.strip().lower()
    if mode not in {"manual", "auto", "hybrid"}:
        raise ValueError("speaker_count_mode must be manual, auto, or hybrid")
    llm_mode = local_llm_mode.strip()
    if llm_mode not in {"disabled", "suggestion-only", "business", "enabled"}:
        raise ValueError(
            "local_llm_mode must be disabled, suggestion-only, business, or enabled"
        )
    model = local_llm_model.strip()
    if not model:
        raise ValueError("local_llm_model must not be blank")
    endpoint = local_llm_endpoint.strip()
    if not endpoint:
        raise ValueError("local_llm_endpoint must not be blank")
    endpoint_policy = local_llm_endpoint_policy.strip()
    if endpoint_policy != "loopback-only":
        raise ValueError("local_llm_endpoint_policy must be loopback-only")
    requested_targets = tuple(target.strip() for target in translation_targets)
    if any(not target for target in requested_targets):
        raise ValueError("translation_targets must not contain blank values")
    if len(set(requested_targets)) != len(requested_targets):
        raise ValueError("translation_targets must not contain duplicates")
    locale = output_locale.strip()
    if not locale:
        raise ValueError("output_locale must not be blank")
    prompt_version = business_prompt_version.strip()
    if not prompt_version:
        raise ValueError("business_prompt_version must not be blank")
    business_requested = bool(requested_targets) or polish or summary
    if business_requested and llm_mode == "disabled":
        raise ValueError(
            "local_llm_mode must enable business processing when variants are requested"
        )

    payload: dict[str, Any] = {
        "jobId": job_id,
        "sourcePath": str(source_path.resolve()),
        "outputDirectory": str(output_directory.resolve()),
        "speakerCountMode": mode,
        "renderPdf": bool(render_pdf),
        "title": title,
        "language": language,
        "localLlmMode": llm_mode,
        "localLlmModel": model,
        "localLlmAutoApply": False,
        "localLlmEndpoint": endpoint,
        "localLlmEndpointPolicy": endpoint_policy,
        "translationTargets": list(requested_targets),
        "polish": bool(polish),
        "summary": bool(summary),
        "outputLocale": locale,
        "businessPromptVersion": prompt_version,
    }

    if mode == "manual":
        if speaker_count is None or speaker_count < 1:
            raise ValueError("manual mode requires a positive speaker_count")
        if any(value is not None for value in (speaker_count_min, speaker_count_max)):
            raise ValueError("manual mode does not accept speaker count bounds")
        if speaker_count_prior is not None:
            raise ValueError("manual mode does not accept speaker_count_prior")
        roles = tuple(role.strip() for role in speaker_roles)
        if not roles:
            roles = tuple(f"角色{index}" for index in range(1, speaker_count + 1))
        if len(roles) != speaker_count:
            raise ValueError("speaker_roles length must equal speaker_count")
        if any(not role for role in roles):
            raise ValueError("speaker_roles must not contain blank values")
        if len(set(roles)) != len(roles):
            raise ValueError("speaker_roles must not contain duplicates")
        payload["speakerCount"] = speaker_count
        payload["speakerRoles"] = list(roles)
        return payload

    if speaker_count is not None or speaker_roles:
        raise ValueError(f"{mode} mode does not accept manual speaker fields")

    if mode == "auto":
        if any(
            value is not None
            for value in (
                speaker_count_min,
                speaker_count_max,
                speaker_count_prior,
            )
        ):
            raise ValueError("auto mode does not accept hybrid count constraints")
        return payload

    if speaker_count_min is None or speaker_count_max is None:
        raise ValueError("hybrid mode requires speaker_count_min and speaker_count_max")
    if speaker_count_min < 1 or speaker_count_max < 1:
        raise ValueError("hybrid speaker count bounds must be positive")
    if speaker_count_min > speaker_count_max:
        raise ValueError("speaker_count_min cannot exceed speaker_count_max")
    if (
        speaker_count_prior is not None
        and not speaker_count_min <= speaker_count_prior <= speaker_count_max
    ):
        raise ValueError("speaker_count_prior must be inside the hybrid bounds")
    payload["speakerCountBounds"] = {
        "min": speaker_count_min,
        "max": speaker_count_max,
    }
    if speaker_count_prior is not None:
        payload["speakerCountPrior"] = speaker_count_prior
    return payload


def command_envelope(
    command_type: str,
    payload: Mapping[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "type": command_type,
        "payload": dict(payload),
    }


def encode_jsonl(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8", errors="strict")


def _write_command(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    stream.write(encode_jsonl(value))
    stream.flush()


def _reader(
    stream: BinaryIO,
    output: queue.Queue[object],
) -> None:
    try:
        while True:
            line = stream.readline()
            if not line:
                break
            output.put(line)
    finally:
        output.put(_EOF)


def _stderr_reader(
    stream: BinaryIO,
    path: Path,
    tail: deque[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as log:
        while True:
            chunk = stream.readline()
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            tail.append(chunk.decode("utf-8", errors="replace").rstrip("\r\n"))


def _process_tree_pids(root_pid: int) -> list[int]:
    if psutil is None:
        return [root_pid]
    try:
        process = psutil.Process(root_pid)
    except psutil.Error:
        return []
    descendants = process.children(recursive=True)
    return [child.pid for child in descendants] + [process.pid]


def terminate_exact_process_tree(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> tuple[int, ...]:
    """Terminate only the process tree rooted at ``process.pid``."""

    if process.poll() is not None:
        return ()
    if psutil is None:
        pid = process.pid
        process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_seconds)
        return (pid,)

    try:
        root = psutil.Process(process.pid)
    except psutil.Error:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_seconds)
        return ()

    try:
        descendants = root.children(recursive=True)
    except psutil.Error:
        descendants = []
    targets = descendants + [root]
    targeted_pids = tuple(item.pid for item in targets)
    for item in targets:
        try:
            item.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(targets, timeout=timeout_seconds)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=timeout_seconds)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)
    return targeted_pids


class ProductionSmokeHarness:
    """Coordinate one worker process without closing stdin prematurely."""

    def __init__(
        self,
        *,
        worker_command: Sequence[str],
        cwd: Path,
        paths: SmokePaths,
        settings: HarnessSettings | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not worker_command:
            raise ValueError("worker_command must not be empty")
        self.worker_command = tuple(worker_command)
        self.cwd = cwd.resolve()
        self.paths = paths
        self.settings = settings or HarnessSettings()
        self.environment = dict(environment) if environment is not None else None

    def _popen(self) -> subprocess.Popen[bytes]:
        kwargs: dict[str, Any] = {
            "args": self.worker_command,
            "cwd": str(self.cwd),
            "env": self.environment,
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(**kwargs)

    def run(self, start_payload: Mapping[str, Any]) -> SmokeResult:
        job_id = str(start_payload.get("jobId") or "")
        if not job_id:
            raise ValueError("start_payload must contain jobId")

        for path in (
            self.paths.event_log,
            self.paths.stderr_log,
            self.paths.result_json,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        process = self._popen()
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise SmokeHarnessError(
                "worker pipes were not created",
                code="PIPE_INITIALIZATION_FAILED",
            )

        started = time.monotonic()
        output: queue.Queue[object] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=self.settings.stderr_tail_lines)
        stdout_thread = threading.Thread(
            target=_reader,
            args=(process.stdout, output),
            name=f"smoke-stdout-{process.pid}",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stderr_reader,
            args=(process.stderr, self.paths.stderr_log, stderr_tail),
            name=f"smoke-stderr-{process.pid}",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        event_count = 0
        terminal_event: dict[str, Any] | None = None
        shutdown_acknowledged = False
        forced_cleanup_pids: tuple[int, ...] = ()
        harness_error: SmokeHarnessError | None = None
        shutdown_sent_at: float | None = None
        deadline = started + self.settings.timeout_seconds

        try:
            _write_command(
                process.stdin,
                command_envelope(
                    "job.start",
                    start_payload,
                    request_id=START_REQUEST_ID,
                ),
            )
            with self.paths.event_log.open("wb") as event_log:
                while True:
                    now = time.monotonic()
                    active_deadline = (
                        shutdown_sent_at + self.settings.shutdown_timeout_seconds
                        if shutdown_sent_at is not None
                        else deadline
                    )
                    remaining = active_deadline - now
                    if remaining <= 0:
                        code = (
                            "SHUTDOWN_TIMEOUT"
                            if shutdown_sent_at is not None
                            else "JOB_TIMEOUT"
                        )
                        harness_error = SmokeHarnessError(
                            "worker did not reach the required state before timeout",
                            code=code,
                            details={
                                "workerPid": process.pid,
                                "jobId": job_id,
                                "stderrTail": list(stderr_tail),
                                "knownProcessTreePids": _process_tree_pids(process.pid),
                            },
                        )
                        break

                    try:
                        item = output.get(timeout=min(0.25, remaining))
                    except queue.Empty:
                        if process.poll() is not None and output.empty():
                            harness_error = SmokeHarnessError(
                                "worker exited before the protocol completed",
                                code="WORKER_EXITED_EARLY",
                                details={
                                    "workerPid": process.pid,
                                    "exitCode": process.returncode,
                                    "jobId": job_id,
                                    "stderrTail": list(stderr_tail),
                                },
                            )
                            break
                        continue

                    if item is _EOF:
                        if shutdown_acknowledged:
                            break
                        harness_error = SmokeHarnessError(
                            "worker stdout closed before shutdown acknowledgement",
                            code="WORKER_STDOUT_CLOSED",
                            details={
                                "workerPid": process.pid,
                                "exitCode": process.poll(),
                                "jobId": job_id,
                                "stderrTail": list(stderr_tail),
                            },
                        )
                        break

                    raw_line = bytes(item)
                    event_log.write(raw_line)
                    if not raw_line.endswith(b"\n"):
                        event_log.write(b"\n")
                    event_log.flush()
                    event_count += 1
                    try:
                        decoded = raw_line.decode("utf-8", errors="strict")
                        event = json.loads(decoded)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        harness_error = SmokeHarnessError(
                            "worker emitted invalid UTF-8 JSONL",
                            code="INVALID_WORKER_JSONL",
                            details={
                                "workerPid": process.pid,
                                "lineNumber": event_count,
                                "exceptionType": type(exc).__name__,
                            },
                        )
                        break
                    if not isinstance(event, dict):
                        harness_error = SmokeHarnessError(
                            "worker event root must be a JSON object",
                            code="INVALID_WORKER_EVENT",
                            details={
                                "workerPid": process.pid,
                                "lineNumber": event_count,
                            },
                        )
                        break

                    event_type = event.get("type")
                    request_id = event.get("requestId")
                    event_job_id = event.get("jobId")
                    if (
                        terminal_event is None
                        and event_type in TARGET_JOB_EVENTS
                        and event_job_id == job_id
                    ):
                        terminal_event = event
                        _write_command(
                            process.stdin,
                            command_envelope(
                                "worker.shutdown",
                                {},
                                request_id=SHUTDOWN_REQUEST_ID,
                            ),
                        )
                        shutdown_sent_at = time.monotonic()
                        continue

                    if (
                        shutdown_sent_at is not None
                        and request_id == SHUTDOWN_REQUEST_ID
                        and event_type == "command.completed"
                    ):
                        shutdown_acknowledged = True
                        break

                    if request_id == START_REQUEST_ID and event_type == "command.rejected":
                        terminal_event = event
                        _write_command(
                            process.stdin,
                            command_envelope(
                                "worker.shutdown",
                                {},
                                request_id=SHUTDOWN_REQUEST_ID,
                            ),
                        )
                        shutdown_sent_at = time.monotonic()
                        continue

                    if event_type == "worker.startup.failed":
                        terminal_event = event
                        harness_error = SmokeHarnessError(
                            "production worker startup failed",
                            code="WORKER_STARTUP_FAILED",
                            details={"event": event, "workerPid": process.pid},
                        )
                        break
        except (BrokenPipeError, OSError) as exc:
            harness_error = SmokeHarnessError(
                "worker protocol pipe failed",
                code="WORKER_PIPE_FAILED",
                details={
                    "workerPid": process.pid,
                    "exceptionType": type(exc).__name__,
                    "stderrTail": list(stderr_tail),
                },
            )
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

            if harness_error is None and shutdown_acknowledged:
                try:
                    process.wait(timeout=self.settings.shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    harness_error = SmokeHarnessError(
                        "worker acknowledged shutdown but did not exit",
                        code="WORKER_EXIT_TIMEOUT",
                        details={
                            "workerPid": process.pid,
                            "jobId": job_id,
                            "stderrTail": list(stderr_tail),
                        },
                    )

            if process.poll() is None:
                forced_cleanup_pids = terminate_exact_process_tree(
                    process,
                    timeout_seconds=self.settings.cleanup_timeout_seconds,
                )
            stdout_thread.join(timeout=self.settings.cleanup_timeout_seconds)
            stderr_thread.join(timeout=self.settings.cleanup_timeout_seconds)
            try:
                process.stdout.close()
            except OSError:
                pass
            try:
                process.stderr.close()
            except OSError:
                pass

        terminal_type = (
            str(terminal_event.get("type"))
            if terminal_event is not None
            else None
        )
        if harness_error is not None:
            status = "harness-failed"
            error = {
                "code": harness_error.code,
                "message": str(harness_error),
                "details": harness_error.details,
            }
        elif terminal_type == "job.failed" or terminal_type == "command.rejected":
            status = "job-failed"
            error = {
                "code": terminal_type.upper().replace(".", "_"),
                "message": "worker reported an unsuccessful job outcome",
                "details": {"event": terminal_event},
            }
        elif terminal_type in {"job.completed", "review.required"}:
            status = "observed"
            error = None
        else:
            status = "harness-failed"
            error = {
                "code": "TERMINAL_EVENT_MISSING",
                "message": "required terminal event was not observed",
                "details": {"terminalType": terminal_type},
            }

        result = SmokeResult(
            status=status,
            terminal_type=terminal_type,
            job_id=job_id,
            worker_pid=process.pid,
            exit_code=process.poll(),
            elapsed_seconds=time.monotonic() - started,
            event_count=event_count,
            shutdown_acknowledged=shutdown_acknowledged,
            forced_cleanup_pids=forced_cleanup_pids,
            event_log=str(self.paths.event_log.resolve()),
            stderr_log=str(self.paths.stderr_log.resolve()),
            result_json=str(self.paths.result_json.resolve()),
            terminal_event=terminal_event,
            error=error,
        )
        self.paths.result_json.write_text(
            json.dumps(
                result.as_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            errors="strict",
        )
        return result


def _load_roles(args: argparse.Namespace) -> tuple[str, ...]:
    if args.speaker_roles_file is not None and args.speaker_role:
        raise ValueError("--speaker-roles-file and --speaker-role are mutually exclusive")
    if args.speaker_roles_file is None:
        return tuple(args.speaker_role or ())
    value = json.loads(args.speaker_roles_file.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("--speaker-roles-file must contain a JSON string array")
    return tuple(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one MediaTranscribeStudio production worker smoke job"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--job-id",
        default=None,
        help="protocol-safe job ID; a smoke UUID is generated when omitted",
    )
    parser.add_argument(
        "--mode",
        choices=("manual", "auto", "hybrid"),
        default="auto",
    )
    parser.add_argument("--speaker-count", type=int)
    parser.add_argument(
        "--speaker-role",
        action="append",
        help="manual mode role; repeat once per speaker",
    )
    parser.add_argument(
        "--speaker-roles-file",
        type=Path,
        help="UTF-8 JSON string array; avoids shell encoding issues",
    )
    parser.add_argument("--speaker-count-min", type=int)
    parser.add_argument("--speaker-count-max", type=int)
    parser.add_argument("--speaker-count-prior", type=int)
    parser.add_argument(
        "--render-pdf",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--title", default="中文说话人分离生产烟雾测试")
    parser.add_argument(
        "--language",
        default="auto",
        help="auto or a BCP-47 source-language tag",
    )
    parser.add_argument(
        "--local-llm-mode",
        choices=("disabled", "suggestion-only", "business", "enabled"),
        default="disabled",
    )
    parser.add_argument("--local-llm-model", default="qwen3.5:4b")
    parser.add_argument(
        "--local-llm-endpoint",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--local-llm-endpoint-policy",
        choices=("loopback-only",),
        default="loopback-only",
    )
    parser.add_argument(
        "--translation-target",
        action="append",
        default=[],
        help="BCP-47 translation target; repeat for multiple derived translations",
    )
    parser.add_argument(
        "--polish",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-locale", default="en")
    parser.add_argument(
        "--business-prompt-version",
        default="business-v1",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--cleanup-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    paths = default_smoke_paths(args.output_dir)
    paths = SmokePaths(
        event_log=args.event_log or paths.event_log,
        stderr_log=args.stderr_log or paths.stderr_log,
        result_json=args.result_json or paths.result_json,
    )
    try:
        roles = _load_roles(args)
        job_id = args.job_id or f"smoke-{args.mode}-{uuid.uuid4().hex[:16]}"
        payload = build_start_payload(
            job_id=job_id,
            source_path=args.source,
            output_directory=args.output_dir,
            speaker_count_mode=args.mode,
            speaker_count=args.speaker_count,
            speaker_roles=roles,
            speaker_count_min=args.speaker_count_min,
            speaker_count_max=args.speaker_count_max,
            speaker_count_prior=args.speaker_count_prior,
            render_pdf=args.render_pdf,
            title=args.title,
            language=args.language,
            local_llm_mode=args.local_llm_mode,
            local_llm_model=args.local_llm_model,
            local_llm_endpoint=args.local_llm_endpoint,
            local_llm_endpoint_policy=args.local_llm_endpoint_policy,
            translation_targets=args.translation_target,
            polish=args.polish,
            summary=args.summary,
            output_locale=args.output_locale,
            business_prompt_version=args.business_prompt_version,
        )
        harness = ProductionSmokeHarness(
            worker_command=(
                str(args.python.resolve()),
                "-m",
                "backend.worker",
                "--config",
                str(args.config.resolve()),
            ),
            cwd=repo_root,
            paths=paths,
            settings=HarnessSettings(
                timeout_seconds=args.timeout_seconds,
                shutdown_timeout_seconds=args.shutdown_timeout_seconds,
                cleanup_timeout_seconds=args.cleanup_timeout_seconds,
            ),
        )
        result = harness.run(payload)
    except (OSError, ValueError, json.JSONDecodeError, SmokeHarnessError) as exc:
        failure = {
            "status": "harness-failed",
            "error": {
                "code": getattr(exc, "code", "INVALID_HARNESS_CONFIGURATION"),
                "message": str(exc),
                "details": getattr(exc, "details", {}),
            },
        }
        paths.result_json.parent.mkdir(parents=True, exist_ok=True)
        paths.result_json.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    if result.status == "observed":
        return 0
    if result.status == "job-failed":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
