"""Concurrent job state machine for the offline headless worker."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping as MappingABC
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapters import (
    AdapterContext,
    ReportRendererAdapter,
    TranscriptionAdapter,
    UnavailableRendererAdapter,
    UnavailableTranscriptionAdapter,
)
from .documents import (
    assemble_transcript_document,
    build_review_queue,
    resolve_speaker_count,
    utc_now,
    validate_segments,
)
from .errors import JobCancelled, WorkerError, invalid_request
from .models import (
    CHECKPOINT_SCHEMA_VERSION,
    JobStatus,
    PROTOCOL_VERSION,
    RenderResult,
    SpeakerCountPolicy,
    StartJobRequest,
    TranscriptionResult,
    validate_job_id,
)
from .paths import PathPolicy
from .persistence import (
    atomic_write_json,
    atomic_write_json_transaction,
    canonical_json_sha256,
    read_json_strict,
    recover_json_transaction,
    validate_strict_json,
)
from .review import (
    assert_raw_text_unchanged,
    merge_speakers,
    open_count,
    rename_speaker,
    resolve_review_item,
    split_speaker,
    validate_review_state,
)


EventSink = Callable[[dict[str, Any]], None]
_TERMINAL = frozenset(
    {
        JobStatus.REVIEW_REQUIRED,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)


@dataclass
class JobRecord:
    request: StartJobRequest
    status: JobStatus = JobStatus.QUEUED
    stage: str = "queued"
    sequence: int = 0
    checkpoint_sequence: int = 0
    cancellation: threading.Event = field(default_factory=threading.Event)
    future: Future[None] | None = None
    error: dict[str, Any] | None = None
    document_hash: str | None = None
    template_hash: str | None = None
    renderer_version: str | None = None
    quality_status: str = "pending"
    quality_report_path: str | None = None
    render_manifest_path: str | None = None
    review_queue_path: str | None = None
    review_open_count: int = 0
    pipeline_metrics_path: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
    capacity_released: bool = False
    followup_operation: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class WorkerService:
    """Own jobs, adapters, state transitions, checkpoints, and events."""

    def __init__(
        self,
        *,
        path_policy: PathPolicy,
        transcription_adapter: TranscriptionAdapter | None = None,
        renderer_adapter: ReportRendererAdapter | None = None,
        event_sink: EventSink | None = None,
        max_workers: int = 1,
        max_pending_jobs: int | None = None,
        count_confidence_threshold: float = 0.75,
        segment_confidence_threshold: float = 0.75,
        low_speaker_margin_threshold: float = 0.18,
        high_speaker_margin_threshold: float = 0.35,
        range_width_threshold: int = 0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_pending_jobs is None:
            max_pending_jobs = max_workers
        if max_pending_jobs < 0:
            raise ValueError("max_pending_jobs must not be negative")
        if high_speaker_margin_threshold <= low_speaker_margin_threshold:
            raise ValueError(
                "high speaker margin threshold must exceed low margin threshold"
            )
        self.path_policy = path_policy
        self.transcription_adapter = (
            transcription_adapter or UnavailableTranscriptionAdapter()
        )
        self.renderer_adapter = renderer_adapter or UnavailableRendererAdapter()
        self.event_sink = event_sink or (lambda _event: None)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mts-worker"
        )
        self._capacity = threading.BoundedSemaphore(max_workers + max_pending_jobs)
        self._max_workers = max_workers
        self._max_pending_jobs = max_pending_jobs
        self.count_confidence_threshold = count_confidence_threshold
        self.segment_confidence_threshold = segment_confidence_threshold
        self.low_speaker_margin_threshold = low_speaker_margin_threshold
        self.high_speaker_margin_threshold = high_speaker_margin_threshold
        self.range_width_threshold = range_width_threshold
        self._jobs: dict[str, JobRecord] = {}
        self._active_outputs: dict[Path, str] = {}
        self._lock = threading.RLock()
        self._closed = False

    def parse_start_payload(self, payload: Any) -> StartJobRequest:
        if not isinstance(payload, Mapping):
            raise invalid_request("job.start payload must be an object")
        allowed_fields = {
            "jobId",
            "sourcePath",
            "outputDirectory",
            "speakerCountMode",
            "speakerCount",
            "speakerRoles",
            "speakerCountBounds",
            "speakerCountPrior",
            "renderPdf",
            "title",
            "language",
            "localLlmMode",
            "localLlmModel",
            "localLlmAutoApply",
        }
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise invalid_request(
                "job.start payload contains unsupported fields",
                fields=unknown_fields,
            )
        job_id = validate_job_id(payload.get("jobId"))
        source = self.path_policy.resolve_source(payload.get("sourcePath"))
        output = self.path_policy.resolve_output(payload.get("outputDirectory"))
        policy = SpeakerCountPolicy.from_payload(payload)
        render_pdf = payload.get("renderPdf", False)
        if not isinstance(render_pdf, bool):
            raise invalid_request("renderPdf must be a boolean")
        title_raw = payload.get("title")
        title = None
        if title_raw is not None:
            if not isinstance(title_raw, str) or not title_raw.strip():
                raise invalid_request("title must be a non-empty string")
            title = title_raw.strip()
            if len(title) > 240:
                raise invalid_request("title exceeds 240 characters")
        language = str(payload.get("language") or "zh-CN").strip()
        if language not in {"zh", "zh-CN", "zh-Hans"}:
            raise invalid_request("language must be zh, zh-CN, or zh-Hans")
        local_llm_mode = str(
            payload.get("localLlmMode") or "disabled"
        ).strip()
        if local_llm_mode not in {"disabled", "suggestion-only"}:
            raise invalid_request(
                "localLlmMode must be disabled or suggestion-only"
            )
        local_llm_model = str(
            payload.get("localLlmModel") or "qwen3.5:4b"
        ).strip()
        if local_llm_model != "qwen3.5:4b":
            raise invalid_request("localLlmModel must be qwen3.5:4b")
        local_llm_auto_apply = payload.get("localLlmAutoApply", False)
        if not isinstance(local_llm_auto_apply, bool):
            raise invalid_request("localLlmAutoApply must be a boolean")
        if local_llm_auto_apply:
            raise invalid_request("localLlmAutoApply is permanently forbidden")
        return StartJobRequest(
            job_id=job_id,
            source_path=source,
            output_directory=output,
            speaker_policy=policy,
            render_pdf=render_pdf,
            title=title,
            language=language,
            local_llm_mode=local_llm_mode,
            local_llm_model=local_llm_model,
        )

    def register(self, request: StartJobRequest) -> JobRecord:
        if not self._capacity.acquire(blocking=False):
            raise WorkerError(
                "WORKER_BACKPRESSURE",
                "worker capacity is full; retry after an active job completes",
                retryable=True,
            )
        record = JobRecord(request=request)
        try:
            with self._lock:
                if self._closed:
                    raise WorkerError("WORKER_CLOSED", "worker is shutting down")
                if request.job_id in self._jobs:
                    raise WorkerError(
                        "DUPLICATE_JOB_ID",
                        "jobId has already been registered",
                        details={"jobId": request.job_id},
                    )
                owner = self._active_outputs.get(request.output_directory)
                if owner is not None:
                    raise WorkerError(
                        "OUTPUT_DIRECTORY_IN_USE",
                        "another active job owns this output directory",
                        details={"jobId": owner},
                    )
                claimed = self.path_policy.create_output_directory(
                    request.output_directory
                )
                if claimed != request.output_directory:
                    raise WorkerError(
                        "OUTPUT_PATH_CHANGED",
                        "outputDirectory changed during exclusive claim",
                    )
                self._jobs[request.job_id] = record
                self._active_outputs[request.output_directory] = request.job_id
            try:
                self._write_checkpoint(record)
            except Exception as exc:
                with self._lock:
                    if self._jobs.get(request.job_id) is record:
                        self._jobs.pop(request.job_id, None)
                    if (
                        self._active_outputs.get(request.output_directory)
                        == request.job_id
                    ):
                        self._active_outputs.pop(request.output_directory, None)
                try:
                    request.output_directory.rmdir()
                except OSError:
                    pass
                if isinstance(exc, WorkerError):
                    raise
                raise WorkerError(
                    "CHECKPOINT_WRITE_FAILED",
                    "initial checkpoint could not be persisted",
                    details={"exceptionType": type(exc).__name__},
                ) from exc
        except Exception:
            self._release_capacity(record)
            raise
        return record

    def launch(self, job_id: str) -> None:
        record = self._get_job(job_id)
        with record.lock:
            if record.future is not None:
                raise WorkerError("JOB_ALREADY_LAUNCHED", "job has already been launched")
            record.future = self.executor.submit(self._run_job, record)

    def start(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self.parse_start_payload(payload)
        record = self.register(request)
        self.launch(request.job_id)
        return self.snapshot(record)

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self._get_job(job_id)
        record.cancellation.set()
        with record.lock:
            if record.status in _TERMINAL:
                return self.snapshot(record)
            future = record.future
            if record.status is JobStatus.QUEUED and future is not None and future.cancel():
                self._finish_cancelled(record)
        return self.snapshot(record)

    def status(self, job_id: str) -> dict[str, Any]:
        return self.snapshot(self._get_job(job_id))

    def review_queue(self, job_id: str) -> dict[str, Any]:
        record = self._get_job(job_id)
        with record.lock:
            document, queue = self._load_review_state(record)
            self._sync_record_from_review_state(record, document, queue)
            self._write_checkpoint(record)
            return {
                "jobId": record.request.job_id,
                "status": record.status.value,
                "openCount": record.review_open_count,
                "documentHash": record.document_hash,
                "speakerCount": document["speakerPolicy"]["resolvedCount"],
                "queue": queue,
            }

    def submit_review(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = self._review_command_payload(
            job_id,
            payload,
            required={
                "jobId",
                "itemId",
                "action",
                "decisionId",
                "reason",
                "evidence",
                "confidence",
                "audit",
            },
            optional={
                "targetSpeakerId",
                "normalizedText",
                "displayText",
                "rawText",
            },
        )
        action = normalized.get("action")
        if action == "accept":
            persisted_action = "accepted"
        elif action == "reject":
            persisted_action = "rejected"
        else:
            raise invalid_request("review.submit action must be accept or reject")
        return self._mutate_review_state(
            self._get_job(job_id),
            command="review.submit",
            mutator=lambda document, queue: resolve_review_item(
                document,
                queue,
                normalized,
                command="review.submit",
                action=persisted_action,
            ),
        )

    def rename_job_speaker(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = self._review_command_payload(
            job_id,
            payload,
            required={
                "jobId",
                "speakerId",
                "name",
                "decisionId",
                "reason",
                "evidence",
                "confidence",
                "audit",
            },
        )
        return self._mutate_review_state(
            self._get_job(job_id),
            command="speaker.rename",
            mutator=lambda document, queue: rename_speaker(
                document, queue, normalized
            ),
        )

    def merge_job_speakers(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = self._review_command_payload(
            job_id,
            payload,
            required={
                "jobId",
                "sourceSpeakerId",
                "targetSpeakerId",
                "decisionId",
                "reason",
                "evidence",
                "confidence",
                "audit",
            },
        )
        return self._mutate_review_state(
            self._get_job(job_id),
            command="speaker.merge",
            mutator=lambda document, queue: merge_speakers(
                document, queue, normalized
            ),
        )

    def split_job_speaker(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = self._review_command_payload(
            job_id,
            payload,
            required={
                "jobId",
                "sourceSpeakerId",
                "segmentIds",
                "decisionId",
                "reason",
                "evidence",
                "confidence",
                "audit",
            },
            optional={"newSpeakerName"},
        )
        return self._mutate_review_state(
            self._get_job(job_id),
            command="speaker.split",
            mutator=lambda document, queue: split_speaker(
                document, queue, normalized
            ),
        )

    def accept_suggestion(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = self._suggestion_payload(job_id, payload)
        record = self._get_job(job_id)

        def accept(
            document: dict[str, Any],
            queue: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            suggestion = self._persisted_suggestion(
                queue,
                item_id=normalized["itemId"],
                suggestion_id=normalized["suggestionId"],
            )
            return resolve_review_item(
                document,
                queue,
                normalized,
                command="suggestion.accept",
                action="accepted",
                suggestion_defaults=suggestion,
            )

        return self._mutate_review_state(
            record,
            command="suggestion.accept",
            mutator=accept,
        )

    def reject_suggestion(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = self._suggestion_payload(job_id, payload)
        record = self._get_job(job_id)

        def reject(
            document: dict[str, Any],
            queue: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            self._persisted_suggestion(
                queue,
                item_id=normalized["itemId"],
                suggestion_id=normalized["suggestionId"],
            )
            return resolve_review_item(
                document,
                queue,
                normalized,
                command="suggestion.reject",
                action="rejected",
            )

        return self._mutate_review_state(
            record,
            command="suggestion.reject",
            mutator=reject,
        )

    def resume(self, job_id: str) -> dict[str, Any]:
        record = self._get_job(job_id)
        with record.lock:
            document, queue = self._load_review_state(record)
            if open_count(queue) != 0:
                raise WorkerError(
                    "REVIEW_INCOMPLETE",
                    "job.resume requires every review item to have a human decision",
                    details={"openCount": open_count(queue)},
                )
            self._sync_record_from_review_state(record, document, queue)
        self._launch_followup(record, operation="resume")
        return self.snapshot(record)

    def rerender(self, job_id: str) -> dict[str, Any]:
        record = self._get_job(job_id)
        with record.lock:
            document, queue = self._load_review_state(record)
            if open_count(queue) != 0:
                raise WorkerError(
                    "REVIEW_INCOMPLETE",
                    "job.rerender refuses to render unresolved review items",
                    details={"openCount": open_count(queue)},
                )
            self._sync_record_from_review_state(record, document, queue)
        self._launch_followup(record, operation="rerender")
        return self.snapshot(record)

    def health(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._jobs.values())
            active_outputs = len(self._active_outputs)
            closed = self._closed
        statuses: dict[str, int] = {}
        for record in records:
            with record.lock:
                key = record.status.value
            statuses[key] = statuses.get(key, 0) + 1
        return {
            "status": "shutting-down" if closed else "ok",
            "workerVersion": "2.0.0",
            "maxWorkers": self._max_workers,
            "maxPendingJobs": self._max_pending_jobs,
            "registeredJobs": len(records),
            "activeOutputClaims": active_outputs,
            "jobsByStatus": statuses,
        }

    def wait(self, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
        record = self._get_job(job_id)
        future = record.future
        if future is not None:
            try:
                future.result(timeout=timeout)
            except (CancelledError, JobCancelled):
                pass
        return self.snapshot(record)

    def snapshot(self, record: JobRecord) -> dict[str, Any]:
        with record.lock:
            value: dict[str, Any] = {
                "jobId": record.request.job_id,
                "status": record.status.value,
                "stage": record.stage,
                "sourcePath": str(record.request.source_path),
                "outputDirectory": str(record.request.output_directory),
                "speakerCountPolicy": record.request.speaker_policy.as_dict(),
                "renderPdf": record.request.render_pdf,
                "cancellationRequested": record.cancellation.is_set(),
                "reviewOpenCount": record.review_open_count,
                "qualityStatus": record.quality_status,
                "documentHash": record.document_hash,
                "reviewQueuePath": record.review_queue_path,
                "qualityReportPath": record.quality_report_path,
                "renderManifestPath": record.render_manifest_path,
                "pipelineMetricsPath": record.pipeline_metrics_path,
                "artifactPaths": list(record.artifact_paths),
                "followupOperation": record.followup_operation,
            }
            if record.error:
                value["error"] = dict(record.error)
            return value

    def shutdown(self, *, cancel: bool = True, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            records = list(self._jobs.values())
        if cancel:
            for record in records:
                if record.status not in _TERMINAL:
                    record.cancellation.set()
                    if record.future is not None:
                        record.future.cancel()
        self.executor.shutdown(wait=wait, cancel_futures=cancel)
        for record in records:
            if record.status is JobStatus.QUEUED and record.cancellation.is_set():
                self._finish_cancelled(record)

    def _get_job(self, job_id: str) -> JobRecord:
        normalized = validate_job_id(job_id)
        with self._lock:
            record = self._jobs.get(normalized)
        if record is None:
            raise WorkerError(
                "JOB_NOT_FOUND",
                "jobId is not known to this worker process",
                details={"jobId": normalized},
            )
        return record

    @staticmethod
    def _review_command_payload(
        job_id: str,
        payload: Mapping[str, Any],
        *,
        required: set[str],
        optional: set[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, MappingABC):
            raise invalid_request("review command payload must be an object")
        optional_fields = optional or set()
        actual = set(payload)
        missing = sorted(required - actual)
        unknown = sorted(actual - required - optional_fields)
        if missing or unknown:
            raise invalid_request(
                "review command payload fields are invalid",
                missingFields=missing,
                unsupportedFields=unknown,
            )
        normalized_job_id = validate_job_id(payload.get("jobId"))
        expected_job_id = validate_job_id(job_id)
        if normalized_job_id != expected_job_id:
            raise invalid_request("payload jobId does not match the requested job")
        return dict(payload)

    def _suggestion_payload(
        self,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._review_command_payload(
            job_id,
            payload,
            required={
                "jobId",
                "itemId",
                "suggestionId",
                "decisionId",
                "reason",
                "evidence",
                "confidence",
                "audit",
            },
            optional={
                "targetSpeakerId",
                "normalizedText",
                "displayText",
                "rawText",
            },
        )

    @staticmethod
    def _persisted_suggestion(
        queue: Mapping[str, Any],
        *,
        item_id: Any,
        suggestion_id: Any,
    ) -> dict[str, Any]:
        if not isinstance(item_id, str) or not item_id.strip():
            raise invalid_request("itemId must be a non-empty string")
        if not isinstance(suggestion_id, str) or not suggestion_id.strip():
            raise invalid_request("suggestionId must be a non-empty string")
        item = next(
            (
                candidate
                for candidate in queue.get("items", [])
                if isinstance(candidate, MappingABC)
                and candidate.get("id") == item_id.strip()
            ),
            None,
        )
        if item is None:
            raise WorkerError(
                "REVIEW_ITEM_NOT_FOUND",
                "review item does not exist",
                details={"itemId": item_id.strip()},
            )
        raw_suggestions: list[Mapping[str, Any]] = []
        suggestions = item.get("suggestions")
        if isinstance(suggestions, list):
            raw_suggestions.extend(
                candidate
                for candidate in suggestions
                if isinstance(candidate, MappingABC)
            )
        singular = item.get("suggestion")
        if isinstance(singular, MappingABC):
            raw_suggestions.append(singular)
        suggestion = next(
            (
                candidate
                for candidate in raw_suggestions
                if candidate.get("id") == suggestion_id.strip()
                or candidate.get("suggestionId") == suggestion_id.strip()
            ),
            None,
        )
        if suggestion is None:
            raise WorkerError(
                "SUGGESTION_NOT_FOUND",
                "suggestion must already exist in the persisted review item",
                details={
                    "itemId": item_id.strip(),
                    "suggestionId": suggestion_id.strip(),
                },
            )
        proposal = suggestion.get("proposal")
        source = proposal if isinstance(proposal, MappingABC) else suggestion
        allowed = {
            "targetSpeakerId",
            "speakerId",
            "normalizedText",
            "displayText",
            "rawText",
        }
        defaults = {
            key: value
            for key, value in source.items()
            if key in allowed
        }
        if not defaults:
            raise WorkerError(
                "SUGGESTION_INVALID",
                "persisted suggestion does not contain an applicable proposal",
            )
        return defaults

    @staticmethod
    def _review_paths(record: JobRecord) -> tuple[Path, Path, Path]:
        output = record.request.output_directory
        return (
            output / "transcript-document.v2.json",
            output / "review" / "review-queue.json",
            output / ".review-transaction.v1.json",
        )

    def _load_review_state(
        self,
        record: JobRecord,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        document_path, queue_path, journal_path = self._review_paths(record)
        recover_json_transaction(journal_path)
        document = read_json_strict(document_path)
        queue = read_json_strict(queue_path)
        return validate_review_state(
            document,
            queue,
            expected_job_id=record.request.job_id,
            high_margin_threshold=self.high_speaker_margin_threshold,
        )

    def _sync_record_from_review_state(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
        queue: Mapping[str, Any],
    ) -> None:
        document_path, queue_path, _journal_path = self._review_paths(record)
        record.document_hash = canonical_json_sha256(document)
        record.review_queue_path = str(queue_path)
        record.review_open_count = open_count(queue)
        for path in (document_path, queue_path):
            text = str(path)
            if text not in record.artifact_paths:
                record.artifact_paths.append(text)
        if record.review_open_count:
            record.quality_status = "review-required"
        elif record.status is JobStatus.REVIEW_REQUIRED:
            record.quality_status = "review-complete"

    def _mutate_review_state(
        self,
        record: JobRecord,
        *,
        command: str,
        mutator: Callable[
            [dict[str, Any], dict[str, Any]],
            tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
        ],
    ) -> dict[str, Any]:
        with record.lock:
            if record.status is not JobStatus.REVIEW_REQUIRED:
                raise WorkerError(
                    "REVIEW_NOT_ACTIVE",
                    "human review mutations require review_required status",
                    details={"status": record.status.value},
                )
            document, queue = self._load_review_state(record)
            before_document = document
            next_document, next_queue, decision = mutator(document, queue)
            assert_raw_text_unchanged(before_document, next_document)
            next_document, next_queue = validate_review_state(
                next_document,
                next_queue,
                expected_job_id=record.request.job_id,
                high_margin_threshold=self.high_speaker_margin_threshold,
            )
            document_path, queue_path, journal_path = self._review_paths(record)
            atomic_write_json_transaction(
                {
                    document_path: next_document,
                    queue_path: next_queue,
                },
                journal_path=journal_path,
            )
            self._sync_record_from_review_state(
                record,
                next_document,
                next_queue,
            )
            record.checkpoint_sequence += 1
            self._write_checkpoint(record)
            response = {
                "jobId": record.request.job_id,
                "status": record.status.value,
                "command": command,
                "decision": decision,
                "openCount": record.review_open_count,
                "documentHash": record.document_hash,
                "speakerCount": next_document["speakerPolicy"]["resolvedCount"],
            }
        self._emit(record, "review.decision.persisted", response)
        return response

    def _launch_followup(self, record: JobRecord, *, operation: str) -> None:
        if operation not in {"resume", "rerender"}:
            raise ValueError("unsupported follow-up operation")
        with record.lock:
            if record.status not in {
                JobStatus.REVIEW_REQUIRED,
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            }:
                raise WorkerError(
                    "JOB_NOT_RESUMABLE",
                    "job is not in a state that permits a follow-up operation",
                    details={"status": record.status.value},
                )
            if record.future is not None and not record.future.done():
                raise WorkerError(
                    "JOB_ALREADY_RUNNING",
                    "job already has an active operation",
                )
            previous_status = record.status
            previous_stage = record.stage
            previous_quality = record.quality_status
            previous_error = record.error
            if not self._capacity.acquire(blocking=False):
                raise WorkerError(
                    "WORKER_BACKPRESSURE",
                    "worker capacity is full; retry after an active job completes",
                    retryable=True,
                )
            claimed = False
            try:
                with self._lock:
                    if self._closed:
                        raise WorkerError(
                            "WORKER_CLOSED",
                            "worker is shutting down",
                        )
                    owner = self._active_outputs.get(
                        record.request.output_directory
                    )
                    if owner not in {None, record.request.job_id}:
                        raise WorkerError(
                            "OUTPUT_DIRECTORY_IN_USE",
                            "another active job owns this output directory",
                            details={"jobId": owner},
                        )
                    self._active_outputs[
                        record.request.output_directory
                    ] = record.request.job_id
                    claimed = True
                record.capacity_released = False
                record.cancellation.clear()
                record.error = None
                record.followup_operation = operation
                record.status = JobStatus.QUEUED
                record.stage = f"{operation}_queued"
                record.quality_status = (
                    "pending" if operation == "rerender" else previous_quality
                )
                record.checkpoint_sequence += 1
                self._write_checkpoint(record)
                record.future = self.executor.submit(
                    self._run_followup,
                    record,
                    operation,
                )
            except Exception:
                record.status = previous_status
                record.stage = previous_stage
                record.quality_status = previous_quality
                record.error = previous_error
                record.followup_operation = None
                if claimed:
                    with self._lock:
                        if (
                            self._active_outputs.get(
                                record.request.output_directory
                            )
                            == record.request.job_id
                        ):
                            self._active_outputs.pop(
                                record.request.output_directory,
                                None,
                            )
                if not record.capacity_released:
                    record.capacity_released = True
                    self._capacity.release()
                raise

    def _run_followup(self, record: JobRecord, operation: str) -> None:
        context = AdapterContext(
            job_id=record.request.job_id,
            output_directory=record.request.output_directory,
            cancellation=record.cancellation,
        )
        try:
            context.raise_if_cancelled()
            self._transition(record, JobStatus.RUNNING, f"{operation}_validation")
            document, queue = self._load_review_state(record)
            if open_count(queue) != 0:
                raise WorkerError(
                    "REVIEW_INCOMPLETE",
                    f"job.{operation} refuses unresolved review items",
                    details={"openCount": open_count(queue)},
                )
            self._sync_record_from_review_state(record, document, queue)
            if operation == "rerender" or record.request.render_pdf:
                self._transition(record, JobStatus.RUNNING, "rendering")
                self._emit(
                    record,
                    "stage.started",
                    {
                        "stage": "rendering",
                        "operation": operation,
                        "adapterId": self.renderer_adapter.adapter_id,
                        "adapterVersion": self.renderer_adapter.version,
                    },
                )
                self._render_persisted_document(record, document, context)
            elif record.quality_status == "review-complete":
                record.quality_status = "not-requested"
            context.raise_if_cancelled()
            with record.lock:
                record.followup_operation = None
            self._transition(record, JobStatus.COMPLETED, "completed")
            self._emit(
                record,
                "job.completed",
                {
                    "status": "completed",
                    "operation": operation,
                    "artifactPaths": list(record.artifact_paths),
                },
            )
        except JobCancelled:
            with record.lock:
                record.followup_operation = None
            self._finish_cancelled(record)
        except WorkerError as exc:
            with record.lock:
                record.followup_operation = None
            self._finish_failed(record, exc)
        except Exception as exc:
            with record.lock:
                record.followup_operation = None
            self._finish_failed(
                record,
                WorkerError(
                    "INTERNAL_ERROR",
                    "follow-up operation failed closed",
                    details={"exceptionType": type(exc).__name__},
                ),
            )
        finally:
            with record.lock:
                record.followup_operation = None
            if record.status in _TERMINAL:
                with self._lock:
                    if (
                        self._active_outputs.get(
                            record.request.output_directory
                        )
                        == record.request.job_id
                    ):
                        self._active_outputs.pop(
                            record.request.output_directory,
                            None,
                        )
                self._release_capacity(record)

    def _render_persisted_document(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
        context: AdapterContext,
    ) -> None:
        render_result = self.renderer_adapter.render(
            document,
            record.request,
            context,
        )
        if not isinstance(render_result, RenderResult):
            raise WorkerError(
                "RENDER_RESULT_INVALID",
                "renderer adapter must return RenderResult",
            )
        render_result.validate()
        self._accept_render_result(record, render_result)

    def _run_job(self, record: JobRecord) -> None:
        context = AdapterContext(
            job_id=record.request.job_id,
            output_directory=record.request.output_directory,
            cancellation=record.cancellation,
        )
        try:
            context.raise_if_cancelled()
            self._transition(record, JobStatus.RUNNING, "transcription")
            self._emit(record, "job.started", {"status": "running"})
            self._emit(
                record,
                "stage.started",
                {
                    "stage": "transcription",
                    "adapterId": self.transcription_adapter.adapter_id,
                    "adapterVersion": self.transcription_adapter.version,
                },
            )
            raw_result = self.transcription_adapter.transcribe(record.request, context)
            result = (
                raw_result
                if isinstance(raw_result, TranscriptionResult)
                else TranscriptionResult.from_mapping(raw_result)
            )
            context.raise_if_cancelled()
            self._transition(record, JobStatus.RUNNING, "validation")
            count, estimate = resolve_speaker_count(
                record.request.speaker_policy, result
            )
            validate_segments(
                result.segments,
                speaker_count=count,
                duration_ms=result.duration_ms,
                high_margin_threshold=self.high_speaker_margin_threshold,
            )
            document = assemble_transcript_document(
                job_id=record.request.job_id,
                source_path=record.request.source_path,
                policy=record.request.speaker_policy,
                estimate=estimate,
                speaker_count=count,
                result=result,
                title=record.request.title,
                language=record.request.language,
                adapter_id=self.transcription_adapter.adapter_id,
                adapter_version=self.transcription_adapter.version,
            )
            transcript_path = (
                record.request.output_directory / "transcript-document.v2.json"
            )
            if result.pipeline_metrics is not None:
                metrics = dict(result.pipeline_metrics)
                try:
                    validate_strict_json(metrics)
                except ValueError as exc:
                    raise WorkerError(
                        "PIPELINE_METRICS_INVALID",
                        "pipeline metrics must contain exact finite JSON values",
                        details={"reason": str(exc)},
                    ) from exc
                reference = metrics.get("referenceEvaluation")
                has_reference = (
                    isinstance(reference, Mapping)
                    and reference.get("available") is True
                )
                if metrics.get("schemaVersion") != "1.0.0":
                    raise WorkerError(
                        "PIPELINE_METRICS_INVALID",
                        "pipeline metrics schemaVersion must be 1.0.0",
                    )
                if metrics.get("jobId") != record.request.job_id:
                    raise WorkerError(
                        "PIPELINE_METRICS_INVALID",
                        "pipeline metrics jobId does not match the job",
                    )
                if metrics.get("durationMs") != result.duration_ms:
                    raise WorkerError(
                        "PIPELINE_METRICS_INVALID",
                        "pipeline metrics durationMs does not match the transcript",
                    )

                def contains_reference_metric(value: Any) -> bool:
                    if isinstance(value, Mapping):
                        return any(
                            key.casefold()
                            in {
                                "quality",
                                "der",
                                "jer",
                                "speakerconfusion",
                                "overlapf1",
                            }
                            or contains_reference_metric(item)
                            for key, item in value.items()
                        )
                    if isinstance(value, list):
                        return any(contains_reference_metric(item) for item in value)
                    return False

                metrics_without_availability = {
                    key: value
                    for key, value in metrics.items()
                    if key != "referenceEvaluation"
                }
                if not has_reference and contains_reference_metric(
                    metrics_without_availability
                ):
                    raise WorkerError(
                        "PIPELINE_METRICS_INVALID",
                        "DER/JER/quality cannot be emitted without reference labels",
                    )
                metrics_path = (
                    record.request.output_directory / "pipeline-metrics.v1.json"
                )
                atomic_write_json(metrics_path, metrics)
                record.pipeline_metrics_path = str(metrics_path)
                record.artifact_paths.append(str(metrics_path))
                self._emit(
                    record,
                    "artifact.created",
                    {
                        "artifactType": "pipeline-metrics-v1",
                        "path": str(metrics_path),
                        "sha256": canonical_json_sha256(metrics),
                    },
                )
            review_queue = build_review_queue(
                job_id=record.request.job_id,
                policy=record.request.speaker_policy,
                estimate=estimate,
                segments=result.segments,
                count_confidence_threshold=self.count_confidence_threshold,
                segment_confidence_threshold=self.segment_confidence_threshold,
                speaker_margin_threshold=self.low_speaker_margin_threshold,
                range_width_threshold=self.range_width_threshold,
            )
            review_path = (
                record.request.output_directory / "review" / "review-queue.json"
            )
            document, review_queue = validate_review_state(
                document,
                review_queue,
                expected_job_id=record.request.job_id,
                high_margin_threshold=self.high_speaker_margin_threshold,
            )
            context.raise_if_cancelled()
            atomic_write_json_transaction(
                {
                    transcript_path: document,
                    review_path: review_queue,
                },
                journal_path=(
                    record.request.output_directory
                    / ".review-transaction.v1.json"
                ),
            )
            self._sync_record_from_review_state(
                record,
                document,
                review_queue,
            )
            self._emit(
                record,
                "artifact.created",
                {
                    "artifactType": "transcript-document-v2",
                    "path": str(transcript_path),
                    "sha256": record.document_hash,
                },
            )
            self._emit(
                record,
                "artifact.created",
                {
                    "artifactType": "review-queue-v2",
                    "path": str(review_path),
                    "sha256": canonical_json_sha256(review_queue),
                },
            )
            if record.review_open_count:
                record.quality_status = "review-required"
                self._transition(
                    record, JobStatus.REVIEW_REQUIRED, "review_required"
                )
                self._emit(
                    record,
                    "review.required",
                    {
                        "reviewQueuePath": str(review_path),
                        "openCount": record.review_open_count,
                    },
                )
                return

            if record.request.render_pdf:
                self._transition(record, JobStatus.RUNNING, "rendering")
                self._emit(
                    record,
                    "stage.started",
                    {
                        "stage": "rendering",
                        "adapterId": self.renderer_adapter.adapter_id,
                        "adapterVersion": self.renderer_adapter.version,
                    },
                )
                self._render_persisted_document(record, document, context)
            else:
                record.quality_status = "not-requested"
            context.raise_if_cancelled()
            self._transition(record, JobStatus.COMPLETED, "completed")
            self._emit(
                record,
                "job.completed",
                {
                    "status": "completed",
                    "artifactPaths": list(record.artifact_paths),
                },
            )
        except JobCancelled:
            self._finish_cancelled(record)
        except WorkerError as exc:
            self._finish_failed(record, exc)
        except Exception as exc:
            self._finish_failed(
                record,
                WorkerError(
                    "INTERNAL_ERROR",
                    "worker failed closed due to an unexpected internal error",
                    details={"exceptionType": type(exc).__name__},
                ),
            )
        finally:
            if record.status in _TERMINAL:
                with self._lock:
                    self._active_outputs.pop(
                        record.request.output_directory, None
                    )
                self._release_capacity(record)

    def _release_capacity(self, record: JobRecord) -> None:
        with record.lock:
            if record.capacity_released:
                return
            record.capacity_released = True
        self._capacity.release()

    def _accept_render_result(
        self, record: JobRecord, result: RenderResult
    ) -> None:
        paths = (
            result.quality_report_path,
            result.render_manifest_path,
            *result.artifact_paths,
        )
        verified = [
            self.path_policy.verify_artifact(
                Path(path), record.request.output_directory
            )
            for path in paths
        ]
        record.template_hash = result.template_hash
        record.renderer_version = result.renderer_version
        record.quality_status = result.quality_status
        record.quality_report_path = str(verified[0])
        record.render_manifest_path = str(verified[1])
        for path in verified[2:]:
            text = str(path)
            if text not in record.artifact_paths:
                record.artifact_paths.append(text)

    def _transition(
        self, record: JobRecord, status: JobStatus, stage: str
    ) -> None:
        with record.lock:
            record.status = status
            record.stage = stage
            record.checkpoint_sequence += 1
        self._write_checkpoint(record)

    def _finish_failed(self, record: JobRecord, error: WorkerError) -> None:
        with record.lock:
            if record.status is JobStatus.CANCELLED:
                return
            record.error = error.as_payload()
            record.quality_status = "failed"
        self._transition(record, JobStatus.FAILED, "failed")
        self._emit(record, "job.failed", error.as_payload())

    def _finish_cancelled(self, record: JobRecord) -> None:
        with record.lock:
            if record.status in {
                JobStatus.CANCELLED,
                JobStatus.COMPLETED,
                JobStatus.REVIEW_REQUIRED,
                JobStatus.FAILED,
            }:
                return
            record.quality_status = "cancelled"
        self._transition(record, JobStatus.CANCELLED, "cancelled")
        self._emit(record, "job.cancelled", {"status": "cancelled"})
        with self._lock:
            self._active_outputs.pop(record.request.output_directory, None)
        self._release_capacity(record)

    def _emit(
        self, record: JobRecord, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        with record.lock:
            sequence = record.sequence
            record.sequence += 1
        event = {
            "schemaVersion": PROTOCOL_VERSION,
            "eventId": f"evt-{uuid.uuid4().hex}",
            "jobId": record.request.job_id,
            "sequence": sequence,
            "timestamp": utc_now(),
            "type": event_type,
            "payload": dict(payload),
        }
        try:
            self.event_sink(event)
        except Exception:
            # A broken observer must not corrupt persisted job state.
            pass

    def _write_checkpoint(self, record: JobRecord) -> None:
        with record.lock:
            checkpoint = {
                "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
                "jobId": record.request.job_id,
                "status": record.status.value,
                "stage": record.stage,
                "sequence": record.checkpoint_sequence,
                "updatedAt": utc_now(),
                "sourcePath": str(record.request.source_path),
                "outputDirectory": str(record.request.output_directory),
                "speakerCountPolicy": record.request.speaker_policy.as_dict(),
                "documentHash": record.document_hash,
                "templateHash": record.template_hash,
                "rendererVersion": record.renderer_version,
                "qualityStatus": record.quality_status,
                "qualityReportPath": record.quality_report_path,
                "renderManifestPath": record.render_manifest_path,
                "reviewQueuePath": record.review_queue_path,
                "reviewOpenCount": record.review_open_count,
                "pipelineMetricsPath": record.pipeline_metrics_path,
                "artifactPaths": list(record.artifact_paths),
                "error": dict(record.error) if record.error else None,
            }
        atomic_write_json(
            record.request.output_directory / "checkpoint.v2.json",
            checkpoint,
        )
