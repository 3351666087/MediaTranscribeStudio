"""Concurrent job state machine for the offline headless worker."""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
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
from .business_processing import (
    BUSINESS_PROMPT_VERSION,
    BusinessProcessingConfig,
    BusinessProcessingRunner,
)
from .documents import (
    assemble_transcript_document,
    build_review_queue,
    resolve_speaker_count,
    utc_now,
    validate_segments,
)
from .errors import JobCancelled, WorkerError, invalid_request
from .final_adjudication import (
    build_final_adjudicated_transcript,
    validate_final_adjudicated_transcript,
)
from .local_llm import LocalLLMConfig, LocalLLMProvider, OllamaLocalProvider
from .language import normalize_language_tag
from .media_probe import MediaProbeResult
from .models import (
    CHECKPOINT_SCHEMA_VERSION,
    JobStatus,
    PROTOCOL_VERSION,
    RenderArtifact,
    RenderResult,
    SpeakerCountPolicy,
    StartJobRequest,
    TranscriptionResult,
    validate_job_id,
)
from .output_orchestration import (
    MediaProbeArtifact,
    OutputExecutionPlan,
    compile_output_execution_plan,
    persist_media_probe_artifact,
    render_planned_report,
)
from .output_publication import (
    OUTPUT_PUBLICATION_SCHEMA_VERSION,
    OutputPublicationManifest,
    publish_output_plans,
)
from .output_recipe import (
    OutputRecipeError,
    compile_output_customizations,
    parse_output_recipe,
    render_recipe_file_name,
)
from .paths import PathPolicy
from .persistence import (
    atomic_publish_json_evidence,
    atomic_write_json,
    atomic_write_json_no_replace,
    atomic_write_json_transaction,
    canonical_json_sha256,
    read_json_strict,
    recover_json_transaction,
    validate_strict_json,
)
from .transcript_exports import export_transcript
from .voice_activity import validate_voice_activity
from .review import (
    assert_raw_text_unchanged,
    merge_speakers,
    open_count,
    rename_speaker,
    resolve_review_item,
    split_speaker,
    validate_review_state,
)
from .semantic_processing import (
    SemanticProcessingRunner,
    attach_semantic_suggestions_to_review,
)


EventSink = Callable[[dict[str, Any]], None]
_LOGGER = logging.getLogger(__name__)
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
    media_probe_result: MediaProbeResult | None = None
    media_probe_artifact: MediaProbeArtifact | None = None
    media_probe_artifact_path: str | None = None
    voice_activity_artifact_path: str | None = None
    output_execution_plans: tuple[OutputExecutionPlan, ...] = ()
    output_plan_paths: list[str] = field(default_factory=list)
    output_plan_hashes: list[str] = field(default_factory=list)
    output_manifest_path: str | None = None
    output_publication_status: str = "not-requested"
    output_publication_manifest_path: str | None = None
    output_publication_manifest_sha256: str | None = None
    output_publication_manifest_file_sha256: str | None = None
    output_publication_receipts: list[dict[str, Any]] = field(
        default_factory=list
    )
    output_publication_error: dict[str, Any] | None = None
    transcript_export_receipts: list[dict[str, Any]] = field(
        default_factory=list
    )
    artifact_paths: list[str] = field(default_factory=list)
    business_status: str = "not-requested"
    business_manifest_path: str | None = None
    business_artifact_paths: list[str] = field(default_factory=list)
    business_provenance: dict[str, Any] | None = None
    business_error: dict[str, Any] | None = None
    semantic_status: str = "not-requested"
    semantic_artifact_path: str | None = None
    semantic_provenance: dict[str, Any] | None = None
    semantic_error: dict[str, Any] | None = None
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
        business_provider: LocalLLMProvider | None = None,
        business_provider_factory: Callable[
            [StartJobRequest], LocalLLMProvider
        ]
        | None = None,
        business_runner_factory: Callable[
            [StartJobRequest, AdapterContext], BusinessProcessingRunner
        ]
        | None = None,
        semantic_provider: LocalLLMProvider | None = None,
        semantic_provider_factory: Callable[
            [StartJobRequest], LocalLLMProvider
        ]
        | None = None,
        semantic_runner_factory: Callable[
            [StartJobRequest, AdapterContext], SemanticProcessingRunner
        ]
        | None = None,
        semantic_required: bool = False,
        semantic_model: str | None = None,
        media_probe: Any | None = None,
        output_publisher: Callable[..., OutputPublicationManifest]
        | None = publish_output_plans,
        subtitle_delivery_executor: Any | None = None,
        subtitle_visual_qa_hook: Callable[..., Any] | None = None,
        heartbeat_interval_seconds: float = 15.0,
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
        if (
            not math.isfinite(heartbeat_interval_seconds)
            or heartbeat_interval_seconds <= 0
        ):
            raise ValueError(
                "heartbeat_interval_seconds must be finite and positive"
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
        self.business_provider = business_provider
        self.business_provider_factory = business_provider_factory
        self.business_runner_factory = business_runner_factory
        self.semantic_provider = semantic_provider
        self.semantic_provider_factory = semantic_provider_factory
        self.semantic_runner_factory = semantic_runner_factory
        self.semantic_required = bool(semantic_required)
        if semantic_model is not None and (
            not isinstance(semantic_model, str) or not semantic_model.strip()
        ):
            raise ValueError("semantic_model must be non-empty text when provided")
        self.semantic_model = (
            semantic_model.strip()
            if isinstance(semantic_model, str)
            else None
        )
        self.media_probe = media_probe
        self.output_publisher = output_publisher
        self.subtitle_delivery_executor = subtitle_delivery_executor
        self.subtitle_visual_qa_hook = subtitle_visual_qa_hook
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._jobs: dict[str, JobRecord] = {}
        self._active_outputs: dict[Path, str] = {}
        self._lock = threading.RLock()
        self._event_emit_lock = threading.Lock()
        self._closed = False
        self._adapter_resources_released = False

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
            "outputCustomization",
            "title",
            "language",
            "localLlmMode",
            "localLlmModel",
            "localLlmAutoApply",
            "translationTargets",
            "summary",
            "outputLocale",
            "businessPromptVersion",
            "localLlmEndpoint",
            "localLlmEndpointPolicy",
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
        render_pdf_supplied = "renderPdf" in payload
        render_pdf = payload.get("renderPdf", False)
        if not isinstance(render_pdf, bool):
            raise invalid_request("renderPdf must be a boolean")
        output_recipe = None
        if "outputCustomization" in payload:
            try:
                output_recipe = parse_output_recipe(
                    payload["outputCustomization"]
                )
            except OutputRecipeError as exc:
                raise invalid_request(
                    "outputCustomization is invalid",
                    reason=str(exc),
                ) from exc
            recipe_render_pdf = output_recipe.render_pdf
            if render_pdf_supplied and render_pdf != recipe_render_pdf:
                raise invalid_request(
                    "renderPdf conflicts with outputCustomization",
                    renderPdf=render_pdf,
                    resolvedRenderPdf=recipe_render_pdf,
                )
            render_pdf = recipe_render_pdf
        title_raw = payload.get("title")
        title = None
        if title_raw is not None:
            if not isinstance(title_raw, str) or not title_raw.strip():
                raise invalid_request("title must be a non-empty string")
            title = title_raw.strip()
            if len(title) > 240:
                raise invalid_request("title exceeds 240 characters")
        language_raw = payload.get("language", "auto")
        try:
            language = normalize_language_tag(language_raw, allow_auto=True)
        except ValueError as exc:
            raise invalid_request(
                "language must be auto or a valid BCP-47 language tag"
            ) from exc
        local_llm_mode = str(
            payload.get("localLlmMode") or "disabled"
        ).strip()
        if local_llm_mode not in {
            "disabled",
            "suggestion-only",
            "business",
            "enabled",
        }:
            raise invalid_request(
                "localLlmMode must be disabled, suggestion-only, business, or enabled"
            )
        local_llm_model_raw = payload.get("localLlmModel", "qwen3.5:9b")
        if not isinstance(local_llm_model_raw, str):
            raise invalid_request("localLlmModel must be a string")
        local_llm_model = local_llm_model_raw.strip() or "qwen3.5:9b"
        if local_llm_model != "qwen3.5:9b":
            raise invalid_request(
                "localLlmModel must identify the production model qwen3.5:9b"
            )
        endpoint_raw = payload.get(
            "localLlmEndpoint", "http://127.0.0.1:11434"
        )
        if not isinstance(endpoint_raw, str):
            raise invalid_request("localLlmEndpoint must be a string")
        endpoint = endpoint_raw.strip() or "http://127.0.0.1:11434"
        endpoint_policy_raw = payload.get(
            "localLlmEndpointPolicy", "loopback-only"
        )
        if not isinstance(endpoint_policy_raw, str):
            raise invalid_request("localLlmEndpointPolicy must be a string")
        endpoint_policy = endpoint_policy_raw.strip() or "loopback-only"
        if endpoint_policy != "loopback-only":
            raise invalid_request(
                "localLlmEndpointPolicy must be loopback-only"
            )
        try:
            LocalLLMConfig(model=local_llm_model, endpoint=endpoint)
        except ValueError as exc:
            raise invalid_request(
                "localLlmEndpoint must be a valid loopback-only endpoint"
            ) from exc
        local_llm_auto_apply = payload.get("localLlmAutoApply", False)
        if not isinstance(local_llm_auto_apply, bool):
            raise invalid_request("localLlmAutoApply must be a boolean")
        if local_llm_auto_apply:
            raise invalid_request("localLlmAutoApply is permanently forbidden")
        raw_targets = payload.get("translationTargets", [])
        if not isinstance(raw_targets, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in raw_targets
        ):
            raise invalid_request(
                "translationTargets must be an array of non-empty language tags"
            )
        if not isinstance(payload.get("summary", False), bool):
            raise invalid_request("summary must be a boolean")
        output_locale_raw = payload.get("outputLocale", "en")
        if not isinstance(output_locale_raw, str):
            raise invalid_request("outputLocale must be a language-tag string")
        prompt_version_raw = payload.get(
            "businessPromptVersion", BUSINESS_PROMPT_VERSION
        )
        if not isinstance(prompt_version_raw, str):
            raise invalid_request("businessPromptVersion must be a string")
        try:
            business_config = BusinessProcessingConfig(
                translation_targets=tuple(raw_targets),
                summary=payload.get("summary", False),
                model=local_llm_model,
                output_locale=output_locale_raw,
                prompt_version=prompt_version_raw,
            )
        except (TypeError, ValueError) as exc:
            raise invalid_request(
                "business processing options are invalid"
            ) from exc
        if business_config.enabled and local_llm_mode == "disabled":
            raise invalid_request(
                "localLlmMode must enable business processing when variants are requested"
            )
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
            local_llm_endpoint=endpoint,
            business_config=business_config,
            output_recipe=output_recipe,
        )

    def register(self, request: StartJobRequest) -> JobRecord:
        if not self._capacity.acquire(blocking=False):
            raise WorkerError(
                "WORKER_BACKPRESSURE",
                "worker capacity is full; retry after an active job completes",
                retryable=True,
            )
        record = JobRecord(request=request)
        if request.business_config.enabled:
            record.business_status = "pending"
        if self.semantic_required or request.local_llm_mode in {
            "suggestion-only",
            "enabled",
        }:
            record.semantic_status = "pending"
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
                "outputCustomization": (
                    record.request.output_recipe.canonical_dict()
                    if record.request.output_recipe is not None
                    else None
                ),
                "outputCustomizationSha256": (
                    record.request.output_recipe.deterministic_hash()
                    if record.request.output_recipe is not None
                    else None
                ),
                "cancellationRequested": record.cancellation.is_set(),
                "reviewOpenCount": record.review_open_count,
                "qualityStatus": record.quality_status,
                "documentHash": record.document_hash,
                "reviewQueuePath": record.review_queue_path,
                "qualityReportPath": record.quality_report_path,
                "renderManifestPath": record.render_manifest_path,
                "pipelineMetricsPath": record.pipeline_metrics_path,
                "mediaProbeArtifactPath": record.media_probe_artifact_path,
                "voiceActivityArtifactPath": (
                    record.voice_activity_artifact_path
                ),
                "outputPlanPaths": list(record.output_plan_paths),
                "outputPlanHashes": list(record.output_plan_hashes),
                "outputManifestPath": record.output_manifest_path,
                "outputPublication": self._output_publication_event_payload(
                    record
                ),
                "transcriptExports": [
                    dict(receipt)
                    for receipt in record.transcript_export_receipts
                ],
                "artifactPaths": list(record.artifact_paths),
                "followupOperation": record.followup_operation,
                "business": {
                    "status": record.business_status,
                    "config": record.request.business_config.as_dict(),
                    "manifestPath": record.business_manifest_path,
                    "artifactPaths": list(record.business_artifact_paths),
                    "provenance": (
                        dict(record.business_provenance)
                        if record.business_provenance
                        else None
                    ),
                    "error": (
                        dict(record.business_error)
                        if record.business_error
                        else None
                    ),
                },
                "semantic": self._semantic_event_payload(record),
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
        with self._lock:
            should_release = wait and not self._adapter_resources_released
            if should_release:
                self._adapter_resources_released = True
        if should_release:
            release = getattr(self.transcription_adapter, "release_resources", None)
            if release is not None:
                if not callable(release):
                    raise WorkerError(
                        "TRANSCRIPTION_ADAPTER_RESOURCE_RELEASE_INVALID",
                        "transcription adapter resource release must be callable",
                    )
                release()

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
            self._persist_final_adjudicated_transcript(
                record,
                document,
                queue,
            )
            self._run_business_processing(record, document, context)
            self._ensure_output_execution_plans(record, document)
            self._execute_transcript_exports(record, document, context)
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
            self._execute_output_publication(record, document, context)
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
                    "business": self._business_event_payload(record),
                    "semantic": self._semantic_event_payload(record),
                    "outputPublication": (
                        self._output_publication_event_payload(record)
                    ),
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
        if record.request.output_recipe is not None:
            report_plan = next(
                (
                    plan
                    for plan in record.output_execution_plans
                    if plan.report_enabled
                ),
                None,
            )
            if report_plan is None:
                raise WorkerError(
                    "OUTPUT_PLAN_MISSING",
                    "the output recipe enables PDF but no report execution plan exists",
                )
            render_result = render_planned_report(
                report_plan,
                renderer=self.renderer_adapter,
                document=document,
                request=record.request,
                context=context,
            )
        else:
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

    def _probe_source_media(
        self,
        record: JobRecord,
        context: AdapterContext,
    ) -> None:
        if self.media_probe is None:
            if record.request.output_recipe is not None:
                raise WorkerError(
                    "MEDIA_PROBE_REQUIRED",
                    "output recipes require a trusted content-driven media probe",
                )
            return
        self._transition(record, JobStatus.RUNNING, "media_probe")
        self._emit(
            record,
            "stage.started",
            {
                "stage": "media_probe",
                "admission": "content-driven",
                "extensionTrusted": False,
            },
        )
        context.raise_if_cancelled()
        raw_probe = self.media_probe.probe(record.request.source_path)
        if not isinstance(raw_probe, MediaProbeResult):
            raise WorkerError(
                "MEDIA_PROBE_RESULT_INVALID",
                "trusted media probe must return MediaProbeResult",
            )
        artifact = persist_media_probe_artifact(
            raw_probe,
            source_path=record.request.source_path,
            output_directory=record.request.output_directory,
        )
        context.raise_if_cancelled()
        record.media_probe_result = raw_probe
        record.media_probe_artifact = artifact
        record.media_probe_artifact_path = str(artifact.path)
        if str(artifact.path) not in record.artifact_paths:
            record.artifact_paths.append(str(artifact.path))
        self._emit(
            record,
            "artifact.created",
            {
                "artifactType": "media-probe-v1",
                "path": str(artifact.path),
                "sha256": artifact.sha256,
            },
        )

    def _ensure_output_execution_plans(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
    ) -> tuple[OutputExecutionPlan, ...]:
        recipe = record.request.output_recipe
        if recipe is None:
            return ()
        if record.output_execution_plans:
            return record.output_execution_plans
        media_probe = record.media_probe_result
        media_probe_artifact = record.media_probe_artifact
        if media_probe is None or media_probe_artifact is None:
            raise WorkerError(
                "MEDIA_PROBE_REQUIRED",
                "output planning requires persisted trusted media-probe evidence",
            )
        speaker_policy = document.get("speakerPolicy")
        if not isinstance(speaker_policy, MappingABC):
            raise WorkerError(
                "OUTPUT_PLAN_INPUT_INVALID",
                "transcript document is missing speakerPolicy",
            )
        speaker_count = speaker_policy.get("resolvedCount")
        if isinstance(speaker_count, bool) or not isinstance(speaker_count, int):
            raise WorkerError(
                "OUTPUT_PLAN_INPUT_INVALID",
                "transcript document has no resolved speaker count",
            )
        language = document.get("language")
        if not isinstance(language, str) or not language.strip():
            raise WorkerError(
                "OUTPUT_PLAN_INPUT_INVALID",
                "transcript document has no persisted language",
            )
        generated_at = document.get("generatedAt")
        if (
            not isinstance(generated_at, str)
            or len(generated_at) < 10
            or generated_at[4:5] != "-"
            or generated_at[7:8] != "-"
        ):
            raise WorkerError(
                "OUTPUT_PLAN_INPUT_INVALID",
                "transcript document has no canonical generated date",
            )
        generated_date = generated_at[:10]
        try:
            compiled = compile_output_customizations(
                recipe,
                source_path=record.request.source_path,
                output_directory=record.request.output_directory,
                media_probe=media_probe,
                media_probe_artifact=media_probe_artifact,
            )
        except OutputRecipeError as exc:
            raise WorkerError(
                "OUTPUT_RECIPE_COMPILE_FAILED",
                "the canonical output recipe could not be compiled",
                details={"reason": str(exc)},
            ) from exc
        plans: list[OutputExecutionPlan] = []
        plan_paths: list[str] = []
        plan_hashes: list[str] = []
        snapshots_root = record.request.output_directory / "output-plans"
        for index, item in enumerate(compiled, start=1):
            plan = compile_output_execution_plan(
                item.customization,
                source_path=record.request.source_path,
                output_directory=record.request.output_directory,
                media_probe=media_probe,
                media_probe_artifact=media_probe_artifact,
                language=language,
                speaker_count=speaker_count,
                generated_date=generated_date,
            )
            mode = item.delivery_mode
            customization_path = (
                snapshots_root
                / f"{index:02d}-{mode}.output-customization.v1.json"
            )
            plan_path = (
                snapshots_root / f"{index:02d}-{mode}.output-plan.v1.json"
            )
            customization_evidence = atomic_publish_json_evidence(
                customization_path,
                item.customization.canonical_dict(),
            )
            plan_evidence = atomic_publish_json_evidence(
                plan_path,
                plan.to_dict(),
            )
            for artifact_type, evidence in (
                ("output-customization-v1", customization_evidence),
                ("output-execution-plan-v1", plan_evidence),
            ):
                text = str(evidence.path)
                if text not in record.artifact_paths:
                    record.artifact_paths.append(text)
                self._emit(
                    record,
                    "artifact.created",
                    {
                        "artifactType": artifact_type,
                        "path": text,
                        "sha256": evidence.sha256,
                    },
                )
            plans.append(plan)
            plan_paths.append(str(plan_evidence.path))
            plan_hashes.append(plan.deterministic_hash())
        record.output_execution_plans = tuple(plans)
        record.output_plan_paths = plan_paths
        record.output_plan_hashes = plan_hashes
        return record.output_execution_plans

    def _execute_transcript_exports(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
        context: AdapterContext,
    ) -> tuple[dict[str, Any], ...]:
        recipe = record.request.output_recipe
        if recipe is None:
            return ()
        if record.transcript_export_receipts:
            return tuple(
                dict(receipt)
                for receipt in record.transcript_export_receipts
            )
        requested = tuple(
            value
            for value in ("json", "txt", "markdown", "html")
            if value in recipe.formats
        )
        if not requested:
            return ()
        speaker_policy = document.get("speakerPolicy")
        if not isinstance(speaker_policy, MappingABC):
            raise WorkerError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                "transcript export requires persisted speakerPolicy",
            )
        speaker_count = speaker_policy.get("resolvedCount")
        if isinstance(speaker_count, bool) or not isinstance(speaker_count, int):
            raise WorkerError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                "transcript export requires a resolved speaker count",
            )
        language = document.get("language")
        if not isinstance(language, str) or not language:
            raise WorkerError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                "transcript export requires a persisted language",
            )
        generated_at = document.get("generatedAt")
        if (
            not isinstance(generated_at, str)
            or len(generated_at) < 10
            or generated_at[4:5] != "-"
            or generated_at[7:8] != "-"
        ):
            raise WorkerError(
                "TRANSCRIPT_EXPORT_DOCUMENT_INVALID",
                "transcript export requires a canonical generated date",
            )
        try:
            output_stem = render_recipe_file_name(
                recipe,
                source_stem=record.request.source_path.stem,
                artifact="transcript",
                language=language,
                generated_date=generated_at[:10],
                speaker_count=speaker_count,
            )
        except OutputRecipeError as exc:
            raise WorkerError(
                "TRANSCRIPT_EXPORT_FILENAME_INVALID",
                "the output recipe could not derive a transcript file name",
                details={"reason": str(exc)},
            ) from exc
        suffixes = {
            "json": ".json",
            "txt": ".txt",
            "markdown": ".md",
            "html": ".html",
        }
        self._transition(record, JobStatus.RUNNING, "transcript_exports")
        self._emit(
            record,
            "stage.started",
            {
                "stage": "transcript_exports",
                "formats": list(requested),
            },
        )
        receipts: list[dict[str, Any]] = []
        for export_format in requested:
            context.raise_if_cancelled()
            receipt = export_transcript(
                document,
                export_format=export_format,
                output_root=record.request.output_directory,
                output_path=output_stem + suffixes[export_format],
            )
            payload = receipt.to_dict()
            receipts.append(payload)
            path_text = str(receipt.path)
            if path_text not in record.artifact_paths:
                record.artifact_paths.append(path_text)
            self._emit(
                record,
                "artifact.created",
                {
                    "artifactType": (
                        f"transcript-export-{export_format}-v1"
                    ),
                    **payload,
                },
            )
        record.transcript_export_receipts = receipts
        return tuple(dict(receipt) for receipt in receipts)

    def _execute_output_publication(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
        context: AdapterContext,
    ) -> dict[str, Any] | None:
        recipe = record.request.output_recipe
        if recipe is None or not recipe.canonical_dict()["subtitles"]["enabled"]:
            record.output_publication_status = "not-requested"
            record.output_publication_error = None
            return None

        plans = tuple(record.output_execution_plans)
        if not plans:
            error = WorkerError(
                "OUTPUT_PUBLICATION_PLANS_REQUIRED",
                "subtitle publication requires canonical output execution plans",
            )
            self._invalidate_output_publication(record, error)
            raise error
        try:
            self._verify_persisted_output_plans(record, plans)
            existing = self._load_valid_output_publication(record, plans)
        except WorkerError as exc:
            self._invalidate_output_publication(record, exc)
            self._emit(
                record,
                "output.publication.invalid",
                self._output_publication_event_payload(record),
            )
            raise

        if existing is not None:
            self._record_output_publication(
                record,
                manifest_path=(
                    record.request.output_directory
                    / "output-publication-manifest.v1.json"
                ),
                manifest_payload=existing,
                manifest_file_sha256=self._exact_file_sha256(
                    record.request.output_directory
                    / "output-publication-manifest.v1.json"
                ),
            )
            self._emit(
                record,
                "output.publication.reused",
                self._output_publication_event_payload(record),
            )
            return existing

        if not callable(self.output_publisher):
            error = WorkerError(
                "OUTPUT_PUBLICATION_RUNTIME_UNAVAILABLE",
                "subtitle publication has no configured transactional publisher",
            )
            self._invalidate_output_publication(record, error)
            raise error
        if self.subtitle_delivery_executor is None:
            error = WorkerError(
                "OUTPUT_PUBLICATION_RUNTIME_UNAVAILABLE",
                "subtitle publication has no configured delivery executor",
            )
            self._invalidate_output_publication(record, error)
            raise error

        self._transition(record, JobStatus.RUNNING, "output_publication")
        record.output_publication_status = "publishing"
        record.output_publication_error = None
        self._emit(
            record,
            "stage.started",
            {
                "stage": "output_publication",
                "recipeSha256": recipe.deterministic_hash(),
                "planSha256": [
                    plan.deterministic_hash() for plan in plans
                ],
            },
        )
        context.raise_if_cancelled()
        manifest: OutputPublicationManifest | None = None
        manifest_published = False
        try:
            raw_manifest = self.output_publisher(
                recipe,
                plans,
                document,
                executor=self.subtitle_delivery_executor,
                visual_qa_hook=self.subtitle_visual_qa_hook,
                subtitle_language=self._publication_language(document),
                subtitle_title=self._publication_title(document),
                make_subtitle_default=False,
                cancellation_check=context.raise_if_cancelled,
            )
            if not isinstance(raw_manifest, OutputPublicationManifest):
                raise WorkerError(
                    "OUTPUT_PUBLICATION_RESULT_INVALID",
                    "transactional publisher must return OutputPublicationManifest",
                    details={
                        "resultType": type(raw_manifest).__name__,
                    },
                )
            manifest = raw_manifest
            manifest_payload = manifest.to_dict()
            self._validate_output_publication_manifest(
                record,
                plans,
                manifest_payload,
                expected_manifest_sha256=manifest.manifest_sha256,
            )
            context.raise_if_cancelled()

            manifest_path = (
                record.request.output_directory
                / "output-publication-manifest.v1.json"
            )
            evidence = atomic_publish_json_evidence(
                manifest_path,
                manifest_payload,
            )
            manifest_published = True
            persisted = read_json_strict(evidence.path)
            if persisted != manifest_payload:
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_MISMATCH",
                    "persisted output publication manifest changed during publication",
                )
            self._validate_output_publication_manifest(
                record,
                plans,
                persisted,
                expected_manifest_sha256=manifest.manifest_sha256,
            )
            self._record_output_publication(
                record,
                manifest_path=evidence.path,
                manifest_payload=persisted,
                manifest_file_sha256=evidence.sha256,
            )
            for receipt in record.output_publication_receipts:
                self._emit(
                    record,
                    "artifact.created",
                    {
                        **dict(receipt),
                        "artifactType": receipt["artifactType"],
                    },
                )
            self._emit(
                record,
                "artifact.created",
                {
                    "artifactType": "output-publication-manifest-v1",
                    "path": str(evidence.path),
                    "sha256": evidence.sha256,
                    "manifestSha256": manifest.manifest_sha256,
                },
            )
            self._emit(
                record,
                "output.publication.completed",
                self._output_publication_event_payload(record),
            )
            return persisted
        except JobCancelled:
            record.output_publication_status = "cancelled"
            record.output_publication_error = None
            if manifest is not None:
                self._rollback_unmanifested_output_publication(
                    record,
                    manifest,
                    cause=WorkerError(
                        "OUTPUT_PUBLICATION_CANCELLED",
                        "output publication was cancelled before its manifest committed",
                    ),
                    remove_manifest=manifest_published,
                )
            raise
        except WorkerError as exc:
            record.output_publication_status = "failed"
            record.output_publication_error = exc.as_payload()
            if manifest is not None:
                self._rollback_unmanifested_output_publication(
                    record,
                    manifest,
                    cause=exc,
                    remove_manifest=manifest_published,
                )
            raise
        except Exception as exc:
            error = WorkerError(
                "OUTPUT_PUBLICATION_FAILED",
                "subtitle and media publication failed closed",
                details={"exceptionType": type(exc).__name__},
            )
            record.output_publication_status = "failed"
            record.output_publication_error = error.as_payload()
            if manifest is not None:
                self._rollback_unmanifested_output_publication(
                    record,
                    manifest,
                    cause=error,
                    remove_manifest=manifest_published,
                )
            raise error from exc

    def _verify_persisted_output_plans(
        self,
        record: JobRecord,
        plans: tuple[OutputExecutionPlan, ...],
    ) -> None:
        expected_hashes = [plan.deterministic_hash() for plan in plans]
        if record.output_plan_hashes != expected_hashes:
            raise WorkerError(
                "OUTPUT_PUBLICATION_PLAN_MISMATCH",
                "in-memory output plan hashes do not match canonical plans",
            )
        if len(record.output_plan_paths) != len(plans):
            raise WorkerError(
                "OUTPUT_PUBLICATION_PLAN_MISMATCH",
                "persisted output plan paths do not match canonical plans",
            )
        for path_text, plan, expected_hash in zip(
            record.output_plan_paths,
            plans,
            expected_hashes,
            strict=True,
        ):
            path = self.path_policy.verify_artifact(
                Path(path_text),
                record.request.output_directory,
            )
            payload = read_json_strict(path)
            if (
                payload != plan.to_dict()
                or canonical_json_sha256(payload) != expected_hash
            ):
                raise WorkerError(
                    "OUTPUT_PUBLICATION_PLAN_MISMATCH",
                    "persisted output plan no longer matches the canonical plan",
                )

    def _load_valid_output_publication(
        self,
        record: JobRecord,
        plans: tuple[OutputExecutionPlan, ...],
    ) -> dict[str, Any] | None:
        manifest_path = (
            record.request.output_directory
            / "output-publication-manifest.v1.json"
        )
        if not manifest_path.exists():
            return None
        if not manifest_path.is_file():
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest path is not a regular file",
            )
        recorded_publication = any(
            (
                record.output_publication_manifest_path,
                record.output_publication_manifest_sha256,
                record.output_publication_receipts,
                record.output_publication_status == "published",
            )
        )
        expected_file_sha256 = (
            record.output_publication_manifest_file_sha256
        )
        if recorded_publication and not expected_file_sha256:
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_FILE_HASH_MISSING",
                "recorded output publication has no exact manifest file hash",
            )
        if expected_file_sha256 is not None:
            actual_file_sha256 = self._exact_file_sha256(manifest_path)
            if actual_file_sha256 != expected_file_sha256:
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_FILE_CHANGED",
                    "output publication manifest bytes changed after publication",
                    details={
                        "expectedSha256": expected_file_sha256,
                        "actualSha256": actual_file_sha256,
                    },
                )
        payload = read_json_strict(manifest_path)
        self._validate_output_publication_manifest(record, plans, payload)
        return payload

    def _validate_output_publication_manifest(
        self,
        record: JobRecord,
        plans: tuple[OutputExecutionPlan, ...],
        payload: Mapping[str, Any],
        *,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        required_keys = {
            "schemaVersion",
            "status",
            "recipeSha256",
            "source",
            "plans",
            "customerArtifacts",
            "internalEvidence",
            "transaction",
            "manifestSha256",
        }
        if set(payload) != required_keys:
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest fields are not canonical",
            )
        if (
            payload.get("schemaVersion")
            != OUTPUT_PUBLICATION_SCHEMA_VERSION
            or payload.get("status") != "published"
        ):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest status or version is unsupported",
            )
        claimed_hash = payload.get("manifestSha256")
        if not isinstance(claimed_hash, str) or len(claimed_hash) != 64:
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest has no valid self hash",
            )
        body = dict(payload)
        body.pop("manifestSha256")
        if canonical_json_sha256(body) != claimed_hash:
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest self hash does not verify",
            )
        if (
            expected_manifest_sha256 is not None
            and claimed_hash != expected_manifest_sha256
        ):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_MISMATCH",
                "publisher and persisted manifest hashes do not match",
            )

        recipe = record.request.output_recipe
        assert recipe is not None
        if payload.get("recipeSha256") != recipe.deterministic_hash():
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_STALE",
                "output publication manifest belongs to another output recipe",
            )

        source = payload.get("source")
        canonical_source = record.request.source_path.resolve(strict=True)
        if not isinstance(source, MappingABC):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest has no source evidence",
            )
        try:
            manifest_source = Path(str(source.get("path"))).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest source path is invalid",
            ) from exc
        if manifest_source != canonical_source:
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_STALE",
                "output publication manifest belongs to another source",
            )
        source_size = canonical_source.stat().st_size
        source_sha256 = self._artifact_sha256(canonical_source)
        if (
            source.get("sizeBytes") != source_size
            or source.get("sha256") != source_sha256
            or source.get("unchanged") is not True
        ):
            raise WorkerError(
                "OUTPUT_PUBLICATION_SOURCE_CHANGED",
                "source media no longer matches output publication evidence",
            )

        expected_plans = [
            {
                "deliveryMode": plan.delivery_mode.value,
                "customizationSha256": plan.customization_sha256,
                "executionPlanSha256": plan.deterministic_hash(),
            }
            for plan in plans
            if plan.subtitle_enabled
        ]
        mode_order = {"sidecar": 0, "soft-mux": 1, "burn-in": 2}
        expected_plans.sort(
            key=lambda item: mode_order[item["deliveryMode"]]
        )
        if payload.get("plans") != expected_plans:
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_STALE",
                "output publication manifest plan hashes are stale",
            )

        internal = payload.get("internalEvidence")
        if not isinstance(internal, MappingABC):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest internal evidence is invalid",
            )
        self._assert_no_private_publication_paths(internal)
        if (
            internal.get("privatePathsExcluded") is not True
            or internal.get("customerAndInternalEvidenceSeparated") is not True
        ):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication manifest does not prove path separation",
            )

        transaction = payload.get("transaction")
        if not isinstance(transaction, MappingABC) or any(
            transaction.get(key) is not True
            for key in (
                "allConflictsCheckedBeforeWrites",
                "mediaQuarantinedBeforePublication",
                "allMediaQaPassedBeforePublication",
                "rollbackSupported",
                "privatePathsExcluded",
                "sourceMediaImmutable",
            )
        ):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication transaction evidence is incomplete",
            )

        receipts = payload.get("customerArtifacts")
        if not isinstance(receipts, list):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                "output publication customer artifacts must be a list",
            )
        expected_receipts = self._expected_output_publication_receipts(recipe)
        actual_receipts: list[tuple[str, str | None, str | None]] = []
        seen_paths: set[Path] = set()
        for receipt in receipts:
            if not isinstance(receipt, MappingABC):
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                    "output publication receipt must be an object",
                )
            if set(receipt) != {
                "artifactType",
                "path",
                "sizeBytes",
                "sha256",
                "subtitleFormat",
                "deliveryMode",
                "visualQaEvidenceSha256",
                "sourceIntegrity",
                "publication",
            }:
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                    "output publication receipt fields are not canonical",
                )
            path_text = receipt.get("path")
            if not isinstance(path_text, str) or not path_text:
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                    "output publication receipt path is invalid",
                )
            path = self.path_policy.verify_artifact(
                Path(path_text),
                record.request.output_directory,
            )
            if path == canonical_source or path in seen_paths:
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                    "output publication receipt path is duplicated or aliases the source",
                )
            seen_paths.add(path)
            size_bytes = receipt.get("sizeBytes")
            digest = receipt.get("sha256")
            if (
                isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or size_bytes <= 0
                or path.stat().st_size != size_bytes
                or not isinstance(digest, str)
                or self._artifact_sha256(path) != digest
            ):
                raise WorkerError(
                    "OUTPUT_PUBLICATION_ARTIFACT_CHANGED",
                    "published customer artifact no longer matches its receipt",
                )
            source_integrity = receipt.get("sourceIntegrity")
            publication = receipt.get("publication")
            if (
                not isinstance(source_integrity, MappingABC)
                or source_integrity.get("unchanged") is not True
                or source_integrity.get("sourceSha256") != source_sha256
                or not isinstance(publication, MappingABC)
                or publication.get("atomic") is not True
                or publication.get("noReplace") is not True
                or publication.get("sourceMediaImmutable") is not True
            ):
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                    "customer artifact receipt lacks source and publication evidence",
                )
            artifact_type = receipt.get("artifactType")
            subtitle_format = receipt.get("subtitleFormat")
            delivery_mode = receipt.get("deliveryMode")
            visual_hash = receipt.get("visualQaEvidenceSha256")
            if artifact_type == "subtitled-media":
                if not isinstance(visual_hash, str) or len(visual_hash) != 64:
                    raise WorkerError(
                        "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                        "published media lacks representative-frame QA evidence",
                    )
            elif visual_hash is not None:
                raise WorkerError(
                    "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                    "sidecar receipt must not claim media visual-QA evidence",
                )
            actual_receipts.append(
                (
                    str(artifact_type),
                    (
                        str(subtitle_format)
                        if subtitle_format is not None
                        else None
                    ),
                    str(delivery_mode) if delivery_mode is not None else None,
                )
            )
        if sorted(actual_receipts) != sorted(expected_receipts):
            raise WorkerError(
                "OUTPUT_PUBLICATION_MANIFEST_STALE",
                "published customer artifacts do not match the output recipe",
            )

    @staticmethod
    def _expected_output_publication_receipts(
        recipe: Any,
    ) -> list[tuple[str, str | None, str | None]]:
        payload = recipe.canonical_dict()
        delivery = payload["delivery"]
        formats = [
            value
            for value in delivery["formats"]
            if value in {"srt", "webvtt", "ass"}
        ]
        expected = [
            ("subtitle-sidecar", value, "sidecar") for value in formats
        ]
        expected.extend(
            ("subtitled-media", "ass", mode)
            for mode in delivery["subtitleModes"]
            if mode in {"soft-mux", "burn-in"}
        )
        return expected

    @classmethod
    def _assert_no_private_publication_paths(cls, value: Any) -> None:
        if isinstance(value, MappingABC):
            for key, item in value.items():
                lowered = str(key).casefold()
                if (
                    lowered == "path"
                    or lowered.endswith("path")
                    or lowered.endswith("paths")
                ) and isinstance(item, str):
                    raise WorkerError(
                        "OUTPUT_PUBLICATION_MANIFEST_INVALID",
                        "internal publication evidence discloses a private path",
                    )
                cls._assert_no_private_publication_paths(item)
        elif isinstance(value, list):
            for item in value:
                cls._assert_no_private_publication_paths(item)

    def _record_output_publication(
        self,
        record: JobRecord,
        *,
        manifest_path: Path,
        manifest_payload: Mapping[str, Any],
        manifest_file_sha256: str,
    ) -> None:
        verified_manifest = self.path_policy.verify_artifact(
            manifest_path,
            record.request.output_directory,
        )
        receipts = [
            dict(receipt)
            for receipt in manifest_payload["customerArtifacts"]
        ]
        record.output_publication_status = "published"
        record.output_publication_manifest_path = str(verified_manifest)
        record.output_publication_manifest_sha256 = str(
            manifest_payload["manifestSha256"]
        )
        record.output_publication_manifest_file_sha256 = (
            manifest_file_sha256
        )
        record.output_publication_receipts = receipts
        record.output_publication_error = None
        record.output_manifest_path = str(verified_manifest)
        for receipt in receipts:
            path_text = str(receipt["path"])
            if path_text not in record.artifact_paths:
                record.artifact_paths.append(path_text)
        manifest_text = str(verified_manifest)
        if manifest_text not in record.artifact_paths:
            record.artifact_paths.append(manifest_text)

    @staticmethod
    def _exact_file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _invalidate_output_publication(
        record: JobRecord,
        error: WorkerError,
    ) -> None:
        invalidated_paths = {
            str(receipt.get("path"))
            for receipt in record.output_publication_receipts
            if isinstance(receipt.get("path"), str)
        }
        if record.output_publication_manifest_path:
            invalidated_paths.add(record.output_publication_manifest_path)
        record.artifact_paths = [
            path
            for path in record.artifact_paths
            if path not in invalidated_paths
        ]
        record.output_publication_status = "failed"
        record.output_publication_manifest_path = None
        record.output_publication_manifest_sha256 = None
        record.output_publication_manifest_file_sha256 = None
        record.output_publication_receipts = []
        record.output_publication_error = error.as_payload()
        record.output_manifest_path = None

    def _rollback_unmanifested_output_publication(
        self,
        record: JobRecord,
        manifest: OutputPublicationManifest,
        *,
        cause: WorkerError,
        remove_manifest: bool = False,
    ) -> None:
        removed = 0
        refused = 0
        source = record.request.source_path.resolve(strict=True)
        for receipt in manifest.customer_artifacts:
            try:
                path = self.path_policy.verify_artifact(
                    receipt.path,
                    record.request.output_directory,
                )
                if (
                    path == source
                    or path.stat().st_size != receipt.size_bytes
                    or self._artifact_sha256(path) != receipt.sha256
                ):
                    refused += 1
                    continue
                path.unlink()
                removed += 1
            except (OSError, WorkerError):
                refused += 1
        if remove_manifest:
            manifest_path = (
                record.request.output_directory
                / "output-publication-manifest.v1.json"
            )
            try:
                persisted = read_json_strict(manifest_path)
                if (
                    persisted.get("manifestSha256")
                    == manifest.manifest_sha256
                ):
                    manifest_path.unlink()
                else:
                    refused += 1
            except (OSError, WorkerError):
                refused += 1
        if refused:
            error = WorkerError(
                "OUTPUT_PUBLICATION_ROLLBACK_INCOMPLETE",
                "unmanifested customer artifacts could not be fully rolled back",
                details={
                    "removedCount": removed,
                    "refusedCount": refused,
                    "causeCode": cause.code,
                },
            )
            record.output_publication_status = "failed"
            record.output_publication_error = error.as_payload()
            raise error from cause

    @staticmethod
    def _publication_language(document: Mapping[str, Any]) -> str:
        value = document.get("language")
        return value if isinstance(value, str) and value.strip() else "und"

    @staticmethod
    def _publication_title(document: Mapping[str, Any]) -> str:
        value = document.get("title")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "MediaTranscribeStudio subtitles"

    @staticmethod
    def _release_stage_runner_resources(runner: Any, *, stage: str) -> None:
        release = getattr(runner, "release_resources", None)
        if not callable(release):
            return
        try:
            release()
        except Exception as exc:
            code = (
                "SEMANTIC_RESOURCE_RELEASE_FAILED"
                if stage == "semantic_processing"
                else "BUSINESS_RESOURCE_RELEASE_FAILED"
            )
            raise WorkerError(
                code,
                f"{stage} could not release its local model resources",
                details={"exceptionType": type(exc).__name__},
                retryable=True,
            ) from exc

    def _business_runner(
        self,
        record: JobRecord,
        context: AdapterContext,
    ) -> BusinessProcessingRunner:
        if self.business_runner_factory is not None:
            return self.business_runner_factory(record.request, context)
        provider = self.business_provider
        if self.business_provider_factory is not None:
            provider = self.business_provider_factory(record.request)
        if provider is None:
            provider = OllamaLocalProvider(
                LocalLLMConfig(
                    model=record.request.business_config.model,
                    endpoint=record.request.local_llm_endpoint,
                )
            )
        return BusinessProcessingRunner(
            provider=provider,
            cancellation_check=context.raise_if_cancelled,
        )

    def _semantic_runner(
        self,
        record: JobRecord,
        context: AdapterContext,
    ) -> SemanticProcessingRunner:
        if self.semantic_runner_factory is not None:
            return self.semantic_runner_factory(record.request, context)
        provider = self.semantic_provider
        if self.semantic_provider_factory is not None:
            provider = self.semantic_provider_factory(record.request)
        if provider is None:
            provider = self.business_provider
        if provider is None and self.business_provider_factory is not None:
            provider = self.business_provider_factory(record.request)
        model = self.semantic_model or record.request.local_llm_model
        if provider is None:
            provider = OllamaLocalProvider(
                LocalLLMConfig(
                    model=model,
                    endpoint=record.request.local_llm_endpoint,
                )
            )
        return SemanticProcessingRunner(
            provider=provider,
            model=model,
            cancellation_check=context.raise_if_cancelled,
        )

    def _run_semantic_processing(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
        review_queue: Mapping[str, Any],
        context: AdapterContext,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if (
            not self.semantic_required
            and record.request.local_llm_mode
            not in {"suggestion-only", "enabled"}
        ):
            record.semantic_status = "not-requested"
            return None, dict(review_queue)
        record.semantic_status = "running"
        record.semantic_error = None
        self._transition(record, JobStatus.RUNNING, "semantic_processing")
        self._emit(
            record,
            "stage.started",
            {
                "stage": "semantic_processing",
                "model": self.semantic_model or record.request.local_llm_model,
                "applicationPolicy": "suggestion-only",
                "autoApply": False,
            },
        )
        try:
            artifact_path = (
                record.request.output_directory
                / "semantic"
                / "semantic-suggestions.v1.json"
            )
            runner = self._semantic_runner(record, context)
            try:
                artifact = runner.run(document)
            except BaseException as primary_error:
                try:
                    self._release_stage_runner_resources(
                        runner,
                        stage="semantic_processing",
                    )
                except Exception as release_error:
                    primary_error.add_note(
                        "semantic_processing resource release also failed: "
                        f"{type(release_error).__name__}"
                    )
                raise
            self._release_stage_runner_resources(
                runner,
                stage="semantic_processing",
            )
            queue = attach_semantic_suggestions_to_review(
                document,
                review_queue,
                artifact,
                artifact_path=str(artifact_path),
            )
            record.semantic_status = str(artifact["status"])
            record.semantic_artifact_path = str(artifact_path)
            provider = artifact.get("provider")
            record.semantic_provenance = {
                "model": artifact["model"],
                "promptVersion": artifact["promptVersion"],
                "provider": dict(provider) if isinstance(provider, Mapping) else {},
                "applicationPolicy": artifact["applicationPolicy"],
                "requiresHumanApproval": artifact["requiresHumanApproval"],
            }
            if artifact["status"] in {"partial", "failed"}:
                record.semantic_error = {
                    "code": "SEMANTIC_PROCESSING_INCOMPLETE",
                    "message": (
                        "local semantic processing did not complete for every segment"
                    ),
                    "retryable": True,
                    "details": {
                        "rejectionCount": artifact["metrics"]["rejectionCount"],
                        "failureCount": artifact["metrics"]["failureCount"],
                        "unresolvedSegmentCount": artifact["metrics"][
                            "unresolvedSegmentCount"
                        ],
                    },
                }
            self._emit(
                record,
                "stage.completed",
                {
                    "stage": "semantic_processing",
                    "status": artifact["status"],
                    "suggestionCount": artifact["metrics"]["suggestionCount"],
                    "rejectionCount": artifact["metrics"]["rejectionCount"],
                    "failureCount": artifact["metrics"]["failureCount"],
                    "autoAppliedCount": 0,
                },
            )
            return artifact, queue
        except JobCancelled:
            record.semantic_status = "cancelled"
            raise
        except WorkerError as exc:
            record.semantic_status = "failed"
            record.semantic_error = exc.as_payload()
            raise
        except Exception as exc:
            error = WorkerError(
                "SEMANTIC_PROCESSING_FAILED",
                "semantic processing failed closed",
                details={"exceptionType": type(exc).__name__},
            )
            record.semantic_status = "failed"
            record.semantic_error = error.as_payload()
            raise error from exc

    def _run_business_processing(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
        context: AdapterContext,
    ) -> None:
        config = record.request.business_config
        if not config.enabled:
            record.business_status = "not-requested"
            return
        transcript_path = (
            record.request.output_directory / "transcript-document.v2.json"
        )
        before_hash = canonical_json_sha256(document)
        record.business_status = "running"
        record.business_error = None
        self._transition(record, JobStatus.RUNNING, "business_processing")
        self._emit(
            record,
            "stage.started",
            {
                "stage": "business_processing",
                "variants": config.as_dict(),
            },
        )
        try:
            runner = self._business_runner(record, context)
            try:
                paths = runner.run(
                    document,
                    output_directory=record.request.output_directory,
                    config=config,
                )
            except BaseException as primary_error:
                try:
                    self._release_stage_runner_resources(
                        runner,
                        stage="business_processing",
                    )
                except Exception as release_error:
                    primary_error.add_note(
                        "business_processing resource release also failed: "
                        f"{type(release_error).__name__}"
                    )
                raise
            self._release_stage_runner_resources(
                runner,
                stage="business_processing",
            )
            context.raise_if_cancelled()
            if canonical_json_sha256(document) != before_hash:
                raise WorkerError(
                    "BUSINESS_TRANSCRIPT_MUTATED",
                    "business processing mutated the in-memory transcript",
                )
            after_document = read_json_strict(transcript_path)
            after_hash = canonical_json_sha256(after_document)
            if after_hash != before_hash:
                raise WorkerError(
                    "BUSINESS_TRANSCRIPT_MUTATED",
                    "business processing changed transcript-document.v2.json",
                )
            verified_paths = [
                self.path_policy.verify_artifact(
                    Path(path),
                    record.request.output_directory,
                )
                for path in paths
            ]
            record.business_artifact_paths = [str(path) for path in verified_paths]
            manifest = next(
                (
                    path
                    for path in verified_paths
                    if path.name == "business-manifest.v1.json"
                ),
                None,
            )
            record.business_manifest_path = (
                str(manifest) if manifest is not None else None
            )
            if manifest is not None:
                manifest_value = read_json_strict(manifest)
                if isinstance(manifest_value.get("config"), MappingABC):
                    record.business_provenance = dict(manifest_value["config"])
            for path in verified_paths:
                if path == manifest:
                    continue
                try:
                    artifact_value = read_json_strict(path)
                except WorkerError:
                    continue
                if not isinstance(artifact_value, MappingABC):
                    continue
                provenance_keys = {
                    "schemaVersion",
                    "variant",
                    "inputHash",
                    "model",
                    "promptVersion",
                    "provider",
                }
                provenance = {
                    key: artifact_value[key]
                    for key in provenance_keys
                    if key in artifact_value
                }
                if provenance:
                    record.business_provenance = {
                        **(record.business_provenance or {}),
                        **provenance,
                    }
                    break
            record.business_status = "completed"
            for path in verified_paths:
                artifact_type = (
                    "business-manifest-v1"
                    if path.name == "business-manifest.v1.json"
                    else "business-variant-v1"
                )
                self._emit(
                    record,
                    "artifact.created",
                    {
                        "artifactType": artifact_type,
                        "path": str(path),
                        "sha256": self._artifact_sha256(path),
                    },
                )
                if str(path) not in record.artifact_paths:
                    record.artifact_paths.append(str(path))
        except JobCancelled:
            record.business_status = "cancelled"
            raise
        except WorkerError as exc:
            record.business_status = "failed"
            record.business_error = exc.as_payload()
            raise
        except Exception as exc:
            record.business_status = "failed"
            error = WorkerError(
                "BUSINESS_PROCESSING_FAILED",
                "business processing failed closed",
                details={"exceptionType": type(exc).__name__},
            )
            record.business_error = error.as_payload()
            raise error from exc

    def _persist_final_adjudicated_transcript(
        self,
        record: JobRecord,
        document: Mapping[str, Any],
        review_queue: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if record.semantic_status == "not-requested":
            return None
        if (
            record.semantic_status != "completed"
            or record.semantic_artifact_path is None
        ):
            raise WorkerError(
                "FINAL_ADJUDICATION_SEMANTIC_INCOMPLETE",
                "final adjudication requires completed semantic processing",
                details={"semanticStatus": record.semantic_status},
            )
        semantic_path = Path(record.semantic_artifact_path)
        semantic_artifact = read_json_strict(semantic_path)
        final_path = (
            record.request.output_directory
            / "final-adjudicated-transcript.v1.json"
        )
        created = False
        if final_path.exists():
            artifact = validate_final_adjudicated_transcript(
                read_json_strict(final_path),
                expected_document=document,
                expected_review_queue=review_queue,
                expected_semantic_artifact=semantic_artifact,
            )
        else:
            artifact = build_final_adjudicated_transcript(
                document,
                review_queue,
                semantic_artifact,
            )
            atomic_write_json_no_replace(final_path, artifact)
            created = True
        final_text = str(final_path)
        if final_text not in record.artifact_paths:
            record.artifact_paths.append(final_text)
        if created:
            self._emit(
                record,
                "artifact.created",
                {
                    "artifactType": "final-adjudicated-transcript-v1",
                    "path": final_text,
                    "sha256": canonical_json_sha256(artifact),
                    "acceptanceSubject": artifact["acceptanceSubject"],
                },
            )
        return artifact

    def _heartbeat_loop(
        self,
        record: JobRecord,
        stop: threading.Event,
        started: float,
    ) -> None:
        while not stop.wait(self.heartbeat_interval_seconds):
            with record.lock:
                if record.status in _TERMINAL:
                    return
                stage = record.stage
                status = record.status.value
            self._emit(
                record,
                "stage.progress",
                {
                    "stage": stage,
                    "kind": "heartbeat",
                    "status": status,
                    "elapsedSeconds": round(
                        max(0.0, time.monotonic() - started),
                        3,
                    ),
                    "intervalSeconds": self.heartbeat_interval_seconds,
                },
            )

    def _run_job(self, record: JobRecord) -> None:
        heartbeat_stop = threading.Event()
        started = time.monotonic()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(record, heartbeat_stop, started),
            name=f"mts-heartbeat-{record.request.job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            self._run_job_body(record)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=self.heartbeat_interval_seconds)

    def _run_job_body(self, record: JobRecord) -> None:
        context = AdapterContext(
            job_id=record.request.job_id,
            output_directory=record.request.output_directory,
            cancellation=record.cancellation,
        )
        try:
            context.raise_if_cancelled()
            self._probe_source_media(record, context)
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
            try:
                raw_result = self.transcription_adapter.transcribe(
                    record.request,
                    context,
                )
            except WorkerError as exc:
                if exc.code in {
                    "NO_SPEECH_DETECTED",
                    "NO_TRANSCRIBABLE_SPEECH",
                }:
                    self._complete_no_speech(record, exc, context)
                    return
                raise
            result = (
                raw_result
                if isinstance(raw_result, TranscriptionResult)
                else TranscriptionResult.from_mapping(raw_result)
            )
            if result.voice_activity is not None:
                self._persist_voice_activity(
                    record,
                    result.voice_activity,
                    context,
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
                language=result.language,
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
            semantic_artifact, review_queue = self._run_semantic_processing(
                record,
                document,
                review_queue,
                context,
            )
            self._transition(record, JobStatus.RUNNING, "validation")
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
            transaction_updates = {
                transcript_path: document,
                review_path: review_queue,
            }
            if semantic_artifact is not None:
                if record.semantic_artifact_path is None:
                    raise WorkerError(
                        "SEMANTIC_ARTIFACT_INVALID",
                        "semantic artifact path was not registered",
                    )
                transaction_updates[
                    Path(record.semantic_artifact_path)
                ] = semantic_artifact
            atomic_write_json_transaction(
                transaction_updates,
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
            if semantic_artifact is not None:
                assert record.semantic_artifact_path is not None
                record.artifact_paths.append(record.semantic_artifact_path)
                self._emit(
                    record,
                    "artifact.created",
                    {
                        "artifactType": "semantic-suggestions-v1",
                        "path": record.semantic_artifact_path,
                        "sha256": canonical_json_sha256(
                            semantic_artifact
                        ),
                        "status": semantic_artifact["status"],
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

            self._persist_final_adjudicated_transcript(
                record,
                document,
                review_queue,
            )
            self._run_business_processing(record, document, context)
            self._ensure_output_execution_plans(record, document)
            self._execute_transcript_exports(record, document, context)
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
            self._execute_output_publication(record, document, context)
            context.raise_if_cancelled()
            self._transition(record, JobStatus.COMPLETED, "completed")
            self._emit(
                record,
                "job.completed",
                {
                    "status": "completed",
                    "artifactPaths": list(record.artifact_paths),
                    "business": self._business_event_payload(record),
                    "semantic": self._semantic_event_payload(record),
                    "outputPublication": (
                        self._output_publication_event_payload(record)
                    ),
                },
            )
        except JobCancelled:
            self._finish_cancelled(record)
        except WorkerError as exc:
            self._finish_failed(record, exc)
        except Exception as exc:
            _LOGGER.exception(
                "unexpected worker job failure jobId=%s stage=%s exceptionType=%s",
                record.request.job_id,
                record.stage,
                type(exc).__name__,
            )
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
        typed_artifacts = (
            list(result.artifacts)
            if result.artifacts
            else [
                RenderArtifact(
                    artifact_type=self._infer_render_artifact_type(path),
                    path=path,
                )
                for path in result.artifact_paths
            ]
        )
        declared_paths = {artifact.path for artifact in typed_artifacts}
        required_artifacts = (
            RenderArtifact(
                artifact_type="pdf-quality-report-v1",
                path=result.quality_report_path,
            ),
            RenderArtifact(
                artifact_type="pdf-render-manifest-v1",
                path=result.render_manifest_path,
            ),
        )
        for artifact in required_artifacts:
            if artifact.path not in declared_paths:
                typed_artifacts.append(artifact)
                declared_paths.add(artifact.path)

        verified_artifacts: list[tuple[str, Path]] = []
        verified_paths: set[Path] = set()
        for artifact in typed_artifacts:
            verified_path = self.path_policy.verify_artifact(
                artifact.path,
                record.request.output_directory,
            )
            if verified_path in verified_paths:
                continue
            verified_paths.add(verified_path)
            verified_artifacts.append((artifact.artifact_type, verified_path))

        quality_report_path = self.path_policy.verify_artifact(
            result.quality_report_path,
            record.request.output_directory,
        )
        render_manifest_path = self.path_policy.verify_artifact(
            result.render_manifest_path,
            record.request.output_directory,
        )
        record.template_hash = result.template_hash
        record.renderer_version = result.renderer_version
        record.quality_status = result.quality_status
        record.quality_report_path = str(quality_report_path)
        record.render_manifest_path = str(render_manifest_path)
        for artifact_type, path in verified_artifacts:
            text = str(path)
            if text not in record.artifact_paths:
                record.artifact_paths.append(text)
            self._emit(
                record,
                "artifact.created",
                {
                    "artifactType": artifact_type,
                    "path": text,
                    "sha256": self._artifact_sha256(path),
                },
            )

    @staticmethod
    def _infer_render_artifact_type(path: Path) -> str:
        name = path.name.casefold()
        suffix = path.suffix.casefold()
        if suffix == ".pdf":
            return "pdf"
        if "quality-report" in name:
            return "pdf-quality-report-v1"
        if "render-manifest" in name or name == "manifest.json":
            return "pdf-render-manifest-v1"
        if "repair-queue" in name:
            return "pdf-repair-queue-v1"
        if "contact-sheet" in name:
            return "pdf-contact-sheet"
        if "report-document" in name:
            return "pdf-report-document-v1"
        if suffix in {".html", ".xhtml"}:
            return "pdf-canonical-xhtml"
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return "pdf-page-evidence"
        return "pdf-render-artifact"

    @staticmethod
    def _artifact_sha256(path: Path) -> str:
        if path.suffix.casefold() == ".json":
            return canonical_json_sha256(read_json_strict(path))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _persist_voice_activity(
        self,
        record: JobRecord,
        value: Mapping[str, Any],
        context: AdapterContext,
    ) -> Path:
        context.raise_if_cancelled()
        normalized = validate_voice_activity(value)
        if normalized["jobId"] != record.request.job_id:
            raise WorkerError(
                "VOICE_ACTIVITY_INVALID",
                "voice activity jobId does not match the active job",
            )
        expected_source_sha256 = (
            record.media_probe_artifact.source_sha256
            if record.media_probe_artifact is not None
            else self._artifact_sha256(record.request.source_path)
        )
        if normalized["sourceSha256"] != expected_source_sha256:
            raise WorkerError(
                "VOICE_ACTIVITY_INVALID",
                "voice activity source hash does not match the input media",
            )
        path = record.request.output_directory / "voice-activity.v1.json"
        atomic_write_json(path, normalized)
        verified_path = self.path_policy.verify_artifact(
            path,
            record.request.output_directory,
        )
        record.voice_activity_artifact_path = str(verified_path)
        if str(verified_path) not in record.artifact_paths:
            record.artifact_paths.append(str(verified_path))
        self._emit(
            record,
            "artifact.created",
            {
                "artifactType": "voice-activity-v1",
                "path": str(verified_path),
                "sha256": canonical_json_sha256(normalized),
            },
        )
        return verified_path

    def _complete_no_speech(
        self,
        record: JobRecord,
        error: WorkerError,
        context: AdapterContext,
    ) -> None:
        raw_voice_activity = error.details.get("voiceActivity")
        if not isinstance(raw_voice_activity, MappingABC):
            raise WorkerError(
                "VOICE_ACTIVITY_INVALID",
                "no-speech completion requires voice activity evidence",
            )
        normalized = validate_voice_activity(raw_voice_activity)
        if normalized["hasTranscribableSpeech"] is not False:
            raise WorkerError(
                "VOICE_ACTIVITY_INVALID",
                "no-speech completion cannot claim transcribable speech",
            )
        self._persist_voice_activity(record, normalized, context)
        with record.lock:
            if record.business_status in {"pending", "running"}:
                record.business_status = "not-applicable-no-speech"
            if record.semantic_status in {"pending", "running"}:
                record.semantic_status = "not-applicable-no-speech"
            record.quality_status = "no-transcribable-speech"
            if record.request.output_recipe is not None:
                record.output_publication_status = (
                    "not-applicable-no-speech"
                )
        self._transition(
            record,
            JobStatus.COMPLETED,
            "completed_no_speech",
        )
        self._emit(
            record,
            "stage.completed",
            {
                "stage": "transcription",
                "outcome": normalized["classification"],
                "hasTranscribableSpeech": False,
            },
        )
        self._emit(
            record,
            "job.completed",
            {
                "status": "completed",
                "disposition": normalized["classification"],
                "hasTranscribableSpeech": False,
                "artifactPaths": list(record.artifact_paths),
                "business": self._business_event_payload(record),
                "semantic": self._semantic_event_payload(record),
                "outputPublication": (
                    self._output_publication_event_payload(record)
                ),
            },
        )

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
            if record.business_status == "running":
                record.business_status = "failed"
                record.business_error = error.as_payload()
            if record.semantic_status == "running":
                record.semantic_status = "failed"
                record.semantic_error = error.as_payload()
            record.quality_status = "failed"
        self._transition(record, JobStatus.FAILED, "failed")
        self._emit(
            record,
            "job.failed",
            {
                **error.as_payload(),
                "business": self._business_event_payload(record),
                "semantic": self._semantic_event_payload(record),
                "outputPublication": self._output_publication_event_payload(
                    record
                ),
            },
        )

    def _finish_cancelled(self, record: JobRecord) -> None:
        with record.lock:
            if record.status in {
                JobStatus.CANCELLED,
                JobStatus.COMPLETED,
                JobStatus.REVIEW_REQUIRED,
                JobStatus.FAILED,
            }:
                return
            if record.business_status == "running":
                record.business_status = "cancelled"
            if record.semantic_status == "running":
                record.semantic_status = "cancelled"
            record.quality_status = "cancelled"
        self._transition(record, JobStatus.CANCELLED, "cancelled")
        self._emit(
            record,
            "job.cancelled",
            {
                "status": "cancelled",
                "business": self._business_event_payload(record),
                "semantic": self._semantic_event_payload(record),
                "outputPublication": self._output_publication_event_payload(
                    record
                ),
            },
        )
        with self._lock:
            self._active_outputs.pop(record.request.output_directory, None)
        self._release_capacity(record)

    @staticmethod
    def _business_event_payload(record: JobRecord) -> dict[str, Any]:
        return {
            "status": record.business_status,
            "config": record.request.business_config.as_dict(),
            "manifestPath": record.business_manifest_path,
            "artifactPaths": list(record.business_artifact_paths),
            "provenance": (
                dict(record.business_provenance)
                if record.business_provenance
                else None
            ),
            "error": (
                dict(record.business_error)
                if record.business_error
                else None
            ),
        }

    @staticmethod
    def _semantic_event_payload(record: JobRecord) -> dict[str, Any]:
        return {
            "required": record.semantic_status != "not-requested",
            "status": record.semantic_status,
            "artifactPath": record.semantic_artifact_path,
            "provenance": (
                dict(record.semantic_provenance)
                if record.semantic_provenance
                else None
            ),
            "error": (
                dict(record.semantic_error)
                if record.semantic_error
                else None
            ),
            "autoApply": False,
        }

    @staticmethod
    def _output_publication_event_payload(
        record: JobRecord,
    ) -> dict[str, Any]:
        return {
            "status": record.output_publication_status,
            "manifestPath": record.output_publication_manifest_path,
            "manifestSha256": record.output_publication_manifest_sha256,
            "manifestFileSha256": (
                record.output_publication_manifest_file_sha256
            ),
            "customerArtifacts": [
                dict(receipt)
                for receipt in record.output_publication_receipts
            ],
            "error": (
                dict(record.output_publication_error)
                if record.output_publication_error
                else None
            ),
        }

    def _emit(
        self, record: JobRecord, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        with self._event_emit_lock:
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
                "language": record.request.language,
                "outputCustomization": (
                    {
                        "sha256": (
                            record.request.output_recipe.deterministic_hash()
                        ),
                        "recipe": (
                            record.request.output_recipe.canonical_dict()
                        ),
                    }
                    if record.request.output_recipe is not None
                    else None
                ),
                "business": {
                    "status": record.business_status,
                    "config": record.request.business_config.as_dict(),
                    "manifestPath": record.business_manifest_path,
                    "artifactPaths": list(record.business_artifact_paths),
                    "provenance": (
                        dict(record.business_provenance)
                        if record.business_provenance
                        else None
                    ),
                    "error": (
                        dict(record.business_error)
                        if record.business_error
                        else None
                    ),
                },
                "semantic": self._semantic_event_payload(record),
                "documentHash": record.document_hash,
                "templateHash": record.template_hash,
                "rendererVersion": record.renderer_version,
                "qualityStatus": record.quality_status,
                "qualityReportPath": record.quality_report_path,
                "renderManifestPath": record.render_manifest_path,
                "reviewQueuePath": record.review_queue_path,
                "reviewOpenCount": record.review_open_count,
                "pipelineMetricsPath": record.pipeline_metrics_path,
                "mediaProbeArtifactPath": record.media_probe_artifact_path,
                "voiceActivityArtifactPath": (
                    record.voice_activity_artifact_path
                ),
                "outputPlanPaths": list(record.output_plan_paths),
                "outputPlanHashes": list(record.output_plan_hashes),
                "outputManifestPath": record.output_manifest_path,
                "outputPublication": self._output_publication_event_payload(
                    record
                ),
                "transcriptExports": [
                    dict(receipt)
                    for receipt in record.transcript_export_receipts
                ],
                "artifactPaths": list(record.artifact_paths),
                "error": dict(record.error) if record.error else None,
            }
        atomic_write_json(
            record.request.output_directory / "checkpoint.v2.json",
            checkpoint,
        )
