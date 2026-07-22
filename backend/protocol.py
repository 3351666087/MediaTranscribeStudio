"""Typed line-delimited JSON command protocol."""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Mapping
from typing import Any, TextIO

from .documents import utc_now
from .errors import WorkerError, invalid_request
from .models import PROTOCOL_VERSION, validate_job_id
from .service import WorkerService


class JsonlEmitter:
    """Serialize each object as exactly one UTF-8 JSON line."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.lock = threading.Lock()

    def emit(self, value: Mapping[str, Any]) -> None:
        line = json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.lock:
            self.stream.write(line + "\n")
            self.stream.flush()


class WorkerProtocol:
    def __init__(
        self,
        service: WorkerService,
        emitter: JsonlEmitter,
        *,
        max_line_bytes: int = 1024 * 1024,
    ) -> None:
        if max_line_bytes < 1024:
            raise ValueError("max_line_bytes must be at least 1024")
        self.service = service
        self.emitter = emitter
        self.max_line_bytes = max_line_bytes
        self.shutdown_requested = False

    def handle_line(self, line: str | bytes) -> None:
        request_id = "unknown"
        try:
            if isinstance(line, bytes):
                if len(line) > self.max_line_bytes:
                    raise WorkerError(
                        "LINE_TOO_LARGE",
                        "JSONL command exceeds the configured byte limit",
                    )
                try:
                    text = line.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise WorkerError(
                        "INVALID_UTF8", "JSONL command is not valid UTF-8"
                    ) from exc
            else:
                text = line
                if len(text.encode("utf-8")) > self.max_line_bytes:
                    raise WorkerError(
                        "LINE_TOO_LARGE",
                        "JSONL command exceeds the configured byte limit",
                    )
            if not text.strip():
                raise invalid_request("empty JSONL commands are not allowed")
            try:
                command = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WorkerError("MALFORMED_JSON", "command is not valid JSON") from exc
            if not isinstance(command, Mapping):
                raise invalid_request("command root must be an object")
            request_id = str(command.get("requestId") or "").strip() or "unknown"
            self._validate_envelope(command)
            command_type = command["type"]
            payload = command["payload"]
            if command_type == "job.start":
                request = self.service.parse_start_payload(payload)
                record = self.service.register(request)
                self._accepted(
                    request_id,
                    {
                        "jobId": request.job_id,
                        "status": record.status.value,
                    },
                )
                self.service.launch(request.job_id)
            elif command_type == "job.cancel":
                job_id = self._job_id_from_payload(payload)
                snapshot = self.service.cancel(job_id)
                self._accepted(request_id, snapshot)
            elif command_type == "job.status":
                job_id = self._job_id_from_payload(payload)
                self._completed(request_id, self.service.status(job_id))
            elif command_type == "review.queue":
                job_id = self._job_id_from_payload(payload)
                self._completed(request_id, self.service.review_queue(job_id))
            elif command_type == "review.submit":
                job_id = self._validate_mutation_payload(
                    payload,
                    command_type=command_type,
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
                self._completed(
                    request_id,
                    self.service.submit_review(job_id, payload),
                )
            elif command_type == "speaker.rename":
                job_id = self._validate_mutation_payload(
                    payload,
                    command_type=command_type,
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
                self._completed(
                    request_id,
                    self.service.rename_job_speaker(job_id, payload),
                )
            elif command_type == "speaker.merge":
                job_id = self._validate_mutation_payload(
                    payload,
                    command_type=command_type,
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
                self._completed(
                    request_id,
                    self.service.merge_job_speakers(job_id, payload),
                )
            elif command_type == "speaker.split":
                job_id = self._validate_mutation_payload(
                    payload,
                    command_type=command_type,
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
                self._completed(
                    request_id,
                    self.service.split_job_speaker(job_id, payload),
                )
            elif command_type == "suggestion.accept":
                job_id = self._validate_mutation_payload(
                    payload,
                    command_type=command_type,
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
                self._completed(
                    request_id,
                    self.service.accept_suggestion(job_id, payload),
                )
            elif command_type == "suggestion.reject":
                job_id = self._validate_mutation_payload(
                    payload,
                    command_type=command_type,
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
                self._completed(
                    request_id,
                    self.service.reject_suggestion(job_id, payload),
                )
            elif command_type == "job.resume":
                job_id = self._job_id_from_payload(payload)
                self._accepted(request_id, self.service.resume(job_id))
            elif command_type == "job.rerender":
                job_id = self._job_id_from_payload(payload)
                self._accepted(request_id, self.service.rerender(job_id))
            elif command_type == "worker.health":
                self._require_empty_payload(payload, command_type)
                self._completed(request_id, self.service.health())
            elif command_type == "worker.shutdown":
                self._require_empty_payload(payload, command_type)
                self._completed(
                    request_id,
                    {
                        "status": "shutdown-requested",
                    },
                )
                self.shutdown_requested = True
            else:
                raise WorkerError(
                    "UNKNOWN_COMMAND",
                    "unsupported command type",
                    details={"type": command_type},
                )
        except WorkerError as exc:
            self._rejected(request_id, exc)
        except Exception as exc:
            self._rejected(
                request_id,
                WorkerError(
                    "INTERNAL_PROTOCOL_ERROR",
                    "protocol failed closed",
                    details={"exceptionType": type(exc).__name__},
                ),
            )

    @staticmethod
    def _validate_envelope(command: Mapping[str, Any]) -> None:
        allowed = {"schemaVersion", "requestId", "type", "payload"}
        if set(command) != allowed:
            raise invalid_request(
                "command must contain exactly schemaVersion, requestId, type, and payload"
            )
        if command.get("schemaVersion") != PROTOCOL_VERSION:
            raise WorkerError(
                "UNSUPPORTED_SCHEMA_VERSION",
                f"schemaVersion must be {PROTOCOL_VERSION}",
            )
        request_id = command.get("requestId")
        if not isinstance(request_id, str) or not request_id.strip():
            raise invalid_request("requestId must be a non-empty string")
        if len(request_id) > 160:
            raise invalid_request("requestId exceeds 160 characters")
        if not isinstance(command.get("type"), str):
            raise invalid_request("type must be a string")
        if not isinstance(command.get("payload"), Mapping):
            raise invalid_request("payload must be an object")

    @staticmethod
    def _job_id_from_payload(payload: Mapping[str, Any]) -> str:
        if set(payload) != {"jobId"}:
            raise invalid_request("payload must contain exactly jobId")
        return validate_job_id(payload.get("jobId"))

    @staticmethod
    def _validate_mutation_payload(
        payload: Mapping[str, Any],
        *,
        command_type: str,
        required: set[str],
        optional: set[str] | None = None,
    ) -> str:
        optional_fields = optional or set()
        actual = set(payload)
        missing = sorted(required - actual)
        unknown = sorted(actual - required - optional_fields)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown fields: {', '.join(unknown)}")
            raise invalid_request(
                f"{command_type} payload has invalid fields ({'; '.join(details)})"
            )
        return validate_job_id(payload.get("jobId"))

    @staticmethod
    def _require_empty_payload(
        payload: Mapping[str, Any],
        command_type: str,
    ) -> None:
        if payload:
            raise invalid_request(
                f"{command_type} payload must be an empty object"
            )

    def _accepted(self, request_id: str, payload: Mapping[str, Any]) -> None:
        self._response("command.accepted", request_id, payload)

    def _completed(self, request_id: str, payload: Mapping[str, Any]) -> None:
        self._response("command.completed", request_id, payload)

    def _rejected(self, request_id: str, error: WorkerError) -> None:
        self._response("command.rejected", request_id, error.as_payload())

    def _response(
        self,
        response_type: str,
        request_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.emitter.emit(
            {
                "schemaVersion": PROTOCOL_VERSION,
                "requestId": request_id,
                "timestamp": utc_now(),
                "type": response_type,
                "payload": dict(payload),
            }
        )


def run_jsonl_loop(
    *,
    input_stream: TextIO,
    protocol: WorkerProtocol,
    service: WorkerService,
) -> None:
    try:
        binary = getattr(input_stream, "buffer", None)
        iterator = binary if binary is not None else input_stream
        for line in iterator:
            protocol.handle_line(line)
            if protocol.shutdown_requested:
                break
    finally:
        service.shutdown(cancel=True, wait=True)
