from __future__ import annotations

import io
import json
import unittest
from copy import deepcopy
from typing import Any

from backend import JsonlEmitter, WorkerProtocol, run_jsonl_loop


def decision_fields() -> dict[str, Any]:
    return {
        "decisionId": "decision-001",
        "reason": "人工复核了本地音频与相邻问答上下文。",
        "evidence": ["audio:0-1000", "context:adjacent-turns"],
        "confidence": 0.98,
        "audit": {
            "actor": "reviewer-001",
            "source": "human",
        },
    }


def command(command_type: str, payload: dict[str, Any], *, request_id: str = "req-1"):
    return {
        "schemaVersion": "1.0.0",
        "requestId": request_id,
        "type": command_type,
        "payload": payload,
    }


class ReviewProtocolServiceStub:
    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.events = events

    def _record(self, name: str, *args: Any) -> dict[str, Any]:
        self.calls.append((name, args))
        return {"method": name, "args": list(args)}

    def review_queue(self, job_id: str) -> dict[str, Any]:
        return self._record("review_queue", job_id)

    def submit_review(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._record("submit_review", job_id, deepcopy(payload))

    def rename_job_speaker(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._record("rename_job_speaker", job_id, deepcopy(payload))

    def merge_job_speakers(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._record("merge_job_speakers", job_id, deepcopy(payload))

    def split_job_speaker(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._record("split_job_speaker", job_id, deepcopy(payload))

    def accept_suggestion(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._record("accept_suggestion", job_id, deepcopy(payload))

    def reject_suggestion(
        self, job_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._record("reject_suggestion", job_id, deepcopy(payload))

    def resume(self, job_id: str) -> dict[str, Any]:
        return self._record("resume", job_id)

    def rerender(self, job_id: str) -> dict[str, Any]:
        return self._record("rerender", job_id)

    def health(self) -> dict[str, Any]:
        return self._record("health")

    def shutdown(self, *, cancel: bool, wait: bool) -> None:
        self.calls.append(("shutdown", (cancel, wait)))
        if self.events is not None:
            self.events.append("service.shutdown")


class TrackingStream(io.StringIO):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def write(self, value: str) -> int:
        self.events.append("response.write")
        return super().write(value)

    def flush(self) -> None:
        self.events.append("response.flush")
        super().flush()


class ReviewProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ReviewProtocolServiceStub()
        self.output = io.StringIO()
        self.protocol = WorkerProtocol(self.service, JsonlEmitter(self.output))

    def dispatch(self, value: dict[str, Any]) -> dict[str, Any]:
        self.protocol.handle_line(json.dumps(value, ensure_ascii=False))
        lines = self.output.getvalue().splitlines()
        self.assertTrue(lines, "protocol must emit exactly one response")
        response = json.loads(lines[-1])
        self.output.seek(0)
        self.output.truncate(0)
        return response

    def valid_payloads(self) -> dict[str, dict[str, Any]]:
        decision = decision_fields()
        return {
            "review.queue": {"jobId": "job-review"},
            "review.submit": {
                "jobId": "job-review",
                "itemId": "segment-0001:SEGMENT_LOW_CONFIDENCE",
                "action": "accept",
                **decision,
            },
            "speaker.rename": {
                "jobId": "job-review",
                "speakerId": "speaker-2",
                "name": "主持人",
                **decision,
            },
            "speaker.merge": {
                "jobId": "job-review",
                "sourceSpeakerId": "speaker-3",
                "targetSpeakerId": "speaker-2",
                **decision,
            },
            "speaker.split": {
                "jobId": "job-review",
                "sourceSpeakerId": "speaker-1",
                "segmentIds": ["segment-0002"],
                "newSpeakerName": "嘉宾",
                **decision,
            },
            "suggestion.accept": {
                "jobId": "job-review",
                "itemId": "segment-0001:SEGMENT_LOW_CONFIDENCE",
                "suggestionId": "suggestion-001",
                **decision,
            },
            "suggestion.reject": {
                "jobId": "job-review",
                "itemId": "segment-0001:SEGMENT_LOW_CONFIDENCE",
                "suggestionId": "suggestion-001",
                **decision,
            },
            "job.resume": {"jobId": "job-review"},
            "job.rerender": {"jobId": "job-review"},
            "worker.health": {},
            "worker.shutdown": {},
        }

    def test_review_commands_route_to_target_service_api(self) -> None:
        expected = {
            "review.queue": ("review_queue", "command.completed"),
            "review.submit": ("submit_review", "command.completed"),
            "speaker.rename": ("rename_job_speaker", "command.completed"),
            "speaker.merge": ("merge_job_speakers", "command.completed"),
            "speaker.split": ("split_job_speaker", "command.completed"),
            "suggestion.accept": ("accept_suggestion", "command.completed"),
            "suggestion.reject": ("reject_suggestion", "command.completed"),
            "job.resume": ("resume", "command.accepted"),
            "job.rerender": ("rerender", "command.accepted"),
            "worker.health": ("health", "command.completed"),
        }
        for index, (command_type, (method, response_type)) in enumerate(
            expected.items(), start=1
        ):
            with self.subTest(command_type=command_type):
                before = len(self.service.calls)
                response = self.dispatch(
                    command(
                        command_type,
                        deepcopy(self.valid_payloads()[command_type]),
                        request_id=f"route-{index}",
                    )
                )
                self.assertEqual(response["requestId"], f"route-{index}")
                self.assertEqual(response["type"], response_type)
                self.assertEqual(len(self.service.calls), before + 1)
                self.assertEqual(self.service.calls[-1][0], method)

    def test_mutation_payloads_reject_unknown_fields_before_service_call(self) -> None:
        mutation_types = (
            "review.submit",
            "speaker.rename",
            "speaker.merge",
            "speaker.split",
            "suggestion.accept",
            "suggestion.reject",
        )
        for command_type in mutation_types:
            with self.subTest(command_type=command_type):
                payload = deepcopy(self.valid_payloads()[command_type])
                payload["unexpected"] = "must-fail-closed"
                before = list(self.service.calls)
                response = self.dispatch(command(command_type, payload))
                self.assertEqual(response["type"], "command.rejected")
                self.assertEqual(response["payload"]["code"], "INVALID_REQUEST")
                self.assertEqual(self.service.calls, before)

    def test_manual_commands_require_decision_evidence_and_audit_fields(self) -> None:
        cases = (
            ("review.submit", "decisionId"),
            ("speaker.rename", "evidence"),
            ("speaker.merge", "audit"),
            ("speaker.split", "decisionId"),
            ("suggestion.accept", "evidence"),
            ("suggestion.reject", "audit"),
        )
        for command_type, missing in cases:
            with self.subTest(command_type=command_type, missing=missing):
                payload = deepcopy(self.valid_payloads()[command_type])
                payload.pop(missing)
                before = list(self.service.calls)
                response = self.dispatch(command(command_type, payload))
                self.assertEqual(response["type"], "command.rejected")
                self.assertEqual(response["payload"]["code"], "INVALID_REQUEST")
                self.assertEqual(self.service.calls, before)

    def test_exact_job_and_empty_payload_contracts_fail_closed(self) -> None:
        for command_type in ("review.queue", "job.resume", "job.rerender"):
            with self.subTest(command_type=command_type):
                response = self.dispatch(
                    command(
                        command_type,
                        {"jobId": "job-review", "unexpected": True},
                    )
                )
                self.assertEqual(response["type"], "command.rejected")
                self.assertEqual(response["payload"]["code"], "INVALID_REQUEST")
        for command_type in ("worker.health", "worker.shutdown"):
            with self.subTest(command_type=command_type):
                response = self.dispatch(
                    command(command_type, {"unexpected": True})
                )
                self.assertEqual(response["type"], "command.rejected")
                self.assertEqual(response["payload"]["code"], "INVALID_REQUEST")
        self.assertEqual(self.service.calls, [])

    def test_envelope_is_exact_and_versioned(self) -> None:
        extra = command("worker.health", {})
        extra["unexpected"] = True
        wrong_version = command("worker.health", {})
        wrong_version["schemaVersion"] = "2.0.0"
        for value, expected_code in (
            (extra, "INVALID_REQUEST"),
            (wrong_version, "UNSUPPORTED_SCHEMA_VERSION"),
        ):
            with self.subTest(expected_code=expected_code):
                response = self.dispatch(value)
                self.assertEqual(response["type"], "command.rejected")
                self.assertEqual(response["payload"]["code"], expected_code)
        self.assertEqual(self.service.calls, [])

    def test_shutdown_response_is_flushed_before_loop_stops_and_service_closes(self) -> None:
        events: list[str] = []
        service = ReviewProtocolServiceStub(events)
        output = TrackingStream(events)
        protocol = WorkerProtocol(service, JsonlEmitter(output))
        input_stream = io.StringIO(
            "\n".join(
                [
                    json.dumps(command("worker.shutdown", {}, request_id="shutdown")),
                    json.dumps(command("worker.health", {}, request_id="too-late")),
                ]
            )
            + "\n"
        )

        run_jsonl_loop(
            input_stream=input_stream,
            protocol=protocol,
            service=service,
        )

        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["requestId"], "shutdown")
        self.assertEqual(responses[0]["type"], "command.completed")
        self.assertTrue(protocol.shutdown_requested)
        self.assertNotIn("health", [name for name, _args in service.calls])
        self.assertEqual(service.calls[-1], ("shutdown", (True, True)))
        self.assertLess(
            events.index("response.flush"),
            events.index("service.shutdown"),
            "shutdown must happen only after the response is observable",
        )


if __name__ == "__main__":
    unittest.main()
