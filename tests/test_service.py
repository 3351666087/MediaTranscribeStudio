from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from backend import (
    JobCancelled,
    JsonlEmitter,
    LocalLLMProvider,
    MappingLocalLLMProvider,
    PathPolicy,
    WorkerError,
    WorkerProtocol,
    WorkerService,
)
from test_worker_support import (
    FakeDynamicRenderer,
    FakeTranscriptionAdapter,
    result_mapping,
)
from test_output_recipe import recipe_payload


def _service(
    root: Path,
    *,
    adapter: Any,
    provider: LocalLLMProvider | None = None,
    runner_factory: Any | None = None,
    renderer: Any | None = None,
    event_sink: Any | None = None,
) -> WorkerService:
    input_root = root / "input"
    output_root = root / "output"
    input_root.mkdir()
    output_root.mkdir()
    (input_root / "source.wav").write_bytes(b"fixture")
    return WorkerService(
        path_policy=PathPolicy(
            allowed_input_roots=[input_root],
            allowed_output_root=output_root,
        ),
        transcription_adapter=adapter,
        renderer_adapter=renderer,
        event_sink=event_sink,
        max_workers=1,
        business_provider=provider,
        business_runner_factory=runner_factory,
    )


def _translation_response(source: str = "你好") -> dict[str, Any]:
    return {
        "id": "segment-0001",
        "speakerId": "speaker-1",
        "startMs": 0,
        "endMs": 1000,
        "sourceTextHash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "text": "Hello",
        "language": "en",
    }


def test_start_payload_parses_business_variants_and_loopback_policy() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
        )
        request = service.parse_start_payload(
            {
                "jobId": "business-parse",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "language": "ja-JP",
                "localLlmMode": "business",
                "localLlmModel": "qwen3.5:4b",
                "localLlmEndpoint": "http://127.0.0.1:11434",
                "localLlmEndpointPolicy": "loopback-only",
                "translationTargets": ["en"],
                "polish": True,
                "summary": True,
                "outputLocale": "en-US",
            }
        )
        assert request.business_config.enabled
        assert request.business_config.translation_targets == ("en",)
        assert request.business_config.output_locale == "en-US"
        assert request.language == "ja-JP"
        assert request.local_llm_endpoint == "http://127.0.0.1:11434"
        service.shutdown()


def test_start_payload_accepts_strict_output_recipe_and_derives_pdf() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
        )
        request = service.parse_start_payload(
            {
                "jobId": "output-recipe-parse",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "outputCustomization": recipe_payload(),
            }
        )

        assert request.output_recipe is not None
        assert request.output_recipe.render_pdf
        assert request.render_pdf
        service.shutdown()


def test_start_payload_rejects_render_pdf_recipe_conflict() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
        )
        with pytest.raises(WorkerError) as raised:
            service.parse_start_payload(
                {
                    "jobId": "output-recipe-conflict",
                    "sourcePath": "source.wav",
                    "outputDirectory": "job",
                    "speakerCountMode": "manual",
                    "speakerCount": 1,
                    "renderPdf": False,
                    "outputCustomization": recipe_payload(),
                }
            )

        assert raised.value.code == "INVALID_REQUEST"
        assert raised.value.details["resolvedRenderPdf"] is True
        service.shutdown()


def test_protocol_job_start_exposes_requested_business_state() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(
                result_mapping(
                    1,
                    segments=[
                        {
                            **result_mapping(1)["segments"][0],
                            "rawText": "你好",
                            "normalizedText": "你好",
                            "displayText": "你好",
                        }
                    ],
                )
            ),
            provider=MappingLocalLLMProvider([_translation_response()]),
        )
        output = io.StringIO()
        protocol = WorkerProtocol(service, JsonlEmitter(output))
        protocol.handle_line(
            json.dumps(
                {
                    "schemaVersion": "1.0.0",
                    "requestId": "request-business-start",
                    "type": "job.start",
                    "payload": {
                        "jobId": "business-protocol",
                        "sourcePath": "source.wav",
                        "outputDirectory": "job",
                        "speakerCountMode": "manual",
                        "speakerCount": 1,
                        "language": "zh-CN",
                        "localLlmMode": "business",
                        "translationTargets": ["en"],
                    },
                }
            )
        )
        response = json.loads(output.getvalue().splitlines()[0])
        assert response["type"] == "command.accepted"
        assert response["payload"]["business"]["status"] == "pending"
        assert response["payload"]["business"]["config"]["translationTargets"] == [
            "en"
        ]
        assert service.wait("business-protocol", timeout=5)["status"] == "completed"
        service.shutdown()


def test_business_variants_run_after_review_and_preserve_transcript() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        segment = result_mapping(
            1,
            segments=[
                {
                    **result_mapping(1)["segments"][0],
                    "rawText": "你好",
                    "normalizedText": "你好",
                    "displayText": "你好",
                    "confidence": 0.50,
                }
            ],
        )
        provider = MappingLocalLLMProvider([_translation_response()])
        events: list[dict[str, Any]] = []
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(segment),
            provider=provider,
            renderer=FakeDynamicRenderer(),
            event_sink=events.append,
        )
        payload = {
            "jobId": "business-review",
            "sourcePath": "source.wav",
            "outputDirectory": "job",
            "speakerCountMode": "manual",
            "speakerCount": 1,
            "language": "zh-CN",
            "localLlmMode": "business",
            "translationTargets": ["en"],
        }
        first = service.start(payload)
        completed = service.wait(first["jobId"], timeout=5)
        assert completed["status"] == "review_required"

        output = root / "output" / "job"
        transcript_path = output / "transcript-document.v2.json"
        before = json.loads(transcript_path.read_text(encoding="utf-8"))
        queue = json.loads(
            (output / "review" / "review-queue.json").read_text(encoding="utf-8")
        )
        item_id = queue["items"][0]["id"]
        def simple_transaction(updates, *, journal_path):
            del journal_path
            for path, value in updates.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        with patch("backend.service.atomic_write_json_transaction", simple_transaction):
            service.submit_review(
                "business-review",
                {
                    "jobId": "business-review",
                    "itemId": item_id,
                    "action": "accept",
                    "decisionId": "decision-1",
                    "reason": "human review confirmed",
                    "evidence": ["audio:0-1000"],
                    "confidence": 0.99,
                    "audit": {"actor": "test", "source": "human"},
                },
            )
        resumed = service.resume("business-review")
        assert resumed["followupOperation"] == "resume"
        final = service.wait("business-review", timeout=5)

        assert final["status"] == "completed"
        assert final["business"]["status"] == "completed"
        assert (
            output / "business" / "translation-en.v1.json"
        ) in [Path(path) for path in final["business"]["artifactPaths"]]
        assert json.loads(transcript_path.read_text(encoding="utf-8")) == before
        checkpoint = json.loads(
            (output / "checkpoint.v2.json").read_text(encoding="utf-8")
        )
        assert checkpoint["business"]["status"] == "completed"
        assert checkpoint["business"]["manifestPath"].endswith(
            "business-manifest.v1.json"
        )
        completed_event = next(
            event for event in events if event["type"] == "job.completed"
        )
        assert (
            completed_event["payload"]["business"]["provenance"]["provider"]["id"]
            == "mapping-fixture"
        )
        service.rerender("business-review")
        rerendered = service.wait("business-review", timeout=5)
        assert rerendered["status"] == "completed"
        assert rerendered["business"]["status"] == "completed"
        assert (output / "transcript.pdf").exists()
        render_events = [
            event
            for event in events
            if event["type"] == "artifact.created"
            and str(event["payload"]["artifactType"]).startswith("pdf")
        ]
        assert {
            event["payload"]["artifactType"] for event in render_events
        } == {
            "pdf",
            "pdf-quality-report-v1",
            "pdf-render-manifest-v1",
        }
        assert all(
            len(event["payload"]["sha256"]) == 64 for event in render_events
        )
        assert all(
            Path(event["payload"]["path"]).is_file() for event in render_events
        )
        render_event_by_type = {
            event["payload"]["artifactType"]: event for event in render_events
        }
        assert render_event_by_type["pdf"]["payload"]["sha256"] == hashlib.sha256(
            (output / "transcript.pdf").read_bytes()
        ).hexdigest()
        for artifact_type in (
            "pdf-quality-report-v1",
            "pdf-render-manifest-v1",
        ):
            artifact_path = Path(
                render_event_by_type[artifact_type]["payload"]["path"]
            )
            canonical_bytes = json.dumps(
                json.loads(artifact_path.read_text(encoding="utf-8")),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            canonical_bytes = f"{canonical_bytes}\n".encode("utf-8")
            assert (
                render_event_by_type[artifact_type]["payload"]["sha256"]
                == hashlib.sha256(canonical_bytes).hexdigest()
            )

        service.shutdown()


class _CancelledBusinessRunner:
    def run(self, document, *, output_directory, config):
        del document, output_directory, config
        raise JobCancelled()


def test_business_cancellation_remains_job_cancelled() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            runner_factory=lambda request, context: _CancelledBusinessRunner(),
        )
        started = service.start(
            {
                "jobId": "business-cancel",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "language": "zh-CN",
                "localLlmMode": "business",
                "summary": True,
            }
        )
        final = service.wait(started["jobId"], timeout=5)
        assert final["status"] == "cancelled"
        assert final["business"]["status"] == "cancelled"
        service.shutdown()
