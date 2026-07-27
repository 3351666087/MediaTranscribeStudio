from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
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
from backend.media_probe import MediaProbe, ProcessLimits, ProcessResult
from backend.output_publication import (
    CustomerArtifactReceipt,
    OutputPublicationManifest,
)
from backend.persistence import (
    atomic_publish_json_evidence as persist_json_evidence,
)
from backend.persistence import canonical_json_sha256, sha256_file
from backend.subtitles import SubtitleFormat, SubtitleOutputMode
from backend.voice_activity import build_voice_activity
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
    media_probe: Any | None = None,
    output_publisher: Any | None = None,
    subtitle_delivery_executor: Any | None = None,
    subtitle_visual_qa_hook: Any | None = None,
    heartbeat_interval_seconds: float = 15.0,
    semantic_required: bool = False,
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
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        max_workers=1,
        business_provider=provider,
        business_runner_factory=runner_factory,
        semantic_required=semantic_required,
        media_probe=media_probe,
        output_publisher=(
            output_publisher
            if output_publisher is not None
            else _fake_output_publisher
        ),
        subtitle_delivery_executor=(
            subtitle_delivery_executor
            if subtitle_delivery_executor is not None
            else object()
        ),
        subtitle_visual_qa_hook=(
            subtitle_visual_qa_hook
            if subtitle_visual_qa_hook is not None
            else (lambda **_: {"passed": True})
        ),
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


class _RecordingProbeRunner:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def run(
        self,
        command: tuple[str, ...] | list[str],
        *,
        limits: ProcessLimits,
    ) -> ProcessResult:
        del limits
        self.order.append("media-probe")
        argv = tuple(command)
        if "-version" in argv:
            tool = Path(argv[0]).name
            return ProcessResult(
                returncode=0,
                stdout=(
                    f"{tool} version 8.0-test\n"
                    "configuration: --enable-gpl --enable-libass\n"
                ).encode(),
                stderr=b"",
                elapsed_ms=1,
            )
        if "-show_streams" in argv:
            return ProcessResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "streams": [
                            {
                                "index": 0,
                                "codec_name": "h264",
                                "codec_type": "video",
                                "width": 1920,
                                "height": 1080,
                                "pix_fmt": "yuv420p",
                                "duration": "8.0",
                                "color_transfer": "bt709",
                                "disposition": {
                                    "default": 1,
                                    "attached_pic": 0,
                                },
                            },
                            {
                                "index": 1,
                                "codec_name": "aac",
                                "codec_type": "audio",
                                "sample_rate": "48000",
                                "channels": 2,
                                "duration": "8.0",
                                "disposition": {"default": 1},
                            },
                        ],
                        "format": {
                            "format_name": "mov,mp4",
                            "format_long_name": "fixture media",
                            "duration": "8.0",
                            "bit_rate": "1000000",
                        },
                        "programs": [],
                        "chapters": [],
                    }
                ).encode(),
                stderr=b"",
                elapsed_ms=2,
            )
        return ProcessResult(
            returncode=0,
            stdout=b"",
            stderr=b"",
            elapsed_ms=3,
        )


class _RecordingTranscriptionAdapter(FakeTranscriptionAdapter):
    def __init__(self, order: list[str]) -> None:
        super().__init__(result_mapping(1))
        self.order = order

    def transcribe(self, request, context):
        self.order.append("transcription")
        return super().transcribe(request, context)


class _UnexpectedFailureTranscriptionAdapter(FakeTranscriptionAdapter):
    def transcribe(self, request, context):
        del request, context
        raise AssertionError("private diagnostic detail")


class _NoSpeechTranscriptionAdapter:
    adapter_id = "no-speech-fixture"
    version = "1"

    def transcribe(self, request, context):
        context.raise_if_cancelled()
        source_sha256 = hashlib.sha256(
            request.source_path.read_bytes()
        ).hexdigest()
        voice_activity = build_voice_activity(
            job_id=request.job_id,
            source_sha256=source_sha256,
            media_duration_ms=2_000,
            normalization_profile="mono-16khz-f32-v1",
            provider={"id": self.adapter_id, "version": self.version},
            windows=(),
            minimum_window_ms=120,
            classification="no-speech-candidates-detected",
            has_transcribable_speech=False,
        )
        raise WorkerError(
            "NO_SPEECH_DETECTED",
            "fixture found no speech",
            details={"voiceActivity": voice_activity},
        )


class _PlannedRenderer(FakeDynamicRenderer):
    def __init__(self) -> None:
        self.output_plans: list[Any] = []

    def render(self, document, request, context, *, output_plan=None):
        self.output_plans.append(output_plan)
        return super().render(document, request, context)


def _fake_output_publisher(
    recipe,
    plans,
    document,
    **kwargs,
) -> OutputPublicationManifest:
    del document, kwargs
    source = plans[0].source_path.resolve(strict=True)
    output_root = plans[0].output_directory.resolve(strict=True)
    source_sha256 = sha256_file(source)
    payload = recipe.canonical_dict()
    receipts: list[CustomerArtifactReceipt] = []
    suffixes = {"srt": ".srt", "webvtt": ".vtt", "ass": ".ass"}
    for subtitle_format in payload["delivery"]["formats"]:
        if subtitle_format not in suffixes:
            continue
        path = output_root / f"published{suffixes[subtitle_format]}"
        path.write_text(
            f"{subtitle_format} publication fixture\n",
            encoding="utf-8",
        )
        receipts.append(
            CustomerArtifactReceipt(
                artifact_type="subtitle-sidecar",
                path=path,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source_sha256=source_sha256,
                subtitle_format=SubtitleFormat(subtitle_format),
                delivery_mode=SubtitleOutputMode.SIDECAR,
            )
        )
    for plan in plans:
        if plan.delivery_mode not in {
            SubtitleOutputMode.SOFT_MUX,
            SubtitleOutputMode.BURN_IN,
        }:
            continue
        assert plan.delivery_output_path is not None
        path = plan.delivery_output_path
        path.write_bytes(
            f"{plan.delivery_mode.value} publication fixture".encode()
        )
        receipts.append(
            CustomerArtifactReceipt(
                artifact_type="subtitled-media",
                path=path,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                source_sha256=source_sha256,
                subtitle_format=SubtitleFormat.ASS,
                delivery_mode=plan.delivery_mode,
                visual_qa_evidence_sha256="b" * 64,
            )
        )
    mode_order = {
        SubtitleOutputMode.SIDECAR: 0,
        SubtitleOutputMode.SOFT_MUX: 1,
        SubtitleOutputMode.BURN_IN: 2,
    }
    return OutputPublicationManifest(
        recipe_sha256=recipe.deterministic_hash(),
        source_path=source,
        source_size_bytes=source.stat().st_size,
        source_sha256=source_sha256,
        plan_sha256=tuple(
            (
                plan.delivery_mode.value,
                plan.customization_sha256,
                plan.deterministic_hash(),
            )
            for plan in sorted(
                (item for item in plans if item.subtitle_enabled),
                key=lambda item: mode_order[item.delivery_mode],
            )
        ),
        customer_artifacts=tuple(receipts),
        internal_evidence={
            "privateAssCarrier": {
                "created": False,
                "customerArtifact": False,
                "pathDisclosed": False,
                "unpredictableName": False,
                "payloadSha256": None,
                "sizeBytes": None,
                "cleanup": "not-created",
                "reason": "public-ass-reused",
            },
            "mediaQuarantine": [],
            "customerAndInternalEvidenceSeparated": True,
            "privatePathsExcluded": True,
        },
    )


def _publication_job_payload(
    job_id: str,
    *,
    formats: list[str] | None = None,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "jobId": job_id,
        "sourcePath": "source.wav",
        "outputDirectory": "job",
        "speakerCountMode": "manual",
        "speakerCount": 1,
        "language": "zh-Hans",
        "outputCustomization": recipe_payload(
            formats=formats or ["pdf", "ass"],
            modes=modes or ["sidecar", "burn-in"],
        ),
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
                "localLlmModel": "qwen3.5:9b",
                "localLlmEndpoint": "http://127.0.0.1:11434",
                "localLlmEndpointPolicy": "loopback-only",
                "translationTargets": ["en"],
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


def test_start_payload_rejects_retired_local_model() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        service = _service(
            Path(temporary),
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
        )
        with pytest.raises(
            WorkerError,
            match="production model qwen3.5:9b",
        ):
            service.parse_start_payload(
                {
                    "jobId": "retired-model",
                    "sourcePath": "source.wav",
                    "outputDirectory": "job",
                    "speakerCountMode": "manual",
                    "speakerCount": 1,
                    "localLlmModel": "qwen3.5:4b",
                }
            )
        service.shutdown()


def test_required_semantic_stage_persists_suggestion_without_mutating_transcript() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = MappingLocalLLMProvider(
            [
                {
                    "results": [
                        {
                            "segmentId": "segment-0001",
                            "decision": "propose",
                            "confidence": 0.82,
                        }
                    ]
                },
                {
                    "results": [
                        {
                            "segmentId": "segment-0001",
                            "decision": "propose",
                            "speakerRanking": ["speaker-1"],
                            "normalizedText": "这是第1位说话人的中文原文。",
                            "textEvidenceCandidateId": "",
                            "confidence": 0.82,
                            "reasonCodes": ["PUNCTUATION_BOUNDARY"],
                            "evidenceRefs": ["segment:segment-0001"],
                        }
                    ]
                }
            ]
        )
        release_calls = 0

        def release_resources() -> None:
            nonlocal release_calls
            release_calls += 1

        provider.release_resources = release_resources  # type: ignore[method-assign]
        transcription = result_mapping(1)
        transcription["segments"][0].update(
            {
                "rawText": "这是第1位说话人的中文原文",
                "normalizedText": "这是第1位说话人的中文原文",
                "displayText": "这是第1位说话人的中文原文",
            }
        )
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(transcription),
            provider=provider,
            semantic_required=True,
        )
        started = service.start(
            {
                "jobId": "semantic-required",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "localLlmMode": "disabled",
            }
        )
        final = service.wait(started["jobId"], timeout=5)

        assert final["status"] == "review_required"
        assert final["semantic"]["required"] is True
        assert final["semantic"]["status"] == "completed"
        output = root / "output" / "job"
        artifact = json.loads(
            (
                output
                / "semantic"
                / "semantic-suggestions.v1.json"
            ).read_text(encoding="utf-8")
        )
        transcript = json.loads(
            (output / "transcript-document.v2.json").read_text(
                encoding="utf-8"
            )
        )
        queue = json.loads(
            (output / "review" / "review-queue.json").read_text(
                encoding="utf-8"
            )
        )
        assert artifact["metrics"]["textSuggestionCount"] == 1
        assert transcript["segments"][0]["rawText"] == (
            "这是第1位说话人的中文原文"
        )
        assert transcript["segments"][0]["normalizedText"] == (
            "这是第1位说话人的中文原文"
        )
        semantic_item = next(
            item
            for item in queue["items"]
            if item["reasonCode"] == "SEMANTIC_TEXT_SUGGESTION"
        )
        assert semantic_item["suggestions"][0]["proposal"] == {
            "normalizedText": "这是第1位说话人的中文原文。",
            "displayText": "这是第1位说话人的中文原文。",
        }
        assert not (
            output / "final-adjudicated-transcript.v1.json"
        ).exists()
        assert release_calls == 1
        service.shutdown()


def test_completed_semantic_job_persists_final_adjudicated_transcript() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        events: list[dict[str, Any]] = []
        provider = MappingLocalLLMProvider(
            [
                {
                    "results": [
                        {
                            "segmentId": "segment-0001",
                            "decision": "abstain",
                            "confidence": 0.9,
                        }
                    ]
                }
            ]
        )
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            provider=provider,
            semantic_required=True,
            event_sink=events.append,
        )

        started = service.start(
            {
                "jobId": "semantic-final",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "localLlmMode": "disabled",
            }
        )
        final = service.wait(started["jobId"], timeout=5)

        assert final["status"] == "completed"
        output = root / "output" / "job"
        final_path = output / "final-adjudicated-transcript.v1.json"
        artifact = json.loads(final_path.read_text(encoding="utf-8"))
        assert artifact["status"] == "adjudication-complete"
        assert artifact["review"]["openCount"] == 0
        assert artifact["semantic"]["status"] == "completed"
        assert artifact["segments"][0]["finalText"] == (
            "这是第1位说话人的中文原文。"
        )
        assert str(final_path) in final["artifactPaths"]
        created = [
            event
            for event in events
            if event["type"] == "artifact.created"
            and event["payload"]["artifactType"]
            == "final-adjudicated-transcript-v1"
        ]
        assert len(created) == 1
        assert created[0]["payload"]["sha256"] == canonical_json_sha256(
            artifact
        )
        service.shutdown()


def test_shutdown_releases_transcription_resources_once() -> None:
    class ReleasableTranscription(FakeTranscriptionAdapter):
        def __init__(self) -> None:
            super().__init__(result_mapping(1))
            self.release_calls = 0

        def release_resources(self) -> None:
            self.release_calls += 1

    with tempfile.TemporaryDirectory() as temporary:
        adapter = ReleasableTranscription()
        service = _service(Path(temporary), adapter=adapter)

        service.shutdown()
        service.shutdown()

        assert adapter.release_calls == 1


def test_long_job_emits_ordered_heartbeat_progress_until_terminal() -> None:
    class SlowTranscription(FakeTranscriptionAdapter):
        def transcribe(self, request: Any, context: Any) -> Any:
            time.sleep(0.08)
            return super().transcribe(request, context)

    with tempfile.TemporaryDirectory() as temporary:
        events: list[dict[str, Any]] = []
        service = _service(
            Path(temporary),
            adapter=SlowTranscription(result_mapping(1)),
            event_sink=events.append,
            heartbeat_interval_seconds=0.01,
        )

        started = service.start(
            {
                "jobId": "heartbeat-progress",
                "sourcePath": "source.wav",
                "outputDirectory": "heartbeat-output",
                "speakerCountMode": "manual",
                "speakerCount": 1,
            }
        )
        final = service.wait(started["jobId"], timeout=5)
        service.shutdown()

        assert final["status"] in {"completed", "review_required"}
        heartbeat_events = [
            event
            for event in events
            if event["type"] == "stage.progress"
            and event["payload"].get("kind") == "heartbeat"
        ]
        assert len(heartbeat_events) >= 2
        assert all(
            event["payload"]["intervalSeconds"] == 0.01
            for event in heartbeat_events
        )
        sequences = [event["sequence"] for event in events]
        assert sequences == list(range(len(events)))
        terminal_index = max(
            index
            for index, event in enumerate(events)
            if event["type"]
            in {"job.completed", "review.required", "job.failed"}
        )
        assert all(
            event["type"] != "stage.progress"
            for event in events[terminal_index + 1 :]
        )


def test_unexpected_job_failure_logs_traceback_but_sanitizes_public_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        events: list[dict[str, Any]] = []
        service = _service(
            root,
            adapter=_UnexpectedFailureTranscriptionAdapter(result_mapping(1)),
            event_sink=events.append,
        )

        with caplog.at_level("ERROR", logger="backend.service"):
            started = service.start(
                {
                    "jobId": "unexpected-transcription-failure",
                    "sourcePath": "source.wav",
                    "outputDirectory": "job",
                    "speakerCountMode": "manual",
                    "speakerCount": 1,
                }
            )
            final = service.wait(started["jobId"], timeout=5)

        assert final["status"] == "failed"
        assert final["error"] == {
            "code": "INTERNAL_ERROR",
            "message": "worker failed closed due to an unexpected internal error",
            "retryable": False,
            "details": {"exceptionType": "AssertionError"},
        }
        failed_event = next(
            event for event in events if event["type"] == "job.failed"
        )
        public_payload = json.dumps(failed_event, sort_keys=True)
        assert "private diagnostic detail" not in public_payload
        assert "service.py" not in public_payload

        diagnostic = next(
            record
            for record in caplog.records
            if record.getMessage().startswith("unexpected worker job failure")
        )
        assert diagnostic.exc_info is not None
        assert diagnostic.getMessage() == (
            "unexpected worker job failure "
            "jobId=unexpected-transcription-failure "
            "stage=transcription exceptionType=AssertionError"
        )
        assert "private diagnostic detail" in caplog.text
        assert "Traceback (most recent call last)" in caplog.text
        service.shutdown()


def test_no_speech_completes_with_voice_activity_artifact() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        events: list[dict[str, Any]] = []
        service = _service(
            root,
            adapter=_NoSpeechTranscriptionAdapter(),
            event_sink=events.append,
        )

        started = service.start(
            {
                "jobId": "no-speech-success",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "auto",
                "translationTargets": [],
            }
        )
        final = service.wait(started["jobId"], timeout=5)

        assert final["status"] == "completed"
        assert final["stage"] == "completed_no_speech"
        assert final["qualityStatus"] == "no-transcribable-speech"
        artifact_path = Path(final["voiceActivityArtifactPath"])
        assert artifact_path.is_file()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert artifact["classification"] == "no-speech-candidates-detected"
        assert artifact["hasTranscribableSpeech"] is False
        assert not (artifact_path.parent / "transcript-document.v2.json").exists()
        completed = next(
            event for event in events if event["type"] == "job.completed"
        )
        assert completed["payload"]["hasTranscribableSpeech"] is False
        assert completed["payload"]["disposition"] == (
            "no-speech-candidates-detected"
        )
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


def test_recipe_job_probes_before_transcription_and_persists_exact_plans() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        order: list[str] = []
        events: list[dict[str, Any]] = []
        renderer = _PlannedRenderer()
        probe = MediaProbe(runner=_RecordingProbeRunner(order))
        service = _service(
            root,
            adapter=_RecordingTranscriptionAdapter(order),
            renderer=renderer,
            event_sink=events.append,
            media_probe=probe,
        )
        source = root / "input" / "source.wav"
        source_hash = sha256_file(source)
        started = service.start(
            {
                "jobId": "recipe-planning",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "language": "zh-Hans",
                "outputCustomization": recipe_payload(
                    formats=[
                        "pdf",
                        "html",
                        "markdown",
                        "txt",
                        "json",
                        "srt",
                        "webvtt",
                        "ass",
                    ],
                    modes=["sidecar", "soft-mux", "burn-in"],
                ),
            }
        )
        final = service.wait(started["jobId"], timeout=5)

        assert final["status"] == "completed"
        assert order.index("media-probe") < order.index("transcription")
        assert sha256_file(source) == source_hash
        output = root / "output" / "job"
        probe_path = output / "media-probe.v1.json"
        assert probe_path.is_file()
        assert final["mediaProbeArtifactPath"] == str(probe_path.resolve())
        plan_paths = [Path(path) for path in final["outputPlanPaths"]]
        assert len(plan_paths) == 3
        assert [path.name for path in plan_paths] == [
            "01-sidecar.output-plan.v1.json",
            "02-soft-mux.output-plan.v1.json",
            "03-burn-in.output-plan.v1.json",
        ]
        plan_payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in plan_paths
        ]
        assert final["outputPlanHashes"] == [
            canonical_json_sha256(payload) for payload in plan_payloads
        ]
        assert [
            receipt["format"] for receipt in final["transcriptExports"]
        ] == ["json", "txt", "markdown", "html"]
        for receipt in final["transcriptExports"]:
            export_path = Path(receipt["path"])
            assert export_path.is_file()
            assert receipt["sha256"] == sha256_file(export_path)
            assert receipt["size"] == export_path.stat().st_size
        assert len(renderer.output_plans) == 1
        assert renderer.output_plans[0].to_dict() == plan_payloads[0]
        assert renderer.output_plans[0].deterministic_hash() == (
            final["outputPlanHashes"][0]
        )
        checkpoint = json.loads(
            (output / "checkpoint.v2.json").read_text(encoding="utf-8")
        )
        assert checkpoint["mediaProbeArtifactPath"] == str(probe_path.resolve())
        assert checkpoint["outputPlanPaths"] == final["outputPlanPaths"]
        assert checkpoint["outputPlanHashes"] == final["outputPlanHashes"]
        assert checkpoint["transcriptExports"] == final["transcriptExports"]
        publication = final["outputPublication"]
        assert publication["status"] == "published"
        manifest_path = output / "output-publication-manifest.v1.json"
        assert publication["manifestPath"] == str(manifest_path.resolve())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_body = dict(manifest)
        claimed_manifest_hash = manifest_body.pop("manifestSha256")
        assert canonical_json_sha256(manifest_body) == claimed_manifest_hash
        assert publication["manifestSha256"] == claimed_manifest_hash
        assert publication["manifestFileSha256"] == sha256_file(manifest_path)
        assert checkpoint["outputPublication"] == publication
        artifact_events = [
            event
            for event in events
            if event["type"] == "artifact.created"
        ]
        assert any(
            event["payload"]["artifactType"] == "media-probe-v1"
            and event["payload"]["sha256"] == sha256_file(probe_path)
            for event in artifact_events
        )
        assert sum(
            event["payload"]["artifactType"] == "output-execution-plan-v1"
            for event in artifact_events
        ) == 3
        assert {
            event["payload"]["artifactType"]
            for event in artifact_events
            if str(event["payload"]["artifactType"]).startswith(
                "transcript-export-"
            )
        } == {
            "transcript-export-json-v1",
            "transcript-export-txt-v1",
            "transcript-export-markdown-v1",
            "transcript-export-html-v1",
        }
        rendering_artifact_indexes = [
            index
            for index, event in enumerate(events)
            if event["type"] == "artifact.created"
            and event["payload"]["artifactType"]
            in {
                "pdf",
                "pdf-quality-report-v1",
                "pdf-render-manifest-v1",
            }
        ]
        publication_stage_index = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "stage.started"
            and event["payload"]["stage"] == "output_publication"
        )
        publication_completed_index = next(
            index
            for index, event in enumerate(events)
            if event["type"] == "output.publication.completed"
        )
        assert rendering_artifact_indexes
        assert max(rendering_artifact_indexes) < publication_stage_index
        assert publication_stage_index < publication_completed_index
        service.shutdown()


def test_recipe_job_fails_closed_without_trusted_media_probe() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
        )
        started = service.start(
            {
                "jobId": "recipe-no-probe",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "outputCustomization": recipe_payload(),
            }
        )
        final = service.wait(started["jobId"], timeout=5)

        assert final["status"] == "failed"
        assert final["error"]["code"] == "MEDIA_PROBE_REQUIRED"
        assert not (root / "output" / "job" / "transcript-document.v2.json").exists()
        service.shutdown()


def test_recipe_job_rejects_untyped_media_probe_result() -> None:
    class WrongProbe:
        def probe(self, source):
            del source
            return {"accepted": True}

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
            media_probe=WrongProbe(),
        )
        started = service.start(
            {
                "jobId": "recipe-wrong-probe",
                "sourcePath": "source.wav",
                "outputDirectory": "job",
                "speakerCountMode": "manual",
                "speakerCount": 1,
                "outputCustomization": recipe_payload(),
            }
        )
        final = service.wait(started["jobId"], timeout=5)

        assert final["status"] == "failed"
        assert final["error"]["code"] == "MEDIA_PROBE_RESULT_INVALID"
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
    def __init__(self) -> None:
        self.release_calls = 0

    def run(self, document, *, output_directory, config):
        del document, output_directory, config
        raise JobCancelled()

    def release_resources(self) -> None:
        self.release_calls += 1
        raise RuntimeError("release failed after cancellation")


def test_business_cancellation_remains_job_cancelled() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        runner = _CancelledBusinessRunner()
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            runner_factory=lambda request, context: runner,
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
        assert runner.release_calls == 1
        service.shutdown()


class _ReleaseFailingBusinessRunner:
    def run(self, document, *, output_directory, config):
        del document, output_directory, config
        return ()

    def release_resources(self) -> None:
        raise RuntimeError("release failed")


def test_business_release_failure_fails_closed_after_successful_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            runner_factory=lambda request, context: _ReleaseFailingBusinessRunner(),
        )
        started = service.start(
            {
                "jobId": "business-release-failure",
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

        assert final["status"] == "failed"
        assert final["business"]["status"] == "failed"
        assert final["error"]["code"] == "BUSINESS_RESOURCE_RELEASE_FAILED"
        service.shutdown()


def test_rerender_reuses_verified_output_publication_without_republishing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        calls: list[tuple[str, ...]] = []
        events: list[dict[str, Any]] = []

        def publisher(recipe, plans, document, **kwargs):
            calls.append(tuple(plan.delivery_mode.value for plan in plans))
            return _fake_output_publisher(
                recipe,
                plans,
                document,
                **kwargs,
            )

        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
            event_sink=events.append,
            media_probe=MediaProbe(runner=_RecordingProbeRunner([])),
            output_publisher=publisher,
        )
        first = service.start(_publication_job_payload("publication-reuse"))
        completed = service.wait(first["jobId"], timeout=5)
        assert completed["status"] == "completed"
        assert len(calls) == 1
        first_publication = completed["outputPublication"]
        first_receipts = {
            receipt["path"]: (
                receipt["sha256"],
                receipt["sizeBytes"],
            )
            for receipt in first_publication["customerArtifacts"]
        }

        service.rerender("publication-reuse")
        rerendered = service.wait("publication-reuse", timeout=5)

        assert rerendered["status"] == "completed"
        assert len(calls) == 1
        assert rerendered["outputPublication"] == first_publication
        assert {
            receipt["path"]: (
                receipt["sha256"],
                receipt["sizeBytes"],
            )
            for receipt in rerendered["outputPublication"]["customerArtifacts"]
        } == first_receipts
        assert sum(
            event["type"] == "output.publication.completed"
            for event in events
        ) == 1
        assert sum(
            event["type"] == "output.publication.reused"
            for event in events
        ) == 1
        service.shutdown()


def test_rerender_rejects_tampered_customer_artifact_without_republishing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        calls = 0

        def publisher(recipe, plans, document, **kwargs):
            nonlocal calls
            calls += 1
            return _fake_output_publisher(
                recipe,
                plans,
                document,
                **kwargs,
            )

        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
            media_probe=MediaProbe(runner=_RecordingProbeRunner([])),
            output_publisher=publisher,
        )
        started = service.start(_publication_job_payload("publication-tamper"))
        completed = service.wait(started["jobId"], timeout=5)
        assert completed["status"] == "completed"
        sidecar = next(
            Path(receipt["path"])
            for receipt in completed["outputPublication"]["customerArtifacts"]
            if receipt["artifactType"] == "subtitle-sidecar"
        )
        sidecar.write_text("tampered\n", encoding="utf-8")

        service.rerender("publication-tamper")
        failed = service.wait("publication-tamper", timeout=5)

        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "OUTPUT_PUBLICATION_ARTIFACT_CHANGED"
        assert failed["outputPublication"]["status"] == "failed"
        assert failed["outputPublication"]["customerArtifacts"] == []
        assert calls == 1
        service.shutdown()


def test_rerender_rejects_byte_only_manifest_change_without_republishing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        calls = 0

        def publisher(recipe, plans, document, **kwargs):
            nonlocal calls
            calls += 1
            return _fake_output_publisher(
                recipe,
                plans,
                document,
                **kwargs,
            )

        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
            media_probe=MediaProbe(runner=_RecordingProbeRunner([])),
            output_publisher=publisher,
        )
        started = service.start(_publication_job_payload("publication-byte-tamper"))
        completed = service.wait(started["jobId"], timeout=5)
        assert completed["status"] == "completed"
        manifest_path = Path(completed["outputPublication"]["manifestPath"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_path.write_text(
            json.dumps(manifest, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        assert canonical_json_sha256(
            {key: value for key, value in manifest.items() if key != "manifestSha256"}
        ) == manifest["manifestSha256"]

        service.rerender("publication-byte-tamper")
        failed = service.wait("publication-byte-tamper", timeout=5)

        assert failed["status"] == "failed"
        assert failed["error"]["code"] == (
            "OUTPUT_PUBLICATION_MANIFEST_FILE_CHANGED"
        )
        assert failed["outputPublication"]["status"] == "failed"
        assert calls == 1
        service.shutdown()


def test_rerender_rejects_missing_recorded_manifest_file_hash() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        calls = 0

        def publisher(recipe, plans, document, **kwargs):
            nonlocal calls
            calls += 1
            return _fake_output_publisher(
                recipe,
                plans,
                document,
                **kwargs,
            )

        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
            media_probe=MediaProbe(runner=_RecordingProbeRunner([])),
            output_publisher=publisher,
        )
        started = service.start(_publication_job_payload("publication-hash-missing"))
        completed = service.wait(started["jobId"], timeout=5)
        assert completed["status"] == "completed"
        record = service._get_job("publication-hash-missing")
        with record.lock:
            record.output_publication_manifest_file_sha256 = None

        service.rerender("publication-hash-missing")
        failed = service.wait("publication-hash-missing", timeout=5)

        assert failed["status"] == "failed"
        assert failed["error"]["code"] == (
            "OUTPUT_PUBLICATION_MANIFEST_FILE_HASH_MISSING"
        )
        assert failed["outputPublication"]["status"] == "failed"
        assert calls == 1
        service.shutdown()


def test_manifest_persistence_failure_rolls_back_publication_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def fail_manifest_only(path: Path, value: Any):
            if path.name == "output-publication-manifest.v1.json":
                raise OSError("injected manifest persistence failure")
            return persist_json_evidence(path, value)

        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
            media_probe=MediaProbe(runner=_RecordingProbeRunner([])),
        )
        with patch(
            "backend.service.atomic_publish_json_evidence",
            side_effect=fail_manifest_only,
        ):
            started = service.start(
                _publication_job_payload("publication-manifest-failure")
            )
            failed = service.wait(started["jobId"], timeout=5)

        output = root / "output" / "job"
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "OUTPUT_PUBLICATION_FAILED"
        assert failed["outputPublication"]["status"] == "failed"
        assert not (output / "output-publication-manifest.v1.json").exists()
        assert not list(output.glob("published.*"))
        for plan_path in failed["outputPlanPaths"]:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            delivery_path = plan.get("deliveryOutputPath")
            if delivery_path:
                assert not Path(delivery_path).exists()
        service.shutdown()


def test_cancellation_after_publisher_return_rolls_back_before_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        holder: dict[str, WorkerService] = {}

        def cancelling_publisher(recipe, plans, document, **kwargs):
            manifest = _fake_output_publisher(
                recipe,
                plans,
                document,
                **kwargs,
            )
            holder["service"].cancel("publication-cancel")
            return manifest

        service = _service(
            root,
            adapter=FakeTranscriptionAdapter(result_mapping(1)),
            renderer=_PlannedRenderer(),
            media_probe=MediaProbe(runner=_RecordingProbeRunner([])),
            output_publisher=cancelling_publisher,
        )
        holder["service"] = service
        started = service.start(_publication_job_payload("publication-cancel"))
        cancelled = service.wait(started["jobId"], timeout=5)

        output = root / "output" / "job"
        assert cancelled["status"] == "cancelled"
        assert cancelled["outputPublication"]["status"] == "cancelled"
        assert not (output / "output-publication-manifest.v1.json").exists()
        assert not list(output.glob("published.*"))
        for plan_path in cancelled["outputPlanPaths"]:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            delivery_path = plan.get("deliveryOutputPath")
            if delivery_path:
                assert not Path(delivery_path).exists()
        service.shutdown()
