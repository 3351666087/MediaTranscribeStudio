"""Shared fixtures for worker-only tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from backend import (
    AdapterContext,
    PathPolicy,
    RenderResult,
    TranscriptionResult,
    WorkerService,
)


def segment_mapping(
    speaker_count: int,
    index: int,
    *,
    confidence: float = 0.96,
    overlapping: bool = False,
    human_locked: bool = False,
    assigned_speaker: str | None = None,
    score_order: list[str] | None = None,
    margin: float | None = None,
    revisions: list[dict[str, Any]] | None = None,
    raw_text: str | None = None,
    normalized_text: str | None = None,
    display_text: str | None = None,
) -> dict[str, Any]:
    canonical = [f"speaker-{number}" for number in range(1, speaker_count + 1)]
    natural = canonical[index % speaker_count]
    assigned = assigned_speaker or natural
    ordered = score_order or [natural, *[item for item in canonical if item != natural]]
    score_by_id: dict[str, float] = {}
    for rank, speaker_id in enumerate(ordered):
        score_by_id[speaker_id] = 0.92 - rank * 0.25
    scores = [
        {"speakerId": speaker_id, "score": score_by_id[speaker_id]}
        for speaker_id in canonical
    ]
    ranked = sorted(scores, key=lambda item: item["score"], reverse=True)
    calculated_margin = (
        ranked[0]["score"] - ranked[1]["score"] if speaker_count > 1 else 2.0
    )
    raw = raw_text or f"这是第{index + 1}位说话人的中文原文。"
    normalized = normalized_text or raw
    display = display_text or normalized
    return {
        "id": f"segment-{index + 1:04d}",
        "startMs": index * 1000,
        "endMs": (index + 1) * 1000,
        "speakerId": assigned,
        "rawText": raw,
        "normalizedText": normalized,
        "displayText": display,
        "confidence": confidence,
        "speakerScores": scores,
        "speakerMargin": calculated_margin if margin is None else margin,
        "overlapping": overlapping,
        "humanLocked": human_locked,
        "revisions": revisions or [],
        "evidence": {
            "voiceprint": {"provider": "CAM++"},
            "boundary": {"provider": "FunASR"},
            "asr": {"provider": "Qwen3-ASR-1.7B"},
        },
    }


def result_mapping(
    speaker_count: int,
    *,
    estimate: bool = True,
    estimate_confidence: float = 0.95,
    candidate_min: int | None = None,
    candidate_max: int | None = None,
    segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "durationMs": speaker_count * 1000,
        "segments": segments
        or [segment_mapping(speaker_count, index) for index in range(speaker_count)],
        "models": [
            {"role": "voiceprint", "name": "CAM++"},
            {"role": "boundary", "name": "FunASR"},
            {"role": "asr", "name": "Qwen3-ASR-1.7B"},
        ],
    }
    if estimate:
        value["speakerCountEstimate"] = {
            "estimatedCount": speaker_count,
            "confidence": estimate_confidence,
            "candidateRange": {
                "min": candidate_min if candidate_min is not None else speaker_count,
                "max": candidate_max if candidate_max is not None else speaker_count,
            },
            "method": "offline-diarization-consensus",
        }
    return value


class FakeTranscriptionAdapter:
    adapter_id = "fake-offline"
    version = "test-1"

    def __init__(self, result: dict[str, Any] | TranscriptionResult) -> None:
        self.result = result

    def transcribe(self, request, context):
        context.raise_if_cancelled()
        return self.result


class BlockingTranscriptionAdapter:
    adapter_id = "blocking-offline"
    version = "test-1"

    def __init__(self) -> None:
        self.entered = threading.Event()

    def transcribe(self, request, context):
        self.entered.set()
        while True:
            context.raise_if_cancelled()
            time.sleep(0.01)


class FakeDynamicRenderer:
    adapter_id = "fake-dynamic-renderer"
    version = "test-2"

    def render(self, document, request, context):
        context.raise_if_cancelled()
        output = request.output_directory
        quality = output / "quality-report.json"
        manifest = output / "render-manifest.json"
        pdf = output / "transcript.pdf"
        quality.write_text('{"status":"passed"}\n', encoding="utf-8")
        manifest.write_text('{"status":"passed"}\n', encoding="utf-8")
        pdf.write_bytes(b"%PDF-1.7\n% worker fixture\n")
        return RenderResult(
            template_hash="a" * 64,
            renderer_version=self.version,
            quality_status="passed",
            quality_report_path=quality,
            render_manifest_path=manifest,
            artifact_paths=(pdf,),
        )


def make_service(
    input_root: Path,
    output_root: Path,
    *,
    adapter=None,
    renderer=None,
    event_sink=None,
    max_workers: int = 1,
) -> WorkerService:
    return WorkerService(
        path_policy=PathPolicy(
            allowed_input_roots=[input_root],
            allowed_output_root=output_root,
        ),
        transcription_adapter=adapter,
        renderer_adapter=renderer,
        event_sink=event_sink,
        max_workers=max_workers,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class FakeLegacyAssembler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assemble(self, segments, **kwargs):
        self.calls.append({"segments": segments, **kwargs})
        return {
            "schemaVersion": "1.0.0",
            "documentId": "legacy-fixture",
        }


class FakeLegacyJavaClient:
    def render(self, report_document, *, job_id, output_directory):
        output = Path(output_directory)
        quality = output / "legacy-quality.json"
        manifest = output / "legacy-manifest.json"
        pdf = output / "legacy.pdf"
        quality.write_text("{}\n", encoding="utf-8")
        manifest.write_text("{}\n", encoding="utf-8")
        pdf.write_bytes(b"%PDF-1.7\n")
        return SimpleNamespace(
            result={
                "templateHash": "b" * 64,
                "rendererVersion": "legacy-test",
                "quality": {"status": "passed"},
            },
            artifact_paths={
                "qualityReportPath": quality,
                "manifestPath": manifest,
                "pdfPath": pdf,
            },
        )
