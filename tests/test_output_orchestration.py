from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from backend.adapters import AdapterContext, JavaPdfRendererAdapter
from backend.errors import WorkerError
from backend.media_probe import (
    MediaProbe,
    MediaProbeError,
    MediaProbeErrorCode,
    ProcessLimits,
    ProcessResult,
)
from backend.output_customization import (
    default_output_customization,
    resolve_output_customization,
)
from backend.output_orchestration import (
    compile_output_execution_plan,
    execute_prepared_subtitle_outputs,
    persist_media_probe_artifact,
    prepare_subtitle_outputs,
)
from backend.persistence import sha256_file
from backend.models import SpeakerCountPolicy, StartJobRequest
from backend.subtitles import SubtitleFormat, SubtitleOutputMode


class ProbeRunner:
    def __init__(
        self,
        *,
        color_transfer: str | None = "bt709",
        hdr: bool = False,
        attached_picture_only: bool = False,
    ) -> None:
        self.color_transfer = color_transfer
        self.hdr = hdr
        self.attached_picture_only = attached_picture_only

    def run(
        self,
        command: tuple[str, ...] | list[str],
        *,
        limits: ProcessLimits,
    ) -> ProcessResult:
        del limits
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
            video: dict[str, Any] = {
                "index": 0,
                "codec_name": "h264",
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "duration": "8.0",
                "color_transfer": self.color_transfer,
                "disposition": {
                    "default": 1,
                    "attached_pic": int(self.attached_picture_only),
                },
            }
            if self.hdr:
                video["color_transfer"] = "smpte2084"
            streams: list[dict[str, Any]] = [video]
            if not self.attached_picture_only:
                streams.append(
                    {
                        "index": 1,
                        "codec_name": "aac",
                        "codec_type": "audio",
                        "sample_rate": "48000",
                        "channels": 2,
                        "duration": "8.0",
                        "disposition": {"default": 1},
                    }
                )
            payload = {
                "streams": streams,
                "format": {
                    "format_name": "mov,mp4",
                    "duration": "8.0",
                    "bit_rate": "1000000",
                },
                "programs": [],
                "chapters": [],
            }
            return ProcessResult(
                returncode=0,
                stdout=json.dumps(payload).encode(),
                stderr=b"",
                elapsed_ms=2,
            )
        return ProcessResult(
            returncode=0,
            stdout=b"",
            stderr=b"",
            elapsed_ms=3,
        )


def _admitted_probe(
    source: Path,
    *,
    color_transfer: str | None = "bt709",
    hdr: bool = False,
) -> Any:
    return MediaProbe(
        runner=ProbeRunner(color_transfer=color_transfer, hdr=hdr)
    ).probe(source)


def _transcript() -> dict[str, Any]:
    return {
        "speakers": [
            {"id": "speaker-1", "displayName": "Alice"},
            {"id": "speaker-2", "displayName": "Bob"},
        ],
        "segments": [
            {
                "startMs": 0,
                "endMs": 3000,
                "speakerId": "speaker-1",
                "rawText": "Hello.",
                "normalizedText": "Hello.",
                "displayText": "Hello.",
            },
            {
                "startMs": 3200,
                "endMs": 6500,
                "speakerId": "speaker-2",
                "rawText": "Welcome.",
                "normalizedText": "Welcome.",
                "displayText": "Welcome.",
            },
        ],
    }


def _artifact(
    tmp_path: Path,
    *,
    color_transfer: str | None = "bt709",
    hdr: bool = False,
) -> tuple[Path, Path, Any, Any]:
    source = tmp_path / "source.extension-is-not-trusted"
    source.write_bytes(b"immutable source fixture")
    output = tmp_path / "job-output"
    output.mkdir()
    probe = _admitted_probe(
        source,
        color_transfer=color_transfer,
        hdr=hdr,
    )
    artifact = persist_media_probe_artifact(
        probe,
        source_path=source,
        output_directory=output,
    )
    return source, output, probe, artifact


def _prepared_soft_mux(
    tmp_path: Path,
) -> tuple[Path, Path, Any]:
    source, output, probe, artifact = _artifact(tmp_path)
    customer_output = output / "rendered.mp4"
    customization = resolve_output_customization(
        {
            "delivery": {
                "mode": "soft-mux",
                "outputTarget": {
                    "binding": "bound",
                    "sourcePath": str(source.resolve()),
                    "outputPath": str(customer_output.resolve()),
                },
            }
        }
    )
    plan = compile_output_execution_plan(
        customization,
        source_path=source,
        output_directory=output,
        media_probe=probe,
        media_probe_artifact=artifact,
        language="en",
        speaker_count=2,
        generated_date="2026-07-23",
    )
    return source, customer_output, prepare_subtitle_outputs(
        plan,
        _transcript(),
    )


def test_probe_artifact_is_exact_hash_bound_and_never_replaced(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)

    assert artifact.path == (output / "media-probe.v1.json").resolve()
    assert artifact.sha256 == sha256_file(artifact.path)
    assert artifact.source_sha256 == sha256_file(source)
    assert json.loads(artifact.path.read_text(encoding="utf-8")) == probe.to_dict()

    with pytest.raises(WorkerError) as raised:
        persist_media_probe_artifact(
            probe,
            source_path=source,
            output_directory=output,
        )
    assert raised.value.code == "MEDIA_PROBE_ARTIFACT_EXISTS"
    assert artifact.sha256 == sha256_file(artifact.path)


def test_attached_cover_art_does_not_admit_a_non_media_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cover-only.bin"
    source.write_bytes(b"cover")

    with pytest.raises(MediaProbeError) as raised:
        MediaProbe(
            runner=ProbeRunner(attached_picture_only=True)
        ).probe(source)

    assert raised.value.code is MediaProbeErrorCode.NO_MEDIA_STREAM


def test_plan_plumbs_canonical_diy_settings_and_existing_subtitle_modules(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    customization = default_output_customization()

    plan = compile_output_execution_plan(
        customization,
        source_path=source,
        output_directory=output,
        media_probe=probe,
        media_probe_artifact=artifact,
        language="en-US",
        speaker_count=2,
        generated_date="2026-07-23",
    )

    assert plan.media_probe_artifact.sha256 == artifact.sha256
    assert plan.report_config == customization.canonical_dict()["report"]
    assert plan.subtitle_config == customization.canonical_dict()["subtitle"]
    assert plan.exports_config == customization.canonical_dict()["exports"]
    assert plan.safety_config["preserveSourceMedia"] is True
    assert plan.reversibility_config["sourceMediaImmutable"] is True
    assert [value for value, _ in plan.sidecar_paths] == [
        SubtitleFormat.ASS,
        SubtitleFormat.SRT,
        SubtitleFormat.WEBVTT,
    ]
    assert all(path.parent == output for _, path in plan.sidecar_paths)
    assert all(path != source for _, path in plan.sidecar_paths)

    prepared = prepare_subtitle_outputs(plan, _transcript())
    assert prepared.media_delivery_plan is None
    assert prepared.arrangement.qa.passed
    assert [item.delivery_plan.mode for item in prepared.sidecars] == [
        SubtitleOutputMode.SIDECAR,
        SubtitleOutputMode.SIDECAR,
        SubtitleOutputMode.SIDECAR,
    ]
    ass = prepared.sidecars[0].payload
    assert f"Style: Default,{plan.subtitle_style.font_family}," in ass
    assert "Alice" in ass
    declarations = prepared.visual_qa_speakers
    assert declarations == prepared.arrangement.visual_qa_speakers()
    assert [item["speakerId"] for item in declarations] == [
        "speaker-1",
        "speaker-2",
    ]
    for assignment in prepared.arrangement.speaker_colors:
        red, green, blue = (
            assignment.color[1:3],
            assignment.color[3:5],
            assignment.color[5:7],
        )
        assert (
            f"Style: {assignment.style_name},"
            f"{plan.subtitle_style.font_family},"
            f"{plan.subtitle_style.font_size},"
            f"&H00{blue}{green}{red}"
        ) in ass
    assert "MTS-Speaker-0001,Alice" in ass
    assert "MTS-Speaker-0002,Bob" in ass
    assert source.read_bytes() == b"immutable source fixture"

    renderer_config = plan.report_renderer_config(
        speaker_policy={"mode": "auto", "resolvedCount": 2},
        renderer_id="java-openhtmltopdf-pdfbox",
    )
    assert renderer_config["presentation"]["report"] == plan.report_config
    assert (
        renderer_config["mediaProbeArtifact"]["sha256"]
        == artifact.sha256
    )


def test_speaker_color_override_wins_for_stable_id_not_display_name(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    customization = resolve_output_customization(
        {
            "subtitle": {
                "speakerColors": {
                    "seed": "orchestration-seed",
                    "overrides": [
                        {"speakerId": "speaker-1", "color": "#12AB34"}
                    ],
                }
            }
        }
    )
    plan = compile_output_execution_plan(
        customization,
        source_path=source,
        output_directory=output,
        media_probe=probe,
        media_probe_artifact=artifact,
        language="en",
        speaker_count=2,
        generated_date="2026-07-23",
    )

    prepared = prepare_subtitle_outputs(plan, _transcript())
    assignments = {
        item.speaker_id: item
        for item in prepared.arrangement.speaker_colors
    }
    ass = prepared.sidecars[0].payload

    assert assignments["speaker-1"].color == "#12AB34"
    assert assignments["speaker-1"].source == "override"
    assert assignments["speaker-1"].palette_index is None
    assert assignments["speaker-2"].source == "palette"
    assert "&H0034AB12" in ass
    assert (
        f"{assignments['speaker-1'].style_name},"
        "Alice,0,0,0,,Alice Hello."
        in ass
    )
    assert prepared.visual_qa_speakers == (
        {"speakerId": "speaker-1", "color": "#12AB34"},
        {
            "speakerId": "speaker-2",
            "color": assignments["speaker-2"].color,
        },
    )


def test_monochrome_mode_does_not_declare_or_render_distinct_speaker_colors(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    customization = resolve_output_customization(
        {
            "subtitle": {
                "speakerColors": {
                    "mode": "monochrome",
                    "algorithm": "monochrome-v1",
                    "minimumDeltaE": 0,
                    "overrides": [
                        {"speakerId": "speaker-1", "color": "#FF0000"}
                    ],
                }
            }
        }
    )
    plan = compile_output_execution_plan(
        customization,
        source_path=source,
        output_directory=output,
        media_probe=probe,
        media_probe_artifact=artifact,
        language="en",
        speaker_count=2,
        generated_date="2026-07-23",
    )

    prepared = prepare_subtitle_outputs(plan, _transcript())
    ass = prepared.sidecars[0].payload

    assert prepared.arrangement.speaker_colors == ()
    assert prepared.visual_qa_speakers == ()
    assert "MTS-SpeakerColor" not in ass
    assert "Style: MTS-Speaker-" not in ass
    assert ass.count(",Default,") == 2
    assert "&H000000FF" not in ass


def test_java_pdf_adapter_receives_exact_planned_report_configuration(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    plan = compile_output_execution_plan(
        default_output_customization(),
        source_path=source,
        output_directory=output,
        media_probe=probe,
        media_probe_artifact=artifact,
        language="en",
        speaker_count=2,
        generated_date="2026-07-23",
    )
    captured: dict[str, Any] = {}
    captured_segments: list[dict[str, Any]] = []

    class Assembler:
        def assemble(self, segments: Any, **kwargs: Any) -> None:
            captured.update(kwargs["config"])
            captured_segments.extend(segments)
            raise WorkerError("TEST_STOP", "captured renderer config")

    adapter = JavaPdfRendererAdapter(assembler=Assembler(), java_client=object())
    request = StartJobRequest(
        job_id="job-output-plan",
        source_path=source,
        output_directory=output,
        speaker_policy=SpeakerCountPolicy.from_payload(
            {"speakerCountMode": "auto"}
        ),
        render_pdf=True,
        language="en",
    )
    document = {
        "documentId": "document-1",
        "generatedAt": "2026-07-23T00:00:00Z",
        "language": "en",
        "source": {"durationMs": 6500},
        "speakerPolicy": {
            "mode": "auto",
            "resolvedCount": 2,
            "speakerIds": ["speaker-1", "speaker-2"],
            "estimate": {"estimatedCount": 2, "confidence": 0.9},
        },
        "speakers": [
            {"id": "speaker-1", "displayName": "Alice"},
            {"id": "speaker-2", "displayName": "Bob"},
        ],
        "segments": [
            {
                "segmentId": "segment-1",
                "startMs": 0,
                "endMs": 3000,
                "speakerId": "speaker-1",
                "speakerScores": [],
                "rawText": "Hello.",
                "normalizedText": "Hello.",
                "displayText": "Hello.",
                "evidence": {
                    "overlap": {
                        "canonicalSpeakerTurns": [
                            {
                                "startMs": 0,
                                "endMs": 3000,
                                "speakerId": "speaker-1",
                                "localSpeaker": "LOCAL_A",
                            }
                        ]
                    },
                    "pyannoteCanonicalMapping": {
                        "accepted": True,
                        "applied": True,
                    },
                },
                "revisions": [
                    {
                        "id": "segment-1:speaker:1",
                        "type": "speaker",
                        "source": "acoustic",
                        "reasonCode": "PYANNOTE_CANONICAL_TRACK_MAPPING",
                        "before": "speaker-2",
                        "after": "speaker-1",
                        "confidence": 0.8,
                        "evidenceRefs": ["pyannote-mapping:segment-1"],
                    }
                ],
            }
        ],
    }
    context = AdapterContext(
        job_id=request.job_id,
        output_directory=output,
        cancellation=threading.Event(),
    )

    with pytest.raises(WorkerError) as raised:
        adapter.render(
            document,
            request,
            context,
            output_plan=plan,
        )

    assert raised.value.code == "TEST_STOP"
    assert captured["presentation"]["report"] == plan.report_config
    assert captured["outputExecutionPlanSha256"] == plan.deterministic_hash()
    assert captured["mediaProbeArtifact"]["sha256"] == artifact.sha256
    assert (
        captured_segments[0]["evidence"]["overlap"]["canonicalSpeakerTurns"][0][
            "localSpeaker"
        ]
        == "LOCAL_A"
    )
    assert captured_segments[0]["evidence"]["pyannoteCanonicalMapping"] == {
        "accepted": True,
        "applied": True,
    }
    assert captured_segments[0]["revisions"][0]["confidence"] == 0.8
    assert captured_segments[0]["revisions"][0]["evidenceRefs"] == [
        "pyannote-mapping:segment-1"
    ]


def test_source_content_mutation_is_rejected_even_when_size_is_unchanged(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    source.write_bytes(b"x" * probe.source_size_bytes)

    with pytest.raises(WorkerError) as raised:
        compile_output_execution_plan(
            default_output_customization(),
            source_path=source,
            output_directory=output,
            media_probe=probe,
            media_probe_artifact=artifact,
            language="en",
            speaker_count=2,
            generated_date="2026-07-23",
        )

    assert raised.value.code == "SOURCE_MEDIA_CHANGED"


def test_bound_sidecar_suffix_must_match_primary_format(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    customization = resolve_output_customization(
        {
            "delivery": {
                "outputTarget": {
                    "binding": "bound",
                    "sourcePath": str(source.resolve()),
                    "outputPath": str((output / "captions.txt").resolve()),
                }
            }
        }
    )

    with pytest.raises(WorkerError) as raised:
        compile_output_execution_plan(
            customization,
            source_path=source,
            output_directory=output,
            media_probe=probe,
            media_probe_artifact=artifact,
            language="en",
            speaker_count=2,
            generated_date="2026-07-23",
        )

    assert (
        raised.value.code
        == "OUTPUT_CUSTOMIZATION_SUBTITLE_SUFFIX_MISMATCH"
    )


@pytest.mark.parametrize(
    ("color_transfer", "hdr", "reason"),
    [
        (None, False, "unknown-dynamic-range"),
        ("unknown", False, "unknown-dynamic-range"),
        ("smpte2084", True, "hdr-source"),
    ],
)
def test_burn_in_fails_closed_for_hdr_or_unknown_dynamic_range(
    tmp_path: Path,
    color_transfer: str | None,
    hdr: bool,
    reason: str,
) -> None:
    source, output, probe, artifact = _artifact(
        tmp_path,
        color_transfer=color_transfer,
        hdr=hdr,
    )
    customization = resolve_output_customization(
        {
            "delivery": {
                "mode": "burn-in",
                "outputTarget": {
                    "binding": "bound",
                    "sourcePath": str(source.resolve()),
                    "outputPath": str((output / "rendered.mp4").resolve()),
                },
                "burnIn": {
                    "strategy": "h264-high-quality",
                    "sourceDynamicRange": "sdr",
                    "dynamicRangeEvidence": {
                        "verified": True,
                        "method": "ffprobe-color-metadata",
                        "probeSha256": artifact.sha256,
                    },
                },
            }
        }
    )

    with pytest.raises(WorkerError) as raised:
        compile_output_execution_plan(
            customization,
            source_path=source,
            output_directory=output,
            media_probe=probe,
            media_probe_artifact=artifact,
            language="en",
            speaker_count=2,
            generated_date="2026-07-23",
        )

    assert raised.value.code == "OUTPUT_CUSTOMIZATION_HDR_UNSAFE"
    assert raised.value.details["reason"] == reason


def test_subtitle_generation_allows_transcripts_without_speaker_metadata(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    plan = compile_output_execution_plan(
        default_output_customization(),
        source_path=source,
        output_directory=output,
        media_probe=probe,
        media_probe_artifact=artifact,
        language="en",
        speaker_count=2,
        generated_date="2026-07-23",
    )
    document = _transcript()
    document.pop("speakers")

    prepared = prepare_subtitle_outputs(plan, document)

    assert prepared.arrangement.qa.passed
    assert "speaker-1" in prepared.sidecars[0].payload


def test_burn_in_requires_visual_qa_hook_before_any_delivery(
    tmp_path: Path,
) -> None:
    source, output, probe, artifact = _artifact(tmp_path)
    customization = resolve_output_customization(
        {
            "delivery": {
                "mode": "burn-in",
                "outputTarget": {
                    "binding": "bound",
                    "sourcePath": str(source.resolve()),
                    "outputPath": str((output / "rendered.mp4").resolve()),
                },
                "burnIn": {
                    "strategy": "h264-high-quality",
                    "sourceDynamicRange": "sdr",
                    "dynamicRangeEvidence": {
                        "verified": True,
                        "method": "ffprobe-color-metadata",
                        "probeSha256": artifact.sha256,
                    },
                },
            }
        }
    )
    plan = compile_output_execution_plan(
        customization,
        source_path=source,
        output_directory=output,
        media_probe=probe,
        media_probe_artifact=artifact,
        language="en",
        speaker_count=2,
        generated_date="2026-07-23",
    )
    prepared = prepare_subtitle_outputs(plan, _transcript())

    class Executor:
        def __init__(self) -> None:
            self.calls = 0

        def deliver(self, *_: Any, **__: Any) -> None:
            self.calls += 1

    executor = Executor()
    with pytest.raises(WorkerError) as raised:
        execute_prepared_subtitle_outputs(
            prepared,
            executor=executor,
            subtitle_language="en",
            subtitle_title="Captions",
            burn_in_strategy="h264-high-quality",
        )

    assert raised.value.code == "SUBTITLE_VISUAL_QA_REQUIRED"
    assert executor.calls == 0


def test_media_is_quarantined_until_visual_qa_passes_then_published(
    tmp_path: Path,
) -> None:
    source, customer_output, prepared = _prepared_soft_mux(tmp_path)
    events: list[str] = []

    class Staged:
        def __init__(self, quarantine_path: Path) -> None:
            self.quarantine_path = quarantine_path
            self.receipt = {"status": "quarantined"}

    class Publication:
        receipt = {"status": "delivered"}

        def to_dict(self) -> dict[str, Any]:
            return {
                "status": "published",
                "atomic": True,
                "noReplace": True,
            }

    class Executor:
        def deliver(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"status": "sidecar-delivered"}

        def stage_media_delivery(
            self,
            *_: Any,
            **__: Any,
        ) -> Staged:
            events.append("stage")
            assert not customer_output.exists()
            quarantine = (
                customer_output.parent
                / ".rendered.mts-quarantine-fixture.mp4"
            )
            quarantine.write_bytes(b"qa-candidate")
            return Staged(quarantine)

        def publish_staged_media(
            self,
            staged: Staged,
            *,
            visual_qa_evidence: dict[str, Any],
        ) -> Publication:
            events.append("publish")
            assert visual_qa_evidence["passed"] is True
            assert staged.quarantine_path.exists()
            assert not customer_output.exists()
            os.link(staged.quarantine_path, customer_output)
            staged.quarantine_path.unlink()
            return Publication()

    def visual_qa(**kwargs: Any) -> dict[str, Any]:
        events.append("visual-qa")
        assert kwargs["source_path"] == source.resolve()
        assert kwargs["rendered_path"].name.startswith(
            ".rendered.mts-quarantine-"
        )
        assert kwargs["rendered_path"].exists()
        assert not customer_output.exists()
        assert (
            kwargs["arrangement"].visual_qa_speakers()
            == prepared.visual_qa_speakers
        )
        ass = next(
            item.payload
            for item in prepared.sidecars
            if item.subtitle_format is SubtitleFormat.ASS
        )
        for assignment in kwargs["arrangement"].speaker_colors:
            red, green, blue = (
                assignment.color[1:3],
                assignment.color[3:5],
                assignment.color[5:7],
            )
            assert (
                f"Style: {assignment.style_name},"
                f"Noto Sans CJK SC,52,&H00{blue}{green}{red}"
            ) in ass
        return {"passed": True, "analysisId": "qa-success"}

    result = execute_prepared_subtitle_outputs(
        prepared,
        executor=Executor(),
        subtitle_language="en",
        subtitle_title="Captions",
        visual_qa_hook=visual_qa,
    )

    assert events == ["stage", "visual-qa", "publish"]
    assert customer_output.read_bytes() == b"qa-candidate"
    assert result["mediaReceipt"] == {"status": "delivered"}
    assert result["mediaPublication"]["atomic"] is True
    assert result["visualQa"]["passed"] is True


def test_failed_visual_qa_rolls_back_quarantine_without_customer_output(
    tmp_path: Path,
) -> None:
    _, customer_output, prepared = _prepared_soft_mux(tmp_path)
    events: list[str] = []

    class Staged:
        def __init__(self, quarantine_path: Path) -> None:
            self.quarantine_path = quarantine_path
            self.receipt = {"status": "quarantined"}

    class Executor:
        def deliver(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"status": "sidecar-delivered"}

        def stage_media_delivery(
            self,
            *_: Any,
            **__: Any,
        ) -> Staged:
            events.append("stage")
            quarantine = (
                customer_output.parent
                / ".rendered.mts-quarantine-failed.mp4"
            )
            quarantine.write_bytes(b"failed-qa-candidate")
            return Staged(quarantine)

        def publish_staged_media(self, *_: Any, **__: Any) -> None:
            events.append("publish")
            raise AssertionError("failed QA must never publish")

        def rollback_staged_media(
            self,
            staged: Staged,
            *,
            reason: str,
        ) -> dict[str, Any]:
            events.append("rollback")
            staged.quarantine_path.unlink()
            return {
                "status": "rolled-back",
                "reason": reason,
                "customerOutputState": "absent",
                "quarantine": {"state": "removed"},
            }

    with pytest.raises(WorkerError) as raised:
        execute_prepared_subtitle_outputs(
            prepared,
            executor=Executor(),
            subtitle_language="en",
            subtitle_title="Captions",
            visual_qa_hook=lambda **_: {
                "passed": False,
                "failureCodes": ["subtitle-clipped"],
            },
        )

    assert raised.value.code == "SUBTITLE_VISUAL_QA_FAILED"
    assert events == ["stage", "rollback"]
    assert raised.value.details["rollback"]["status"] == "rolled-back"
    assert (
        raised.value.details["rollback"]["customerOutputState"]
        == "absent"
    )
    assert not customer_output.exists()
    assert not list(
        customer_output.parent.glob("*.mts-quarantine-failed.mp4")
    )
