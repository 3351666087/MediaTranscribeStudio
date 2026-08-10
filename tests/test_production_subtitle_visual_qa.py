from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.output_orchestration import MediaProbeArtifact, OutputExecutionPlan
from backend.production_subtitle_visual_qa import (
    ProductionSubtitleVisualQAHook,
)
from backend.subtitle_visual_qa import default_subtitle_visual_qa_policy
from backend.subtitles import (
    CuePolicy,
    SpeakerColorAssignment,
    SubtitleArrangement,
    SubtitleCue,
    SubtitleFormat,
    SubtitleOutputMode,
    SubtitleQAReport,
    SubtitleStyle,
    arrange_cues,
)


class FakeProbe:
    def probe(self, _path: Path) -> Any:
        return SimpleNamespace(
            streams=(
                SimpleNamespace(
                    index=0,
                    attached_picture=False,
                    width=320,
                    height=240,
                ),
            ),
            video_stream_indexes=(0,),
        )


class FakeCollector:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def collect(self, request: dict[str, Any], **kwargs: Any) -> Any:
        self.calls.append((request, kwargs))
        return SimpleNamespace(
            qa_request={"fixture": "qa-request"},
            to_dict=lambda: {
                "evidenceArtifactSha256": "a" * 64,
                "qaRequestSha256": "b" * 64,
                "selection": {"selectionArtifactSha256": "c" * 64},
                "sourceMedia": {"sha256": "d" * 64},
                "renderArtifact": {"sha256": "e" * 64},
            },
        )


def _arrangement() -> SubtitleArrangement:
    return SubtitleArrangement(
        cues=(
            SubtitleCue(
                number=1,
                start_ms=100,
                end_ms=1_100,
                text="[Speaker 1] Hello",
                source_text="Hello",
                source_segment_index=0,
                speaker="Speaker 1",
                speaker_id="speaker-1",
            ),
        ),
        qa=SubtitleQAReport(
            passed=True,
            issues=(),
            cue_count=1,
            source_text_preserved=True,
            maximum_observed_reading_speed=8.0,
        ),
        speaker_colors=(
            SpeakerColorAssignment(
                speaker_id="speaker-1",
                color="#176b87",
                source="palette",
                palette_index=0,
                style_name="MTS-Speaker-0001",
            ),
        ),
    )


def _plan(
    tmp_path: Path,
    *,
    mode: SubtitleOutputMode,
    style: SubtitleStyle | None = None,
) -> OutputExecutionPlan:
    source = tmp_path / "source.mp4"
    probe_path = tmp_path / "media-probe.json"
    probe_path.write_text("{}", encoding="utf-8")
    return OutputExecutionPlan(
        customization_sha256="f" * 64,
        source_path=source,
        output_directory=tmp_path,
        media_probe_artifact=MediaProbeArtifact(
            path=probe_path,
            size_bytes=2,
            sha256="1" * 64,
            source_path=source,
            source_size_bytes=6,
            source_sha256="2" * 64,
            probe_fingerprint_sha256="3" * 64,
        ),
        report_enabled=False,
        report_config={},
        exports_config={},
        safety_config={
            "preserveSourceMedia": True,
            "overwriteSourceMedia": False,
        },
        reversibility_config={
            "sourceMediaImmutable": True,
            "derivedArtifactOnly": True,
        },
        subtitle_enabled=True,
        subtitle_config={},
        subtitle_formats=(SubtitleFormat.ASS,),
        subtitle_style=style or SubtitleStyle(),
        cue_policy=CuePolicy(),
        subtitle_theme="youtube-clean",
        speaker_color_mode="automatic",
        speaker_color_seed="fixture",
        speaker_color_overrides={},
        sidecar_paths=(),
        delivery_mode=mode,
        delivery_output_path=tmp_path / f"{mode.value}.mp4",
        subtitle_codec=None,
        burn_in_strategy=None,
        visual_qa_required=True,
        presentation_limitations=(),
    )


def _hook(collector: FakeCollector) -> ProductionSubtitleVisualQAHook:
    return ProductionSubtitleVisualQAHook(
        ffmpeg_path="unused-ffmpeg",
        ffprobe_path="unused-ffprobe",
        probe=FakeProbe(),
        collector=collector,
        evaluator=lambda request: {
            "kind": "subtitle-visual-qa-result",
            "passed": request == {"fixture": "qa-request"},
        },
    )


def test_soft_mux_binds_private_ass_and_execution_plan(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    rendered = tmp_path / "rendered.mp4"
    ass = tmp_path / "canonical.ass"
    source.write_bytes(b"source")
    rendered.write_bytes(b"rendered")
    ass.write_text("[Script Info]\n", encoding="utf-8")
    plan = _plan(tmp_path, mode=SubtitleOutputMode.SOFT_MUX)
    collector = FakeCollector()

    result = _hook(collector)(
        source_path=source,
        rendered_path=rendered,
        arrangement=_arrangement(),
        delivery_receipt={
            "mode": "soft-mux",
            "subtitlePath": str(ass),
        },
        execution_plan=plan,
    )

    assert result["passed"] is True
    assert result["deliveryMode"] == "soft-mux"
    assert result["executionPlanSha256"] == plan.deterministic_hash()
    request, kwargs = collector.calls[0]
    assert request["speakers"] == [
        {"speakerId": "speaker-1", "color": "#176B87"}
    ]
    assert request["cues"][0]["styleId"] == "MTS-Speaker-0001"
    assert request["cues"][0]["bounds"] == {
        "x": 6,
        "y": 96,
        "width": 308,
        "height": 143,
    }
    assert kwargs["canonical_ass_overlay_path"] == ass.resolve()
    assert len(kwargs["canonical_ass_overlay_sha256"]) == 64
    assert kwargs["delivery_receipt"]["mode"] == "soft-mux"


def test_burn_in_binds_ass_carrier_for_background_only_contrast(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    rendered = tmp_path / "rendered.mp4"
    source.write_bytes(b"source")
    rendered.write_bytes(b"rendered")
    ass = tmp_path / "canonical.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    collector = FakeCollector()

    result = _hook(collector)(
        source_path=source,
        rendered_path=rendered,
        arrangement=_arrangement(),
        delivery_receipt={"mode": "burn-in", "subtitlePath": str(ass)},
        execution_plan=_plan(tmp_path, mode=SubtitleOutputMode.BURN_IN),
    )

    assert result["passed"] is True
    _, kwargs = collector.calls[0]
    assert set(kwargs) == {
        "delivery_receipt",
        "canonical_ass_overlay_path",
        "canonical_ass_overlay_sha256",
    }
    assert kwargs["canonical_ass_overlay_path"] == ass.resolve()


def test_production_policy_preserves_background_coverage_contract(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, mode=SubtitleOutputMode.BURN_IN)

    policy = ProductionSubtitleVisualQAHook._policy(
        plan,
        has_speaker_colors=True,
    )

    expected_contrast = default_subtitle_visual_qa_policy()["contrast"]
    expected_contrast["minimumRatio"] = 3.0
    assert policy["contrast"] == expected_contrast


def test_boxed_dark_style_requires_only_effective_dark_background(
    tmp_path: Path,
) -> None:
    plan = _plan(
        tmp_path,
        mode=SubtitleOutputMode.BURN_IN,
        style=SubtitleStyle(
            background_color="#000000",
            background_opacity=0.72,
        ),
    )

    policy = ProductionSubtitleVisualQAHook._policy(
        plan,
        has_speaker_colors=True,
    )

    assert policy["contrast"]["minimumRatio"] == 3.0
    assert policy["contrast"]["requiredBackgroundClasses"] == ["dark"]
    assert policy["contrast"]["darkMaximumLuminance"] == 0.35
    assert policy["contrast"]["lightMinimumLuminance"] == 0.65


def test_jfk_wrapping_emits_contract_valid_rendered_lines(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    rendered = tmp_path / "rendered.mp4"
    source.write_bytes(b"source")
    rendered.write_bytes(b"rendered")
    ass = tmp_path / "canonical.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    collector = FakeCollector()
    text = (
        "What your country can do for you, ask what you can do for your country."
    )
    arrangement = arrange_cues(
        [
            {
                "startMs": 6490,
                "endMs": 12150,
                "text": text,
                "speaker": "speaker-1",
                "speakerId": "speaker-1",
            }
        ],
        policy=CuePolicy(
            max_characters_per_line=38,
            max_lines=2,
            max_reading_speed=20,
            min_cue_ms=800,
            max_cue_ms=7000,
            gap_ms=80,
            include_speaker_labels=True,
            speaker_label_template="{speaker} ",
        ),
    )

    _hook(collector)(
        source_path=source,
        rendered_path=rendered,
        arrangement=arrangement,
        delivery_receipt={"mode": "burn-in", "subtitlePath": str(ass)},
        execution_plan=_plan(tmp_path, mode=SubtitleOutputMode.BURN_IN),
    )

    request, _ = collector.calls[0]
    assert "".join(cue.source_text for cue in arrangement.cues) == text
    assert all(
        line and line == line.strip()
        for cue in request["cues"]
        for line in cue["renderedLines"]
    )


def test_missing_execution_plan_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    rendered = tmp_path / "rendered.mp4"
    source.write_bytes(b"source")
    rendered.write_bytes(b"rendered")

    with pytest.raises(TypeError, match="execution plan"):
        _hook(FakeCollector())(
            source_path=source,
            rendered_path=rendered,
            arrangement=_arrangement(),
            delivery_receipt={"mode": "burn-in"},
        )
