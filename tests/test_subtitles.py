from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from backend.subtitles import (
    CuePolicy,
    SourceProtectionError,
    SubtitleFormat,
    SubtitleOutputMode,
    SubtitleStyle,
    SubtitleTheme,
    arrange_cues,
    build_subtitle_output_plan,
    export_subtitles,
    style_for_theme,
)


def test_chinese_punctuation_wrap_preserves_every_source_character() -> None:
    text = "今天确认发布计划，明天完成回归测试。随后检查字幕质量！"
    result = arrange_cues(
        [{"start": 0, "end": 6, "text": text, "speaker": "主持人"}],
        policy=CuePolicy(
            max_characters_per_line=10,
            max_lines=2,
            max_reading_speed=12,
            min_cue_ms=500,
            max_cue_ms=3000,
            gap_ms=50,
        ),
    )

    assert result.qa.passed
    assert result.qa.source_text_preserved
    assert "".join(cue.source_text for cue in result.cues) == text
    assert all(len(line) <= 10 for cue in result.cues for line in cue.text.split("\n"))


def test_english_long_line_is_split_without_loss_or_overlap() -> None:
    text = (
        "A deliberately long English sentence demonstrates readable line "
        "breaking without deleting punctuation, spaces, or Unicode."
    )
    policy = CuePolicy(
        max_characters_per_line=18,
        max_lines=2,
        max_reading_speed=14,
        min_cue_ms=600,
        max_cue_ms=2600,
        gap_ms=75,
    )
    result = arrange_cues(
        [{"startMs": 100, "endMs": 3000, "text": text}],
        policy=policy,
    )

    assert len(result.cues) > 1
    assert "".join(cue.source_text for cue in result.cues) == text
    for left, right in zip(result.cues, result.cues[1:], strict=False):
        assert right.start_ms - left.end_ms >= policy.gap_ms


def test_speaker_labels_and_unicode_survive_all_text_exports() -> None:
    result = arrange_cues(
        [
            {
                "start": 0,
                "end": 3,
                "text": "你好，世界 👋 café مرحبا",
                "speaker": "嘉宾甲",
            }
        ],
        policy=CuePolicy(
            max_characters_per_line=24,
            max_lines=2,
            include_speaker_labels=True,
            speaker_label_template="【{speaker}】",
        ),
    )

    assert "【嘉宾甲】" in result.cues[0].text
    for subtitle_format in SubtitleFormat:
        exported = export_subtitles(result, subtitle_format)
        assert "嘉宾甲" in exported
        assert "你好" in exported
        assert "café" in exported
        assert "مرحبا" in exported


def test_overlapping_segments_are_repaired_monotonically_and_recorded() -> None:
    policy = CuePolicy(min_cue_ms=800, max_cue_ms=2500, gap_ms=120)
    result = arrange_cues(
        [
            {"start": 0, "end": 2, "text": "first", "speaker": "A"},
            {"start": 1, "end": 2.2, "text": "second", "speaker": "B"},
        ],
        policy=policy,
    )

    assert result.cues[1].start_ms >= result.cues[0].end_ms + policy.gap_ms
    assert any(repair.startswith("shifted-overlap:segment-2") for repair in result.qa.repairs)


def test_extremely_long_unbroken_unicode_is_hard_split_within_limits() -> None:
    text = "超" * 200 + "🚀" * 20
    policy = CuePolicy(
        max_characters_per_line=8,
        max_lines=2,
        max_reading_speed=20,
        min_cue_ms=400,
        max_cue_ms=1200,
        gap_ms=20,
    )
    result = arrange_cues(
        [{"start": 0, "end": 3, "text": text}],
        policy=policy,
    )

    assert "".join(cue.source_text for cue in result.cues) == text
    assert all(len(cue.text.split("\n")) <= 2 for cue in result.cues)
    assert all(
        len(line) <= 8 for cue in result.cues for line in cue.text.split("\n")
    )


def test_srt_vtt_and_ass_have_expected_syntax_and_timestamps() -> None:
    result = arrange_cues(
        [{"startMs": 1234, "endMs": 3234, "text": "Caption text"}]
    )

    srt = export_subtitles(result, "srt")
    vtt = export_subtitles(result, "webvtt")
    ass = export_subtitles(result, "ass", theme="youtube-bold")

    assert "00:00:01,234 --> 00:00:03,234" in srt
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:01.234 --> 00:00:03.234" in vtt
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert "Dialogue: 0,0:00:01.23,0:00:03.23" in ass


def test_all_named_themes_resolve_and_custom_requires_style() -> None:
    for theme in SubtitleTheme:
        if theme is SubtitleTheme.CUSTOM:
            continue
        assert style_for_theme(theme).font_family

    with pytest.raises(ValueError, match="requires custom_style"):
        style_for_theme(SubtitleTheme.CUSTOM)
    custom = SubtitleStyle(font_family="User Font", font_size=44)
    assert style_for_theme(SubtitleTheme.CUSTOM, custom_style=custom) is custom


def test_source_is_never_an_output_and_plans_are_non_executing(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    sidecar = tmp_path / "source.zh-CN.ass"
    muxed = tmp_path / "source.subtitled.mkv"
    burned = tmp_path / "source.burned.mp4"

    with pytest.raises(SourceProtectionError, match="immutable"):
        build_subtitle_output_plan(
            source_path=source,
            output_path=source,
            subtitle_format="ass",
            mode="burn-in",
            subtitle_path=sidecar,
        )

    sidecar_plan = build_subtitle_output_plan(
        source_path=source,
        output_path=sidecar,
        subtitle_format="ass",
        mode="sidecar",
    )
    assert sidecar_plan.ffmpeg is None
    assert sidecar_plan.source_preserved
    assert sidecar_plan.workflow_reversible

    mux_plan = build_subtitle_output_plan(
        source_path=source,
        output_path=muxed,
        subtitle_path=sidecar,
        subtitle_format="ass",
        mode=SubtitleOutputMode.SOFT_MUX,
    )
    assert mux_plan.ffmpeg is not None
    assert mux_plan.ffmpeg.operation == "soft-mux"
    assert "-c" in mux_plan.ffmpeg.arguments
    assert "copy" in mux_plan.ffmpeg.arguments
    assert not mux_plan.ffmpeg.execution_permitted
    assert mux_plan.derived_media_reversible

    burn_plan = build_subtitle_output_plan(
        source_path=source,
        output_path=burned,
        subtitle_path=sidecar,
        subtitle_format=SubtitleFormat.ASS,
        mode=SubtitleOutputMode.BURN_IN,
    )
    assert burn_plan.ffmpeg is not None
    assert burn_plan.ffmpeg.operation == "burn-in"
    assert "-vf" in burn_plan.ffmpeg.arguments
    assert burn_plan.source_preserved
    assert burn_plan.workflow_reversible
    assert not burn_plan.derived_media_reversible
    assert not burn_plan.ffmpeg.ready_for_execution


def test_subtitle_schema_covers_modes_formats_themes_fonts_and_protection() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "subtitle-output.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    payload = {
        "schemaVersion": "1.0.0",
        "mode": "burn-in",
        "format": "ass",
        "theme": "custom",
        "sourcePath": "D:\\media\\source.mp4",
        "outputPath": "D:\\media\\source.captioned.mp4",
        "subtitlePath": "D:\\media\\source.ass",
        "sourceProtection": {
            "preserveSource": True,
            "overwriteSource": False,
            "copyToNewFile": True,
        },
        "cuePolicy": {
            "maxCharactersPerLine": 22,
            "maxLines": 2,
            "maxReadingSpeed": 17,
            "minCueMs": 900,
            "maxCueMs": 7000,
            "gapMs": 80,
            "punctuationPriority": ["。", "！", "？"],
        },
        "style": {
            "font": {
                "family": "Noto Sans CJK SC",
                "fallbacks": ["Noto Sans", "Segoe UI"],
                "size": 58,
                "weight": 600,
                "italic": False,
            },
            "primaryColor": "#FFFFFF",
            "activeWordColor": "#FFD60A",
            "outline": {"color": "#000000", "width": 3},
            "shadow": {"depth": 1.2},
            "background": {"color": "#111827", "opacity": 0.4},
            "safeArea": {"horizontal": 90, "vertical": 70},
            "alignment": 2,
        },
    }

    assert not list(validator.iter_errors(payload))
    unsafe = dict(payload)
    unsafe["sourceProtection"] = {
        "preserveSource": False,
        "overwriteSource": True,
        "copyToNewFile": False,
    }
    assert list(validator.iter_errors(unsafe))
