"""Pure subtitle-domain primitives, layout QA, and non-executing media plans.

This module deliberately does not invoke ffmpeg, ffprobe, a shell, or any
other external process.  It owns deterministic cue composition and describes
future media operations without granting them execution authority.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import ntpath
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any


SUBTITLE_SCHEMA_VERSION = "1.0.0"
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_DEFAULT_PUNCTUATION = tuple("。！？!?；;：:,，、.")
_SPEAKER_COLOR_ALGORITHMS = {
    "automatic": "oklch-hash-v1",
    "accessible": "accessible-oklch-hash-v1",
    "monochrome": "monochrome-v1",
}
_SPEAKER_COLOR_DEFAULT_SEED = "MediaTranscribeStudio"
_GOLDEN_ANGLE_DEGREES = 137.50776405003785
_SPEAKER_COLOR_INDEX_MODULUS = 2_147_483_647


class SubtitleError(ValueError):
    """Base error for invalid subtitle-domain input."""


class SubtitleQAError(SubtitleError):
    """Raised when generated cues violate a hard subtitle invariant."""


class SourceProtectionError(SubtitleError):
    """Raised when a requested output could replace or alias the source."""


class SubtitleFormat(str, Enum):
    SRT = "srt"
    WEBVTT = "webvtt"
    ASS = "ass"


class SubtitleOutputMode(str, Enum):
    SIDECAR = "sidecar"
    SOFT_MUX = "soft-mux"
    BURN_IN = "burn-in"


class SubtitleTheme(str, Enum):
    YOUTUBE_CLEAN = "youtube-clean"
    YOUTUBE_BOLD = "youtube-bold"
    MINIMAL_GLASS = "minimal-glass"
    KARAOKE_HIGHLIGHT = "karaoke-highlight"
    SPEAKER_COLOR = "speaker-color"
    DOCUMENTARY = "documentary"
    NEWS_LOWER_THIRD = "news-lower-third"
    CUSTOM = "custom"


@dataclass(frozen=True)
class SubtitleStyle:
    """Portable style intent shared by UI, ASS export, and future renderers."""

    font_family: str = "Noto Sans CJK SC"
    font_fallbacks: tuple[str, ...] = (
        "Noto Sans",
        "Segoe UI",
        "Arial Unicode MS",
    )
    font_size: int = 58
    font_weight: int = 600
    italic: bool = False
    primary_color: str = "#FFFFFF"
    active_word_color: str = "#FFD60A"
    outline_color: str = "#000000"
    outline_width: float = 3.0
    shadow_depth: float = 1.2
    background_color: str = "#111827"
    background_opacity: float = 0.0
    margin_horizontal: int = 90
    margin_vertical: int = 70
    alignment: int = 2

    def __post_init__(self) -> None:
        if not self.font_family.strip():
            raise SubtitleError("font_family must be non-empty")
        if any(not value.strip() for value in self.font_fallbacks):
            raise SubtitleError("font_fallbacks must contain non-empty names")
        if not 8 <= self.font_size <= 240:
            raise SubtitleError("font_size must be between 8 and 240")
        if self.font_weight < 100 or self.font_weight > 900:
            raise SubtitleError("font_weight must be between 100 and 900")
        for field_name in (
            "primary_color",
            "active_word_color",
            "outline_color",
            "background_color",
        ):
            if not _HEX_COLOR.fullmatch(getattr(self, field_name)):
                raise SubtitleError(f"{field_name} must be a #RRGGBB color")
        if not 0.0 <= self.outline_width <= 12.0:
            raise SubtitleError("outline_width must be between 0 and 12")
        if not 0.0 <= self.shadow_depth <= 12.0:
            raise SubtitleError("shadow_depth must be between 0 and 12")
        if not 0.0 <= self.background_opacity <= 1.0:
            raise SubtitleError("background_opacity must be between 0 and 1")
        if self.margin_horizontal < 0 or self.margin_vertical < 0:
            raise SubtitleError("subtitle margins cannot be negative")
        if self.alignment not in range(1, 10):
            raise SubtitleError("alignment must use the ASS keypad range 1..9")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["font_fallbacks"] = list(self.font_fallbacks)
        return value


@dataclass(frozen=True)
class SpeakerColorAssignment:
    """One auditable speaker-to-ASS-style color binding."""

    speaker_id: str
    color: str
    source: str
    palette_index: int | None
    style_name: str

    def __post_init__(self) -> None:
        if not self.speaker_id.strip():
            raise SubtitleError("speaker color speaker_id must be non-empty")
        if not _HEX_COLOR.fullmatch(self.color):
            raise SubtitleError("speaker color must be a #RRGGBB color")
        if self.source not in {"override", "palette"}:
            raise SubtitleError(
                "speaker color source must be override or palette"
            )
        if self.source == "override" and self.palette_index is not None:
            raise SubtitleError(
                "override speaker colors cannot declare a palette index"
            )
        if self.source == "palette" and (
            isinstance(self.palette_index, bool)
            or not isinstance(self.palette_index, int)
            or self.palette_index < 0
        ):
            raise SubtitleError(
                "palette speaker colors require a non-negative palette index"
            )
        if not re.fullmatch(r"MTS-Speaker-\d{4,}", self.style_name):
            raise SubtitleError("speaker color style_name is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "speakerId": self.speaker_id,
            "color": self.color.upper(),
            "source": self.source,
            "paletteIndex": self.palette_index,
            "styleName": self.style_name,
        }

    def visual_qa_dict(self) -> dict[str, str]:
        """Return the exact declaration accepted by subtitle visual QA."""

        return {
            "speakerId": self.speaker_id,
            "color": self.color.upper(),
        }


_THEME_STYLES = MappingProxyType(
    {
        SubtitleTheme.YOUTUBE_CLEAN: SubtitleStyle(),
        SubtitleTheme.YOUTUBE_BOLD: SubtitleStyle(
            font_family="Noto Sans CJK SC",
            font_size=64,
            font_weight=800,
            active_word_color="#FFE66D",
            outline_width=4.0,
            shadow_depth=1.6,
            margin_vertical=78,
        ),
        SubtitleTheme.MINIMAL_GLASS: SubtitleStyle(
            font_family="Inter",
            font_fallbacks=("Noto Sans CJK SC", "Segoe UI", "Noto Sans"),
            font_size=52,
            font_weight=550,
            outline_width=0.8,
            shadow_depth=1.0,
            background_color="#111827",
            background_opacity=0.68,
            margin_horizontal=110,
            margin_vertical=82,
        ),
        SubtitleTheme.KARAOKE_HIGHLIGHT: SubtitleStyle(
            font_family="Noto Sans CJK SC",
            font_size=62,
            font_weight=750,
            active_word_color="#FFD60A",
            outline_width=3.5,
            shadow_depth=1.4,
        ),
        SubtitleTheme.SPEAKER_COLOR: SubtitleStyle(
            font_family="Noto Sans CJK SC",
            font_size=58,
            font_weight=700,
            outline_width=3.2,
            shadow_depth=1.2,
        ),
        SubtitleTheme.DOCUMENTARY: SubtitleStyle(
            font_family="Noto Serif CJK SC",
            font_fallbacks=("Noto Serif", "Georgia", "Noto Sans CJK SC"),
            font_size=50,
            font_weight=500,
            primary_color="#F7F3E8",
            active_word_color="#F7F3E8",
            outline_width=2.2,
            shadow_depth=1.0,
            margin_vertical=74,
        ),
        SubtitleTheme.NEWS_LOWER_THIRD: SubtitleStyle(
            font_family="Source Han Sans SC",
            font_fallbacks=("Noto Sans CJK SC", "Segoe UI", "Noto Sans"),
            font_size=48,
            font_weight=700,
            primary_color="#FFFFFF",
            active_word_color="#7DD3FC",
            outline_width=0.8,
            shadow_depth=0.6,
            background_color="#0F2747",
            background_opacity=0.90,
            margin_horizontal=72,
            margin_vertical=54,
            alignment=1,
        ),
    }
)


def style_for_theme(
    theme: SubtitleTheme | str,
    *,
    custom_style: SubtitleStyle | None = None,
) -> SubtitleStyle:
    """Resolve one preset, requiring explicit style data for ``custom``."""

    normalized = _coerce_enum(SubtitleTheme, theme, "theme")
    if normalized is SubtitleTheme.CUSTOM:
        if custom_style is None:
            raise SubtitleError("custom theme requires custom_style")
        return custom_style
    if custom_style is not None:
        raise SubtitleError("custom_style is only valid for the custom theme")
    return _THEME_STYLES[normalized]


@dataclass(frozen=True)
class CuePolicy:
    """Hard layout and timing limits for deterministic cue composition."""

    max_characters_per_line: int = 22
    max_lines: int = 2
    max_reading_speed: float = 17.0
    min_cue_ms: int = 900
    max_cue_ms: int = 7000
    gap_ms: int = 80
    punctuation_priority: tuple[str, ...] = _DEFAULT_PUNCTUATION
    include_speaker_labels: bool = False
    speaker_label_template: str = "[{speaker}] "

    def __post_init__(self) -> None:
        if not 4 <= self.max_characters_per_line <= 160:
            raise SubtitleError(
                "max_characters_per_line must be between 4 and 160"
            )
        if not 1 <= self.max_lines <= 6:
            raise SubtitleError("max_lines must be between 1 and 6")
        if not math.isfinite(self.max_reading_speed):
            raise SubtitleError("max_reading_speed must be finite")
        if not 1.0 <= self.max_reading_speed <= 100.0:
            raise SubtitleError("max_reading_speed must be between 1 and 100")
        if not 100 <= self.min_cue_ms <= 30_000:
            raise SubtitleError("min_cue_ms must be between 100 and 30000")
        if not self.min_cue_ms <= self.max_cue_ms <= 60_000:
            raise SubtitleError(
                "max_cue_ms must be between min_cue_ms and 60000"
            )
        if not 0 <= self.gap_ms <= 5000:
            raise SubtitleError("gap_ms must be between 0 and 5000")
        if any(not isinstance(item, str) or len(item) != 1 for item in self.punctuation_priority):
            raise SubtitleError(
                "punctuation_priority must contain single-character strings"
            )
        if self.include_speaker_labels and "{speaker}" not in self.speaker_label_template:
            raise SubtitleError(
                "speaker_label_template must contain {speaker} when labels are enabled"
            )


@dataclass(frozen=True)
class SubtitleSegment:
    """One source transcript segment expressed in integer milliseconds."""

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start_ms, bool) or not isinstance(self.start_ms, int):
            raise SubtitleError("start_ms must be an integer")
        if isinstance(self.end_ms, bool) or not isinstance(self.end_ms, int):
            raise SubtitleError("end_ms must be an integer")
        if self.start_ms < 0:
            raise SubtitleError("segment start cannot be negative")
        if self.end_ms <= self.start_ms:
            raise SubtitleError("segment end must be after segment start")
        if not isinstance(self.text, str) or not self.text:
            raise SubtitleError("segment text must be a non-empty string")
        if self.speaker is not None and (
            not isinstance(self.speaker, str) or not self.speaker.strip()
        ):
            raise SubtitleError("speaker must be a non-empty string when present")
        if self.speaker_id is not None and (
            not isinstance(self.speaker_id, str) or not self.speaker_id.strip()
        ):
            raise SubtitleError(
                "speaker_id must be a non-empty string when present"
            )

    @classmethod
    def from_value(
        cls,
        value: SubtitleSegment | Mapping[str, Any],
    ) -> SubtitleSegment:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise SubtitleError("segments must be SubtitleSegment values or mappings")

        has_milliseconds = "startMs" in value or "endMs" in value
        has_seconds = "start" in value or "end" in value
        if has_milliseconds and has_seconds:
            raise SubtitleError(
                "segment timing cannot mix start/end with startMs/endMs"
            )
        if has_milliseconds:
            if "startMs" not in value or "endMs" not in value:
                raise SubtitleError("segment requires both startMs and endMs")
            start_ms = _integer_milliseconds(value["startMs"], "startMs")
            end_ms = _integer_milliseconds(value["endMs"], "endMs")
        else:
            if "start" not in value or "end" not in value:
                raise SubtitleError("segment requires start/end or startMs/endMs")
            start_ms = _seconds_to_milliseconds(value["start"], "start")
            end_ms = _seconds_to_milliseconds(value["end"], "end")

        speaker_id_value = value.get("speakerId")
        speaker_value = value.get("speaker", speaker_id_value)
        return cls(
            start_ms=start_ms,
            end_ms=end_ms,
            text=value.get("text"),  # type: ignore[arg-type]
            speaker=speaker_value,  # type: ignore[arg-type]
            speaker_id=speaker_id_value,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SubtitleCue:
    number: int
    start_ms: int
    end_ms: int
    text: str
    source_text: str
    source_segment_index: int
    speaker: str | None = None
    speaker_id: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def reading_units(self) -> int:
        return _reading_units(self.text)


@dataclass(frozen=True)
class SubtitleQAIssue:
    code: str
    message: str
    cue_number: int | None = None


@dataclass(frozen=True)
class SubtitleQAReport:
    passed: bool
    issues: tuple[SubtitleQAIssue, ...]
    cue_count: int
    source_text_preserved: bool
    maximum_observed_reading_speed: float
    repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubtitleArrangement:
    cues: tuple[SubtitleCue, ...]
    qa: SubtitleQAReport
    speaker_colors: tuple[SpeakerColorAssignment, ...] = ()

    def visual_qa_speakers(self) -> tuple[dict[str, str], ...]:
        """Declare only colors that are actually rendered into the ASS."""

        return tuple(
            assignment.visual_qa_dict()
            for assignment in self.speaker_colors
        )


@dataclass(frozen=True)
class SourceBoundCueResult:
    """Cue arrangement plus evidence for any source-timeline fallback."""

    arrangement: SubtitleArrangement
    configured_policy: CuePolicy
    effective_policy: CuePolicy
    configured_policy_qa: SubtitleQAReport
    source_duration_ms: int
    source_segments_monotonic_nonoverlapping: bool
    source_bound_fallback_applied: bool

    def timing_policy_evidence(self) -> dict[str, Any]:
        return {
            "configuredGapMs": self.configured_policy.gap_ms,
            "effectiveGapMs": self.effective_policy.gap_ms,
            "configuredMinCueMs": self.configured_policy.min_cue_ms,
            "effectiveMinCueMs": self.effective_policy.min_cue_ms,
            "configuredMaxReadingSpeed": (
                self.configured_policy.max_reading_speed
            ),
            "effectiveMaxReadingSpeed": (
                self.effective_policy.max_reading_speed
            ),
            "sourceDurationMs": self.source_duration_ms,
            "lastCueEndMs": self.arrangement.cues[-1].end_ms,
            "withinSourceDuration": True,
            "sourceSegmentsMonotonicNonoverlapping": (
                self.source_segments_monotonic_nonoverlapping
            ),
            "zeroGapApplied": self.effective_policy.gap_ms == 0,
            "sourceBoundFallbackApplied": (
                self.source_bound_fallback_applied
            ),
            "configuredPolicyQaPassed": self.configured_policy_qa.passed,
            "configuredPolicyIssues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "cueNumber": issue.cue_number,
                }
                for issue in self.configured_policy_qa.issues
            ],
        }


def arrange_cues(
    segments: Sequence[SubtitleSegment | Mapping[str, Any]],
    *,
    policy: CuePolicy | None = None,
) -> SubtitleArrangement:
    """Compose monotonic, non-overlapping cues without dropping source text.

    ``start``/``end`` mapping fields are interpreted as seconds.  Existing
    worker-style ``startMs``/``endMs`` fields are also accepted.  Input order
    is authoritative; overlapping timings are shifted forward and recorded as
    deterministic repairs rather than silently reordering transcript text.
    """

    resolved_policy = policy or CuePolicy()
    normalized = tuple(SubtitleSegment.from_value(value) for value in segments)
    if not normalized:
        raise SubtitleError("at least one subtitle segment is required")

    cues: list[SubtitleCue] = []
    repairs: list[str] = []
    previous_end = -resolved_policy.gap_ms

    for segment_index, segment in enumerate(normalized):
        label = _speaker_label(segment.speaker, resolved_policy)
        chunks = _compose_segment_chunks(
            segment.text,
            label=label,
            policy=resolved_policy,
        )
        required_durations = tuple(
            _required_duration_ms(text, resolved_policy)
            for _, text in chunks
        )
        durations = _allocate_durations(
            required_durations,
            source_duration_ms=segment.end_ms - segment.start_ms,
            gap_ms=resolved_policy.gap_ms,
            maximum_ms=resolved_policy.max_cue_ms,
        )

        segment_start = max(
            segment.start_ms,
            previous_end + resolved_policy.gap_ms,
        )
        if segment_start > segment.start_ms:
            repairs.append(
                f"shifted-overlap:segment-{segment_index + 1}:"
                f"{segment.start_ms}->{segment_start}"
            )

        cursor = segment_start
        for chunk_index, ((source_text, display_text), duration_ms) in enumerate(
            zip(chunks, durations, strict=True)
        ):
            if chunk_index:
                cursor += resolved_policy.gap_ms
            end_ms = cursor + duration_ms
            cue = SubtitleCue(
                number=len(cues) + 1,
                start_ms=cursor,
                end_ms=end_ms,
                text=display_text,
                source_text=source_text,
                source_segment_index=segment_index,
                speaker=segment.speaker,
                speaker_id=segment.speaker_id,
            )
            cues.append(cue)
            cursor = end_ms
        previous_end = cues[-1].end_ms

    report = audit_cues(
        cues,
        policy=resolved_policy,
        source_segments=normalized,
        repairs=repairs,
    )
    if not report.passed:
        detail = "; ".join(
            f"{issue.code}: {issue.message}" for issue in report.issues[:5]
        )
        raise SubtitleQAError(f"generated subtitle cues failed QA: {detail}")
    return SubtitleArrangement(cues=tuple(cues), qa=report)


def audit_cues(
    cues: Sequence[SubtitleCue],
    *,
    policy: CuePolicy,
    source_segments: Sequence[SubtitleSegment] | None = None,
    repairs: Sequence[str] = (),
) -> SubtitleQAReport:
    """Audit final cue timing, layout, speed, numbering, and text fidelity."""

    issues: list[SubtitleQAIssue] = []
    previous_end: int | None = None
    maximum_speed = 0.0

    for expected_number, cue in enumerate(cues, start=1):
        if cue.number != expected_number:
            issues.append(
                SubtitleQAIssue(
                    "numbering",
                    f"expected cue number {expected_number}, got {cue.number}",
                    cue.number,
                )
            )
        if cue.start_ms < 0 or cue.end_ms <= cue.start_ms:
            issues.append(
                SubtitleQAIssue(
                    "invalid-time-range",
                    "cue must have a non-negative start and a positive duration",
                    cue.number,
                )
            )
        if previous_end is not None:
            if cue.start_ms < previous_end:
                issues.append(
                    SubtitleQAIssue(
                        "overlap",
                        f"cue starts at {cue.start_ms} before {previous_end}",
                        cue.number,
                    )
                )
            elif cue.start_ms - previous_end < policy.gap_ms:
                issues.append(
                    SubtitleQAIssue(
                        "gap",
                        f"cue gap is below {policy.gap_ms} ms",
                        cue.number,
                    )
                )
        previous_end = cue.end_ms

        if not policy.min_cue_ms <= cue.duration_ms <= policy.max_cue_ms:
            issues.append(
                SubtitleQAIssue(
                    "duration",
                    (
                        f"cue duration {cue.duration_ms} ms is outside "
                        f"{policy.min_cue_ms}..{policy.max_cue_ms}"
                    ),
                    cue.number,
                )
            )

        lines = cue.text.split("\n")
        if len(lines) > policy.max_lines:
            issues.append(
                SubtitleQAIssue(
                    "line-count",
                    f"cue has {len(lines)} lines; maximum is {policy.max_lines}",
                    cue.number,
                )
            )
        for line in lines:
            if not line or line != line.strip():
                issues.append(
                    SubtitleQAIssue(
                        "line-whitespace",
                        "cue lines must be non-empty and have no edge whitespace",
                        cue.number,
                    )
                )
            if len(line) > policy.max_characters_per_line:
                issues.append(
                    SubtitleQAIssue(
                        "line-length",
                        (
                            f"line has {len(line)} characters; maximum is "
                            f"{policy.max_characters_per_line}"
                        ),
                        cue.number,
                    )
                )

        speed = cue.reading_units / max(cue.duration_ms / 1000.0, 0.001)
        maximum_speed = max(maximum_speed, speed)
        if speed > policy.max_reading_speed + 1e-9:
            issues.append(
                SubtitleQAIssue(
                    "reading-speed",
                    (
                        f"reading speed {speed:.3f} exceeds "
                        f"{policy.max_reading_speed:.3f}"
                    ),
                    cue.number,
                )
            )

    source_preserved = True
    if source_segments is not None:
        reconstructed = [""] * len(source_segments)
        for cue in cues:
            if not 0 <= cue.source_segment_index < len(source_segments):
                source_preserved = False
                issues.append(
                    SubtitleQAIssue(
                        "source-index",
                        "cue references an unknown source segment",
                        cue.number,
                    )
                )
                continue
            reconstructed[cue.source_segment_index] += cue.source_text
        for index, (actual, source) in enumerate(
            zip(reconstructed, source_segments, strict=True)
        ):
            if actual != source.text:
                source_preserved = False
                issues.append(
                    SubtitleQAIssue(
                        "source-text",
                        f"source segment {index + 1} was changed or lost",
                    )
                )

    return SubtitleQAReport(
        passed=not issues,
        issues=tuple(issues),
        cue_count=len(cues),
        source_text_preserved=source_preserved,
        maximum_observed_reading_speed=maximum_speed,
        repairs=tuple(repairs),
    )


def arrange_cues_within_source_duration(
    segments: Sequence[SubtitleSegment | Mapping[str, Any]],
    *,
    source_duration_ms: int,
    policy: CuePolicy | None = None,
) -> SourceBoundCueResult:
    """Arrange cues without allowing presentation timing to outlive the source.

    When already-monotonic source segments overflow only because presentation
    timing is stricter than their persisted timeline, a source-fit preview is
    attempted. It removes inter-cue gaps and lowers the effective minimum cue
    duration only as far as the shortest source segment can contain. The
    original policy is still audited and returned as evidence; this fallback
    does not grant readability or release approval.
    """

    if (
        isinstance(source_duration_ms, bool)
        or not isinstance(source_duration_ms, int)
        or source_duration_ms < 1
    ):
        raise SubtitleError("source_duration_ms must be a positive integer")
    resolved_policy = policy or CuePolicy()
    normalized = tuple(SubtitleSegment.from_value(value) for value in segments)
    configured_arrangement = arrange_cues(
        normalized,
        policy=resolved_policy,
    )
    source_segments_are_monotonic = all(
        right.start_ms >= left.end_ms
        for left, right in zip(normalized, normalized[1:], strict=False)
    )
    fallback_applied = (
        source_segments_are_monotonic
        and configured_arrangement.cues[-1].end_ms > source_duration_ms
    )
    effective_policy = resolved_policy
    arrangement = configured_arrangement
    if fallback_applied:
        # Keep each cue anchored to its source segment. A fixed domain minimum
        # can still push short, contiguous ASR turns past the real media end.
        source_fit_min_cue_ms = min(
            resolved_policy.min_cue_ms,
            *(
                (segment.end_ms - segment.start_ms)
                // len(
                    _compose_segment_chunks(
                        segment.text,
                        label=_speaker_label(segment.speaker, resolved_policy),
                        policy=resolved_policy,
                    )
                )
                for segment in normalized
            ),
        )
        effective_policy = replace(
            resolved_policy,
            gap_ms=0,
            max_reading_speed=100.0,
            min_cue_ms=max(100, source_fit_min_cue_ms),
        )
        arrangement = arrange_cues(normalized, policy=effective_policy)
    if arrangement.cues[-1].end_ms > source_duration_ms:
        raise SubtitleQAError(
            "generated subtitle cues exceed the persisted source duration"
        )
    configured_policy_qa = audit_cues(
        arrangement.cues,
        policy=resolved_policy,
        source_segments=normalized,
        repairs=arrangement.qa.repairs,
    )
    return SourceBoundCueResult(
        arrangement=arrangement,
        configured_policy=resolved_policy,
        effective_policy=effective_policy,
        configured_policy_qa=configured_policy_qa,
        source_duration_ms=source_duration_ms,
        source_segments_monotonic_nonoverlapping=(
            source_segments_are_monotonic
        ),
        source_bound_fallback_applied=fallback_applied,
    )


def export_subtitles(
    cues: SubtitleArrangement | Sequence[SubtitleCue],
    subtitle_format: SubtitleFormat | str,
    *,
    theme: SubtitleTheme | str = SubtitleTheme.YOUTUBE_CLEAN,
    custom_style: SubtitleStyle | None = None,
    title: str = "MediaTranscribeStudio subtitles",
    speaker_color_mode: str | None = None,
    speaker_color_seed: str | None = None,
    speaker_color_overrides: Mapping[str, str] | None = None,
    speaker_color_algorithm: str | None = None,
    speaker_color_assignments: Sequence[SpeakerColorAssignment] | None = None,
) -> str:
    """Export an already-QA'd cue sequence to SRT, WebVTT, or ASS text."""

    arrangement = cues if isinstance(cues, SubtitleArrangement) else None
    values = arrangement.cues if arrangement is not None else tuple(cues)
    if not values:
        raise SubtitleError("cannot export an empty cue sequence")
    resolved_format = _coerce_enum(
        SubtitleFormat,
        subtitle_format,
        "subtitle_format",
    )
    resolved_theme = _coerce_enum(SubtitleTheme, theme, "theme")
    if resolved_format is SubtitleFormat.SRT:
        return _export_srt(values)
    if resolved_format is SubtitleFormat.WEBVTT:
        return _export_webvtt(values)
    style = style_for_theme(resolved_theme, custom_style=custom_style)
    resolved_color_mode = speaker_color_mode
    if (
        resolved_color_mode is None
        and resolved_theme is SubtitleTheme.SPEAKER_COLOR
    ):
        resolved_color_mode = "automatic"
    if (
        speaker_color_assignments is None
        and arrangement is not None
        and arrangement.speaker_colors
    ):
        speaker_color_assignments = arrangement.speaker_colors
        if resolved_color_mode is None:
            resolved_color_mode = "automatic"
    if speaker_color_assignments is None:
        assignments = resolve_speaker_colors(
            values,
            mode=resolved_color_mode,
            seed=speaker_color_seed,
            overrides=speaker_color_overrides,
            algorithm=speaker_color_algorithm,
        )
    else:
        assignments = tuple(speaker_color_assignments)
        _validate_speaker_color_assignments(
            values,
            assignments,
            mode=resolved_color_mode,
        )
    return _export_ass(
        values,
        style=style,
        title=title,
        speaker_color_mode=resolved_color_mode,
        speaker_color_seed=speaker_color_seed,
        speaker_color_algorithm=speaker_color_algorithm,
        speaker_color_assignments=assignments,
    )


@dataclass(frozen=True)
class FFmpegExecutionPlan:
    """A reviewable argument plan; this object never launches a process."""

    executable: str
    arguments: tuple[str, ...]
    operation: str
    requires_content_probe: bool
    unresolved_capabilities: tuple[str, ...]
    execution_permitted: bool = False

    @property
    def ready_for_execution(self) -> bool:
        return (
            self.execution_permitted
            and not self.requires_content_probe
            and not self.unresolved_capabilities
        )


@dataclass(frozen=True)
class SubtitleOutputPlan:
    """Source-preserving output intent for sidecar, soft mux, or burn-in."""

    schema_version: str
    mode: SubtitleOutputMode
    subtitle_format: SubtitleFormat
    source_path: str
    output_path: str
    subtitle_path: str
    source_preserved: bool
    overwrites_source: bool
    creates_new_file: bool
    workflow_reversible: bool
    derived_media_reversible: bool
    ffmpeg: FFmpegExecutionPlan | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "mode": self.mode.value,
            "format": self.subtitle_format.value,
            "sourcePath": self.source_path,
            "outputPath": self.output_path,
            "subtitlePath": self.subtitle_path,
            "sourceProtection": {
                "preserveSource": self.source_preserved,
                "overwriteSource": self.overwrites_source,
                "copyToNewFile": self.creates_new_file,
            },
            "reversibility": {
                "workflowReversible": self.workflow_reversible,
                "derivedMediaReversible": self.derived_media_reversible,
            },
            "ffmpeg": (
                {
                    "executable": self.ffmpeg.executable,
                    "arguments": list(self.ffmpeg.arguments),
                    "operation": self.ffmpeg.operation,
                    "requiresContentProbe": self.ffmpeg.requires_content_probe,
                    "unresolvedCapabilities": list(
                        self.ffmpeg.unresolved_capabilities
                    ),
                    "executionPermitted": self.ffmpeg.execution_permitted,
                    "readyForExecution": self.ffmpeg.ready_for_execution,
                }
                if self.ffmpeg is not None
                else None
            ),
        }


def build_subtitle_output_plan(
    *,
    source_path: str | Path,
    output_path: str | Path,
    subtitle_format: SubtitleFormat | str,
    mode: SubtitleOutputMode | str,
    subtitle_path: str | Path | None = None,
    subtitle_codec: str | None = None,
    video_encoder: str | None = None,
) -> SubtitleOutputPlan:
    """Build a non-executing, source-preserving output plan.

    Sidecar output performs no FFmpeg operation.  Soft mux copies existing
    source streams and adds a selectable subtitle stream.  Burn-in necessarily
    re-encodes video while copying audio when the probed container permits it.
    FFprobe capability validation and encoder selection remain mandatory
    downstream gates.
    """

    resolved_mode = _coerce_enum(SubtitleOutputMode, mode, "mode")
    resolved_format = _coerce_enum(
        SubtitleFormat,
        subtitle_format,
        "subtitle_format",
    )
    source = os.fspath(source_path)
    output = os.fspath(output_path)
    if not source or not output:
        raise SourceProtectionError("source_path and output_path are required")
    if _paths_alias(source, output):
        raise SourceProtectionError(
            "source and output must be different; source media is immutable"
        )

    if resolved_mode is SubtitleOutputMode.SIDECAR:
        if subtitle_path is not None and not _paths_alias(
            os.fspath(subtitle_path),
            output,
        ):
            raise SubtitleError(
                "sidecar mode uses output_path as the subtitle path"
            )
        return SubtitleOutputPlan(
            schema_version=SUBTITLE_SCHEMA_VERSION,
            mode=resolved_mode,
            subtitle_format=resolved_format,
            source_path=source,
            output_path=output,
            subtitle_path=output,
            source_preserved=True,
            overwrites_source=False,
            creates_new_file=True,
            workflow_reversible=True,
            derived_media_reversible=True,
            ffmpeg=None,
        )

    if subtitle_path is None:
        raise SubtitleError(f"{resolved_mode.value} mode requires subtitle_path")
    subtitle = os.fspath(subtitle_path)
    if _paths_alias(source, subtitle):
        raise SourceProtectionError("subtitle_path cannot alias source media")
    if _paths_alias(output, subtitle):
        raise SourceProtectionError(
            "derived media output cannot overwrite the sidecar subtitle"
        )

    if resolved_mode is SubtitleOutputMode.SOFT_MUX:
        codec = subtitle_codec or "<probe-selected-subtitle-codec>"
        unresolved = () if subtitle_codec else ("subtitle-codec",)
        ffmpeg = FFmpegExecutionPlan(
            executable="ffmpeg",
            arguments=(
                "-nostdin",
                "-hide_banner",
                "-i",
                source,
                "-i",
                subtitle,
                "-map",
                "0",
                "-map",
                "1:0",
                "-c",
                "copy",
                "-c:s",
                codec,
                "-metadata:s:s:0",
                "language=und",
                "-disposition:s:0",
                "default",
                "-n",
                output,
            ),
            operation="soft-mux",
            requires_content_probe=True,
            unresolved_capabilities=unresolved,
        )
        return SubtitleOutputPlan(
            schema_version=SUBTITLE_SCHEMA_VERSION,
            mode=resolved_mode,
            subtitle_format=resolved_format,
            source_path=source,
            output_path=output,
            subtitle_path=subtitle,
            source_preserved=True,
            overwrites_source=False,
            creates_new_file=True,
            workflow_reversible=True,
            derived_media_reversible=True,
            ffmpeg=ffmpeg,
        )

    encoder = video_encoder or "<probe-selected-video-encoder>"
    unresolved = () if video_encoder else ("video-encoder",)
    subtitle_filter = _subtitle_filter_argument(subtitle)
    ffmpeg = FFmpegExecutionPlan(
        executable="ffmpeg",
        arguments=(
            "-nostdin",
            "-hide_banner",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            subtitle_filter,
            "-c:v",
            encoder,
            "-c:a",
            "copy",
            "-map_metadata",
            "0",
            "-n",
            output,
        ),
        operation="burn-in",
        requires_content_probe=True,
        unresolved_capabilities=unresolved,
    )
    return SubtitleOutputPlan(
        schema_version=SUBTITLE_SCHEMA_VERSION,
        mode=resolved_mode,
        subtitle_format=resolved_format,
        source_path=source,
        output_path=output,
        subtitle_path=subtitle,
        source_preserved=True,
        overwrites_source=False,
        creates_new_file=True,
        workflow_reversible=True,
        derived_media_reversible=False,
        ffmpeg=ffmpeg,
    )


def _coerce_enum(
    enum_type: type[Enum],
    value: Enum | str,
    field_name: str,
) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise SubtitleError(f"{field_name} must be one of: {allowed}") from exc


def _integer_milliseconds(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubtitleError(f"{field_name} must be an integer")
    return value


def _seconds_to_milliseconds(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise SubtitleError(f"{field_name} must be numeric")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise SubtitleError(f"{field_name} must be numeric") from exc
    if not math.isfinite(seconds):
        raise SubtitleError(f"{field_name} must be finite")
    return int(round(seconds * 1000))


def _speaker_label(speaker: str | None, policy: CuePolicy) -> str:
    if not policy.include_speaker_labels or speaker is None:
        return ""
    return policy.speaker_label_template.format(speaker=speaker)


def _reading_units(text: str) -> int:
    return max(1, sum(not character.isspace() for character in text))


def _visible_whitespace(text: str) -> str:
    """Keep source character positions while making layout whitespace safe."""

    return "".join(" " if character.isspace() else character for character in text)


def _compose_segment_chunks(
    text: str,
    *,
    label: str,
    policy: CuePolicy,
) -> tuple[tuple[str, str], ...]:
    visual_capacity = policy.max_characters_per_line * policy.max_lines
    if len(label) >= visual_capacity:
        raise SubtitleError(
            "speaker label leaves no room for subtitle text under cue limits"
        )
    max_reading_units = max(
        1,
        math.floor(policy.max_reading_speed * policy.max_cue_ms / 1000),
    )
    label_units = _reading_units(label) if label else 0
    if label_units >= max_reading_units:
        raise SubtitleError(
            "speaker label alone exceeds maximum reading-speed capacity"
        )

    chunk_limit = min(
        visual_capacity - len(label),
        max_reading_units - label_units,
    )
    if chunk_limit < 1:
        raise SubtitleError("cue policy cannot fit subtitle text")

    chunks: list[tuple[str, str]] = []
    pending_source_whitespace = ""
    remaining = text
    while remaining:
        split_at = _preferred_break(
            remaining,
            chunk_limit,
            policy.punctuation_priority,
        )
        source_candidate = remaining[:split_at]
        if not source_candidate.strip():
            pending_source_whitespace += source_candidate
            remaining = remaining[split_at:]
            continue
        source_text = pending_source_whitespace + source_candidate
        display_source = label + source_text.strip()
        lines = _wrap_exact(
            display_source,
            policy.max_characters_per_line,
            policy.punctuation_priority,
        )
        if len(lines) > policy.max_lines:
            permitted_display_length = sum(
                len(line) for line in lines[: policy.max_lines]
            )
            split_at = permitted_display_length - len(label)
            if split_at < 1:
                raise SubtitleQAError(
                    "speaker label prevents text from fitting within max_lines"
                )
            source_candidate = remaining[:split_at]
            source_text = pending_source_whitespace + source_candidate
            display_source = label + source_text.strip()
            lines = _wrap_exact(
                display_source,
                policy.max_characters_per_line,
                policy.punctuation_priority,
            )
        remaining = remaining[split_at:]
        pending_source_whitespace = ""
        chunks.append((source_text, "\n".join(lines)))
    if pending_source_whitespace:
        if not chunks:
            raise SubtitleError("subtitle text must contain visible characters")
        source_text, display_text = chunks[-1]
        chunks[-1] = (
            source_text + pending_source_whitespace,
            display_text,
        )
    return tuple(chunks)


def _preferred_break(
    text: str,
    limit: int,
    punctuation: Sequence[str],
) -> int:
    if len(text) <= limit:
        return len(text)
    window = text[:limit]
    threshold = max(1, int(limit * 0.45))
    punctuation_set = frozenset(punctuation)
    punctuation_candidates = [
        index
        for index, character in enumerate(window, start=1)
        if character in punctuation_set and index >= threshold
    ]
    if punctuation_candidates:
        return punctuation_candidates[-1]
    whitespace_candidates = [
        index
        for index, character in enumerate(window, start=1)
        if character.isspace() and index >= threshold
    ]
    if whitespace_candidates:
        return whitespace_candidates[-1]
    return limit


def _wrap_exact(
    text: str,
    width: int,
    punctuation: Sequence[str],
) -> tuple[str, ...]:
    lines: list[str] = []
    # Preserve one character per source code point so chunk boundaries remain
    # index-compatible, while preventing embedded tabs/newlines from leaking
    # into the rendered-line evidence contract.
    remaining = _visible_whitespace(text)
    while remaining:
        split_at = _preferred_break(remaining, width, punctuation)
        rendered_line = remaining[:split_at].strip()
        remaining = remaining[split_at:]
        remaining = remaining.lstrip()
        if rendered_line:
            lines.append(rendered_line)
    if not lines:
        raise SubtitleError("subtitle text must contain visible characters")
    return tuple(lines)


def _required_duration_ms(text: str, policy: CuePolicy) -> int:
    speed_duration = math.ceil(
        _reading_units(text) / policy.max_reading_speed * 1000
    )
    return min(
        policy.max_cue_ms,
        max(policy.min_cue_ms, speed_duration),
    )


def _allocate_durations(
    required: Sequence[int],
    *,
    source_duration_ms: int,
    gap_ms: int,
    maximum_ms: int,
) -> tuple[int, ...]:
    durations = list(required)
    available_for_cues = source_duration_ms - gap_ms * max(0, len(durations) - 1)
    extra = max(0, available_for_cues - sum(durations))
    while extra:
        candidates = [
            index
            for index, duration in enumerate(durations)
            if duration < maximum_ms
        ]
        if not candidates:
            break
        share = max(1, math.ceil(extra / len(candidates)))
        spent = 0
        for index in candidates:
            addition = min(share, maximum_ms - durations[index], extra - spent)
            durations[index] += addition
            spent += addition
            if spent >= extra:
                break
        if spent == 0:
            break
        extra -= spent
    return tuple(durations)


def _format_timestamp(milliseconds: int, *, separator: str) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        f"{separator}{millis:03d}"
    )


def _export_srt(cues: Sequence[SubtitleCue]) -> str:
    blocks = []
    for number, cue in enumerate(cues, start=1):
        blocks.append(
            "\n".join(
                (
                    str(number),
                    (
                        f"{_format_timestamp(cue.start_ms, separator=',')} --> "
                        f"{_format_timestamp(cue.end_ms, separator=',')}"
                    ),
                    cue.text,
                )
            )
        )
    return "\n\n".join(blocks) + "\n"


def _export_webvtt(cues: Sequence[SubtitleCue]) -> str:
    blocks = ["WEBVTT"]
    for cue in cues:
        blocks.append(
            "\n".join(
                (
                    str(cue.number),
                    (
                        f"{_format_timestamp(cue.start_ms, separator='.')} --> "
                        f"{_format_timestamp(cue.end_ms, separator='.')}"
                    ),
                    cue.text,
                )
            )
        )
    return "\n\n".join(blocks) + "\n"


def _ass_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    centiseconds = min(99, int(round(millis / 10)))
    if centiseconds == 100:
        seconds += 1
        centiseconds = 0
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _ass_color(color: str, *, alpha: int = 0) -> str:
    red, green, blue = color[1:3], color[3:5], color[5:7]
    return f"&H{alpha:02X}{blue}{green}{red}".upper()


def _ass_escape(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\r", r"\N")
        .replace("\n", r"\N")
    )


def _speaker_key(cue: SubtitleCue) -> str:
    return cue.speaker_id or cue.speaker or "unknown"


def resolve_speaker_colors(
    cues: Sequence[SubtitleCue],
    *,
    mode: str | None,
    seed: str | None = None,
    overrides: Mapping[str, str] | None = None,
    algorithm: str | None = None,
) -> tuple[SpeakerColorAssignment, ...]:
    """Resolve deterministic, high-separation colors for observed speakers.

    ``monochrome`` and ``None`` deliberately return no declarations.  Colored
    modes use stable speaker identifiers, explicit overrides first, and a
    seeded OKLCH candidate search for all remaining speakers.  The chosen
    palette index is retained so even very large speaker sets remain
    reproducible and auditable without a fixed speaker-count ceiling.
    """

    if mode is None or mode == "monochrome":
        if algorithm not in {None, "monochrome-v1"}:
            raise SubtitleError(
                "speaker color algorithm is incompatible with monochrome mode"
            )
        return ()
    if mode not in {"automatic", "accessible"}:
        raise SubtitleError(f"unsupported speaker color mode: {mode!r}")
    expected_algorithm = _SPEAKER_COLOR_ALGORITHMS[mode]
    if algorithm is not None and algorithm != expected_algorithm:
        raise SubtitleError(
            f"speaker color algorithm {algorithm!r} is incompatible "
            f"with mode {mode!r}"
        )
    resolved_seed = (
        _SPEAKER_COLOR_DEFAULT_SEED if seed is None else seed
    )
    if not isinstance(resolved_seed, str) or not resolved_seed:
        raise SubtitleError("speaker color seed must be a non-empty string")

    aliases: dict[str, set[str]] = {}
    for cue in cues:
        speaker_id = _speaker_key(cue)
        aliases.setdefault(speaker_id, set()).add(speaker_id)
        if cue.speaker:
            aliases[speaker_id].add(cue.speaker)
    if not aliases:
        return ()

    normalized_overrides: dict[str, str] = {}
    for raw_speaker_id, raw_color in (overrides or {}).items():
        if not isinstance(raw_speaker_id, str) or not raw_speaker_id.strip():
            raise SubtitleError(
                "speaker color override identifiers must be non-empty strings"
            )
        if not isinstance(raw_color, str) or not _HEX_COLOR.fullmatch(
            raw_color
        ):
            raise SubtitleError(
                "speaker color overrides must use #RRGGBB colors"
            )
        normalized_overrides[raw_speaker_id] = raw_color.upper()

    speaker_ids = sorted(aliases, key=lambda value: value.encode("utf-8"))
    resolved_overrides: dict[str, str] = {}
    for speaker_id in speaker_ids:
        override_color = normalized_overrides.get(speaker_id)
        if override_color is None:
            matching_aliases = sorted(
                aliases[speaker_id] & normalized_overrides.keys(),
                key=lambda value: value.encode("utf-8"),
            )
            if matching_aliases:
                override_color = normalized_overrides[matching_aliases[0]]
        if override_color is not None:
            resolved_overrides[speaker_id] = override_color

    assignments: list[SpeakerColorAssignment] = []
    used_colors = set(resolved_overrides.values())
    used_oklab = [
        _hex_to_oklab(color)
        for color in sorted(used_colors)
    ]
    for ordinal, speaker_id in enumerate(speaker_ids, start=1):
        override_color = resolved_overrides.get(speaker_id)
        if override_color is not None:
            color = override_color
            source = "override"
            palette_index = None
        else:
            color, palette_index = _select_speaker_palette_color(
                speaker_id,
                seed=resolved_seed,
                mode=mode,
                used_colors=used_colors,
                used_oklab=used_oklab,
            )
            source = "palette"
            used_colors.add(color)
            used_oklab.append(_hex_to_oklab(color))
        assignments.append(
            SpeakerColorAssignment(
                speaker_id=speaker_id,
                color=color,
                source=source,
                palette_index=palette_index,
                style_name=f"MTS-Speaker-{ordinal:04d}",
            )
        )
    return tuple(assignments)


def _select_speaker_palette_color(
    speaker_id: str,
    *,
    seed: str,
    mode: str,
    used_colors: set[str],
    used_oklab: Sequence[tuple[float, float, float]],
) -> tuple[str, int]:
    digest = hashlib.sha256(
        f"{seed}\0{speaker_id}".encode("utf-8")
    ).digest()
    base_index = (
        int.from_bytes(digest[:8], "big")
        % _SPEAKER_COLOR_INDEX_MODULUS
    )
    step = (
        int.from_bytes(digest[8:16], "big")
        % (_SPEAKER_COLOR_INDEX_MODULUS - 1)
    ) + 1
    candidate_count = 48 if len(used_oklab) < 64 else 16
    comparison = (
        tuple(used_oklab)
        if len(used_oklab) <= 256
        else tuple(used_oklab[-256:])
    )
    best: tuple[float, str, int] | None = None
    for attempt in range(candidate_count):
        palette_index = (
            base_index + attempt * step
        ) % _SPEAKER_COLOR_INDEX_MODULUS
        color, oklab = _speaker_palette_candidate(
            palette_index,
            seed=seed,
            mode=mode,
        )
        if color in used_colors:
            continue
        minimum_distance = min(
            (
                math.dist(oklab, existing)
                for existing in comparison
            ),
            default=float("inf"),
        )
        candidate = (minimum_distance, color, palette_index)
        if best is None or candidate > best:
            best = candidate
    if best is not None:
        return best[1], best[2]

    # The 24-bit color space is finite, but this deterministic probe makes
    # collision handling explicit and bounded for very large N.
    palette_index = (
        base_index + candidate_count * step
    ) % _SPEAKER_COLOR_INDEX_MODULUS
    for _ in range(1 << 16):
        color, _ = _speaker_palette_candidate(
            palette_index,
            seed=seed,
            mode=mode,
        )
        if color not in used_colors:
            return color, palette_index
        palette_index = (
            palette_index + step
        ) % _SPEAKER_COLOR_INDEX_MODULUS
    raise SubtitleError("speaker color palette exhausted its collision probe")


def _speaker_palette_candidate(
    palette_index: int,
    *,
    seed: str,
    mode: str,
) -> tuple[str, tuple[float, float, float]]:
    seed_digest = hashlib.sha256(seed.encode("utf-8")).digest()
    seed_phase = (
        int.from_bytes(seed_digest[:8], "big") / float(1 << 64)
    ) * 360.0
    ring = palette_index // 24
    hue = (
        seed_phase
        + palette_index * _GOLDEN_ANGLE_DEGREES
        + (ring % 17) * 7.125
    ) % 360.0
    if mode == "accessible":
        lightness_values = (0.82, 0.74, 0.87, 0.69)
        chroma_values = (0.17, 0.19, 0.14, 0.20)
    else:
        lightness_values = (0.78, 0.70, 0.84, 0.65)
        chroma_values = (0.18, 0.20, 0.15, 0.21)
    lightness = lightness_values[ring % len(lightness_values)]
    chroma = chroma_values[
        (ring // len(lightness_values)) % len(chroma_values)
    ]
    return _oklch_to_srgb_hex(lightness, chroma, hue)


def _oklch_to_srgb_hex(
    lightness: float,
    chroma: float,
    hue_degrees: float,
) -> tuple[str, tuple[float, float, float]]:
    hue_radians = math.radians(hue_degrees)
    for gamut_step in range(12):
        adjusted_chroma = chroma * (0.88**gamut_step)
        alpha = adjusted_chroma * math.cos(hue_radians)
        beta = adjusted_chroma * math.sin(hue_radians)
        linear = _oklab_to_linear_srgb(lightness, alpha, beta)
        if all(-1e-7 <= channel <= 1.0000001 for channel in linear):
            rgb = tuple(
                round(
                    max(0.0, min(1.0, _linear_to_srgb(channel)))
                    * 255
                )
                for channel in linear
            )
            color = "#{:02X}{:02X}{:02X}".format(*rgb)
            return color, (lightness, alpha, beta)

    # This should only be reached for future out-of-range policy values.
    red, green, blue = colorsys.hls_to_rgb(
        (hue_degrees % 360.0) / 360.0,
        max(0.0, min(1.0, lightness)),
        0.72,
    )
    color = "#{:02X}{:02X}{:02X}".format(
        round(red * 255),
        round(green * 255),
        round(blue * 255),
    )
    return color, _hex_to_oklab(color)


def _oklab_to_linear_srgb(
    lightness: float,
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    long = lightness + 0.3963377774 * alpha + 0.2158037573 * beta
    medium = lightness - 0.1055613458 * alpha - 0.0638541728 * beta
    short = lightness - 0.0894841775 * alpha - 1.2914855480 * beta
    long, medium, short = long**3, medium**3, short**3
    return (
        +4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short,
        -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short,
        -0.0041960863 * long - 0.7034186147 * medium + 1.7076147010 * short,
    )


def _linear_to_srgb(channel: float) -> float:
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * (channel ** (1 / 2.4)) - 0.055


def _hex_to_oklab(color: str) -> tuple[float, float, float]:
    red, green, blue = (
        int(color[index : index + 2], 16) / 255
        for index in (1, 3, 5)
    )

    def linear(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = linear(red), linear(green), linear(blue)
    long = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = (
        0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    )
    short = (
        0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    )
    long, medium, short = (
        math.copysign(abs(value) ** (1 / 3), value)
        for value in (long, medium, short)
    )
    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


def _validate_speaker_color_assignments(
    cues: Sequence[SubtitleCue],
    assignments: Sequence[SpeakerColorAssignment],
    *,
    mode: str | None,
) -> None:
    if mode is None or mode == "monochrome":
        if assignments:
            raise SubtitleError(
                "disabled speaker colors cannot declare color assignments"
            )
        return
    expected = {_speaker_key(cue) for cue in cues}
    actual = {assignment.speaker_id for assignment in assignments}
    if len(actual) != len(assignments):
        raise SubtitleError("speaker color assignments repeat speaker_id")
    if actual != expected:
        raise SubtitleError(
            "speaker color assignments must exactly cover rendered speakers"
        )
    style_names = {assignment.style_name for assignment in assignments}
    if len(style_names) != len(assignments):
        raise SubtitleError("speaker color assignments repeat style_name")


def _ass_style_line(
    name: str,
    style: SubtitleStyle,
    *,
    primary_color: str | None = None,
) -> str:
    background_alpha = 255 - round(style.background_opacity * 255)
    border_style = 3 if style.background_opacity > 0 else 1
    bold = -1 if style.font_weight >= 600 else 0
    italic = -1 if style.italic else 0
    return (
        f"Style: {name},"
        f"{style.font_family},{style.font_size},"
        f"{_ass_color(primary_color or style.primary_color)},"
        f"{_ass_color(style.active_word_color)},"
        f"{_ass_color(style.outline_color)},"
        f"{_ass_color(style.background_color, alpha=background_alpha)},"
        f"{bold},{italic},0,0,100,100,0,0,{border_style},"
        f"{style.outline_width:.1f},{style.shadow_depth:.1f},"
        f"{style.alignment},{style.margin_horizontal},"
        f"{style.margin_horizontal},{style.margin_vertical},1"
    )


def _export_ass(
    cues: Sequence[SubtitleCue],
    *,
    style: SubtitleStyle,
    title: str,
    speaker_color_mode: str | None,
    speaker_color_seed: str | None,
    speaker_color_algorithm: str | None,
    speaker_color_assignments: Sequence[SpeakerColorAssignment],
) -> str:
    _validate_speaker_color_assignments(
        cues,
        speaker_color_assignments,
        mode=speaker_color_mode,
    )
    assignment_by_id = {
        assignment.speaker_id: assignment
        for assignment in speaker_color_assignments
    }
    audit_lines: list[str] = []
    if speaker_color_assignments:
        resolved_seed = (
            _SPEAKER_COLOR_DEFAULT_SEED
            if speaker_color_seed is None
            else speaker_color_seed
        )
        algorithm = (
            speaker_color_algorithm
            or _SPEAKER_COLOR_ALGORITHMS[speaker_color_mode or "automatic"]
        )
        audit_lines.extend(
            (
                f"; MTS-SpeakerColor-Mode: {speaker_color_mode}",
                f"; MTS-SpeakerColor-Algorithm: {algorithm}",
                (
                    "; MTS-SpeakerColor-Seed: "
                    + json.dumps(resolved_seed, ensure_ascii=True)
                ),
                (
                    "; MTS-SpeakerColor-Seed-SHA256: "
                    + hashlib.sha256(
                        resolved_seed.encode("utf-8")
                    ).hexdigest()
                ),
                (
                    "; MTS-SpeakerColor-Count: "
                    f"{len(speaker_color_assignments)}"
                ),
            )
        )
        for assignment in speaker_color_assignments:
            audit_lines.append(
                "; MTS-SpeakerColor-Assignment: "
                + json.dumps(
                    assignment.as_dict(),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    header = [
        "[Script Info]",
        f"Title: {title.replace(chr(10), ' ').replace(chr(13), ' ')}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "PlayResX: 1920",
        "PlayResY: 1080",
        *audit_lines,
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        _ass_style_line("Default", style),
        *(
            _ass_style_line(
                assignment.style_name,
                style,
                primary_color=assignment.color,
            )
            for assignment in speaker_color_assignments
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []
    for cue in cues:
        text = _ass_escape(cue.text)
        assignment = assignment_by_id.get(_speaker_key(cue))
        style_name = assignment.style_name if assignment else "Default"
        name = (cue.speaker or "").replace(",", " ")
        events.append(
            "Dialogue: "
            f"0,{_ass_timestamp(cue.start_ms)},{_ass_timestamp(cue.end_ms)},"
            f"{style_name},{name},0,0,0,,{text}"
        )
    return "\n".join((*header, *events)) + "\n"


def _path_key(path: str) -> str:
    if re.match(r"^[A-Za-z]:[\\/]", path) or path.startswith(("\\\\", "//")):
        return ntpath.normcase(ntpath.abspath(ntpath.normpath(path)))
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def _paths_alias(left: str, right: str) -> bool:
    return _path_key(left) == _path_key(right)


def _subtitle_filter_argument(path: str) -> str:
    normalized = path.replace("\\", "/")
    escaped = (
        normalized.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace(",", r"\,")
        .replace(";", r"\;")
    )
    return f"subtitles=filename='{escaped}'"


__all__ = [
    "CuePolicy",
    "FFmpegExecutionPlan",
    "SpeakerColorAssignment",
    "SourceBoundCueResult",
    "SourceProtectionError",
    "SUBTITLE_SCHEMA_VERSION",
    "SubtitleArrangement",
    "SubtitleCue",
    "SubtitleError",
    "SubtitleFormat",
    "SubtitleOutputMode",
    "SubtitleOutputPlan",
    "SubtitleQAError",
    "SubtitleQAIssue",
    "SubtitleQAReport",
    "SubtitleSegment",
    "SubtitleStyle",
    "SubtitleTheme",
    "arrange_cues",
    "arrange_cues_within_source_duration",
    "audit_cues",
    "build_subtitle_output_plan",
    "export_subtitles",
    "resolve_speaker_colors",
    "style_for_theme",
]
