"""Deterministic, offline hard gates for rendered subtitle representative frames.

This module is deliberately an evidence consumer.  It does not render media,
open image files, inspect the operating-system font registry, invoke a GUI,
or access a network.  Upstream code supplies immutable representative-frame
hashes, subtitle geometry, sampled rendered pixels, and renderer/font evidence.
Missing or contradictory evidence fails the relevant gate closed.

Positive font installation or embedding claims are accepted only when they
carry both a verification method and SHA-256-bound evidence.  A successful
glyph/render check does not, by itself, claim that a font is installed or
embedded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SUBTITLE_VISUAL_QA_SCHEMA_VERSION = "1.0.0"
SUBTITLE_VISUAL_QA_REQUEST_KIND = "subtitle-visual-qa-request"
SUBTITLE_VISUAL_QA_RESULT_KIND = "subtitle-visual-qa-result"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_CODEPOINT = re.compile(r"^U\+[0-9A-F]{4,6}$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_TRUE_WORD_TIMING_SOURCES = frozenset(
    {"forced-aligner", "native-word-timestamps", "human-authored"}
)
_WORD_TIMING_SOURCES = _TRUE_WORD_TIMING_SOURCES | frozenset(
    {"segment-interpolation", "synthetic-even-split"}
)
_FONT_EVIDENCE_METHODS = frozenset(
    {
        "not-provided",
        "renderer-glyph-map",
        "font-cmap-and-shaping",
        "directwrite-enumeration",
        "fontconfig-scan",
        "libass-render-report",
        "manual-pixel-audit",
        "container-font-inspection",
    }
)
_INSTALLATION_STATUSES = frozenset({"not-asserted", "verified-installed"})
_EMBEDDING_STATUSES = frozenset(
    {"not-asserted", "verified-embedded", "verified-not-embedded"}
)
_BACKGROUND_CLASSES = frozenset({"dark", "light"})

_GATE_ORDER = (
    "samplingEvidence",
    "safeArea",
    "clipping",
    "contrast",
    "fontGlyph",
    "lineReading",
    "speakerColor",
    "visualOverlap",
    "karaokeAuthenticity",
)


class SubtitleVisualQAInputError(ValueError):
    """Raised when the request is malformed or internally ambiguous."""


@dataclass(frozen=True)
class Rect:
    """Integer pixel rectangle using an exclusive right/bottom edge."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, other: Rect) -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def intersection_area(self, other: Rect) -> int:
        width = max(0, min(self.right, other.right) - max(self.x, other.x))
        height = max(0, min(self.bottom, other.bottom) - max(self.y, other.y))
        return width * height


@dataclass(frozen=True)
class Violation:
    """One deterministic, machine-readable hard-gate failure."""

    gate: str
    code: str
    message: str
    evidence_path: str
    frame_id: str | None = None
    cue_id: str | None = None
    style_id: str | None = None
    measured: int | float | str | None = None
    threshold: int | float | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "code": self.code,
            "message": self.message,
            "evidencePath": self.evidence_path,
            "frameId": self.frame_id,
            "cueId": self.cue_id,
            "styleId": self.style_id,
            "measured": self.measured,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class GateResult:
    """Result of one named gate."""

    name: str
    checked: int
    violations: tuple[Violation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checked": self.checked,
            "violations": [item.to_dict() for item in self.violations],
        }


@dataclass(frozen=True)
class SubtitleVisualQAMetrics:
    frames_expected: int
    frames_evaluated: int
    cues_evaluated: int
    cue_instances_evaluated: int
    contrast_samples_evaluated: int
    minimum_contrast_ratio: float | None
    maximum_reading_speed: float | None
    minimum_speaker_delta_e_2000: float | None
    maximum_overlap_area_px: int
    verified_word_timing_cues: int
    background_classes_covered: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "framesExpected": self.frames_expected,
            "framesEvaluated": self.frames_evaluated,
            "cuesEvaluated": self.cues_evaluated,
            "cueInstancesEvaluated": self.cue_instances_evaluated,
            "contrastSamplesEvaluated": self.contrast_samples_evaluated,
            "minimumContrastRatio": self.minimum_contrast_ratio,
            "maximumReadingSpeed": self.maximum_reading_speed,
            "minimumSpeakerDeltaE2000": self.minimum_speaker_delta_e_2000,
            "maximumOverlapAreaPx": self.maximum_overlap_area_px,
            "verifiedWordTimingCues": self.verified_word_timing_cues,
            "backgroundClassesCovered": list(self.background_classes_covered),
        }


@dataclass(frozen=True)
class FontTruthSummary:
    evidence_instances: int
    installation_verified: int
    installation_not_asserted: int
    embedding_verified: int
    embedding_not_asserted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "claimPolicy": "positive-claims-require-sha256-bound-evidence",
            "evidenceInstances": self.evidence_instances,
            "installation": {
                "verified": self.installation_verified,
                "notAsserted": self.installation_not_asserted,
            },
            "embedding": {
                "verified": self.embedding_verified,
                "notAsserted": self.embedding_not_asserted,
            },
        }


@dataclass(frozen=True)
class SubtitleVisualQAResult:
    """Immutable result with canonical JSON and deterministic input binding."""

    analysis_id: str
    input_sha256: str
    gates: tuple[GateResult, ...]
    metrics: SubtitleVisualQAMetrics
    font_truth: FontTruthSummary

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def to_dict(self) -> dict[str, Any]:
        failures = sorted(
            {item.code for gate in self.gates for item in gate.violations}
        )
        return {
            "kind": SUBTITLE_VISUAL_QA_RESULT_KIND,
            "schemaVersion": SUBTITLE_VISUAL_QA_SCHEMA_VERSION,
            "analysisId": self.analysis_id,
            "passed": self.passed,
            "failClosed": True,
            "inputSha256": self.input_sha256,
            "gateOrder": list(_GATE_ORDER),
            "gates": {gate.name: gate.to_dict() for gate in self.gates},
            "metrics": self.metrics.to_dict(),
            "fontTruth": self.font_truth.to_dict(),
            "failureCodes": failures,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def default_subtitle_visual_qa_policy() -> dict[str, Any]:
    """Return detached, explicit defaults suitable for a request contract."""

    return {
        "safeArea": {
            "leftRatio": 0.05,
            "rightRatio": 0.05,
            "topRatio": 0.05,
            "bottomRatio": 0.05,
        },
        "contrast": {
            "minimumRatio": 4.5,
            "minimumForegroundPixels": 32,
            "minimumBackgroundPixels": 32,
            "requiredBackgroundClasses": ["dark", "light"],
            "darkMaximumLuminance": 0.35,
            "lightMinimumLuminance": 0.65,
        },
        "layout": {
            "maxLines": 2,
            "maxCharactersPerLine": 42,
            "maxReadingSpeed": 17.0,
            "maximumOverlapAreaPx": 0,
        },
        "speakerColor": {
            "minimumDeltaE2000": 18.0,
        },
        "karaoke": {
            "requireVerifiedWordTiming": True,
        },
    }


def canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Serialize JSON deterministically while rejecting NaN and Infinity."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SubtitleVisualQAInputError(
            "value is not canonical JSON data"
        ) from exc


def deterministic_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def evaluate_subtitle_visual_qa(
    request: Mapping[str, Any],
) -> SubtitleVisualQAResult:
    """Evaluate every hard gate against supplied offline rendering evidence.

    Structural ambiguity raises :class:`SubtitleVisualQAInputError`.  Missing
    or inadequate QA evidence remains a valid request but produces a failed
    gate, making it impossible to publish a passing result by omission.
    """

    root = _object(request, "$")
    _keys(
        root,
        required={
            "kind",
            "schemaVersion",
            "analysisId",
            "renderArtifact",
            "policy",
            "sampling",
            "speakers",
            "cues",
            "frames",
        },
        allowed={
            "kind",
            "schemaVersion",
            "analysisId",
            "renderArtifact",
            "policy",
            "sampling",
            "speakers",
            "cues",
            "frames",
        },
        path="$",
    )
    if _string(root["kind"], "$.kind") != SUBTITLE_VISUAL_QA_REQUEST_KIND:
        raise SubtitleVisualQAInputError("$.kind is unsupported")
    if (
        _string(root["schemaVersion"], "$.schemaVersion")
        != SUBTITLE_VISUAL_QA_SCHEMA_VERSION
    ):
        raise SubtitleVisualQAInputError("$.schemaVersion is unsupported")
    analysis_id = _bounded_string(root["analysisId"], "$.analysisId", 1, 160)

    _parse_render_artifact(root["renderArtifact"])
    policy = _parse_policy(root["policy"])
    sampling = _parse_sampling(root["sampling"])
    speakers = _parse_speakers(root["speakers"])
    cues = _parse_cues(root["cues"], speakers)
    frames = _parse_frames(root["frames"], cues)

    buckets: dict[str, list[Violation]] = {
        gate: [] for gate in _GATE_ORDER
    }
    checked = {gate: 0 for gate in _GATE_ORDER}

    represented_cues: set[str] = set()
    contrast_ratios: list[float] = []
    reading_speeds: list[float] = []
    speaker_deltas: list[float] = []
    overlap_areas: list[int] = []
    background_classes: set[str] = set()
    style_background_classes: dict[str, set[str]] = defaultdict(set)
    verified_word_timing_cues = 0
    cue_instances = 0
    font_evidence_instances = 0
    installation_verified = 0
    installation_not_asserted = 0
    embedding_verified = 0
    embedding_not_asserted = 0

    expected_frame_ids = tuple(sampling["expectedFrameIds"])
    actual_frame_ids = tuple(frame["frameId"] for frame in frames)
    expected_set = set(expected_frame_ids)
    actual_set = set(actual_frame_ids)
    checked["samplingEvidence"] += len(expected_frame_ids) + len(cues)
    for frame_id in sorted(expected_set - actual_set):
        _fail(
            buckets,
            "samplingEvidence",
            "expected-frame-missing",
            "A frame named by deterministic sampling evidence is missing.",
            "$.frames",
            frame_id=frame_id,
        )
    for frame_id in sorted(actual_set - expected_set):
        _fail(
            buckets,
            "samplingEvidence",
            "unexpected-frame",
            "A supplied frame is not bound by expectedFrameIds.",
            "$.frames",
            frame_id=frame_id,
        )

    for frame_index, frame in enumerate(frames):
        frame_path = f"$.frames[{frame_index}]"
        frame_rect = Rect(0, 0, frame["widthPx"], frame["heightPx"])
        safe_rect = _safe_rect(frame["widthPx"], frame["heightPx"], policy)
        instances = frame["instances"]
        seen_frame_cues: set[str] = set()

        for instance_index, instance in enumerate(instances):
            cue_instances += 1
            cue_id = instance["cueId"]
            cue = cues[cue_id]
            style_id = cue["styleId"]
            instance_path = f"{frame_path}.instances[{instance_index}]"
            represented_cues.add(cue_id)

            checked["samplingEvidence"] += 1
            if cue_id in seen_frame_cues:
                _fail(
                    buckets,
                    "samplingEvidence",
                    "duplicate-cue-instance",
                    "A frame cannot contain the same cue more than once.",
                    f"{instance_path}.cueId",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                )
            seen_frame_cues.add(cue_id)
            if not cue["startMs"] <= frame["timestampMs"] <= cue["endMs"]:
                _fail(
                    buckets,
                    "samplingEvidence",
                    "frame-outside-cue-time",
                    "Representative frame timestamp is outside the cue interval.",
                    f"{frame_path}.timestampMs",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                    measured=frame["timestampMs"],
                    threshold=f"{cue['startMs']}..{cue['endMs']}",
                )

            bounds = instance["bounds"]
            ink = instance["inkBounds"]

            checked["safeArea"] += 1
            if not safe_rect.contains(ink):
                _fail(
                    buckets,
                    "safeArea",
                    "outside-safe-area",
                    "Rendered subtitle ink leaves the configured safe area.",
                    f"{instance_path}.inkBounds",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                    measured=_rect_text(ink),
                    threshold=_rect_text(safe_rect),
                )

            checked["clipping"] += 1
            if not frame_rect.contains(bounds):
                _fail(
                    buckets,
                    "clipping",
                    "bounds-outside-frame",
                    "Subtitle bounds leave the rendered frame.",
                    f"{instance_path}.bounds",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                    measured=_rect_text(bounds),
                    threshold=_rect_text(frame_rect),
                )
            if not frame_rect.contains(ink):
                _fail(
                    buckets,
                    "clipping",
                    "ink-outside-frame",
                    "Visible subtitle ink leaves the rendered frame.",
                    f"{instance_path}.inkBounds",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                    measured=_rect_text(ink),
                    threshold=_rect_text(frame_rect),
                )
            if (
                ink.x == 0
                or ink.y == 0
                or ink.right == frame["widthPx"]
                or ink.bottom == frame["heightPx"]
            ):
                _fail(
                    buckets,
                    "clipping",
                    "ink-touches-frame-edge",
                    "Visible subtitle ink geometrically touches a frame edge.",
                    f"{instance_path}.inkBounds",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                )
            if not bounds.contains(ink):
                _fail(
                    buckets,
                    "clipping",
                    "ink-outside-declared-bounds",
                    "Visible subtitle ink is not contained by declared bounds.",
                    f"{instance_path}.inkBounds",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                )
            if instance["clippedPixelCount"] != 0:
                _fail(
                    buckets,
                    "clipping",
                    "clipped-pixels-detected",
                    "Renderer evidence reports clipped subtitle pixels.",
                    f"{instance_path}.clippedPixelCount",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                    measured=instance["clippedPixelCount"],
                    threshold=0,
                )
            if instance["edgeTouchingPixelCount"] != 0:
                _fail(
                    buckets,
                    "clipping",
                    "frame-edge-touch-detected",
                    "Subtitle ink touches a frame edge, so clipping cannot be excluded.",
                    f"{instance_path}.edgeTouchingPixelCount",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                    measured=instance["edgeTouchingPixelCount"],
                    threshold=0,
                )
            if instance["overflowDetected"]:
                _fail(
                    buckets,
                    "clipping",
                    "renderer-overflow-detected",
                    "Renderer evidence reports subtitle overflow.",
                    f"{instance_path}.overflowDetected",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                )

            samples = instance["contrastSamples"]
            checked["contrast"] += max(1, len(samples))
            if not samples:
                _fail(
                    buckets,
                    "contrast",
                    "contrast-evidence-missing",
                    "Every rendered cue instance requires sampled pixel evidence.",
                    f"{instance_path}.contrastSamples",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                )
            for sample_index, sample in enumerate(samples):
                sample_path = (
                    f"{instance_path}.contrastSamples[{sample_index}]"
                )
                background_class = sample["backgroundClass"]
                foreground_luminance = relative_luminance(
                    sample["foregroundRgb"]
                )
                background_luminance = relative_luminance(
                    sample["backgroundRgb"]
                )
                ratio = contrast_ratio(
                    sample["foregroundRgb"], sample["backgroundRgb"]
                )
                contrast_ratios.append(ratio)

                valid_class = True
                if (
                    background_class == "dark"
                    and background_luminance
                    > policy["contrast"]["darkMaximumLuminance"]
                ):
                    valid_class = False
                if (
                    background_class == "light"
                    and background_luminance
                    < policy["contrast"]["lightMinimumLuminance"]
                ):
                    valid_class = False
                if not valid_class:
                    _fail(
                        buckets,
                        "contrast",
                        "background-class-contradicted",
                        "Declared background class contradicts sampled luminance.",
                        f"{sample_path}.backgroundClass",
                        frame_id=frame["frameId"],
                        cue_id=cue_id,
                        style_id=style_id,
                        measured=_round_metric(background_luminance),
                        threshold=(
                            policy["contrast"]["darkMaximumLuminance"]
                            if background_class == "dark"
                            else policy["contrast"]["lightMinimumLuminance"]
                        ),
                    )
                else:
                    background_classes.add(background_class)
                    style_background_classes[style_id].add(background_class)
                if not sample["sampledFromRenderedFrame"]:
                    _fail(
                        buckets,
                        "contrast",
                        "unrendered-pixel-sample",
                        "Contrast samples must come from the rendered frame.",
                        f"{sample_path}.sampledFromRenderedFrame",
                        frame_id=frame["frameId"],
                        cue_id=cue_id,
                        style_id=style_id,
                    )
                if (
                    sample["foregroundPixelCount"]
                    < policy["contrast"]["minimumForegroundPixels"]
                ):
                    _fail(
                        buckets,
                        "contrast",
                        "foreground-sample-too-small",
                        "Foreground sample contains too few rendered pixels.",
                        f"{sample_path}.foregroundPixelCount",
                        frame_id=frame["frameId"],
                        cue_id=cue_id,
                        style_id=style_id,
                        measured=sample["foregroundPixelCount"],
                        threshold=policy["contrast"]["minimumForegroundPixels"],
                    )
                if (
                    sample["backgroundPixelCount"]
                    < policy["contrast"]["minimumBackgroundPixels"]
                ):
                    _fail(
                        buckets,
                        "contrast",
                        "background-sample-too-small",
                        "Background sample contains too few rendered pixels.",
                        f"{sample_path}.backgroundPixelCount",
                        frame_id=frame["frameId"],
                        cue_id=cue_id,
                        style_id=style_id,
                        measured=sample["backgroundPixelCount"],
                        threshold=policy["contrast"]["minimumBackgroundPixels"],
                    )
                if ratio < policy["contrast"]["minimumRatio"]:
                    _fail(
                        buckets,
                        "contrast",
                        "contrast-below-threshold",
                        "Rendered subtitle contrast is below the hard minimum.",
                        sample_path,
                        frame_id=frame["frameId"],
                        cue_id=cue_id,
                        style_id=style_id,
                        measured=_round_metric(ratio),
                        threshold=policy["contrast"]["minimumRatio"],
                    )
                if foreground_luminance == background_luminance:
                    # This is redundant with the ratio gate but gives a useful
                    # forensic code for exact foreground/background collisions.
                    _fail(
                        buckets,
                        "contrast",
                        "foreground-background-identical",
                        "Sampled foreground and background luminance are identical.",
                        sample_path,
                        frame_id=frame["frameId"],
                        cue_id=cue_id,
                        style_id=style_id,
                    )

            checked["fontGlyph"] += 1
            font_evidence = instance["fontEvidence"]
            if font_evidence is None:
                _fail(
                    buckets,
                    "fontGlyph",
                    "font-evidence-missing",
                    "Rendered font and glyph evidence is required.",
                    f"{instance_path}.fontEvidence",
                    frame_id=frame["frameId"],
                    cue_id=cue_id,
                    style_id=style_id,
                )
            else:
                font_evidence_instances += 1
                installation = font_evidence["installation"]
                embedding = font_evidence["embedding"]
                if installation["status"] == "verified-installed":
                    installation_verified += 1
                else:
                    installation_not_asserted += 1
                if embedding["status"] in {
                    "verified-embedded",
                    "verified-not-embedded",
                }:
                    embedding_verified += 1
                else:
                    embedding_not_asserted += 1
                _evaluate_font_evidence(
                    buckets,
                    font_evidence,
                    cue,
                    frame_id=frame["frameId"],
                    instance_path=instance_path,
                )

        for left_index, left in enumerate(instances):
            for right_index in range(left_index + 1, len(instances)):
                right = instances[right_index]
                checked["visualOverlap"] += 1
                area = left["inkBounds"].intersection_area(right["inkBounds"])
                overlap_areas.append(area)
                if area > policy["layout"]["maximumOverlapAreaPx"]:
                    _fail(
                        buckets,
                        "visualOverlap",
                        "subtitle-ink-overlap",
                        "Two rendered subtitle instances overlap visually.",
                        (
                            f"{frame_path}.instances[{left_index}].inkBounds"
                            f"|{frame_path}.instances[{right_index}].inkBounds"
                        ),
                        frame_id=frame["frameId"],
                        cue_id=f"{left['cueId']}|{right['cueId']}",
                        measured=area,
                        threshold=policy["layout"]["maximumOverlapAreaPx"],
                    )

    for cue_id, cue in sorted(cues.items()):
        checked["samplingEvidence"] += 1
        if cue_id not in represented_cues:
            _fail(
                buckets,
                "samplingEvidence",
                "cue-not-represented",
                "Every cue in the representative set must appear in a frame.",
                "$.frames",
                cue_id=cue_id,
                style_id=cue["styleId"],
            )

        checked["lineReading"] += 1
        _evaluate_line_and_reading(
            buckets,
            cue,
            policy,
            reading_speeds,
        )

        checked["karaokeAuthenticity"] += 1
        if _evaluate_karaoke(buckets, cue, policy):
            verified_word_timing_cues += 1

    required_backgrounds = set(
        policy["contrast"]["requiredBackgroundClasses"]
    )
    for style_id in sorted({cue["styleId"] for cue in cues.values()}):
        for background_class in sorted(
            required_backgrounds - style_background_classes[style_id]
        ):
            checked["contrast"] += 1
            _fail(
                buckets,
                "contrast",
                "required-background-class-missing",
                "A subtitle style lacks valid contrast evidence for a required background class.",
                "$.frames[*].instances[*].contrastSamples",
                style_id=style_id,
                measured=background_class,
                threshold="required",
            )

    speaker_items = sorted(speakers.values(), key=lambda item: item["speakerId"])
    if len(speaker_items) < 2:
        checked["speakerColor"] = 1
    else:
        for left_index, left in enumerate(speaker_items):
            for right in speaker_items[left_index + 1 :]:
                checked["speakerColor"] += 1
                delta = delta_e_2000(left["color"], right["color"])
                speaker_deltas.append(delta)
                if delta < policy["speakerColor"]["minimumDeltaE2000"]:
                    _fail(
                        buckets,
                        "speakerColor",
                        "speaker-colors-not-distinct",
                        "Speaker colors are insufficiently distinguishable.",
                        "$.speakers",
                        cue_id=f"{left['speakerId']}|{right['speakerId']}",
                        measured=_round_metric(delta),
                        threshold=policy["speakerColor"][
                            "minimumDeltaE2000"
                        ],
                    )

    gates = tuple(
        GateResult(
            name=gate,
            checked=checked[gate],
            violations=tuple(
                sorted(buckets[gate], key=_violation_sort_key)
            ),
        )
        for gate in _GATE_ORDER
    )
    metrics = SubtitleVisualQAMetrics(
        frames_expected=len(expected_frame_ids),
        frames_evaluated=len(frames),
        cues_evaluated=len(cues),
        cue_instances_evaluated=cue_instances,
        contrast_samples_evaluated=len(contrast_ratios),
        minimum_contrast_ratio=(
            _round_metric(min(contrast_ratios)) if contrast_ratios else None
        ),
        maximum_reading_speed=(
            _round_metric(max(reading_speeds)) if reading_speeds else None
        ),
        minimum_speaker_delta_e_2000=(
            _round_metric(min(speaker_deltas)) if speaker_deltas else None
        ),
        maximum_overlap_area_px=max(overlap_areas, default=0),
        verified_word_timing_cues=verified_word_timing_cues,
        background_classes_covered=tuple(sorted(background_classes)),
    )
    font_truth = FontTruthSummary(
        evidence_instances=font_evidence_instances,
        installation_verified=installation_verified,
        installation_not_asserted=installation_not_asserted,
        embedding_verified=embedding_verified,
        embedding_not_asserted=embedding_not_asserted,
    )
    return SubtitleVisualQAResult(
        analysis_id=analysis_id,
        input_sha256=deterministic_sha256(root),
        gates=gates,
        metrics=metrics,
        font_truth=font_truth,
    )


def relative_luminance(color: str) -> float:
    """Return WCAG relative luminance for one ``#RRGGBB`` color."""

    red, green, blue = _rgb(color)

    def linear(component: int) -> float:
        value = component / 255.0
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * linear(red)
        + 0.7152 * linear(green)
        + 0.0722 * linear(blue)
    )


def contrast_ratio(foreground: str, background: str) -> float:
    """Return WCAG contrast ratio for two sRGB colors."""

    left = relative_luminance(foreground)
    right = relative_luminance(background)
    return (max(left, right) + 0.05) / (min(left, right) + 0.05)


def delta_e_2000(left: str, right: str) -> float:
    """Return CIEDE2000 color difference for two sRGB colors."""

    lab1 = _rgb_to_lab(left)
    lab2 = _rgb_to_lab(right)
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    mean_c = (c1 + c2) / 2.0
    g = 0.5 * (
        1.0
        - math.sqrt(
            (mean_c**7) / (mean_c**7 + 25.0**7)
            if mean_c
            else 0.0
        )
    )
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = math.hypot(a1_prime, b1)
    c2_prime = math.hypot(a2_prime, b2)
    h1_prime = _hue_degrees(a1_prime, b1)
    h2_prime = _hue_degrees(a2_prime, b2)

    delta_l = l2 - l1
    delta_c = c2_prime - c1_prime
    if c1_prime * c2_prime == 0:
        delta_h_angle = 0.0
    elif abs(h2_prime - h1_prime) <= 180.0:
        delta_h_angle = h2_prime - h1_prime
    elif h2_prime <= h1_prime:
        delta_h_angle = h2_prime - h1_prime + 360.0
    else:
        delta_h_angle = h2_prime - h1_prime - 360.0
    delta_h = (
        2.0
        * math.sqrt(c1_prime * c2_prime)
        * math.sin(math.radians(delta_h_angle / 2.0))
    )

    mean_l = (l1 + l2) / 2.0
    mean_c_prime = (c1_prime + c2_prime) / 2.0
    if c1_prime * c2_prime == 0:
        mean_h = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0:
        mean_h = (h1_prime + h2_prime) / 2.0
    elif h1_prime + h2_prime < 360.0:
        mean_h = (h1_prime + h2_prime + 360.0) / 2.0
    else:
        mean_h = (h1_prime + h2_prime - 360.0) / 2.0

    t = (
        1.0
        - 0.17 * math.cos(math.radians(mean_h - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * mean_h))
        + 0.32 * math.cos(math.radians(3.0 * mean_h + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * mean_h - 63.0))
    )
    s_l = 1.0 + (
        0.015 * (mean_l - 50.0) ** 2
        / math.sqrt(20.0 + (mean_l - 50.0) ** 2)
    )
    s_c = 1.0 + 0.045 * mean_c_prime
    s_h = 1.0 + 0.015 * mean_c_prime * t
    delta_theta = 30.0 * math.exp(
        -(((mean_h - 275.0) / 25.0) ** 2)
    )
    r_c = 2.0 * math.sqrt(
        (mean_c_prime**7) / (mean_c_prime**7 + 25.0**7)
        if mean_c_prime
        else 0.0
    )
    r_t = -r_c * math.sin(math.radians(2.0 * delta_theta))
    l_term = delta_l / s_l
    c_term = delta_c / s_c
    h_term = delta_h / s_h
    return math.sqrt(
        l_term**2 + c_term**2 + h_term**2 + r_t * c_term * h_term
    )


def _parse_render_artifact(value: Any) -> dict[str, Any]:
    item = _object(value, "$.renderArtifact")
    _keys(
        item,
        required={
            "artifactSha256",
            "renderer",
            "rendererVersion",
            "renderConfigurationSha256",
        },
        allowed={
            "artifactSha256",
            "renderer",
            "rendererVersion",
            "renderConfigurationSha256",
        },
        path="$.renderArtifact",
    )
    return {
        "artifactSha256": _sha256(
            item["artifactSha256"], "$.renderArtifact.artifactSha256"
        ),
        "renderer": _bounded_string(
            item["renderer"], "$.renderArtifact.renderer", 1, 160
        ),
        "rendererVersion": _bounded_string(
            item["rendererVersion"],
            "$.renderArtifact.rendererVersion",
            1,
            320,
        ),
        "renderConfigurationSha256": _sha256(
            item["renderConfigurationSha256"],
            "$.renderArtifact.renderConfigurationSha256",
        ),
    }


def _parse_policy(value: Any) -> dict[str, Any]:
    policy = _object(value, "$.policy")
    _keys(
        policy,
        required={"safeArea", "contrast", "layout", "speakerColor", "karaoke"},
        allowed={"safeArea", "contrast", "layout", "speakerColor", "karaoke"},
        path="$.policy",
    )
    safe = _object(policy["safeArea"], "$.policy.safeArea")
    _keys(
        safe,
        required={"leftRatio", "rightRatio", "topRatio", "bottomRatio"},
        allowed={"leftRatio", "rightRatio", "topRatio", "bottomRatio"},
        path="$.policy.safeArea",
    )
    safe_result = {
        key: _number_between(
            safe[key], f"$.policy.safeArea.{key}", 0.0, 0.25
        )
        for key in ("leftRatio", "rightRatio", "topRatio", "bottomRatio")
    }
    if safe_result["leftRatio"] + safe_result["rightRatio"] >= 1.0:
        raise SubtitleVisualQAInputError(
            "$.policy.safeArea horizontal ratios leave no content area"
        )
    if safe_result["topRatio"] + safe_result["bottomRatio"] >= 1.0:
        raise SubtitleVisualQAInputError(
            "$.policy.safeArea vertical ratios leave no content area"
        )

    contrast = _object(policy["contrast"], "$.policy.contrast")
    _keys(
        contrast,
        required={
            "minimumRatio",
            "minimumForegroundPixels",
            "minimumBackgroundPixels",
            "requiredBackgroundClasses",
            "darkMaximumLuminance",
            "lightMinimumLuminance",
        },
        allowed={
            "minimumRatio",
            "minimumForegroundPixels",
            "minimumBackgroundPixels",
            "requiredBackgroundClasses",
            "darkMaximumLuminance",
            "lightMinimumLuminance",
        },
        path="$.policy.contrast",
    )
    required_classes = _unique_strings(
        contrast["requiredBackgroundClasses"],
        "$.policy.contrast.requiredBackgroundClasses",
        allowed=_BACKGROUND_CLASSES,
        minimum=1,
    )
    contrast_result = {
        "minimumRatio": _number_between(
            contrast["minimumRatio"],
            "$.policy.contrast.minimumRatio",
            1.0,
            21.0,
        ),
        "minimumForegroundPixels": _integer_between(
            contrast["minimumForegroundPixels"],
            "$.policy.contrast.minimumForegroundPixels",
            1,
            10_000_000,
        ),
        "minimumBackgroundPixels": _integer_between(
            contrast["minimumBackgroundPixels"],
            "$.policy.contrast.minimumBackgroundPixels",
            1,
            10_000_000,
        ),
        "requiredBackgroundClasses": required_classes,
        "darkMaximumLuminance": _number_between(
            contrast["darkMaximumLuminance"],
            "$.policy.contrast.darkMaximumLuminance",
            0.0,
            1.0,
        ),
        "lightMinimumLuminance": _number_between(
            contrast["lightMinimumLuminance"],
            "$.policy.contrast.lightMinimumLuminance",
            0.0,
            1.0,
        ),
    }
    if (
        contrast_result["darkMaximumLuminance"]
        >= contrast_result["lightMinimumLuminance"]
    ):
        raise SubtitleVisualQAInputError(
            "$.policy.contrast dark threshold must be below light threshold"
        )

    layout = _object(policy["layout"], "$.policy.layout")
    _keys(
        layout,
        required={
            "maxLines",
            "maxCharactersPerLine",
            "maxReadingSpeed",
            "maximumOverlapAreaPx",
        },
        allowed={
            "maxLines",
            "maxCharactersPerLine",
            "maxReadingSpeed",
            "maximumOverlapAreaPx",
        },
        path="$.policy.layout",
    )
    layout_result = {
        "maxLines": _integer_between(
            layout["maxLines"], "$.policy.layout.maxLines", 1, 6
        ),
        "maxCharactersPerLine": _integer_between(
            layout["maxCharactersPerLine"],
            "$.policy.layout.maxCharactersPerLine",
            1,
            240,
        ),
        "maxReadingSpeed": _number_between(
            layout["maxReadingSpeed"],
            "$.policy.layout.maxReadingSpeed",
            1.0,
            100.0,
        ),
        "maximumOverlapAreaPx": _integer_between(
            layout["maximumOverlapAreaPx"],
            "$.policy.layout.maximumOverlapAreaPx",
            0,
            100_000_000,
        ),
    }

    speaker = _object(policy["speakerColor"], "$.policy.speakerColor")
    _keys(
        speaker,
        required={"minimumDeltaE2000"},
        allowed={"minimumDeltaE2000"},
        path="$.policy.speakerColor",
    )
    speaker_result = {
        "minimumDeltaE2000": _number_between(
            speaker["minimumDeltaE2000"],
            "$.policy.speakerColor.minimumDeltaE2000",
            0.1,
            100.0,
        )
    }

    karaoke = _object(policy["karaoke"], "$.policy.karaoke")
    _keys(
        karaoke,
        required={"requireVerifiedWordTiming"},
        allowed={"requireVerifiedWordTiming"},
        path="$.policy.karaoke",
    )
    karaoke_result = {
        "requireVerifiedWordTiming": _boolean(
            karaoke["requireVerifiedWordTiming"],
            "$.policy.karaoke.requireVerifiedWordTiming",
        )
    }
    if not karaoke_result["requireVerifiedWordTiming"]:
        raise SubtitleVisualQAInputError(
            "$.policy.karaoke.requireVerifiedWordTiming must be true"
        )

    return {
        "safeArea": safe_result,
        "contrast": contrast_result,
        "layout": layout_result,
        "speakerColor": speaker_result,
        "karaoke": karaoke_result,
    }


def _parse_sampling(value: Any) -> dict[str, Any]:
    sampling = _object(value, "$.sampling")
    _keys(
        sampling,
        required={"strategy", "selectionArtifactSha256", "expectedFrameIds"},
        allowed={"strategy", "selectionArtifactSha256", "expectedFrameIds"},
        path="$.sampling",
    )
    return {
        "strategy": _bounded_string(
            sampling["strategy"], "$.sampling.strategy", 1, 160
        ),
        "selectionArtifactSha256": _sha256(
            sampling["selectionArtifactSha256"],
            "$.sampling.selectionArtifactSha256",
        ),
        "expectedFrameIds": _unique_strings(
            sampling["expectedFrameIds"],
            "$.sampling.expectedFrameIds",
            minimum=1,
            maximum=100_000,
        ),
    }


def _parse_speakers(value: Any) -> dict[str, dict[str, Any]]:
    values = _array(value, "$.speakers", minimum=1, maximum=10_000)
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(values):
        path = f"$.speakers[{index}]"
        item = _object(raw, path)
        _keys(
            item,
            required={"speakerId", "color"},
            allowed={"speakerId", "color"},
            path=path,
        )
        speaker_id = _bounded_string(
            item["speakerId"], f"{path}.speakerId", 1, 160
        )
        color = _color(item["color"], f"{path}.color")
        if speaker_id in result:
            raise SubtitleVisualQAInputError(
                f"{path}.speakerId duplicates {speaker_id!r}"
            )
        result[speaker_id] = {"speakerId": speaker_id, "color": color}
    return result


def _parse_cues(
    value: Any,
    speakers: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    values = _array(value, "$.cues", minimum=1, maximum=1_000_000)
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(values):
        path = f"$.cues[{index}]"
        item = _object(raw, path)
        _keys(
            item,
            required={
                "cueId",
                "startMs",
                "endMs",
                "text",
                "speakerId",
                "styleId",
                "renderedLines",
                "karaokeMode",
                "wordTimingEvidence",
            },
            allowed={
                "cueId",
                "startMs",
                "endMs",
                "text",
                "speakerId",
                "styleId",
                "renderedLines",
                "karaokeMode",
                "wordTimingEvidence",
            },
            path=path,
        )
        cue_id = _bounded_string(item["cueId"], f"{path}.cueId", 1, 160)
        if cue_id in result:
            raise SubtitleVisualQAInputError(
                f"{path}.cueId duplicates {cue_id!r}"
            )
        start_ms = _integer_between(
            item["startMs"], f"{path}.startMs", 0, 604_800_000
        )
        end_ms = _integer_between(
            item["endMs"], f"{path}.endMs", 1, 604_800_000
        )
        if end_ms <= start_ms:
            raise SubtitleVisualQAInputError(
                f"{path}.endMs must be greater than startMs"
            )
        text = _bounded_string(item["text"], f"{path}.text", 1, 100_000)
        speaker_id = _bounded_string(
            item["speakerId"], f"{path}.speakerId", 1, 160
        )
        if speaker_id not in speakers:
            raise SubtitleVisualQAInputError(
                f"{path}.speakerId does not reference $.speakers"
            )
        style_id = _bounded_string(
            item["styleId"], f"{path}.styleId", 1, 160
        )
        rendered_lines_raw = _array(
            item["renderedLines"],
            f"{path}.renderedLines",
            minimum=1,
            maximum=20,
        )
        rendered_lines = tuple(
            _bounded_string(
                line,
                f"{path}.renderedLines[{line_index}]",
                1,
                10_000,
            )
            for line_index, line in enumerate(rendered_lines_raw)
        )
        karaoke_mode = _enum_string(
            item["karaokeMode"],
            f"{path}.karaokeMode",
            {"none", "word-progress"},
        )
        word_evidence = _parse_word_timing_evidence(
            item["wordTimingEvidence"],
            f"{path}.wordTimingEvidence",
        )
        result[cue_id] = {
            "cueId": cue_id,
            "startMs": start_ms,
            "endMs": end_ms,
            "text": text,
            "speakerId": speaker_id,
            "styleId": style_id,
            "renderedLines": rendered_lines,
            "karaokeMode": karaoke_mode,
            "wordTimingEvidence": word_evidence,
            "path": path,
        }
    return result


def _parse_word_timing_evidence(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _object(value, path)
    _keys(
        item,
        required={
            "source",
            "verified",
            "evidenceArtifactSha256",
            "cueTextSha256",
            "words",
        },
        allowed={
            "source",
            "verified",
            "evidenceArtifactSha256",
            "cueTextSha256",
            "words",
        },
        path=path,
    )
    words_raw = _array(item["words"], f"{path}.words", maximum=100_000)
    words: list[dict[str, Any]] = []
    for index, raw in enumerate(words_raw):
        word_path = f"{path}.words[{index}]"
        word = _object(raw, word_path)
        _keys(
            word,
            required={"text", "startMs", "endMs"},
            allowed={"text", "startMs", "endMs"},
            path=word_path,
        )
        words.append(
            {
                "text": _bounded_string(
                    word["text"], f"{word_path}.text", 1, 1_000
                ),
                "startMs": _integer_between(
                    word["startMs"], f"{word_path}.startMs", 0, 604_800_000
                ),
                "endMs": _integer_between(
                    word["endMs"], f"{word_path}.endMs", 1, 604_800_000
                ),
            }
        )
    return {
        "source": _enum_string(
            item["source"], f"{path}.source", _WORD_TIMING_SOURCES
        ),
        "verified": _boolean(item["verified"], f"{path}.verified"),
        "evidenceArtifactSha256": _sha256(
            item["evidenceArtifactSha256"],
            f"{path}.evidenceArtifactSha256",
        ),
        "cueTextSha256": _sha256(
            item["cueTextSha256"], f"{path}.cueTextSha256"
        ),
        "words": tuple(words),
    }


def _parse_frames(
    value: Any,
    cues: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    values = _array(value, "$.frames", maximum=100_000)
    result: list[dict[str, Any]] = []
    frame_ids: set[str] = set()
    for frame_index, raw in enumerate(values):
        path = f"$.frames[{frame_index}]"
        item = _object(raw, path)
        _keys(
            item,
            required={
                "frameId",
                "timestampMs",
                "widthPx",
                "heightPx",
                "imageSha256",
                "instances",
            },
            allowed={
                "frameId",
                "timestampMs",
                "widthPx",
                "heightPx",
                "imageSha256",
                "instances",
            },
            path=path,
        )
        frame_id = _bounded_string(
            item["frameId"], f"{path}.frameId", 1, 160
        )
        if frame_id in frame_ids:
            raise SubtitleVisualQAInputError(
                f"{path}.frameId duplicates {frame_id!r}"
            )
        frame_ids.add(frame_id)
        instances_raw = _array(
            item["instances"], f"{path}.instances", maximum=1_000
        )
        instances = tuple(
            _parse_instance(instance, f"{path}.instances[{index}]", cues)
            for index, instance in enumerate(instances_raw)
        )
        result.append(
            {
                "frameId": frame_id,
                "timestampMs": _integer_between(
                    item["timestampMs"],
                    f"{path}.timestampMs",
                    0,
                    604_800_000,
                ),
                "widthPx": _integer_between(
                    item["widthPx"], f"{path}.widthPx", 16, 32_768
                ),
                "heightPx": _integer_between(
                    item["heightPx"], f"{path}.heightPx", 16, 32_768
                ),
                "imageSha256": _sha256(
                    item["imageSha256"], f"{path}.imageSha256"
                ),
                "instances": instances,
            }
        )
    return tuple(result)


def _parse_instance(
    value: Any,
    path: str,
    cues: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    item = _object(value, path)
    _keys(
        item,
        required={
            "cueId",
            "bounds",
            "inkBounds",
            "clippedPixelCount",
            "edgeTouchingPixelCount",
            "overflowDetected",
            "fontEvidence",
            "contrastSamples",
        },
        allowed={
            "cueId",
            "bounds",
            "inkBounds",
            "clippedPixelCount",
            "edgeTouchingPixelCount",
            "overflowDetected",
            "fontEvidence",
            "contrastSamples",
        },
        path=path,
    )
    cue_id = _bounded_string(item["cueId"], f"{path}.cueId", 1, 160)
    if cue_id not in cues:
        raise SubtitleVisualQAInputError(
            f"{path}.cueId does not reference $.cues"
        )
    samples_raw = _array(
        item["contrastSamples"], f"{path}.contrastSamples", maximum=1_000
    )
    samples = tuple(
        _parse_contrast_sample(
            sample, f"{path}.contrastSamples[{index}]"
        )
        for index, sample in enumerate(samples_raw)
    )
    sample_ids = [sample["sampleId"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise SubtitleVisualQAInputError(
            f"{path}.contrastSamples contains duplicate sampleId values"
        )
    return {
        "cueId": cue_id,
        "bounds": _parse_rect(item["bounds"], f"{path}.bounds"),
        "inkBounds": _parse_rect(item["inkBounds"], f"{path}.inkBounds"),
        "clippedPixelCount": _integer_between(
            item["clippedPixelCount"],
            f"{path}.clippedPixelCount",
            0,
            1_000_000_000,
        ),
        "edgeTouchingPixelCount": _integer_between(
            item["edgeTouchingPixelCount"],
            f"{path}.edgeTouchingPixelCount",
            0,
            1_000_000_000,
        ),
        "overflowDetected": _boolean(
            item["overflowDetected"], f"{path}.overflowDetected"
        ),
        "fontEvidence": _parse_font_evidence(
            item["fontEvidence"], f"{path}.fontEvidence"
        ),
        "contrastSamples": samples,
    }


def _parse_rect(value: Any, path: str) -> Rect:
    item = _object(value, path)
    _keys(
        item,
        required={"x", "y", "width", "height"},
        allowed={"x", "y", "width", "height"},
        path=path,
    )
    return Rect(
        x=_integer_between(item["x"], f"{path}.x", 0, 1_000_000),
        y=_integer_between(item["y"], f"{path}.y", 0, 1_000_000),
        width=_integer_between(
            item["width"], f"{path}.width", 1, 1_000_000
        ),
        height=_integer_between(
            item["height"], f"{path}.height", 1, 1_000_000
        ),
    )


def _parse_contrast_sample(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path)
    _keys(
        item,
        required={
            "sampleId",
            "backgroundClass",
            "foregroundRgb",
            "backgroundRgb",
            "foregroundPixelCount",
            "backgroundPixelCount",
            "sampledFromRenderedFrame",
            "sampleArtifactSha256",
        },
        allowed={
            "sampleId",
            "backgroundClass",
            "foregroundRgb",
            "backgroundRgb",
            "foregroundPixelCount",
            "backgroundPixelCount",
            "sampledFromRenderedFrame",
            "sampleArtifactSha256",
        },
        path=path,
    )
    return {
        "sampleId": _bounded_string(
            item["sampleId"], f"{path}.sampleId", 1, 160
        ),
        "backgroundClass": _enum_string(
            item["backgroundClass"],
            f"{path}.backgroundClass",
            _BACKGROUND_CLASSES,
        ),
        "foregroundRgb": _color(
            item["foregroundRgb"], f"{path}.foregroundRgb"
        ),
        "backgroundRgb": _color(
            item["backgroundRgb"], f"{path}.backgroundRgb"
        ),
        "foregroundPixelCount": _integer_between(
            item["foregroundPixelCount"],
            f"{path}.foregroundPixelCount",
            1,
            1_000_000_000,
        ),
        "backgroundPixelCount": _integer_between(
            item["backgroundPixelCount"],
            f"{path}.backgroundPixelCount",
            1,
            1_000_000_000,
        ),
        "sampledFromRenderedFrame": _boolean(
            item["sampledFromRenderedFrame"],
            f"{path}.sampledFromRenderedFrame",
        ),
        "sampleArtifactSha256": _sha256(
            item["sampleArtifactSha256"],
            f"{path}.sampleArtifactSha256",
        ),
    }


def _parse_font_evidence(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _object(value, path)
    _keys(
        item,
        required={
            "requestedFamilies",
            "resolvedFamily",
            "resolutionVerified",
            "glyphCoverageVerified",
            "verificationMethod",
            "evidenceArtifactSha256",
            "expectedRenderableCodePoints",
            "coveredRenderableCodePoints",
            "missingCodePoints",
            "tofuGlyphCount",
            "installation",
            "embedding",
        },
        allowed={
            "requestedFamilies",
            "resolvedFamily",
            "resolutionVerified",
            "glyphCoverageVerified",
            "verificationMethod",
            "evidenceArtifactSha256",
            "expectedRenderableCodePoints",
            "coveredRenderableCodePoints",
            "missingCodePoints",
            "tofuGlyphCount",
            "installation",
            "embedding",
        },
        path=path,
    )
    resolved_family = item["resolvedFamily"]
    if resolved_family is not None:
        resolved_family = _bounded_string(
            resolved_family, f"{path}.resolvedFamily", 1, 320
        )
    evidence_hash = item["evidenceArtifactSha256"]
    if evidence_hash is not None:
        evidence_hash = _sha256(
            evidence_hash, f"{path}.evidenceArtifactSha256"
        )
    missing = _unique_strings(
        item["missingCodePoints"],
        f"{path}.missingCodePoints",
        maximum=100_000,
    )
    for index, codepoint in enumerate(missing):
        if not _CODEPOINT.fullmatch(codepoint):
            raise SubtitleVisualQAInputError(
                f"{path}.missingCodePoints[{index}] must be U+XXXX"
            )
    return {
        "requestedFamilies": _unique_strings(
            item["requestedFamilies"],
            f"{path}.requestedFamilies",
            minimum=1,
            maximum=64,
        ),
        "resolvedFamily": resolved_family,
        "resolutionVerified": _boolean(
            item["resolutionVerified"], f"{path}.resolutionVerified"
        ),
        "glyphCoverageVerified": _boolean(
            item["glyphCoverageVerified"], f"{path}.glyphCoverageVerified"
        ),
        "verificationMethod": _enum_string(
            item["verificationMethod"],
            f"{path}.verificationMethod",
            _FONT_EVIDENCE_METHODS,
        ),
        "evidenceArtifactSha256": evidence_hash,
        "expectedRenderableCodePoints": _integer_between(
            item["expectedRenderableCodePoints"],
            f"{path}.expectedRenderableCodePoints",
            0,
            1_000_000,
        ),
        "coveredRenderableCodePoints": _integer_between(
            item["coveredRenderableCodePoints"],
            f"{path}.coveredRenderableCodePoints",
            0,
            1_000_000,
        ),
        "missingCodePoints": missing,
        "tofuGlyphCount": _integer_between(
            item["tofuGlyphCount"],
            f"{path}.tofuGlyphCount",
            0,
            1_000_000,
        ),
        "installation": _parse_font_claim(
            item["installation"],
            f"{path}.installation",
            statuses=_INSTALLATION_STATUSES,
        ),
        "embedding": _parse_font_claim(
            item["embedding"],
            f"{path}.embedding",
            statuses=_EMBEDDING_STATUSES,
        ),
    }


def _parse_font_claim(
    value: Any,
    path: str,
    *,
    statuses: frozenset[str],
) -> dict[str, Any]:
    item = _object(value, path)
    _keys(
        item,
        required={
            "status",
            "verificationMethod",
            "evidenceArtifactSha256",
            "fontArtifactSha256",
        },
        allowed={
            "status",
            "verificationMethod",
            "evidenceArtifactSha256",
            "fontArtifactSha256",
        },
        path=path,
    )
    status = _enum_string(item["status"], f"{path}.status", statuses)
    method = _enum_string(
        item["verificationMethod"],
        f"{path}.verificationMethod",
        _FONT_EVIDENCE_METHODS,
    )
    evidence_hash = item["evidenceArtifactSha256"]
    font_hash = item["fontArtifactSha256"]
    if evidence_hash is not None:
        evidence_hash = _sha256(
            evidence_hash, f"{path}.evidenceArtifactSha256"
        )
    if font_hash is not None:
        font_hash = _sha256(font_hash, f"{path}.fontArtifactSha256")
    if status == "not-asserted":
        if method != "not-provided" or evidence_hash is not None or font_hash is not None:
            raise SubtitleVisualQAInputError(
                f"{path} cannot carry claim evidence when status is not-asserted"
            )
    elif (
        method == "not-provided"
        or evidence_hash is None
        or font_hash is None
    ):
        raise SubtitleVisualQAInputError(
            f"{path} positive claim requires method and SHA-256 evidence"
        )
    return {
        "status": status,
        "verificationMethod": method,
        "evidenceArtifactSha256": evidence_hash,
        "fontArtifactSha256": font_hash,
    }


def _evaluate_font_evidence(
    buckets: dict[str, list[Violation]],
    evidence: Mapping[str, Any],
    cue: Mapping[str, Any],
    *,
    frame_id: str,
    instance_path: str,
) -> None:
    cue_id = str(cue["cueId"])
    style_id = str(cue["styleId"])
    expected = _renderable_codepoint_count(str(cue["text"]))
    common = {
        "frame_id": frame_id,
        "cue_id": cue_id,
        "style_id": style_id,
    }
    if not evidence["resolutionVerified"]:
        _fail(
            buckets,
            "fontGlyph",
            "font-resolution-unverified",
            "The renderer did not verify the resolved font.",
            f"{instance_path}.fontEvidence.resolutionVerified",
            **common,
        )
    if evidence["resolvedFamily"] is None:
        _fail(
            buckets,
            "fontGlyph",
            "resolved-font-missing",
            "Resolved font family evidence is missing.",
            f"{instance_path}.fontEvidence.resolvedFamily",
            **common,
        )
    if (
        evidence["verificationMethod"] == "not-provided"
        or evidence["evidenceArtifactSha256"] is None
    ):
        _fail(
            buckets,
            "fontGlyph",
            "font-verification-artifact-missing",
            "Font/glyph verification must be bound to a SHA-256 artifact.",
            f"{instance_path}.fontEvidence",
            **common,
        )
    if not evidence["glyphCoverageVerified"]:
        _fail(
            buckets,
            "fontGlyph",
            "glyph-coverage-unverified",
            "Glyph coverage was not verified by the renderer evidence.",
            f"{instance_path}.fontEvidence.glyphCoverageVerified",
            **common,
        )
    if evidence["expectedRenderableCodePoints"] != expected:
        _fail(
            buckets,
            "fontGlyph",
            "expected-codepoint-count-mismatch",
            "Glyph evidence does not match the cue's renderable code-point count.",
            f"{instance_path}.fontEvidence.expectedRenderableCodePoints",
            measured=evidence["expectedRenderableCodePoints"],
            threshold=expected,
            **common,
        )
    if evidence["coveredRenderableCodePoints"] != expected:
        _fail(
            buckets,
            "fontGlyph",
            "glyph-coverage-incomplete",
            "Not every renderable cue code point is covered.",
            f"{instance_path}.fontEvidence.coveredRenderableCodePoints",
            measured=evidence["coveredRenderableCodePoints"],
            threshold=expected,
            **common,
        )
    if evidence["missingCodePoints"]:
        _fail(
            buckets,
            "fontGlyph",
            "missing-glyphs-detected",
            "Renderer evidence lists missing Unicode code points.",
            f"{instance_path}.fontEvidence.missingCodePoints",
            measured=",".join(evidence["missingCodePoints"]),
            threshold="empty",
            **common,
        )
    if evidence["tofuGlyphCount"] != 0:
        _fail(
            buckets,
            "fontGlyph",
            "tofu-glyphs-detected",
            "Renderer evidence reports replacement/tofu glyphs.",
            f"{instance_path}.fontEvidence.tofuGlyphCount",
            measured=evidence["tofuGlyphCount"],
            threshold=0,
            **common,
        )


def _evaluate_line_and_reading(
    buckets: dict[str, list[Violation]],
    cue: Mapping[str, Any],
    policy: Mapping[str, Any],
    reading_speeds: list[float],
) -> None:
    path = str(cue["path"])
    cue_id = str(cue["cueId"])
    style_id = str(cue["styleId"])
    lines = tuple(cue["renderedLines"])
    maximum_lines = policy["layout"]["maxLines"]
    maximum_characters = policy["layout"]["maxCharactersPerLine"]
    if len(lines) > maximum_lines:
        _fail(
            buckets,
            "lineReading",
            "line-count-exceeded",
            "Rendered subtitle uses too many lines.",
            f"{path}.renderedLines",
            cue_id=cue_id,
            style_id=style_id,
            measured=len(lines),
            threshold=maximum_lines,
        )
    for index, line in enumerate(lines):
        count = _renderable_codepoint_count(line)
        if count > maximum_characters:
            _fail(
                buckets,
                "lineReading",
                "line-length-exceeded",
                "Rendered subtitle line exceeds the character limit.",
                f"{path}.renderedLines[{index}]",
                cue_id=cue_id,
                style_id=style_id,
                measured=count,
                threshold=maximum_characters,
            )
    if _without_whitespace("".join(lines)) != _without_whitespace(str(cue["text"])):
        _fail(
            buckets,
            "lineReading",
            "rendered-text-mismatch",
            "Rendered lines do not reconstruct the cue's non-whitespace text.",
            f"{path}.renderedLines",
            cue_id=cue_id,
            style_id=style_id,
        )
    duration_seconds = (cue["endMs"] - cue["startMs"]) / 1_000.0
    speed = _renderable_codepoint_count(str(cue["text"])) / duration_seconds
    reading_speeds.append(speed)
    if speed > policy["layout"]["maxReadingSpeed"]:
        _fail(
            buckets,
            "lineReading",
            "reading-speed-exceeded",
            "Cue reading speed exceeds the hard maximum.",
            path,
            cue_id=cue_id,
            style_id=style_id,
            measured=_round_metric(speed),
            threshold=policy["layout"]["maxReadingSpeed"],
        )


def _evaluate_karaoke(
    buckets: dict[str, list[Violation]],
    cue: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    if cue["karaokeMode"] == "none":
        return False
    path = str(cue["path"])
    cue_id = str(cue["cueId"])
    style_id = str(cue["styleId"])
    evidence = cue["wordTimingEvidence"]
    if not policy["karaoke"]["requireVerifiedWordTiming"]:
        # The policy parser rejects this state; retain the branch as a
        # defense-in-depth guard if policy construction changes.
        _fail(
            buckets,
            "karaokeAuthenticity",
            "karaoke-policy-not-fail-closed",
            "Word-progress karaoke cannot disable timing verification.",
            "$.policy.karaoke.requireVerifiedWordTiming",
            cue_id=cue_id,
            style_id=style_id,
        )
        return False
    if evidence is None:
        _fail(
            buckets,
            "karaokeAuthenticity",
            "word-timing-evidence-missing",
            "Word-progress karaoke requires true word-level timing evidence.",
            f"{path}.wordTimingEvidence",
            cue_id=cue_id,
            style_id=style_id,
        )
        return False
    valid = True
    if evidence["source"] not in _TRUE_WORD_TIMING_SOURCES:
        valid = False
        _fail(
            buckets,
            "karaokeAuthenticity",
            "synthetic-word-timing-source",
            "Interpolated or evenly split segment timing is not true word timing.",
            f"{path}.wordTimingEvidence.source",
            cue_id=cue_id,
            style_id=style_id,
            measured=evidence["source"],
            threshold="verified true word timing",
        )
    if not evidence["verified"]:
        valid = False
        _fail(
            buckets,
            "karaokeAuthenticity",
            "word-timing-unverified",
            "Word timing evidence is not verified.",
            f"{path}.wordTimingEvidence.verified",
            cue_id=cue_id,
            style_id=style_id,
        )
    expected_hash = hashlib.sha256(str(cue["text"]).encode("utf-8")).hexdigest()
    if evidence["cueTextSha256"] != expected_hash:
        valid = False
        _fail(
            buckets,
            "karaokeAuthenticity",
            "word-timing-text-hash-mismatch",
            "Word timing evidence is not bound to the exact cue text.",
            f"{path}.wordTimingEvidence.cueTextSha256",
            cue_id=cue_id,
            style_id=style_id,
        )
    words = tuple(evidence["words"])
    if not words:
        valid = False
        _fail(
            buckets,
            "karaokeAuthenticity",
            "word-timing-empty",
            "Word-progress karaoke contains no word timing entries.",
            f"{path}.wordTimingEvidence.words",
            cue_id=cue_id,
            style_id=style_id,
        )
    previous_end: int | None = None
    for index, word in enumerate(words):
        word_path = f"{path}.wordTimingEvidence.words[{index}]"
        if word["endMs"] <= word["startMs"]:
            valid = False
            _fail(
                buckets,
                "karaokeAuthenticity",
                "word-duration-invalid",
                "Every word timing entry must have positive duration.",
                word_path,
                cue_id=cue_id,
                style_id=style_id,
            )
        if word["startMs"] < cue["startMs"] or word["endMs"] > cue["endMs"]:
            valid = False
            _fail(
                buckets,
                "karaokeAuthenticity",
                "word-outside-cue-time",
                "Word timing leaves the parent cue interval.",
                word_path,
                cue_id=cue_id,
                style_id=style_id,
            )
        if previous_end is not None and word["startMs"] < previous_end:
            valid = False
            _fail(
                buckets,
                "karaokeAuthenticity",
                "word-timing-overlap",
                "Word timing entries overlap or are out of order.",
                word_path,
                cue_id=cue_id,
                style_id=style_id,
            )
        previous_end = word["endMs"]
    if _without_whitespace("".join(word["text"] for word in words)) != (
        _without_whitespace(str(cue["text"]))
    ):
        valid = False
        _fail(
            buckets,
            "karaokeAuthenticity",
            "word-text-reconstruction-mismatch",
            "Timed word entries do not reconstruct the cue text.",
            f"{path}.wordTimingEvidence.words",
            cue_id=cue_id,
            style_id=style_id,
        )
    return valid


def _safe_rect(
    width: int,
    height: int,
    policy: Mapping[str, Any],
) -> Rect:
    safe = policy["safeArea"]
    left = math.ceil(width * safe["leftRatio"])
    right_margin = math.ceil(width * safe["rightRatio"])
    top = math.ceil(height * safe["topRatio"])
    bottom_margin = math.ceil(height * safe["bottomRatio"])
    return Rect(
        left,
        top,
        width - left - right_margin,
        height - top - bottom_margin,
    )


def _rgb_to_lab(color: str) -> tuple[float, float, float]:
    red, green, blue = _rgb(color)

    def linear(component: int) -> float:
        value = component / 255.0
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    r = linear(red)
    g = linear(green)
    b = linear(blue)
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883

    def pivot(value: float) -> float:
        if value > 216.0 / 24_389.0:
            return value ** (1.0 / 3.0)
        return (24_389.0 / 27.0 * value + 16.0) / 116.0

    fx = pivot(x)
    fy = pivot(y)
    fz = pivot(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _hue_degrees(a_value: float, b_value: float) -> float:
    if a_value == 0.0 and b_value == 0.0:
        return 0.0
    angle = math.degrees(math.atan2(b_value, a_value))
    return angle + 360.0 if angle < 0.0 else angle


def _renderable_codepoint_count(text: str) -> int:
    return sum(
        1
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("C")
    )


def _without_whitespace(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _fail(
    buckets: dict[str, list[Violation]],
    gate: str,
    code: str,
    message: str,
    evidence_path: str,
    *,
    frame_id: str | None = None,
    cue_id: str | None = None,
    style_id: str | None = None,
    measured: int | float | str | None = None,
    threshold: int | float | str | None = None,
) -> None:
    buckets[gate].append(
        Violation(
            gate=gate,
            code=code,
            message=message,
            evidence_path=evidence_path,
            frame_id=frame_id,
            cue_id=cue_id,
            style_id=style_id,
            measured=measured,
            threshold=threshold,
        )
    )


def _violation_sort_key(item: Violation) -> tuple[str, ...]:
    return (
        item.code,
        item.frame_id or "",
        item.cue_id or "",
        item.style_id or "",
        item.evidence_path,
        item.message,
    )


def _rect_text(rect: Rect) -> str:
    return f"{rect.x},{rect.y},{rect.width},{rect.height}"


def _round_metric(value: float) -> float:
    return round(value, 6)


def _rgb(color: str) -> tuple[int, int, int]:
    if not isinstance(color, str) or not _HEX_COLOR.fullmatch(color):
        raise SubtitleVisualQAInputError("color must be #RRGGBB")
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SubtitleVisualQAInputError(f"{path} must be an object")
    return dict(value)


def _array(
    value: Any,
    path: str,
    *,
    minimum: int = 0,
    maximum: int = 1_000_000,
) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise SubtitleVisualQAInputError(f"{path} must be an array")
    result = list(value)
    if not minimum <= len(result) <= maximum:
        raise SubtitleVisualQAInputError(
            f"{path} must contain {minimum}..{maximum} items"
        )
    return result


def _keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    path: str,
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    extra = sorted(keys - allowed)
    if missing:
        raise SubtitleVisualQAInputError(
            f"{path} is missing required keys: {', '.join(missing)}"
        )
    if extra:
        raise SubtitleVisualQAInputError(
            f"{path} contains unsupported keys: {', '.join(extra)}"
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise SubtitleVisualQAInputError(f"{path} must be a string")
    return value


def _bounded_string(
    value: Any,
    path: str,
    minimum: int,
    maximum: int,
) -> str:
    result = _string(value, path)
    if not minimum <= len(result) <= maximum:
        raise SubtitleVisualQAInputError(
            f"{path} length must be {minimum}..{maximum}"
        )
    if result != result.strip():
        raise SubtitleVisualQAInputError(
            f"{path} cannot have leading or trailing whitespace"
        )
    if _CONTROL_CHARACTERS.search(result):
        raise SubtitleVisualQAInputError(
            f"{path} cannot contain control characters"
        )
    return result


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SubtitleVisualQAInputError(f"{path} must be a boolean")
    return value


def _integer_between(
    value: Any,
    path: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SubtitleVisualQAInputError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise SubtitleVisualQAInputError(
            f"{path} must be between {minimum} and {maximum}"
        )
    return value


def _number_between(
    value: Any,
    path: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubtitleVisualQAInputError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise SubtitleVisualQAInputError(
            f"{path} must be finite and between {minimum} and {maximum}"
        )
    return result


def _enum_string(
    value: Any,
    path: str,
    allowed: set[str] | frozenset[str],
) -> str:
    result = _string(value, path)
    if result not in allowed:
        raise SubtitleVisualQAInputError(
            f"{path} must be one of {', '.join(sorted(allowed))}"
        )
    return result


def _sha256(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _SHA256.fullmatch(result):
        raise SubtitleVisualQAInputError(
            f"{path} must be a lowercase SHA-256 digest"
        )
    return result


def _color(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _HEX_COLOR.fullmatch(result):
        raise SubtitleVisualQAInputError(f"{path} must be #RRGGBB")
    return result.upper()


def _unique_strings(
    value: Any,
    path: str,
    *,
    allowed: set[str] | frozenset[str] | None = None,
    minimum: int = 0,
    maximum: int = 100_000,
) -> tuple[str, ...]:
    values = _array(value, path, minimum=minimum, maximum=maximum)
    result: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        item = _bounded_string(raw, f"{path}[{index}]", 1, 320)
        if allowed is not None and item not in allowed:
            raise SubtitleVisualQAInputError(
                f"{path}[{index}] contains an unsupported value"
            )
        if item in seen:
            raise SubtitleVisualQAInputError(
                f"{path}[{index}] duplicates {item!r}"
            )
        seen.add(item)
        result.append(item)
    return tuple(result)


__all__ = [
    "SUBTITLE_VISUAL_QA_REQUEST_KIND",
    "SUBTITLE_VISUAL_QA_RESULT_KIND",
    "SUBTITLE_VISUAL_QA_SCHEMA_VERSION",
    "GateResult",
    "Rect",
    "SubtitleVisualQAInputError",
    "SubtitleVisualQAMetrics",
    "SubtitleVisualQAResult",
    "Violation",
    "canonical_json",
    "contrast_ratio",
    "default_subtitle_visual_qa_policy",
    "delta_e_2000",
    "deterministic_sha256",
    "evaluate_subtitle_visual_qa",
    "relative_luminance",
]
