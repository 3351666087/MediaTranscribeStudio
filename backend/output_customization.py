"""Pure, reversible, and fail-closed output customization.

This module is a configuration orchestration boundary.  It does not render a
report, generate subtitles, invoke FFmpeg, inspect fonts, or mutate media.
Instead, it resolves a complete immutable customization snapshot that can be
compiled into the existing report-style, subtitle-output, subtitle-delivery,
and subtitle-visual-QA contracts.

The public contract deliberately keeps safety invariants in the canonical
payload.  A consumer cannot claim source overwrite, verified fonts, HDR
burn-in, or word-level karaoke merely by setting a presentation option.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import ntpath
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


OUTPUT_CUSTOMIZATION_SCHEMA_VERSION = "1.0.0"
OUTPUT_CUSTOMIZATION_KIND = "output-customization"

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1F\x7F]")
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_TEMPLATE_TOKEN = re.compile(r"\{([A-Za-z][A-Za-z0-9]*)\}")
_ALLOWED_FILENAME_TOKENS = frozenset(
    {"sourceStem", "artifact", "language", "date", "speakerCount"}
)

_REPORT_LAYOUTS = frozenset(
    {
        "modern-editorial",
        "conversation-focus",
        "executive-brief",
        "compact-review",
        "accessible-large-print",
        "archive-monochrome",
    }
)
_REPORT_DENSITIES = frozenset({"relaxed", "comfortable", "compact"})
_PAPER_SIZES = frozenset({"A4", "A5", "Letter", "Legal"})
_ORIENTATIONS = frozenset({"portrait", "landscape"})
_COVER_STYLES = frozenset({"immersive", "editorial", "minimal", "none"})
_CHAPTER_SOURCES = frozenset(
    {"semantic-sections", "fixed-interval", "manual-markers"}
)
_LEGEND_POSITIONS = frozenset(
    {"front-matter", "before-transcript", "appendix"}
)
_TIMESTAMP_FORMATS = frozenset(
    {
        "hh:mm:ss",
        "hh:mm:ss.mmm",
        "mm:ss",
        "seconds",
        "smpte-non-drop",
        "smpte-drop-frame",
    }
)
_TIMESTAMP_PLACEMENTS = frozenset({"inline", "gutter", "block"})
_SMPTE_FRAME_RATES = frozenset({23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0})
_DROP_FRAME_RATES = frozenset({29.97, 59.94})

_SUBTITLE_FORMATS = frozenset({"srt", "webvtt", "ass"})
_SUBTITLE_THEMES = frozenset(
    {
        "youtube-clean",
        "youtube-bold",
        "minimal-glass",
        "karaoke-highlight",
        "speaker-color",
        "documentary",
        "news-lower-third",
        "custom",
    }
)
_SUBTITLE_ALIGNMENTS = frozenset(
    {
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "middle-center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    }
)
_SPEAKER_COLOR_MODES = frozenset({"automatic", "accessible", "monochrome"})
_SPEAKER_COLOR_ALGORITHMS = frozenset(
    {"oklch-hash-v1", "accessible-oklch-hash-v1", "monochrome-v1"}
)
_KARAOKE_MODES = frozenset({"off", "word-progress"})
_KARAOKE_SOURCES = frozenset(
    {"forced-aligner", "native-word-timestamps", "human-authored"}
)

_DELIVERY_MODES = frozenset({"sidecar", "soft-mux", "burn-in"})
_BURN_IN_STRATEGIES = frozenset(
    {
        "h264-high-quality",
        "h265-high-quality",
        "vp9-high-quality",
        "av1-high-quality",
        "prores-422-hq",
        "ffv1-lossless",
    }
)
_SOFT_MUX_CONTAINERS = frozenset(
    {"source-compatible", "mp4", "mov", "matroska", "webm"}
)
_SOFT_MUX_CODECS = frozenset(
    {"probe-selected", "mov_text", "subrip", "webvtt", "ass"}
)
_DYNAMIC_RANGES = frozenset({"unknown", "sdr", "hdr"})
_DYNAMIC_RANGE_METHODS = frozenset(
    {"ffprobe-color-metadata", "container-and-frame-probe", "human-verified"}
)

_FONT_AVAILABILITY_CLAIMS = frozenset(
    {"not-asserted", "verified-installed", "verified-packaged"}
)
_FONT_EMBEDDING_CLAIMS = frozenset(
    {"not-asserted", "verified-embedded", "verified-not-embedded"}
)
_FONT_EVIDENCE_METHODS = frozenset(
    {
        "windows-font-enumeration",
        "fontconfig-match",
        "coretext-enumeration",
        "package-manifest",
        "font-file-scan",
        "pdfbox-font-inspection",
        "media-attachment-inspection",
    }
)

_REPORT_EXPORT_ORDER = ("pdf", "html", "docx", "odt")
_TRANSCRIPT_EXPORT_ORDER = ("json", "txt", "markdown", "html", "docx", "pdf")
_DATA_EXPORT_ORDER = ("json", "csv", "tsv")
_SUBTITLE_EXPORT_ORDER = ("srt", "webvtt", "ass")
_PACKAGE_FORMATS = frozenset({"directory", "zip"})


class OutputCustomizationError(ValueError):
    """Raised when an output customization violates the public contract."""


@dataclass(frozen=True)
class OutputCustomization:
    """An immutable canonical output-customization snapshot."""

    _canonical_json: str

    def canonical_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of this snapshot."""

        return json.loads(self._canonical_json)

    def canonical_json(self) -> str:
        """Return deterministic UTF-8 canonical JSON."""

        return self._canonical_json

    def deterministic_hash(self) -> str:
        """Return the lowercase SHA-256 of :meth:`canonical_json`."""

        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutputCustomizationChange:
    """A verified before/after snapshot pair for reversible UI edits."""

    before_json: str
    after_json: str
    before_sha256: str
    after_sha256: str

    @classmethod
    def create(
        cls,
        before: OutputCustomization,
        after: OutputCustomization,
    ) -> OutputCustomizationChange:
        return cls(
            before_json=before.canonical_json(),
            after_json=after.canonical_json(),
            before_sha256=before.deterministic_hash(),
            after_sha256=after.deterministic_hash(),
        )

    def verify(self) -> None:
        """Fail if either stored snapshot or digest has been tampered with."""

        _verify_snapshot(self.before_json, self.before_sha256, "before")
        _verify_snapshot(self.after_json, self.after_sha256, "after")

    def before(self) -> OutputCustomization:
        self.verify()
        return validate_output_customization(json.loads(self.before_json))

    def after(self) -> OutputCustomization:
        self.verify()
        return validate_output_customization(json.loads(self.after_json))

    def revert(self, current: OutputCustomization) -> OutputCustomization:
        """Restore ``before`` only when ``current`` exactly matches ``after``."""

        self.verify()
        if not isinstance(current, OutputCustomization):
            raise OutputCustomizationError(
                "current must be an OutputCustomization snapshot"
            )
        if current.deterministic_hash() != self.after_sha256:
            raise OutputCustomizationError(
                "cannot revert: current customization does not match the "
                "verified after snapshot"
            )
        return self.before()


def _default_font_pack() -> dict[str, Any]:
    return {
        "id": "noto-global-sans",
        "primary": "Noto Sans CJK SC",
        "fallbacks": {
            "latin": ["Inter", "Segoe UI", "Arial"],
            "cjk": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Yu Gothic UI",
                "Malgun Gothic",
            ],
            "rtl": ["Noto Sans Arabic", "Noto Sans Hebrew", "Segoe UI"],
            "symbols": ["Noto Sans Symbols 2", "Segoe UI Symbol"],
        },
        "embeddingPolicy": "require-embedded",
        "availabilityClaim": "not-asserted",
        "embeddingClaim": "not-asserted",
        "evidence": None,
    }


def _default_payload() -> dict[str, Any]:
    return {
        "schemaVersion": OUTPUT_CUSTOMIZATION_SCHEMA_VERSION,
        "kind": OUTPUT_CUSTOMIZATION_KIND,
        "profile": "studio-balanced",
        "report": {
            "enabled": True,
            "layout": "modern-editorial",
            "density": "comfortable",
            "paper": {
                "size": "A4",
                "orientation": "portrait",
                "marginsMm": {
                    "top": 18.0,
                    "right": 17.0,
                    "bottom": 19.0,
                    "left": 17.0,
                },
            },
            "fontPack": _default_font_pack(),
            "cover": {
                "enabled": True,
                "style": "editorial",
                "showTitle": True,
                "showSubtitle": True,
                "showSourceMetadata": True,
                "artwork": None,
            },
            "header": {
                "enabled": True,
                "text": "",
                "showDocumentTitle": True,
                "showChapter": False,
                "showPageNumber": False,
            },
            "footer": {
                "enabled": True,
                "text": "",
                "showDocumentTitle": False,
                "showChapter": False,
                "showPageNumber": True,
            },
            "watermark": {
                "enabled": False,
                "text": "",
                "opacity": 0.08,
                "rotationDegrees": -30.0,
                "repeat": "diagonal",
            },
            "chapters": {
                "enabled": True,
                "source": "semantic-sections",
                "intervalMinutes": None,
                "includeTimestamps": True,
                "pageBreakBefore": False,
            },
            "speakerLegend": {
                "enabled": True,
                "position": "before-transcript",
                "showColor": True,
                "showDisplayName": True,
                "showSpeakingTime": False,
            },
            "timestamp": {
                "format": "hh:mm:ss.mmm",
                "placement": "gutter",
                "showEnd": False,
                "frameRate": None,
            },
            "accentColor": "#6D5DFB",
            "highContrast": False,
        },
        "subtitle": {
            "enabled": True,
            "format": "ass",
            "theme": "youtube-clean",
            "fontPack": _default_font_pack(),
            "fontSizePx": 52,
            "fontWeight": 700,
            "italic": False,
            "lineHeight": 1.18,
            "alignment": "bottom-center",
            "safeArea": {
                "horizontalPercent": 5.0,
                "topPercent": 5.0,
                "bottomPercent": 8.0,
            },
            "foregroundColor": "#FFFFFF",
            "activeWordColor": "#FFD84D",
            "outline": {"color": "#000000", "widthPx": 3.0},
            "shadow": {
                "color": "#000000",
                "offsetXPx": 0.0,
                "offsetYPx": 2.0,
                "blurPx": 4.0,
                "opacity": 0.72,
            },
            "background": {
                "color": "#000000",
                "opacity": 0.36,
                "radiusPx": 10.0,
                "paddingHorizontalPx": 18.0,
                "paddingVerticalPx": 10.0,
            },
            "cuePolicy": {
                "maxCharactersPerLine": 42,
                "maxLines": 2,
                "maxReadingSpeed": 20.0,
                "minCueMs": 800,
                "maxCueMs": 7000,
                "gapMs": 80,
            },
            "speakerLabels": {
                "enabled": True,
                "template": "{speaker}",
                "position": "inline",
            },
            "speakerColors": {
                "mode": "automatic",
                "algorithm": "oklch-hash-v1",
                "seed": "MediaTranscribeStudio",
                "minimumDeltaE": 18.0,
                "collisionFallback": "label-and-pattern",
                "overrides": [],
            },
            "karaoke": {"mode": "off", "evidence": None},
        },
        "delivery": {
            "mode": "sidecar",
            "outputTarget": {
                "binding": "deferred",
                "sourcePath": None,
                "outputPath": None,
            },
            "overwriteExisting": False,
            "atomicPublication": True,
            "preserveMetadata": True,
            "preserveChapters": True,
            "preserveAudio": True,
            "softMux": {
                "container": "source-compatible",
                "subtitleCodec": "probe-selected",
            },
            "burnIn": {
                "strategy": None,
                "sourceDynamicRange": "unknown",
                "dynamicRangeEvidence": None,
                "requireVisualQa": True,
            },
            "executionReady": False,
        },
        "exports": {
            "reportFormats": ["pdf"],
            "transcriptFormats": ["json", "txt"],
            "subtitleAlternates": ["srt", "webvtt"],
            "dataFormats": ["json"],
            "packageFormat": "directory",
            "fileNameTemplate": "{sourceStem}-{artifact}",
            "checksumManifest": True,
            "provenanceManifest": True,
        },
        "safety": {
            "preserveSourceMedia": True,
            "overwriteSourceMedia": False,
            "overwriteExistingOutputs": False,
            "transcriptAuthoritative": True,
            "presentationOnly": True,
            "fontClaimsRequireEvidence": True,
            "rejectFakeKaraoke": True,
            "rejectHdrBurnIn": True,
            "visualQaRequiredForBurnIn": True,
        },
        "reversibility": {
            "workflowReversible": True,
            "sourceMediaImmutable": True,
            "derivedArtifactOnly": True,
            "derivedMediaBitstreamReversible": True,
            "revertAction": "delete-derived-artifacts",
        },
    }


def default_output_customization() -> OutputCustomization:
    """Return the fully materialized safe default snapshot."""

    return validate_output_customization(_default_payload())


def resolve_output_customization(
    overrides: Mapping[str, Any] | None = None,
) -> OutputCustomization:
    """Resolve a strict partial override against documented safe defaults.

    Unknown keys are rejected before merging.  Derived readiness and
    reversibility fields are recomputed only when the caller did not
    explicitly provide them; an explicitly incorrect derived claim fails.
    """

    if overrides is None:
        return default_output_customization()
    raw_overrides = _mapping(overrides, "output customization overrides")
    base = _default_payload()
    merged = _deep_merge_known(base, raw_overrides, "")
    _fill_derived_fields(merged, raw_overrides)
    return validate_output_customization(merged)


def validate_output_customization(
    value: Mapping[str, Any],
) -> OutputCustomization:
    """Validate and normalize a complete output-customization payload."""

    normalized = _parse_root(value)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return OutputCustomization(canonical)


def apply_output_customization_patch(
    current: OutputCustomization,
    overrides: Mapping[str, Any],
) -> OutputCustomizationChange:
    """Apply a strict patch and retain a hash-bound inverse snapshot."""

    if not isinstance(current, OutputCustomization):
        raise OutputCustomizationError(
            "current must be an OutputCustomization snapshot"
        )
    raw_overrides = _mapping(overrides, "output customization patch")
    base = current.canonical_dict()
    merged = _deep_merge_known(base, raw_overrides, "")
    _fill_derived_fields(merged, raw_overrides)
    after = validate_output_customization(merged)
    return OutputCustomizationChange.create(current, after)


def canonical_output_customization_dict(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_output_customization(value).canonical_dict()


def canonical_output_customization_json(value: Mapping[str, Any]) -> str:
    return validate_output_customization(value).canonical_json()


def deterministic_output_customization_hash(
    value: Mapping[str, Any],
) -> str:
    return validate_output_customization(value).deterministic_hash()


def _verify_snapshot(payload: str, expected_hash: str, name: str) -> None:
    if not isinstance(payload, str):
        raise OutputCustomizationError(f"{name}_json must be a string")
    if not _SHA256.fullmatch(expected_hash):
        raise OutputCustomizationError(
            f"{name}_sha256 must be a lowercase SHA-256 digest"
        )
    actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual != expected_hash:
        raise OutputCustomizationError(f"{name} snapshot hash mismatch")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise OutputCustomizationError(
            f"{name} snapshot is not valid JSON"
        ) from exc
    canonical = validate_output_customization(decoded).canonical_json()
    if canonical != payload:
        raise OutputCustomizationError(
            f"{name} snapshot is not canonical output-customization JSON"
        )


def _fill_derived_fields(
    merged: dict[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    delivery_override = overrides.get("delivery")
    explicit_ready = isinstance(delivery_override, Mapping) and (
        "executionReady" in delivery_override
    )
    target = merged.get("delivery", {}).get("outputTarget", {})
    expected_ready = target.get("binding") == "bound"
    if not explicit_ready:
        merged["delivery"]["executionReady"] = expected_ready

    if "reversibility" not in overrides:
        mode = merged.get("delivery", {}).get("mode")
        merged["reversibility"] = _expected_reversibility(mode)


def _expected_reversibility(mode: Any) -> dict[str, Any]:
    return {
        "workflowReversible": True,
        "sourceMediaImmutable": True,
        "derivedArtifactOnly": True,
        "derivedMediaBitstreamReversible": mode in {"sidecar", "soft-mux"},
        "revertAction": "delete-derived-artifacts",
    }


def _deep_merge_known(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    path: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        field = f"{path}.{key}" if path else key
        if key not in base:
            raise OutputCustomizationError(f"{field} is an unknown property")
        base_value = base[key]
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge_known(base_value, value, field)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_root(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "output customization")
    _exact_keys(
        payload,
        {
            "schemaVersion",
            "kind",
            "profile",
            "report",
            "subtitle",
            "delivery",
            "exports",
            "safety",
            "reversibility",
        },
        "output customization",
    )
    if payload["schemaVersion"] != OUTPUT_CUSTOMIZATION_SCHEMA_VERSION:
        raise OutputCustomizationError(
            "schemaVersion must be "
            f"{OUTPUT_CUSTOMIZATION_SCHEMA_VERSION!r}"
        )
    if payload["kind"] != OUTPUT_CUSTOMIZATION_KIND:
        raise OutputCustomizationError(
            f"kind must be {OUTPUT_CUSTOMIZATION_KIND!r}"
        )
    profile = _enum(
        payload["profile"],
        "profile",
        {
            "studio-balanced",
            "editorial",
            "compact-review",
            "accessibility",
            "custom",
        },
    )
    report = _parse_report(payload["report"])
    subtitle = _parse_subtitle(payload["subtitle"])
    delivery = _parse_delivery(payload["delivery"])
    exports = _parse_exports(payload["exports"])
    safety = _parse_safety(payload["safety"])
    reversibility = _parse_reversibility(
        payload["reversibility"], delivery["mode"]
    )

    if report["enabled"] and not exports["reportFormats"]:
        raise OutputCustomizationError(
            "enabled reports require at least one exports.reportFormats entry"
        )
    if not report["enabled"] and exports["reportFormats"]:
        raise OutputCustomizationError(
            "disabled reports require exports.reportFormats to be empty"
        )
    if not subtitle["enabled"] and exports["subtitleAlternates"]:
        raise OutputCustomizationError(
            "disabled subtitles require exports.subtitleAlternates to be empty"
        )
    if subtitle["format"] in exports["subtitleAlternates"]:
        raise OutputCustomizationError(
            "exports.subtitleAlternates cannot repeat subtitle.format"
        )

    return {
        "schemaVersion": OUTPUT_CUSTOMIZATION_SCHEMA_VERSION,
        "kind": OUTPUT_CUSTOMIZATION_KIND,
        "profile": profile,
        "report": report,
        "subtitle": subtitle,
        "delivery": delivery,
        "exports": exports,
        "safety": safety,
        "reversibility": reversibility,
    }


def _parse_report(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "report")
    _exact_keys(
        payload,
        {
            "enabled",
            "layout",
            "density",
            "paper",
            "fontPack",
            "cover",
            "header",
            "footer",
            "watermark",
            "chapters",
            "speakerLegend",
            "timestamp",
            "accentColor",
            "highContrast",
        },
        "report",
    )
    return {
        "enabled": _boolean(payload["enabled"], "report.enabled"),
        "layout": _enum(payload["layout"], "report.layout", _REPORT_LAYOUTS),
        "density": _enum(
            payload["density"], "report.density", _REPORT_DENSITIES
        ),
        "paper": _parse_paper(payload["paper"]),
        "fontPack": _parse_font_pack(payload["fontPack"], "report.fontPack"),
        "cover": _parse_cover(payload["cover"]),
        "header": _parse_running_text(payload["header"], "report.header"),
        "footer": _parse_running_text(payload["footer"], "report.footer"),
        "watermark": _parse_watermark(payload["watermark"]),
        "chapters": _parse_chapters(payload["chapters"]),
        "speakerLegend": _parse_speaker_legend(payload["speakerLegend"]),
        "timestamp": _parse_timestamp(payload["timestamp"]),
        "accentColor": _color(payload["accentColor"], "report.accentColor"),
        "highContrast": _boolean(
            payload["highContrast"], "report.highContrast"
        ),
    }


def _parse_paper(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "report.paper")
    _exact_keys(payload, {"size", "orientation", "marginsMm"}, "report.paper")
    margins = _mapping(payload["marginsMm"], "report.paper.marginsMm")
    _exact_keys(
        margins,
        {"top", "right", "bottom", "left"},
        "report.paper.marginsMm",
    )
    return {
        "size": _enum(payload["size"], "report.paper.size", _PAPER_SIZES),
        "orientation": _enum(
            payload["orientation"],
            "report.paper.orientation",
            _ORIENTATIONS,
        ),
        "marginsMm": {
            side: _number(
                margins[side],
                f"report.paper.marginsMm.{side}",
                minimum=0,
                maximum=50,
            )
            for side in ("top", "right", "bottom", "left")
        },
    }


def _parse_font_pack(value: Any, name: str) -> dict[str, Any]:
    payload = _mapping(value, name)
    _exact_keys(
        payload,
        {
            "id",
            "primary",
            "fallbacks",
            "embeddingPolicy",
            "availabilityClaim",
            "embeddingClaim",
            "evidence",
        },
        name,
    )
    fallbacks = _mapping(payload["fallbacks"], f"{name}.fallbacks")
    _exact_keys(
        fallbacks,
        {"latin", "cjk", "rtl", "symbols"},
        f"{name}.fallbacks",
    )
    availability = _enum(
        payload["availabilityClaim"],
        f"{name}.availabilityClaim",
        _FONT_AVAILABILITY_CLAIMS,
    )
    embedding = _enum(
        payload["embeddingClaim"],
        f"{name}.embeddingClaim",
        _FONT_EMBEDDING_CLAIMS,
    )
    evidence = _parse_font_evidence(payload["evidence"], f"{name}.evidence")
    claims_verified = availability != "not-asserted" or embedding != "not-asserted"
    if claims_verified and evidence is None:
        raise OutputCustomizationError(
            f"{name} verified font claims require SHA-256-bound evidence"
        )
    if not claims_verified and evidence is not None:
        raise OutputCustomizationError(
            f"{name}.evidence requires a verified availability or embedding "
            "claim"
        )

    return {
        "id": _identifier(payload["id"], f"{name}.id"),
        "primary": _text(
            payload["primary"], f"{name}.primary", minimum=1, maximum=160
        ),
        "fallbacks": {
            script: _text_list(
                fallbacks[script],
                f"{name}.fallbacks.{script}",
                minimum_items=1,
                maximum_items=24,
            )
            for script in ("latin", "cjk", "rtl", "symbols")
        },
        "embeddingPolicy": _enum(
            payload["embeddingPolicy"],
            f"{name}.embeddingPolicy",
            {"require-embedded", "prefer-embedded", "allow-not-embedded"},
        ),
        "availabilityClaim": availability,
        "embeddingClaim": embedding,
        "evidence": evidence,
    }


def _parse_font_evidence(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = _mapping(value, name)
    _exact_keys(
        payload,
        {"verified", "method", "manifestSha256", "fontFileSha256"},
        name,
    )
    verified = _boolean(payload["verified"], f"{name}.verified")
    if not verified:
        raise OutputCustomizationError(f"{name}.verified must be true")
    hashes = _sha256_list(
        payload["fontFileSha256"],
        f"{name}.fontFileSha256",
        minimum_items=1,
        maximum_items=256,
    )
    return {
        "verified": True,
        "method": _enum(
            payload["method"],
            f"{name}.method",
            _FONT_EVIDENCE_METHODS,
        ),
        "manifestSha256": _sha256(
            payload["manifestSha256"], f"{name}.manifestSha256"
        ),
        "fontFileSha256": sorted(hashes),
    }


def _parse_cover(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "report.cover")
    _exact_keys(
        payload,
        {
            "enabled",
            "style",
            "showTitle",
            "showSubtitle",
            "showSourceMetadata",
            "artwork",
        },
        "report.cover",
    )
    enabled = _boolean(payload["enabled"], "report.cover.enabled")
    style = _enum(payload["style"], "report.cover.style", _COVER_STYLES)
    if not enabled and style != "none":
        raise OutputCustomizationError(
            "disabled report.cover requires report.cover.style='none'"
        )
    if enabled and style == "none":
        raise OutputCustomizationError(
            "enabled report.cover cannot use report.cover.style='none'"
        )
    artwork = _parse_artwork(payload["artwork"])
    return {
        "enabled": enabled,
        "style": style,
        "showTitle": _boolean(
            payload["showTitle"], "report.cover.showTitle"
        ),
        "showSubtitle": _boolean(
            payload["showSubtitle"], "report.cover.showSubtitle"
        ),
        "showSourceMetadata": _boolean(
            payload["showSourceMetadata"],
            "report.cover.showSourceMetadata",
        ),
        "artwork": artwork,
    }


def _parse_artwork(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = _mapping(value, "report.cover.artwork")
    _exact_keys(
        payload,
        {"assetId", "sha256", "fit", "opacity"},
        "report.cover.artwork",
    )
    return {
        "assetId": _identifier(
            payload["assetId"], "report.cover.artwork.assetId"
        ),
        "sha256": _sha256(
            payload["sha256"], "report.cover.artwork.sha256"
        ),
        "fit": _enum(
            payload["fit"],
            "report.cover.artwork.fit",
            {"cover", "contain", "fill"},
        ),
        "opacity": _number(
            payload["opacity"],
            "report.cover.artwork.opacity",
            minimum=0,
            maximum=1,
        ),
    }


def _parse_running_text(value: Any, name: str) -> dict[str, Any]:
    payload = _mapping(value, name)
    _exact_keys(
        payload,
        {
            "enabled",
            "text",
            "showDocumentTitle",
            "showChapter",
            "showPageNumber",
        },
        name,
    )
    return {
        "enabled": _boolean(payload["enabled"], f"{name}.enabled"),
        "text": _text(
            payload["text"],
            f"{name}.text",
            minimum=0,
            maximum=240,
            allow_surrounding_whitespace=True,
        ),
        "showDocumentTitle": _boolean(
            payload["showDocumentTitle"], f"{name}.showDocumentTitle"
        ),
        "showChapter": _boolean(
            payload["showChapter"], f"{name}.showChapter"
        ),
        "showPageNumber": _boolean(
            payload["showPageNumber"], f"{name}.showPageNumber"
        ),
    }


def _parse_watermark(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "report.watermark")
    _exact_keys(
        payload,
        {"enabled", "text", "opacity", "rotationDegrees", "repeat"},
        "report.watermark",
    )
    enabled = _boolean(payload["enabled"], "report.watermark.enabled")
    text = _text(
        payload["text"],
        "report.watermark.text",
        minimum=0,
        maximum=120,
        allow_surrounding_whitespace=True,
    )
    if enabled and not text.strip():
        raise OutputCustomizationError(
            "enabled report.watermark requires visible text"
        )
    return {
        "enabled": enabled,
        "text": text,
        "opacity": _number(
            payload["opacity"],
            "report.watermark.opacity",
            minimum=0.01,
            maximum=0.35,
        ),
        "rotationDegrees": _number(
            payload["rotationDegrees"],
            "report.watermark.rotationDegrees",
            minimum=-90,
            maximum=90,
        ),
        "repeat": _enum(
            payload["repeat"],
            "report.watermark.repeat",
            {"none", "diagonal", "grid"},
        ),
    }


def _parse_chapters(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "report.chapters")
    _exact_keys(
        payload,
        {
            "enabled",
            "source",
            "intervalMinutes",
            "includeTimestamps",
            "pageBreakBefore",
        },
        "report.chapters",
    )
    source = _enum(
        payload["source"], "report.chapters.source", _CHAPTER_SOURCES
    )
    interval = payload["intervalMinutes"]
    if source == "fixed-interval":
        interval_value: float | None = _number(
            interval,
            "report.chapters.intervalMinutes",
            minimum=1,
            maximum=240,
        )
    else:
        if interval is not None:
            raise OutputCustomizationError(
                "report.chapters.intervalMinutes is only valid for "
                "source='fixed-interval'"
            )
        interval_value = None
    return {
        "enabled": _boolean(payload["enabled"], "report.chapters.enabled"),
        "source": source,
        "intervalMinutes": interval_value,
        "includeTimestamps": _boolean(
            payload["includeTimestamps"],
            "report.chapters.includeTimestamps",
        ),
        "pageBreakBefore": _boolean(
            payload["pageBreakBefore"], "report.chapters.pageBreakBefore"
        ),
    }


def _parse_speaker_legend(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "report.speakerLegend")
    _exact_keys(
        payload,
        {
            "enabled",
            "position",
            "showColor",
            "showDisplayName",
            "showSpeakingTime",
        },
        "report.speakerLegend",
    )
    return {
        "enabled": _boolean(
            payload["enabled"], "report.speakerLegend.enabled"
        ),
        "position": _enum(
            payload["position"],
            "report.speakerLegend.position",
            _LEGEND_POSITIONS,
        ),
        "showColor": _boolean(
            payload["showColor"], "report.speakerLegend.showColor"
        ),
        "showDisplayName": _boolean(
            payload["showDisplayName"],
            "report.speakerLegend.showDisplayName",
        ),
        "showSpeakingTime": _boolean(
            payload["showSpeakingTime"],
            "report.speakerLegend.showSpeakingTime",
        ),
    }


def _parse_timestamp(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "report.timestamp")
    _exact_keys(
        payload,
        {"format", "placement", "showEnd", "frameRate"},
        "report.timestamp",
    )
    timestamp_format = _enum(
        payload["format"], "report.timestamp.format", _TIMESTAMP_FORMATS
    )
    frame_rate_raw = payload["frameRate"]
    if timestamp_format.startswith("smpte-"):
        frame_rate = _number(
            frame_rate_raw,
            "report.timestamp.frameRate",
            minimum=1,
            maximum=120,
        )
        if frame_rate not in _SMPTE_FRAME_RATES:
            raise OutputCustomizationError(
                "report.timestamp.frameRate is not a supported SMPTE rate"
            )
        if (
            timestamp_format == "smpte-drop-frame"
            and frame_rate not in _DROP_FRAME_RATES
        ):
            raise OutputCustomizationError(
                "SMPTE drop-frame requires 29.97 or 59.94 fps"
            )
    else:
        if frame_rate_raw is not None:
            raise OutputCustomizationError(
                "report.timestamp.frameRate is only valid for SMPTE formats"
            )
        frame_rate = None
    return {
        "format": timestamp_format,
        "placement": _enum(
            payload["placement"],
            "report.timestamp.placement",
            _TIMESTAMP_PLACEMENTS,
        ),
        "showEnd": _boolean(
            payload["showEnd"], "report.timestamp.showEnd"
        ),
        "frameRate": frame_rate,
    }


def _parse_subtitle(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle")
    _exact_keys(
        payload,
        {
            "enabled",
            "format",
            "theme",
            "fontPack",
            "fontSizePx",
            "fontWeight",
            "italic",
            "lineHeight",
            "alignment",
            "safeArea",
            "foregroundColor",
            "activeWordColor",
            "outline",
            "shadow",
            "background",
            "cuePolicy",
            "speakerLabels",
            "speakerColors",
            "karaoke",
        },
        "subtitle",
    )
    subtitle_format = _enum(
        payload["format"], "subtitle.format", _SUBTITLE_FORMATS
    )
    theme = _enum(payload["theme"], "subtitle.theme", _SUBTITLE_THEMES)
    karaoke = _parse_karaoke(payload["karaoke"])
    if karaoke["mode"] == "word-progress" and subtitle_format != "ass":
        raise OutputCustomizationError(
            "word-progress karaoke requires subtitle.format='ass'"
        )
    if theme == "karaoke-highlight" and karaoke["mode"] != "word-progress":
        raise OutputCustomizationError(
            "karaoke-highlight requires verified word-progress timing"
        )

    return {
        "enabled": _boolean(payload["enabled"], "subtitle.enabled"),
        "format": subtitle_format,
        "theme": theme,
        "fontPack": _parse_font_pack(
            payload["fontPack"], "subtitle.fontPack"
        ),
        "fontSizePx": _integer(
            payload["fontSizePx"],
            "subtitle.fontSizePx",
            minimum=8,
            maximum=240,
        ),
        "fontWeight": _integer(
            payload["fontWeight"],
            "subtitle.fontWeight",
            minimum=100,
            maximum=900,
            multiple_of=50,
        ),
        "italic": _boolean(payload["italic"], "subtitle.italic"),
        "lineHeight": _number(
            payload["lineHeight"],
            "subtitle.lineHeight",
            minimum=0.8,
            maximum=2.5,
        ),
        "alignment": _enum(
            payload["alignment"],
            "subtitle.alignment",
            _SUBTITLE_ALIGNMENTS,
        ),
        "safeArea": _parse_safe_area(payload["safeArea"]),
        "foregroundColor": _color(
            payload["foregroundColor"], "subtitle.foregroundColor"
        ),
        "activeWordColor": _color(
            payload["activeWordColor"], "subtitle.activeWordColor"
        ),
        "outline": _parse_outline(payload["outline"]),
        "shadow": _parse_shadow(payload["shadow"]),
        "background": _parse_background(payload["background"]),
        "cuePolicy": _parse_cue_policy(payload["cuePolicy"]),
        "speakerLabels": _parse_speaker_labels(payload["speakerLabels"]),
        "speakerColors": _parse_speaker_colors(payload["speakerColors"]),
        "karaoke": karaoke,
    }


def _parse_safe_area(value: Any) -> dict[str, float]:
    payload = _mapping(value, "subtitle.safeArea")
    _exact_keys(
        payload,
        {"horizontalPercent", "topPercent", "bottomPercent"},
        "subtitle.safeArea",
    )
    return {
        key: _number(
            payload[key],
            f"subtitle.safeArea.{key}",
            minimum=0,
            maximum=30,
        )
        for key in ("horizontalPercent", "topPercent", "bottomPercent")
    }


def _parse_outline(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.outline")
    _exact_keys(payload, {"color", "widthPx"}, "subtitle.outline")
    return {
        "color": _color(payload["color"], "subtitle.outline.color"),
        "widthPx": _number(
            payload["widthPx"],
            "subtitle.outline.widthPx",
            minimum=0,
            maximum=12,
        ),
    }


def _parse_shadow(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.shadow")
    _exact_keys(
        payload,
        {"color", "offsetXPx", "offsetYPx", "blurPx", "opacity"},
        "subtitle.shadow",
    )
    return {
        "color": _color(payload["color"], "subtitle.shadow.color"),
        "offsetXPx": _number(
            payload["offsetXPx"],
            "subtitle.shadow.offsetXPx",
            minimum=-24,
            maximum=24,
        ),
        "offsetYPx": _number(
            payload["offsetYPx"],
            "subtitle.shadow.offsetYPx",
            minimum=-24,
            maximum=24,
        ),
        "blurPx": _number(
            payload["blurPx"],
            "subtitle.shadow.blurPx",
            minimum=0,
            maximum=32,
        ),
        "opacity": _number(
            payload["opacity"],
            "subtitle.shadow.opacity",
            minimum=0,
            maximum=1,
        ),
    }


def _parse_background(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.background")
    _exact_keys(
        payload,
        {
            "color",
            "opacity",
            "radiusPx",
            "paddingHorizontalPx",
            "paddingVerticalPx",
        },
        "subtitle.background",
    )
    return {
        "color": _color(payload["color"], "subtitle.background.color"),
        "opacity": _number(
            payload["opacity"],
            "subtitle.background.opacity",
            minimum=0,
            maximum=1,
        ),
        "radiusPx": _number(
            payload["radiusPx"],
            "subtitle.background.radiusPx",
            minimum=0,
            maximum=80,
        ),
        "paddingHorizontalPx": _number(
            payload["paddingHorizontalPx"],
            "subtitle.background.paddingHorizontalPx",
            minimum=0,
            maximum=120,
        ),
        "paddingVerticalPx": _number(
            payload["paddingVerticalPx"],
            "subtitle.background.paddingVerticalPx",
            minimum=0,
            maximum=120,
        ),
    }


def _parse_cue_policy(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.cuePolicy")
    _exact_keys(
        payload,
        {
            "maxCharactersPerLine",
            "maxLines",
            "maxReadingSpeed",
            "minCueMs",
            "maxCueMs",
            "gapMs",
        },
        "subtitle.cuePolicy",
    )
    minimum = _integer(
        payload["minCueMs"],
        "subtitle.cuePolicy.minCueMs",
        minimum=100,
        maximum=30_000,
    )
    maximum = _integer(
        payload["maxCueMs"],
        "subtitle.cuePolicy.maxCueMs",
        minimum=100,
        maximum=60_000,
    )
    if minimum > maximum:
        raise OutputCustomizationError(
            "subtitle.cuePolicy.minCueMs cannot exceed maxCueMs"
        )
    return {
        "maxCharactersPerLine": _integer(
            payload["maxCharactersPerLine"],
            "subtitle.cuePolicy.maxCharactersPerLine",
            minimum=4,
            maximum=160,
        ),
        "maxLines": _integer(
            payload["maxLines"],
            "subtitle.cuePolicy.maxLines",
            minimum=1,
            maximum=6,
        ),
        "maxReadingSpeed": _number(
            payload["maxReadingSpeed"],
            "subtitle.cuePolicy.maxReadingSpeed",
            minimum=1,
            maximum=100,
        ),
        "minCueMs": minimum,
        "maxCueMs": maximum,
        "gapMs": _integer(
            payload["gapMs"],
            "subtitle.cuePolicy.gapMs",
            minimum=0,
            maximum=5000,
        ),
    }


def _parse_speaker_labels(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.speakerLabels")
    _exact_keys(
        payload,
        {"enabled", "template", "position"},
        "subtitle.speakerLabels",
    )
    template = _text(
        payload["template"],
        "subtitle.speakerLabels.template",
        minimum=1,
        maximum=80,
    )
    if "{speaker}" not in template:
        raise OutputCustomizationError(
            "subtitle.speakerLabels.template must contain {speaker}"
        )
    return {
        "enabled": _boolean(
            payload["enabled"], "subtitle.speakerLabels.enabled"
        ),
        "template": template,
        "position": _enum(
            payload["position"],
            "subtitle.speakerLabels.position",
            {"inline", "line-above", "badge"},
        ),
    }


def _parse_speaker_colors(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.speakerColors")
    _exact_keys(
        payload,
        {
            "mode",
            "algorithm",
            "seed",
            "minimumDeltaE",
            "collisionFallback",
            "overrides",
        },
        "subtitle.speakerColors",
    )
    mode = _enum(
        payload["mode"],
        "subtitle.speakerColors.mode",
        _SPEAKER_COLOR_MODES,
    )
    algorithm = _enum(
        payload["algorithm"],
        "subtitle.speakerColors.algorithm",
        _SPEAKER_COLOR_ALGORITHMS,
    )
    expected_algorithm = {
        "automatic": "oklch-hash-v1",
        "accessible": "accessible-oklch-hash-v1",
        "monochrome": "monochrome-v1",
    }[mode]
    if algorithm != expected_algorithm:
        raise OutputCustomizationError(
            "subtitle.speakerColors.algorithm is incompatible with mode"
        )
    minimum_delta = _number(
        payload["minimumDeltaE"],
        "subtitle.speakerColors.minimumDeltaE",
        minimum=0,
        maximum=100,
    )
    if mode == "monochrome" and minimum_delta != 0:
        raise OutputCustomizationError(
            "monochrome speaker colors require minimumDeltaE=0"
        )
    if mode != "monochrome" and minimum_delta < 10:
        raise OutputCustomizationError(
            "colored speaker modes require minimumDeltaE >= 10"
        )

    raw_overrides = payload["overrides"]
    if not isinstance(raw_overrides, list):
        raise OutputCustomizationError(
            "subtitle.speakerColors.overrides must be an array"
        )
    normalized_overrides: list[dict[str, str]] = []
    seen_speakers: set[str] = set()
    for index, item in enumerate(raw_overrides):
        name = f"subtitle.speakerColors.overrides[{index}]"
        entry = _mapping(item, name)
        _exact_keys(entry, {"speakerId", "color"}, name)
        speaker_id = _text(
            entry["speakerId"],
            f"{name}.speakerId",
            minimum=1,
            maximum=256,
        )
        if speaker_id in seen_speakers:
            raise OutputCustomizationError(
                "subtitle.speakerColors.overrides cannot repeat speakerId"
            )
        seen_speakers.add(speaker_id)
        normalized_overrides.append(
            {
                "speakerId": speaker_id,
                "color": _color(entry["color"], f"{name}.color"),
            }
        )
    normalized_overrides.sort(key=lambda item: item["speakerId"])
    return {
        "mode": mode,
        "algorithm": algorithm,
        "seed": _text(
            payload["seed"],
            "subtitle.speakerColors.seed",
            minimum=1,
            maximum=128,
        ),
        "minimumDeltaE": minimum_delta,
        "collisionFallback": _constant(
            payload["collisionFallback"],
            "label-and-pattern",
            "subtitle.speakerColors.collisionFallback",
        ),
        "overrides": normalized_overrides,
    }


def _parse_karaoke(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.karaoke")
    _exact_keys(payload, {"mode", "evidence"}, "subtitle.karaoke")
    mode = _enum(payload["mode"], "subtitle.karaoke.mode", _KARAOKE_MODES)
    raw_evidence = payload["evidence"]
    if mode == "off":
        if raw_evidence is not None:
            raise OutputCustomizationError(
                "subtitle.karaoke.evidence must be null when karaoke is off"
            )
        evidence = None
    else:
        evidence = _parse_karaoke_evidence(raw_evidence)
    return {"mode": mode, "evidence": evidence}


def _parse_karaoke_evidence(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "subtitle.karaoke.evidence")
    _exact_keys(
        payload,
        {
            "source",
            "verified",
            "subtitleSha256",
            "transcriptSha256",
            "wordCount",
        },
        "subtitle.karaoke.evidence",
    )
    if not _boolean(
        payload["verified"], "subtitle.karaoke.evidence.verified"
    ):
        raise OutputCustomizationError(
            "subtitle.karaoke.evidence.verified must be true"
        )
    return {
        "source": _enum(
            payload["source"],
            "subtitle.karaoke.evidence.source",
            _KARAOKE_SOURCES,
        ),
        "verified": True,
        "subtitleSha256": _sha256(
            payload["subtitleSha256"],
            "subtitle.karaoke.evidence.subtitleSha256",
        ),
        "transcriptSha256": _sha256(
            payload["transcriptSha256"],
            "subtitle.karaoke.evidence.transcriptSha256",
        ),
        "wordCount": _integer(
            payload["wordCount"],
            "subtitle.karaoke.evidence.wordCount",
            minimum=1,
            maximum=100_000_000,
        ),
    }


def _parse_delivery(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "delivery")
    _exact_keys(
        payload,
        {
            "mode",
            "outputTarget",
            "overwriteExisting",
            "atomicPublication",
            "preserveMetadata",
            "preserveChapters",
            "preserveAudio",
            "softMux",
            "burnIn",
            "executionReady",
        },
        "delivery",
    )
    mode = _enum(payload["mode"], "delivery.mode", _DELIVERY_MODES)
    target = _parse_output_target(payload["outputTarget"])
    ready = _boolean(payload["executionReady"], "delivery.executionReady")
    expected_ready = target["binding"] == "bound"
    if ready != expected_ready:
        raise OutputCustomizationError(
            "delivery.executionReady must exactly reflect bound output paths"
        )
    overwrite = _boolean(
        payload["overwriteExisting"], "delivery.overwriteExisting"
    )
    if overwrite:
        raise OutputCustomizationError(
            "delivery.overwriteExisting must remain false"
        )
    atomic = _boolean(
        payload["atomicPublication"], "delivery.atomicPublication"
    )
    if not atomic:
        raise OutputCustomizationError(
            "delivery.atomicPublication must remain true"
        )
    soft_mux = _parse_soft_mux(payload["softMux"])
    burn_in = _parse_burn_in(payload["burnIn"])
    if mode == "burn-in":
        if burn_in["strategy"] is None:
            raise OutputCustomizationError(
                "burn-in delivery requires an explicit high-quality strategy"
            )
        if burn_in["sourceDynamicRange"] != "sdr":
            raise OutputCustomizationError(
                "burn-in requires verified SDR; HDR and unknown inputs fail "
                "closed"
            )
        if burn_in["dynamicRangeEvidence"] is None:
            raise OutputCustomizationError(
                "burn-in requires SHA-256-bound dynamic-range evidence"
            )
        if not burn_in["requireVisualQa"]:
            raise OutputCustomizationError(
                "burn-in requires representative-frame visual QA"
            )
    elif burn_in["strategy"] is not None:
        raise OutputCustomizationError(
            "delivery.burnIn.strategy must be null unless mode='burn-in'"
        )

    return {
        "mode": mode,
        "outputTarget": target,
        "overwriteExisting": False,
        "atomicPublication": True,
        "preserveMetadata": _boolean(
            payload["preserveMetadata"], "delivery.preserveMetadata"
        ),
        "preserveChapters": _boolean(
            payload["preserveChapters"], "delivery.preserveChapters"
        ),
        "preserveAudio": _boolean(
            payload["preserveAudio"], "delivery.preserveAudio"
        ),
        "softMux": soft_mux,
        "burnIn": burn_in,
        "executionReady": ready,
    }


def _parse_output_target(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "delivery.outputTarget")
    _exact_keys(
        payload,
        {"binding", "sourcePath", "outputPath"},
        "delivery.outputTarget",
    )
    binding = _enum(
        payload["binding"],
        "delivery.outputTarget.binding",
        {"deferred", "bound"},
    )
    if binding == "deferred":
        if payload["sourcePath"] is not None or payload["outputPath"] is not None:
            raise OutputCustomizationError(
                "deferred delivery paths must both be null"
            )
        source = output = None
    else:
        source = _path(payload["sourcePath"], "delivery.outputTarget.sourcePath")
        output = _path(payload["outputPath"], "delivery.outputTarget.outputPath")
        if _paths_alias(source, output):
            raise OutputCustomizationError(
                "delivery output cannot alias or overwrite source media"
            )
    return {
        "binding": binding,
        "sourcePath": source,
        "outputPath": output,
    }


def _parse_soft_mux(value: Any) -> dict[str, str]:
    payload = _mapping(value, "delivery.softMux")
    _exact_keys(
        payload,
        {"container", "subtitleCodec"},
        "delivery.softMux",
    )
    return {
        "container": _enum(
            payload["container"],
            "delivery.softMux.container",
            _SOFT_MUX_CONTAINERS,
        ),
        "subtitleCodec": _enum(
            payload["subtitleCodec"],
            "delivery.softMux.subtitleCodec",
            _SOFT_MUX_CODECS,
        ),
    }


def _parse_burn_in(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "delivery.burnIn")
    _exact_keys(
        payload,
        {
            "strategy",
            "sourceDynamicRange",
            "dynamicRangeEvidence",
            "requireVisualQa",
        },
        "delivery.burnIn",
    )
    strategy_raw = payload["strategy"]
    strategy = (
        None
        if strategy_raw is None
        else _enum(
            strategy_raw,
            "delivery.burnIn.strategy",
            _BURN_IN_STRATEGIES,
        )
    )
    dynamic_range = _enum(
        payload["sourceDynamicRange"],
        "delivery.burnIn.sourceDynamicRange",
        _DYNAMIC_RANGES,
    )
    raw_evidence = payload["dynamicRangeEvidence"]
    if dynamic_range == "unknown":
        if raw_evidence is not None:
            raise OutputCustomizationError(
                "unknown dynamic range cannot carry verified range evidence"
            )
        evidence = None
    else:
        evidence = _parse_dynamic_range_evidence(raw_evidence)
    return {
        "strategy": strategy,
        "sourceDynamicRange": dynamic_range,
        "dynamicRangeEvidence": evidence,
        "requireVisualQa": _boolean(
            payload["requireVisualQa"], "delivery.burnIn.requireVisualQa"
        ),
    }


def _parse_dynamic_range_evidence(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "delivery.burnIn.dynamicRangeEvidence")
    _exact_keys(
        payload,
        {"verified", "method", "probeSha256"},
        "delivery.burnIn.dynamicRangeEvidence",
    )
    if not _boolean(
        payload["verified"],
        "delivery.burnIn.dynamicRangeEvidence.verified",
    ):
        raise OutputCustomizationError(
            "delivery.burnIn.dynamicRangeEvidence.verified must be true"
        )
    return {
        "verified": True,
        "method": _enum(
            payload["method"],
            "delivery.burnIn.dynamicRangeEvidence.method",
            _DYNAMIC_RANGE_METHODS,
        ),
        "probeSha256": _sha256(
            payload["probeSha256"],
            "delivery.burnIn.dynamicRangeEvidence.probeSha256",
        ),
    }


def _parse_exports(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "exports")
    _exact_keys(
        payload,
        {
            "reportFormats",
            "transcriptFormats",
            "subtitleAlternates",
            "dataFormats",
            "packageFormat",
            "fileNameTemplate",
            "checksumManifest",
            "provenanceManifest",
        },
        "exports",
    )
    checksum = _boolean(
        payload["checksumManifest"], "exports.checksumManifest"
    )
    provenance = _boolean(
        payload["provenanceManifest"], "exports.provenanceManifest"
    )
    if not checksum or not provenance:
        raise OutputCustomizationError(
            "verifiable exports require checksum and provenance manifests"
        )
    return {
        "reportFormats": _ordered_enum_list(
            payload["reportFormats"],
            "exports.reportFormats",
            _REPORT_EXPORT_ORDER,
            minimum_items=0,
        ),
        "transcriptFormats": _ordered_enum_list(
            payload["transcriptFormats"],
            "exports.transcriptFormats",
            _TRANSCRIPT_EXPORT_ORDER,
            minimum_items=1,
        ),
        "subtitleAlternates": _ordered_enum_list(
            payload["subtitleAlternates"],
            "exports.subtitleAlternates",
            _SUBTITLE_EXPORT_ORDER,
            minimum_items=0,
        ),
        "dataFormats": _ordered_enum_list(
            payload["dataFormats"],
            "exports.dataFormats",
            _DATA_EXPORT_ORDER,
            minimum_items=1,
        ),
        "packageFormat": _enum(
            payload["packageFormat"],
            "exports.packageFormat",
            _PACKAGE_FORMATS,
        ),
        "fileNameTemplate": _filename_template(payload["fileNameTemplate"]),
        "checksumManifest": True,
        "provenanceManifest": True,
    }


def _parse_safety(value: Any) -> dict[str, bool]:
    payload = _mapping(value, "safety")
    expected = {
        "preserveSourceMedia": True,
        "overwriteSourceMedia": False,
        "overwriteExistingOutputs": False,
        "transcriptAuthoritative": True,
        "presentationOnly": True,
        "fontClaimsRequireEvidence": True,
        "rejectFakeKaraoke": True,
        "rejectHdrBurnIn": True,
        "visualQaRequiredForBurnIn": True,
    }
    _exact_keys(payload, set(expected), "safety")
    for key, expected_value in expected.items():
        actual = _boolean(payload[key], f"safety.{key}")
        if actual is not expected_value:
            raise OutputCustomizationError(
                f"safety.{key} must remain {str(expected_value).lower()}"
            )
    return expected


def _parse_reversibility(value: Any, mode: str) -> dict[str, Any]:
    payload = _mapping(value, "reversibility")
    expected = _expected_reversibility(mode)
    _exact_keys(payload, set(expected), "reversibility")
    normalized = {
        "workflowReversible": _boolean(
            payload["workflowReversible"],
            "reversibility.workflowReversible",
        ),
        "sourceMediaImmutable": _boolean(
            payload["sourceMediaImmutable"],
            "reversibility.sourceMediaImmutable",
        ),
        "derivedArtifactOnly": _boolean(
            payload["derivedArtifactOnly"],
            "reversibility.derivedArtifactOnly",
        ),
        "derivedMediaBitstreamReversible": _boolean(
            payload["derivedMediaBitstreamReversible"],
            "reversibility.derivedMediaBitstreamReversible",
        ),
        "revertAction": _constant(
            payload["revertAction"],
            "delete-derived-artifacts",
            "reversibility.revertAction",
        ),
    }
    if normalized != expected:
        raise OutputCustomizationError(
            "reversibility must match the selected delivery mode and immutable "
            "source policy"
        )
    return normalized


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OutputCustomizationError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise OutputCustomizationError(f"{name} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise OutputCustomizationError(
            f"{name} is missing: {', '.join(missing)}"
        )
    if unknown:
        raise OutputCustomizationError(
            f"{name} has unknown properties: {', '.join(unknown)}"
        )


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OutputCustomizationError(f"{name} must be a boolean")
    return value


def _number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OutputCustomizationError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise OutputCustomizationError(f"{name} must be finite")
    if not minimum <= normalized <= maximum:
        raise OutputCustomizationError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return normalized


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
    multiple_of: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutputCustomizationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise OutputCustomizationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    if multiple_of is not None and value % multiple_of:
        raise OutputCustomizationError(
            f"{name} must be a multiple of {multiple_of}"
        )
    return value


def _text(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
    allow_surrounding_whitespace: bool = False,
) -> str:
    if not isinstance(value, str):
        raise OutputCustomizationError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if _CONTROL_CHARACTERS.search(normalized):
        raise OutputCustomizationError(f"{name} cannot contain control characters")
    if not allow_surrounding_whitespace and normalized != normalized.strip():
        raise OutputCustomizationError(
            f"{name} cannot start or end with whitespace"
        )
    if not minimum <= len(normalized) <= maximum:
        raise OutputCustomizationError(
            f"{name} length must be between {minimum} and {maximum}"
        )
    if minimum and not normalized.strip():
        raise OutputCustomizationError(f"{name} must contain visible text")
    return normalized


def _identifier(value: Any, name: str) -> str:
    normalized = _text(value, name, minimum=1, maximum=128)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", normalized):
        raise OutputCustomizationError(
            f"{name} must contain only letters, digits, dot, underscore, or hyphen"
        )
    return normalized


def _enum(value: Any, name: str, allowed: set[str] | frozenset[str]) -> str:
    if not isinstance(value, str):
        raise OutputCustomizationError(f"{name} must be a string")
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise OutputCustomizationError(f"{name} must be one of: {choices}")
    return value


def _constant(value: Any, expected: Any, name: str) -> Any:
    if value != expected:
        raise OutputCustomizationError(f"{name} must be {expected!r}")
    return expected


def _color(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX_COLOR.fullmatch(value):
        raise OutputCustomizationError(f"{name} must be a #RRGGBB color")
    return value.upper()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OutputCustomizationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _sha256_list(
    value: Any,
    name: str,
    *,
    minimum_items: int,
    maximum_items: int,
) -> list[str]:
    if not isinstance(value, list):
        raise OutputCustomizationError(f"{name} must be an array")
    if not minimum_items <= len(value) <= maximum_items:
        raise OutputCustomizationError(
            f"{name} must contain between {minimum_items} and "
            f"{maximum_items} entries"
        )
    normalized = [_sha256(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if len(set(normalized)) != len(normalized):
        raise OutputCustomizationError(f"{name} cannot contain duplicates")
    return normalized


def _text_list(
    value: Any,
    name: str,
    *,
    minimum_items: int,
    maximum_items: int,
) -> list[str]:
    if not isinstance(value, list):
        raise OutputCustomizationError(f"{name} must be an array")
    if not minimum_items <= len(value) <= maximum_items:
        raise OutputCustomizationError(
            f"{name} must contain between {minimum_items} and "
            f"{maximum_items} entries"
        )
    normalized = [
        _text(item, f"{name}[{index}]", minimum=1, maximum=160)
        for index, item in enumerate(value)
    ]
    if len(set(normalized)) != len(normalized):
        raise OutputCustomizationError(f"{name} cannot contain duplicates")
    return normalized


def _ordered_enum_list(
    value: Any,
    name: str,
    order: tuple[str, ...],
    *,
    minimum_items: int,
) -> list[str]:
    if not isinstance(value, list):
        raise OutputCustomizationError(f"{name} must be an array")
    if len(value) < minimum_items:
        raise OutputCustomizationError(
            f"{name} must contain at least {minimum_items} entries"
        )
    allowed = set(order)
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(_enum(item, f"{name}[{index}]", allowed))
    if len(set(normalized)) != len(normalized):
        raise OutputCustomizationError(f"{name} cannot contain duplicates")
    selected = set(normalized)
    return [item for item in order if item in selected]


def _filename_template(value: Any) -> str:
    template = _text(
        value,
        "exports.fileNameTemplate",
        minimum=1,
        maximum=160,
    )
    if any(separator in template for separator in ("/", "\\")):
        raise OutputCustomizationError(
            "exports.fileNameTemplate cannot contain path separators"
        )
    tokens = set(_TEMPLATE_TOKEN.findall(template))
    if not tokens:
        raise OutputCustomizationError(
            "exports.fileNameTemplate requires at least one supported token"
        )
    unknown = sorted(tokens - _ALLOWED_FILENAME_TOKENS)
    if unknown:
        raise OutputCustomizationError(
            "exports.fileNameTemplate has unknown tokens: "
            + ", ".join(unknown)
        )
    unmatched = _TEMPLATE_TOKEN.sub("", template)
    if "{" in unmatched or "}" in unmatched:
        raise OutputCustomizationError(
            "exports.fileNameTemplate contains malformed token braces"
        )
    return template


def _path(value: Any, name: str) -> str:
    return _text(value, name, minimum=1, maximum=4096)


def _path_key(path: str) -> str:
    if _WINDOWS_PATH.match(path) or path.startswith(("\\\\", "//")):
        return ntpath.normcase(ntpath.abspath(ntpath.normpath(path)))
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def _paths_alias(left: str, right: str) -> bool:
    return _path_key(left) == _path_key(right)
