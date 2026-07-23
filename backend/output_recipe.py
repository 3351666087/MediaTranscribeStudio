"""Strict native-UI output recipes and canonical execution snapshots.

The TypeScript desktop application intentionally exposes a small, human
readable recipe instead of the much larger evidence-bearing production
``OutputCustomization`` contract.  This module is the trust boundary between
those two representations:

* UI recipes are validated fail-closed with exact keys and bounded values.
* The accepted recipe is stored as deterministic canonical JSON.
* One canonical production customization is compiled for every requested
  subtitle delivery mode; no selected mode is silently dropped.
* Source media is always immutable and every derived media path remains
  inside the job output directory.

The recipe is presentation-only.  It cannot change transcript text, speaker
identity, timestamps, review locks, or acoustic evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .output_customization import (
    OutputCustomization,
    OutputCustomizationError,
    resolve_output_customization,
)


OUTPUT_RECIPE_SCHEMA_VERSION = "1.0.0"

_REPORT_TEMPLATES = (
    "soft-glass",
    "editorial",
    "academic",
    "compact",
)
_REPORT_FONTS = (
    "system-sans",
    "humanist",
    "serif",
    "mono-accent",
    "custom",
)
_PAGE_SIZES = ("a4", "letter", "legal", "screen")
_DENSITIES = ("airy", "balanced", "compact")
_SUBTITLE_THEMES = (
    "youtube-clean",
    "soft-bubble",
    "cinema",
    "high-contrast",
)
_SUBTITLE_SIZES = ("small", "medium", "large", "x-large")
_SAFE_AREAS = ("standard", "broadcast", "generous")
_POSITIONS = ("bottom", "smart", "top")
_SPEAKER_PALETTES = (
    "adaptive-spectrum",
    "candy",
    "ocean",
    "high-contrast",
    "monochrome",
)

# DOCX is deliberately not exposed at this boundary until a real exporter and
# package-level validation gate exist.  Accepting a dead control would be less
# product-safe than rejecting it explicitly.
_EXPORT_FORMATS = (
    "pdf",
    "html",
    "markdown",
    "txt",
    "json",
    "srt",
    "webvtt",
    "ass",
)
_SUBTITLE_FORMATS = ("srt", "webvtt", "ass")
_SUBTITLE_MODES = ("sidecar", "soft-mux", "burn-in")
_CHAPTER_STYLES = ("semantic", "interval", "none")
_TIMESTAMP_STYLES = ("segment", "paragraph", "chapter")
_FILENAME_TOKENS = frozenset(
    {"sourceStem", "artifact", "language", "date", "speakerCount"}
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_BIDI_CONTROL = re.compile(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_TEMPLATE_TOKEN = re.compile(r"\{([A-Za-z][A-Za-z0-9]*)\}")
_PATH_SEPARATOR = re.compile(r"[\\/]")
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class OutputRecipeError(ValueError):
    """Raised when the native desktop output recipe is unsafe or malformed."""


@dataclass(frozen=True)
class OutputRecipe:
    """Immutable canonical native-UI output recipe."""

    _canonical_json: str

    def canonical_json(self) -> str:
        return self._canonical_json

    def canonical_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    def deterministic_hash(self) -> str:
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()

    @property
    def formats(self) -> tuple[str, ...]:
        return tuple(self.canonical_dict()["delivery"]["formats"])

    @property
    def subtitle_modes(self) -> tuple[str, ...]:
        return tuple(self.canonical_dict()["delivery"]["subtitleModes"])

    @property
    def render_pdf(self) -> bool:
        return "pdf" in self.formats


@dataclass(frozen=True)
class CompiledOutputCustomization:
    """One rich production customization compiled from a simple UI recipe."""

    delivery_mode: str
    customization: OutputCustomization


def parse_output_recipe(value: Any) -> OutputRecipe:
    """Validate the exact TypeScript recipe and return canonical JSON."""

    root = _mapping(value, "outputCustomization")
    _exact_keys(
        root,
        {"schemaVersion", "report", "subtitles", "delivery", "finishing"},
        "outputCustomization",
    )
    if root["schemaVersion"] != OUTPUT_RECIPE_SCHEMA_VERSION:
        raise OutputRecipeError(
            f"outputCustomization.schemaVersion must be "
            f"{OUTPUT_RECIPE_SCHEMA_VERSION!r}"
        )

    report = _mapping(root["report"], "outputCustomization.report")
    _exact_keys(
        report,
        {
            "template",
            "font",
            "customFontFamily",
            "pageSize",
            "density",
            "accentColor",
        },
        "outputCustomization.report",
    )
    font = _enum(
        report["font"],
        "outputCustomization.report.font",
        _REPORT_FONTS,
    )
    custom_font = _bounded_text(
        report["customFontFamily"],
        "outputCustomization.report.customFontFamily",
        maximum=160,
        allow_empty=True,
    )
    if font == "custom" and not custom_font:
        raise OutputRecipeError(
            "outputCustomization.report.customFontFamily is required when "
            "font='custom'"
        )
    normalized_report = {
        "template": _enum(
            report["template"],
            "outputCustomization.report.template",
            _REPORT_TEMPLATES,
        ),
        "font": font,
        "customFontFamily": custom_font,
        "pageSize": _enum(
            report["pageSize"],
            "outputCustomization.report.pageSize",
            _PAGE_SIZES,
        ),
        "density": _enum(
            report["density"],
            "outputCustomization.report.density",
            _DENSITIES,
        ),
        "accentColor": _color(
            report["accentColor"],
            "outputCustomization.report.accentColor",
        ),
    }

    subtitles = _mapping(
        root["subtitles"], "outputCustomization.subtitles"
    )
    _exact_keys(
        subtitles,
        {
            "enabled",
            "theme",
            "size",
            "safeArea",
            "position",
            "speakerPalette",
            "backgroundOpacity",
            "maximumLines",
            "avoidVisualCollisions",
            "wordProgressHighlight",
        },
        "outputCustomization.subtitles",
    )
    subtitles_enabled = _boolean(
        subtitles["enabled"], "outputCustomization.subtitles.enabled"
    )
    word_progress = _boolean(
        subtitles["wordProgressHighlight"],
        "outputCustomization.subtitles.wordProgressHighlight",
    )
    if word_progress:
        raise OutputRecipeError(
            "wordProgressHighlight requires verified word-level timings and "
            "is unavailable for this pipeline"
        )
    normalized_subtitles = {
        "enabled": subtitles_enabled,
        "theme": _enum(
            subtitles["theme"],
            "outputCustomization.subtitles.theme",
            _SUBTITLE_THEMES,
        ),
        "size": _enum(
            subtitles["size"],
            "outputCustomization.subtitles.size",
            _SUBTITLE_SIZES,
        ),
        "safeArea": _enum(
            subtitles["safeArea"],
            "outputCustomization.subtitles.safeArea",
            _SAFE_AREAS,
        ),
        "position": _enum(
            subtitles["position"],
            "outputCustomization.subtitles.position",
            _POSITIONS,
        ),
        "speakerPalette": _enum(
            subtitles["speakerPalette"],
            "outputCustomization.subtitles.speakerPalette",
            _SPEAKER_PALETTES,
        ),
        "backgroundOpacity": _integer(
            subtitles["backgroundOpacity"],
            "outputCustomization.subtitles.backgroundOpacity",
            minimum=0,
            maximum=100,
        ),
        "maximumLines": _integer(
            subtitles["maximumLines"],
            "outputCustomization.subtitles.maximumLines",
            minimum=1,
            maximum=3,
        ),
        "avoidVisualCollisions": _boolean(
            subtitles["avoidVisualCollisions"],
            "outputCustomization.subtitles.avoidVisualCollisions",
        ),
        "wordProgressHighlight": False,
    }

    delivery = _mapping(root["delivery"], "outputCustomization.delivery")
    _exact_keys(
        delivery,
        {
            "formats",
            "subtitleModes",
            "includeMediaMetadata",
            "preserveSourceMedia",
            "fileNamePattern",
        },
        "outputCustomization.delivery",
    )
    formats = _unique_enum_list(
        delivery["formats"],
        "outputCustomization.delivery.formats",
        _EXPORT_FORMATS,
        minimum=1,
    )
    modes = _unique_enum_list(
        delivery["subtitleModes"],
        "outputCustomization.delivery.subtitleModes",
        _SUBTITLE_MODES,
        minimum=1 if subtitles_enabled else 0,
    )
    if not subtitles_enabled:
        if modes:
            raise OutputRecipeError(
                "subtitleModes must be empty while subtitles are disabled"
            )
        forbidden = [value for value in formats if value in _SUBTITLE_FORMATS]
        if forbidden:
            raise OutputRecipeError(
                "subtitle formats are forbidden while subtitles are disabled"
            )
    elif not modes:
        raise OutputRecipeError(
            "at least one subtitle delivery mode is required while subtitles "
            "are enabled"
        )
    if any(mode in {"soft-mux", "burn-in"} for mode in modes) and not subtitles_enabled:
        raise OutputRecipeError(
            "media subtitle delivery requires subtitles to be enabled"
        )
    if delivery["preserveSourceMedia"] is not True:
        raise OutputRecipeError(
            "outputCustomization.delivery.preserveSourceMedia must remain true"
        )
    file_name_pattern = _filename_template(delivery["fileNamePattern"])
    normalized_delivery = {
        "formats": list(formats),
        "subtitleModes": list(modes),
        "includeMediaMetadata": _boolean(
            delivery["includeMediaMetadata"],
            "outputCustomization.delivery.includeMediaMetadata",
        ),
        "preserveSourceMedia": True,
        "fileNamePattern": file_name_pattern,
    }

    finishing = _mapping(
        root["finishing"], "outputCustomization.finishing"
    )
    _exact_keys(
        finishing,
        {
            "includeCover",
            "includeChapters",
            "includeTimestamps",
            "includeHeader",
            "includeFooter",
            "includeSpeakerIndex",
            "includeConfidenceNotes",
            "chapterStyle",
            "timestampStyle",
            "customTitle",
        },
        "outputCustomization.finishing",
    )
    include_chapters = _boolean(
        finishing["includeChapters"],
        "outputCustomization.finishing.includeChapters",
    )
    chapter_style = _enum(
        finishing["chapterStyle"],
        "outputCustomization.finishing.chapterStyle",
        _CHAPTER_STYLES,
    )
    if include_chapters and chapter_style == "none":
        raise OutputRecipeError(
            "chapterStyle cannot be 'none' while includeChapters is true"
        )
    if not include_chapters and chapter_style != "none":
        raise OutputRecipeError(
            "chapterStyle must be 'none' while includeChapters is false"
        )
    normalized_finishing = {
        "includeCover": _boolean(
            finishing["includeCover"],
            "outputCustomization.finishing.includeCover",
        ),
        "includeChapters": include_chapters,
        "includeTimestamps": _boolean(
            finishing["includeTimestamps"],
            "outputCustomization.finishing.includeTimestamps",
        ),
        "includeHeader": _boolean(
            finishing["includeHeader"],
            "outputCustomization.finishing.includeHeader",
        ),
        "includeFooter": _boolean(
            finishing["includeFooter"],
            "outputCustomization.finishing.includeFooter",
        ),
        "includeSpeakerIndex": _boolean(
            finishing["includeSpeakerIndex"],
            "outputCustomization.finishing.includeSpeakerIndex",
        ),
        "includeConfidenceNotes": _boolean(
            finishing["includeConfidenceNotes"],
            "outputCustomization.finishing.includeConfidenceNotes",
        ),
        "chapterStyle": chapter_style,
        "timestampStyle": _enum(
            finishing["timestampStyle"],
            "outputCustomization.finishing.timestampStyle",
            _TIMESTAMP_STYLES,
        ),
        "customTitle": _bounded_text(
            finishing["customTitle"],
            "outputCustomization.finishing.customTitle",
            maximum=160,
            allow_empty=True,
        ),
    }

    normalized = {
        "schemaVersion": OUTPUT_RECIPE_SCHEMA_VERSION,
        "report": normalized_report,
        "subtitles": normalized_subtitles,
        "delivery": normalized_delivery,
        "finishing": normalized_finishing,
    }
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return OutputRecipe(canonical)


def compile_output_customizations(
    recipe: OutputRecipe,
    *,
    source_path: Path,
    output_directory: Path,
    media_probe: Any,
    media_probe_artifact: Any,
) -> tuple[CompiledOutputCustomization, ...]:
    """Compile every requested delivery mode into a rich canonical snapshot."""

    if not isinstance(recipe, OutputRecipe):
        raise OutputRecipeError("recipe must be an OutputRecipe")
    source = source_path.resolve(strict=True)
    output_root = output_directory.resolve(strict=True)
    payload = recipe.canonical_dict()
    modes = list(payload["delivery"]["subtitleModes"])
    if not payload["subtitles"]["enabled"]:
        # A report/transcript-only recipe still needs one plan so the Java
        # renderer receives exact presentation settings.
        modes = ["sidecar"]

    compiled: list[CompiledOutputCustomization] = []
    for mode in modes:
        overrides = _canonical_overrides(
            payload,
            mode=mode,
            source_path=source,
            output_directory=output_root,
            media_probe=media_probe,
            media_probe_artifact=media_probe_artifact,
        )
        try:
            customization = resolve_output_customization(overrides)
        except OutputCustomizationError as exc:
            raise OutputRecipeError(
                f"output recipe could not compile delivery mode {mode!r}: {exc}"
            ) from exc
        compiled.append(
            CompiledOutputCustomization(
                delivery_mode=mode,
                customization=customization,
            )
        )
    return tuple(compiled)


def render_recipe_file_name(
    recipe: OutputRecipe,
    *,
    source_stem: str,
    artifact: str,
    language: str,
    generated_date: str,
    speaker_count: int,
) -> str:
    """Render a bounded file stem from the recipe's safe token template."""

    if not isinstance(recipe, OutputRecipe):
        raise OutputRecipeError("recipe must be an OutputRecipe")
    template = recipe.canonical_dict()["delivery"]["fileNamePattern"]
    values = {
        "sourceStem": source_stem,
        "artifact": artifact,
        "language": language,
        "date": generated_date,
        "speakerCount": str(speaker_count),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    rendered = unicodedata.normalize("NFC", rendered)
    rendered = _WINDOWS_INVALID.sub("-", rendered)
    rendered = re.sub(r"\s+", " ", rendered).strip(" .")
    if not rendered or rendered in {".", ".."}:
        raise OutputRecipeError("fileNamePattern produced an empty file name")
    # Leave ample headroom for suffixes and Windows output paths.
    return rendered[:180].rstrip(" .")


def _canonical_overrides(
    recipe: Mapping[str, Any],
    *,
    mode: str,
    source_path: Path,
    output_directory: Path,
    media_probe: Any,
    media_probe_artifact: Any,
) -> dict[str, Any]:
    report = recipe["report"]
    subtitles = recipe["subtitles"]
    delivery = recipe["delivery"]
    finishing = recipe["finishing"]
    formats = tuple(delivery["formats"])
    subtitle_formats = [value for value in formats if value in _SUBTITLE_FORMATS]

    # ASS is the internal high-fidelity carrier for styled media delivery.
    # It is added only when required for soft-mux/burn-in execution.
    if subtitles["enabled"] and mode in {"soft-mux", "burn-in"}:
        ordered_subtitle_formats = [
            "ass",
            *[value for value in subtitle_formats if value != "ass"],
        ]
    else:
        ordered_subtitle_formats = list(subtitle_formats)
    if subtitles["enabled"] and not ordered_subtitle_formats:
        ordered_subtitle_formats = ["ass"]

    primary_subtitle = (
        ordered_subtitle_formats[0] if ordered_subtitle_formats else "ass"
    )
    subtitle_alternates = ordered_subtitle_formats[1:]
    report_enabled = "pdf" in formats
    transcript_formats = [
        value
        for value in ("json", "txt", "markdown", "html")
        if value in formats
    ]
    if not transcript_formats:
        # The immutable transcript JSON is always retained as provenance even
        # when the user asks only for PDF or subtitles.
        transcript_formats = ["json"]
    page_size, orientation = _paper(report["pageSize"])
    chapters_enabled = bool(finishing["includeChapters"])
    chapter_source = {
        "semantic": "semantic-sections",
        "interval": "fixed-interval",
        "none": "semantic-sections",
    }[finishing["chapterStyle"]]
    timestamp_placement = {
        "segment": "gutter",
        "paragraph": "inline",
        "chapter": "block",
    }[finishing["timestampStyle"]]

    target: dict[str, Any]
    if mode == "sidecar":
        target = {
            "binding": "deferred",
            "sourcePath": None,
            "outputPath": None,
        }
    else:
        source_stem = source_path.stem
        artifact = "subtitled-soft-mux" if mode == "soft-mux" else "subtitled"
        output_stem = _render_file_name_payload(
            recipe,
            source_stem=source_stem,
            artifact=artifact,
            language="und",
            generated_date="output",
            speaker_count="dynamic",
        )
        suffix = ".mkv" if mode == "soft-mux" else ".mp4"
        output_path = output_directory / f"{output_stem}{suffix}"
        target = {
            "binding": "bound",
            "sourcePath": str(source_path),
            "outputPath": str(output_path),
        }

    burn_in: dict[str, Any] = {
        "strategy": None,
        "sourceDynamicRange": "unknown",
        "dynamicRangeEvidence": None,
        "requireVisualQa": True,
    }
    if mode == "burn-in":
        if not getattr(media_probe, "video_stream_indexes", ()):
            raise OutputRecipeError("burn-in is unavailable for audio-only media")
        if bool(getattr(media_probe, "has_hdr_video", True)):
            raise OutputRecipeError(
                "HDR video burn-in is rejected; choose sidecar or soft-mux"
            )
        probe_sha256 = getattr(media_probe_artifact, "sha256", None)
        if not isinstance(probe_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", probe_sha256
        ):
            raise OutputRecipeError(
                "burn-in requires SHA-256-bound media probe evidence"
            )
        burn_in = {
            "strategy": "h264-high-quality",
            "sourceDynamicRange": "sdr",
            "dynamicRangeEvidence": {
                "verified": True,
                "method": "container-and-frame-probe",
                "probeSha256": probe_sha256,
            },
            "requireVisualQa": True,
        }

    return {
        "profile": "custom",
        "report": {
            "enabled": report_enabled,
            "layout": {
                "soft-glass": "modern-editorial",
                "editorial": "conversation-focus",
                "academic": "executive-brief",
                "compact": "compact-review",
            }[report["template"]],
            "density": {
                "airy": "relaxed",
                "balanced": "comfortable",
                "compact": "compact",
            }[report["density"]],
            "paper": {
                "size": page_size,
                "orientation": orientation,
            },
            "fontPack": _font_pack(
                report["font"], report["customFontFamily"]
            ),
            "cover": {
                "enabled": bool(finishing["includeCover"]),
                "style": (
                    "none"
                    if not finishing["includeCover"]
                    else {
                        "soft-glass": "immersive",
                        "editorial": "editorial",
                        "academic": "minimal",
                        "compact": "minimal",
                    }[report["template"]]
                ),
                "showTitle": True,
                "showSubtitle": True,
                "showSourceMetadata": bool(delivery["includeMediaMetadata"]),
            },
            "header": {
                "enabled": bool(finishing["includeHeader"]),
                "showDocumentTitle": True,
                "showChapter": chapters_enabled,
                "showPageNumber": False,
            },
            "footer": {
                "enabled": bool(finishing["includeFooter"]),
                "showDocumentTitle": False,
                "showChapter": False,
                "showPageNumber": True,
            },
            "chapters": {
                "enabled": chapters_enabled,
                "source": chapter_source,
                "intervalMinutes": (
                    5 if finishing["chapterStyle"] == "interval" else None
                ),
                "includeTimestamps": bool(finishing["includeTimestamps"]),
                "pageBreakBefore": False,
            },
            "speakerLegend": {
                "enabled": bool(finishing["includeSpeakerIndex"]),
                "position": "before-transcript",
                "showColor": True,
                "showDisplayName": True,
                "showSpeakingTime": bool(
                    finishing["includeConfidenceNotes"]
                ),
            },
            "timestamp": {
                "format": "hh:mm:ss.mmm",
                "placement": timestamp_placement,
                "showEnd": False,
                "frameRate": None,
            },
            "accentColor": report["accentColor"].upper(),
            "highContrast": report["template"] == "academic",
        },
        "subtitle": {
            "enabled": bool(subtitles["enabled"]),
            "format": primary_subtitle,
            "theme": {
                "youtube-clean": "youtube-clean",
                "soft-bubble": "minimal-glass",
                "cinema": "documentary",
                "high-contrast": "youtube-bold",
            }[subtitles["theme"]],
            "fontPack": _font_pack(
                report["font"], report["customFontFamily"]
            ),
            "fontSizePx": {
                "small": 42,
                "medium": 52,
                "large": 62,
                "x-large": 72,
            }[subtitles["size"]],
            "fontWeight": (
                800 if subtitles["theme"] == "high-contrast" else 700
            ),
            "alignment": {
                "bottom": "bottom-center",
                "smart": "bottom-center",
                "top": "top-center",
            }[subtitles["position"]],
            "safeArea": _safe_area(subtitles["safeArea"]),
            "background": {
                "opacity": subtitles["backgroundOpacity"] / 100.0,
            },
            "cuePolicy": {
                "maxLines": subtitles["maximumLines"],
                "maxCharactersPerLine": (
                    38 if subtitles["avoidVisualCollisions"] else 44
                ),
            },
            "speakerColors": _speaker_colors(
                subtitles["speakerPalette"]
            ),
            "karaoke": {"mode": "off", "evidence": None},
        },
        "delivery": {
            "mode": mode,
            "outputTarget": target,
            "preserveMetadata": bool(delivery["includeMediaMetadata"]),
            "preserveChapters": True,
            "preserveAudio": True,
            "softMux": {
                "container": (
                    "matroska" if mode == "soft-mux" else "source-compatible"
                ),
                "subtitleCodec": (
                    "ass"
                    if mode == "soft-mux"
                    else "probe-selected"
                ),
            },
            "burnIn": burn_in,
        },
        "exports": {
            "reportFormats": ["pdf"] if report_enabled else [],
            "transcriptFormats": transcript_formats,
            "subtitleAlternates": subtitle_alternates,
            "dataFormats": ["json"],
            "fileNameTemplate": delivery["fileNamePattern"],
            "packageFormat": "directory",
            "checksumManifest": True,
            "provenanceManifest": True,
        },
    }


def _font_pack(font: str, custom_family: str) -> dict[str, Any]:
    primary = {
        "system-sans": "Noto Sans CJK SC",
        "humanist": "Segoe UI",
        "serif": "Noto Serif CJK SC",
        "mono-accent": "JetBrains Mono",
        "custom": custom_family,
    }[font]
    return {
        "id": f"ui-{font}",
        "primary": primary,
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


def _paper(value: str) -> tuple[str, str]:
    return {
        "a4": ("A4", "portrait"),
        "letter": ("Letter", "portrait"),
        "legal": ("Legal", "portrait"),
        # The canonical renderer currently models physical sheets.  Screen
        # mode uses a wide Letter canvas instead of silently ignoring the UI.
        "screen": ("Letter", "landscape"),
    }[value]


def _safe_area(value: str) -> dict[str, float]:
    horizontal, top, bottom = {
        "standard": (4.0, 4.0, 6.0),
        "broadcast": (5.0, 5.0, 8.0),
        "generous": (8.0, 8.0, 12.0),
    }[value]
    return {
        "horizontalPercent": horizontal,
        "topPercent": top,
        "bottomPercent": bottom,
    }


def _speaker_colors(value: str) -> dict[str, Any]:
    if value == "monochrome":
        mode = "monochrome"
        algorithm = "monochrome-v1"
    elif value == "high-contrast":
        mode = "accessible"
        algorithm = "accessible-oklch-hash-v1"
    else:
        mode = "automatic"
        algorithm = "oklch-hash-v1"
    return {
        "mode": mode,
        "algorithm": algorithm,
        "seed": f"MediaTranscribeStudio:{value}",
        "minimumDeltaE": 22.0 if value == "high-contrast" else 18.0,
        "collisionFallback": "label-and-pattern",
        "overrides": [],
    }


def _render_file_name_payload(
    recipe: Mapping[str, Any],
    *,
    source_stem: str,
    artifact: str,
    language: str,
    generated_date: str,
    speaker_count: str,
) -> str:
    rendered = recipe["delivery"]["fileNamePattern"]
    values = {
        "sourceStem": source_stem,
        "artifact": artifact,
        "language": language,
        "date": generated_date,
        "speakerCount": speaker_count,
    }
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    rendered = unicodedata.normalize("NFC", rendered)
    rendered = _WINDOWS_INVALID.sub("-", rendered)
    rendered = re.sub(r"\s+", " ", rendered).strip(" .")
    if not rendered:
        raise OutputRecipeError("fileNamePattern produced an empty file name")
    return rendered[:180].rstrip(" .")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OutputRecipeError(f"{field} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise OutputRecipeError(f"{field} has invalid fields: {'; '.join(details)}")


def _enum(value: Any, field: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise OutputRecipeError(
            f"{field} must be one of {', '.join(repr(item) for item in choices)}"
        )
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise OutputRecipeError(f"{field} must be a boolean")
    return value


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutputRecipeError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise OutputRecipeError(
            f"{field} must be between {minimum} and {maximum}"
        )
    return value


def _bounded_text(
    value: Any,
    field: str,
    *,
    maximum: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        raise OutputRecipeError(f"{field} must be a string")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not allow_empty and not normalized:
        raise OutputRecipeError(f"{field} must not be empty")
    if len(normalized) > maximum:
        raise OutputRecipeError(f"{field} exceeds {maximum} characters")
    if _CONTROL.search(normalized) or _BIDI_CONTROL.search(normalized):
        raise OutputRecipeError(f"{field} contains unsafe control characters")
    return normalized


def _color(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_COLOR.fullmatch(value):
        raise OutputRecipeError(f"{field} must be a six-digit hexadecimal color")
    return value.lower()


def _unique_enum_list(
    value: Any,
    field: str,
    choices: Sequence[str],
    *,
    minimum: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OutputRecipeError(f"{field} must be an array")
    if len(value) < minimum or len(value) > len(choices):
        raise OutputRecipeError(
            f"{field} must contain between {minimum} and {len(choices)} entries"
        )
    result: list[str] = []
    for index, item in enumerate(value):
        selected = _enum(item, f"{field}[{index}]", choices)
        if selected in result:
            raise OutputRecipeError(f"{field}[{index}] duplicates {selected!r}")
        result.append(selected)
    return tuple(result)


def _filename_template(value: Any) -> str:
    template = _bounded_text(
        value,
        "outputCustomization.delivery.fileNamePattern",
        maximum=160,
        allow_empty=False,
    )
    if _PATH_SEPARATOR.search(template):
        raise OutputRecipeError("fileNamePattern cannot contain path separators")
    tokens = _TEMPLATE_TOKEN.findall(template)
    unknown = sorted(set(tokens) - _FILENAME_TOKENS)
    if unknown:
        raise OutputRecipeError(
            "fileNamePattern has unknown tokens: " + ", ".join(unknown)
        )
    if not tokens:
        raise OutputRecipeError(
            "fileNamePattern must contain at least one supported token"
        )
    stripped = _TEMPLATE_TOKEN.sub("", template)
    if "{" in stripped or "}" in stripped:
        raise OutputRecipeError("fileNamePattern contains malformed token braces")
    if os.path.basename(template) != template:
        raise OutputRecipeError("fileNamePattern must be a file name, not a path")
    return template


__all__ = [
    "CompiledOutputCustomization",
    "OUTPUT_RECIPE_SCHEMA_VERSION",
    "OutputRecipe",
    "OutputRecipeError",
    "compile_output_customizations",
    "parse_output_recipe",
    "render_recipe_file_name",
]
