"""Pure, offline report-appearance configuration.

The domain model in this module is intentionally presentation-only.  It does
not accept transcript text, write files, resolve fonts, fetch assets, render
HTML, or mutate source artifacts.  A resolved configuration is immutable and
can be serialized canonically for caching, provenance, and renderer hand-off.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any


REPORT_STYLE_SCHEMA_VERSION = "1.1.0"
FONT_AVAILABILITY_DECLARATION = "declared-not-verified"
FONT_EMBEDDING_DECLARATION = "not-embedded"

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SAFE_LOGO_PATH = re.compile(
    r"^(?!\s)"
    r"(?![\\/])"
    r"(?!.*[\\/]{2})"
    r"(?!\.{1,2}(?:[\\/]|$))"
    r"(?!.*[\\/]\.{1,2}(?:[\\/]|$))"
    r"(?!.*[. ][\\/])"
    r"(?!.*[. ]$)"
    r"[^\x00-\x1F\x7F:%?#]{1,512}"
    r"\.(?:[Pp][Nn][Gg]|[Jj][Pp][Ee]?[Gg]|[Ww][Ee][Bb][Pp])$"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1F\x7F]")


class ReportStyleError(ValueError):
    """Raised when a report-appearance value violates the public contract."""


class ReportStylePreset(str, Enum):
    """Stable identifiers for the shipped presets and the editable baseline."""

    MODERN_EDITORIAL = "modern-editorial"
    CONVERSATION_FOCUS = "conversation-focus"
    EXECUTIVE_BRIEF = "executive-brief"
    COMPACT_REVIEW = "compact-review"
    ACCESSIBLE_HIGH_CONTRAST = "accessible-high-contrast"
    ARCHIVE_MONOCHROME = "archive-monochrome"
    CUSTOM = "custom"


class LayoutTemplate(str, Enum):
    MODERN_EDITORIAL = "modern-editorial"
    CONVERSATION_FOCUS = "conversation-focus"
    EXECUTIVE_BRIEF = "executive-brief"
    COMPACT_REVIEW = "compact-review"
    ACCESSIBLE_LARGE_PRINT = "accessible-large-print"
    ARCHIVE_MONOCHROME = "archive-monochrome"


class Density(str, Enum):
    RELAXED = "relaxed"
    COMFORTABLE = "comfortable"
    COMPACT = "compact"


class CoverStyle(str, Enum):
    IMMERSIVE = "immersive"
    EDITORIAL = "editorial"
    MINIMAL = "minimal"
    NONE = "none"


class SpeakerColorMode(str, Enum):
    DISTINCT = "distinct"
    ACCENT = "accent"
    MONOCHROME = "monochrome"
    ACCESSIBLE = "accessible"


class TimestampDetail(str, Enum):
    NONE = "none"
    SECTION = "section"
    SEGMENT = "segment"
    MILLISECONDS = "milliseconds"


class PageSize(str, Enum):
    A4 = "A4"
    A5 = "A5"
    LETTER = "Letter"
    LEGAL = "Legal"


class PageOrientation(str, Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


@dataclass(frozen=True)
class FontFallbacks:
    """Ordered fallback declarations by writing-system family."""

    latin: tuple[str, ...]
    cjk: tuple[str, ...]
    rtl: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "latin": list(self.latin),
            "cjk": list(self.cjk),
            "rtl": list(self.rtl),
        }


@dataclass(frozen=True)
class FontPolicy:
    """Font intent, without claiming installation, resolution, or embedding."""

    family: str
    fallbacks: FontFallbacks

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "fallbacks": self.fallbacks.as_dict(),
            "availability": FONT_AVAILABILITY_DECLARATION,
            "embedding": FONT_EMBEDDING_DECLARATION,
        }


@dataclass(frozen=True)
class SectionInclusion:
    cover: bool
    table_of_contents: bool
    metadata: bool
    speaker_directory: bool
    transcript: bool
    translation: bool
    summary: bool
    quality_appendix: bool
    provenance: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "cover": self.cover,
            "tableOfContents": self.table_of_contents,
            "metadata": self.metadata,
            "speakerDirectory": self.speaker_directory,
            "transcript": self.transcript,
            "translation": self.translation,
            "summary": self.summary,
            "qualityAppendix": self.quality_appendix,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class PagePolicy:
    size: PageSize
    orientation: PageOrientation

    def as_dict(self) -> dict[str, str]:
        return {"size": self.size.value, "orientation": self.orientation.value}


@dataclass(frozen=True)
class MarginPolicy:
    top_mm: float
    right_mm: float
    bottom_mm: float
    left_mm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "topMm": self.top_mm,
            "rightMm": self.right_mm,
            "bottomMm": self.bottom_mm,
            "leftMm": self.left_mm,
        }


@dataclass(frozen=True)
class RunningText:
    enabled: bool
    text: str
    show_document_title: bool
    show_page_number: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "text": self.text,
            "showDocumentTitle": self.show_document_title,
            "showPageNumber": self.show_page_number,
        }


@dataclass(frozen=True)
class LogoPolicy:
    enabled: bool
    path: str | None
    alt_text: str
    max_width_mm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": self.path,
            "altText": self.alt_text,
            "maxWidthMm": self.max_width_mm,
        }


@dataclass(frozen=True)
class ReportStyleConfig:
    """A fully resolved, immutable, versioned report style."""

    preset: ReportStylePreset
    layout_template: LayoutTemplate
    font: FontPolicy
    density: Density
    cover_style: CoverStyle
    speaker_color_mode: SpeakerColorMode
    timestamp_detail: TimestampDetail
    section_inclusion: SectionInclusion
    page: PagePolicy
    margins: MarginPolicy
    header: RunningText
    footer: RunningText
    logo: LogoPolicy
    brand_accent: str
    high_contrast: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReportStyleConfig:
        """Parse a complete contract payload with strict key and type checks."""

        payload = _mapping(value, "report style")
        _exact_keys(
            payload,
            {
                "schemaVersion",
                "preset",
                "layoutTemplate",
                "font",
                "density",
                "coverStyle",
                "speakerColorMode",
                "timestampDetail",
                "sectionInclusion",
                "page",
                "margins",
                "header",
                "footer",
                "logo",
                "brandAccent",
                "highContrast",
                "sourceProtection",
            },
            "report style",
        )
        if payload["schemaVersion"] != REPORT_STYLE_SCHEMA_VERSION:
            raise ReportStyleError(
                "schemaVersion must be "
                f"{REPORT_STYLE_SCHEMA_VERSION!r}"
            )
        _validate_source_protection(payload["sourceProtection"])

        return cls(
            preset=_enum_value(
                ReportStylePreset, payload["preset"], "preset"
            ),
            layout_template=_enum_value(
                LayoutTemplate,
                payload["layoutTemplate"],
                "layoutTemplate",
            ),
            font=_parse_font(payload["font"]),
            density=_enum_value(Density, payload["density"], "density"),
            cover_style=_enum_value(
                CoverStyle, payload["coverStyle"], "coverStyle"
            ),
            speaker_color_mode=_enum_value(
                SpeakerColorMode,
                payload["speakerColorMode"],
                "speakerColorMode",
            ),
            timestamp_detail=_enum_value(
                TimestampDetail,
                payload["timestampDetail"],
                "timestampDetail",
            ),
            section_inclusion=_parse_sections(payload["sectionInclusion"]),
            page=_parse_page(payload["page"]),
            margins=_parse_margins(payload["margins"]),
            header=_parse_running_text(payload["header"], "header"),
            footer=_parse_running_text(payload["footer"], "footer"),
            logo=_parse_logo(payload["logo"]),
            brand_accent=_color(payload["brandAccent"], "brandAccent"),
            high_contrast=_boolean(payload["highContrast"], "highContrast"),
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Return the fully materialized JSON representation."""

        return {
            "schemaVersion": REPORT_STYLE_SCHEMA_VERSION,
            "preset": self.preset.value,
            "layoutTemplate": self.layout_template.value,
            "font": self.font.as_dict(),
            "density": self.density.value,
            "coverStyle": self.cover_style.value,
            "speakerColorMode": self.speaker_color_mode.value,
            "timestampDetail": self.timestamp_detail.value,
            "sectionInclusion": self.section_inclusion.as_dict(),
            "page": self.page.as_dict(),
            "margins": self.margins.as_dict(),
            "header": self.header.as_dict(),
            "footer": self.footer.as_dict(),
            "logo": self.logo.as_dict(),
            "brandAccent": self.brand_accent,
            "highContrast": self.high_contrast,
            "sourceProtection": {
                "preserveOriginalTranscript": True,
                "mutatesOriginalTranscript": False,
                "presentationOnly": True,
            },
        }

    def canonical_json(self) -> str:
        """Return stable UTF-8 JSON suitable for hashing and provenance."""

        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def deterministic_hash(self) -> str:
        """Return a SHA-256 digest of the canonical JSON representation."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportStyleError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ReportStyleError(f"{name} keys must be strings")
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
        raise ReportStyleError(f"{name} is missing: {', '.join(missing)}")
    if unknown:
        raise ReportStyleError(
            f"{name} has unknown properties: {', '.join(unknown)}"
        )


def _enum_value(
    enum_type: type[Enum],
    value: Any,
    name: str,
) -> Any:
    if not isinstance(value, str):
        raise ReportStyleError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ReportStyleError(f"{name} must be one of: {allowed}") from exc


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ReportStyleError(f"{name} must be a boolean")
    return value


def _number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportStyleError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ReportStyleError(f"{name} must be finite")
    if not minimum <= normalized <= maximum:
        raise ReportStyleError(
            f"{name} must be between {minimum:g} and {maximum:g}"
        )
    return normalized


def _text(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
    allow_surrounding_whitespace: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ReportStyleError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if _CONTROL_CHARACTERS.search(normalized):
        raise ReportStyleError(f"{name} cannot contain control characters")
    if not allow_surrounding_whitespace and normalized != normalized.strip():
        raise ReportStyleError(
            f"{name} cannot start or end with whitespace"
        )
    if not minimum <= len(normalized) <= maximum:
        raise ReportStyleError(
            f"{name} length must be between {minimum} and {maximum}"
        )
    if minimum and not normalized.strip():
        raise ReportStyleError(f"{name} must contain a visible character")
    return normalized


def _string_sequence(
    value: Any,
    name: str,
    *,
    minimum_items: int = 1,
    maximum_items: int = 12,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ReportStyleError(f"{name} must be an array")
    if not minimum_items <= len(value) <= maximum_items:
        raise ReportStyleError(
            f"{name} must contain between "
            f"{minimum_items} and {maximum_items} entries"
        )
    normalized = tuple(
        _text(item, f"{name}[{index}]", minimum=1, maximum=160)
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise ReportStyleError(f"{name} cannot contain duplicate entries")
    return normalized


def _parse_font(value: Any) -> FontPolicy:
    payload = _mapping(value, "font")
    _exact_keys(
        payload,
        {"family", "fallbacks", "availability", "embedding"},
        "font",
    )
    if payload["availability"] != FONT_AVAILABILITY_DECLARATION:
        raise ReportStyleError(
            "font.availability must declare fonts as not verified"
        )
    if payload["embedding"] != FONT_EMBEDDING_DECLARATION:
        raise ReportStyleError(
            "font.embedding cannot claim that fonts are embedded"
        )
    fallback_payload = _mapping(payload["fallbacks"], "font.fallbacks")
    _exact_keys(
        fallback_payload,
        {"latin", "cjk", "rtl"},
        "font.fallbacks",
    )
    return FontPolicy(
        family=_text(
            payload["family"], "font.family", minimum=1, maximum=160
        ),
        fallbacks=FontFallbacks(
            latin=_string_sequence(
                fallback_payload["latin"], "font.fallbacks.latin"
            ),
            cjk=_string_sequence(
                fallback_payload["cjk"], "font.fallbacks.cjk"
            ),
            rtl=_string_sequence(
                fallback_payload["rtl"], "font.fallbacks.rtl"
            ),
        ),
    )


def _parse_sections(value: Any) -> SectionInclusion:
    payload = _mapping(value, "sectionInclusion")
    expected = {
        "cover",
        "tableOfContents",
        "metadata",
        "speakerDirectory",
        "transcript",
        "translation",
        "summary",
        "qualityAppendix",
        "provenance",
    }
    _exact_keys(payload, expected, "sectionInclusion")
    normalized = {
        key: _boolean(payload[key], f"sectionInclusion.{key}")
        for key in expected
    }
    content_keys = (
        "transcript",
        "translation",
        "summary",
        "qualityAppendix",
        "provenance",
    )
    if not any(normalized[key] for key in content_keys):
        raise ReportStyleError(
            "sectionInclusion must enable at least one content section"
        )
    return SectionInclusion(
        cover=normalized["cover"],
        table_of_contents=normalized["tableOfContents"],
        metadata=normalized["metadata"],
        speaker_directory=normalized["speakerDirectory"],
        transcript=normalized["transcript"],
        translation=normalized["translation"],
        summary=normalized["summary"],
        quality_appendix=normalized["qualityAppendix"],
        provenance=normalized["provenance"],
    )


def _parse_page(value: Any) -> PagePolicy:
    payload = _mapping(value, "page")
    _exact_keys(payload, {"size", "orientation"}, "page")
    return PagePolicy(
        size=_enum_value(PageSize, payload["size"], "page.size"),
        orientation=_enum_value(
            PageOrientation, payload["orientation"], "page.orientation"
        ),
    )


def _parse_margins(value: Any) -> MarginPolicy:
    payload = _mapping(value, "margins")
    expected = {"topMm", "rightMm", "bottomMm", "leftMm"}
    _exact_keys(payload, expected, "margins")
    return MarginPolicy(
        top_mm=_number(
            payload["topMm"], "margins.topMm", minimum=0, maximum=50
        ),
        right_mm=_number(
            payload["rightMm"], "margins.rightMm", minimum=0, maximum=50
        ),
        bottom_mm=_number(
            payload["bottomMm"],
            "margins.bottomMm",
            minimum=0,
            maximum=50,
        ),
        left_mm=_number(
            payload["leftMm"], "margins.leftMm", minimum=0, maximum=50
        ),
    )


def _parse_running_text(value: Any, name: str) -> RunningText:
    payload = _mapping(value, name)
    _exact_keys(
        payload,
        {"enabled", "text", "showDocumentTitle", "showPageNumber"},
        name,
    )
    return RunningText(
        enabled=_boolean(payload["enabled"], f"{name}.enabled"),
        text=_text(
            payload["text"],
            f"{name}.text",
            minimum=0,
            maximum=200,
            allow_surrounding_whitespace=True,
        ),
        show_document_title=_boolean(
            payload["showDocumentTitle"], f"{name}.showDocumentTitle"
        ),
        show_page_number=_boolean(
            payload["showPageNumber"], f"{name}.showPageNumber"
        ),
    )


def _safe_logo_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ReportStyleError("logo.path must be a string or null")
    normalized = unicodedata.normalize("NFC", value)
    if not _SAFE_LOGO_PATH.fullmatch(normalized):
        raise ReportStyleError(
            "logo.path must be a safe local relative PNG, JPEG, or WebP path"
        )
    return normalized.replace("\\", "/")


def _parse_logo(value: Any) -> LogoPolicy:
    payload = _mapping(value, "logo")
    _exact_keys(
        payload,
        {"enabled", "path", "altText", "maxWidthMm"},
        "logo",
    )
    enabled = _boolean(payload["enabled"], "logo.enabled")
    raw_path = payload["path"]
    path = None if raw_path is None else _safe_logo_path(raw_path)
    alt_text = _text(
        payload["altText"],
        "logo.altText",
        minimum=0,
        maximum=200,
        allow_surrounding_whitespace=True,
    )
    if enabled and path is None:
        raise ReportStyleError("enabled logo requires logo.path")
    if enabled and not alt_text.strip():
        raise ReportStyleError("enabled logo requires non-empty logo.altText")
    return LogoPolicy(
        enabled=enabled,
        path=path,
        alt_text=alt_text,
        max_width_mm=_number(
            payload["maxWidthMm"],
            "logo.maxWidthMm",
            minimum=5,
            maximum=80,
        ),
    )


def _color(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX_COLOR.fullmatch(value):
        raise ReportStyleError(f"{name} must be a #RRGGBB color")
    return value.upper()


def _validate_source_protection(value: Any) -> None:
    payload = _mapping(value, "sourceProtection")
    _exact_keys(
        payload,
        {
            "preserveOriginalTranscript",
            "mutatesOriginalTranscript",
            "presentationOnly",
        },
        "sourceProtection",
    )
    expected = {
        "preserveOriginalTranscript": True,
        "mutatesOriginalTranscript": False,
        "presentationOnly": True,
    }
    if dict(payload) != expected:
        raise ReportStyleError(
            "sourceProtection must preserve the original transcript and "
            "declare presentation-only behavior"
        )


def _deep_merge(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _base_payload(preset: str) -> dict[str, Any]:
    return {
        "schemaVersion": REPORT_STYLE_SCHEMA_VERSION,
        "preset": preset,
        "layoutTemplate": "modern-editorial",
        "font": {
            "family": "Noto Sans CJK SC",
            "fallbacks": {
                "latin": ["Inter", "Segoe UI", "Arial"],
                "cjk": [
                    "Noto Sans CJK SC",
                    "Source Han Sans SC",
                    "Yu Gothic UI",
                    "Malgun Gothic",
                ],
                "rtl": [
                    "Noto Sans Arabic",
                    "Noto Sans Hebrew",
                    "Segoe UI",
                ],
            },
            "availability": FONT_AVAILABILITY_DECLARATION,
            "embedding": FONT_EMBEDDING_DECLARATION,
        },
        "density": "comfortable",
        "coverStyle": "editorial",
        "speakerColorMode": "distinct",
        "timestampDetail": "segment",
        "sectionInclusion": {
            "cover": True,
            "tableOfContents": True,
            "metadata": True,
            "speakerDirectory": True,
            "transcript": True,
            "translation": True,
            "summary": True,
            "qualityAppendix": True,
            "provenance": True,
        },
        "page": {"size": "A4", "orientation": "portrait"},
        "margins": {
            "topMm": 18.0,
            "rightMm": 17.0,
            "bottomMm": 19.0,
            "leftMm": 17.0,
        },
        "header": {
            "enabled": True,
            "text": "",
            "showDocumentTitle": True,
            "showPageNumber": False,
        },
        "footer": {
            "enabled": True,
            "text": "",
            "showDocumentTitle": False,
            "showPageNumber": True,
        },
        "logo": {
            "enabled": False,
            "path": None,
            "altText": "",
            "maxWidthMm": 28.0,
        },
        "brandAccent": "#6D5DFB",
        "highContrast": False,
        "sourceProtection": {
            "preserveOriginalTranscript": True,
            "mutatesOriginalTranscript": False,
            "presentationOnly": True,
        },
    }


def _with_changes(
    preset: ReportStylePreset,
    changes: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _base_payload(preset.value)
    return _deep_merge(payload, changes)


_RAW_PRESETS: dict[ReportStylePreset, dict[str, Any]] = {
    ReportStylePreset.MODERN_EDITORIAL: _base_payload(
        ReportStylePreset.MODERN_EDITORIAL.value
    ),
    ReportStylePreset.CONVERSATION_FOCUS: _with_changes(
        ReportStylePreset.CONVERSATION_FOCUS,
        {
            "layoutTemplate": "conversation-focus",
            "coverStyle": "immersive",
            "speakerColorMode": "distinct",
            "timestampDetail": "segment",
            "brandAccent": "#2E90FA",
            "sectionInclusion": {
                "translation": False,
                "summary": False,
            },
        },
    ),
    ReportStylePreset.EXECUTIVE_BRIEF: _with_changes(
        ReportStylePreset.EXECUTIVE_BRIEF,
        {
            "layoutTemplate": "executive-brief",
            "font": {
                "family": "Source Serif 4",
                "fallbacks": {
                    "latin": ["Source Serif 4", "Georgia", "Times New Roman"],
                    "cjk": [
                        "Noto Serif CJK SC",
                        "Source Han Serif SC",
                        "Yu Mincho",
                        "Batang",
                    ],
                    "rtl": [
                        "Noto Naskh Arabic",
                        "Noto Serif Hebrew",
                        "Arial",
                    ],
                },
            },
            "density": "comfortable",
            "coverStyle": "minimal",
            "speakerColorMode": "accent",
            "timestampDetail": "section",
            "sectionInclusion": {
                "speakerDirectory": False,
                "translation": False,
                "qualityAppendix": False,
            },
            "brandAccent": "#315C8C",
        },
    ),
    ReportStylePreset.COMPACT_REVIEW: _with_changes(
        ReportStylePreset.COMPACT_REVIEW,
        {
            "layoutTemplate": "compact-review",
            "density": "compact",
            "coverStyle": "none",
            "speakerColorMode": "accent",
            "timestampDetail": "milliseconds",
            "sectionInclusion": {
                "cover": False,
                "tableOfContents": False,
                "translation": False,
                "summary": False,
            },
            "margins": {
                "topMm": 12.0,
                "rightMm": 12.0,
                "bottomMm": 14.0,
                "leftMm": 12.0,
            },
            "brandAccent": "#475467",
        },
    ),
    ReportStylePreset.ACCESSIBLE_HIGH_CONTRAST: _with_changes(
        ReportStylePreset.ACCESSIBLE_HIGH_CONTRAST,
        {
            "layoutTemplate": "accessible-large-print",
            "font": {
                "family": "Atkinson Hyperlegible Next",
                "fallbacks": {
                    "latin": [
                        "Atkinson Hyperlegible Next",
                        "Verdana",
                        "Arial",
                    ],
                    "cjk": [
                        "Noto Sans CJK SC",
                        "Source Han Sans SC",
                        "Yu Gothic UI",
                        "Malgun Gothic",
                    ],
                    "rtl": [
                        "Noto Sans Arabic",
                        "Noto Sans Hebrew",
                        "Arial",
                    ],
                },
            },
            "density": "relaxed",
            "coverStyle": "minimal",
            "speakerColorMode": "accessible",
            "timestampDetail": "segment",
            "brandAccent": "#005FCC",
            "highContrast": True,
            "margins": {
                "topMm": 22.0,
                "rightMm": 22.0,
                "bottomMm": 22.0,
                "leftMm": 22.0,
            },
        },
    ),
    ReportStylePreset.ARCHIVE_MONOCHROME: _with_changes(
        ReportStylePreset.ARCHIVE_MONOCHROME,
        {
            "layoutTemplate": "archive-monochrome",
            "font": {
                "family": "Noto Serif",
                "fallbacks": {
                    "latin": ["Noto Serif", "Georgia", "Times New Roman"],
                    "cjk": [
                        "Noto Serif CJK SC",
                        "Source Han Serif SC",
                        "Yu Mincho",
                        "Batang",
                    ],
                    "rtl": [
                        "Noto Naskh Arabic",
                        "Noto Serif Hebrew",
                        "Arial",
                    ],
                },
            },
            "density": "compact",
            "coverStyle": "minimal",
            "speakerColorMode": "monochrome",
            "timestampDetail": "segment",
            "page": {"size": "Letter", "orientation": "portrait"},
            "brandAccent": "#222222",
            "highContrast": True,
        },
    ),
    ReportStylePreset.CUSTOM: _with_changes(
        ReportStylePreset.CUSTOM,
        {
            "layoutTemplate": "modern-editorial",
            "coverStyle": "minimal",
        },
    ),
}


REPORT_STYLE_PRESETS = MappingProxyType(
    {
        preset.value: ReportStyleConfig.from_dict(payload)
        for preset, payload in _RAW_PRESETS.items()
    }
)


def list_report_style_presets(
    *,
    include_custom: bool = True,
) -> tuple[str, ...]:
    """Return stable preset identifiers in UI display order."""

    return tuple(
        preset.value
        for preset in ReportStylePreset
        if include_custom or preset is not ReportStylePreset.CUSTOM
    )


def get_report_style_preset(
    preset: ReportStylePreset | str,
) -> ReportStyleConfig:
    """Return the immutable effective defaults for one preset."""

    normalized = _enum_value(ReportStylePreset, preset, "preset")
    return REPORT_STYLE_PRESETS[normalized.value]


def effective_defaults(
    preset: ReportStylePreset | str = ReportStylePreset.MODERN_EDITORIAL,
) -> dict[str, Any]:
    """Return a detached canonical dictionary for safe UI editing."""

    return copy.deepcopy(get_report_style_preset(preset).canonical_dict())


def resolve_report_style(
    preset: ReportStylePreset | str = ReportStylePreset.MODERN_EDITORIAL,
    overrides: Mapping[str, Any] | None = None,
) -> ReportStyleConfig:
    """Resolve partial nested overrides over a stable preset.

    The input mapping is never mutated.  Unknown fields, unsafe logo paths,
    false source-protection claims, and invalid nested values fail closed.
    """

    normalized = _enum_value(ReportStylePreset, preset, "preset")
    base = effective_defaults(normalized)
    if overrides is None:
        return ReportStyleConfig.from_dict(base)
    override_payload = _mapping(overrides, "overrides")
    if (
        "preset" in override_payload
        and override_payload["preset"] != normalized.value
    ):
        raise ReportStyleError(
            "overrides.preset must match the selected base preset"
        )
    merged = _deep_merge(base, override_payload)
    merged["preset"] = normalized.value
    return ReportStyleConfig.from_dict(merged)


def validate_report_style(
    value: ReportStyleConfig | Mapping[str, Any],
) -> ReportStyleConfig:
    """Validate and normalize one complete report-style payload."""

    if isinstance(value, ReportStyleConfig):
        return value
    return ReportStyleConfig.from_dict(value)


def canonical_report_style_dict(
    value: ReportStyleConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return the canonical detached representation."""

    return validate_report_style(value).canonical_dict()


def deterministic_report_style_hash(
    value: ReportStyleConfig | Mapping[str, Any],
) -> str:
    """Validate and hash one report style deterministically."""

    return validate_report_style(value).deterministic_hash()


__all__ = [
    "CoverStyle",
    "Density",
    "FONT_AVAILABILITY_DECLARATION",
    "FONT_EMBEDDING_DECLARATION",
    "FontFallbacks",
    "FontPolicy",
    "LayoutTemplate",
    "LogoPolicy",
    "MarginPolicy",
    "PageOrientation",
    "PagePolicy",
    "PageSize",
    "REPORT_STYLE_PRESETS",
    "REPORT_STYLE_SCHEMA_VERSION",
    "ReportStyleConfig",
    "ReportStyleError",
    "ReportStylePreset",
    "RunningText",
    "SectionInclusion",
    "SpeakerColorMode",
    "TimestampDetail",
    "canonical_report_style_dict",
    "deterministic_report_style_hash",
    "effective_defaults",
    "get_report_style_preset",
    "list_report_style_presets",
    "resolve_report_style",
    "validate_report_style",
]
