from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from backend.report_styles import (
    FONT_AVAILABILITY_DECLARATION,
    FONT_EMBEDDING_DECLARATION,
    REPORT_STYLE_SCHEMA_VERSION,
    ReportStyleError,
    ReportStylePreset,
    canonical_report_style_dict,
    deterministic_report_style_hash,
    effective_defaults,
    get_report_style_preset,
    list_report_style_presets,
    resolve_report_style,
    validate_report_style,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "report-style.schema.json"
)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _assert_rejected_by_python_and_schema(
    payload: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    with pytest.raises(ReportStyleError):
        validate_report_style(payload)
    assert list(validator.iter_errors(payload))


def test_schema_is_draft_2020_12_and_versioned(
    validator: Draft202012Validator,
) -> None:
    assert validator.schema["$schema"].endswith("draft/2020-12/schema")
    assert validator.schema["properties"]["schemaVersion"]["const"] == "1.0.0"
    assert REPORT_STYLE_SCHEMA_VERSION == "1.0.0"


def test_six_shipped_presets_plus_custom_are_stable() -> None:
    assert list_report_style_presets(include_custom=False) == (
        "modern-editorial",
        "conversation-focus",
        "executive-brief",
        "compact-review",
        "accessible-high-contrast",
        "archive-monochrome",
    )
    assert list_report_style_presets()[-1] == "custom"
    assert len(list_report_style_presets()) == 7


def test_every_preset_is_valid_effective_and_source_preserving(
    validator: Draft202012Validator,
) -> None:
    hashes: set[str] = set()
    for preset in ReportStylePreset:
        config = get_report_style_preset(preset)
        payload = config.canonical_dict()

        assert payload["preset"] == preset.value
        assert not list(validator.iter_errors(payload))
        assert validate_report_style(payload) == config
        assert payload["sourceProtection"] == {
            "preserveOriginalTranscript": True,
            "mutatesOriginalTranscript": False,
            "presentationOnly": True,
        }
        assert payload["font"]["availability"] == (
            FONT_AVAILABILITY_DECLARATION
        )
        assert payload["font"]["embedding"] == FONT_EMBEDDING_DECLARATION
        assert payload["font"]["fallbacks"]["latin"]
        assert payload["font"]["fallbacks"]["cjk"]
        assert payload["font"]["fallbacks"]["rtl"]
        hashes.add(config.deterministic_hash())

    assert len(hashes) == len(ReportStylePreset)


def test_effective_defaults_are_detached_and_presets_are_immutable() -> None:
    first = effective_defaults("modern-editorial")
    second = effective_defaults("modern-editorial")
    first["font"]["fallbacks"]["cjk"].append("User Font")
    first["margins"]["leftMm"] = 49

    assert "User Font" not in second["font"]["fallbacks"]["cjk"]
    assert second["margins"]["leftMm"] == 17.0

    preset = get_report_style_preset("modern-editorial")
    with pytest.raises(FrozenInstanceError):
        preset.high_contrast = True  # type: ignore[misc]


def test_partial_overrides_deep_merge_without_mutating_the_caller(
    validator: Draft202012Validator,
) -> None:
    overrides = {
        "density": "relaxed",
        "font": {
            "family": "Noto Sans",
            "fallbacks": {
                "rtl": ["Noto Sans Arabic", "Noto Sans Hebrew"],
            },
        },
        "margins": {"leftMm": 23},
        "logo": {
            "enabled": True,
            "path": r"branding\客户-logo.PNG",
            "altText": "Customer logo",
        },
        "brandAccent": "#aabbcc",
    }
    original = copy.deepcopy(overrides)

    config = resolve_report_style("custom", overrides)
    payload = config.canonical_dict()

    assert overrides == original
    assert payload["density"] == "relaxed"
    assert payload["font"]["family"] == "Noto Sans"
    assert payload["font"]["fallbacks"]["latin"] == [
        "Inter",
        "Segoe UI",
        "Arial",
    ]
    assert payload["font"]["fallbacks"]["rtl"] == [
        "Noto Sans Arabic",
        "Noto Sans Hebrew",
    ]
    assert payload["margins"]["leftMm"] == 23.0
    assert payload["margins"]["rightMm"] == 17.0
    assert payload["logo"]["path"] == "branding/客户-logo.PNG"
    assert payload["brandAccent"] == "#AABBCC"
    assert not list(validator.iter_errors(payload))


def test_canonical_hash_is_order_independent_and_normalizes_equivalent_values() -> None:
    payload = resolve_report_style(
        "custom",
        {
            "brandAccent": "#abcdef",
            "logo": {
                "enabled": True,
                "path": r"assets\logo.webp",
                "altText": "Brand",
            },
        },
    ).canonical_dict()
    reordered = {key: payload[key] for key in reversed(tuple(payload))}
    reordered["brandAccent"] = "#ABCDEF"
    reordered["logo"] = dict(reordered["logo"])
    reordered["logo"]["path"] = "assets/logo.webp"

    assert canonical_report_style_dict(payload) == (
        canonical_report_style_dict(reordered)
    )
    assert deterministic_report_style_hash(payload) == (
        deterministic_report_style_hash(reordered)
    )
    assert len(deterministic_report_style_hash(payload)) == 64


def test_hash_changes_when_effective_presentation_changes() -> None:
    base = get_report_style_preset("modern-editorial")
    changed = resolve_report_style(
        "modern-editorial",
        {"timestampDetail": "milliseconds"},
    )

    assert base.deterministic_hash() != changed.deterministic_hash()


def test_canonical_json_is_utf8_stable_and_contains_no_transcript_content() -> None:
    config = resolve_report_style(
        "custom",
        {
            "header": {"text": "会议记录"},
            "logo": {
                "enabled": True,
                "path": "品牌/标识.png",
                "altText": "品牌标识",
            },
        },
    )
    canonical = config.canonical_json()

    assert "会议记录" in canonical
    assert "\\u4f1a" not in canonical
    assert "transcriptText" not in canonical
    assert '"mutatesOriginalTranscript":false' in canonical


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("https://example.com/logo.png", "remote URL"),
        ("file:///C:/logo.png", "file URL"),
        ("data:image/png;base64,AAAA", "data URL"),
        (r"C:\assets\logo.png", "absolute drive path"),
        ("D:/assets/logo.png", "forward-slash drive path"),
        (r"\\server\share\logo.png", "UNC path"),
        ("//server/share/logo.png", "network path"),
        ("/opt/assets/logo.png", "rooted path"),
        ("../logo.png", "leading traversal"),
        ("assets/../logo.png", "nested traversal"),
        (r"assets\.\logo.png", "dot segment"),
        ("assets//logo.png", "empty segment"),
        (r"assets\\logo.png", "empty backslash segment"),
        ("assets/%2e%2e/logo.png", "encoded traversal"),
        ("assets/logo.svg", "active or unsupported image type"),
        ("assets/logo.png?cache=1", "query suffix"),
        (" assets/logo.png", "leading whitespace"),
        ("assets /logo.png", "Windows trailing segment space"),
        ("assets./logo.png", "Windows trailing segment dot"),
        ("assets/logo.png ", "trailing whitespace"),
    ],
)
def test_unsafe_logo_paths_fail_closed_in_python_and_schema(
    path: str,
    reason: str,
    validator: Draft202012Validator,
) -> None:
    payload = effective_defaults("custom")
    payload["logo"] = {
        "enabled": True,
        "path": path,
        "altText": f"Rejected: {reason}",
        "maxWidthMm": 28,
    }
    _assert_rejected_by_python_and_schema(payload, validator)


@pytest.mark.parametrize(
    "path",
    [
        "assets/logo.png",
        r"assets\logo.JPEG",
        "branding/客户/logo.webp",
        "logo.JPG",
    ],
)
def test_safe_local_relative_logo_paths_are_accepted_and_canonicalized(
    path: str,
    validator: Draft202012Validator,
) -> None:
    config = resolve_report_style(
        "custom",
        {
            "logo": {
                "enabled": True,
                "path": path,
                "altText": "Local brand mark",
            }
        },
    )
    payload = config.canonical_dict()

    assert payload["logo"]["path"] == path.replace("\\", "/")
    assert not list(validator.iter_errors(payload))


@pytest.mark.parametrize(
    "logo",
    [
        {
            "enabled": True,
            "path": None,
            "altText": "Missing image",
            "maxWidthMm": 28,
        },
        {
            "enabled": True,
            "path": "assets/logo.png",
            "altText": "",
            "maxWidthMm": 28,
        },
        {
            "enabled": True,
            "path": "assets/logo.png",
            "altText": "   ",
            "maxWidthMm": 28,
        },
        {
            "enabled": False,
            "path": "https://example.com/logo.png",
            "altText": "",
            "maxWidthMm": 28,
        },
    ],
)
def test_logo_dependency_and_security_rules_match_schema(
    logo: dict[str, Any],
    validator: Draft202012Validator,
) -> None:
    payload = effective_defaults("custom")
    payload["logo"] = logo
    _assert_rejected_by_python_and_schema(payload, validator)


def test_disabled_logo_can_retain_a_safe_reversible_local_choice(
    validator: Draft202012Validator,
) -> None:
    config = resolve_report_style(
        "custom",
        {
            "logo": {
                "enabled": False,
                "path": "branding/logo.png",
                "altText": "",
            }
        },
    )
    payload = config.canonical_dict()

    assert payload["logo"]["path"] == "branding/logo.png"
    assert not list(validator.iter_errors(payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schemaVersion",), "2.0.0"),
        (("layoutTemplate",), "unknown-layout"),
        (("density",), "dense"),
        (("coverStyle",), "animated"),
        (("speakerColorMode",), "random"),
        (("timestampDetail",), "frames"),
        (("page", "size"), "A3"),
        (("page", "orientation"), "diagonal"),
        (("margins", "topMm"), -1),
        (("margins", "leftMm"), 51),
        (("margins", "rightMm"), True),
        (("brandAccent",), "#FFF"),
        (("highContrast",), 1),
        (("font", "availability"), "installed"),
        (("font", "embedding"), "embedded"),
    ],
)
def test_enum_range_type_and_truth_claim_failures_match_schema(
    path: tuple[str, ...],
    value: Any,
    validator: Draft202012Validator,
) -> None:
    payload = effective_defaults("modern-editorial")
    cursor: dict[str, Any] = payload
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    _assert_rejected_by_python_and_schema(payload, validator)


@pytest.mark.parametrize(
    "fallbacks",
    [
        {"latin": [], "cjk": ["Noto Sans CJK SC"], "rtl": ["Arial"]},
        {
            "latin": ["Inter", "Inter"],
            "cjk": ["Noto Sans CJK SC"],
            "rtl": ["Arial"],
        },
        {
            "latin": [" Inter"],
            "cjk": ["Noto Sans CJK SC"],
            "rtl": ["Arial"],
        },
        {
            "latin": ["Inter"],
            "cjk": ["Noto Sans CJK SC"],
            "rtl": [],
        },
    ],
)
def test_font_fallback_declarations_are_strict_and_script_complete(
    fallbacks: dict[str, list[str]],
    validator: Draft202012Validator,
) -> None:
    payload = effective_defaults("custom")
    payload["font"]["fallbacks"] = fallbacks
    _assert_rejected_by_python_and_schema(payload, validator)


def test_at_least_one_content_section_is_required(
    validator: Draft202012Validator,
) -> None:
    payload = effective_defaults("custom")
    for key in (
        "transcript",
        "translation",
        "polishNotes",
        "summary",
        "qualityAppendix",
        "provenance",
    ):
        payload["sectionInclusion"][key] = False
    _assert_rejected_by_python_and_schema(payload, validator)


@pytest.mark.parametrize(
    "source_protection",
    [
        {
            "preserveOriginalTranscript": False,
            "mutatesOriginalTranscript": False,
            "presentationOnly": True,
        },
        {
            "preserveOriginalTranscript": True,
            "mutatesOriginalTranscript": True,
            "presentationOnly": True,
        },
        {
            "preserveOriginalTranscript": True,
            "mutatesOriginalTranscript": False,
            "presentationOnly": False,
        },
    ],
)
def test_source_transcript_protection_is_a_hard_invariant(
    source_protection: dict[str, bool],
    validator: Draft202012Validator,
) -> None:
    payload = effective_defaults("custom")
    payload["sourceProtection"] = source_protection
    _assert_rejected_by_python_and_schema(payload, validator)


def test_unknown_and_missing_properties_fail_closed(
    validator: Draft202012Validator,
) -> None:
    unknown = effective_defaults("custom")
    unknown["transcriptText"] = "must never enter the appearance domain"
    _assert_rejected_by_python_and_schema(unknown, validator)

    nested_unknown = effective_defaults("custom")
    nested_unknown["font"]["remoteUrl"] = "https://example.com/font.woff2"
    _assert_rejected_by_python_and_schema(nested_unknown, validator)

    missing = effective_defaults("custom")
    del missing["timestampDetail"]
    _assert_rejected_by_python_and_schema(missing, validator)


def test_control_characters_fail_closed_in_text_fields(
    validator: Draft202012Validator,
) -> None:
    payload = effective_defaults("custom")
    payload["header"]["text"] = "Header\nInjected"
    _assert_rejected_by_python_and_schema(payload, validator)


def test_custom_base_can_be_fully_edited_but_preset_identity_cannot_be_spoofed() -> None:
    config = resolve_report_style(
        ReportStylePreset.CUSTOM,
        {
            "layoutTemplate": "archive-monochrome",
            "page": {"size": "Legal", "orientation": "landscape"},
            "header": {
                "enabled": False,
                "text": "Retained while disabled",
            },
            "sectionInclusion": {
                "transcript": False,
                "summary": True,
            },
        },
    )
    assert config.preset is ReportStylePreset.CUSTOM
    assert config.page.size.value == "Legal"
    assert not config.header.enabled
    assert config.header.text == "Retained while disabled"

    with pytest.raises(ReportStyleError, match="must match"):
        resolve_report_style(
            "custom",
            {"preset": "modern-editorial"},
        )


def test_resolver_rejects_unknown_partial_override_keys() -> None:
    with pytest.raises(ReportStyleError, match="unknown properties"):
        resolve_report_style(
            "custom",
            {"margins": {"gutterMm": 12}},
        )
