from __future__ import annotations

import json

import pytest

from backend.persistence import validate_strict_json
from backend.semantic_language_calibration import (
    SEMANTIC_LANGUAGE_CALIBRATION_ARTIFACT_TYPE,
    SEMANTIC_LANGUAGE_CALIBRATION_SCHEMA_VERSION,
    build_semantic_language_calibration,
    calibrate_visible_language,
)


def _calibrate(language: str, text: str) -> dict:
    result = build_semantic_language_calibration(language, text)
    validate_strict_json(result)
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def test_english_surface_against_cantonese_claim_requests_language_and_asr() -> None:
    result = _calibrate("yue", "from our department")

    assert result["scriptProfile"]["latin"] == 17
    assert result["scriptProfile"]["han"] == 0
    assert result["codeSwitchLikely"] is False
    assert result["claimedLanguageConflict"] is True
    assert result["scriptVariantConflict"] is False
    assert result["recommendedDomains"] == ["language-span", "asr-text"]


def test_han_plus_english_phrase_is_code_switch_not_claim_conflict() -> None:
    result = _calibrate("zh", "念的是 electric engineering")

    assert result["scriptProfile"]["han"] == 3
    assert result["scriptProfile"]["latin"] == 19
    assert result["codeSwitchLikely"] is True
    assert result["claimedLanguageConflict"] is False
    assert result["scriptVariantConflict"] is False
    assert result["recommendedDomains"] == ["language-span", "asr-text"]


def test_traditional_cantonese_surface_is_compatible() -> None:
    result = _calibrate("yue", "如果我們要去申請")

    assert result["scriptProfile"]["han"] == 8
    assert result["codeSwitchLikely"] is False
    assert result["claimedLanguageConflict"] is False
    assert result["scriptVariantConflict"] is False
    assert result["recommendedDomains"] == []


def test_simplified_han_against_explicit_hant_claim_is_flagged() -> None:
    result = _calibrate(
        "yue-Hant-HK",
        "对我来说，这似乎不合理；这肯定不公平。",
    )

    assert result["scriptProfile"]["han"] > 0
    assert result["scriptProfile"]["simplifiedHanCount"] >= 2
    assert result["scriptProfile"]["traditionalHanCount"] == 0
    assert result["codeSwitchLikely"] is False
    assert result["claimedLanguageConflict"] is True
    assert result["scriptVariantConflict"] is True
    assert result["recommendedDomains"] == ["language-span", "asr-text"]


@pytest.mark.parametrize(
    ("language", "text", "script"),
    [
        ("ar-EG", "هذا نص عربي واضح.", "arabic"),
        ("es-419", "Esta es una frase normal.", "latin"),
        ("hi-IN", "यह एक सामान्य वाक्य है।", "devanagari"),
        ("ja-JP", "これは日本語の文章です。", "kana"),
        ("ko-KR", "이것은 한국어 문장입니다.", "hangul"),
    ],
)
def test_normal_single_language_surfaces_are_not_escalated(
    language: str,
    text: str,
    script: str,
) -> None:
    result = _calibrate(language, text)

    assert result["scriptProfile"][script] > 0
    assert result["codeSwitchLikely"] is False
    assert result["claimedLanguageConflict"] is False
    assert result["scriptVariantConflict"] is False
    assert result["recommendedDomains"] == []


def test_technical_literals_do_not_create_language_conflict() -> None:
    result = _calibrate(
        "zh",
        "2024 API v2 https://example.com",
    )

    assert result["scriptProfile"]["letterCount"] > 0
    assert result["scriptProfile"]["meaningfulLetterCount"] == 0
    assert result["scriptProfile"]["protectedLetterCount"] > 0
    assert result["codeSwitchLikely"] is False
    assert result["claimedLanguageConflict"] is False
    assert result["recommendedDomains"] == []


@pytest.mark.parametrize(
    ("language", "text", "script"),
    [
        ("sr-Latn-RS", "Ovo je normalna recenica.", "latin"),
        ("sr-Cyrl-RS", "Ово је нормална реченица.", "cyrillic"),
        ("az-Arab-IR", "بو عادی بیر جمله‌دیر.", "arabic"),
    ],
)
def test_explicit_script_subtag_overrides_language_default(
    language: str,
    text: str,
    script: str,
) -> None:
    result = _calibrate(language, text)

    assert result["scriptProfile"][script] > 0
    assert result["claimedLanguageConflict"] is False
    assert result["recommendedDomains"] == []


def test_explicit_script_subtag_conflict_is_flagged() -> None:
    result = _calibrate("sr-Latn-RS", "Ово је написано ћирилицом.")

    assert result["scriptProfile"]["cyrillic"] > 0
    assert result["claimedLanguageConflict"] is True
    assert result["recommendedDomains"] == ["language-span", "asr-text"]


def test_result_is_deterministic_and_alias_matches() -> None:
    first = _calibrate("sr-Latn-RS", "A normal Latin sentence.")
    second = calibrate_visible_language("sr-Latn-RS", "A normal Latin sentence.")

    assert first == second
    assert first["schemaVersion"] == SEMANTIC_LANGUAGE_CALIBRATION_SCHEMA_VERSION
    assert first["artifactType"] == SEMANTIC_LANGUAGE_CALIBRATION_ARTIFACT_TYPE


@pytest.mark.parametrize(
    "language",
    ["", "auto", "not a tag", "zh-汉"],
)
def test_invalid_claimed_language_fails_closed(language: str) -> None:
    with pytest.raises(ValueError):
        build_semantic_language_calibration(language, "text")


def test_visible_text_must_be_a_string() -> None:
    with pytest.raises(ValueError, match="visible_text"):
        build_semantic_language_calibration("en", None)  # type: ignore[arg-type]
