from __future__ import annotations

from tools.audit_first_principles_batch import (
    _language_code,
    _target_chinese_script,
    _text_error,
    _tokens,
)


def test_language_aliases_are_compared_at_primary_code() -> None:
    assert _language_code("English") == "en"
    assert _language_code("en-US") == "en"
    assert _language_code("Tagalog") == "tl"


def test_text_error_is_deterministic_and_token_aware() -> None:
    assert _tokens("Hello 世界") == ["hello", "世", "界"]
    result = _text_error("Hello 世界", "Hello 世界")
    assert result == {
        "referenceTokens": 3,
        "hypothesisTokens": 3,
        "editDistance": 0,
        "normalizedError": 0.0,
    }


def test_target_chinese_script_rejects_wrong_language_translation() -> None:
    assert _target_chinese_script("这是中文翻译 PayPal")["targetScriptPass"] is True
    assert _target_chinese_script("tradução em português")["targetScriptPass"] is False
    assert _target_chinese_script("배우가 сказал")["targetScriptPass"] is False
