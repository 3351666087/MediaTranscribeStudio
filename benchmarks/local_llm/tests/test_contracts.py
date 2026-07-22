from __future__ import annotations

import json

from local_llm_bench.contracts import parse_and_validate


SAMPLE_ID = "0123456789abcdef"
SOURCE = "我们先看一下这个方案。"


def _response(**overrides):
    payload = {
        "schemaVersion": "1.0",
        "sampleId": SAMPLE_ID,
        "decision": "unchanged",
        "normalizedText": SOURCE,
        "reasonCodes": [],
        "riskFlags": [],
        "needsHumanReview": False,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_accepts_strict_unchanged_contract():
    result = parse_and_validate(
        _response(),
        sample_id=SAMPLE_ID,
        source_text=SOURCE,
        input_risk_flags=(),
    )
    assert result.valid


def test_rejects_extra_speaker_field():
    result = parse_and_validate(
        _response(speaker="speaker-1"),
        sample_id=SAMPLE_ID,
        source_text=SOURCE,
        input_risk_flags=(),
    )
    assert not result.schema_valid
    assert result.forbidden_capability_attempt
    assert "additional_properties" in result.errors


def test_overlap_must_escalate_without_editing():
    result = parse_and_validate(
        _response(
            decision="normalized",
            normalizedText="我们先看这个方案。",
            reasonCodes=["filler_removal"],
        ),
        sample_id=SAMPLE_ID,
        source_text=SOURCE,
        input_risk_flags=("overlap_candidate",),
    )
    assert not result.safety_valid
    assert "risk_not_escalated" in result.errors
    assert "risky_text_modified" in result.errors


def test_unglossed_term_correction_is_forbidden():
    result = parse_and_validate(
        _response(
            decision="normalized",
            normalizedText="我们先看一下该方案。",
            reasonCodes=["glossary_correction"],
        ),
        sample_id=SAMPLE_ID,
        source_text=SOURCE,
        input_risk_flags=(),
    )
    assert not result.safety_valid
    assert "unglossed_term_correction" in result.errors


def test_json_code_fence_is_not_silently_accepted():
    result = parse_and_validate(
        "```json\n" + _response() + "\n```",
        sample_id=SAMPLE_ID,
        source_text=SOURCE,
        input_risk_flags=(),
    )
    assert not result.json_valid
