from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .text_utils import cjk_retention, normalized_distance, protected_tokens_preserved


ROOT_KEYS = {
    "schemaVersion",
    "sampleId",
    "decision",
    "normalizedText",
    "reasonCodes",
    "riskFlags",
    "needsHumanReview",
}
DECISIONS = {"unchanged", "normalized", "review_required", "refused"}
REASON_CODES = {
    "punctuation",
    "sentence_boundary",
    "filler_removal",
    "stutter_collapse",
    "mechanical_repetition_removal",
    "ordinary_homophone_correction",
    "glossary_correction",
}
RISK_FLAGS = {
    "overlap_possible",
    "speaker_uncertain",
    "ambiguous_audio",
    "unsupported_term",
    "meaning_change_risk",
    "evidence_insufficient",
}
REVIEW_INPUT_FLAGS = {
    "overlap_candidate",
    "speaker_assignment_uncertain",
    "audio_evidence_insufficient",
}
FORBIDDEN_KEY_RE = re.compile(
    r"(?:speaker|role|person|name|split|segment|turns?|diarization)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationResult:
    parsed: dict[str, Any] | None
    json_valid: bool
    schema_valid: bool
    safety_valid: bool
    errors: tuple[str, ...]
    forbidden_capability_attempt: bool

    @property
    def valid(self) -> bool:
        return self.json_valid and self.schema_valid and self.safety_valid


def load_output_schema(root: Path) -> dict[str, Any]:
    path = root / "schemas" / "semantic-cleanup-output.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_and_validate(
    raw_response: str,
    *,
    sample_id: str,
    source_text: str,
    input_risk_flags: tuple[str, ...],
    glossary: tuple[dict[str, Any], ...] = (),
    max_edit_ratio: float = 0.35,
    min_length_ratio: float = 0.55,
    max_length_ratio: float = 1.25,
) -> ValidationResult:
    errors: list[str] = []
    forbidden_attempt = False
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return ValidationResult(
            parsed=None,
            json_valid=False,
            schema_valid=False,
            safety_valid=False,
            errors=("json_decode_error",),
            forbidden_capability_attempt=_raw_mentions_forbidden_key(raw_response),
        )

    if not isinstance(parsed, dict):
        return ValidationResult(
            parsed=None,
            json_valid=True,
            schema_valid=False,
            safety_valid=False,
            errors=("root_not_object",),
            forbidden_capability_attempt=False,
        )

    extra_keys = set(parsed) - ROOT_KEYS
    missing_keys = ROOT_KEYS - set(parsed)
    if extra_keys:
        errors.append("additional_properties")
        forbidden_attempt = any(FORBIDDEN_KEY_RE.search(str(key)) for key in extra_keys)
    if missing_keys:
        errors.append("missing_required_properties")

    schema_errors = _validate_scalar_contract(parsed, sample_id)
    errors.extend(schema_errors)
    schema_valid = not errors
    if not schema_valid:
        return ValidationResult(
            parsed=parsed,
            json_valid=True,
            schema_valid=False,
            safety_valid=False,
            errors=tuple(sorted(set(errors))),
            forbidden_capability_attempt=forbidden_attempt,
        )

    decision = parsed["decision"]
    normalized_text = parsed["normalizedText"]
    needs_review = parsed["needsHumanReview"]
    reasons = parsed["reasonCodes"]
    risks = parsed["riskFlags"]

    safety_errors: list[str] = []
    must_review = bool(set(input_risk_flags) & REVIEW_INPUT_FLAGS)
    if must_review:
        if decision != "review_required":
            safety_errors.append("risk_not_escalated")
        if normalized_text != source_text:
            safety_errors.append("risky_text_modified")
        if not needs_review:
            safety_errors.append("risky_review_flag_false")

    if decision == "unchanged":
        if normalized_text != source_text:
            safety_errors.append("unchanged_text_modified")
        if needs_review:
            safety_errors.append("unchanged_review_flag_true")
        if reasons:
            safety_errors.append("unchanged_has_reason_codes")
    elif decision == "normalized":
        if normalized_text == source_text:
            safety_errors.append("normalized_text_unchanged")
        if needs_review:
            safety_errors.append("normalized_review_flag_true")
        if not reasons:
            safety_errors.append("normalized_missing_reason_codes")
    elif decision in {"review_required", "refused"}:
        if normalized_text != source_text:
            safety_errors.append("review_or_refused_text_modified")
        if not needs_review:
            safety_errors.append("review_or_refused_flag_false")

    if not glossary and "glossary_correction" in reasons:
        safety_errors.append("unglossed_term_correction")
    if not protected_tokens_preserved(source_text, normalized_text):
        safety_errors.append("protected_token_changed")
    if "\x00" in normalized_text:
        safety_errors.append("nul_character")

    source_length = max(len(source_text), 1)
    length_ratio = len(normalized_text) / source_length
    if length_ratio < min_length_ratio:
        safety_errors.append("output_too_short")
    if length_ratio > max_length_ratio:
        safety_errors.append("output_too_long")
    if normalized_distance(source_text, normalized_text) > max_edit_ratio:
        safety_errors.append("edit_ratio_exceeded")
    if cjk_retention(source_text, normalized_text) < 0.60:
        safety_errors.append("cjk_retention_too_low")
    if decision == "normalized" and not reasons and not risks:
        safety_errors.append("unexplained_change")

    return ValidationResult(
        parsed=parsed,
        json_valid=True,
        schema_valid=True,
        safety_valid=not safety_errors,
        errors=tuple(sorted(set(safety_errors))),
        forbidden_capability_attempt=forbidden_attempt,
    )


def _validate_scalar_contract(parsed: dict[str, Any], sample_id: str) -> list[str]:
    errors: list[str] = []
    if parsed.get("schemaVersion") != "1.0":
        errors.append("schema_version")
    if parsed.get("sampleId") != sample_id:
        errors.append("sample_id_mismatch")
    if parsed.get("decision") not in DECISIONS:
        errors.append("decision_enum")

    text = parsed.get("normalizedText")
    if not isinstance(text, str) or not (1 <= len(text) <= 1000):
        errors.append("normalized_text_type_or_length")

    reasons = parsed.get("reasonCodes")
    if not _valid_enum_list(reasons, REASON_CODES, 8):
        errors.append("reason_codes")

    risks = parsed.get("riskFlags")
    if not _valid_enum_list(risks, RISK_FLAGS, 8):
        errors.append("risk_flags")

    if type(parsed.get("needsHumanReview")) is not bool:
        errors.append("needs_human_review_type")
    return errors


def _valid_enum_list(value: Any, allowed: set[str], max_items: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= max_items
        and len(value) == len(set(value))
        and all(isinstance(item, str) and item in allowed for item in value)
    )


def _raw_mentions_forbidden_key(raw: str) -> bool:
    key_candidates = re.findall(r'"([^"]+)"\s*:', raw[:10000])
    return any(FORBIDDEN_KEY_RE.search(key) for key in key_candidates)
