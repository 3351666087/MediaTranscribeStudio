from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from backend.asr_evidence import build_asr_candidate_set


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
SHA256 = "a" * 64
LANGUAGE_TAGS = ("en-US", "sr-Latn-RS", "und", "mul")
INVALID_LANGUAGE_TAGS = (
    "auto",
    "AUTO",
    "en-a",
    "e",
    "123",
    "en--US",
    " en-US",
    "en-US ",
)


def _provider() -> dict[str, Any]:
    return {
        "id": "ollama-loopback",
        "version": "native-json-v1",
        "networkPolicy": "loopback-only",
    }


def _report_document(language: str) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "documentId": "contract-report-001",
        "generatedAt": "2026-07-22T00:00:00Z",
        "language": language,
        "source": {
            "fileName": "fixture.wav",
            "mediaType": "audio/wav",
            "durationMs": 1_000,
            "sha256": SHA256,
        },
        "speakerPolicy": {
            "mode": "auto",
            "resolvedCount": 1,
            "requireExactSet": True,
            "speakerIds": ["speaker-1"],
        },
        "speakers": [
            {
                "id": "speaker-1",
                "order": 1,
                "displayName": "Speaker 1",
                "shortLabel": "S1",
                "colorToken": "speaker.1",
            }
        ],
        "segments": [
            {
                "id": "segment-001",
                "startMs": 0,
                "endMs": 1_000,
                "speakerId": "speaker-1",
                "rawText": "Immutable source text.",
                "normalizedText": "Immutable source text.",
                "displayText": "Immutable source text.",
                "language": language,
                "confidence": 0.99,
                "evidence": {
                    "asr": {
                        "provider": "qwen3-asr",
                        "model": "Qwen3-ASR-1.7B",
                        "confidence": 0.99,
                    },
                    "boundary": {
                        "provider": "funasr",
                        "confidence": 0.99,
                    },
                    "speaker": {
                        "provider": "cam++",
                        "assignment": "speaker-1",
                        "locked": False,
                        "scores": [
                            {
                                "speakerId": "speaker-1",
                                "score": 0.99,
                            }
                        ],
                    },
                },
                "revisions": [],
            }
        ],
        "provenance": {
            "pipelineVersion": "ultimate-contract-test",
            "models": [
                {
                    "role": "asr",
                    "name": "Qwen3-ASR-1.7B",
                }
            ],
            "offline": True,
        },
    }


def _semantic_arbitration(language: str) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "requestId": "semantic-contract-001",
        "segmentId": "segment-001",
        "constraints": {
            "language": language,
            "translationAllowed": False,
            "stylePolishAllowed": False,
            "hallucinationAllowed": False,
            "speakerChangeAllowed": False,
            "overlapSplitAllowed": False,
            "rawTextMutable": False,
        },
        "proposal": {
            "action": "keep",
            "normalizedText": "Immutable source text.",
            "speakerId": "speaker-1",
            "reasonCodes": ["SOURCE_PRESERVED"],
            "evidenceSegmentIds": ["segment-001"],
            "application": "suggestion",
        },
        "validation": {
            "schemaValid": True,
            "sourceTraceable": True,
            "textBoundarySafe": True,
            "speakerConstraintSafe": True,
            "deterministicDecision": "suggest",
        },
    }


def _business_request(language: str) -> dict[str, Any]:
    return {
        "schemaVersion": "1.2.0",
        "translationTargets": [language],
        "summary": True,
        "model": "qwen3.5:4b",
        "outputLocale": language,
        "promptVersion": "business-v2",
    }


def _translation_output(language: str) -> dict[str, Any]:
    return {
        "schemaVersion": "1.1.0",
        "variant": f"translation:{language}",
        "inputHash": SHA256,
        "model": "qwen3.5:4b",
        "promptVersion": "translation-v1",
        "provider": _provider(),
        "temperature": 0,
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
        "status": "completed",
        "sourceLanguage": "en",
        "targetLanguage": language,
        "segments": [
            {
                "id": "segment-001",
                "speakerId": "speaker-1",
                "startMs": 0,
                "endMs": 1_000,
                "sourceTextHash": SHA256,
                "text": "Translated text.",
                "language": language,
            }
        ],
    }


def _summary_output(language: str) -> dict[str, Any]:
    evidence = {
        "text": "Evidence-grounded point.",
        "evidenceSegmentIds": ["segment-001"],
        "timeRange": {
            "startMs": 0,
            "endMs": 1_000,
        },
    }
    return {
        "schemaVersion": "1.1.0",
        "variant": "summary",
        "inputHash": SHA256,
        "model": "qwen3.5:4b",
        "promptVersion": "summary-v1",
        "provider": _provider(),
        "temperature": 0,
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
        "status": "completed",
        "language": language,
        "executiveSummary": "Evidence-grounded summary.",
        "keyPoints": [copy.deepcopy(evidence)],
        "topics": [copy.deepcopy(evidence)],
        "actionItems": [copy.deepcopy(evidence)],
    }


SCHEMA_FIXTURES: tuple[
    tuple[str, Callable[[str], dict[str, Any]]], ...
] = (
    ("report-document.schema.json", _report_document),
    ("semantic-arbitration.schema.json", _semantic_arbitration),
    ("business-processing-request.schema.json", _business_request),
    ("translation-output.schema.json", _translation_output),
    ("summary-output.schema.json", _summary_output),
)


def _validator(filename: str) -> Draft202012Validator:
    schema = json.loads((CONTRACTS / filename).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize(("filename", "factory"), SCHEMA_FIXTURES)
def test_contract_is_valid_draft_2020_12_and_accepts_global_language_tags(
    filename: str,
    factory: Callable[[str], dict[str, Any]],
) -> None:
    validator = _validator(filename)
    for language in LANGUAGE_TAGS:
        validator.validate(factory(language))


@pytest.mark.parametrize(("filename", "factory"), SCHEMA_FIXTURES)
def test_persisted_contracts_reject_request_only_and_malformed_language_tags(
    filename: str,
    factory: Callable[[str], dict[str, Any]],
) -> None:
    validator = _validator(filename)
    for language in INVALID_LANGUAGE_TAGS:
        with pytest.raises(ValidationError):
            validator.validate(factory(language))


def test_report_locale_is_optional_and_uses_persisted_language_tags() -> None:
    validator = _validator("report-document.schema.json")

    omitted = _report_document("ar-SA")
    validator.validate(omitted)

    for report_locale in LANGUAGE_TAGS:
        document = _report_document("ar-SA")
        document["reportLocale"] = report_locale
        validator.validate(document)
        assert document["language"] == "ar-SA"

    for report_locale in INVALID_LANGUAGE_TAGS:
        document = _report_document("ar-SA")
        document["reportLocale"] = report_locale
        with pytest.raises(ValidationError):
            validator.validate(document)


def test_asr_evidence_schema_accepts_canonical_and_rejects_unsafe_shapes() -> None:
    validator = _validator("asr-evidence.schema.json")
    value = build_asr_candidate_set(
        model_id="Qwen3-ASR-1.7B",
        model_revision="revision-fixture",
        model_manifest_sha256=SHA256,
        model_identity_status="manifest-bound",
        source_audio_sha256=SHA256,
        normalization_profile="mono-16khz-f32-v1",
        source_window_id="window-1",
        start_ms=0,
        end_ms=1_000,
        hypotheses=[
            {
                "text": "Hello world",
                "language": "en-US",
                "tokens": [
                    {"text": "Hello", "startMs": 0, "endMs": 400},
                    {"text": "world", "startMs": 500, "endMs": 900},
                ],
                "acousticScore": -0.1,
                "acousticScoreStatus": "available",
                "decodeScore": -0.2,
                "decodeScoreStatus": "available",
            }
        ],
    )

    validator.validate(value)

    malformed_hash = copy.deepcopy(value)
    malformed_hash["candidateSetSha256"] = "not-a-sha256"
    with pytest.raises(ValidationError):
        validator.validate(malformed_hash)

    ineligible_scores = copy.deepcopy(value)
    ineligible_scores["nBest"][0]["decodeScore"] = None
    ineligible_scores["nBest"][0]["decodeScoreStatus"] = "provider-unavailable"
    with pytest.raises(ValidationError):
        validator.validate(ineligible_scores)

    projected_without_parent = copy.deepcopy(value)
    projected_without_parent["candidateSetType"] = "projection-derived-top1"
    projected_without_parent["nBest"][0]["acousticScore"] = None
    projected_without_parent["nBest"][0]["acousticScoreStatus"] = "projection-derived"
    projected_without_parent["nBest"][0]["decodeScore"] = None
    projected_without_parent["nBest"][0]["decodeScoreStatus"] = "projection-derived"
    projected_without_parent["nBest"][0]["lexicalRepairEligible"] = False
    with pytest.raises(ValidationError):
        validator.validate(projected_without_parent)
