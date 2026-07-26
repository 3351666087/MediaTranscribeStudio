from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from backend.business_processing import (
    BUSINESS_PROMPT_VERSION,
    BUSINESS_REQUEST_SCHEMA_VERSION,
    BUSINESS_SCHEMA_VERSION,
    BusinessProcessingConfig,
    BusinessProcessingRunner,
    _variant_input_hash,
)
from backend.errors import JobCancelled, WorkerError
from backend.local_llm import MappingLocalLLMProvider
from backend.persistence import canonical_json_sha256


def _document() -> dict[str, object]:
    return {
        "documentId": "doc-business-test-001",
        "language": "zh-CN",
        "segments": [
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1200,
                "speakerId": "speaker-1",
                "rawText": "我们今天确认发布计划。",
                "normalizedText": "我们今天确认发布计划。",
                "displayText": "我们今天确认发布计划。",
            },
            {
                "id": "segment-2",
                "startMs": 1300,
                "endMs": 2500,
                "speakerId": "speaker-2",
                "rawText": "我会在周五前完成验证。",
                "normalizedText": "我会在周五前完成验证。",
                "displayText": "我会在周五前完成验证。",
            },
        ],
    }


def _single_segment_document(
    text: str,
    *,
    language: str = "en",
) -> dict[str, object]:
    return {
        "documentId": "doc-semantic-guard-001",
        "language": language,
        "segments": [
            {
                "id": "segment-1",
                "startMs": 0,
                "endMs": 1_200,
                "speakerId": "speaker-1",
                "language": language,
                "rawText": text,
                "normalizedText": text,
                "displayText": text,
            }
        ],
    }


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _translated_segment(
    segment_id: str,
    speaker_id: str,
    start_ms: int,
    end_ms: int,
    source_text: str,
    text: str,
) -> dict[str, object]:
    return {
        "id": segment_id,
        "speakerId": speaker_id,
        "startMs": start_ms,
        "endMs": end_ms,
        "sourceTextHash": _source_hash(source_text),
        "text": text,
        "language": "en",
    }


def _summary_item(text: str, segment_id: str, start_ms: int, end_ms: int) -> dict[str, object]:
    return {
        "text": text,
        "evidenceSegmentIds": [segment_id],
        "timeRange": {"startMs": start_ms, "endMs": end_ms},
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _valid_translation_responses() -> list[dict[str, object]]:
    document = _document()
    segments = document["segments"]
    assert isinstance(segments, list)
    first = segments[0]
    second = segments[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    return [
        _translated_segment(
            "segment-1",
            "speaker-1",
            0,
            1200,
            str(first["normalizedText"]),
            "Today we confirmed the release plan.",
        ),
        _translated_segment(
            "segment-2",
            "speaker-2",
            1300,
            2500,
            str(second["normalizedText"]),
            "I will finish validation by Friday.",
        ),
    ]


def test_business_config_normalizes_language_tags_and_is_schema_ready() -> None:
    config = BusinessProcessingConfig(
        translation_targets=("zh_CN", "en"),
        summary=True,
        model=" qwen3.5:4b ",
        output_locale="en-US",
    )

    assert config.translation_targets == ("zh-CN", "en")
    assert config.model == "qwen3.5:4b"
    assert config.output_locale == "en-US"
    assert config.as_dict()["schemaVersion"] == BUSINESS_REQUEST_SCHEMA_VERSION
    assert config.enabled is True

    with pytest.raises(ValueError, match="unique"):
        BusinessProcessingConfig(translation_targets=("en", "en"))
    with pytest.raises(ValueError, match="unsupported business promptVersion"):
        BusinessProcessingConfig(
            summary=True,
            prompt_version="business-v999",
        )


@pytest.mark.parametrize(
    ("field", "config"),
    [
        (
            "translation target",
            BusinessProcessingConfig,
        ),
        (
            "summary output locale",
            BusinessProcessingConfig,
        ),
    ],
)
def test_business_config_rejects_request_only_auto_language(
    field: str,
    config: type[BusinessProcessingConfig],
) -> None:
    kwargs = (
        {"translation_targets": ("auto",)}
        if field == "translation target"
        else {"output_locale": "auto"}
    )
    with pytest.raises(ValueError, match="request input"):
        config(**kwargs)


def test_business_processing_rejects_request_only_auto_document_language(
    tmp_path: Path,
) -> None:
    document = _document()
    document["language"] = "auto"

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([])
        ).run(
            document,
            output_directory=tmp_path,
            config=BusinessProcessingConfig(summary=True),
        )

    assert error.value.code == "BUSINESS_INPUT_INVALID"


def test_runner_creates_translation_summary_and_manifest(
    tmp_path: Path,
) -> None:
    document = _document()
    original = copy.deepcopy(document)
    provider = MappingLocalLLMProvider(
        [
            _translated_segment(
                "segment-1",
                "speaker-1",
                0,
                1200,
                "我们今天确认发布计划。",
                "Today we confirmed the release plan.",
            ),
            _translated_segment(
                "segment-2",
                "speaker-2",
                1300,
                2500,
                "我会在周五前完成验证。",
                "I will finish validation by Friday.",
            ),
            {
                "executiveSummary": "会议确认了发布计划，并安排了验证工作。",
                "keyPoints": [
                    _summary_item(
                        "确认发布计划。",
                        "segment-1",
                        0,
                        1200,
                    )
                ],
                "topics": [
                    _summary_item(
                        "发布与验证。",
                        "segment-1",
                        0,
                        2500,
                    )
                ],
                "actionItems": [
                    {
                        **_summary_item(
                            "在周五前完成验证。",
                            "segment-2",
                            1300,
                            2500,
                        ),
                        "owner": "speaker-2",
                        "dueDate": "Friday",
                    }
                ],
            },
        ]
    )
    config = BusinessProcessingConfig(
        translation_targets=("en",),
        summary=True,
        output_locale="zh-CN",
    )

    artifacts = BusinessProcessingRunner(provider=provider).run(
        document,
        output_directory=tmp_path,
        config=config,
    )

    assert [path.name for path in artifacts] == [
        "translation-en.v1.json",
        "summary.v1.json",
        "business-manifest.v1.json",
    ]
    assert document == original

    translation = _read_json(tmp_path / "business" / "translation-en.v1.json")
    summary = _read_json(tmp_path / "business" / "summary.v1.json")
    manifest = _read_json(tmp_path / "business" / "business-manifest.v1.json")

    assert translation["status"] == "completed"
    assert translation["targetLanguage"] == "en"
    assert translation["applicationPolicy"] == "suggestion-only"
    assert translation["requiresHumanApproval"] is True
    assert translation["segments"][0]["speakerId"] == "speaker-1"
    assert translation["segments"][0]["startMs"] == 0
    assert summary["actionItems"][0]["evidenceSegmentIds"] == ["segment-2"]
    assert summary["applicationPolicy"] == "suggestion-only"
    assert summary["requiresHumanApproval"] is True
    assert summary["keyPoints"][0]["timeRange"] == {
        "startMs": 0,
        "endMs": 1200,
    }
    assert summary["topics"][0]["timeRange"] == {
        "startMs": 0,
        "endMs": 1200,
    }
    assert summary["actionItems"][0]["timeRange"] == {
        "startMs": 1300,
        "endMs": 2500,
    }
    assert summary["promptVersion"] == BUSINESS_PROMPT_VERSION
    assert manifest["rawTranscriptImmutable"] is True
    assert manifest["sourceDocumentHash"] == canonical_json_sha256(original)
    assert manifest["applicationPolicy"] == "suggestion-only"
    assert manifest["requiresHumanApproval"] is True
    assert [Path(path).name for path in manifest["artifacts"]] == [
        "translation-en.v1.json",
        "summary.v1.json",
    ]


def test_same_language_translation_skips_model_without_changing_source(
    tmp_path: Path,
) -> None:
    provider = MappingLocalLLMProvider([])
    config = BusinessProcessingConfig(translation_targets=("zh_CN",))

    BusinessProcessingRunner(provider=provider).run(
        _document(),
        output_directory=tmp_path,
        config=config,
    )

    output = _read_json(tmp_path / "business" / "translation-zh-CN.v1.json")
    assert output["status"] == "skipped-same-language"
    assert output["sourceLanguage"] == "zh-CN"
    assert output["targetLanguage"] == "zh-CN"
    assert output["segments"][1]["text"] == "我会在周五前完成验证。"


def test_checkpoint_reuse_matches_artifact_input_hash(tmp_path: Path) -> None:
    document = _document()
    config = BusinessProcessingConfig(translation_targets=("en",))
    response = _translated_segment(
        "segment-1",
        "speaker-1",
        0,
        1200,
        "我们今天确认发布计划。",
        "Today we confirmed the release plan.",
    )
    response_two = _translated_segment(
        "segment-2",
        "speaker-2",
        1300,
        2500,
        "我会在周五前完成验证。",
        "I will finish validation by Friday.",
    )

    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider([response, response_two])
    ).run(document, output_directory=tmp_path, config=config)

    artifact = _read_json(tmp_path / "business" / "translation-en.v1.json")
    checkpoint = _read_json(
        tmp_path / "business" / "checkpoints" / "translation-en.json"
    )
    assert checkpoint["inputHash"] == artifact["inputHash"]

    cached_artifacts = BusinessProcessingRunner(
        provider=MappingLocalLLMProvider([])
    ).run(document, output_directory=tmp_path, config=config)
    assert cached_artifacts[0].name == "translation-en.v1.json"


def test_checkpoint_rejects_artifact_output_hash_tampering(tmp_path: Path) -> None:
    document = _document()
    config = BusinessProcessingConfig(translation_targets=("en",))
    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider(_valid_translation_responses())
    ).run(document, output_directory=tmp_path, config=config)

    artifact_path = tmp_path / "business" / "translation-en.v1.json"
    artifact = _read_json(artifact_path)
    artifact["segments"][0]["speakerId"] = "speaker-999"
    _write_json(artifact_path, artifact)

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([])
        ).run(document, output_directory=tmp_path, config=config)

    assert error.value.code == "BUSINESS_ARTIFACT_INTEGRITY_FAILED"
    assert "outputHash mismatch" in error.value.message


def test_checkpoint_rejects_rehashed_semantic_tampering(tmp_path: Path) -> None:
    document = _document()
    config = BusinessProcessingConfig(translation_targets=("en",))
    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider(_valid_translation_responses())
    ).run(document, output_directory=tmp_path, config=config)

    artifact_path = tmp_path / "business" / "translation-en.v1.json"
    checkpoint_path = (
        tmp_path / "business" / "checkpoints" / "translation-en.json"
    )
    artifact = _read_json(artifact_path)
    artifact["segments"][0]["speakerId"] = "speaker-999"
    _write_json(artifact_path, artifact)
    checkpoint = _read_json(checkpoint_path)
    checkpoint["outputHash"] = canonical_json_sha256(artifact)
    _write_json(checkpoint_path, checkpoint)

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([])
        ).run(document, output_directory=tmp_path, config=config)

    assert error.value.code == "BUSINESS_ARTIFACT_INTEGRITY_FAILED"
    assert error.value.details["reasonCode"] == "BUSINESS_OUTPUT_INVALID"


def test_checkpoint_rejects_rehashed_public_schema_violation(
    tmp_path: Path,
) -> None:
    document = _document()
    config = BusinessProcessingConfig(translation_targets=("en",))
    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider(_valid_translation_responses())
    ).run(document, output_directory=tmp_path, config=config)

    artifact_path = tmp_path / "business" / "translation-en.v1.json"
    checkpoint_path = (
        tmp_path / "business" / "checkpoints" / "translation-en.json"
    )
    artifact = _read_json(artifact_path)
    del artifact["segments"][0]["text"]
    _write_json(artifact_path, artifact)
    checkpoint = _read_json(checkpoint_path)
    checkpoint["outputHash"] = canonical_json_sha256(artifact)
    _write_json(checkpoint_path, checkpoint)

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([])
        ).run(document, output_directory=tmp_path, config=config)

    assert error.value.code == "BUSINESS_ARTIFACT_INTEGRITY_FAILED"
    assert "public contract" in error.value.details["reason"]


def test_variant_input_hash_separates_derived_artifact_types() -> None:
    document = _document()
    segments = tuple(copy.deepcopy(document["segments"]))

    translation_hash = _variant_input_hash(
        document=document,
        segments=segments,
        variant="translation:en",
    )
    summary_hash = _variant_input_hash(
        document=document,
        segments=segments,
        variant="summary",
    )

    assert translation_hash != summary_hash


def test_translation_cannot_change_speaker_or_timing(tmp_path: Path) -> None:
    invalid = _translated_segment(
        "segment-1",
        "speaker-9",
        50,
        1250,
        "我们今天确认发布计划。",
        "Today we confirmed the release plan.",
    )
    provider = MappingLocalLLMProvider([invalid])

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(provider=provider).run(
            _document(),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("en",)),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"
    assert not (tmp_path / "business" / "translation-en.v1.json").exists()


def test_translation_must_return_the_requested_language(tmp_path: Path) -> None:
    invalid = _translated_segment(
        "segment-1",
        "speaker-1",
        0,
        1200,
        "我们今天确认发布计划。",
        "Aujourd'hui, nous avons confirmé le plan.",
    )
    invalid["language"] = "fr"

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([invalid])
        ).run(
            _document(),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("en",)),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"


def test_model_outputs_reject_malformed_or_request_only_language_tags(
    tmp_path: Path,
) -> None:
    response = _translated_segment(
        "segment-1",
        "speaker-1",
        0,
        1200,
        "我们今天确认发布计划。",
        "Today we confirmed the release plan.",
    )
    config = BusinessProcessingConfig(translation_targets=("en",))

    for invalid_language in ("auto", "en-a"):
        invalid_response = dict(response)
        invalid_response["language"] = invalid_language
        with pytest.raises(WorkerError) as error:
            BusinessProcessingRunner(
                provider=MappingLocalLLMProvider([invalid_response])
            ).run(
                _document(),
                output_directory=tmp_path / invalid_language.replace("-", "_"),
                config=config,
            )
        assert error.value.code == "BUSINESS_OUTPUT_INVALID"


def test_model_language_output_is_canonicalized(tmp_path: Path) -> None:
    responses = [
        {
            **_translated_segment(
                "segment-1",
                "speaker-1",
                0,
                1200,
                "我们今天确认发布计划。",
                "Today we confirmed the release plan.",
            ),
            "language": "EN_us",
        },
        {
            **_translated_segment(
                "segment-2",
                "speaker-2",
                1300,
                2500,
                "我会在周五前完成验证。",
                "I will finish validation by Friday.",
            ),
            "language": "EN_us",
        },
    ]

    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider(responses)
    ).run(
        _document(),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(translation_targets=("en-US",)),
    )

    output = _read_json(tmp_path / "business" / "translation-en-US.v1.json")
    assert output["targetLanguage"] == "en-US"
    assert {segment["language"] for segment in output["segments"]} == {"en-US"}


def test_multilingual_translation_skips_segments_already_in_target_language(
    tmp_path: Path,
) -> None:
    document = _document()
    document["language"] = "mul"
    document["segments"][0]["language"] = "en"
    document["segments"][0]["rawText"] = "The release plan is confirmed."
    document["segments"][0]["normalizedText"] = "The release plan is confirmed."
    document["segments"][0]["displayText"] = "The release plan is confirmed."
    document["segments"][1]["language"] = "zh-CN"
    translated_second = _translated_segment(
        "segment-2",
        "speaker-2",
        1300,
        2500,
        "我会在周五前完成验证。",
        "I will finish validation by Friday.",
    )

    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider([translated_second])
    ).run(
        document,
        output_directory=tmp_path,
        config=BusinessProcessingConfig(translation_targets=("en",)),
    )

    output = _read_json(tmp_path / "business" / "translation-en.v1.json")
    assert output["status"] == "completed"
    assert output["segments"][0]["text"] == "The release plan is confirmed."
    assert output["segments"][0]["language"] == "en"
    assert output["segments"][1]["text"] == "I will finish validation by Friday."


def test_summary_requires_valid_evidence_references(tmp_path: Path) -> None:
    invalid_summary = {
        "executiveSummary": "Summary",
        "keyPoints": [
            _summary_item("Unsupported claim", "missing-segment", 0, 100)
        ],
        "topics": [],
        "actionItems": [],
    }

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([invalid_summary])
        ).run(
            _document(),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(summary=True),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"


def test_summary_retries_when_prose_does_not_match_requested_script(
    tmp_path: Path,
) -> None:
    invalid = {
        "executiveSummary": "Ang buod ay nasa maling wika.",
        "keyPoints": [_summary_item("Maling wika.", "segment-1", 0, 1200)],
        "topics": [],
        "actionItems": [],
    }
    valid = {
        "executiveSummary": "会议摘要使用请求的中文输出。",
        "keyPoints": [_summary_item("确认发布计划。", "segment-1", 0, 1200)],
        "topics": [],
        "actionItems": [],
    }

    class RetryingProvider(MappingLocalLLMProvider):
        business_generation_attempts = 2

        def __init__(self) -> None:
            super().__init__([invalid, valid])
            self.prompts: list[str] = []

        def generate_json(self, **kwargs: object) -> Mapping[str, object]:
            self.prompts.append(str(kwargs["user_prompt"]))
            return super().generate_json(**kwargs)

    provider = RetryingProvider()
    BusinessProcessingRunner(provider=provider).run(
        _document(),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(summary=True, output_locale="zh-CN"),
    )

    output = _read_json(tmp_path / "business" / "summary.v1.json")
    assert output["executiveSummary"] == "会议摘要使用请求的中文输出。"
    assert "retryCorrection=" not in provider.prompts[0]
    assert "retryAttempt=2" in provider.prompts[1]


def test_translation_rejects_short_untranslated_source_copy(tmp_path: Path) -> None:
    source = "kasi"
    copied = {
        "id": "segment-1",
        "speakerId": "speaker-1",
        "startMs": 0,
        "endMs": 1200,
        "sourceTextHash": _source_hash(source),
        "text": source,
        "language": "zh-CN",
    }

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([copied])
        ).run(
            _single_segment_document(source, language="tl"),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"
    assert not (tmp_path / "business" / "translation-zh-CN.v1.json").exists()


def test_translation_retry_adds_fixed_validation_feedback(tmp_path: Path) -> None:
    source = "kasi"
    prompts: list[str] = []

    class RetryingProvider:
        provider_id = "retrying-fixture"
        provider_version = "1"
        network_policy = "loopback-only"
        business_batch_size = 1
        business_translation_segment_attempts = 2

        def generate_json(self, **kwargs: object) -> dict[str, object]:
            prompt = str(kwargs["user_prompt"])
            prompts.append(prompt)
            return {
                "id": "segment-1",
                "speakerId": "speaker-1",
                "startMs": 0,
                "endMs": 1200,
                "sourceTextHash": _source_hash(source),
                "text": "因为" if "retryCorrection=" in prompt else source,
                "language": "zh-CN",
            }

    BusinessProcessingRunner(provider=RetryingProvider()).run(
        _single_segment_document(source, language="tl"),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
    )

    assert len(prompts) == 2
    assert "retryCorrection=" not in prompts[0]
    assert "retryAttempt=2" in prompts[1]
    output = _read_json(tmp_path / "business" / "translation-zh-CN.v1.json")
    assert output["segments"][0]["text"] == "因为"


def test_translation_rejects_punctuation_only_output_for_lexical_source(
    tmp_path: Path,
) -> None:
    source = "na"
    punctuation_only = {
        "id": "segment-1",
        "speakerId": "speaker-1",
        "startMs": 0,
        "endMs": 1200,
        "sourceTextHash": _source_hash(source),
        "text": "。",
        "language": "zh-CN",
    }

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([punctuation_only])
        ).run(
            _single_segment_document(source, language="tl"),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"
    assert not (tmp_path / "business" / "translation-zh-CN.v1.json").exists()


def test_summary_derives_multi_segment_time_range_and_discards_model_range(
    tmp_path: Path,
) -> None:
    summary_response = {
        "executiveSummary": "Summary",
        "keyPoints": [
            {
                "text": "Release and validation are linked.",
                "evidenceSegmentIds": ["segment-2", "segment-1"],
                "timeRange": {"startMs": 999_999, "endMs": 1_000_000},
            }
        ],
        "topics": [],
        "actionItems": [],
    }

    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider([summary_response])
    ).run(
        _document(),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(summary=True),
    )

    output = _read_json(tmp_path / "business" / "summary.v1.json")
    assert output["keyPoints"][0]["timeRange"] == {
        "startMs": 0,
        "endMs": 2500,
    }


def test_prompt_registry_drives_executed_prompt_and_provenance(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class CapturingProvider:
        provider_id = "capturing-fixture"
        provider_version = "1"

        def generate_json(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "executiveSummary": "Summary",
                "keyPoints": [
                    {
                        "text": "Grounded point.",
                        "evidenceSegmentIds": ["segment-1"],
                    }
                ],
                "topics": [],
                "actionItems": [],
            }

    BusinessProcessingRunner(provider=CapturingProvider()).run(
        _document(),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(summary=True),
    )

    output = _read_json(tmp_path / "business" / "summary.v1.json")
    assert output["promptVersion"] == BUSINESS_PROMPT_VERSION
    assert "Do not return or infer timeRange" in str(captured["user_prompt"])
    assert "requested outputLanguage" in str(captured["system_prompt"])
    assert "language label never substitutes" in str(
        captured["system_prompt"]
    )


def test_ollama_capability_style_batching_reduces_translation_calls(
    tmp_path: Path,
) -> None:
    document = _document()
    segments = document["segments"]
    assert isinstance(segments, list)
    first = segments[0]
    second = segments[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    calls: list[dict[str, object]] = []

    class BatchProvider:
        provider_id = "batch-fixture"
        provider_version = "1"
        business_batch_size = 8
        business_batch_character_limit = 7_000

        def generate_json(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return {
                "segments": [
                    _translated_segment(
                        "segment-1",
                        "speaker-1",
                        0,
                        1200,
                        str(first["normalizedText"]),
                        "Today we confirmed the release plan.",
                    ),
                    _translated_segment(
                        "segment-2",
                        "speaker-2",
                        1300,
                        2500,
                        str(second["normalizedText"]),
                        "I will finish validation by Friday.",
                    ),
                ]
            }

    BusinessProcessingRunner(provider=BatchProvider()).run(
        document,
        output_directory=tmp_path,
        config=BusinessProcessingConfig(translation_targets=("en",)),
    )

    assert len(calls) == 1
    assert "segments array" in str(calls[0]["user_prompt"])
    response_schema = calls[0]["response_schema"]
    assert isinstance(response_schema, dict)
    segment_array = response_schema["properties"]["segments"]
    assert segment_array["minItems"] == 2
    assert segment_array["maxItems"] == 2
    output = _read_json(tmp_path / "business" / "translation-en.v1.json")
    assert [item["id"] for item in output["segments"]] == [
        "segment-1",
        "segment-2",
    ]


def test_hierarchical_summary_is_bounded_and_retains_source_evidence(
    tmp_path: Path,
) -> None:
    document = _document()
    segments = document["segments"]
    assert isinstance(segments, list)
    for index in range(3, 5):
        segments.append(
            {
                "id": f"segment-{index}",
                "startMs": (index - 1) * 1300,
                "endMs": (index - 1) * 1300 + 1200,
                "speakerId": f"speaker-{index}",
                "rawText": f"Source statement {index}.",
                "normalizedText": f"Source statement {index}.",
                "displayText": f"Source statement {index}.",
            }
        )
    calls: list[dict[str, object]] = []

    class HierarchicalProvider:
        provider_id = "hierarchical-fixture"
        provider_version = "1"
        business_summary_segment_limit = 2
        business_summary_character_limit = 10_000
        business_summary_reduce_size = 2

        def generate_json(self, **kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            prompt = str(kwargs["user_prompt"])
            if "partialSummaries=" in prompt:
                return {
                    "executiveSummary": "Merged grounded summary.",
                    "keyPoints": [
                        {
                            "text": "Both bounded chunks contributed evidence.",
                            "evidenceSegmentIds": ["segment-1", "segment-3"],
                        }
                    ],
                    "topics": [],
                    "actionItems": [],
                }
            segment_id = "segment-1" if "segment-1" in prompt else "segment-3"
            return {
                "executiveSummary": f"Partial for {segment_id}.",
                "keyPoints": [
                    {
                        "text": f"Grounded point for {segment_id}.",
                        "evidenceSegmentIds": [segment_id],
                    }
                ],
                "topics": [],
                "actionItems": [],
            }

    BusinessProcessingRunner(provider=HierarchicalProvider()).run(
        document,
        output_directory=tmp_path,
        config=BusinessProcessingConfig(summary=True),
    )

    assert len(calls) == 3
    assert sum("partialSummaries=" in str(call["user_prompt"]) for call in calls) == 1
    output = _read_json(tmp_path / "business" / "summary.v1.json")
    assert output["keyPoints"][0]["evidenceSegmentIds"] == [
        "segment-1",
        "segment-3",
    ]
    assert output["keyPoints"][0]["timeRange"] == {
        "startMs": 0,
        "endMs": 3800,
    }


@pytest.mark.parametrize(
    "invalid_item",
    [
        {
            "evidenceSegmentIds": ["segment-1"],
        },
        {
            "text": "Point",
            "evidenceSegmentIds": ["segment-1", "segment-1"],
        },
        {
            "text": "Point",
            "evidenceSegmentIds": ["segment-1"],
            "unexpected": True,
        },
    ],
)
def test_summary_rejects_items_outside_the_public_contract(
    tmp_path: Path,
    invalid_item: dict[str, object],
) -> None:
    response = {
        "executiveSummary": "Summary",
        "keyPoints": [invalid_item],
        "topics": [],
        "actionItems": [],
    }

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([response])
        ).run(
            _document(),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(summary=True),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"
    assert not (tmp_path / "business" / "summary.v1.json").exists()


def test_job_cancellation_is_not_wrapped_as_a_business_provider_failure(
    tmp_path: Path,
) -> None:
    def cancel() -> None:
        raise JobCancelled()

    with pytest.raises(JobCancelled):
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([]),
            cancellation_check=cancel,
        ).run(
            _document(),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(summary=True),
        )


def _reliability_document(
    count: int,
    *,
    source_text: str | None = None,
) -> dict[str, object]:
    languages = ("en", "es", "fr", "de")
    segments: list[dict[str, object]] = []
    for index in range(1, count + 1):
        text = source_text or (
            f"Release planning source segment {index} keeps every immutable "
            "speaker, timestamp, and review-lock field."
        )
        segments.append(
            {
                "id": f"segment-{index}",
                "startMs": (index - 1) * 1_100,
                "endMs": (index - 1) * 1_100 + 1_000,
                "speakerId": f"speaker-{(index - 1) % 9 + 1}",
                "humanLocked": index % 3 == 0,
                "language": languages[(index - 1) % len(languages)],
                "rawText": text,
                "normalizedText": text,
                "displayText": text,
            }
        )
    return {
        "schemaVersion": "2.0.0",
        "documentId": "business-reliability-document",
        "language": "mul" if count > 1 else "en",
        "segments": segments,
    }


def _reliability_prompt_items(
    user_prompt: str,
) -> tuple[list[dict[str, object]], bool]:
    if "\nsegments=" in user_prompt:
        payload = user_prompt.split("\nsegments=", 1)[1]
        values = ast.literal_eval(payload)
        assert isinstance(values, list)
        return [dict(item) for item in values], True
    payload = user_prompt.split("\nsegment=", 1)[1]
    value = ast.literal_eval(payload)
    assert isinstance(value, dict)
    return [dict(value)], False


def _reliability_translation(item: dict[str, object]) -> dict[str, object]:
    return {
        "id": item["id"],
        "speakerId": item["speakerId"],
        "startMs": item["startMs"],
        "endMs": item["endMs"],
        "sourceTextHash": item["sourceTextHash"],
        "text": (
            "\u8fd9\u662f\u5b8c\u6574\u4e14\u53ef\u5ba1\u8ba1\u7684"
            f"\u4e2d\u6587\u7ffb\u8bd1 {item['id']}\u3002"
        ),
        "language": "zh-CN",
    }


class _ReliabilityTranslationProvider:
    provider_id = "reliability-fixture"
    provider_version = "1"
    network_policy = "loopback-only"
    business_translation_segment_attempts = 3

    def __init__(
        self,
        *,
        batch_size: int = 16,
        character_limit: int = 20_000,
        first_batch_mode: str | None = None,
    ) -> None:
        self.business_batch_size = batch_size
        self.business_batch_character_limit = character_limit
        self.first_batch_mode = first_batch_mode
        self.batch_calls: list[tuple[str, ...]] = []
        self.single_calls: list[str] = []

    def generate_json(self, **kwargs: object) -> dict[str, object]:
        items, is_batch = _reliability_prompt_items(
            str(kwargs["user_prompt"])
        )
        outputs = [_reliability_translation(item) for item in items]
        if not is_batch:
            self.single_calls.append(str(items[0]["id"]))
            return outputs[0]

        self.batch_calls.append(tuple(str(item["id"]) for item in items))
        if len(self.batch_calls) == 1:
            if self.first_batch_mode == "omit":
                return {
                    "segments": [
                        output
                        for output in outputs
                        if output["id"] != "segment-3"
                    ]
                }
            if self.first_batch_mode == "duplicate":
                return {
                    "segments": [
                        outputs[0],
                        outputs[0],
                        *outputs[2:],
                    ]
                }
            if self.first_batch_mode == "out-of-order":
                return {
                    "segments": [
                        outputs[1],
                        outputs[0],
                        *outputs[2:],
                    ]
                }
        return {"segments": outputs}


def test_large_multilingual_translation_is_complete_and_metadata_lossless(
    tmp_path: Path,
) -> None:
    document = _reliability_document(96)
    original = copy.deepcopy(document)
    provider = _ReliabilityTranslationProvider(batch_size=13)

    BusinessProcessingRunner(provider=provider).run(
        document,
        output_directory=tmp_path,
        config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
    )

    artifact = _read_json(
        tmp_path / "business" / "translation-zh-CN.v1.json"
    )
    source_segments = original["segments"]
    output_segments = artifact["segments"]
    assert isinstance(source_segments, list)
    assert isinstance(output_segments, list)
    assert len(output_segments) == len(source_segments) == 96
    assert document == original
    assert [
        (
            output["id"],
            output["startMs"],
            output["endMs"],
            output["speakerId"],
            output["humanLocked"],
        )
        for output in output_segments
    ] == [
        (
            source["id"],
            source["startMs"],
            source["endMs"],
            source["speakerId"],
            source["humanLocked"],
        )
        for source in source_segments
    ]
    assert provider.batch_calls
    assert not provider.single_calls


def test_partial_translation_omission_retries_only_the_missing_segment(
    tmp_path: Path,
) -> None:
    provider = _ReliabilityTranslationProvider(
        batch_size=8,
        first_batch_mode="omit",
    )
    BusinessProcessingRunner(provider=provider).run(
        _reliability_document(8),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
    )

    artifact = _read_json(
        tmp_path / "business" / "translation-zh-CN.v1.json"
    )
    assert [item["id"] for item in artifact["segments"]] == [
        f"segment-{index}" for index in range(1, 9)
    ]
    assert provider.single_calls == ["segment-3"]
    progress = _read_json(
        tmp_path
        / "business"
        / "checkpoints"
        / "translation-zh-CN.progress.json"
    )
    missing = next(
        item for item in progress["segments"] if item["id"] == "segment-3"
    )
    assert missing["status"] == "translated"
    assert missing["errors"][0]["phase"] == "batch"
    assert "omitted" in missing["errors"][0]["message"]


@pytest.mark.parametrize(
    ("mode", "message_fragment"),
    [
        ("duplicate", "duplicate segment IDs"),
        ("out-of-order", "out of source order"),
    ],
)
def test_invalid_batch_identity_or_order_is_rejected_audited_and_merged_by_source(
    tmp_path: Path,
    mode: str,
    message_fragment: str,
) -> None:
    provider = _ReliabilityTranslationProvider(
        batch_size=4,
        first_batch_mode=mode,
    )
    BusinessProcessingRunner(provider=provider).run(
        _reliability_document(4),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
    )

    artifact = _read_json(
        tmp_path / "business" / "translation-zh-CN.v1.json"
    )
    assert [item["id"] for item in artifact["segments"]] == [
        "segment-1",
        "segment-2",
        "segment-3",
        "segment-4",
    ]
    assert provider.single_calls == [
        "segment-1",
        "segment-2",
        "segment-3",
        "segment-4",
    ]
    progress = _read_json(
        tmp_path
        / "business"
        / "checkpoints"
        / "translation-zh-CN.progress.json"
    )
    assert all(
        any(
            message_fragment in error["message"]
            for error in item["errors"]
        )
        for item in progress["segments"]
    )


def test_business_character_boundary_accepts_exact_segment_and_rejects_overflow(
    tmp_path: Path,
) -> None:
    exact_provider = _ReliabilityTranslationProvider(
        batch_size=1,
        character_limit=256,
    )
    BusinessProcessingRunner(provider=exact_provider).run(
        _reliability_document(1, source_text="a" * 256),
        output_directory=tmp_path / "exact",
        config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
    )
    assert exact_provider.single_calls == ["segment-1"]

    overflow_provider = _ReliabilityTranslationProvider(
        batch_size=1,
        character_limit=256,
    )
    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(provider=overflow_provider).run(
            _reliability_document(1, source_text="a" * 257),
            output_directory=tmp_path / "overflow",
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )

    assert error.value.code == "BUSINESS_CONTEXT_LIMIT_EXCEEDED"
    assert error.value.details == {
        "segmentId": "segment-1",
        "segmentCharacters": 257,
        "maximumCharacters": 256,
    }
    assert not overflow_provider.single_calls
    assert not (
        tmp_path
        / "overflow"
        / "business"
        / "translation-zh-CN.v1.json"
    ).exists()


def test_translation_preserves_human_lock_and_raw_document(
    tmp_path: Path,
) -> None:
    source = "The desktop app should accept 3 PDF files."
    document = _single_segment_document(source)
    segment = document["segments"][0]
    assert isinstance(segment, dict)
    segment["humanLocked"] = True
    original = copy.deepcopy(document)
    translation = {
        "id": "segment-1",
        "speakerId": "speaker-1",
        "startMs": 0,
        "endMs": 1_200,
        "sourceTextHash": _source_hash(source),
        "text": "\u684c\u9762\u5e94\u7528\u5e94\u652f\u6301\u63a5\u6536"
        " 3 \u4e2a PDF \u6587\u4ef6\u3002",
        "language": "zh-CN",
    }
    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider([translation])
    ).run(
        document,
        output_directory=tmp_path,
        config=BusinessProcessingConfig(
            translation_targets=("zh-CN",),
        ),
    )

    translated = _read_json(
        tmp_path / "business" / "translation-zh-CN.v1.json"
    )
    assert document == original
    assert translated["segments"][0]["humanLocked"] is True


def test_model_cannot_change_the_immutable_human_lock(
    tmp_path: Path,
) -> None:
    document = _reliability_document(1)
    segments = document["segments"]
    assert isinstance(segments, list)
    segments[0]["humanLocked"] = True
    provider = _ReliabilityTranslationProvider(batch_size=1)
    original_generate = provider.generate_json

    def changed_lock(**kwargs: object) -> dict[str, object]:
        output = original_generate(**kwargs)
        output["humanLocked"] = False
        return output

    provider.generate_json = changed_lock  # type: ignore[method-assign]
    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(provider=provider).run(
            document,
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"
    assert "human lock" in error.value.details["cause"]["message"]
    assert not (
        tmp_path / "business" / "translation-zh-CN.v1.json"
    ).exists()


def test_duplicate_source_segment_ids_fail_before_model_invocation(
    tmp_path: Path,
) -> None:
    document = _reliability_document(2)
    segments = document["segments"]
    assert isinstance(segments, list)
    segments[1]["id"] = "segment-1"
    provider = _ReliabilityTranslationProvider()

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(provider=provider).run(
            document,
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )

    assert error.value.code == "BUSINESS_INPUT_INVALID"
    assert error.value.details["duplicateSegmentIds"] == ["segment-1"]
    assert not provider.batch_calls
    assert not provider.single_calls


@pytest.mark.parametrize(
    ("network_policy", "endpoint"),
    [
        ("remote-allowed", None),
        ("loopback-only", "https://example.com"),
    ],
)
def test_business_processing_rejects_non_loopback_provider_before_any_call(
    tmp_path: Path,
    network_policy: str,
    endpoint: str | None,
) -> None:
    provider = _ReliabilityTranslationProvider()
    provider.network_policy = network_policy
    if endpoint is not None:
        provider.endpoint = endpoint

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(provider=provider).run(
            _reliability_document(1),
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )

    assert error.value.code == "BUSINESS_PROVIDER_POLICY_INVALID"
    assert error.value.details["reason"]
    assert not provider.batch_calls
    assert not provider.single_calls


def test_source_mutation_during_model_call_fails_before_artifact_publication(
    tmp_path: Path,
) -> None:
    document = _reliability_document(1)

    class MutatingProvider(_ReliabilityTranslationProvider):
        def generate_json(self, **kwargs: object) -> dict[str, object]:
            segments = document["segments"]
            assert isinstance(segments, list)
            segments[0]["rawText"] = "tampered"
            return super().generate_json(**kwargs)

    provider = MutatingProvider(batch_size=1)
    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(provider=provider).run(
            document,
            output_directory=tmp_path,
            config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
        )

    assert error.value.code == "BUSINESS_SOURCE_MUTATED"
    assert error.value.details["expectedDocumentHash"]
    assert error.value.details["actualDocumentHash"]
    assert not (
        tmp_path / "business" / "translation-zh-CN.v1.json"
    ).exists()
