from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from backend.business_processing import (
    BUSINESS_PROMPT_VERSION,
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


def _polished_segment(
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
        "language": "zh-CN",
        "diffReason": "clarified punctuation",
    }


def _polished_segment_for_language(
    source_text: str,
    text: str,
    *,
    language: str,
) -> dict[str, object]:
    return {
        "id": "segment-1",
        "speakerId": "speaker-1",
        "startMs": 0,
        "endMs": 1_200,
        "sourceTextHash": _source_hash(source_text),
        "text": text,
        "language": language,
        "diffReason": "Conservative readability suggestion.",
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
        polish=True,
        summary=True,
        model=" qwen3.5:4b ",
        output_locale="en-US",
    )

    assert config.translation_targets == ("zh-CN", "en")
    assert config.model == "qwen3.5:4b"
    assert config.output_locale == "en-US"
    assert config.as_dict()["schemaVersion"] == BUSINESS_SCHEMA_VERSION
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


def test_runner_creates_translation_polish_summary_and_manifest(
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
            _polished_segment(
                "segment-1",
                "speaker-1",
                0,
                1200,
                "我们今天确认发布计划。",
                "我们今天确认发布计划。",
            ),
            _polished_segment(
                "segment-2",
                "speaker-2",
                1300,
                2500,
                "我会在周五前完成验证。",
                "我会在周五前完成验证。",
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
        polish=True,
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
        "polished-transcript.v1.json",
        "summary.v1.json",
        "business-manifest.v1.json",
    ]
    assert document == original

    translation = _read_json(tmp_path / "business" / "translation-en.v1.json")
    polish = _read_json(tmp_path / "business" / "polished-transcript.v1.json")
    summary = _read_json(tmp_path / "business" / "summary.v1.json")
    manifest = _read_json(tmp_path / "business" / "business-manifest.v1.json")

    assert translation["status"] == "completed"
    assert translation["targetLanguage"] == "en"
    assert translation["segments"][0]["speakerId"] == "speaker-1"
    assert translation["segments"][0]["startMs"] == 0
    assert polish["language"] == "zh-CN"
    assert polish["applicationPolicy"] == "suggestion-only"
    assert polish["requiresHumanApproval"] is True
    assert polish["diff"] == []
    assert summary["actionItems"][0]["evidenceSegmentIds"] == ["segment-2"]
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
    assert len(manifest["artifacts"]) == 3


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
    polish_hash = _variant_input_hash(
        document=document,
        segments=segments,
        variant="polish:source",
    )

    assert translation_hash != polish_hash


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


@pytest.mark.parametrize("task", ["translation", "polish"])
def test_model_outputs_reject_malformed_or_request_only_language_tags(
    tmp_path: Path,
    task: str,
) -> None:
    if task == "translation":
        response = _translated_segment(
            "segment-1",
            "speaker-1",
            0,
            1200,
            "我们今天确认发布计划。",
            "Today we confirmed the release plan.",
        )
        config = BusinessProcessingConfig(translation_targets=("en",))
    else:
        response = _polished_segment(
            "segment-1",
            "speaker-1",
            0,
            1200,
            "我们今天确认发布计划。",
            "我们今天确认发布计划。",
        )
        config = BusinessProcessingConfig(polish=True)

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
    assert captured["system_prompt"] == (
        "You are an offline evidence-grounded meeting summarizer. "
        "Output strict JSON only."
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


def test_polish_rejects_empty_diff_reason_before_persistence(
    tmp_path: Path,
) -> None:
    document = _document()
    segments = document["segments"]
    assert isinstance(segments, list)
    first = segments[0]
    assert isinstance(first, dict)
    source_text = str(first["normalizedText"])
    response = _polished_segment(
        "segment-1",
        "speaker-1",
        0,
        1200,
        source_text,
        source_text,
    )
    response["diffReason"] = ""

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([response])
        ).run(
            document,
            output_directory=tmp_path,
            config=BusinessProcessingConfig(polish=True),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"
    assert not (
        tmp_path / "business" / "polished-transcript.v1.json"
    ).exists()


@pytest.mark.parametrize(
    ("source_text", "polished_text", "guard"),
    [
        (
            "The desktop app should accept 3 PDF files.",
            "The desktop app must accept 3 PDF files.",
            "modality",
        ),
        (
            "The desktop app should accept 3 PDF files.",
            "The desktop app should accept 4 PDF files.",
            "protected-literals",
        ),
        (
            "The desktop app should not overwrite report.pdf.",
            "The desktop app should overwrite report.pdf.",
            "negation",
        ),
        (
            "Should the desktop app accept 3 PDF files?",
            "The desktop app should accept 3 PDF files.",
            "question-intent",
        ),
        (
            "The desktop app should accept 3 PDF files.",
            "桌面应用应该接受 3 个 PDF 文件。",
            "source-script",
        ),
    ],
)
def test_polish_rejects_high_confidence_semantic_drift(
    tmp_path: Path,
    source_text: str,
    polished_text: str,
    guard: str,
) -> None:
    document = _single_segment_document(source_text)
    response = _polished_segment_for_language(
        source_text,
        polished_text,
        language="en",
    )

    with pytest.raises(WorkerError) as error:
        BusinessProcessingRunner(
            provider=MappingLocalLLMProvider([response])
        ).run(
            document,
            output_directory=tmp_path,
            config=BusinessProcessingConfig(polish=True),
        )

    assert error.value.code == "BUSINESS_OUTPUT_INVALID"
    assert error.value.details["guard"] == guard
    assert not (
        tmp_path / "business" / "polished-transcript.v1.json"
    ).exists()


def test_polish_accepts_conservative_rewrite_and_marks_it_suggestion_only(
    tmp_path: Path,
) -> None:
    source_text = "Should the desktop app accept 3 PDF files by drag and drop?"
    polished_text = "Should the desktop app accept 3 PDF files via drag-and-drop?"
    response = _polished_segment_for_language(
        source_text,
        polished_text,
        language="en",
    )

    BusinessProcessingRunner(
        provider=MappingLocalLLMProvider([response])
    ).run(
        _single_segment_document(source_text),
        output_directory=tmp_path,
        config=BusinessProcessingConfig(polish=True),
    )

    artifact = _read_json(
        tmp_path / "business" / "polished-transcript.v1.json"
    )
    assert artifact["applicationPolicy"] == "suggestion-only"
    assert artifact["requiresHumanApproval"] is True
    assert artifact["segments"][0]["text"] == polished_text
    assert artifact["diff"][0]["before"] == source_text
    assert artifact["diff"][0]["after"] == polished_text


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
