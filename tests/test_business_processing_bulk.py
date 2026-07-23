from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from backend.business_processing import (
    BusinessProcessingConfig,
    BusinessProcessingRunner,
)
from backend.errors import WorkerError
from backend.local_llm import LocalLLMConfig, LocalLLMError, OllamaLocalProvider


def _bulk_document(
    count: int,
    *,
    chinese_segment_ids: set[str] | None = None,
) -> dict[str, Any]:
    chinese_ids = chinese_segment_ids or set()
    segments: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        segment_id = f"segment-{index}"
        if segment_id in chinese_ids:
            text = f"这是第 {index} 段已经是中文的源文本。"
            language = "zh-CN"
        else:
            text = (
                f"This is English source segment number {index} about the release "
                "plan, validation status, and the next concrete action."
            )
            language = "en"
        segments.append(
            {
                "id": segment_id,
                "startMs": (index - 1) * 1_250,
                "endMs": (index - 1) * 1_250 + 1_100,
                "speakerId": f"speaker-{(index - 1) % 7 + 1}",
                "language": language,
                "rawText": text,
                "normalizedText": text,
                "displayText": text,
            }
        )
    return {
        "schemaVersion": "2.0.0",
        "documentId": "bulk-translation-document",
        "language": "mul" if chinese_ids else "en",
        "segments": segments,
    }


def _parse_prompt_items(user_prompt: str) -> tuple[list[dict[str, Any]], bool]:
    if "\nsegments=" in user_prompt:
        payload = user_prompt.split("\nsegments=", 1)[1]
        values = ast.literal_eval(payload)
        assert isinstance(values, list)
        return [dict(item) for item in values], True
    payload = user_prompt.split("\nsegment=", 1)[1]
    value = ast.literal_eval(payload)
    assert isinstance(value, dict)
    return [dict(value)], False


def _translated_segment(
    item: Mapping[str, Any],
    *,
    text: str | None = None,
) -> dict[str, Any]:
    index = int(str(item["id"]).split("-")[-1])
    return {
        "id": item["id"],
        "speakerId": item["speakerId"],
        "startMs": item["startMs"],
        "endMs": item["endMs"],
        "sourceTextHash": item["sourceTextHash"],
        "text": text or f"这是第 {index} 段的完整中文翻译，包含发布计划、验证状态和下一步行动。",
        "language": "zh-CN",
    }


BatchHook = Callable[
    [int, list[dict[str, Any]], list[dict[str, Any]]],
    Mapping[str, Any],
]
SingleHook = Callable[[str, int, dict[str, Any]], Mapping[str, Any]]


class ScriptedBulkProvider:
    provider_id = "bulk-fixture"
    provider_version = "1"
    business_batch_size = 20
    business_batch_character_limit = 100_000
    business_translation_segment_attempts = 3

    def __init__(
        self,
        *,
        batch_hook: BatchHook | None = None,
        single_hook: SingleHook | None = None,
    ) -> None:
        self.batch_hook = batch_hook
        self.single_hook = single_hook
        self.batch_calls: list[tuple[str, ...]] = []
        self.single_calls: list[str] = []
        self.calls_by_segment: Counter[str] = Counter()

    def generate_json(self, **kwargs: object) -> Mapping[str, Any]:
        items, is_batch = _parse_prompt_items(str(kwargs["user_prompt"]))
        for item in items:
            self.calls_by_segment[str(item["id"])] += 1
        if is_batch:
            self.batch_calls.append(tuple(str(item["id"]) for item in items))
            outputs = [_translated_segment(item) for item in items]
            if self.batch_hook is not None:
                return self.batch_hook(len(self.batch_calls), items, outputs)
            return {"segments": outputs}
        item = items[0]
        segment_id = str(item["id"])
        self.single_calls.append(segment_id)
        attempt = sum(1 for value in self.single_calls if value == segment_id)
        if self.single_hook is not None:
            return self.single_hook(segment_id, attempt, item)
        return _translated_segment(item)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _translation_path(root: Path) -> Path:
    return root / "business" / "translation-zh-CN.v1.json"


def _progress_path(root: Path) -> Path:
    return (
        root
        / "business"
        / "checkpoints"
        / "translation-zh-CN.progress.json"
    )


def _run_translation(
    provider: ScriptedBulkProvider,
    document: Mapping[str, Any],
    root: Path,
) -> tuple[Path, ...]:
    return BusinessProcessingRunner(provider=provider).run(
        document,
        output_directory=root,
        config=BusinessProcessingConfig(translation_targets=("zh-CN",)),
    )


def test_bulk_120_english_segments_translate_completely_in_bounded_batches(
    tmp_path: Path,
) -> None:
    document = _bulk_document(120)
    immutable_source = copy.deepcopy(document)
    provider = ScriptedBulkProvider()

    artifacts = _run_translation(provider, document, tmp_path)

    assert document == immutable_source
    assert [path.name for path in artifacts] == [
        "translation-zh-CN.v1.json",
        "business-manifest.v1.json",
    ]
    assert len(provider.batch_calls) == 6
    assert not provider.single_calls
    assert all(
        len(call) <= provider.business_batch_size
        for call in provider.batch_calls
    )
    assert set(provider.calls_by_segment.values()) == {1}

    output = _read_json(_translation_path(tmp_path))
    assert output["status"] == "completed"
    assert len(output["segments"]) == 120
    for source, translated in zip(
        document["segments"],
        output["segments"],
        strict=True,
    ):
        assert translated["id"] == source["id"]
        assert translated["speakerId"] == source["speakerId"]
        assert translated["startMs"] == source["startMs"]
        assert translated["endMs"] == source["endMs"]
        assert translated["language"] == "zh-CN"
        assert translated["text"] != source["normalizedText"]
        assert "中文翻译" in translated["text"]

    manifest = _read_json(
        tmp_path / "business" / "business-manifest.v1.json"
    )
    completeness = manifest["completeness"]["translations"]["zh-CN"]
    assert completeness == {
        "total": 120,
        "translated": 120,
        "copied": 0,
        "skipped": 0,
        "failed": 0,
        "pending": 0,
        "completed": 120,
        "complete": True,
    }


def test_ollama_translation_batch_limit_is_bounded_by_output_budget() -> None:
    small = OllamaLocalProvider(
        LocalLLMConfig(context_tokens=4_096, output_tokens=1_024)
    )
    large = OllamaLocalProvider(
        LocalLLMConfig(context_tokens=8_192, output_tokens=4_096)
    )

    assert small.business_batch_character_limit == 512
    assert large.business_batch_character_limit == 2_048
    assert small.business_translation_segment_attempts == 3


def test_bulk_mixed_language_legally_copies_only_target_language_segments(
    tmp_path: Path,
) -> None:
    chinese_ids = {f"segment-{index}" for index in range(1, 111, 10)}
    document = _bulk_document(110, chinese_segment_ids=chinese_ids)
    provider = ScriptedBulkProvider()

    _run_translation(provider, document, tmp_path)

    assert not (chinese_ids & set(provider.calls_by_segment))
    output = _read_json(_translation_path(tmp_path))
    output_by_id = {item["id"]: item for item in output["segments"]}
    source_by_id = {item["id"]: item for item in document["segments"]}
    for segment_id in chinese_ids:
        assert (
            output_by_id[segment_id]["text"]
            == source_by_id[segment_id]["normalizedText"]
        )
        assert output_by_id[segment_id]["language"] == "zh-CN"

    manifest = _read_json(
        tmp_path / "business" / "business-manifest.v1.json"
    )
    completeness = manifest["completeness"]["translations"]["zh-CN"]
    assert completeness["total"] == 110
    assert completeness["translated"] == 99
    assert completeness["copied"] == 11
    assert completeness["skipped"] == 0
    assert completeness["failed"] == 0
    assert completeness["complete"] is True


def test_partial_empty_and_source_language_batch_items_retry_per_segment_only(
    tmp_path: Path,
) -> None:
    fault_ids = {"segment-3", "segment-5", "segment-7"}

    def partial_first_batch(
        batch_number: int,
        items: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        if batch_number != 1:
            return {"segments": outputs}
        mutated: list[dict[str, Any]] = []
        for item, output in zip(items, outputs, strict=True):
            segment_id = str(item["id"])
            if segment_id == "segment-3":
                continue
            if segment_id == "segment-5":
                output = {**output, "text": ""}
            if segment_id == "segment-7":
                output = {**output, "text": str(item["sourceText"])}
            mutated.append(output)
        return {"segments": mutated}

    document = _bulk_document(105)
    provider = ScriptedBulkProvider(batch_hook=partial_first_batch)

    _run_translation(provider, document, tmp_path)

    assert set(provider.single_calls) == fault_ids
    assert len(provider.single_calls) == len(fault_ids)
    for segment_id in fault_ids:
        assert provider.calls_by_segment[segment_id] == 2
    for index in range(1, 21):
        segment_id = f"segment-{index}"
        if segment_id not in fault_ids:
            assert provider.calls_by_segment[segment_id] == 1
    output = _read_json(_translation_path(tmp_path))
    assert len(output["segments"]) == 105
    assert all("中文翻译" in item["text"] for item in output["segments"])


def test_whole_batch_provider_failure_degrades_to_bounded_single_segment_calls(
    tmp_path: Path,
) -> None:
    def fail_first_batch(
        batch_number: int,
        items: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        del items, outputs
        if batch_number == 1:
            raise LocalLLMError("simulated truncated batch response")
        raise AssertionError("the first failed batch should recover by segment")

    document = _bulk_document(20)
    provider = ScriptedBulkProvider(batch_hook=fail_first_batch)

    _run_translation(provider, document, tmp_path)

    assert len(provider.batch_calls) == 1
    assert len(provider.single_calls) == 20
    assert set(provider.calls_by_segment.values()) == {2}
    manifest = _read_json(
        tmp_path / "business" / "business-manifest.v1.json"
    )
    assert (
        manifest["completeness"]["translations"]["zh-CN"]["translated"]
        == 20
    )


def test_retry_exhaustion_persists_failure_and_never_claims_completion(
    tmp_path: Path,
) -> None:
    def poisoned_batch(
        batch_number: int,
        items: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        assert batch_number == 1
        poisoned = [
            (
                {**output, "text": str(item["sourceText"])}
                if item["id"] == "segment-5"
                else output
            )
            for item, output in zip(items, outputs, strict=True)
        ]
        return {"segments": poisoned}

    def poisoned_single(
        segment_id: str,
        attempt: int,
        item: dict[str, Any],
    ) -> Mapping[str, Any]:
        del attempt
        assert segment_id == "segment-5"
        return _translated_segment(item, text=str(item["sourceText"]))

    document = _bulk_document(120)
    provider = ScriptedBulkProvider(
        batch_hook=poisoned_batch,
        single_hook=poisoned_single,
    )

    with pytest.raises(WorkerError) as raised:
        _run_translation(provider, document, tmp_path)

    assert raised.value.code == "BUSINESS_OUTPUT_INVALID"
    assert "incomplete" in raised.value.message
    assert raised.value.details["segmentId"] == "segment-5"
    assert raised.value.details["attemptsThisRun"] == 3
    assert raised.value.details["completeness"] == {
        "total": 120,
        "translated": 19,
        "copied": 0,
        "skipped": 0,
        "failed": 1,
        "pending": 100,
        "completed": 19,
        "complete": False,
    }
    assert not _translation_path(tmp_path).exists()
    assert not (
        tmp_path / "business" / "business-manifest.v1.json"
    ).exists()

    progress = _read_json(_progress_path(tmp_path))
    assert progress["status"] == "failed"
    assert progress["completeness"] == raised.value.details["completeness"]
    segment_five = progress["segments"][4]
    assert segment_five["status"] == "failed"
    assert segment_five["attempts"] == 3
    assert segment_five["output"] is None
    assert segment_five["lastError"]["code"] == "BUSINESS_OUTPUT_INVALID"


def test_resume_reuses_completed_segments_and_only_processes_unresolved_work(
    tmp_path: Path,
) -> None:
    def first_run_batch(
        batch_number: int,
        items: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        assert batch_number == 1
        return {
            "segments": [
                (
                    {**output, "text": str(item["sourceText"])}
                    if item["id"] == "segment-5"
                    else output
                )
                for item, output in zip(items, outputs, strict=True)
            ]
        }

    def first_run_single(
        segment_id: str,
        attempt: int,
        item: dict[str, Any],
    ) -> Mapping[str, Any]:
        del attempt
        assert segment_id == "segment-5"
        return _translated_segment(item, text=str(item["sourceText"]))

    document = _bulk_document(120)
    first_provider = ScriptedBulkProvider(
        batch_hook=first_run_batch,
        single_hook=first_run_single,
    )
    with pytest.raises(WorkerError):
        _run_translation(first_provider, document, tmp_path)

    completed_first_run = {
        f"segment-{index}"
        for index in range(1, 21)
        if index != 5
    }
    second_provider = ScriptedBulkProvider()

    _run_translation(second_provider, document, tmp_path)

    assert not (completed_first_run & set(second_provider.calls_by_segment))
    assert second_provider.calls_by_segment["segment-5"] == 1
    assert set(second_provider.calls_by_segment) == (
        {"segment-5"}
        | {f"segment-{index}" for index in range(21, 121)}
    )
    assert all(
        count == 1 for count in second_provider.calls_by_segment.values()
    )

    output = _read_json(_translation_path(tmp_path))
    assert len(output["segments"]) == 120
    progress = _read_json(_progress_path(tmp_path))
    assert progress["status"] == "completed"
    assert progress["completeness"]["complete"] is True
    assert progress["segments"][4]["attempts"] == 4
    assert progress["segments"][0]["attempts"] == 1


def test_resume_checkpoint_tampering_fails_closed_before_model_reuse(
    tmp_path: Path,
) -> None:
    def poisoned_batch(
        batch_number: int,
        items: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
    ) -> Mapping[str, Any]:
        assert batch_number == 1
        return {
            "segments": [
                (
                    {**output, "text": str(item["sourceText"])}
                    if item["id"] == "segment-5"
                    else output
                )
                for item, output in zip(items, outputs, strict=True)
            ]
        }

    document = _bulk_document(101)
    first_provider = ScriptedBulkProvider(
        batch_hook=poisoned_batch,
        single_hook=lambda _segment_id, _attempt, item: _translated_segment(
            item,
            text=str(item["sourceText"]),
        ),
    )
    with pytest.raises(WorkerError):
        _run_translation(first_provider, document, tmp_path)

    progress_path = _progress_path(tmp_path)
    progress = _read_json(progress_path)
    progress["segments"][0]["output"]["speakerId"] = "speaker-999"
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    second_provider = ScriptedBulkProvider()

    with pytest.raises(WorkerError) as raised:
        _run_translation(second_provider, document, tmp_path)

    assert raised.value.code == "BUSINESS_CHECKPOINT_INVALID"
    assert not second_provider.calls_by_segment
    assert not _translation_path(tmp_path).exists()
