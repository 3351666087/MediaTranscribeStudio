from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ENV_PATHS = {
    "final": "MTS_BENCH_FINAL_TRANSCRIPT",
    "pre": "MTS_BENCH_PRE_TRANSCRIPT",
    "turn_corrections": "MTS_BENCH_TURN_CORRECTIONS",
    "sentence_decisions": "MTS_BENCH_SENTENCE_DECISIONS",
}
_TEXT_KEY_RE = re.compile(
    r"(?:^|_)(?:text|transcript|utterance|sentence|content|caption|段落|文本|原文|正文)(?:$|_)",
    re.IGNORECASE,
)
_SPEAKER_KEY_RE = re.compile(
    (
        r"^(?:speaker(?:_id|_name|_label)?|role(?:_id|_name)?|"
        r"person(?:_id|_name)?|name|alias|participant(?:_id|_name)?|"
        r"说话人|角色|姓名|人物)$"
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Example:
    sample_id: str
    source_turn_id: str
    split: str
    kind: str
    start_ms: int
    end_ms: int
    source_text: str
    target_text: str
    previous_text: str
    next_text: str
    input_risk_flags: tuple[str, ...]


@dataclass(frozen=True)
class DatasetBundle:
    examples: tuple[Example, ...]
    source_hashes: dict[str, str]
    combined_fingerprint: str
    duration_ms: int
    split_boundary_ms: int
    counts: dict[str, int]
    source_texts_for_privacy_check: tuple[str, ...]
    speaker_truth_for_privacy_check: tuple[str, ...]


def load_from_environment(
    *,
    dev_ratio: float,
    max_dev: int | None,
    max_heldout: int | None,
    max_safety: int | None,
) -> DatasetBundle:
    paths: dict[str, Path] = {}
    for logical_name, environment_name in ENV_PATHS.items():
        raw = os.environ.get(environment_name)
        if not raw:
            raise RuntimeError(f"missing_environment_variable:{environment_name}")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"source_file_missing:{environment_name}")
        paths[logical_name] = path

    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    source_hashes = {
        name: _sha256_file(path)
        for name, path in paths.items()
    }
    combined_fingerprint = hashlib.sha256(
        "|".join(f"{name}:{source_hashes[name]}" for name in sorted(source_hashes)).encode("ascii")
    ).hexdigest()
    return build_dataset(
        payloads["pre"],
        payloads["final"],
        payloads["turn_corrections"],
        payloads["sentence_decisions"],
        source_hashes=source_hashes,
        combined_fingerprint=combined_fingerprint,
        dev_ratio=dev_ratio,
        max_dev=max_dev,
        max_heldout=max_heldout,
        max_safety=max_safety,
    )


def build_dataset(
    pre_payload: Mapping[str, Any],
    final_payload: Mapping[str, Any],
    turn_corrections_payload: Mapping[str, Any],
    sentence_decisions_payload: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, str],
    combined_fingerprint: str,
    dev_ratio: float,
    max_dev: int | None,
    max_heldout: int | None,
    max_safety: int | None,
) -> DatasetBundle:
    if not 0.5 <= dev_ratio <= 0.9:
        raise ValueError("dev_ratio_out_of_range")

    pre_turns = _require_turns(pre_payload, "pre")
    final_turns = _require_turns(final_payload, "final")
    duration_ms = int(pre_payload.get("duration_ms") or max(turn["end_ms"] for turn in pre_turns))
    split_boundary_ms = int(duration_ms * dev_ratio)

    split_ids = {str(value) for value in sentence_decisions_payload.get("split_turns", {})}
    override_ids = {
        str(value)
        for value in sentence_decisions_payload.get("whole_turn_overrides", {})
    }
    corrected_turn_ids = {
        str(value)
        for value in turn_corrections_payload.get("turn_text", {})
    }

    final_by_original: dict[str, list[Mapping[str, Any]]] = {}
    for turn in final_turns:
        original_id = str(turn.get("original_turn_id", turn.get("turn_id")))
        final_by_original.setdefault(original_id, []).append(turn)

    all_private_texts: list[str] = []
    all_private_speaker_values: list[str] = []
    for payload in (
        pre_payload,
        final_payload,
        turn_corrections_payload,
        sentence_decisions_payload,
    ):
        text_values, speaker_values = collect_private_values(payload)
        all_private_texts.extend(text_values)
        all_private_speaker_values.extend(speaker_values)

    pre_by_id = {str(turn["turn_id"]): turn for turn in pre_turns}
    ordered_ids = [str(turn["turn_id"]) for turn in pre_turns]
    semantic: list[Example] = []
    safety: list[Example] = []

    for position, turn_id in enumerate(ordered_ids):
        pre_turn = pre_by_id[turn_id]
        source_text = str(pre_turn.get("final_text", "")).strip()
        if not source_text:
            continue
        previous_text = _neighbor_text(pre_turns, position - 1)
        next_text = _neighbor_text(pre_turns, position + 1)
        start_ms = int(pre_turn["start_ms"])
        end_ms = int(pre_turn["end_ms"])
        split_name = "dev" if (start_ms + end_ms) / 2 < split_boundary_ms else "heldout"
        aligned = final_by_original.get(turn_id, [])

        if turn_id in split_ids:
            safety.append(
                _make_example(
                    combined_fingerprint,
                    turn_id,
                    split_name,
                    "overlap_safety",
                    start_ms,
                    end_ms,
                    source_text,
                    source_text,
                    previous_text,
                    next_text,
                    ("overlap_candidate",),
                )
            )
            continue

        if turn_id in override_ids:
            safety.append(
                _make_example(
                    combined_fingerprint,
                    turn_id,
                    split_name,
                    "speaker_safety",
                    start_ms,
                    end_ms,
                    source_text,
                    source_text,
                    previous_text,
                    next_text,
                    ("speaker_assignment_uncertain",),
                )
            )
            continue

        if len(aligned) != 1:
            continue
        final_turn = aligned[0]
        if str(pre_turn.get("speaker_id")) != str(final_turn.get("speaker_id")):
            continue
        target_text = str(final_turn.get("final_text", "")).strip()
        if not target_text:
            continue
        semantic.append(
            _make_example(
                combined_fingerprint,
                turn_id,
                split_name,
                "semantic_cleanup",
                start_ms,
                end_ms,
                source_text,
                target_text,
                previous_text,
                next_text,
                (),
            )
        )

    semantic_dev = [item for item in semantic if item.split == "dev"]
    semantic_heldout = [item for item in semantic if item.split == "heldout"]
    selected_dev = deterministic_even_sample(semantic_dev, max_dev)
    selected_heldout = deterministic_even_sample(semantic_heldout, max_heldout)
    selected_safety = deterministic_stratified_safety_sample(safety, max_safety)
    selected = sorted(
        [*selected_dev, *selected_heldout, *selected_safety],
        key=lambda item: (item.start_ms, item.kind, item.sample_id),
    )

    counts = {
        "preTurns": len(pre_turns),
        "finalTurns": len(final_turns),
        "manualSplitSourceTurns": len(split_ids),
        "wholeTurnOverrides": len(override_ids),
        "turnTextCorrections": len(corrected_turn_ids),
        "semanticEligible": len(semantic),
        "semanticDevAvailable": len(semantic_dev),
        "semanticHeldoutAvailable": len(semantic_heldout),
        "safetyAvailable": len(safety),
        "selectedDev": len(selected_dev),
        "selectedHeldout": len(selected_heldout),
        "selectedSafety": len(selected_safety),
        "selectedOverlapSafety": sum(
            item.kind == "overlap_safety" for item in selected_safety
        ),
        "selectedSpeakerSafety": sum(
            item.kind == "speaker_safety" for item in selected_safety
        ),
        "selectedTotal": len(selected),
    }
    return DatasetBundle(
        examples=tuple(selected),
        source_hashes=dict(source_hashes),
        combined_fingerprint=combined_fingerprint,
        duration_ms=duration_ms,
        split_boundary_ms=split_boundary_ms,
        counts=counts,
        source_texts_for_privacy_check=tuple(
            _ordered_unique(
                text.strip()
                for text in all_private_texts
                if isinstance(text, str) and len(text.strip()) >= 4
            )
        ),
        speaker_truth_for_privacy_check=tuple(
            _ordered_unique(
                text.strip()
                for text in all_private_speaker_values
                if isinstance(text, str) and text.strip()
            )
        ),
    )


def deterministic_even_sample(items: Sequence[Example], limit: int | None) -> list[Example]:
    ordered = sorted(items, key=lambda item: (item.start_ms, item.end_ms, item.sample_id))
    if limit is None or limit <= 0 or len(ordered) <= limit:
        return list(ordered)
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [ordered[index] for index in sorted(indices)]


def deterministic_stratified_safety_sample(
    items: Sequence[Example],
    limit: int | None,
) -> list[Example]:
    ordered = sorted(items, key=lambda item: (item.start_ms, item.end_ms, item.sample_id))
    if limit is None or limit <= 0 or len(ordered) <= limit:
        return ordered

    strata: dict[str, list[Example]] = {}
    for item in ordered:
        strata.setdefault(item.kind, []).append(item)
    non_empty = [kind for kind in sorted(strata) if strata[kind]]
    if limit < len(non_empty):
        representatives = [
            deterministic_even_sample(strata[kind], 1)[0]
            for kind in non_empty
        ]
        return sorted(
            deterministic_even_sample(representatives, limit),
            key=lambda item: (item.start_ms, item.end_ms, item.sample_id),
        )

    allocations = {kind: 1 for kind in non_empty}
    remaining = limit - len(non_empty)
    while remaining > 0:
        candidates = [
            kind
            for kind in non_empty
            if allocations[kind] < len(strata[kind])
        ]
        if not candidates:
            break
        kind = max(
            candidates,
            key=lambda value: (
                len(strata[value]) / allocations[value],
                len(strata[value]),
                value,
            ),
        )
        allocations[kind] += 1
        remaining -= 1

    selected = [
        item
        for kind in non_empty
        for item in deterministic_even_sample(strata[kind], allocations[kind])
    ]
    return sorted(
        selected,
        key=lambda item: (item.start_ms, item.end_ms, item.sample_id),
    )


def make_runtime_input(example: Example) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0",
        "sampleId": example.sample_id,
        "language": "zh-CN",
        "sourceText": example.source_text,
        "context": {
            "previousText": example.previous_text[:500],
            "nextText": example.next_text[:500],
        },
        "glossary": [],
        "riskFlags": list(example.input_risk_flags),
        "constraints": {
            "speakerLocked": True,
            "turnBoundaryLocked": True,
            "translationForbidden": True,
            "summarizationForbidden": True,
            "stylePolishingForbidden": True,
            "unglossedTermCorrectionForbidden": True,
        },
    }


def _make_example(
    fingerprint: str,
    turn_id: str,
    split: str,
    kind: str,
    start_ms: int,
    end_ms: int,
    source_text: str,
    target_text: str,
    previous_text: str,
    next_text: str,
    risk_flags: tuple[str, ...],
) -> Example:
    sample_id = hashlib.sha256(
        f"{fingerprint}|{turn_id}|{kind}".encode("utf-8")
    ).hexdigest()[:16]
    return Example(
        sample_id=sample_id,
        source_turn_id=turn_id,
        split=split,
        kind=kind,
        start_ms=start_ms,
        end_ms=end_ms,
        source_text=source_text,
        target_text=target_text,
        previous_text=previous_text,
        next_text=next_text,
        input_risk_flags=risk_flags,
    )


def _require_turns(payload: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{label}_turns_missing")
    required = {"turn_id", "start_ms", "end_ms", "final_text", "speaker_id"}
    for turn in turns:
        if not isinstance(turn, Mapping) or not required.issubset(turn):
            raise ValueError(f"{label}_turn_contract")
    return turns


def _neighbor_text(turns: Sequence[Mapping[str, Any]], index: int) -> str:
    if index < 0 or index >= len(turns):
        return ""
    return str(turns[index].get("final_text", "")).strip()


def _text_fields(turn: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("final_text", "qwen_text", "funasr_text", "original_final_text"):
        value = turn.get(key)
        if isinstance(value, str):
            values.append(value)
    return values


def collect_private_values(payload: Any) -> tuple[list[str], list[str]]:
    text_values: list[str] = []
    speaker_values: list[str] = []

    def visit(value: Any, inherited_kind: str | None = None) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                kind = inherited_kind
                if _SPEAKER_KEY_RE.search(key):
                    kind = "speaker"
                elif _TEXT_KEY_RE.search(key):
                    kind = "text"
                visit(child, kind)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child, inherited_kind)
            return
        if inherited_kind == "text" and isinstance(value, str):
            text_values.append(value)
        elif inherited_kind == "speaker" and isinstance(value, (str, int)):
            speaker_values.append(str(value))

    visit(payload)
    return text_values, speaker_values


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
