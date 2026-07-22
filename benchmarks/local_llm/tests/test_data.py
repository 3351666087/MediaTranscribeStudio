from __future__ import annotations

from local_llm_bench.data import build_dataset, deterministic_even_sample


def _turn(turn_id, start, end, speaker, text, **extra):
    value = {
        "turn_id": turn_id,
        "start_ms": start,
        "end_ms": end,
        "speaker_id": speaker,
        "final_text": text,
    }
    value.update(extra)
    return value


def test_continuous_time_partition_and_safety_exclusion():
    pre = {
        "duration_ms": 1000,
        "turns": [
            _turn(0, 0, 100, 0, "先看第一项"),
            _turn(1, 200, 300, 1, "再看第二项"),
            _turn(2, 750, 850, 2, "最后确认"),
            _turn(3, 900, 980, 3, "这里有串话"),
        ],
    }
    final = {
        "turns": [
            _turn(10, 0, 100, 0, "先看第一项。", original_turn_id=0),
            _turn(11, 200, 300, 1, "再看第二项。", original_turn_id=1),
            _turn(12, 750, 850, 2, "最后确认。", original_turn_id=2),
            _turn(13, 900, 940, 3, "串话甲", original_turn_id=3),
            _turn(14, 940, 980, 4, "串话乙", original_turn_id=3),
        ]
    }
    turn_corrections = {"turn_text": {"0": "先看第一项。", "1": "再看第二项。"}}
    sentence_decisions = {
        "split_turns": {"3": {"segments": []}},
        "whole_turn_overrides": {},
    }
    bundle = build_dataset(
        pre,
        final,
        turn_corrections,
        sentence_decisions,
        source_hashes={"pre": "a", "final": "b"},
        combined_fingerprint="f" * 64,
        dev_ratio=0.7,
        max_dev=None,
        max_heldout=None,
        max_safety=None,
    )
    semantic = [item for item in bundle.examples if item.kind == "semantic_cleanup"]
    safety = [item for item in bundle.examples if item.kind == "overlap_safety"]
    assert [item.split for item in semantic] == ["dev", "dev", "heldout"]
    assert len(safety) == 1
    assert safety[0].input_risk_flags == ("overlap_candidate",)
    assert bundle.split_boundary_ms == 700


def test_even_sampling_keeps_temporal_coverage():
    pre = {
        "duration_ms": 1000,
        "turns": [_turn(i, i * 100, i * 100 + 50, 0, f"合成样本{i}") for i in range(10)],
    }
    final = {
        "turns": [
            _turn(i, i * 100, i * 100 + 50, 0, f"合成样本{i}。", original_turn_id=i)
            for i in range(10)
        ]
    }
    bundle = build_dataset(
        pre,
        final,
        {"turn_text": {}},
        {"split_turns": {}, "whole_turn_overrides": {}},
        source_hashes={},
        combined_fingerprint="e" * 64,
        dev_ratio=0.7,
        max_dev=None,
        max_heldout=None,
        max_safety=None,
    )
    sampled = deterministic_even_sample(
        [item for item in bundle.examples if item.split == "dev"],
        3,
    )
    assert len(sampled) == 3
    assert sampled[0].start_ms == 0
    assert sampled[-1].start_ms == 600
