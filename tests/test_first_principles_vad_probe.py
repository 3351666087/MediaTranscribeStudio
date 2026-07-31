from __future__ import annotations

from tools.run_first_principles_vad_probe import _normalize_intervals


def test_normalize_vad_intervals_discards_invalid_values() -> None:
    assert _normalize_intervals([[0, 100], [200, 200], [-1, 4], [300, 500]]) == [
        {"startMs": 0, "endMs": 100},
        {"startMs": 300, "endMs": 500},
    ]
