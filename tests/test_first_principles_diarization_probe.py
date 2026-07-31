from __future__ import annotations

from tools.run_first_principles_diarization_probe import _turns


class _Interval:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _Annotation:
    def itertracks(self, *, yield_label: bool):
        assert yield_label
        return [
            (_Interval(1.0, 2.0), "track", "SPEAKER_1"),
            (_Interval(2.0, 2.0), "track", "ignored"),
            (_Interval(0.0, 0.5), "track", "SPEAKER_0"),
        ]


def test_turns_are_sorted_and_empty_intervals_are_ignored() -> None:
    assert _turns(_Annotation()) == [
        {"startSeconds": 0.0, "endSeconds": 0.5, "localSpeaker": "SPEAKER_0"},
        {"startSeconds": 1.0, "endSeconds": 2.0, "localSpeaker": "SPEAKER_1"},
    ]
