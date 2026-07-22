from __future__ import annotations

import unittest

from backend.speaker_sequence_decoder import (
    CARDINALITY_CHANGE_REVIEW_REQUIRED,
    SequenceDecoderConfig,
    SequenceSegment,
    SpeakerEmission,
    decode_speaker_sequence,
)


def _segment(
    number: int,
    original: str,
    scores: dict[str, float],
    *,
    start_ms: int | None = None,
    human_lock: str | None = None,
    anchor: bool = False,
) -> SequenceSegment:
    resolved_start = (
        (number - 1) * 1000
        if start_ms is None
        else start_ms
    )
    return SequenceSegment(
        segment_id=f"segment-{number}",
        start_ms=resolved_start,
        end_ms=resolved_start + 900,
        original_speaker_id=original,
        speaker_scores=tuple(
            SpeakerEmission(speaker_id=speaker_id, score=score)
            for speaker_id, score in scores.items()
        ),
        human_locked_speaker_id=human_lock,
        strong_acoustic_anchor=anchor,
    )


def _speaker_map(result) -> dict[str, str]:
    return {
        assignment.segment_id: assignment.speaker_id
        for assignment in result.assignments
    }


class SpeakerSequenceDecoderTests(unittest.TestCase):
    def test_acoustic_emission_overrides_weak_continuity(self) -> None:
        config = SequenceDecoderConfig(auto_anchor_min_score=2.0)
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
                _segment(
                    2,
                    "speaker-2",
                    {"speaker-2": 0.80, "speaker-1": 0.65},
                ),
                _segment(
                    3,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
            ),
            config,
        )

        self.assertEqual(
            [item.speaker_id for item in result.assignments],
            ["speaker-1", "speaker-2", "speaker-1"],
        )

    def test_continuity_only_breaks_a_near_acoustic_tie(self) -> None:
        config = SequenceDecoderConfig(auto_anchor_min_score=2.0)
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
                _segment(
                    2,
                    "speaker-2",
                    {"speaker-2": 0.701, "speaker-1": 0.700},
                ),
                _segment(
                    3,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
            ),
            config,
        )

        center = result.assignments[1]
        self.assertEqual(center.speaker_id, "speaker-1")
        self.assertTrue(center.changed)
        self.assertIn("SEQUENCE_REASSIGNED", center.reason_codes)
        self.assertFalse(result.review_required)

    def test_large_gap_removes_most_continuity_influence(self) -> None:
        config = SequenceDecoderConfig(auto_anchor_min_score=2.0)
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
                _segment(
                    2,
                    "speaker-2",
                    {"speaker-2": 0.701, "speaker-1": 0.700},
                    start_ms=60_000,
                ),
                _segment(
                    3,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                    start_ms=120_000,
                ),
            ),
            config,
        )

        self.assertEqual(result.assignments[1].speaker_id, "speaker-2")

    def test_human_lock_is_a_hard_constraint_outside_top_m(self) -> None:
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    {
                        "speaker-1": 0.95,
                        "speaker-2": 0.50,
                        "speaker-3": 0.40,
                    },
                    human_lock="speaker-9",
                ),
            ),
            SequenceDecoderConfig(top_m=2),
        )

        assignment = result.assignments[0]
        self.assertEqual(assignment.speaker_id, "speaker-9")
        self.assertIn("HUMAN_LOCKED", assignment.reason_codes)

    def test_explicit_strong_anchor_is_never_changed(self) -> None:
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
                _segment(
                    2,
                    "speaker-2",
                    {"speaker-1": 0.71, "speaker-2": 0.70},
                    anchor=True,
                ),
                _segment(
                    3,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
            )
        )

        center = result.assignments[1]
        self.assertEqual(center.speaker_id, "speaker-2")
        self.assertIn(
            "STRONG_ACOUSTIC_ANCHOR",
            center.reason_codes,
        )

    def test_strong_acoustic_b_in_a_b_a_is_not_smoothed(self) -> None:
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    {"speaker-1": 0.95, "speaker-2": 0.05},
                ),
                _segment(
                    2,
                    "speaker-2",
                    {"speaker-2": 0.93, "speaker-1": 0.50},
                ),
                _segment(
                    3,
                    "speaker-1",
                    {"speaker-1": 0.96, "speaker-2": 0.04},
                ),
            )
        )

        center = result.assignments[1]
        self.assertEqual(center.speaker_id, "speaker-2")
        self.assertIn(
            "STRONG_ACOUSTIC_ANCHOR",
            center.reason_codes,
        )
        self.assertNotIn("SEQUENCE_REASSIGNED", center.reason_codes)

    def test_last_high_confidence_support_is_restored_for_review(self) -> None:
        config = SequenceDecoderConfig(
            auto_anchor_min_score=2.0,
            high_confidence_support_min_score=0.65,
            high_confidence_support_min_margin=0.005,
        )
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
                _segment(
                    2,
                    "speaker-2",
                    {"speaker-2": 0.700, "speaker-1": 0.690},
                ),
                _segment(
                    3,
                    "speaker-1",
                    {"speaker-1": 0.90, "speaker-2": 0.10},
                ),
            ),
            config,
        )

        center = result.assignments[1]
        self.assertEqual(center.speaker_id, "speaker-2")
        self.assertFalse(center.changed)
        self.assertEqual(center.review_status, "REVIEW_REQUIRED")
        self.assertEqual(
            center.reason_codes,
            (CARDINALITY_CHANGE_REVIEW_REQUIRED,),
        )
        self.assertTrue(result.review_required)

    def test_decoder_is_deterministic_and_preserves_input_order(self) -> None:
        config = SequenceDecoderConfig(auto_anchor_min_score=2.0)
        chronological = (
            _segment(
                1,
                "speaker-1",
                {"speaker-1": 0.80, "speaker-2": 0.20},
            ),
            _segment(
                2,
                "speaker-2",
                {"speaker-2": 0.61, "speaker-1": 0.60},
            ),
            _segment(
                3,
                "speaker-1",
                {"speaker-1": 0.80, "speaker-2": 0.20},
            ),
        )
        shuffled = (
            chronological[2],
            chronological[0],
            chronological[1],
        )

        first = decode_speaker_sequence(chronological, config)
        second = decode_speaker_sequence(shuffled, config)
        third = decode_speaker_sequence(shuffled, config)

        self.assertEqual(_speaker_map(first), _speaker_map(second))
        self.assertEqual(second, third)
        self.assertEqual(
            [item.segment_id for item in second.assignments],
            ["segment-3", "segment-1", "segment-2"],
        )

    def test_duplicate_scores_use_deterministic_maximum(self) -> None:
        segment = SequenceSegment(
            segment_id="segment-1",
            start_ms=0,
            end_ms=900,
            original_speaker_id="speaker-2",
            speaker_scores=(
                SpeakerEmission("speaker-2", 0.10),
                SpeakerEmission("speaker-1", 0.70),
                SpeakerEmission("speaker-2", 0.80),
            ),
        )

        result = decode_speaker_sequence(
            (segment,),
            SequenceDecoderConfig(auto_anchor_min_score=2.0),
        )

        self.assertEqual(result.assignments[0].speaker_id, "speaker-2")
        self.assertEqual(result.assignments[0].acoustic_score, 0.80)

    def test_top_m_supports_large_global_speaker_inventory(self) -> None:
        scores = {
            f"speaker-{number}": number / 1000
            for number in range(1, 130)
        }
        result = decode_speaker_sequence(
            (
                _segment(
                    1,
                    "speaker-1",
                    scores,
                ),
            ),
            SequenceDecoderConfig(
                top_m=4,
                auto_anchor_min_score=2.0,
            ),
        )

        self.assertEqual(
            result.assignments[0].speaker_id,
            "speaker-129",
        )

    def test_transition_prior_cannot_be_configured_as_primary(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "combined transition differential",
        ):
            decode_speaker_sequence(
                (
                    _segment(
                        1,
                        "speaker-1",
                        {"speaker-1": 0.8},
                    ),
                ),
                SequenceDecoderConfig(
                    continuity_bonus=0.04,
                    switch_penalty=0.02,
                ),
            )

    def test_conflicting_hard_constraints_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "conflicting human lock",
        ):
            decode_speaker_sequence(
                (
                    _segment(
                        1,
                        "speaker-1",
                        {"speaker-1": 0.9, "speaker-2": 0.1},
                        human_lock="speaker-2",
                        anchor=True,
                    ),
                )
            )

    def test_non_integer_top_m_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "top_m must be a positive integer",
        ):
            decode_speaker_sequence(
                (
                    _segment(
                        1,
                        "speaker-1",
                        {"speaker-1": 0.8},
                    ),
                ),
                SequenceDecoderConfig(top_m=3.0),  # type: ignore[arg-type]
            )

    def test_empty_sequence_is_a_valid_noop(self) -> None:
        result = decode_speaker_sequence(())

        self.assertEqual(result.assignments, ())
        self.assertEqual(result.total_score, 0.0)
        self.assertFalse(result.review_required)


if __name__ == "__main__":
    unittest.main()
