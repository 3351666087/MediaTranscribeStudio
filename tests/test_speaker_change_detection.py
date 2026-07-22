from __future__ import annotations

import math
import unittest

from backend.speaker_change_detection import (
    NOT_REQUIRED,
    OVERLAP_NOT_EVALUATED,
    OVERLAP_RISK_NOT_EVALUATED,
    REVIEW_REQUIRED,
    AsrBoundary,
    EnergyValley,
    Pcm16kMonoTimeline,
    SpeakerChangeDetectionConfig,
    SpeakerEmbeddingWindow,
    plan_speaker_changes,
)


def _vector_with_cosine(cosine: float) -> tuple[float, float]:
    return (cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)))


def _window(
    window_id: str,
    start_ms: int,
    end_ms: int,
    embedding: tuple[float, ...],
    *,
    confidence: float = 0.98,
    overlap_risk: bool = False,
) -> SpeakerEmbeddingWindow:
    return SpeakerEmbeddingWindow(
        window_id=window_id,
        start_ms=start_ms,
        end_ms=end_ms,
        embedding=embedding,
        confidence=confidence,
        overlap_risk=overlap_risk,
        evidence={"model": "CAM++", "fixture": True},
    )


class SpeakerChangeDetectionTests(unittest.TestCase):
    def test_boundary_evidence_cannot_create_a_change_candidate(self) -> None:
        windows = (
            _window("w1", 0, 1000, (1.0, 0.0)),
            _window("w2", 1000, 2000, (1.0, 0.0)),
        )

        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=windows,
            energy_valleys=(EnergyValley(950, 1.0, "energy-1"),),
            asr_boundaries=(AsrBoundary(1020, 1.0, "asr-1"),),
        )

        self.assertEqual(plan.evaluated_edge_count, 1)
        self.assertEqual(plan.candidate_edge_count, 0)
        self.assertEqual(plan.proposals, ())
        self.assertEqual(plan.automatic_splits_ms, ())

    def test_strong_cam_plus_change_auto_splits_at_energy_valley(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 1000, 2000, (0.0, 1.0)),
            ),
            energy_valleys=(EnergyValley(930, 0.92, "energy-1"),),
        )

        self.assertEqual(len(plan.proposals), 1)
        proposal = plan.proposals[0]
        self.assertEqual(proposal.acoustic_boundary_ms, 1000)
        self.assertEqual(proposal.split_ms, 930)
        self.assertEqual(proposal.boundary_source, "ENERGY_VALLEY")
        self.assertEqual(proposal.boundary_marker_id, "energy-1")
        self.assertEqual(proposal.review_status, NOT_REQUIRED)
        self.assertTrue(proposal.apply_automatically)
        self.assertEqual(plan.automatic_splits_ms, (930,))

    def test_asr_boundary_is_optional_weak_localization(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 1000, 2000, (0.0, 1.0)),
            ),
            asr_boundaries=(AsrBoundary(1060, 0.88, "asr-1"),),
        )

        proposal = plan.proposals[0]
        self.assertEqual(proposal.split_ms, 1060)
        self.assertEqual(proposal.boundary_source, "ASR_BOUNDARY")
        self.assertEqual(
            proposal.evidence_codes,
            ("CAM_PLUS_ADJACENT_CHANGE", "ASR_BOUNDARY_LOCALIZATION"),
        )
        self.assertTrue(proposal.apply_automatically)

    def test_localization_never_increases_acoustic_scores(self) -> None:
        windows = (
            _window("w1", 0, 1000, (1.0, 0.0)),
            _window("w2", 1000, 2000, (0.0, 1.0)),
        )
        acoustic_only = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=windows,
        ).proposals[0]
        localized = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=windows,
            energy_valleys=(EnergyValley(940, 1.0),),
            asr_boundaries=(AsrBoundary(945, 1.0),),
        ).proposals[0]

        self.assertEqual(localized.change_score, acoustic_only.change_score)
        self.assertEqual(
            localized.acoustic_confidence,
            acoustic_only.acoustic_confidence,
        )
        self.assertNotEqual(localized.split_ms, acoustic_only.split_ms)

    def test_low_change_score_is_review_only(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 1000, 2000, _vector_with_cosine(0.75)),
            ),
            energy_valleys=(EnergyValley(980, 1.0),),
        )

        proposal = plan.proposals[0]
        self.assertAlmostEqual(proposal.change_score, 0.25)
        self.assertEqual(proposal.review_status, REVIEW_REQUIRED)
        self.assertIn("LOW_CHANGE_SCORE", proposal.review_reasons)
        self.assertIn("LOW_ACOUSTIC_CONFIDENCE", proposal.review_reasons)
        self.assertFalse(proposal.apply_automatically)
        self.assertEqual(plan.automatic_splits_ms, ())

    def test_low_embedding_confidence_is_review_only(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window(
                    "w1",
                    0,
                    1000,
                    (1.0, 0.0),
                    confidence=0.45,
                ),
                _window("w2", 1000, 2000, (0.0, 1.0)),
            ),
        )

        proposal = plan.proposals[0]
        self.assertEqual(proposal.review_status, REVIEW_REQUIRED)
        self.assertIn(
            "LOW_EMBEDDING_CONFIDENCE",
            proposal.review_reasons,
        )
        self.assertFalse(proposal.apply_automatically)

    def test_overlap_risk_is_not_reported_as_overlap_detection(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window(
                    "w1",
                    0,
                    1000,
                    (1.0, 0.0),
                    overlap_risk=True,
                ),
                _window("w2", 1000, 2000, (0.0, 1.0)),
            ),
        )

        proposal = plan.proposals[0]
        self.assertEqual(proposal.review_status, REVIEW_REQUIRED)
        self.assertIn(
            "OVERLAP_RISK_NOT_EVALUATED",
            proposal.review_reasons,
        )
        self.assertEqual(
            proposal.overlap_detection_status,
            OVERLAP_RISK_NOT_EVALUATED,
        )
        self.assertEqual(
            plan.overlap_detection_status,
            OVERLAP_NOT_EVALUATED,
        )
        payload = plan.as_dict()
        self.assertFalse(payload["overlapDetectorRun"])
        self.assertNotIn("overlapDetected", payload)
        self.assertFalse(proposal.apply_automatically)

    def test_split_near_vad_edge_is_review_only(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 400, (1.0, 0.0)),
                _window("w2", 300, 700, (0.0, 1.0)),
            ),
            energy_valleys=(EnergyValley(300, 1.0),),
        )

        proposal = plan.proposals[0]
        self.assertEqual(proposal.split_ms, 300)
        self.assertEqual(proposal.review_status, REVIEW_REQUIRED)
        self.assertIn(
            "SHORT_RESULTING_INTERVAL",
            proposal.review_reasons,
        )
        self.assertFalse(proposal.apply_automatically)

    def test_two_changes_that_create_a_short_middle_turn_both_require_review(
        self,
    ) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 600, 1600, (0.0, 1.0)),
                _window("w3", 1200, 2200, (1.0, 0.0)),
            ),
        )

        self.assertEqual(
            tuple(proposal.split_ms for proposal in plan.proposals),
            (800, 1400),
        )
        self.assertTrue(
            all(
                proposal.review_status == REVIEW_REQUIRED
                for proposal in plan.proposals
            )
        )
        self.assertTrue(
            all(
                "SHORT_RESULTING_INTERVAL" in proposal.review_reasons
                for proposal in plan.proposals
            )
        )
        self.assertEqual(plan.automatic_splits_ms, ())

    def test_nearby_change_peaks_are_suppressed_deterministically(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0, 0.0)),
                _window("w2", 100, 1100, (0.0, 1.0, 0.0)),
                _window("w3", 200, 1200, (0.0, 0.0, 1.0)),
            ),
        )

        self.assertEqual(plan.candidate_edge_count, 2)
        self.assertEqual(plan.suppressed_peak_count, 1)
        self.assertEqual(len(plan.proposals), 1)
        self.assertEqual(plan.proposals[0].left_window_id, "w1")
        self.assertEqual(plan.proposals[0].right_window_id, "w2")
        self.assertEqual(
            plan.proposals[0].support_window_ids,
            ("w1", "w2", "w3"),
        )
        self.assertIn(
            "NEARBY_CHANGE_PEAKS_MERGED",
            plan.proposals[0].evidence_codes,
        )

    def test_boundary_marker_order_and_ties_are_deterministic(self) -> None:
        windows = (
            _window("w1", 0, 1000, (1.0, 0.0)),
            _window("w2", 1000, 2000, (0.0, 1.0)),
        )
        markers = (
            EnergyValley(1050, 0.8, "later"),
            EnergyValley(950, 0.8, "earlier"),
        )
        forward = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=windows,
            energy_valleys=markers,
        )
        reverse = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=windows,
            energy_valleys=tuple(reversed(markers)),
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.proposals[0].split_ms, 950)
        self.assertEqual(forward.proposals[0].boundary_marker_id, "earlier")

    def test_energy_wins_an_exact_localization_tie_with_asr(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 1000, 2000, (0.0, 1.0)),
            ),
            energy_valleys=(EnergyValley(975, 0.9, "energy"),),
            asr_boundaries=(AsrBoundary(975, 0.9, "asr"),),
        )

        self.assertEqual(
            plan.proposals[0].boundary_source,
            "ENERGY_VALLEY",
        )
        self.assertEqual(plan.proposals[0].boundary_marker_id, "energy")

    def test_low_confidence_boundary_marker_is_ignored(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 1000, 2000, (0.0, 1.0)),
            ),
            energy_valleys=(EnergyValley(900, 0.1, "weak"),),
        )

        proposal = plan.proposals[0]
        self.assertEqual(proposal.split_ms, 1000)
        self.assertEqual(proposal.boundary_source, "ACOUSTIC_MIDPOINT")
        self.assertIsNone(proposal.boundary_marker_confidence)

    def test_sparse_window_coverage_fails_closed(self) -> None:
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=4000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 2000, 3000, (0.0, 1.0)),
            ),
        )

        proposal = plan.proposals[0]
        self.assertEqual(proposal.review_status, REVIEW_REQUIRED)
        self.assertIn(
            "SPARSE_ACOUSTIC_COVERAGE",
            proposal.review_reasons,
        )
        self.assertIn(
            "BOUNDARY_LOCALIZATION_UNCERTAIN",
            proposal.review_reasons,
        )
        self.assertFalse(proposal.apply_automatically)

    def test_shared_pcm_timeline_is_referenced_without_decoder_contract(
        self,
    ) -> None:
        pcm = Pcm16kMonoTimeline(
            buffer_id="pcm:fixture",
            sample_count=64_000,
        )
        plan = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(
                _window("w1", 0, 1000, (1.0, 0.0)),
                _window("w2", 1000, 2000, (0.0, 1.0)),
            ),
            pcm_timeline=pcm,
        )

        self.assertEqual(pcm.duration_ms, 4000)
        self.assertEqual(plan.pcm_buffer_id, "pcm:fixture")
        self.assertEqual(plan.as_dict()["pcmBufferId"], "pcm:fixture")

    def test_shared_pcm_must_be_16khz_mono(self) -> None:
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            Pcm16kMonoTimeline(
                buffer_id="pcm:bad-rate",
                sample_count=16_000,
                sample_rate_hz=8_000,
            )
        with self.assertRaisesRegex(ValueError, "mono"):
            Pcm16kMonoTimeline(
                buffer_id="pcm:bad-channels",
                sample_count=16_000,
                channel_count=2,
            )

    def test_vad_must_fit_inside_shared_pcm(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds shared PCM"):
            plan_speaker_changes(
                vad_start_ms=0,
                vad_end_ms=1001,
                windows=(),
                pcm_timeline=Pcm16kMonoTimeline(
                    buffer_id="pcm:one-second",
                    sample_count=16_000,
                ),
            )

    def test_unsorted_windows_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly time-sorted"):
            plan_speaker_changes(
                vad_start_ms=0,
                vad_end_ms=3000,
                windows=(
                    _window("w2", 1000, 2000, (0.0, 1.0)),
                    _window("w1", 0, 1000, (1.0, 0.0)),
                ),
            )

    def test_mismatched_embedding_dimensions_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dimension"):
            plan_speaker_changes(
                vad_start_ms=0,
                vad_end_ms=3000,
                windows=(
                    _window("w1", 0, 1000, (1.0, 0.0)),
                    _window("w2", 1000, 2000, (0.0, 1.0, 0.0)),
                ),
            )

    def test_zero_norm_and_non_finite_embeddings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero norm"):
            _window("zero", 0, 1000, (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "finite number"):
            _window("nan", 0, 1000, (math.nan, 0.0))

    def test_empty_and_single_window_inputs_are_valid_noops(self) -> None:
        empty = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(),
        )
        single = plan_speaker_changes(
            vad_start_ms=0,
            vad_end_ms=3000,
            windows=(_window("w1", 0, 1000, (1.0, 0.0)),),
        )

        self.assertEqual(empty.proposals, ())
        self.assertEqual(empty.evaluated_edge_count, 0)
        self.assertEqual(single.proposals, ())
        self.assertEqual(single.evaluated_edge_count, 0)
        self.assertFalse(empty.review_required)

    def test_configuration_rejects_unsafe_threshold_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "must exceed"):
            SpeakerChangeDetectionConfig(
                min_review_change_score=0.5,
                min_auto_change_score=0.5,
            )


if __name__ == "__main__":
    unittest.main()
