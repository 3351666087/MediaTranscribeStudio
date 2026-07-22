from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from reporting.report_document_assembler import (
    ReportAssemblyError,
    ReportDocumentAssembler,
    canonical_speaker_ids,
)


def speaker_scores(
    winner: str,
    speaker_ids: tuple[str, ...],
    *,
    margin: float = 0.30,
) -> list[dict[str, object]]:
    winner_index = speaker_ids.index(winner)
    values = [0.10 + index * 0.01 for index in range(len(speaker_ids))]
    values[winner_index] = 0.90
    if len(speaker_ids) > 1:
        runner_up = (winner_index + 1) % len(speaker_ids)
        values[runner_up] = 0.90 - margin
    return [
        {"speaker": f"legacy-{index + 1}", "score": value}
        for index, value in enumerate(values)
    ]


def synthetic_segments(count: int = 5) -> list[dict[str, object]]:
    speaker_ids = canonical_speaker_ids(count)
    output: list[dict[str, object]] = []
    for index in range(count):
        speaker_id = speaker_ids[index]
        output.append(
            {
                "id": f"segment-{index + 1:03d}",
                "start": float(index * 2),
                "end": float(index * 2 + 1.5),
                "speaker": f"legacy-{index + 1}",
                "raw_text": f"合成测试原文{index + 1}。",
                "normalized_text": f"合成测试原文{index + 1}。",
                "display_text": f"合成测试原文{index + 1}。",
                "confidence": 0.92,
                "speaker_scores": speaker_scores(speaker_id, speaker_ids),
                "boundary_confidence": 0.91,
            }
        )
    return output


class ReportDocumentAssemblerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assembler = ReportDocumentAssembler()
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / "synthetic.wav"
        self.source.write_bytes(b"synthetic-media")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def assemble(self, segments=None, **kwargs):
        materialized = list(segments or synthetic_segments())
        duration_ms = kwargs.pop(
            "duration_ms",
            max(
                10_000,
                int(max(float(segment["end"]) for segment in materialized) * 1000)
                + 500,
            ),
        )
        return self.assembler.assemble(
            materialized,
            source_path=self.source,
            duration_ms=duration_ms,
            document_id="synthetic-test-document",
            generated_at="2026-07-21T00:00:00Z",
            privacy={
                "containsRealMeetingText": False,
                "exportApproved": True,
            },
            **kwargs,
        )

    def test_auto_mode_supports_arbitrary_speaker_counts(self) -> None:
        for count in (1, 2, 5, 8, 13):
            with self.subTest(count=count):
                speaker_ids = canonical_speaker_ids(count)
                document = self.assemble(
                    synthetic_segments(count),
                    speaker_count_mode="auto",
                    speaker_count_confidence=0.93,
                )
                self.assertEqual(
                    [speaker["id"] for speaker in document["speakers"]],
                    list(speaker_ids),
                )
                self.assertEqual(
                    document["speakerPolicy"]["resolvedCount"],
                    count,
                )
                self.assertEqual(document["speakerPolicy"]["mode"], "auto")
                for segment in document["segments"]:
                    self.assertEqual(
                        [
                            score["speakerId"]
                            for score in segment["evidence"]["speaker"]["scores"]
                        ],
                        list(speaker_ids),
                    )

    def test_manual_mode_requires_the_requested_count(self) -> None:
        with self.assertRaisesRegex(
            ReportAssemblyError,
            "manual speaker_count_mode requires speaker_count",
        ):
            self.assemble(speaker_count_mode="manual")
        with self.assertRaisesRegex(ReportAssemblyError, "resolved 4 speakers"):
            self.assemble(
                synthetic_segments(5),
                speaker_count_mode="manual",
                speaker_count=4,
            )
        document = self.assemble(
            synthetic_segments(7),
            speaker_count_mode="manual",
            speaker_count=7,
        )
        self.assertEqual(document["speakerPolicy"]["requestedCount"], 7)
        self.assertEqual(document["speakerPolicy"]["resolvedCount"], 7)

    def test_hybrid_mode_enforces_bounds(self) -> None:
        document = self.assemble(
            synthetic_segments(6),
            speaker_count_mode="hybrid",
            minimum_speaker_count=4,
            maximum_speaker_count=8,
            speaker_count_confidence=0.88,
            speaker_count_candidates=[
                {"count": 6, "confidence": 0.88},
                {"count": 5, "confidence": 0.08},
            ],
        )
        self.assertEqual(document["speakerPolicy"]["resolvedCount"], 6)
        with self.assertRaisesRegex(ReportAssemblyError, "exceeds maximum"):
            self.assemble(
                synthetic_segments(6),
                speaker_count_mode="hybrid",
                maximum_speaker_count=5,
            )

    def test_explicit_alias_mapping_can_collapse_legacy_cluster_aliases(self) -> None:
        segments = synthetic_segments()
        alias = copy.deepcopy(segments[0])
        alias["id"] = "segment-006"
        alias["start"] = 9.2
        alias["end"] = 9.8
        alias["speaker"] = "legacy-1-alias"
        segments.append(alias)
        mapping = {
            "legacy-1": "speaker-1",
            "legacy-1-alias": "speaker-1",
            "legacy-2": "speaker-2",
            "legacy-3": "speaker-3",
            "legacy-4": "speaker-4",
            "legacy-5": "speaker-5",
        }
        document = self.assemble(segments, speaker_mapping=mapping)
        self.assertEqual(document["segments"][-1]["speakerId"], "speaker-1")

    def test_adds_revision_for_normalized_and_display_text_changes(self) -> None:
        segments = synthetic_segments()
        segments[0]["raw_text"] = "PDF要检查。"
        segments[0]["normalized_text"] = "PDF 要检查。"
        segments[0]["display_text"] = "PDF 要检查！"
        document = self.assemble(segments)
        revisions = document["segments"][0]["revisions"]
        self.assertEqual(len(revisions), 2)
        self.assertEqual(revisions[0]["before"], "PDF要检查。")
        self.assertEqual(revisions[0]["after"], "PDF 要检查。")
        self.assertEqual(revisions[1]["before"], "PDF 要检查。")
        self.assertEqual(revisions[1]["after"], "PDF 要检查！")

    def test_rejects_invalid_boundaries(self) -> None:
        segments = synthetic_segments()
        segments[2]["start"] = 1.0
        with self.assertRaisesRegex(ReportAssemblyError, "not monotonic"):
            self.assemble(segments)
        segments = synthetic_segments()
        segments[-1]["end"] = 10.1
        with self.assertRaisesRegex(ReportAssemblyError, "exceeds media duration"):
            self.assemble(segments, duration_ms=10_000)

    def test_rejects_incomplete_speaker_evidence_by_default(self) -> None:
        segments = synthetic_segments()
        segments[0]["speaker_scores"] = segments[0]["speaker_scores"][:3]
        with self.assertRaisesRegex(ReportAssemblyError, "missing scores"):
            self.assemble(segments)

    def test_partial_vector_fails_even_when_compatibility_flag_is_enabled(self) -> None:
        assembler = ReportDocumentAssembler(
            allow_neutral_compatibility_evidence=True
        )
        segments = synthetic_segments()
        segments[0]["speaker_scores"] = segments[0]["speaker_scores"][:3]
        with self.assertRaisesRegex(ReportAssemblyError, "partial vectors"):
            assembler.assemble(
                segments,
                source_path=self.source,
                duration_ms=10_000,
                document_id="synthetic-compat-document",
                generated_at="2026-07-21T00:00:00Z",
            )

    def test_locked_speaker_cannot_be_overridden_by_llm(self) -> None:
        segments = synthetic_segments()
        segments[0]["speaker_locked"] = True
        segments[0]["revisions"] = [
            {
                "revisionId": "speaker-change-001",
                "type": "speaker",
                "source": "llm",
                "reasonCode": "CONTEXT_REASSIGN",
                "before": "legacy-2",
                "after": "legacy-1",
            }
        ]
        with self.assertRaisesRegex(ReportAssemblyError, "LLM revisions"):
            self.assemble(segments)

    def test_high_margin_speaker_cannot_be_overridden_by_llm(self) -> None:
        segments = synthetic_segments()
        segments[0]["speaker_scores"] = speaker_scores(
            "speaker-2",
            canonical_speaker_ids(5),
            margin=0.40,
        )
        segments[0]["revisions"] = [
            {
                "revisionId": "speaker-change-002",
                "type": "speaker",
                "source": "llm",
                "reasonCode": "CONTEXT_REASSIGN",
                "before": "legacy-2",
                "after": "legacy-1",
            }
        ]
        with self.assertRaisesRegex(ReportAssemblyError, "forbidden"):
            self.assemble(segments)

    def test_config_hash_and_source_hash_are_real(self) -> None:
        document = self.assemble(config={"speakerCount": 5, "offline": True})
        self.assertEqual(len(document["source"]["sha256"]), 64)
        self.assertEqual(len(document["provenance"]["configSha256"]), 64)


if __name__ == "__main__":
    unittest.main()
