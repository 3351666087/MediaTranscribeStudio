from __future__ import annotations

import copy
import hashlib
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

from backend.adapters import AdapterContext
from backend.documents import build_review_queue
from backend.errors import JobCancelled, WorkerError
from backend.models import (
    SpeakerCountPolicy,
    SpeakerScore,
    StartJobRequest,
    TranscriptSegment,
)
from backend.production_runners import LocalERes2NetV2Verifier
from backend.speaker_pipeline import (
    AsrHypothesis,
    EmbeddingRecord,
    InMemoryStageCache,
    NoOverlapAdapter,
    OverlapDecision,
    PreparedAudio,
    ReviewProposal,
    ReviewCandidate,
    SpeakerPipeline,
    SpeakerPipelineConfig,
    SpeechWindow,
    UnavailableOverlapAdapter,
    _ClusterResult,
)


class FakePreparationAdapter:
    adapter_id = "normalize-vad-boundary-fixture"
    version = "1"

    def __init__(
        self,
        window_count: int,
        *,
        boundary_conflict_ids: set[str] | None = None,
        locked: dict[str, str] | None = None,
        window_order: tuple[int, ...] | None = None,
        window_ranges: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        self.window_count = window_count
        self.boundary_conflict_ids = boundary_conflict_ids or set()
        self.locked = locked or {}
        self.window_order = window_order
        self.window_ranges = dict(window_ranges or {})
        self.calls = 0

    def prepare(self, source_path, *, normalization_profile, context):
        context.raise_if_cancelled()
        self.calls += 1
        source_fingerprint = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        windows_by_number = {
            index + 1: SpeechWindow(
                window_id=f"window-{index + 1}",
                start_ms=self.window_ranges.get(
                    f"window-{index + 1}",
                    (index * 1000, (index + 1) * 1000),
                )[0],
                end_ms=self.window_ranges.get(
                    f"window-{index + 1}",
                    (index * 1000, (index + 1) * 1000),
                )[1],
                boundary_conflict=f"window-{index + 1}"
                in self.boundary_conflict_ids,
                locked_speaker_id=self.locked.get(f"window-{index + 1}"),
                metadata={"turnId": f"turn-{index + 1}"},
            )
            for index in range(self.window_count)
        }
        order = self.window_order or tuple(range(1, self.window_count + 1))
        if sorted(order) != list(range(1, self.window_count + 1)):
            raise AssertionError("window_order must be a complete permutation")
        windows = tuple(windows_by_number[index] for index in order)
        return PreparedAudio(
            duration_ms=max(window.end_ms for window in windows),
            source_fingerprint=source_fingerprint,
            normalization_profile=normalization_profile,
            windows=windows,
            stage_durations_ms={
                "decode": 1.0,
                "normalize": 2.0,
                "vad": 3.0,
                "boundary": 4.0,
            },
        )


class FailingPreparationAdapter:
    adapter_id = "normalize-vad-boundary-failing-fixture"
    version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def prepare(self, source_path, *, normalization_profile, context):
        context.raise_if_cancelled()
        self.calls += 1
        raise WorkerError(
            "FUNASR_VAD_INFERENCE_FAILED",
            "synthetic VAD inference failure",
        )


class FakeAsrAdapter:
    adapter_id = "qwen3-asr-1.7b-fixture"
    version = "1"

    def __init__(
        self,
        mode: str = "valid",
        *,
        language_by_window_id: Mapping[str, str] | None = None,
    ) -> None:
        self.mode = mode
        self.language_by_window_id = dict(language_by_window_id or {})
        self.calls: list[tuple[str, ...]] = []
        self.requested_languages: list[str] = []

    def transcribe_batch(
        self,
        prepared,
        windows,
        context,
        *,
        requested_language: str,
    ):
        context.raise_if_cancelled()
        self.calls.append(tuple(window.window_id for window in windows))
        self.requested_languages.append(requested_language)
        if self.mode == "string":
            return "not-an-array"
        values = []
        for window in windows:
            evidence = {
                "model": "Qwen3-ASR-1.7B",
                "timestamps": [
                    {
                        "text": f"词{index + 1}",
                        "startMs": start_ms,
                        "endMs": min(start_ms + 900, window.end_ms),
                    }
                    for index, start_ms in enumerate(
                        range(window.start_ms, window.end_ms, 1_000)
                    )
                ],
            }
            language = self.language_by_window_id.get(window.window_id)
            if language is not None:
                evidence["language"] = language
            if self.mode == "reject-last" and window is windows[-1]:
                values.append(
                    AsrHypothesis(
                        window_id=window.window_id,
                        text="",
                        confidence=0.0,
                        evidence={
                            **evidence,
                            "disposition": "rejected-non-lexical",
                            "rejectionReason": (
                                "EMPTY_AFTER_INDIVIDUAL_RETRY"
                            ),
                        },
                    )
                )
                continue
            values.append(
                AsrHypothesis(
                    window_id=window.window_id,
                    text=f"中文原文{window.window_id}",
                    normalized_text=f"中文原文{window.window_id}",
                    display_text=f"中文原文{window.window_id}",
                    confidence=0.98,
                    evidence=evidence,
                )
            )
        if self.mode == "missing":
            return values[:-1]
        if self.mode == "duplicate" and values:
            return [*values, values[0]]
        return values


class RejectingLanguageAsrAdapter(FakeAsrAdapter):
    def validate_requested_language(self, requested_language: str) -> None:
        raise WorkerError(
            "ASR_LANGUAGE_UNSUPPORTED",
            "synthetic unsupported language",
            details={"requestedLanguage": requested_language},
        )


class FakeCamPlusAdapter:
    adapter_id = "CAM++"
    version = "1"

    def __init__(
        self,
        speaker_count: int,
        *,
        identical: bool = False,
        vectors_by_window_id: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.speaker_count = speaker_count
        self.identical = identical
        self.vectors_by_window_id = {
            window_id: tuple(float(value) for value in vector)
            for window_id, vector in (vectors_by_window_id or {}).items()
        }
        self.calls: list[tuple[str, ...]] = []

    def embed_batch(self, prepared, windows, context):
        context.raise_if_cancelled()
        self.calls.append(tuple(window.window_id for window in windows))
        output = []
        for window in windows:
            index = int(window.window_id.rsplit("-", 1)[1]) - 1
            if window.window_id in self.vectors_by_window_id:
                vector = self.vectors_by_window_id[window.window_id]
            elif self.identical:
                vector = [1.0, *([0.0] * max(0, self.speaker_count - 1))]
            else:
                vector = [0.0] * self.speaker_count
                vector[index % self.speaker_count] = 1.0
            output.append(
                EmbeddingRecord(
                    window_id=window.window_id,
                    vector=tuple(vector),
                    confidence=0.99,
                    evidence={"model": "CAM++"},
                )
            )
        return output


class FakeOverlapAdapter:
    adapter_id = "overlap-fixture"
    version = "1"

    def __init__(self, overlapping_ids: set[str] | None = None) -> None:
        self.overlapping_ids = overlapping_ids or set()
        self.calls: list[tuple[str, ...]] = []

    def detect_batch(self, prepared, windows, context):
        context.raise_if_cancelled()
        self.calls.append(tuple(window.window_id for window in windows))
        return [
            OverlapDecision(
                window_id=window.window_id,
                overlapping=window.window_id in self.overlapping_ids,
                confidence=0.99,
                evidence={"detector": "fixture"},
            )
            for window in windows
        ]


class FakePyannoteOverlapAdapter(FakeOverlapAdapter):
    adapter_id = "pyannote-community-1"
    version = "2.2.0"

    def __init__(
        self,
        *,
        inconsistent_digest: bool = False,
        observed_speaker_count: int = 2,
    ) -> None:
        super().__init__()
        self.inconsistent_digest = inconsistent_digest
        self.observed_speaker_count = observed_speaker_count

    def detect_batch(self, prepared, windows, context):
        context.raise_if_cancelled()
        self.calls.append(tuple(window.window_id for window in windows))
        output = []
        for index, window in enumerate(windows):
            local_speaker = (
                "LOCAL_A"
                if self.observed_speaker_count == 1 or window.start_ms < 3_000
                else "LOCAL_B"
            )
            local_speakers = (
                ["LOCAL_A"]
                if self.observed_speaker_count == 1
                else ["LOCAL_A", "LOCAL_B"]
            )
            digest_character = (
                "b" if self.inconsistent_digest and index == 1 else "a"
            )
            output.append(
                OverlapDecision(
                    window_id=window.window_id,
                    overlapping=False,
                    confidence=0.5,
                    evidence={
                        "detectorStatus": "EVALUATED",
                        "overlapDetectorRun": True,
                        "reviewStatus": "NOT_REQUIRED",
                        "confidenceKind": "binary-annotation-no-posterior",
                        "calibratedConfidence": False,
                        "fullTimelineInference": {
                            "scope": "full-normalized-timeline",
                            "startMs": 0,
                            "endMs": prepared.duration_ms,
                            "turnCount": 6,
                            "localSpeakerCount": len(local_speakers),
                            "localSpeakers": local_speakers,
                            "speakerTurnsSha256": digest_character * 64,
                        },
                        "speakerTurns": [
                            {
                                "startMs": window.start_ms,
                                "endMs": window.end_ms,
                                "localSpeaker": local_speaker,
                            }
                        ],
                        "overlapIntervals": [],
                        "localSpeakerCount": 1,
                    },
                )
            )
        return output


class FakeSecondaryVerifier:
    adapter_id = "ERes2NetV2"
    version = "1"

    def __init__(
        self,
        *,
        mutate_text: bool = False,
        exit_reasons: Mapping[str, str] | None = None,
        speaker_ids: Mapping[str, str] | None = None,
    ) -> None:
        self.mutate_text = mutate_text
        self.exit_reasons = dict(exit_reasons or {})
        self.speaker_ids = dict(speaker_ids or {})
        self.calls: list[tuple[str, ...]] = []

    def review_batch(self, candidates, segments, context):
        context.raise_if_cancelled()
        self.calls.append(tuple(candidate.segment_id for candidate in candidates))
        return [
            ReviewProposal(
                segment_id=candidate.segment_id,
                source="acoustic",
                speaker_id=self.speaker_ids.get(candidate.segment_id),
                normalized_text=(
                    "非法改写" if self.mutate_text else None
                ),
                reason_code="ERES2NETV2_VERIFY",
                evidence_refs=(f"eres2netv2:{candidate.segment_id}",),
                confidence=0.91,
                exit_reason=self.exit_reasons.get(
                    candidate.segment_id,
                    "VERIFIED_NO_CHANGE",
                ),
                resource={"ramMb": 64.0, "vramMb": 128.0},
            )
            for candidate in candidates
        ]


class FakePyannoteAudit(FakeSecondaryVerifier):
    adapter_id = "pyannote-community-1"
    telemetry_enabled = False


class SpeakerPipelineProductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "meeting.mov"
        self.source.write_bytes(b"synthetic media for deterministic hashing")
        self.output = self.root / "output"
        self.output.mkdir()
        self.eres_model = self.root / "eres2netv2-model"
        self.eres_model.mkdir()
        self.audio_primary = self.root / "prepared-primary.wav"
        self.audio_secondary = self.root / "prepared-secondary.wav"
        self.audio_primary.write_bytes(b"fixture")
        self.audio_secondary.write_bytes(b"fixture")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(
        self,
        count: int,
        mode: str,
        *,
        job_id: str = "pipeline-job",
        language: str = "auto",
    ) -> StartJobRequest:
        payload: dict[str, object] = {"speakerCountMode": mode}
        if mode == "manual":
            payload["speakerCount"] = count
        elif mode == "hybrid":
            payload["speakerCountBounds"] = {"min": count, "max": count}
            payload["speakerCountPrior"] = count
        return StartJobRequest(
            job_id=job_id,
            source_path=self.source,
            output_directory=self.output,
            speaker_policy=SpeakerCountPolicy.from_payload(payload),
            language=language,
        )

    def context(self, job_id: str = "pipeline-job") -> AdapterContext:
        return AdapterContext(job_id, self.output, threading.Event())

    def transcript_segment(
        self,
        segment_id: str,
        start_ms: int,
        end_ms: int,
        speaker_id: str,
        *,
        margin: float = 0.05,
        scores: Sequence[tuple[str, float]] | None = None,
        overlapping: bool = False,
        human_locked: bool = False,
        boundary_conflict: bool = False,
        audio_path: str | Path | None = None,
    ) -> TranscriptSegment:
        score_rows = scores or (
            (speaker_id, 0.60),
            (
                "speaker-2" if speaker_id == "speaker-1" else "speaker-1",
                0.55,
            ),
        )
        preparation: dict[str, object] = {
            "provider": {"id": "fixture", "version": "1"}
        }
        if audio_path is not None:
            preparation["audioPath"] = str(audio_path)
        return TranscriptSegment(
            segment_id=segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            speaker_id=speaker_id,
            raw_text=f"中文原文{segment_id}",
            normalized_text=f"中文原文{segment_id}",
            display_text=f"中文原文{segment_id}",
            confidence=0.98,
            speaker_scores=tuple(
                SpeakerScore(score_speaker_id, score)
                for score_speaker_id, score in score_rows
            ),
            speaker_margin=margin,
            overlapping=overlapping,
            human_locked=human_locked,
            evidence={
                "preparation": preparation,
                "boundary": {
                    "provider": {"id": "fixture", "version": "1"},
                    "conflict": boundary_conflict,
                },
            },
            turn_id=f"turn-{segment_id}",
        )

    def assert_cascade_schema(self, event: Mapping[str, object]) -> None:
        self.assertEqual(
            set(event),
            {
                "stage",
                "provider",
                "triggerReason",
                "candidateRange",
                "cache",
                "latencyMs",
                "resource",
                "confidence",
                "invoked",
                "exitReason",
            },
        )
        candidate_range = event["candidateRange"]
        self.assertIsInstance(candidate_range, Mapping)
        self.assertEqual(
            set(candidate_range),
            {
                "scope",
                "sourceCount",
                "candidateCount",
                "maxCandidates",
                "segmentIds",
                "startMs",
                "endMs",
            },
        )
        cache = event["cache"]
        self.assertIsInstance(cache, Mapping)
        self.assertEqual(
            set(cache),
            {
                "stage",
                "requests",
                "hits",
                "misses",
                "recomputations",
            },
        )
        resource = event["resource"]
        self.assertIsInstance(resource, Mapping)
        self.assertEqual(set(resource), {"ramMb", "vramMb"})

    def test_global_sequence_decode_preserves_strong_acoustic_a_b_a(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(2)
        segments = (
            self.transcript_segment(
                "left",
                0,
                900,
                "speaker-1",
                margin=0.80,
                scores=(("speaker-1", 0.95), ("speaker-2", 0.15)),
            ),
            self.transcript_segment(
                "center",
                900,
                1_200,
                "speaker-2",
                margin=0.78,
                scores=(("speaker-2", 0.94), ("speaker-1", 0.16)),
            ),
            self.transcript_segment(
                "right",
                1_200,
                2_100,
                "speaker-1",
                margin=0.80,
                scores=(("speaker-1", 0.95), ("speaker-2", 0.15)),
            ),
        )

        decoded = pipeline._decode_global_speaker_sequence(segments)

        self.assertEqual(
            [segment.speaker_id for segment in decoded],
            ["speaker-1", "speaker-2", "speaker-1"],
        )
        center = decoded[1]
        self.assertEqual(center.revisions, ())
        self.assertEqual(
            center.evidence["speakerSequenceDecode"],
            {
                "method": "acoustic-topm-viterbi-v1",
                "beforeSpeakerId": "speaker-2",
                "afterSpeakerId": "speaker-2",
                "acousticScore": 0.94,
                "reviewStatus": "NOT_REQUIRED",
                "reasonCodes": ["STRONG_ACOUSTIC_ANCHOR"],
                "applied": False,
            },
        )

    def test_global_sequence_decode_uses_only_weak_path_tie_breaking(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(2)
        chronological = {
            "left": self.transcript_segment(
                "left",
                0,
                900,
                "speaker-1",
                scores=(("speaker-1", 0.95), ("speaker-2", 0.05)),
            ),
            "ambiguous": self.transcript_segment(
                "ambiguous",
                900,
                1_200,
                "speaker-2",
                margin=0.01,
                scores=(("speaker-2", 0.60), ("speaker-1", 0.59)),
            ),
            "right": self.transcript_segment(
                "right",
                1_200,
                2_100,
                "speaker-1",
                scores=(("speaker-1", 0.95), ("speaker-2", 0.05)),
            ),
            "speaker-2-anchor": self.transcript_segment(
                "speaker-2-anchor",
                2_100,
                3_000,
                "speaker-2",
                scores=(("speaker-2", 0.96), ("speaker-1", 0.04)),
            ),
        }
        input_order = (
            chronological["speaker-2-anchor"],
            chronological["right"],
            chronological["ambiguous"],
            chronological["left"],
        )

        decoded = pipeline._decode_global_speaker_sequence(input_order)
        by_id = {segment.segment_id: segment for segment in decoded}

        self.assertEqual(
            [segment.segment_id for segment in decoded],
            [segment.segment_id for segment in input_order],
        )
        self.assertEqual(by_id["ambiguous"].speaker_id, "speaker-1")
        self.assertEqual(
            by_id["ambiguous"].revisions[-1].reason_code,
            "GLOBAL_ACOUSTIC_SEQUENCE_DECODE",
        )
        self.assertEqual(
            by_id["ambiguous"].revisions[-1].source,
            "deterministic",
        )
        self.assertTrue(
            by_id["ambiguous"].evidence["speakerSequenceDecode"]["applied"]
        )

    def test_global_sequence_decode_human_lock_is_a_hard_constraint(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(2)
        locked = self.transcript_segment(
            "locked",
            0,
            1_000,
            "speaker-2",
            margin=-0.90,
            scores=(("speaker-1", 0.95), ("speaker-2", 0.05)),
            human_locked=True,
        )

        decoded = pipeline._decode_global_speaker_sequence((locked,))

        self.assertEqual(decoded[0].speaker_id, "speaker-2")
        self.assertEqual(decoded[0].revisions, ())
        self.assertIn(
            "HUMAN_LOCKED",
            decoded[0].evidence["speakerSequenceDecode"]["reasonCodes"],
        )

    def test_global_sequence_decode_preserves_cardinality_and_escalates(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(2)
        segments = (
            self.transcript_segment(
                "speaker-1-a",
                0,
                1_000,
                "speaker-1",
                margin=0.0,
                scores=(("speaker-1", 0.60), ("speaker-2", 0.60)),
            ),
            self.transcript_segment(
                "only-speaker-2",
                1_000,
                2_000,
                "speaker-2",
                margin=0.0,
                scores=(("speaker-1", 0.60), ("speaker-2", 0.60)),
            ),
            self.transcript_segment(
                "speaker-1-b",
                2_000,
                3_000,
                "speaker-1",
                margin=0.0,
                scores=(("speaker-1", 0.60), ("speaker-2", 0.60)),
            ),
        )

        decoded = pipeline._decode_global_speaker_sequence(segments)
        protected = next(
            segment
            for segment in decoded
            if segment.segment_id == "only-speaker-2"
        )

        self.assertEqual(
            {segment.speaker_id for segment in decoded},
            {"speaker-1", "speaker-2"},
        )
        self.assertEqual(
            protected.evidence["speakerSequenceDecode"]["reviewStatus"],
            "REVIEW_REQUIRED",
        )
        self.assertIn(
            "CARDINALITY_CHANGE_REVIEW_REQUIRED",
            protected.evidence["speakerSequenceDecode"]["reasonCodes"],
        )

    def test_global_sequence_decode_supports_129_speaker_score_inventory(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(129)
        scores = tuple(
            (
                f"speaker-{index}",
                0.99 if index == 129 else 0.50 - (index / 10_000),
            )
            for index in range(1, 130)
        )
        segment = self.transcript_segment(
            "speaker-129-segment",
            0,
            1_000,
            "speaker-129",
            margin=0.49,
            scores=scores,
        )

        decoded = pipeline._decode_global_speaker_sequence((segment,))

        self.assertEqual(decoded[0].speaker_id, "speaker-129")
        self.assertEqual(
            decoded[0].evidence["speakerSequenceDecode"]["acousticScore"],
            0.99,
        )
        self.assertEqual(
            decoded[0].evidence["speakerSequenceDecode"]["reviewStatus"],
            "NOT_REQUIRED",
        )

    def test_pyannote_tracks_map_to_canonical_speakers_without_changing_n(
        self,
    ) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            2,
            secondary=FakeSecondaryVerifier(),
            pyannote=FakePyannoteAudit(),
            config=SpeakerPipelineConfig(pyannote_mode="fallback"),
        )
        base = (
            self.transcript_segment(
                "one",
                0,
                1000,
                "speaker-1",
                margin=0.8,
                scores=(("speaker-1", 0.9), ("speaker-2", 0.1)),
            ),
            self.transcript_segment(
                "two",
                1000,
                2000,
                "speaker-1",
                margin=0.6,
                scores=(("speaker-1", 0.8), ("speaker-2", 0.2)),
            ),
            self.transcript_segment(
                "three",
                2000,
                3000,
                "speaker-1",
                margin=0.5,
                scores=(("speaker-1", 0.3), ("speaker-2", 0.8)),
            ),
            self.transcript_segment(
                "four",
                3000,
                4000,
                "speaker-2",
                margin=0.7,
                scores=(("speaker-1", 0.2), ("speaker-2", 0.9)),
            ),
        )
        local_by_segment = {
            "one": "LOCAL_A",
            "two": "LOCAL_A",
            "three": "LOCAL_B",
            "four": "LOCAL_B",
        }
        segments = tuple(
            replace(
                segment,
                evidence={
                    **dict(segment.evidence),
                    "overlap": {
                        "provider": {
                            "id": "pyannote-community-1",
                            "version": "2.0.0",
                        },
                        "speakerTurns": [
                            {
                                "startMs": segment.start_ms,
                                "endMs": segment.end_ms,
                                "localSpeaker": local_by_segment[
                                    segment.segment_id
                                ],
                            }
                        ],
                        "overlapIntervals": [],
                    },
                },
            )
            for segment in base
        )

        mapped = pipeline._apply_pyannote_canonical_mapping(segments)

        self.assertEqual(
            [segment.speaker_id for segment in mapped],
            ["speaker-1", "speaker-1", "speaker-2", "speaker-2"],
        )
        self.assertEqual(
            {segment.speaker_id for segment in mapped},
            {"speaker-1", "speaker-2"},
        )
        changed = mapped[2]
        self.assertEqual(
            changed.revisions[-1].reason_code,
            "PYANNOTE_CANONICAL_TRACK_MAPPING",
        )
        evidence = changed.evidence["pyannoteCanonicalMapping"]
        self.assertTrue(evidence["accepted"])
        self.assertTrue(evidence["applied"])
        self.assertGreater(
            evidence["mappingMargin"],
            evidence["mappingMarginThreshold"],
        )
        self.assertEqual(
            changed.evidence["overlap"]["canonicalSpeakerTurns"],
            [
                {
                    "startMs": 2000,
                    "endMs": 3000,
                    "speakerId": "speaker-2",
                    "localSpeaker": "LOCAL_B",
                }
            ],
        )

    def test_ambiguous_pyannote_mapping_fails_closed(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            2,
            secondary=FakeSecondaryVerifier(),
            pyannote=FakePyannoteAudit(),
            config=SpeakerPipelineConfig(pyannote_mode="fallback"),
        )
        base = (
            self.transcript_segment(
                "one",
                0,
                1000,
                "speaker-1",
                scores=(("speaker-1", 0.5), ("speaker-2", 0.5)),
            ),
            self.transcript_segment(
                "two",
                1000,
                2000,
                "speaker-2",
                scores=(("speaker-1", 0.5), ("speaker-2", 0.5)),
            ),
        )
        segments = tuple(
            replace(
                segment,
                evidence={
                    **dict(segment.evidence),
                    "overlap": {
                        "provider": {
                            "id": "pyannote-community-1",
                            "version": "2.0.0",
                        },
                        "speakerTurns": [
                            {
                                "startMs": segment.start_ms,
                                "endMs": segment.end_ms,
                                "localSpeaker": (
                                    "LOCAL_A"
                                    if segment.segment_id == "one"
                                    else "LOCAL_B"
                                ),
                            }
                        ],
                        "overlapIntervals": [],
                    },
                },
            )
            for segment in base
        )

        mapped = pipeline._apply_pyannote_canonical_mapping(segments)

        self.assertEqual(
            [segment.speaker_id for segment in mapped],
            ["speaker-1", "speaker-2"],
        )
        for segment in mapped:
            evidence = segment.evidence["pyannoteCanonicalMapping"]
            self.assertFalse(evidence["accepted"])
            self.assertFalse(evidence["applied"])
            self.assertIn(
                "PYANNOTE_MAPPING_MARGIN_BELOW_THRESHOLD",
                evidence["blockers"],
            )
            self.assertNotIn(
                "canonicalSpeakerTurns",
                segment.evidence["overlap"],
            )

    def test_global_sequence_decode_ignores_semantic_role_hints(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(2)
        base = self.transcript_segment(
            "semantic-trap",
            0,
            1_000,
            "speaker-1",
            margin=0.70,
            scores=(("speaker-1", 0.90), ("speaker-2", 0.20)),
        )
        trapped = TranscriptSegment(
            **{
                **base.__dict__,
                "evidence": {
                    **dict(base.evidence),
                    "semanticHint": {
                        "suggestedSpeakerId": "speaker-2",
                        "confidence": 1.0,
                    },
                    "questionAnswerRelation": {
                        "suggestedSpeakerId": "speaker-2",
                    },
                },
            }
        )

        plain_result = pipeline._decode_global_speaker_sequence((base,))
        semantic_result = pipeline._decode_global_speaker_sequence((trapped,))

        self.assertEqual(plain_result[0].speaker_id, "speaker-1")
        self.assertEqual(semantic_result[0].speaker_id, "speaker-1")
        self.assertEqual(
            plain_result[0].evidence["speakerSequenceDecode"],
            semantic_result[0].evidence["speakerSequenceDecode"],
        )

    def test_strong_acoustic_a_b_a_is_preserved_without_review(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            2,
            config=SpeakerPipelineConfig(low_margin_threshold=0.18),
        )
        segments = (
            self.transcript_segment(
                "left",
                0,
                900,
                "speaker-1",
                margin=0.45,
                scores=(("speaker-1", 0.90), ("speaker-2", 0.10)),
            ),
            self.transcript_segment(
                "center",
                900,
                1_200,
                "speaker-2",
                margin=0.70,
                scores=(("speaker-2", 0.90), ("speaker-1", 0.20)),
            ),
            self.transcript_segment(
                "right",
                1_200,
                2_100,
                "speaker-1",
                margin=0.45,
                scores=(("speaker-1", 0.90), ("speaker-2", 0.10)),
            ),
            self.transcript_segment(
                "speaker-2-anchor",
                2_100,
                3_000,
                "speaker-2",
                margin=0.45,
                scores=(("speaker-2", 0.90), ("speaker-1", 0.10)),
            ),
        )

        stabilized = pipeline._stabilize_temporal_assignments(segments)
        center = next(
            segment
            for segment in stabilized
            if segment.segment_id == "center"
        )

        self.assertEqual(center.speaker_id, "speaker-2")
        self.assertEqual(center.revisions, ())
        self.assertEqual(
            center.evidence["temporalStabilization"]["reviewStatus"],
            "NOT_REQUIRED",
        )
        self.assertEqual(
            center.evidence["temporalStabilization"]["reasonCode"],
            "SKIPPED_STRONG_ACOUSTIC_EVIDENCE",
        )
        self.assertNotIn(
            "TEMPORAL_CONTINUITY_UNRESOLVED",
            {
                item.reason_code
                for item in center.revisions
            },
        )

    def test_low_margin_a_b_a_blocker_remains_review_required(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            2,
            config=SpeakerPipelineConfig(low_margin_threshold=0.18),
        )
        segments = (
            self.transcript_segment("left", 0, 900, "speaker-1"),
            self.transcript_segment(
                "center",
                900,
                1_200,
                "speaker-2",
                margin=0.05,
                scores=(("speaker-2", 0.60), ("speaker-1", 0.55)),
                boundary_conflict=True,
            ),
            self.transcript_segment("right", 1_200, 2_100, "speaker-1"),
            self.transcript_segment(
                "speaker-2-anchor",
                2_100,
                3_000,
                "speaker-2",
            ),
        )

        stabilized = pipeline._stabilize_temporal_assignments(segments)
        center = next(
            segment
            for segment in stabilized
            if segment.segment_id == "center"
        )

        self.assertEqual(center.speaker_id, "speaker-2")
        self.assertEqual(
            center.evidence["temporalStabilization"]["reviewStatus"],
            "REVIEW_REQUIRED",
        )
        self.assertIn(
            "BOUNDARY_CONFLICT",
            center.evidence["temporalStabilization"]["blockers"],
        )

    def test_safe_short_a_b_a_is_stabilized_without_reordering_input(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            2,
            config=SpeakerPipelineConfig(
                low_margin_threshold=0.18,
                temporal_short_segment_ms=1_500,
                temporal_max_gap_ms=750,
            ),
        )
        chronological = {
            "left": self.transcript_segment("left", 0, 900, "speaker-1"),
            "center": self.transcript_segment(
                "center",
                900,
                1_200,
                "speaker-2",
                margin=0.05,
                scores=(("speaker-2", 0.60), ("speaker-1", 0.55)),
            ),
            "right": self.transcript_segment(
                "right",
                1_200,
                2_100,
                "speaker-1",
            ),
            "speaker-1-tail": self.transcript_segment(
                "speaker-1-tail",
                2_100,
                3_000,
                "speaker-1",
            ),
            "speaker-2-anchor": self.transcript_segment(
                "speaker-2-anchor",
                3_000,
                3_900,
                "speaker-2",
            ),
        }
        input_order = (
            chronological["speaker-2-anchor"],
            chronological["right"],
            chronological["center"],
            chronological["speaker-1-tail"],
            chronological["left"],
        )

        stabilized = pipeline._stabilize_temporal_assignments(input_order)
        by_id = {segment.segment_id: segment for segment in stabilized}

        self.assertEqual(
            [segment.segment_id for segment in stabilized],
            [segment.segment_id for segment in input_order],
        )
        self.assertEqual(by_id["center"].speaker_id, "speaker-1")
        self.assertEqual(
            by_id["center"].evidence["temporalStabilization"]["reviewStatus"],
            "RESOLVED",
        )
        self.assertTrue(
            by_id["center"].evidence["temporalStabilization"]["applied"]
        )
        self.assertEqual(
            {segment.speaker_id for segment in stabilized},
            {"speaker-1", "speaker-2"},
        )

    def test_adjacent_alternating_islands_are_never_ping_pong_reassigned(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            2,
            config=SpeakerPipelineConfig(low_margin_threshold=0.18),
        )
        segments = tuple(
            self.transcript_segment(
                f"segment-{index + 1}",
                index * 300,
                (index + 1) * 300,
                speaker_id,
                margin=0.05,
                scores=(
                    (speaker_id, 0.60),
                    (
                        "speaker-2"
                        if speaker_id == "speaker-1"
                        else "speaker-1",
                        0.55,
                    ),
                ),
            )
            for index, speaker_id in enumerate(
                (
                    "speaker-1",
                    "speaker-2",
                    "speaker-1",
                    "speaker-2",
                    "speaker-1",
                )
            )
        )

        stabilized = pipeline._stabilize_temporal_assignments(segments)

        self.assertEqual(
            [segment.speaker_id for segment in stabilized],
            [segment.speaker_id for segment in segments],
        )
        for segment in stabilized[1:4]:
            evidence = segment.evidence["temporalStabilization"]
            self.assertEqual(evidence["reviewStatus"], "REVIEW_REQUIRED")
            self.assertFalse(evidence["applied"])
            self.assertIn(
                "ADJACENT_ISLAND_AMBIGUITY",
                evidence["blockers"],
            )

    def test_temporal_blockers_are_complete_stable_and_fail_closed(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            3,
            config=SpeakerPipelineConfig(
                low_margin_threshold=0.18,
                temporal_short_segment_ms=1_500,
                temporal_max_gap_ms=750,
            ),
        )
        segments = (
            self.transcript_segment(
                "left",
                0,
                1_000,
                "speaker-1",
                overlapping=True,
            ),
            self.transcript_segment(
                "center",
                900,
                3_000,
                "speaker-2",
                margin=0.05,
                scores=(
                    ("speaker-2", 0.60),
                    ("speaker-3", 0.55),
                    ("speaker-1", 0.50),
                ),
                overlapping=True,
                human_locked=True,
                boundary_conflict=True,
            ),
            self.transcript_segment(
                "right",
                4_000,
                5_000,
                "speaker-1",
            ),
        )

        stabilized = pipeline._stabilize_temporal_assignments(segments)
        center = stabilized[1]
        temporal = center.evidence["temporalStabilization"]

        self.assertEqual(center.speaker_id, "speaker-2")
        self.assertEqual(center.revisions, ())
        self.assertEqual(temporal["reviewStatus"], "REVIEW_REQUIRED")
        self.assertEqual(
            temporal["blockers"],
            [
                "DURATION_EXCEEDS_LIMIT",
                "TEMPORAL_OVERLAP",
                "GAP_EXCEEDS_LIMIT",
                "HUMAN_LOCKED",
                "OVERLAP_PROTECTED",
                "NEIGHBOR_OVERLAP",
                "BOUNDARY_CONFLICT",
                "NEIGHBOR_SPEAKER_OUTSIDE_TOP2",
                "CARDINALITY_CHANGE_REVIEW_REQUIRED",
            ],
        )

    def test_temporal_stabilization_never_removes_last_speaker_support(self) -> None:
        pipeline, _, _, _, _ = self.pipeline(
            2,
            config=SpeakerPipelineConfig(low_margin_threshold=0.18),
        )
        segments = (
            self.transcript_segment("left", 0, 900, "speaker-1"),
            self.transcript_segment(
                "only-speaker-2",
                900,
                1_200,
                "speaker-2",
                margin=0.05,
                scores=(("speaker-2", 0.60), ("speaker-1", 0.55)),
            ),
            self.transcript_segment("right", 1_200, 2_100, "speaker-1"),
        )

        stabilized = pipeline._stabilize_temporal_assignments(segments)
        center = stabilized[1]

        self.assertEqual(center.speaker_id, "speaker-2")
        self.assertEqual(
            {segment.speaker_id for segment in stabilized},
            {"speaker-1", "speaker-2"},
        )
        self.assertEqual(
            center.evidence["temporalStabilization"]["reviewStatus"],
            "REVIEW_REQUIRED",
        )
        self.assertIn(
            "CARDINALITY_CHANGE_REVIEW_REQUIRED",
            center.evidence["temporalStabilization"]["blockers"],
        )

    def test_eres_batch_deduplicates_clips_and_preserves_candidate_order(self) -> None:
        verifier = LocalERes2NetV2Verifier(model_path=self.eres_model)
        segments = {
            "ref-1": self.transcript_segment(
                "ref-1",
                0,
                1_000,
                "speaker-1",
                margin=0.80,
                audio_path=self.audio_primary,
            ),
            "ref-2": self.transcript_segment(
                "ref-2",
                1_000,
                2_000,
                "speaker-2",
                margin=0.80,
                audio_path=self.audio_primary,
            ),
            "candidate-1": self.transcript_segment(
                "candidate-1",
                2_000,
                2_500,
                "speaker-1",
                scores=(("speaker-1", 0.60), ("speaker-2", 0.55)),
                audio_path=self.audio_primary,
            ),
            "candidate-2": self.transcript_segment(
                "candidate-2",
                2_000,
                2_500,
                "speaker-2",
                scores=(("speaker-2", 0.60), ("speaker-1", 0.55)),
                audio_path=self.audio_primary,
            ),
        }
        candidates = (
            ReviewCandidate("candidate-2", ("LOW_MARGIN",), False),
            ReviewCandidate("candidate-1", ("LOW_MARGIN",), False),
        )

        with (
            patch(
                "backend.production_runners._load_audio",
                return_value=([0.0] * 10_000, 1_000),
            ) as load_audio,
            patch.object(
                verifier,
                "_embeddings",
                return_value=[
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 0.0),
                ],
            ) as embeddings,
            patch(
                "backend.production_runners._resource_snapshot",
                return_value={"ramMb": 1.0, "vramMb": 2.0},
            ),
        ):
            proposals = verifier.review_batch(
                candidates,
                segments,
                self.context("eres-dedup"),
            )

        load_audio.assert_called_once_with(str(self.audio_primary))
        embeddings.assert_called_once()
        self.assertEqual(len(embeddings.call_args.args[0]), 3)
        self.assertEqual(
            [proposal.segment_id for proposal in proposals],
            ["candidate-2", "candidate-1"],
        )

    def test_eres_cross_audio_references_load_each_source_once(self) -> None:
        verifier = LocalERes2NetV2Verifier(model_path=self.eres_model)
        segments = {
            "candidate": self.transcript_segment(
                "candidate",
                2_000,
                2_500,
                "speaker-1",
                scores=(("speaker-1", 0.60), ("speaker-2", 0.55)),
                audio_path=self.audio_primary,
            ),
            "ref-1": self.transcript_segment(
                "ref-1",
                0,
                1_000,
                "speaker-1",
                margin=0.80,
                audio_path=self.audio_primary,
            ),
            "ref-2": self.transcript_segment(
                "ref-2",
                1_000,
                2_000,
                "speaker-2",
                margin=0.80,
                audio_path=self.audio_secondary,
            ),
        }

        with (
            patch(
                "backend.production_runners._load_audio",
                return_value=([0.0] * 10_000, 1_000),
            ) as load_audio,
            patch.object(
                verifier,
                "_embeddings",
                return_value=[
                    (1.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                ],
            ) as embeddings,
            patch(
                "backend.production_runners._resource_snapshot",
                return_value={"ramMb": 0.0, "vramMb": 0.0},
            ),
        ):
            verifier.review_batch(
                (ReviewCandidate("candidate", ("LOW_MARGIN",), False),),
                segments,
                self.context("eres-cross-audio"),
            )

        self.assertEqual(load_audio.call_count, 2)
        self.assertEqual(
            {call.args[0] for call in load_audio.call_args_list},
            {str(self.audio_primary), str(self.audio_secondary)},
        )
        embeddings.assert_called_once()

    def test_eres_no_or_single_reference_does_not_touch_audio_or_model(self) -> None:
        verifier = LocalERes2NetV2Verifier(model_path=self.eres_model)
        no_reference = {
            "candidate": self.transcript_segment(
                "candidate",
                0,
                500,
                "speaker-1",
                scores=(("speaker-1", 0.60), ("speaker-2", 0.55)),
            )
        }
        duplicate_scores_single_reference = {
            "candidate": self.transcript_segment(
                "candidate",
                0,
                500,
                "speaker-1",
                scores=(("speaker-1", 0.60), ("speaker-1", 0.55)),
            ),
            "reference": self.transcript_segment(
                "reference",
                500,
                1_500,
                "speaker-1",
                margin=0.80,
            ),
        }

        with (
            patch("backend.production_runners._load_audio") as load_audio,
            patch.object(verifier, "_embeddings") as embeddings,
            patch(
                "backend.production_runners._resource_snapshot",
                return_value={"ramMb": 0.0, "vramMb": 0.0},
            ),
        ):
            no_reference_proposal = verifier.review_batch(
                (ReviewCandidate("candidate", ("LOW_MARGIN",), False),),
                no_reference,
                self.context("eres-no-reference"),
            )[0]
            single_reference_proposal = verifier.review_batch(
                (ReviewCandidate("candidate", ("LOW_MARGIN",), False),),
                duplicate_scores_single_reference,
                self.context("eres-single-reference"),
            )[0]

        self.assertEqual(no_reference_proposal.exit_reason, "NO_REFERENCE")
        self.assertEqual(
            single_reference_proposal.exit_reason,
            "SINGLE_REFERENCE_INSUFFICIENT",
        )
        load_audio.assert_not_called()
        embeddings.assert_not_called()

    def test_eres_reference_selection_and_cache_material_are_stable(self) -> None:
        verifier = LocalERes2NetV2Verifier(model_path=self.eres_model)
        segments = {
            "candidate": self.transcript_segment(
                "candidate",
                2_000,
                2_500,
                "speaker-1",
                scores=(("speaker-1", 0.60), ("speaker-2", 0.55)),
                audio_path=self.audio_primary,
            ),
            "speaker-1-z": self.transcript_segment(
                "speaker-1-z",
                0,
                1_000,
                "speaker-1",
                margin=0.80,
                audio_path=self.audio_secondary,
            ),
            "speaker-1-a": self.transcript_segment(
                "speaker-1-a",
                0,
                1_000,
                "speaker-1",
                margin=0.80,
                audio_path=self.audio_primary,
            ),
            "speaker-2": self.transcript_segment(
                "speaker-2",
                1_000,
                2_000,
                "speaker-2",
                margin=0.80,
                audio_path=self.audio_secondary,
            ),
        }

        material = verifier.cache_material(
            ReviewCandidate("candidate", ("LOW_MARGIN",), False),
            segments,
        )

        self.assertEqual(
            [item["segmentId"] for item in material["references"]],
            ["speaker-1-a", "speaker-2"],
        )
        self.assertEqual(
            [item["audioPath"] for item in material["references"]],
            [str(self.audio_primary), str(self.audio_secondary)],
        )

    def test_eres_invalid_candidates_and_embedding_dimensions_fail_closed(self) -> None:
        verifier = LocalERes2NetV2Verifier(model_path=self.eres_model)
        segment = self.transcript_segment(
            "candidate",
            0,
            500,
            "speaker-1",
        )

        for candidates in (
            (ReviewCandidate("missing", ("LOW_MARGIN",), False),),
            (
                ReviewCandidate("candidate", ("LOW_MARGIN",), False),
                ReviewCandidate("candidate", ("SHORT_SEGMENT",), False),
            ),
        ):
            with self.subTest(
                candidates=[candidate.segment_id for candidate in candidates]
            ):
                with self.assertRaises(WorkerError) as captured:
                    verifier.review_batch(
                        candidates,
                        {"candidate": segment},
                        self.context("eres-invalid-candidate"),
                    )
                self.assertEqual(
                    captured.exception.code,
                    "ERES2NETV2_CANDIDATE_INVALID",
                )

        with self.assertRaises(WorkerError) as captured:
            verifier.cache_material(
                ReviewCandidate("missing", ("LOW_MARGIN",), False),
                {"candidate": segment},
            )
        self.assertEqual(
            captured.exception.code,
            "ERES2NETV2_CANDIDATE_INVALID",
        )

        with self.assertRaises(WorkerError) as captured:
            verifier._cosine((1.0, 0.0), (1.0,))
        self.assertEqual(
            captured.exception.code,
            "ERES2NETV2_RESULT_INVALID",
        )
        self.assertEqual(
            captured.exception.details,
            {"leftDimensions": 2, "rightDimensions": 1},
        )

    def pipeline(
        self,
        count: int,
        *,
        windows: int | None = None,
        cache: InMemoryStageCache | None = None,
        preparation: FakePreparationAdapter | None = None,
        asr: FakeAsrAdapter | None = None,
        cam: FakeCamPlusAdapter | None = None,
        overlap: FakeOverlapAdapter | None = None,
        secondary: FakeSecondaryVerifier | None = None,
        pyannote: FakePyannoteAudit | None = None,
        config: SpeakerPipelineConfig | None = None,
    ):
        preparation = preparation or FakePreparationAdapter(windows or count)
        asr = asr or FakeAsrAdapter()
        cam = cam or FakeCamPlusAdapter(count)
        overlap = overlap or FakeOverlapAdapter()
        pipeline = SpeakerPipeline(
            preparation_adapter=preparation,
            asr_adapter=asr,
            embedding_adapter=cam,
            overlap_adapter=overlap,
            secondary_adapter=secondary,
            pyannote_adapter=pyannote,
            cache=cache,
            config=config,
        )
        return pipeline, preparation, asr, cam, overlap

    def test_unavailable_and_compatibility_overlap_adapters_are_fail_closed(
        self,
    ) -> None:
        window = SpeechWindow(
            window_id="window-1",
            start_ms=0,
            end_ms=1_000,
        )
        expected_evidence = {
            "detectorStatus": "UNAVAILABLE",
            "overlapDetectorRun": False,
            "reviewStatus": "REVIEW_REQUIRED",
            "reasonCode": "OVERLAP_DETECTOR_UNAVAILABLE",
        }
        for adapter in (UnavailableOverlapAdapter(), NoOverlapAdapter()):
            with self.subTest(adapter=adapter.adapter_id):
                self.assertNotEqual(adapter.version, "1")
                decisions = adapter.detect_batch(
                    None,
                    (window,),
                    self.context(f"{adapter.adapter_id}-decision"),
                )
                self.assertEqual(len(decisions), 1)
                decision = decisions[0]
                self.assertFalse(decision.overlapping)
                self.assertEqual(decision.confidence, 0.0)
                self.assertEqual(dict(decision.evidence), expected_evidence)
                serialized = decision.as_dict()
                self.assertEqual(serialized["confidence"], 0.0)
                self.assertEqual(serialized["evidence"], expected_evidence)
                restored = OverlapDecision.from_mapping(serialized)
                self.assertEqual(restored, decision)

    def test_non_lexical_asr_window_is_audited_and_excluded_from_diarization(
        self,
    ) -> None:
        asr = FakeAsrAdapter(mode="reject-last")
        pipeline, _, _, cam, overlap = self.pipeline(
            1,
            windows=3,
            asr=asr,
        )

        result = pipeline.transcribe(
            self.request(1, "manual", job_id="non-lexical-window"),
            self.context("non-lexical-window"),
        )

        self.assertEqual(len(result.segments), 2)
        self.assertEqual(cam.calls, [("window-1", "window-2")])
        self.assertEqual(overlap.calls, [("window-1", "window-2")])
        self.assertEqual(
            result.pipeline_metrics["policy"][
                "asrRejectedNonLexicalWindowCount"
            ],
            1,
        )
        rejection = next(
            item
            for item in result.pipeline_metrics["cascade"]
            if item["stage"] == "asr-non-lexical-rejection"
        )
        self.assertEqual(
            rejection["candidateRange"]["segmentIds"],
            ["window-3"],
        )
        self.assertEqual(
            rejection["exitReason"],
            "REJECTED_WITHOUT_FABRICATED_TEXT",
        )

    def test_manual_count_partitions_one_continuous_vad_window(self) -> None:
        preparation = FakePreparationAdapter(
            1,
            window_ranges={"window-1": (0, 5_000)},
        )
        pipeline, _, asr, cam, overlap = self.pipeline(
            5,
            windows=1,
            preparation=preparation,
        )

        result = pipeline.transcribe(
            self.request(5, "manual", job_id="manual-continuous-vad"),
            self.context("manual-continuous-vad"),
        )

        self.assertEqual(len(result.segments), 5)
        self.assertEqual(
            {segment.speaker_id for segment in result.segments},
            {f"speaker-{index}" for index in range(1, 6)},
        )
        self.assertEqual(
            [(segment.start_ms, segment.end_ms) for segment in result.segments],
            [
                (0, 1_000),
                (1_000, 2_000),
                (2_000, 3_000),
                (3_000, 4_000),
                (4_000, 5_000),
            ],
        )
        self.assertEqual(
            [segment.raw_text for segment in result.segments],
            ["词1", "词2", "词3", "词4", "词5"],
        )
        self.assertEqual(
            {segment.turn_id for segment in result.segments},
            {"turn-1"},
        )
        self.assertTrue(
            all(
                segment.evidence["speakerCountPartition"]["reviewRequired"]
                for segment in result.segments
            )
        )
        self.assertTrue(
            all(
                segment.evidence["asr"]["asrProjection"]["sourceWindowId"]
                == "window-1"
                for segment in result.segments
            )
        )
        expected_windows = tuple(
            f"window-1.cardinality-{index:02d}" for index in range(1, 6)
        )
        self.assertEqual(asr.calls, [("window-1",)])
        self.assertEqual(cam.calls, [expected_windows])
        self.assertEqual(overlap.calls, [expected_windows])
        self.assertTrue(
            result.pipeline_metrics["policy"]["speakerCountPartitionApplied"]
        )

    def test_auto_count_samples_long_vad_and_merges_same_voice(self) -> None:
        preparation = FakePreparationAdapter(
            1,
            window_ranges={"window-1": (0, 5_000)},
        )
        pipeline, _, asr, cam, overlap = self.pipeline(
            1,
            windows=1,
            preparation=preparation,
            cam=FakeCamPlusAdapter(1, identical=True),
        )

        result = pipeline.transcribe(
            self.request(1, "auto", job_id="auto-continuous-vad"),
            self.context("auto-continuous-vad"),
        )

        expected_windows = tuple(
            f"window-1.cardinality-{index:02d}" for index in range(1, 6)
        )
        self.assertEqual(asr.calls, [("window-1",)])
        self.assertEqual(cam.calls, [expected_windows])
        self.assertEqual(overlap.calls, [expected_windows])
        self.assertIsNotNone(result.speaker_count_estimate)
        self.assertEqual(result.speaker_count_estimate.estimated_count, 1)
        self.assertEqual(
            {segment.speaker_id for segment in result.segments},
            {"speaker-1"},
        )
        self.assertTrue(
            all(
                segment.evidence["speakerCountPartition"]["method"]
                == "auto-acoustic-contiguous-partition-v1"
                for segment in result.segments
            )
        )
        self.assertEqual(
            result.pipeline_metrics["policy"]["speakerCountPartitionMode"],
            "auto",
        )

    def test_auto_count_short_audio_caps_evidence_and_requires_review(
        self,
    ) -> None:
        preparation = FakePreparationAdapter(
            1,
            window_ranges={"window-1": (0, 2_020)},
        )
        pipeline, _, _, cam, _ = self.pipeline(
            1,
            windows=1,
            preparation=preparation,
            cam=FakeCamPlusAdapter(1, identical=True),
        )

        result = pipeline.transcribe(
            self.request(1, "auto", job_id="auto-short-audio"),
            self.context("auto-short-audio"),
        )

        self.assertEqual(
            cam.calls,
            [
                (
                    "window-1.cardinality-01",
                    "window-1.cardinality-02",
                )
            ],
        )
        self.assertEqual(result.speaker_count_estimate.estimated_count, 1)
        policy = result.pipeline_metrics["policy"]
        self.assertTrue(policy["speakerCountPartitionApplied"])
        self.assertTrue(policy["speakerCountPartitionCapacityLimited"])
        self.assertEqual(policy["speakerCountPartitionTargetEvidenceWindows"], 3)
        self.assertEqual(policy["speakerCountPartitionEvidenceWindows"], 2)
        self.assertTrue(
            all(
                segment.evidence["speakerCountPartition"][
                    "capacityLimited"
                ]
                is True
                for segment in result.segments
            )
        )
        self.assertTrue(
            all(
                segment.evidence["speakerCountPartition"]["reasonCode"]
                == "AUTO_COUNT_EVIDENCE_CAPACITY_LIMITED"
                for segment in result.segments
            )
        )

    def test_auto_count_uses_audited_full_timeline_pyannote_prior(self) -> None:
        overlap = FakePyannoteOverlapAdapter()
        second_axis = (1.0 - 0.99**2) ** 0.5
        vectors = {
            **{
                f"window-{index}": (1.0, 0.0, (index - 2) * 0.001)
                for index in range(1, 4)
            },
            **{
                f"window-{index}": (
                    0.99,
                    second_axis,
                    (index - 5) * 0.001,
                )
                for index in range(4, 7)
            },
        }
        pipeline, _, _, _, _ = self.pipeline(
            6,
            overlap=overlap,
            cam=FakeCamPlusAdapter(2, vectors_by_window_id=vectors),
            secondary=FakeSecondaryVerifier(),
            pyannote=FakePyannoteAudit(),
            config=SpeakerPipelineConfig(pyannote_mode="fallback"),
        )

        result = pipeline.transcribe(
            self.request(6, "auto", job_id="pyannote-count-prior"),
            self.context("pyannote-count-prior"),
        )

        self.assertEqual(result.speaker_count_estimate.estimated_count, 2)
        self.assertLessEqual(result.speaker_count_estimate.candidate_min, 1)
        self.assertGreaterEqual(result.speaker_count_estimate.candidate_max, 2)
        policy = result.pipeline_metrics["policy"]
        self.assertEqual(policy["pyannoteSpeakerCountPriorStatus"], "eligible")
        self.assertEqual(policy["pyannoteSpeakerCountPriorObserved"], 2)
        self.assertEqual(policy["pyannoteSpeakerCountPriorUsed"], 2)
        self.assertTrue(policy["pyannoteSpeakerCountPriorApplied"])
        self.assertFalse(policy["pyannoteSpeakerCountPriorConflict"])
        self.assertIn(
            "PYANNOTE_FULL_TIMELINE_PRIOR:1->2",
            policy["speakerCountCorrectionPath"],
        )

    def test_pyannote_prior_does_not_override_large_acoustic_gap(self) -> None:
        overlap = FakePyannoteOverlapAdapter()
        pipeline, _, _, _, _ = self.pipeline(
            6,
            overlap=overlap,
            secondary=FakeSecondaryVerifier(),
            pyannote=FakePyannoteAudit(),
            config=SpeakerPipelineConfig(pyannote_mode="fallback"),
        )

        result = pipeline.transcribe(
            self.request(6, "auto", job_id="pyannote-count-conflict"),
            self.context("pyannote-count-conflict"),
        )

        self.assertEqual(result.speaker_count_estimate.estimated_count, 6)
        policy = result.pipeline_metrics["policy"]
        self.assertEqual(policy["pyannoteSpeakerCountPriorObserved"], 2)
        self.assertEqual(policy["pyannoteSpeakerCountPriorUsed"], 2)
        self.assertFalse(policy["pyannoteSpeakerCountPriorApplied"])
        self.assertTrue(policy["pyannoteSpeakerCountPriorConflict"])

    def test_inconsistent_pyannote_full_timeline_evidence_is_not_used(
        self,
    ) -> None:
        overlap = FakePyannoteOverlapAdapter(inconsistent_digest=True)
        pipeline, _, _, _, _ = self.pipeline(
            6,
            overlap=overlap,
            secondary=FakeSecondaryVerifier(),
            pyannote=FakePyannoteAudit(),
            config=SpeakerPipelineConfig(pyannote_mode="fallback"),
        )

        result = pipeline.transcribe(
            self.request(6, "auto", job_id="pyannote-count-inconsistent"),
            self.context("pyannote-count-inconsistent"),
        )

        self.assertEqual(result.speaker_count_estimate.estimated_count, 6)
        policy = result.pipeline_metrics["policy"]
        self.assertEqual(
            policy["pyannoteSpeakerCountPriorStatus"],
            "inconsistent-full-timeline-evidence",
        )
        self.assertIsNone(policy["pyannoteSpeakerCountPriorUsed"])
        self.assertFalse(policy["pyannoteSpeakerCountPriorApplied"])

    def test_single_pyannote_track_is_conflict_evidence_not_count_prior(
        self,
    ) -> None:
        overlap = FakePyannoteOverlapAdapter(observed_speaker_count=1)
        pipeline, _, _, _, _ = self.pipeline(
            6,
            overlap=overlap,
            secondary=FakeSecondaryVerifier(),
            pyannote=FakePyannoteAudit(),
            config=SpeakerPipelineConfig(pyannote_mode="fallback"),
        )

        result = pipeline.transcribe(
            self.request(6, "auto", job_id="pyannote-single-track"),
            self.context("pyannote-single-track"),
        )

        self.assertEqual(result.speaker_count_estimate.estimated_count, 6)
        policy = result.pipeline_metrics["policy"]
        self.assertEqual(
            policy["pyannoteSpeakerCountPriorStatus"],
            "single-track-not-independent-count-evidence",
        )
        self.assertEqual(policy["pyannoteSpeakerCountPriorObserved"], 1)
        self.assertIsNone(policy["pyannoteSpeakerCountPriorUsed"])
        self.assertFalse(policy["pyannoteSpeakerCountPriorApplied"])
        self.assertTrue(policy["pyannoteSpeakerCountPriorConflict"])

    def test_speaker_partition_requires_forced_alignment_timestamps(
        self,
    ) -> None:
        class MissingTimestampAsr(FakeAsrAdapter):
            def transcribe_batch(self, *args, **kwargs):
                values = super().transcribe_batch(*args, **kwargs)
                return [
                    replace(item, evidence={"model": "Qwen3-ASR-1.7B"})
                    for item in values
                ]

        preparation = FakePreparationAdapter(
            1,
            window_ranges={"window-1": (0, 5_000)},
        )
        pipeline, _, _, _, _ = self.pipeline(
            1,
            windows=1,
            preparation=preparation,
            asr=MissingTimestampAsr(),
        )

        with self.assertRaises(WorkerError) as captured:
            pipeline.transcribe(
                self.request(1, "auto", job_id="auto-missing-alignment"),
                self.context("auto-missing-alignment"),
            )

        self.assertEqual(
            captured.exception.code,
            "ASR_TIMESTAMPS_REQUIRED_FOR_SPEAKER_PARTITION",
        )

    def test_manual_count_fails_when_audio_cannot_support_evidence_windows(
        self,
    ) -> None:
        pipeline, _, _, _, _ = self.pipeline(5, windows=1)

        with self.assertRaises(WorkerError) as captured:
            pipeline.transcribe(
                self.request(5, "manual", job_id="manual-too-short"),
                self.context("manual-too-short"),
            )

        self.assertEqual(
            captured.exception.code,
            "SPEAKER_COUNT_AUDIO_TOO_SHORT",
        )
        self.assertEqual(
            captured.exception.details["minimumPartitionMs"],
            700,
        )

    def test_gpu_stage_resources_release_once_in_strict_execution_order(
        self,
    ) -> None:
        events: list[str] = []

        class LifecycleAsr(FakeAsrAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.release_calls = 0

            def transcribe_batch(self, *args, **kwargs):
                events.append("asr-inference")
                return super().transcribe_batch(*args, **kwargs)

            def release_resources(self) -> None:
                self.release_calls += 1
                events.append("asr-release")

        class LifecycleCam(FakeCamPlusAdapter):
            def __init__(self) -> None:
                super().__init__(2)
                self.release_calls = 0

            def release_resources(self) -> None:
                self.release_calls += 1
                events.append("cam-release")

            def embed_batch(self, *args, **kwargs):
                events.append("cam-inference")
                return super().embed_batch(*args, **kwargs)

        cache = InMemoryStageCache()
        asr = LifecycleAsr()
        cam = LifecycleCam()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            cache=cache,
            asr=asr,
            cam=cam,
        )
        request = self.request(2, "manual", job_id="resource-order")

        pipeline.transcribe(request, self.context("resource-order"))

        self.assertEqual(
            events[:5],
            [
                "cam-release",
                "asr-inference",
                "asr-release",
                "cam-inference",
                "cam-release",
            ],
        )
        self.assertEqual(asr.release_calls, 1)
        self.assertEqual(cam.release_calls, 2)

        pipeline.transcribe(request, self.context("resource-order"))

        self.assertEqual(asr.release_calls, 2)
        self.assertEqual(cam.release_calls, 4)
        self.assertEqual(events.count("asr-inference"), 1)
        self.assertEqual(events.count("cam-inference"), 1)

    def test_worker_residency_retains_models_until_explicit_release(self) -> None:
        class LifecycleAsr(FakeAsrAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.release_calls = 0

            def release_resources(self) -> None:
                self.release_calls += 1

        class LifecycleCam(FakeCamPlusAdapter):
            def __init__(self) -> None:
                super().__init__(2)
                self.release_calls = 0

            def release_resources(self) -> None:
                self.release_calls += 1

        class LifecycleOverlap(FakeOverlapAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.release_calls = 0

            def release_resources(self) -> None:
                self.release_calls += 1

        asr = LifecycleAsr()
        cam = LifecycleCam()
        overlap = LifecycleOverlap()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            asr=asr,
            cam=cam,
            overlap=overlap,
            config=SpeakerPipelineConfig(model_residency="worker"),
        )

        pipeline.transcribe(
            self.request(2, "manual", job_id="worker-residency"),
            self.context("worker-residency"),
        )

        self.assertEqual(asr.release_calls, 0)
        self.assertEqual(cam.release_calls, 0)
        self.assertEqual(overlap.release_calls, 0)

        pipeline.release_resources()

        self.assertEqual(asr.release_calls, 1)
        self.assertEqual(cam.release_calls, 1)
        self.assertEqual(overlap.release_calls, 1)

    def test_cam_eres_and_pyannote_release_in_strict_cascade_order(
        self,
    ) -> None:
        events: list[str] = []

        class LifecycleAsr(FakeAsrAdapter):
            def transcribe_batch(self, *args, **kwargs):
                events.append("asr-inference")
                return super().transcribe_batch(*args, **kwargs)

            def release_resources(self) -> None:
                events.append("asr-release")

        class LifecycleCam(FakeCamPlusAdapter):
            def release_resources(self) -> None:
                events.append("cam-release")

            def embed_batch(self, *args, **kwargs):
                events.append("cam-inference")
                return super().embed_batch(*args, **kwargs)

        class LifecycleSecondary(FakeSecondaryVerifier):
            def review_batch(self, *args, **kwargs):
                events.append("eres-inference")
                return super().review_batch(*args, **kwargs)

            def release_resources(self) -> None:
                events.append("eres-release")

        class LifecyclePyannote(FakePyannoteAudit):
            def review_batch(self, *args, **kwargs):
                events.append("pyannote-inference")
                return super().review_batch(*args, **kwargs)

            def release_resources(self) -> None:
                events.append("pyannote-release")

        secondary = LifecycleSecondary(
            exit_reasons={
                f"window-{index}": "LOW_MARGIN_UNRESOLVED"
                for index in range(1, 5)
            }
        )
        pyannote = LifecyclePyannote()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=8,
            preparation=FakePreparationAdapter(
                8,
                boundary_conflict_ids={
                    f"window-{index}" for index in range(1, 9)
                },
            ),
            asr=LifecycleAsr(),
            cam=LifecycleCam(
                2,
                vectors_by_window_id={
                    "window-1": (1.0, 0.0),
                    "window-2": (1.0, 0.0),
                    "window-3": (0.0, 1.0),
                    "window-4": (0.0, 1.0),
                    "window-5": (1.0, 0.0),
                    "window-6": (1.0, 0.0),
                    "window-7": (0.0, 1.0),
                    "window-8": (0.0, 1.0),
                },
            ),
            secondary=secondary,
            pyannote=pyannote,
            config=SpeakerPipelineConfig(
                high_margin_threshold=3.0,
                max_secondary_fraction=0.5,
                pyannote_mode="fallback",
            ),
        )

        pipeline.transcribe(
            self.request(2, "manual", job_id="strict-cascade-release"),
            self.context("strict-cascade-release"),
        )

        self.assertEqual(
            events,
            [
                "cam-release",
                "asr-inference",
                "asr-release",
                "cam-inference",
                "cam-release",
                "eres-inference",
                "eres-release",
                "pyannote-inference",
                "pyannote-release",
            ],
        )

    def test_secondary_resources_release_on_error_and_cancellation(
        self,
    ) -> None:
        class FailingSecondary(FakeSecondaryVerifier):
            def __init__(self, *, cancel: bool) -> None:
                super().__init__()
                self.cancel = cancel
                self.release_calls = 0

            def review_batch(self, candidates, segments, context):
                if self.cancel:
                    context.cancellation.set()
                    context.raise_if_cancelled()
                raise WorkerError(
                    "ERES2NETV2_INFERENCE_FAILED",
                    "synthetic verifier failure",
                )

            def release_resources(self) -> None:
                self.release_calls += 1

        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                secondary = FailingSecondary(cancel=cancel)
                pipeline, _, _, _, _ = self.pipeline(
                    2,
                    windows=4,
                    preparation=FakePreparationAdapter(
                        4,
                        boundary_conflict_ids={
                            f"window-{index}" for index in range(1, 5)
                        },
                    ),
                    secondary=secondary,
                    config=SpeakerPipelineConfig(
                        high_margin_threshold=3.0,
                        max_secondary_fraction=0.5,
                    ),
                )
                with self.assertRaises(
                    JobCancelled if cancel else WorkerError
                ):
                    pipeline.transcribe(
                        self.request(
                            2,
                            "manual",
                            job_id=f"secondary-release-{cancel}",
                        ),
                        self.context(f"secondary-release-{cancel}"),
                    )
                self.assertEqual(secondary.release_calls, 1)

    def test_asr_resources_release_before_all_non_lexical_failure(
        self,
    ) -> None:
        class AllNonLexicalAsr(FakeAsrAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.release_calls = 0

            def transcribe_batch(
                self,
                prepared,
                windows,
                context,
                *,
                requested_language: str,
            ):
                context.raise_if_cancelled()
                self.calls.append(
                    tuple(window.window_id for window in windows)
                )
                self.requested_languages.append(requested_language)
                return [
                    AsrHypothesis(
                        window_id=window.window_id,
                        text="",
                        confidence=0.0,
                        evidence={
                            "disposition": "rejected-non-lexical",
                            "rejectionReason": (
                                "EMPTY_AFTER_INDIVIDUAL_RETRY"
                            ),
                        },
                    )
                    for window in windows
                ]

            def release_resources(self) -> None:
                self.release_calls += 1

        asr = AllNonLexicalAsr()
        pipeline, _, _, cam, _ = self.pipeline(2, asr=asr)

        with self.assertRaises(WorkerError) as captured:
            pipeline.transcribe(
                self.request(2, "manual", job_id="all-non-lexical"),
                self.context("all-non-lexical"),
            )

        self.assertEqual(
            captured.exception.code,
            "NO_TRANSCRIBABLE_SPEECH",
        )
        voice_activity = captured.exception.details["voiceActivity"]
        self.assertEqual(
            voice_activity["classification"],
            "no-lexical-speech-detected",
        )
        self.assertTrue(voice_activity["hasSpeechCandidates"])
        self.assertFalse(voice_activity["hasTranscribableSpeech"])
        self.assertEqual(asr.release_calls, 1)
        self.assertEqual(cam.calls, [])

    def test_asr_resources_release_once_when_asr_stage_raises(self) -> None:
        class FailingAsr(FakeAsrAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.release_calls = 0

            def transcribe_batch(self, *args, **kwargs):
                raise WorkerError(
                    "QWEN3_ASR_INFERENCE_FAILED",
                    "synthetic ASR failure",
                )

            def release_resources(self) -> None:
                self.release_calls += 1

        asr = FailingAsr()
        pipeline, _, _, cam, _ = self.pipeline(
            2,
            asr=asr,
            config=SpeakerPipelineConfig(model_residency="worker"),
        )

        with self.assertRaises(WorkerError) as captured:
            pipeline.transcribe(
                self.request(2, "manual", job_id="asr-release-error"),
                self.context("asr-release-error"),
            )

        self.assertEqual(
            captured.exception.code,
            "QWEN3_ASR_INFERENCE_FAILED",
        )
        self.assertEqual(asr.release_calls, 1)
        self.assertEqual(cam.calls, [])

    def test_default_overlap_unavailable_routes_human_review_without_eres(
        self,
    ) -> None:
        secondary = FakeSecondaryVerifier()
        pipeline = SpeakerPipeline(
            preparation_adapter=FakePreparationAdapter(2),
            asr_adapter=FakeAsrAdapter(),
            embedding_adapter=FakeCamPlusAdapter(1),
            secondary_adapter=secondary,
        )
        self.assertIsInstance(
            pipeline.overlap_adapter,
            UnavailableOverlapAdapter,
        )
        self.assertNotIsInstance(pipeline.overlap_adapter, NoOverlapAdapter)
        result = pipeline.transcribe(
            self.request(1, "manual", job_id="overlap-unavailable"),
            self.context("overlap-unavailable"),
        )

        self.assertEqual(secondary.calls, [])
        for segment in result.segments:
            with self.subTest(segment=segment.segment_id):
                self.assertFalse(segment.overlapping)
                overlap = segment.evidence["overlap"]
                self.assertEqual(overlap["confidence"], 0.0)
                self.assertEqual(overlap["detectorStatus"], "UNAVAILABLE")
                self.assertFalse(overlap["overlapDetectorRun"])
                self.assertEqual(
                    overlap["reviewStatus"],
                    "REVIEW_REQUIRED",
                )
                self.assertEqual(
                    overlap["reasonCode"],
                    "OVERLAP_DETECTOR_UNAVAILABLE",
                )
                selective = segment.evidence["selectiveReview"]
                self.assertEqual(
                    selective["reasonCode"],
                    "OVERLAP_DETECTOR_UNAVAILABLE",
                )
                self.assertEqual(
                    selective["exitReason"],
                    "OVERLAP_DETECTOR_UNAVAILABLE",
                )

        queue = build_review_queue(
            job_id="overlap-unavailable",
            policy=SpeakerCountPolicy.from_payload(
                {"speakerCountMode": "manual", "speakerCount": 1}
            ),
            estimate=result.speaker_count_estimate,
            segments=result.segments,
            count_confidence_threshold=0.0,
            segment_confidence_threshold=0.0,
            speaker_margin_threshold=-1.0,
            range_width_threshold=8,
        )
        overlap_items = [
            item
            for item in queue["items"]
            if item["reasonCode"] == "OVERLAP_DETECTOR_UNAVAILABLE"
        ]
        self.assertEqual(len(overlap_items), len(result.segments))
        self.assertNotIn(
            "OVERLAP_REVIEW_REQUIRED",
            {item["reasonCode"] for item in queue["items"]},
        )

    def test_real_detector_non_overlap_is_recorded_as_evaluated(self) -> None:
        overlap_adapter = FakeOverlapAdapter()
        pipeline = SpeakerPipeline(
            preparation_adapter=FakePreparationAdapter(2),
            asr_adapter=FakeAsrAdapter(),
            embedding_adapter=FakeCamPlusAdapter(1),
            overlap_adapter=overlap_adapter,
        )
        result = pipeline.transcribe(
            self.request(1, "manual", job_id="overlap-evaluated"),
            self.context("overlap-evaluated"),
        )

        self.assertEqual(
            overlap_adapter.calls,
            [("window-1", "window-2")],
        )
        for segment in result.segments:
            with self.subTest(segment=segment.segment_id):
                overlap = segment.evidence["overlap"]
                self.assertFalse(segment.overlapping)
                self.assertEqual(overlap["confidence"], 0.99)
                self.assertEqual(overlap["detectorStatus"], "EVALUATED")
                self.assertTrue(overlap["overlapDetectorRun"])
                self.assertEqual(overlap["reviewStatus"], "NOT_REQUIRED")
                self.assertNotEqual(
                    overlap.get("reasonCode"),
                    "OVERLAP_DETECTOR_UNAVAILABLE",
                )

    def test_dynamic_n_and_all_three_count_modes(self) -> None:
        for count in (1, 2, 5, 8, 13):
            for mode in ("auto", "manual", "hybrid"):
                with self.subTest(count=count, mode=mode):
                    pipeline, _, _, cam, _ = self.pipeline(count)
                    result = pipeline.transcribe(
                        self.request(
                            count,
                            mode,
                            job_id=f"dynamic-{mode}-{count}",
                        ),
                        self.context(f"dynamic-{mode}-{count}"),
                    )
                    self.assertEqual(
                        result.speaker_count_estimate.estimated_count,
                        count,
                    )
                    self.assertEqual(
                        {segment.speaker_id for segment in result.segments},
                        {
                            f"speaker-{index}"
                            for index in range(1, count + 1)
                        },
                    )
                    self.assertEqual(sum(len(batch) for batch in cam.calls), count)
                    self.assertEqual(
                        result.pipeline_metrics["policy"]["localLlmMode"],
                        "disabled",
                    )
                    self.assertFalse(
                        result.pipeline_metrics["policy"]["localLlmAutoApply"]
                    )
                    self.assertIn(
                        "source-hashing",
                        result.pipeline_metrics["runtime"]["stages"],
                    )

    def test_stage_cache_hit_and_single_item_corruption_recompute(self) -> None:
        cache = InMemoryStageCache()
        pipeline, preparation, asr, cam, _ = self.pipeline(5, cache=cache)
        request = self.request(5, "manual")
        pipeline.transcribe(request, self.context())
        self.assertEqual(preparation.calls, 1)
        self.assertEqual(sum(len(batch) for batch in asr.calls), 5)
        self.assertEqual(sum(len(batch) for batch in cam.calls), 5)
        self.assertEqual(
            {stage for stage, _ in cache.keys()}
            >= {
                "normalize",
                "vad",
                "boundary",
                "asr",
                "campp-embedding",
                "overlap",
                "clustering",
            },
            True,
        )

        corrupt_key = cache.keys("asr")[0][1]
        cache.set_raw("asr", corrupt_key, {"broken": True})
        second = pipeline.transcribe(request, self.context())
        self.assertEqual(preparation.calls, 1)
        self.assertEqual(sum(len(batch) for batch in asr.calls), 6)
        self.assertEqual(sum(len(batch) for batch in cam.calls), 5)
        asr_cache = second.pipeline_metrics["cache"]["byStage"]["asr"]
        self.assertEqual(asr_cache["hits"], 4)
        self.assertEqual(asr_cache["recomputations"], 1)

    def test_malformed_v7_clustering_cache_recomputes_at_the_same_key(
        self,
    ) -> None:
        cache = InMemoryStageCache()
        pipeline, preparation, asr, cam, _ = self.pipeline(5, cache=cache)
        request = self.request(5, "manual", job_id="malformed-cluster-cache")
        first = pipeline.transcribe(
            request,
            self.context("malformed-cluster-cache"),
        )

        cluster_key = cache.keys("clustering")[0][1]
        cached = cache.read("clustering", cluster_key)
        self.assertTrue(cached.hit)
        malformed = copy.deepcopy(cached.value)
        malformed["selectionMethod"] = (
            "dynamic-n-adaptive-resample-stability-v9"
        )
        cache.set_raw("clustering", cluster_key, malformed)

        second = pipeline.transcribe(
            request,
            self.context("malformed-cluster-cache"),
        )

        self.assertEqual(
            second.speaker_count_estimate.estimated_count,
            first.speaker_count_estimate.estimated_count,
        )
        self.assertEqual(preparation.calls, 1)
        self.assertEqual(sum(len(batch) for batch in asr.calls), 5)
        self.assertEqual(sum(len(batch) for batch in cam.calls), 5)
        clustering_cache = second.pipeline_metrics["cache"]["byStage"][
            "clustering"
        ]
        self.assertEqual(clustering_cache["hits"], 0)
        self.assertEqual(clustering_cache["misses"], 1)
        self.assertEqual(clustering_cache["recomputations"], 1)
        repaired = cache.read("clustering", cluster_key)
        self.assertTrue(repaired.hit)
        self.assertEqual(
            repaired.value["selectionMethod"],
            "dynamic-n-adaptive-resample-stability-v12",
        )

    def test_asr_language_isolated_cache_reuses_acoustic_stages(self) -> None:
        cache = InMemoryStageCache()
        pipeline, preparation, asr, cam, overlap = self.pipeline(5, cache=cache)

        pipeline.transcribe(
            self.request(5, "manual", language="auto"),
            self.context(),
        )
        second = pipeline.transcribe(
            self.request(5, "manual", language="en-US"),
            self.context(),
        )

        self.assertEqual(preparation.calls, 1)
        self.assertEqual(sum(len(batch) for batch in asr.calls), 10)
        self.assertEqual(sum(len(batch) for batch in cam.calls), 5)
        self.assertEqual(sum(len(batch) for batch in overlap.calls), 5)
        self.assertEqual(asr.requested_languages, ["auto", "en-US"])

        cache_by_stage = second.pipeline_metrics["cache"]["byStage"]
        self.assertEqual(cache_by_stage["asr"]["misses"], 5)
        for stage in (
            "normalize",
            "vad",
            "boundary",
            "campp-embedding",
            "overlap",
        ):
            self.assertGreater(cache_by_stage[stage]["hits"], 0)

    def test_auto_language_persists_detected_english_not_auto(self) -> None:
        asr = FakeAsrAdapter(
            language_by_window_id={
                "window-1": "English",
                "window-2": "English",
            }
        )
        pipeline, _, _, _, _ = self.pipeline(2, asr=asr)

        result = pipeline.transcribe(
            self.request(2, "manual", language="auto"),
            self.context(),
        )

        self.assertEqual(result.language, "en")
        self.assertEqual({segment.language for segment in result.segments}, {"en"})
        self.assertNotEqual(result.language, "auto")

    def test_explicit_language_persists_canonical_requested_tag(self) -> None:
        asr = FakeAsrAdapter(
            language_by_window_id={
                "window-1": "Chinese",
                "window-2": "Chinese",
            }
        )
        pipeline, _, _, _, _ = self.pipeline(2, asr=asr)

        result = pipeline.transcribe(
            self.request(2, "manual", language="en-US"),
            self.context(),
        )

        self.assertEqual(result.language, "en-US")
        self.assertEqual(
            {segment.language for segment in result.segments},
            {"en-US"},
        )

    def test_auto_language_persists_mul_for_mixed_content(self) -> None:
        asr = FakeAsrAdapter(
            language_by_window_id={
                "window-1": "English",
                "window-2": "Chinese",
            }
        )
        pipeline, _, _, _, _ = self.pipeline(2, asr=asr)

        result = pipeline.transcribe(
            self.request(2, "manual", language="auto"),
            self.context(),
        )

        self.assertEqual(result.language, "mul")
        self.assertEqual(
            [segment.language for segment in result.segments],
            ["en", "zh"],
        )
        self.assertEqual(
            [segment.speaker_id for segment in result.segments],
            ["speaker-1", "speaker-2"],
        )

    def test_one_speaker_can_switch_languages_across_windows(self) -> None:
        asr = FakeAsrAdapter(
            language_by_window_id={
                "window-1": "English",
                "window-2": "Chinese",
                "window-3": "Spanish",
            }
        )
        pipeline, _, _, _, _ = self.pipeline(
            1,
            windows=3,
            asr=asr,
        )

        result = pipeline.transcribe(
            self.request(1, "manual", language="auto"),
            self.context(),
        )

        self.assertEqual(result.language, "mul")
        self.assertEqual(
            [segment.language for segment in result.segments],
            ["en", "zh", "es"],
        )
        self.assertEqual(
            {segment.speaker_id for segment in result.segments},
            {"speaker-1"},
        )

    def test_secondary_verifier_is_selective_and_records_full_telemetry(self) -> None:
        secondary = FakeSecondaryVerifier()
        pipeline, _, _, cam, _ = self.pipeline(
            2,
            windows=8,
            cam=FakeCamPlusAdapter(2, identical=True),
            secondary=secondary,
            config=SpeakerPipelineConfig(
                low_margin_threshold=0.18,
                high_margin_threshold=0.35,
                max_secondary_fraction=0.25,
            ),
        )
        result = pipeline.transcribe(
            self.request(2, "manual"),
            self.context(),
        )
        primary_count = sum(len(batch) for batch in cam.calls)
        secondary_count = sum(len(batch) for batch in secondary.calls)
        self.assertEqual(primary_count, 8)
        self.assertGreater(secondary_count, 0)
        self.assertLess(secondary_count, primary_count)
        self.assertLessEqual(secondary_count, 2)
        events = result.pipeline_metrics["escalations"]
        self.assertEqual(len(events), secondary_count)
        for event in events:
            self.assertTrue(event["reasons"])
            self.assertEqual(len(event["cacheKey"]), 64)
            self.assertIn("latencyMs", event)
            self.assertIn("resource", event)
            self.assertIn("confidence", event)
            self.assertIn("exitReason", event)
        self.assertEqual(
            result.pipeline_metrics["policy"]["secondaryVoiceprint"],
            "ERes2NetV2",
        )
        self.assertFalse(
            result.pipeline_metrics["policy"]["fullCorpusSecondaryRunAllowed"]
        )

    def test_secondary_fraction_uses_strict_floor_budget(self) -> None:
        for window_count, expected_secondary in ((2, 0), (3, 0), (4, 1), (8, 2)):
            with self.subTest(
                windows=window_count,
                expected_secondary=expected_secondary,
            ):
                secondary = FakeSecondaryVerifier()
                cam = FakeCamPlusAdapter(2)
                pipeline, _, _, _, _ = self.pipeline(
                    2,
                    windows=window_count,
                    preparation=FakePreparationAdapter(
                        window_count,
                        boundary_conflict_ids={
                            f"window-{index}"
                            for index in range(1, window_count + 1)
                        },
                    ),
                    cam=cam,
                    secondary=secondary,
                    config=SpeakerPipelineConfig(
                        high_margin_threshold=3.0,
                        max_secondary_fraction=0.25,
                    ),
                )
                result = pipeline.transcribe(
                    self.request(
                        2,
                        "manual",
                        job_id=f"floor-budget-{window_count}",
                    ),
                    self.context(f"floor-budget-{window_count}"),
                )
                cam_ids = {
                    segment_id
                    for batch in cam.calls
                    for segment_id in batch
                }
                secondary_ids = {
                    segment_id
                    for batch in secondary.calls
                    for segment_id in batch
                }
                self.assertEqual(len(cam_ids), window_count)
                self.assertEqual(len(secondary_ids), expected_secondary)
                self.assertLessEqual(
                    result.pipeline_metrics["routing"]["escalationRate"],
                    0.25,
                )
                self.assertTrue(secondary_ids.issubset(cam_ids))
                cascade = {
                    event["stage"]: event
                    for event in result.pipeline_metrics["cascade"]
                }
                secondary_event = cascade["secondary-review"]
                self.assertEqual(
                    secondary_event["candidateRange"]["maxCandidates"],
                    expected_secondary,
                )
                self.assertEqual(
                    secondary_event["candidateRange"]["candidateCount"],
                    expected_secondary,
                )
                if expected_secondary == 0:
                    self.assertEqual(secondary.calls, [])
                    self.assertFalse(secondary_event["invoked"])
                    self.assertEqual(
                        secondary_event["exitReason"],
                        "SECONDARY_FRACTION_BUDGET_ZERO",
                    )
                    difficult = [
                        segment
                        for segment in result.segments
                        if isinstance(
                            segment.evidence.get("selectiveReview"),
                            Mapping,
                        )
                    ]
                    self.assertEqual(len(difficult), window_count)
                    self.assertTrue(
                        all(
                            segment.evidence["selectiveReview"][
                                "reviewStatus"
                            ]
                            == "REVIEW_REQUIRED"
                            for segment in difficult
                        )
                    )

    def test_campp_is_full_corpus_and_eres_is_bounded_subset(self) -> None:
        secondary = FakeSecondaryVerifier()
        cam = FakeCamPlusAdapter(2)
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=8,
            preparation=FakePreparationAdapter(
                8,
                boundary_conflict_ids={
                    f"window-{index}" for index in range(1, 9)
                },
            ),
            cam=cam,
            secondary=secondary,
            config=SpeakerPipelineConfig(
                high_margin_threshold=3.0,
                max_secondary_fraction=0.25,
            ),
        )
        result = pipeline.transcribe(
            self.request(2, "manual", job_id="cascade-subset"),
            self.context("cascade-subset"),
        )
        cam_ids = [segment_id for batch in cam.calls for segment_id in batch]
        secondary_ids = [
            segment_id for batch in secondary.calls for segment_id in batch
        ]
        self.assertEqual(
            cam_ids,
            [f"window-{index}" for index in range(1, 9)],
        )
        self.assertEqual(len(secondary_ids), 2)
        self.assertTrue(set(secondary_ids).issubset(cam_ids))
        cascade = {
            event["stage"]: event
            for event in result.pipeline_metrics["cascade"]
        }
        self.assertEqual(
            cascade["campp-embedding"]["triggerReason"],
            "ALL_SPEECH_WINDOWS",
        )
        self.assertEqual(
            cascade["secondary-review"]["triggerReason"],
            "CAMPP_DIFFICULT_SEGMENTS",
        )
        self.assertEqual(
            result.pipeline_metrics["policy"]["maxSecondaryFraction"],
            0.25,
        )
        self.assertEqual(
            result.pipeline_metrics["policy"]["resolvedSpeakerCount"],
            2,
        )
        self.assertEqual(
            result.pipeline_metrics["policy"]["speakerCountMode"],
            "manual",
        )

    def test_pyannote_fallback_receives_only_eres_unresolved(self) -> None:
        secondary = FakeSecondaryVerifier(
            exit_reasons={
                "window-2": "LOW_MARGIN_UNRESOLVED",
                "window-4": "NO_REFERENCE",
            }
        )
        pyannote = FakePyannoteAudit()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=8,
            preparation=FakePreparationAdapter(
                8,
                boundary_conflict_ids={
                    f"window-{index}" for index in range(1, 9)
                },
            ),
            cam=FakeCamPlusAdapter(
                2,
                vectors_by_window_id={
                    "window-1": (1.0, 0.0),
                    "window-2": (1.0, 0.0),
                    "window-3": (0.0, 1.0),
                    "window-4": (0.0, 1.0),
                    "window-5": (1.0, 0.0),
                    "window-6": (1.0, 0.0),
                    "window-7": (0.0, 1.0),
                    "window-8": (0.0, 1.0),
                },
            ),
            secondary=secondary,
            pyannote=pyannote,
            config=SpeakerPipelineConfig(
                high_margin_threshold=3.0,
                max_secondary_fraction=0.5,
                pyannote_mode="fallback",
            ),
        )
        result = pipeline.transcribe(
            self.request(2, "manual", job_id="fallback-order"),
            self.context("fallback-order"),
        )
        self.assertEqual(
            secondary.calls,
            [("window-1", "window-2", "window-3", "window-4")],
        )
        self.assertEqual(pyannote.calls, [("window-2", "window-4")])
        by_id = {segment.segment_id: segment for segment in result.segments}
        for segment_id in ("window-2", "window-4"):
            self.assertEqual(
                by_id[segment_id].evidence["pyannoteFallback"][
                    "reviewStatus"
                ],
                "REVIEW_REQUIRED",
            )
        cascade = result.pipeline_metrics["cascade"]
        self.assertEqual(
            [event["stage"] for event in cascade],
            ["campp-embedding", "secondary-review", "pyannote-fallback"],
        )
        fallback_event = cascade[-1]
        self.assertEqual(
            fallback_event["triggerReason"],
            "ERES_UNRESOLVED_ONLY",
        )
        self.assertEqual(
            fallback_event["candidateRange"]["scope"],
            "eres-unresolved-only",
        )
        self.assertEqual(
            fallback_event["candidateRange"]["segmentIds"],
            ["window-2", "window-4"],
        )
        self.assertEqual(
            fallback_event["candidateRange"]["sourceCount"],
            4,
        )
        self.assertEqual(
            fallback_event["candidateRange"]["maxCandidates"],
            4,
        )
        self.assertEqual(
            fallback_event["candidateRange"]["candidateCount"],
            2,
        )
        for event in cascade:
            self.assert_cascade_schema(event)

    def test_unknown_eres_exit_reason_is_fail_closed_before_pyannote(self) -> None:
        secondary = FakeSecondaryVerifier(
            speaker_ids={"window-1": "speaker-2"},
            exit_reasons={"window-1": "FUTURE_UNKNOWN_EXIT"},
        )
        pyannote = FakePyannoteAudit()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=4,
            preparation=FakePreparationAdapter(
                4,
                boundary_conflict_ids={
                    f"window-{index}" for index in range(1, 5)
                },
            ),
            cam=FakeCamPlusAdapter(
                2,
                vectors_by_window_id={
                    "window-1": (1.0, 0.0),
                    "window-2": (1.0, 0.0),
                    "window-3": (0.0, 1.0),
                    "window-4": (0.0, 1.0),
                },
            ),
            secondary=secondary,
            pyannote=pyannote,
            config=SpeakerPipelineConfig(
                high_margin_threshold=3.0,
                max_secondary_fraction=0.25,
                pyannote_mode="fallback",
            ),
        )
        result = pipeline.transcribe(
            self.request(2, "manual", job_id="unknown-eres-exit"),
            self.context("unknown-eres-exit"),
        )
        self.assertEqual(secondary.calls, [("window-1",)])
        self.assertEqual(pyannote.calls, [("window-1",)])
        by_id = {segment.segment_id: segment for segment in result.segments}
        reviewed = by_id["window-1"]
        self.assertEqual(reviewed.speaker_id, "speaker-1")
        self.assertEqual(
            reviewed.evidence["selectiveReview"]["proposalExitReason"],
            "FUTURE_UNKNOWN_EXIT",
        )
        self.assertEqual(
            reviewed.evidence["selectiveReview"]["reviewStatus"],
            "REVIEW_REQUIRED",
        )
        self.assertFalse(
            reviewed.evidence["selectiveReview"]["applied"]
        )
        self.assertEqual(
            reviewed.evidence["pyannoteFallback"]["reviewStatus"],
            "REVIEW_REQUIRED",
        )

    def test_pyannote_fallback_cannot_bypass_eres(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "secondary_adapter is required",
        ):
            self.pipeline(
                2,
                pyannote=FakePyannoteAudit(),
                config=SpeakerPipelineConfig(pyannote_mode="fallback"),
            )

    def test_campp_cascade_cache_audit_reports_full_hits(self) -> None:
        cache = InMemoryStageCache()
        cam = FakeCamPlusAdapter(2)
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=4,
            cache=cache,
            cam=cam,
        )
        request = self.request(2, "manual", job_id="campp-cache")
        pipeline.transcribe(request, self.context("campp-cache"))
        second = pipeline.transcribe(request, self.context("campp-cache"))
        campp_event = next(
            event
            for event in second.pipeline_metrics["cascade"]
            if event["stage"] == "campp-embedding"
        )
        self.assertEqual(
            campp_event["cache"],
            {
                "stage": "campp-embedding",
                "requests": 4,
                "hits": 4,
                "misses": 0,
                "recomputations": 0,
            },
        )
        self.assertEqual(sum(len(batch) for batch in cam.calls), 4)

    def test_all_count_modes_fail_closed_on_silent_cardinality_change(self) -> None:
        for mode in ("manual", "auto", "hybrid"):
            with self.subTest(mode=mode):
                secondary = FakeSecondaryVerifier(
                    speaker_ids={"window-1": "speaker-2"},
                    exit_reasons={
                        "window-1": "VERIFIED_SPEAKER_CHANGE"
                    },
                )
                pipeline, _, _, _, _ = self.pipeline(
                    2,
                    windows=2,
                    preparation=FakePreparationAdapter(
                        2,
                        boundary_conflict_ids={"window-1"},
                    ),
                    secondary=secondary,
                    config=SpeakerPipelineConfig(
                        high_margin_threshold=3.0,
                        max_secondary_fraction=0.75,
                    ),
                )
                result = pipeline.transcribe(
                    self.request(
                        2,
                        mode,
                        job_id=f"cardinality-{mode}",
                    ),
                    self.context(f"cardinality-{mode}"),
                )
                by_id = {
                    segment.segment_id: segment
                    for segment in result.segments
                }
                self.assertEqual(
                    {
                        segment.speaker_id
                        for segment in result.segments
                    },
                    {"speaker-1", "speaker-2"},
                )
                self.assertEqual(
                    by_id["window-1"].speaker_id,
                    "speaker-1",
                )
                self.assertEqual(
                    by_id["window-1"].evidence["selectiveReview"][
                        "reviewStatus"
                    ],
                    "REVIEW_REQUIRED",
                )
                self.assertEqual(
                    by_id["window-1"].evidence["selectiveReview"][
                        "exitReason"
                    ],
                    "CARDINALITY_CHANGE_REVIEW_REQUIRED",
                )
                self.assertFalse(
                    by_id["window-1"].evidence["selectiveReview"][
                        "applied"
                    ]
                )

    def test_protected_overlap_lock_and_high_margin_never_reach_secondary(self) -> None:
        preparation = FakePreparationAdapter(
            4,
            boundary_conflict_ids={"window-1", "window-2", "window-3"},
            locked={"window-1": "speaker-1"},
        )
        overlap = FakeOverlapAdapter({"window-2"})
        secondary = FakeSecondaryVerifier()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=4,
            preparation=preparation,
            overlap=overlap,
            secondary=secondary,
        )
        result = pipeline.transcribe(
            self.request(2, "manual"),
            self.context(),
        )
        called = {
            segment_id for batch in secondary.calls for segment_id in batch
        }
        self.assertNotIn("window-1", called)
        self.assertNotIn("window-2", called)
        self.assertNotIn("window-3", called)
        self.assertTrue(
            next(
                segment
                for segment in result.segments
                if segment.segment_id == "window-1"
            ).human_locked
        )
        self.assertTrue(
            next(
                segment
                for segment in result.segments
                if segment.segment_id == "window-2"
            ).overlapping
        )

    def test_secondary_cannot_mutate_text_overlap_or_turn_structure(self) -> None:
        secondary = FakeSecondaryVerifier(mutate_text=True)
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=8,
            cam=FakeCamPlusAdapter(2, identical=True),
            secondary=secondary,
        )
        with self.assertRaisesRegex(
            WorkerError,
            "cannot modify text, boundary, turn, or overlap",
        ):
            pipeline.transcribe(
                self.request(2, "manual"),
                self.context(),
            )

    def test_batch_adapters_fail_closed_on_string_missing_and_duplicate(self) -> None:
        for mode in ("string", "missing", "duplicate"):
            with self.subTest(mode=mode):
                pipeline, _, _, _, _ = self.pipeline(
                    2,
                    asr=FakeAsrAdapter(mode),
                )
                with self.assertRaises(WorkerError) as captured:
                    pipeline.transcribe(
                        self.request(2, "manual"),
                        self.context(),
                    )
                self.assertEqual(
                    captured.exception.code,
                    "PIPELINE_ADAPTER_RESULT_INVALID",
                )

    def test_pyannote_is_disabled_by_default_and_audit_never_mutates(self) -> None:
        pyannote = FakePyannoteAudit()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=8,
            cam=FakeCamPlusAdapter(2, identical=True),
            pyannote=pyannote,
        )
        result = pipeline.transcribe(
            self.request(2, "manual"),
            self.context(),
        )
        self.assertEqual(pyannote.calls, [])
        self.assertEqual(
            result.pipeline_metrics["policy"]["pyannoteMode"],
            "disabled",
        )
        self.assertFalse(
            result.pipeline_metrics["policy"]["pyannoteTelemetryEnabled"]
        )

    def test_pyannote_audit_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "pyannote_mode must be disabled or fallback",
        ):
            SpeakerPipelineConfig(pyannote_mode="audit")

    def test_fail_closed_review_evidence_and_queue_reasons(self) -> None:
        for window_count, expected_exit in (
            (2, "SECONDARY_BUDGET_EXHAUSTED"),
            (4, "SECONDARY_BUDGET_EXHAUSTED"),
        ):
            with self.subTest(window_count=window_count):
                pipeline, _, _, _, _ = self.pipeline(
                    2,
                    windows=window_count,
                    preparation=FakePreparationAdapter(
                        window_count,
                        boundary_conflict_ids={
                            f"window-{index}"
                            for index in range(1, window_count + 1)
                        },
                    ),
                    secondary=FakeSecondaryVerifier(),
                    config=SpeakerPipelineConfig(
                        high_margin_threshold=3.0,
                        max_secondary_fraction=0.25,
                    ),
                )
                result = pipeline.transcribe(
                    self.request(
                        2,
                        "manual",
                        job_id=f"fail-closed-{window_count}",
                    ),
                    self.context(f"fail-closed-{window_count}"),
                )
                by_id = {segment.segment_id: segment for segment in result.segments}
                self.assertTrue(
                    all(
                        isinstance(segment.evidence.get("selectiveReview"), Mapping)
                        for segment in result.segments
                    )
                )
                self.assertIn(
                    expected_exit,
                    {
                        by_id[segment_id].evidence["selectiveReview"]["exitReason"]
                        for segment_id in by_id
                    },
                )
                queue = build_review_queue(
                    job_id="fail-closed",
                    policy=SpeakerCountPolicy.from_payload(
                        {"speakerCountMode": "manual", "speakerCount": 2}
                    ),
                    estimate=result.speaker_count_estimate,
                    segments=result.segments,
                    count_confidence_threshold=0.75,
                    segment_confidence_threshold=0.0,
                    speaker_margin_threshold=-1.0,
                    range_width_threshold=8,
                )
                open_reasons = {
                    item["reasonCode"]
                    for item in queue["items"]
                    if item["status"] == "open"
                }
                self.assertIn("SECONDARY_BUDGET_EXHAUSTED", open_reasons)

    def test_protected_and_disabled_secondary_have_explicit_exit_reasons(self) -> None:
        preparation = FakePreparationAdapter(
            2,
            boundary_conflict_ids={"window-1", "window-2"},
            locked={"window-1": "speaker-1"},
        )
        pipeline, _, _, _, _ = self.pipeline(
            2,
            windows=2,
            preparation=preparation,
            secondary=None,
            config=SpeakerPipelineConfig(
                high_margin_threshold=3.0,
                max_secondary_fraction=0.75,
            ),
        )
        result = pipeline.transcribe(
            self.request(2, "manual", job_id="disabled-review"),
            self.context("disabled-review"),
        )
        exits = {
            segment.segment_id: segment.evidence["selectiveReview"]["exitReason"]
            for segment in result.segments
        }
        self.assertEqual(exits["window-1"], "PROTECTED_REQUIRES_HUMAN_REVIEW")
        self.assertEqual(exits["window-2"], "SECONDARY_ADAPTER_DISABLED")

    def test_cancellation_is_checked_before_source_hashing_and_models(self) -> None:
        pipeline, preparation, _, _, _ = self.pipeline(2)
        cancellation = threading.Event()
        cancellation.set()
        with self.assertRaises(JobCancelled):
            pipeline.transcribe(
                self.request(2, "manual"),
                AdapterContext("pipeline-job", self.output, cancellation),
            )
        self.assertEqual(preparation.calls, 0)

    def test_unsupported_asr_language_fails_before_hashing_and_preparation(
        self,
    ) -> None:
        preparation = FakePreparationAdapter(2)
        asr = RejectingLanguageAsrAdapter()
        pipeline, _, _, _, _ = self.pipeline(
            2,
            preparation=preparation,
            asr=asr,
        )

        with patch(
            "backend.speaker_pipeline._sha256_file",
            side_effect=AssertionError("source hashing must not run"),
        ):
            with self.assertRaises(WorkerError) as captured:
                pipeline.transcribe(
                    self.request(2, "manual", job_id="language-preflight"),
                    self.context("language-preflight"),
                )

        self.assertEqual(captured.exception.code, "ASR_LANGUAGE_UNSUPPORTED")
        self.assertEqual(preparation.calls, 0)
        self.assertEqual(asr.calls, [])

    def test_vad_failure_stops_before_asr_embedding_and_overlap(self) -> None:
        preparation = FailingPreparationAdapter()
        pipeline, _, asr, cam, overlap = self.pipeline(
            2,
            preparation=preparation,
        )

        with self.assertRaises(WorkerError) as captured:
            pipeline.transcribe(
                self.request(2, "manual", job_id="vad-failure"),
                self.context("vad-failure"),
            )

        self.assertEqual(
            captured.exception.code,
            "FUNASR_VAD_INFERENCE_FAILED",
        )
        self.assertEqual(preparation.calls, 1)
        self.assertEqual(asr.calls, [])
        self.assertEqual(cam.calls, [])
        self.assertEqual(overlap.calls, [])

    def test_dynamic_count_fail_closed_flag_forces_count_uncertainty_review(
        self,
    ) -> None:
        pipeline = SpeakerPipeline(
            preparation_adapter=FakePreparationAdapter(1),
            asr_adapter=FakeAsrAdapter(),
            embedding_adapter=FakeCamPlusAdapter(1),
            overlap_adapter=NoOverlapAdapter(),
        )
        segment = TranscriptSegment(
            segment_id="segment-1",
            start_ms=0,
            end_ms=1_000,
            speaker_id="speaker-1",
            raw_text="你好",
            normalized_text="你好",
            display_text="你好",
            confidence=0.99,
            speaker_scores=(SpeakerScore("speaker-1", 0.99),),
            speaker_margin=0.99,
        )
        window = SpeechWindow(
            window_id="segment-1",
            start_ms=0,
            end_ms=1_000,
        )
        clusters = _ClusterResult(
            count=1,
            confidence=0.99,
            candidate_min=1,
            candidate_max=1,
            assignments=(0,),
            scores=((0.99,),),
            low_confidence_fail_closed=True,
        )

        candidates = pipeline._select_candidates(
            (segment,),
            (window,),
            clusters,
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn("COUNT_UNCERTAINTY", candidates[0].reasons)


if __name__ == "__main__":
    unittest.main()
