from __future__ import annotations

import copy
import hashlib
import json

import pytest

from backend.asr_evidence import (
    ASR_MODEL_MANIFEST_NAME,
    AsrEvidenceError,
    build_asr_candidate_set,
    model_identity_from_manifest,
    project_asr_candidate_set,
    validate_asr_candidate_set,
)
from backend.errors import WorkerError
from backend.models import SpeakerScore, TranscriptSegment
from backend.speaker_pipeline import AsrHypothesis


SOURCE_SHA256 = "a" * 64
MANIFEST_SHA256 = "b" * 64


def _tokens(text: str = "Hello world") -> list[dict[str, object]]:
    left, right = text.split(" ", 1)
    return [
        {"text": left, "startMs": 100, "endMs": 450},
        {"text": right, "startMs": 500, "endMs": 900},
    ]


def _candidate_set(
    *,
    model_identity_status: str = "manifest-bound",
    scores_available: bool = True,
    nbest: bool = False,
) -> dict:
    status = "available" if scores_available else "provider-unavailable"
    score = -0.1 if scores_available else None
    hypotheses = [
        {
            "text": "Hello world",
            "language": "en",
            "tokens": _tokens(),
            "acousticScore": score,
            "acousticScoreStatus": status,
            "decodeScore": score,
            "decodeScoreStatus": status,
        }
    ]
    if nbest:
        hypotheses.append(
            {
                "text": "Hallo world",
                "language": "en",
                "tokens": _tokens("Hallo world"),
                "acousticScore": -0.3,
                "acousticScoreStatus": "available",
                "decodeScore": -0.4,
                "decodeScoreStatus": "available",
            }
        )
    return build_asr_candidate_set(
        model_id="Qwen3-ASR-1.7B",
        model_revision="revision-fixture",
        model_manifest_sha256=MANIFEST_SHA256,
        model_identity_status=model_identity_status,
        source_audio_sha256=SOURCE_SHA256,
        normalization_profile="mono-16khz-f32-v1",
        source_window_id="window-1",
        start_ms=0,
        end_ms=1_000,
        hypotheses=hypotheses,
    )


def test_candidate_ids_and_set_hash_are_stable_and_rank_bound() -> None:
    first = _candidate_set(nbest=True)
    second = _candidate_set(nbest=True)

    assert first == second
    assert first["candidateSetType"] == "provider-nbest"
    assert [item["rank"] for item in first["nBest"]] == [1, 2]
    assert len({item["candidateId"] for item in first["nBest"]}) == 2
    assert all(item["lexicalRepairEligible"] for item in first["nBest"])
    assert validate_asr_candidate_set(first) == first


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["nBest"].reverse(),
        lambda value: value["nBest"][0].__setitem__("text", "Tampered"),
        lambda value: value["nBest"][0].__setitem__("decodeScore", -99.0),
        lambda value: value["nBest"][0]["tokens"][0].__setitem__("startMs", 101),
        lambda value: value.__setitem__("modelManifestSha256", "c" * 64),
        lambda value: value.__setitem__("sourceWindowSha256", "d" * 64),
        lambda value: value.__setitem__("candidateSetSha256", "e" * 64),
    ],
)
def test_candidate_evidence_rejects_reordering_and_tampering(mutate) -> None:
    value = copy.deepcopy(_candidate_set(nbest=True))
    mutate(value)

    with pytest.raises(AsrEvidenceError):
        validate_asr_candidate_set(value)


def test_unavailable_or_unverified_scores_cannot_authorize_lexical_repair() -> None:
    unavailable = _candidate_set(scores_available=False)
    unverified = _candidate_set(model_identity_status="unverified-local")

    assert unavailable["nBest"][0]["acousticScore"] is None
    assert unavailable["nBest"][0]["decodeScore"] is None
    assert unavailable["nBest"][0]["lexicalRepairEligible"] is False
    assert unverified["nBest"][0]["lexicalRepairEligible"] is False


def test_manifest_bound_scored_token_evidence_is_lexical_repair_eligible() -> None:
    value = _candidate_set()

    assert value["modelIdentityStatus"] == "manifest-bound"
    assert value["nBest"][0]["tokens"]
    assert value["nBest"][0]["acousticScoreStatus"] == "available"
    assert value["nBest"][0]["decodeScoreStatus"] == "available"
    assert value["nBest"][0]["lexicalRepairEligible"] is True


def test_projection_preserves_parent_identity_without_fabricating_scores() -> None:
    source = _candidate_set()
    projected = project_asr_candidate_set(
        source,
        source_text="Hello world",
        target_text="Hello",
        target_tokens=[{"text": "Hello", "startMs": 100, "endMs": 450}],
        target_window_id="window-1.cardinality-01",
        target_start_ms=0,
        target_end_ms=500,
    )

    candidate = projected["nBest"][0]
    assert projected["candidateSetType"] == "projection-derived-top1"
    assert projected["sourceStartMs"] == 0
    assert projected["sourceEndMs"] == 500
    assert candidate["parentCandidateId"] == source["nBest"][0]["candidateId"]
    assert candidate["candidateId"] != candidate["parentCandidateId"]
    assert candidate["acousticScoreStatus"] == "projection-derived"
    assert candidate["decodeScoreStatus"] == "projection-derived"
    assert candidate["lexicalRepairEligible"] is False
    assert validate_asr_candidate_set(
        projected,
        expected_text="Hello",
        expected_start_ms=0,
        expected_end_ms=500,
    ) == projected


def test_projection_clips_cross_boundary_token_without_losing_parent_hash() -> None:
    source = _candidate_set()
    projected = project_asr_candidate_set(
        source,
        source_text="Hello world",
        target_text="world",
        target_tokens=[{"text": "world", "startMs": 450, "endMs": 550}],
        target_window_id="window-1.cardinality-02",
        target_start_ms=500,
        target_end_ms=1_000,
    )

    assert projected["nBest"][0]["tokens"] == [
        {"index": 0, "text": "world", "startMs": 500, "endMs": 550}
    ]
    assert (
        projected["nBest"][0]["parentCandidateId"]
        == source["nBest"][0]["candidateId"]
    )


def test_expected_segment_text_and_boundaries_fail_closed() -> None:
    value = _candidate_set()

    with pytest.raises(AsrEvidenceError, match="rawText"):
        validate_asr_candidate_set(value, expected_text="Other text")
    with pytest.raises(AsrEvidenceError, match="sourceStartMs"):
        validate_asr_candidate_set(value, expected_start_ms=1)
    with pytest.raises(AsrEvidenceError, match="sourceEndMs"):
        validate_asr_candidate_set(value, expected_end_ms=999)


def test_model_identity_binds_exact_manifest_bytes(tmp_path) -> None:
    manifest = {
        "modelId": "Qwen3-ASR-1.7B",
        "revision": "0123456789abcdef",
    }
    raw = json.dumps(manifest, separators=(",", ":")).encode()
    (tmp_path / ASR_MODEL_MANIFEST_NAME).write_bytes(raw)

    identity = model_identity_from_manifest(tmp_path, injected_fixture=False)

    assert identity == {
        "modelRevision": "0123456789abcdef",
        "modelManifestSha256": hashlib.sha256(raw).hexdigest(),
        "modelIdentityStatus": "manifest-bound",
    }


def test_transcript_segment_round_trip_revalidates_candidate_evidence() -> None:
    segment = TranscriptSegment(
        segment_id="segment-1",
        start_ms=0,
        end_ms=1_000,
        speaker_id="speaker-1",
        raw_text="Hello world",
        normalized_text="Hello world",
        display_text="Hello world",
        confidence=0.9,
        speaker_scores=(SpeakerScore("speaker-1", 0.9),),
        speaker_margin=0.9,
        evidence={"asr": {"provider": "fixture", **_candidate_set()}},
        language="en",
    )

    assert TranscriptSegment.from_mapping(segment.as_dict(), 0) == segment

    tampered = segment.as_dict()
    tampered["evidence"]["asr"]["nBest"][0]["text"] = "Tampered"
    with pytest.raises(WorkerError) as captured:
        TranscriptSegment.from_mapping(tampered, 0)
    assert captured.value.code == "ASR_CANDIDATE_EVIDENCE_INVALID"


def test_asr_cache_deserialization_revalidates_candidate_hashes() -> None:
    value = {
        "windowId": "window-1",
        "text": "Hello world",
        "confidence": 0.9,
        "evidence": _candidate_set(),
    }

    assert AsrHypothesis.from_mapping(value).text == "Hello world"

    value["evidence"]["candidateSetSha256"] = "f" * 64
    with pytest.raises(ValueError, match="immutable or traceable"):
        AsrHypothesis.from_mapping(value)


def test_candidate_set_type_cannot_misrepresent_cardinality_or_projection() -> None:
    with pytest.raises(AsrEvidenceError, match="provider-nbest"):
        build_asr_candidate_set(
            model_id="Qwen3-ASR-1.7B",
            model_revision="revision-fixture",
            model_manifest_sha256=MANIFEST_SHA256,
            model_identity_status="manifest-bound",
            source_audio_sha256=SOURCE_SHA256,
            normalization_profile="mono-16khz-f32-v1",
            source_window_id="window-1",
            start_ms=0,
            end_ms=1_000,
            hypotheses=[
                {
                    "text": "Hello world",
                    "language": "en",
                    "tokens": _tokens(),
                    "acousticScore": -0.1,
                    "acousticScoreStatus": "available",
                    "decodeScore": -0.1,
                    "decodeScoreStatus": "available",
                }
            ],
            candidate_set_type="provider-nbest",
        )

    with pytest.raises(AsrEvidenceError, match="parent candidate"):
        build_asr_candidate_set(
            model_id="Qwen3-ASR-1.7B",
            model_revision="revision-fixture",
            model_manifest_sha256=MANIFEST_SHA256,
            model_identity_status="manifest-bound",
            source_audio_sha256=SOURCE_SHA256,
            normalization_profile="mono-16khz-f32-v1",
            source_window_id="window-1",
            start_ms=0,
            end_ms=1_000,
            hypotheses=[
                {
                    "text": "Hello world",
                    "language": "en",
                    "tokens": _tokens(),
                    "acousticScoreStatus": "projection-derived",
                    "decodeScoreStatus": "projection-derived",
                }
            ],
            candidate_set_type="projection-derived-top1",
        )
