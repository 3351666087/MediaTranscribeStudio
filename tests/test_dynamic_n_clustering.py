from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from backend import speaker_pipeline
from backend.errors import WorkerError
from backend.models import SpeakerCountPolicy, StartJobRequest
from backend.speaker_pipeline import (
    EmbeddingRecord,
    SpeakerPipelineConfig,
    SpeechWindow,
)
from benchmarks.speaker_scaling.synthetic import generate_scenario


SPEAKER_COUNTS = (1, 2, 3, 5, 8, 13, 32, 129)


@dataclass(frozen=True)
class SyntheticClusteringCase:
    windows: tuple[SpeechWindow, ...]
    embeddings: tuple[EmbeddingRecord, ...]
    truth_by_window_id: Mapping[str, int]


def _request(
    mode: str,
    *,
    manual_count: int | None = None,
    bounds: tuple[int, int] | None = None,
    prior: int | None = None,
) -> StartJobRequest:
    payload: dict[str, object] = {"speakerCountMode": mode}
    if manual_count is not None:
        payload["speakerCount"] = manual_count
    if bounds is not None:
        payload["speakerCountBounds"] = {
            "min": bounds[0],
            "max": bounds[1],
        }
    if prior is not None:
        payload["speakerCountPrior"] = prior
    return StartJobRequest(
        job_id=f"dynamic-n-{mode}",
        source_path=Path("synthetic-dynamic-n.wav"),
        output_directory=Path("synthetic-output"),
        speaker_policy=SpeakerCountPolicy.from_payload(payload),
    )


def _config(**overrides: object) -> SpeakerPipelineConfig:
    values: dict[str, object] = {
        "cluster_similarity_threshold": 0.72,
        "max_auto_speakers": 256,
        "max_clustering_windows": 10_000,
        "max_clustering_work_items": 10_000_000,
        "kmeans_iterations": 2,
        "max_count_uncertainty_candidates": 9,
        "auto_count_confidence_threshold": 0.70,
    }
    values.update(overrides)
    return SpeakerPipelineConfig(**values)


def _basis_case(group_sizes: Sequence[int]) -> SyntheticClusteringCase:
    if not group_sizes or any(size < 1 for size in group_sizes):
        raise ValueError("group_sizes must contain positive values")
    dimension = len(group_sizes)
    windows: list[SpeechWindow] = []
    embeddings: list[EmbeddingRecord] = []
    truth: dict[str, int] = {}
    sequence = 0
    for sample_index in range(max(group_sizes)):
        for speaker_index, group_size in enumerate(group_sizes):
            if sample_index >= group_size:
                continue
            window_id = (
                f"speaker-{speaker_index + 1:03d}-sample-{sample_index + 1:03d}"
            )
            vector = [0.0] * dimension
            vector[speaker_index] = 1.0
            windows.append(
                SpeechWindow(
                    window_id=window_id,
                    start_ms=sequence * 1_000,
                    end_ms=(sequence + 1) * 1_000,
                    metadata={"syntheticSpeaker": speaker_index},
                )
            )
            embeddings.append(
                EmbeddingRecord(
                    window_id=window_id,
                    vector=tuple(vector),
                    confidence=1.0,
                    evidence={"fixture": "orthogonal-basis"},
                )
            )
            truth[window_id] = speaker_index
            sequence += 1
    return SyntheticClusteringCase(
        windows=tuple(windows),
        embeddings=tuple(embeddings),
        truth_by_window_id=truth,
    )


def _close_voice_case(
    *,
    cosine_similarity: float,
    samples_per_speaker: int,
) -> SyntheticClusteringCase:
    if not 0.0 < cosine_similarity < 1.0:
        raise ValueError("cosine_similarity must be between zero and one")
    second_axis = math.sqrt(1.0 - cosine_similarity**2)
    jitters = tuple(
        (index - (samples_per_speaker - 1) / 2.0) * 0.004
        for index in range(samples_per_speaker)
    )
    windows: list[SpeechWindow] = []
    embeddings: list[EmbeddingRecord] = []
    truth: dict[str, int] = {}
    sequence = 0
    for sample_index, jitter in enumerate(jitters):
        for speaker_index, vector in enumerate(
            (
                (1.0, 0.0, jitter),
                (cosine_similarity, second_axis, jitter),
            )
        ):
            window_id = (
                f"close-{speaker_index + 1}-sample-{sample_index + 1:03d}"
            )
            windows.append(
                SpeechWindow(
                    window_id=window_id,
                    start_ms=sequence * 800,
                    end_ms=sequence * 800 + 700,
                    metadata={"syntheticSpeaker": speaker_index},
                )
            )
            embeddings.append(
                EmbeddingRecord(
                    window_id=window_id,
                    vector=vector,
                    confidence=1.0,
                    evidence={
                        "fixture": "close-voices",
                        "crossSpeakerCosine": cosine_similarity,
                    },
                )
            )
            truth[window_id] = speaker_index
            sequence += 1
    return SyntheticClusteringCase(
        windows=tuple(windows),
        embeddings=tuple(embeddings),
        truth_by_window_id=truth,
    )


def _permuted(case: SyntheticClusteringCase) -> SyntheticClusteringCase:
    size = len(case.windows)
    order = tuple(range(0, size, 2)) + tuple(reversed(range(1, size, 2)))
    assert sorted(order) == list(range(size))
    return SyntheticClusteringCase(
        windows=tuple(case.windows[index] for index in order),
        embeddings=tuple(case.embeddings[index] for index in order),
        truth_by_window_id=case.truth_by_window_id,
    )


def _cluster(
    case: SyntheticClusteringCase,
    request: StartJobRequest,
    config: SpeakerPipelineConfig | None = None,
):
    result = speaker_pipeline._cluster(
        case.embeddings,
        case.windows,
        request,
        config or _config(),
    )
    assert 1 <= result.candidate_min <= result.count <= result.candidate_max
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.assignments) == len(case.windows)
    assert len(result.scores) == len(case.windows)
    assert all(len(row) == result.count for row in result.scores)
    assert set(result.assignments) == set(range(result.count))
    for assignment, row in zip(result.assignments, result.scores):
        assert row[assignment] >= max(row) - 1e-12
    return result


def _assignments_by_truth(
    case: SyntheticClusteringCase,
    assignments: Sequence[int],
) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for window, assignment in zip(case.windows, assignments):
        grouped.setdefault(
            case.truth_by_window_id[window.window_id],
            set(),
        ).add(assignment)
    return grouped


def _assert_exact_truth_partition(
    case: SyntheticClusteringCase,
    assignments: Sequence[int],
) -> None:
    grouped = _assignments_by_truth(case, assignments)
    assert all(len(cluster_ids) == 1 for cluster_ids in grouped.values())
    assert len({next(iter(cluster_ids)) for cluster_ids in grouped.values()}) == len(
        grouped
    )


def _result_signature(
    case: SyntheticClusteringCase,
    result: object,
) -> tuple[object, ...]:
    assignments = {
        window.window_id: assignment
        for window, assignment in zip(case.windows, result.assignments)
    }
    scores = {
        window.window_id: row
        for window, row in zip(case.windows, result.scores)
    }
    return (
        result.count,
        result.confidence,
        result.candidate_min,
        result.candidate_max,
        assignments,
        scores,
    )


@pytest.mark.parametrize("speaker_count", SPEAKER_COUNTS)
def test_manual_mode_honors_exact_k(speaker_count: int) -> None:
    case = _basis_case((2,) * speaker_count)
    result = _cluster(
        case,
        _request("manual", manual_count=speaker_count),
    )

    assert result.count == speaker_count
    assert result.confidence == 1.0
    assert result.candidate_min == speaker_count
    assert result.candidate_max == speaker_count
    _assert_exact_truth_partition(case, result.assignments)


@pytest.mark.parametrize("mode", ("auto", "hybrid"))
@pytest.mark.parametrize("speaker_count", SPEAKER_COUNTS)
def test_dynamic_n_recovers_well_separated_speakers(
    mode: str,
    speaker_count: int,
) -> None:
    case = _basis_case((2,) * speaker_count)
    if mode == "auto":
        request = _request("auto")
    else:
        lower = max(1, speaker_count - 2)
        upper = speaker_count + 2
        request = _request(
            "hybrid",
            bounds=(lower, upper),
            prior=speaker_count + 1,
        )

    result = _cluster(case, request)

    assert result.count == speaker_count
    assert result.candidate_min <= speaker_count <= result.candidate_max
    _assert_exact_truth_partition(case, result.assignments)


@pytest.mark.parametrize("mode", ("manual", "auto", "hybrid"))
def test_clustering_is_exactly_deterministic_under_input_permutation(
    mode: str,
) -> None:
    case = _basis_case((3,) * 13)
    shuffled = _permuted(case)
    if mode == "manual":
        request = _request("manual", manual_count=13)
    elif mode == "auto":
        request = _request("auto")
    else:
        request = _request("hybrid", bounds=(10, 16), prior=12)

    original = _cluster(case, request)
    reordered = _cluster(shuffled, request)

    assert _result_signature(case, original) == _result_signature(
        shuffled,
        reordered,
    )


def test_low_confidence_count_exposes_all_plausible_candidates() -> None:
    case = _close_voice_case(
        cosine_similarity=0.97,
        samples_per_speaker=8,
    )
    confidence_threshold = 0.80

    result = _cluster(
        case,
        _request("auto"),
        _config(
            cluster_similarity_threshold=0.99,
            auto_count_confidence_threshold=confidence_threshold,
        ),
    )

    assert result.confidence < confidence_threshold
    assert result.candidate_min < result.candidate_max
    assert result.candidate_min <= 1 <= result.candidate_max
    assert result.candidate_min <= 2 <= result.candidate_max


def test_hybrid_bounds_are_hard_even_when_acoustics_are_out_of_range() -> None:
    case = _basis_case((4,) * 5)

    result = _cluster(
        case,
        _request("hybrid", bounds=(2, 4), prior=3),
    )

    assert 2 <= result.count <= 4
    assert 2 <= result.candidate_min <= result.count
    assert result.count <= result.candidate_max <= 4


def test_hybrid_prior_breaks_an_acoustically_ambiguous_tie_only() -> None:
    case = _close_voice_case(
        cosine_similarity=0.97,
        samples_per_speaker=8,
    )
    config = _config(
        cluster_similarity_threshold=0.99,
        auto_count_confidence_threshold=0.80,
    )

    prefer_one = _cluster(
        case,
        _request("hybrid", bounds=(1, 3), prior=1),
        config,
    )
    prefer_two = _cluster(
        case,
        _request("hybrid", bounds=(1, 3), prior=2),
        config,
    )

    assert prefer_one.count == 1
    assert prefer_two.count == 2
    for result in (prefer_one, prefer_two):
        assert result.confidence < config.auto_count_confidence_threshold
        assert result.candidate_min <= 1 <= result.candidate_max
        assert result.candidate_min <= 2 <= result.candidate_max


def test_hybrid_prior_cannot_override_strong_acoustic_evidence() -> None:
    case = _basis_case((5, 5, 5))

    result = _cluster(
        case,
        _request("hybrid", bounds=(2, 4), prior=2),
    )

    assert result.count == 3
    _assert_exact_truth_partition(case, result.assignments)


def test_singleton_outlier_does_not_silently_create_a_speaker() -> None:
    case = _basis_case((6, 6, 6, 1))
    confidence_threshold = 0.80

    result = _cluster(
        case,
        _request("auto"),
        _config(auto_count_confidence_threshold=confidence_threshold),
    )

    assert result.count == 3
    assert result.confidence < confidence_threshold
    assert result.candidate_min <= 3 <= result.candidate_max
    assert result.candidate_min <= 4 <= result.candidate_max
    cluster_sizes = sorted(result.assignments.count(index) for index in range(3))
    assert cluster_sizes == [6, 6, 7]


def test_small_but_repeated_cluster_is_preserved() -> None:
    case = _basis_case((10, 10, 3))

    result = _cluster(case, _request("auto"))

    assert result.count == 3
    assert sorted(
        result.assignments.count(index) for index in range(result.count)
    ) == [3, 10, 10]
    _assert_exact_truth_partition(case, result.assignments)


def test_repeated_nearby_voices_remain_distinct() -> None:
    case = _close_voice_case(
        cosine_similarity=0.94,
        samples_per_speaker=12,
    )

    result = _cluster(case, _request("auto"))

    assert result.count == 2
    assert result.candidate_min <= 1 <= result.candidate_max
    assert result.candidate_min <= 2 <= result.candidate_max
    _assert_exact_truth_partition(case, result.assignments)


@pytest.mark.parametrize("mode", ("auto", "hybrid"))
@pytest.mark.parametrize("speaker_count", (3, 5, 8, 13))
def test_embedded_close_voice_pair_uses_residual_collapse_correction(
    mode: str,
    speaker_count: int,
) -> None:
    case = generate_scenario(
        "near-voices",
        speaker_count=speaker_count,
        samples_per_speaker=3,
    )
    request = (
        _request("auto")
        if mode == "auto"
        else _request(
            "hybrid",
            bounds=(max(1, speaker_count - 2), speaker_count + 2),
            prior=speaker_count,
        )
    )

    result = _cluster(case, request)

    assert result.count == speaker_count
    assert result.selection_method == "dynamic-n-adaptive-resample-stability-v7"
    assert result.under_split_detected
    assert "CLOSE_VOICE_RESIDUAL_COLLAPSE" in result.correction_path
    assert result.candidate_min <= speaker_count - 1
    assert result.candidate_max >= speaker_count
    _assert_exact_truth_partition(case, result.assignments)


def test_close_voice_residual_collapse_uses_strict_stability_guard() -> None:
    near = _cluster(
        generate_scenario(
            "near-voices",
            speaker_count=8,
            samples_per_speaker=3,
        ),
        _request("auto"),
    )

    merged_score = next(
        candidate
        for candidate in near.count_candidates
        if candidate.count == 7
    )
    separated_score = next(
        candidate
        for candidate in near.count_candidates
        if candidate.count == 8
    )

    assert near.count == 8
    assert "CLOSE_VOICE_RESIDUAL_COLLAPSE" in near.correction_path
    assert separated_score.stability >= 0.90
    assert separated_score.stability >= merged_score.stability - 0.08
    assert separated_score.objective >= merged_score.objective - 0.23
    assert separated_score.bootstrap_support == pytest.approx(1.0)
    assert separated_score.stability_components["coverage"] == pytest.approx(
        1.0
    )
    assert separated_score.stability_effective_unique_runs >= 2


def test_stability_masks_have_exact_size_preserve_locks_and_are_diverse() -> None:
    case = _basis_case((3,) * 8)
    locks = {0: 0, 7: 1, 23: 7}
    target = max(8, math.ceil(len(case.windows) * 0.80), len(locks))

    masks = [
        speaker_pipeline._stability_retained_indices(
            windows=case.windows,
            count=8,
            locks=locks,
            run_index=run_index,
        )
        for run_index in range(10)
    ]

    assert all(len(mask) == target for mask in masks)
    assert all(set(locks) <= set(mask) for mask in masks)
    assert len(set(masks)) > 5


def test_stability_masks_are_deterministic_and_permutation_invariant() -> None:
    case = _basis_case((3,) * 8)
    locked_ids = {
        case.windows[0].window_id: 0,
        case.windows[7].window_id: 1,
    }
    original_locks = {
        index: locked_ids[window.window_id]
        for index, window in enumerate(case.windows)
        if window.window_id in locked_ids
    }
    permuted_windows = tuple(reversed(case.windows))
    permuted_locks = {
        index: locked_ids[window.window_id]
        for index, window in enumerate(permuted_windows)
        if window.window_id in locked_ids
    }

    for run_index in range(10):
        original = speaker_pipeline._stability_retained_indices(
            windows=case.windows,
            count=8,
            locks=original_locks,
            run_index=run_index,
        )
        repeated = speaker_pipeline._stability_retained_indices(
            windows=case.windows,
            count=8,
            locks=original_locks,
            run_index=run_index,
        )
        permuted = speaker_pipeline._stability_retained_indices(
            windows=permuted_windows,
            count=8,
            locks=permuted_locks,
            run_index=run_index,
        )

        assert original == repeated
        assert {
            case.windows[index].window_id for index in original
        } == {
            permuted_windows[index].window_id for index in permuted
        }


def test_partition_stability_is_invariant_to_label_permutation() -> None:
    reference_centroids = ((1.0, 0.0), (0.0, 1.0))
    replicate_centroids = tuple(reversed(reference_centroids))
    label_map = speaker_pipeline._maximum_weight_label_map(
        reference_centroids,
        replicate_centroids,
    )

    agreement = speaker_pipeline._partition_agreement(
        (0, 0, 1, 1),
        (1, 1, 0, 0),
        label_map=label_map,
    )

    assert label_map == {0: 1, 1: 0}
    assert agreement == pytest.approx(
        {
            "adjustedRand": 1.0,
            "pairwiseJaccard": 1.0,
            "coassociationAgreement": 1.0,
            "alignedAccuracy": 1.0,
        }
    )


@pytest.mark.parametrize("mode", ("auto", "hybrid"))
@pytest.mark.parametrize("speaker_count", (1, 3, 8, 13))
def test_absolute_singleton_is_a_reviewable_count_not_a_persistent_speaker(
    mode: str,
    speaker_count: int,
) -> None:
    case = generate_scenario(
        "singleton-outlier",
        speaker_count=speaker_count,
        samples_per_speaker=3,
    )
    request = (
        _request("auto")
        if mode == "auto"
        else _request(
            "hybrid",
            bounds=(max(1, speaker_count - 2), speaker_count + 2),
            prior=speaker_count,
        )
    )

    result = _cluster(case, request)

    assert result.count == speaker_count
    assert result.over_split_detected
    assert "ABSOLUTE_SINGLETON_OUTLIER_AMBIGUITY" in result.correction_path
    assert (
        "PERSISTENT_COUNT_SELECTED_SINGLETON_REVIEW_REQUIRED"
        in result.confidence_reasons
    )
    assert result.candidate_min <= speaker_count
    assert result.candidate_max >= speaker_count + 1
    assert result.low_confidence_fail_closed
    singleton_candidate = next(
        candidate
        for candidate in result.count_candidates
        if candidate.count == speaker_count + 1
    )
    assert singleton_candidate.singleton_count == 1
    assert singleton_candidate.tiny_cluster_count == 1
    assert singleton_candidate.minimum_cluster_size == 1


@pytest.mark.parametrize("mode", ("manual", "auto", "hybrid"))
def test_work_item_budget_fails_closed_for_every_count_mode(mode: str) -> None:
    case = _basis_case((3, 3, 3))
    if mode == "manual":
        request = _request("manual", manual_count=3)
    elif mode == "auto":
        request = _request("auto")
    else:
        request = _request("hybrid", bounds=(2, 4), prior=3)
    work_item_limit = 8

    with pytest.raises(WorkerError) as captured:
        speaker_pipeline._cluster(
            case.embeddings,
            case.windows,
            request,
            _config(max_clustering_work_items=work_item_limit),
        )

    error = captured.value
    assert error.code == "CLUSTERING_RESOURCE_LIMIT_EXCEEDED"
    assert error.details["maxWorkItems"] == work_item_limit
    assert error.details["workItems"] > work_item_limit


@pytest.mark.parametrize("mode", ("manual", "auto", "hybrid"))
def test_same_fixture_succeeds_when_work_item_budget_is_sufficient(
    mode: str,
) -> None:
    case = _basis_case((3, 3, 3))
    if mode == "manual":
        request = _request("manual", manual_count=3)
    elif mode == "auto":
        request = _request("auto")
    else:
        request = _request("hybrid", bounds=(2, 4), prior=3)

    result = _cluster(
        case,
        request,
        _config(max_clustering_work_items=10_000),
    )

    assert result.count == 3
    _assert_exact_truth_partition(case, result.assignments)
