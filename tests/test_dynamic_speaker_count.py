from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
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
    _ClusterResult,
)


@dataclass(frozen=True)
class _Case:
    windows: tuple[SpeechWindow, ...]
    embeddings: tuple[EmbeddingRecord, ...]
    truth_by_window_id: Mapping[str, int]


def _request(
    mode: str,
    *,
    count: int | None = None,
    bounds: tuple[int, int] | None = None,
    prior: int | None = None,
) -> StartJobRequest:
    payload: dict[str, object] = {"speakerCountMode": mode}
    if count is not None:
        payload["speakerCount"] = count
    if bounds is not None:
        payload["speakerCountBounds"] = {
            "min": bounds[0],
            "max": bounds[1],
        }
    if prior is not None:
        payload["speakerCountPrior"] = prior
    return StartJobRequest(
        job_id=f"dynamic-speaker-count-{mode}",
        source_path=Path("dynamic-speaker-count.wav"),
        output_directory=Path("dynamic-speaker-count-output"),
        speaker_policy=SpeakerCountPolicy.from_payload(payload),
    )


def _config(**overrides: object) -> SpeakerPipelineConfig:
    values: dict[str, object] = {
        "cluster_similarity_threshold": 0.72,
        "max_auto_speakers": None,
        "max_clustering_windows": 10_000,
        "max_clustering_work_items": 10_000_000,
        "kmeans_iterations": 2,
        "count_stability_runs": 3,
        "max_count_uncertainty_candidates": 9,
        "auto_count_confidence_threshold": 0.70,
    }
    values.update(overrides)
    return SpeakerPipelineConfig(**values)


def _sequence_case(
    sequence: Sequence[int],
    *,
    dimension: int | None = None,
) -> _Case:
    if not sequence or min(sequence) < 0:
        raise ValueError("sequence must contain non-negative speaker indexes")
    vector_dimension = dimension or max(sequence) + 1
    windows: list[SpeechWindow] = []
    embeddings: list[EmbeddingRecord] = []
    truth: dict[str, int] = {}
    occurrences: dict[int, int] = {}
    for index, speaker_index in enumerate(sequence):
        occurrences[speaker_index] = occurrences.get(speaker_index, 0) + 1
        window_id = (
            f"speaker-{speaker_index + 1:03d}-"
            f"sample-{occurrences[speaker_index]:03d}"
        )
        vector = [0.0] * vector_dimension
        vector[speaker_index] = 1.0
        windows.append(
            SpeechWindow(
                window_id=window_id,
                start_ms=index * 800,
                end_ms=index * 800 + 700,
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
    return _Case(tuple(windows), tuple(embeddings), truth)


def _basis_case(group_sizes: Sequence[int]) -> _Case:
    sequence = [
        speaker_index
        for sample_index in range(max(group_sizes))
        for speaker_index, size in enumerate(group_sizes)
        if sample_index < size
    ]
    return _sequence_case(sequence, dimension=len(group_sizes))


def _close_voice_case(cosine_similarity: float, samples: int) -> _Case:
    second_axis = math.sqrt(1.0 - cosine_similarity**2)
    sequence = range(samples)
    windows: list[SpeechWindow] = []
    embeddings: list[EmbeddingRecord] = []
    truth: dict[str, int] = {}
    for sample_index in sequence:
        jitter = (sample_index - (samples - 1) / 2.0) * 0.004
        for speaker_index, vector in enumerate(
            (
                (1.0, 0.0, jitter),
                (cosine_similarity, second_axis, jitter),
            )
        ):
            index = len(windows)
            window_id = f"close-{speaker_index + 1}-{sample_index + 1:03d}"
            windows.append(
                SpeechWindow(
                    window_id=window_id,
                    start_ms=index * 800,
                    end_ms=index * 800 + 700,
                )
            )
            embeddings.append(
                EmbeddingRecord(
                    window_id=window_id,
                    vector=vector,
                    confidence=1.0,
                    evidence={"fixture": "near-voice"},
                )
            )
            truth[window_id] = speaker_index
    return _Case(tuple(windows), tuple(embeddings), truth)


def _cluster(
    case: _Case,
    request: StartJobRequest,
    config: SpeakerPipelineConfig | None = None,
    *,
    pyannote_count_prior: int | None = None,
) -> _ClusterResult:
    result = speaker_pipeline._cluster(
        case.embeddings,
        case.windows,
        request,
        config or _config(),
        pyannote_count_prior=pyannote_count_prior,
    )
    assert 1 <= result.candidate_min <= result.count <= result.candidate_max
    assert len(result.assignments) == len(case.windows)
    assert len(result.scores) == len(case.windows)
    return result


def _assert_exact_partition(case: _Case, result: _ClusterResult) -> None:
    cluster_by_truth: dict[int, set[int]] = {}
    for window, assignment in zip(case.windows, result.assignments):
        cluster_by_truth.setdefault(
            case.truth_by_window_id[window.window_id],
            set(),
        ).add(assignment)
    assert all(len(values) == 1 for values in cluster_by_truth.values())
    assert len({next(iter(values)) for values in cluster_by_truth.values()}) == len(
        cluster_by_truth
    )


@pytest.mark.parametrize("speaker_count", (1, 2, 5, 13, 64, 128, 129))
def test_auto_dynamic_n_has_no_fixed_cardinality_ceiling(
    speaker_count: int,
) -> None:
    case = _basis_case((2,) * speaker_count)

    result = _cluster(case, _request("auto"))

    assert result.count == speaker_count
    assert result.candidate_min <= speaker_count <= result.candidate_max
    _assert_exact_partition(case, result)


def test_pyannote_prior_cannot_override_large_singleton_objective_gap() -> None:
    case = _basis_case((1,) * 12)

    result = _cluster(
        case,
        _request("auto"),
        pyannote_count_prior=3,
    )

    assert result.count == 12
    assert result.candidate_min <= 3
    assert result.candidate_max >= 12
    assert result.low_confidence_fail_closed
    assert "PYANNOTE_COUNT_PRIOR_CONFLICT" in result.confidence_reasons


def test_pyannote_prior_breaks_close_voice_acoustic_tie_with_review() -> None:
    case = _close_voice_case(0.99, 4)

    acoustic_only = _cluster(case, _request("auto"))
    reconciled = _cluster(
        case,
        _request("auto"),
        pyannote_count_prior=2,
    )

    assert acoustic_only.count == 1
    assert reconciled.count == 2
    assert reconciled.candidate_min <= 1
    assert reconciled.candidate_max >= 2
    assert reconciled.low_confidence_fail_closed
    assert "PYANNOTE_FULL_TIMELINE_PRIOR:1->2" in reconciled.correction_path
    assert (
        "PYANNOTE_COUNT_PRIOR_APPLIED_WITH_REVIEW"
        in reconciled.confidence_reasons
    )


def test_pyannote_prior_cannot_override_persistent_acoustic_contradiction() -> None:
    case = _basis_case((2,) * 6)

    result = _cluster(
        case,
        _request("auto"),
        pyannote_count_prior=2,
    )

    assert result.count == 6
    assert result.candidate_min <= 2
    assert result.candidate_max >= 6
    assert result.low_confidence_fail_closed
    assert not any(
        item.startswith("PYANNOTE_FULL_TIMELINE_PRIOR:")
        for item in result.correction_path
    )
    assert "PYANNOTE_COUNT_PRIOR_CONFLICT" in result.confidence_reasons


def test_manual_129_is_exact_and_never_fail_closed() -> None:
    case = _basis_case((1,) * 129)

    result = _cluster(case, _request("manual", count=129))

    assert result.count == 129
    assert result.confidence == 1.0
    assert (result.candidate_min, result.candidate_max) == (129, 129)
    assert result.low_confidence_fail_closed is False
    _assert_exact_partition(case, result)


def test_hybrid_bounds_are_hard_and_prior_only_breaks_near_voice_tie() -> None:
    strong = _basis_case((4,) * 5)
    bounded = _cluster(
        strong,
        _request("hybrid", bounds=(2, 4), prior=3),
    )
    assert 2 <= bounded.count <= 4
    assert 2 <= bounded.candidate_min <= bounded.candidate_max <= 4

    ambiguous = _close_voice_case(0.97, 8)
    config = _config(
        cluster_similarity_threshold=0.99,
        auto_count_confidence_threshold=0.80,
    )
    prefer_one = _cluster(
        ambiguous,
        _request("hybrid", bounds=(1, 3), prior=1),
        config,
    )
    prefer_two = _cluster(
        ambiguous,
        _request("hybrid", bounds=(1, 3), prior=2),
        config,
    )
    assert (prefer_one.count, prefer_two.count) == (1, 2)
    assert prefer_one.low_confidence_fail_closed
    assert prefer_two.low_confidence_fail_closed


def test_near_voices_preserve_resolvable_pair_and_fail_closed_when_ambiguous() -> None:
    resolvable = _cluster(_close_voice_case(0.94, 12), _request("auto"))
    assert resolvable.count == 2
    assert resolvable.candidate_min <= 1 <= resolvable.candidate_max
    assert resolvable.candidate_min <= 2 <= resolvable.candidate_max
    _assert_exact_partition(_close_voice_case(0.94, 12), resolvable)

    ambiguous = _cluster(
        _close_voice_case(0.97, 8),
        _request("auto"),
        _config(
            cluster_similarity_threshold=0.99,
            auto_count_confidence_threshold=0.80,
        ),
    )
    assert ambiguous.candidate_min <= 1 <= ambiguous.candidate_max
    assert ambiguous.candidate_min <= 2 <= ambiguous.candidate_max
    assert ambiguous.low_confidence_fail_closed
    assert "FAIL_CLOSED_REVIEW_REQUIRED" in ambiguous.confidence_reasons


def test_imbalance_outlier_aba_and_fragmentation_corrections() -> None:
    imbalanced = _cluster(_basis_case((10, 10, 3)), _request("auto"))
    assert imbalanced.count == 3
    assert sorted(imbalanced.assignments.count(index) for index in range(3)) == [
        3,
        10,
        10,
    ]

    outlier = _cluster(
        _basis_case((6, 6, 6, 1)),
        _request("auto"),
        _config(auto_count_confidence_threshold=0.80),
    )
    assert outlier.count == 3
    assert outlier.over_split_detected
    assert outlier.low_confidence_fail_closed
    assert outlier.candidate_min <= 3 <= outlier.candidate_max
    assert outlier.candidate_min <= 4 <= outlier.candidate_max

    aba = _cluster(
        _sequence_case((0, 0, 0, 1, 1, 1, 1, 0, 0, 0)),
        _request("auto"),
    )
    assert aba.count == 2
    _assert_exact_partition(
        _sequence_case((0, 0, 0, 1, 1, 1, 1, 0, 0, 0)),
        aba,
    )

    fragmented_case = _sequence_case(tuple(index % 2 for index in range(40)))
    fragmented = _cluster(fragmented_case, _request("auto"))
    assert fragmented.count == 2
    _assert_exact_partition(fragmented_case, fragmented)


def test_human_locks_survive_dynamic_count_selection() -> None:
    case = _basis_case((4, 4))
    windows = list(case.windows)
    windows[0] = replace(windows[0], locked_speaker_id="speaker-1")
    windows[1] = replace(windows[1], locked_speaker_id="speaker-2")
    locked_case = replace(case, windows=tuple(windows))

    result = _cluster(locked_case, _request("auto"))

    assert result.count == 2
    assert result.assignments[0] == 0
    assert result.assignments[1] == 1


def test_score_decomposition_is_finite_auditable_and_cache_safe() -> None:
    case = _basis_case((3,) * 5)
    result = _cluster(case, _request("auto"))

    assert result.count_candidates
    required = {
        "compactness",
        "silhouette",
        "calinskiHarabasz",
        "daviesBouldin",
        "eigengap",
        "stability",
        "separation",
        "consensusSupport",
    }
    for candidate in result.count_candidates:
        assert required <= set(candidate.weighted_contributions)
        numeric_values = (
            candidate.objective,
            candidate.calinski_harabasz,
            candidate.davies_bouldin,
            candidate.stability,
            candidate.bootstrap_support,
            candidate.consensus_support,
            *candidate.weighted_contributions.values(),
        )
        assert all(math.isfinite(float(value)) for value in numeric_values)
        assert 0 <= candidate.metric_votes <= 7

    restored = _ClusterResult.from_mapping(
        result.as_dict(),
        expected_rows=len(case.windows),
    )
    assert restored.as_dict() == result.as_dict()
    serialized = result.as_dict()
    assert serialized["confidenceKind"] == (
        "bounded-decision-score-uncalibrated-v1"
    )
    assert serialized["confidenceIntervalKind"] == (
        "plausible-count-range-not-calibrated-credible-interval"
    )
    assert serialized["legalRange"] == {
        "min": result.legal_min,
        "max": result.legal_max,
    }
    assert serialized["evaluatedCounts"] == list(result.evaluated_counts)
    assert serialized["plannedCounts"] == list(result.planned_counts)


@pytest.fixture(scope="module")
def valid_v7_cluster_cache_payload() -> tuple[int, dict[str, object]]:
    case = _basis_case((3,) * 5)
    result = _cluster(case, _request("auto"))
    return len(case.windows), result.as_dict()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-stability-component",
        "unequal-resample-objective-lengths",
        "frequency-sum-mismatch",
        "support-mismatch",
        "truncation-flag-reason-conflict",
        "unknown-truncation-reason",
        "candidate-evaluated-planned-conflict",
        "non-finite-number",
        "missing-v7-field",
        "old-v6-method",
    ),
)
def test_v7_cluster_cache_rejects_malformed_audit_payloads(
    valid_v7_cluster_cache_payload: tuple[int, dict[str, object]],
    mutation: str,
) -> None:
    expected_rows, source = valid_v7_cluster_cache_payload
    payload = copy.deepcopy(source)
    candidates = payload["countCandidates"]
    assert isinstance(candidates, list)
    first_candidate = candidates[0]
    assert isinstance(first_candidate, dict)

    if mutation == "missing-stability-component":
        components = first_candidate["stabilityComponents"]
        assert isinstance(components, dict)
        components.pop("coverage")
    elif mutation == "unequal-resample-objective-lengths":
        objectives = first_candidate["resampleObjectives"]
        assert isinstance(objectives, list)
        objectives.pop()
    elif mutation == "frequency-sum-mismatch":
        frequency = payload["resampleObjectiveWinnerFrequency"]
        assert isinstance(frequency, dict)
        for count in tuple(frequency):
            frequency[count] = float(frequency[count]) * 0.5
    elif mutation == "support-mismatch":
        current = float(payload["resampleObjectiveWinnerSupport"])
        payload["resampleObjectiveWinnerSupport"] = (
            0.0 if current > 0.0 else 1.0
        )
    elif mutation == "truncation-flag-reason-conflict":
        payload["countSearchTruncated"] = True
        payload["searchTruncationReason"] = "resource-limit"
    elif mutation == "unknown-truncation-reason":
        payload["countSearchTruncated"] = True
        payload["adaptiveBudgetLimited"] = True
        payload["searchTruncationReason"] = "unknown-reason"
    elif mutation == "candidate-evaluated-planned-conflict":
        selected = int(payload["count"])
        planned = payload["plannedCounts"]
        assert isinstance(planned, list)
        payload["plannedCounts"] = [
            count for count in planned if int(count) != selected
        ]
    elif mutation == "non-finite-number":
        first_candidate["objective"] = float("nan")
    elif mutation == "missing-v7-field":
        payload.pop("searchExhaustive")
    elif mutation == "old-v6-method":
        payload["selectionMethod"] = (
            "dynamic-n-adaptive-resample-stability-v6"
        )
    else:  # pragma: no cover - protects the parameter table itself.
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises((TypeError, ValueError, WorkerError)):
        _ClusterResult.from_mapping(payload, expected_rows=expected_rows)


def test_resample_objective_winner_frequency_is_auditable() -> None:
    stability_runs = 4
    result = _cluster(
        _basis_case((3,) * 5),
        _request("auto"),
        _config(count_stability_runs=stability_runs),
    )

    assert all(
        len(candidate.resample_objectives) == stability_runs
        for candidate in result.count_candidates
    )
    assert sum(
        result.resample_objective_winner_frequency.values()
    ) == pytest.approx(1.0)
    assert set(result.resample_objective_winner_frequency) <= set(
        result.evaluated_counts
    )
    assert result.resample_objective_winner_support == pytest.approx(
        result.resample_objective_winner_frequency.get(result.count, 0.0)
    )
    serialized_frequency = result.as_dict()[
        "resampleObjectiveWinnerFrequency"
    ]
    assert serialized_frequency == {
        str(count): support
        for count, support in sorted(
            result.resample_objective_winner_frequency.items()
        )
    }
    assert result.as_dict()["resampleObjectiveWinnerSupport"] == pytest.approx(
        result.resample_objective_winner_support
    )
    # This is deliberately the winner of each pre-correction resample
    # objective.  A deterministic residual-collapse correction may select a
    # final K that won no pre-correction replicate.
    if result.count not in result.resample_objective_winner_frequency:
        assert result.resample_objective_winner_support == 0.0


@pytest.mark.parametrize(
    ("maximum_candidates", "expected"),
    (
        (1, (12,)),
        (2, (12, 13)),
        (3, (12, 13, 14)),
    ),
)
def test_low_budget_search_preserves_acoustic_anchors(
    maximum_candidates: int,
    expected: tuple[int, ...],
) -> None:
    order = speaker_pipeline._candidate_count_order(
        lower=4,
        upper=40,
        leader_estimate=12,
        spectral_estimate=13,
        prior=14,
        maximum_candidates=maximum_candidates,
    )

    assert order == expected


def test_adaptive_search_finds_hidden_optimum_with_wrong_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _basis_case((3,) * 13)
    monkeypatch.setattr(
        speaker_pipeline,
        "_leader_count",
        lambda vectors, threshold, maximum, work_item_limit: (1, 0),
    )
    monkeypatch.setattr(
        speaker_pipeline,
        "_graph_eigengap_profile",
        lambda vectors, landmark_limit: speaker_pipeline._EigengapProfile(
            scores={2: 1.0},
            estimate=2,
            method="forced-wrong-anchor-test",
            landmark_count=0,
            work_items=0,
        ),
    )

    result = _cluster(
        case,
        _request("auto"),
        _config(max_count_uncertainty_candidates=9),
    )

    assert result.leader_estimate == 1
    assert result.spectral_estimate == 2
    assert result.count == 13
    assert {12, 13, 14} <= set(result.evaluated_counts)
    assert result.planned_counts[:5] == (
        1,
        2,
        3,
        20,
        11,
    )
    assert tuple(
        count
        for count in result.planned_counts
        if 12 <= count <= 14
    ) == (12, 13, 14)
    assert result.decision_locally_bracketed
    assert not result.count_search_truncated


def test_input_permutation_is_deterministic() -> None:
    case = _basis_case((3,) * 13)
    reversed_case = _Case(
        windows=tuple(reversed(case.windows)),
        embeddings=tuple(reversed(case.embeddings)),
        truth_by_window_id=case.truth_by_window_id,
    )
    original = _cluster(case, _request("auto"))
    reordered = _cluster(reversed_case, _request("auto"))

    original_by_id = {
        window.window_id: assignment
        for window, assignment in zip(case.windows, original.assignments)
    }
    reordered_by_id = {
        window.window_id: assignment
        for window, assignment in zip(
            reversed_case.windows,
            reordered.assignments,
        )
    }
    assert original.count == reordered.count == 13
    assert original_by_id == reordered_by_id


def test_resource_bounded_search_is_auditable_and_fail_closed() -> None:
    result = _cluster(
        _basis_case((3, 3, 3)),
        _request("auto"),
        _config(max_clustering_work_items=300),
    )

    assert result.count_search_truncated
    assert result.low_confidence_fail_closed
    assert result.confidence <= 0.50
    assert result.candidate_min == 1
    assert result.candidate_max == 9
    assert "RESOURCE_BOUNDED_SEARCH_TRUNCATED" in result.confidence_reasons
    assert (
        result.search_truncation_reason
        == "adaptive-budget-unresolved-local-bracket"
    )
    assert not result.search_exhaustive
    assert not result.decision_locally_bracketed
    assert result.adaptive_budget_limited
    assert not result.resource_truncated
    assert result.resource_skipped_counts == ()
    assert result.minimum_required_work_items == 193
    assert result.resource_affordable_max_count == 1
    serialized = result.as_dict()
    assert serialized["legalRange"] == {"min": 1, "max": 9}
    assert serialized["evaluatedCounts"] == [1]
    assert serialized["evaluatedRange"] == {"min": 1, "max": 1}
    assert serialized["searchTruncationReason"] == (
        "adaptive-budget-unresolved-local-bracket"
    )
    assert serialized["searchExhaustive"] is False
    assert serialized["decisionLocallyBracketed"] is False
    assert serialized["adaptiveBudgetLimited"] is True
    assert serialized["resourceTruncated"] is False
    assert serialized["resourceSkippedCounts"] == []
    assert serialized["minimumRequiredWorkItems"] == 193
    assert serialized["resourceAffordableMaxCount"] == 1
    assert serialized["countSearchTruncated"] is True
    assert serialized["lowConfidenceFailClosed"] is True
