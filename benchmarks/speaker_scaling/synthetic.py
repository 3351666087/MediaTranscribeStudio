"""Deterministic synthetic inputs for speaker-clustering scalability tests.

This module constructs only the backend's lightweight value objects. It never
opens media and never instantiates a model adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from backend.speaker_pipeline import EmbeddingRecord, SpeechWindow


SCENARIO_NAMES = (
    "orthogonal",
    "near-voices",
    "singleton-outlier",
    "shuffled-orthogonal",
)
NEAR_VOICE_COSINE = 0.94


@dataclass(frozen=True)
class SyntheticCase:
    """One deterministic clustering input and its persistent-speaker truth."""

    scenario: str
    windows: tuple[SpeechWindow, ...]
    embeddings: tuple[EmbeddingRecord, ...]
    truth_by_window_id: Mapping[str, int | None]
    persistent_speaker_count: int
    embedding_dimension: int
    reference_centers: tuple[tuple[float, ...], ...]
    outlier_window_ids: frozenset[str] = frozenset()
    near_voice_cosine: float | None = None

    @property
    def input_is_time_ordered(self) -> bool:
        starts = tuple(window.start_ms for window in self.windows)
        return starts == tuple(sorted(starts))


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 0.0:
        raise ValueError("synthetic vector must have non-zero magnitude")
    return tuple(value / magnitude for value in vector)


def _basis(dimension: int, index: int) -> tuple[float, ...]:
    vector = [0.0] * dimension
    vector[index] = 1.0
    return tuple(vector)


def _build_persistent_case(
    *,
    scenario: str,
    centers: Sequence[Sequence[float]],
    samples_per_speaker: int,
    near_voice_cosine: float | None = None,
) -> SyntheticCase:
    windows: list[SpeechWindow] = []
    embeddings: list[EmbeddingRecord] = []
    truth: dict[str, int | None] = {}
    sequence = 0
    dimension = len(centers[0])
    jitter_axis = dimension - 1

    for sample_index in range(samples_per_speaker):
        centered_index = sample_index - (samples_per_speaker - 1) / 2.0
        jitter = centered_index * 0.001 if scenario == "near-voices" else 0.0
        for speaker_index, center in enumerate(centers):
            vector = list(center)
            if jitter:
                vector[jitter_axis] += jitter
            window_id = (
                f"{scenario}-speaker-{speaker_index + 1:04d}"
                f"-sample-{sample_index + 1:04d}"
            )
            windows.append(
                SpeechWindow(
                    window_id=window_id,
                    start_ms=sequence * 1_000,
                    end_ms=sequence * 1_000 + 900,
                    metadata={
                        "synthetic": True,
                        "syntheticScenario": scenario,
                        "syntheticSpeaker": speaker_index,
                    },
                )
            )
            embeddings.append(
                EmbeddingRecord(
                    window_id=window_id,
                    vector=_unit(vector),
                    confidence=1.0,
                    evidence={
                        "fixture": scenario,
                        "loadsModel": False,
                        "loadsMedia": False,
                    },
                )
            )
            truth[window_id] = speaker_index
            sequence += 1

    return SyntheticCase(
        scenario=scenario,
        windows=tuple(windows),
        embeddings=tuple(embeddings),
        truth_by_window_id=truth,
        persistent_speaker_count=len(centers),
        embedding_dimension=dimension,
        reference_centers=tuple(_unit(center) for center in centers),
        near_voice_cosine=near_voice_cosine,
    )


def _orthogonal(
    speaker_count: int,
    samples_per_speaker: int,
    *,
    scenario: str = "orthogonal",
) -> SyntheticCase:
    centers = tuple(_basis(speaker_count, index) for index in range(speaker_count))
    return _build_persistent_case(
        scenario=scenario,
        centers=centers,
        samples_per_speaker=samples_per_speaker,
    )


def _near_voices(
    speaker_count: int,
    samples_per_speaker: int,
) -> SyntheticCase:
    dimension = speaker_count + 1
    centers: list[tuple[float, ...]] = [_basis(dimension, 0)]
    if speaker_count >= 2:
        second = [0.0] * dimension
        second[0] = NEAR_VOICE_COSINE
        second[1] = math.sqrt(1.0 - NEAR_VOICE_COSINE**2)
        centers.append(tuple(second))
    for speaker_index in range(2, speaker_count):
        centers.append(_basis(dimension, speaker_index))
    return _build_persistent_case(
        scenario="near-voices",
        centers=centers,
        samples_per_speaker=samples_per_speaker,
        near_voice_cosine=NEAR_VOICE_COSINE if speaker_count >= 2 else None,
    )


def _singleton_outlier(
    speaker_count: int,
    samples_per_speaker: int,
) -> SyntheticCase:
    dimension = speaker_count + 1
    centers = tuple(_basis(dimension, index) for index in range(speaker_count))
    base = _build_persistent_case(
        scenario="singleton-outlier",
        centers=centers,
        samples_per_speaker=samples_per_speaker,
    )
    window_id = "singleton-outlier-ignored-0001"
    sequence = len(base.windows)
    outlier_window = SpeechWindow(
        window_id=window_id,
        start_ms=sequence * 1_000,
        end_ms=sequence * 1_000 + 900,
        metadata={
            "synthetic": True,
            "syntheticScenario": "singleton-outlier",
            "syntheticOutlier": True,
        },
    )
    outlier_embedding = EmbeddingRecord(
        window_id=window_id,
        vector=_basis(dimension, speaker_count),
        confidence=1.0,
        evidence={
            "fixture": "singleton-outlier",
            "persistentSpeakerTruth": False,
            "loadsModel": False,
            "loadsMedia": False,
        },
    )
    truth = dict(base.truth_by_window_id)
    truth[window_id] = None
    return SyntheticCase(
        scenario="singleton-outlier",
        windows=base.windows + (outlier_window,),
        embeddings=base.embeddings + (outlier_embedding,),
        truth_by_window_id=truth,
        persistent_speaker_count=speaker_count,
        embedding_dimension=dimension,
        reference_centers=base.reference_centers,
        outlier_window_ids=frozenset((window_id,)),
    )


def _shuffled_orthogonal(
    speaker_count: int,
    samples_per_speaker: int,
) -> SyntheticCase:
    base = _orthogonal(
        speaker_count,
        samples_per_speaker,
        scenario="shuffled-orthogonal",
    )
    order = tuple(reversed(range(len(base.windows))))
    return SyntheticCase(
        scenario=base.scenario,
        windows=tuple(base.windows[index] for index in order),
        embeddings=tuple(base.embeddings[index] for index in order),
        truth_by_window_id=base.truth_by_window_id,
        persistent_speaker_count=base.persistent_speaker_count,
        embedding_dimension=base.embedding_dimension,
        reference_centers=base.reference_centers,
    )


def generate_scenario(
    scenario: str,
    *,
    speaker_count: int,
    samples_per_speaker: int,
) -> SyntheticCase:
    """Create a named synthetic case without model or media I/O."""

    if scenario not in SCENARIO_NAMES:
        raise ValueError(
            f"scenario must be one of {', '.join(SCENARIO_NAMES)}"
        )
    if isinstance(speaker_count, bool) or speaker_count < 1:
        raise ValueError("speaker_count must be a positive integer")
    if isinstance(samples_per_speaker, bool) or samples_per_speaker < 1:
        raise ValueError("samples_per_speaker must be a positive integer")

    if scenario == "orthogonal":
        return _orthogonal(speaker_count, samples_per_speaker)
    if scenario == "near-voices":
        return _near_voices(speaker_count, samples_per_speaker)
    if scenario == "singleton-outlier":
        return _singleton_outlier(speaker_count, samples_per_speaker)
    return _shuffled_orthogonal(speaker_count, samples_per_speaker)
