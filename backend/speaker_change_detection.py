"""Deterministic, fail-closed planning for speaker changes inside one VAD span.

This module is intentionally pure computation:

* it never opens media, invokes FFmpeg, or loads a speaker model;
* callers pass time-sorted CAM++ (or equivalent) sliding-window embeddings;
* energy valleys and optional ASR boundaries may localize an acoustic change,
  but can never create one;
* a shared 16 kHz mono PCM buffer is represented only by an immutable timeline
  descriptor, so upstream preparation can decode once and reuse the buffer.

The planner does not perform overlap detection.  An upstream overlap-risk flag
is treated as a blocker and is reported as unverified risk, never as detected
overlap or verified non-overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence


REVIEW_REQUIRED = "REVIEW_REQUIRED"
NOT_REQUIRED = "NOT_REQUIRED"
OVERLAP_NOT_EVALUATED = "NOT_EVALUATED"
OVERLAP_RISK_NOT_EVALUATED = "RISK_FLAGGED_NOT_EVALUATED"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _probability(value: Any, field_name: str) -> float:
    result = _finite_number(value, field_name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _bounded_positive_int(value: Any, field_name: str) -> int:
    if not _is_int(value) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _unique_ordered(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class Pcm16kMonoTimeline:
    """Reference metadata for an upstream-owned, reusable normalized PCM buffer.

    ``buffer_id`` identifies the already decoded buffer.  Samples are not
    copied into this planner and the planner has no API for decoding them.
    """

    buffer_id: str
    sample_count: int
    sample_rate_hz: int = 16_000
    channel_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.buffer_id, str) or not self.buffer_id.strip():
            raise ValueError("buffer_id must be non-empty text")
        object.__setattr__(self, "buffer_id", self.buffer_id.strip())
        _bounded_positive_int(self.sample_count, "sample_count")
        if self.sample_rate_hz != 16_000:
            raise ValueError("shared PCM must be 16 kHz")
        if self.channel_count != 1:
            raise ValueError("shared PCM must be mono")

    @property
    def duration_ms(self) -> int:
        return math.ceil(self.sample_count * 1000 / self.sample_rate_hz)


@dataclass(frozen=True)
class SpeakerEmbeddingWindow:
    """One time-sorted sliding-window speaker embedding."""

    window_id: str
    start_ms: int
    end_ms: int
    embedding: tuple[float, ...]
    confidence: float = 1.0
    overlap_risk: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise ValueError("window_id must be non-empty text")
        object.__setattr__(self, "window_id", self.window_id.strip())
        if (
            not _is_int(self.start_ms)
            or not _is_int(self.end_ms)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValueError("embedding window must have a valid time range")
        if not isinstance(self.embedding, tuple) or not self.embedding:
            raise ValueError("embedding must be a non-empty tuple")
        normalized = tuple(
            _finite_number(value, f"{self.window_id}.embedding")
            for value in self.embedding
        )
        if math.sqrt(sum(value * value for value in normalized)) <= 1e-12:
            raise ValueError("embedding must have non-zero norm")
        object.__setattr__(self, "embedding", normalized)
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, f"{self.window_id}.confidence"),
        )
        if not isinstance(self.overlap_risk, bool):
            raise ValueError("overlap_risk must be boolean")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")

    @property
    def center_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2


@dataclass(frozen=True)
class EnergyValley:
    timestamp_ms: int
    confidence: float = 1.0
    marker_id: str | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ValueError("energy valley timestamp_ms must be non-negative")
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, "energy valley confidence"),
        )
        if self.marker_id is not None and (
            not isinstance(self.marker_id, str) or not self.marker_id.strip()
        ):
            raise ValueError("energy valley marker_id must be non-empty text")
        if self.marker_id is not None:
            object.__setattr__(self, "marker_id", self.marker_id.strip())


@dataclass(frozen=True)
class AsrBoundary:
    timestamp_ms: int
    confidence: float = 1.0
    marker_id: str | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.timestamp_ms) or self.timestamp_ms < 0:
            raise ValueError("ASR boundary timestamp_ms must be non-negative")
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, "ASR boundary confidence"),
        )
        if self.marker_id is not None and (
            not isinstance(self.marker_id, str) or not self.marker_id.strip()
        ):
            raise ValueError("ASR boundary marker_id must be non-empty text")
        if self.marker_id is not None:
            object.__setattr__(self, "marker_id", self.marker_id.strip())


@dataclass(frozen=True)
class SpeakerChangeDetectionConfig:
    """Thresholds for conservative automatic split planning."""

    min_review_change_score: float = 0.14
    min_auto_change_score: float = 0.42
    min_embedding_confidence: float = 0.78
    min_auto_acoustic_confidence: float = 0.62
    min_resulting_interval_ms: int = 700
    peak_merge_radius_ms: int = 160
    boundary_search_radius_ms: int = 600
    min_boundary_search_radius_ms: int = 100
    min_boundary_marker_confidence: float = 0.20
    max_auto_center_gap_ms: int = 1_200
    max_auto_boundary_uncertainty_ms: int = 450

    def __post_init__(self) -> None:
        review = _probability(
            self.min_review_change_score,
            "min_review_change_score",
        )
        automatic = _probability(
            self.min_auto_change_score,
            "min_auto_change_score",
        )
        if automatic <= review:
            raise ValueError(
                "min_auto_change_score must exceed min_review_change_score"
            )
        embedding_confidence = _probability(
            self.min_embedding_confidence, "min_embedding_confidence"
        )
        acoustic_confidence = _probability(
            self.min_auto_acoustic_confidence,
            "min_auto_acoustic_confidence",
        )
        marker_confidence = _probability(
            self.min_boundary_marker_confidence,
            "min_boundary_marker_confidence",
        )
        object.__setattr__(self, "min_review_change_score", review)
        object.__setattr__(self, "min_auto_change_score", automatic)
        object.__setattr__(
            self,
            "min_embedding_confidence",
            embedding_confidence,
        )
        object.__setattr__(
            self,
            "min_auto_acoustic_confidence",
            acoustic_confidence,
        )
        object.__setattr__(
            self,
            "min_boundary_marker_confidence",
            marker_confidence,
        )
        for field_name in (
            "min_resulting_interval_ms",
            "peak_merge_radius_ms",
            "boundary_search_radius_ms",
            "min_boundary_search_radius_ms",
            "max_auto_center_gap_ms",
            "max_auto_boundary_uncertainty_ms",
        ):
            _bounded_positive_int(getattr(self, field_name), field_name)
        if self.min_boundary_search_radius_ms > self.boundary_search_radius_ms:
            raise ValueError(
                "min_boundary_search_radius_ms must not exceed "
                "boundary_search_radius_ms"
            )


@dataclass(frozen=True)
class SplitProposal:
    """One auditable VAD-internal split proposal."""

    proposal_id: str
    split_ms: int
    acoustic_boundary_ms: int
    boundary_uncertainty_ms: int
    left_window_id: str
    right_window_id: str
    support_window_ids: tuple[str, ...]
    change_score: float
    acoustic_confidence: float
    boundary_source: str
    boundary_marker_confidence: float | None
    boundary_marker_id: str | None
    review_status: str
    review_reasons: tuple[str, ...]
    evidence_codes: tuple[str, ...]
    apply_automatically: bool
    overlap_risk: bool
    overlap_detection_status: str

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "proposalId": self.proposal_id,
            "splitMs": self.split_ms,
            "acousticBoundaryMs": self.acoustic_boundary_ms,
            "boundaryUncertaintyMs": self.boundary_uncertainty_ms,
            "leftWindowId": self.left_window_id,
            "rightWindowId": self.right_window_id,
            "supportWindowIds": list(self.support_window_ids),
            "changeScore": self.change_score,
            "acousticConfidence": self.acoustic_confidence,
            "boundarySource": self.boundary_source,
            "reviewStatus": self.review_status,
            "reviewReasons": list(self.review_reasons),
            "evidenceCodes": list(self.evidence_codes),
            "applyAutomatically": self.apply_automatically,
            "overlapRisk": self.overlap_risk,
            "overlapDetectionStatus": self.overlap_detection_status,
        }
        if self.boundary_marker_confidence is not None:
            value["boundaryMarkerConfidence"] = self.boundary_marker_confidence
        if self.boundary_marker_id is not None:
            value["boundaryMarkerId"] = self.boundary_marker_id
        return value


@dataclass(frozen=True)
class SpeakerChangePlan:
    """Deterministic result for one VAD interval."""

    vad_start_ms: int
    vad_end_ms: int
    proposals: tuple[SplitProposal, ...]
    evaluated_edge_count: int
    candidate_edge_count: int
    suppressed_peak_count: int
    pcm_buffer_id: str | None
    method: str = "cam-plus-adjacent-change-v1"
    overlap_detection_status: str = OVERLAP_NOT_EVALUATED

    @property
    def review_required(self) -> bool:
        return any(
            proposal.review_status == REVIEW_REQUIRED
            for proposal in self.proposals
        )

    @property
    def automatic_splits_ms(self) -> tuple[int, ...]:
        return tuple(
            proposal.split_ms
            for proposal in self.proposals
            if proposal.apply_automatically
        )

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "vadStartMs": self.vad_start_ms,
            "vadEndMs": self.vad_end_ms,
            "method": self.method,
            "proposals": [proposal.as_dict() for proposal in self.proposals],
            "evaluatedEdgeCount": self.evaluated_edge_count,
            "candidateEdgeCount": self.candidate_edge_count,
            "suppressedPeakCount": self.suppressed_peak_count,
            "reviewRequired": self.review_required,
            "automaticSplitsMs": list(self.automatic_splits_ms),
            "overlapDetectionStatus": self.overlap_detection_status,
            "overlapDetectorRun": False,
        }
        if self.pcm_buffer_id is not None:
            value["pcmBufferId"] = self.pcm_buffer_id
        return value


@dataclass(frozen=True)
class _EdgeCandidate:
    acoustic_boundary_ms: int
    boundary_uncertainty_ms: int
    center_gap_ms: int
    left_window_id: str
    right_window_id: str
    support_window_ids: tuple[str, ...]
    change_score: float
    acoustic_confidence: float
    embedding_confidence: float
    overlap_risk: bool
    merged_peak_count: int = 1


@dataclass(frozen=True)
class _BoundaryChoice:
    timestamp_ms: int
    source: str
    marker_confidence: float | None
    marker_id: str | None


def _cosine_change_score(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    cosine = dot / (left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return round(max(0.0, min(1.0, 1.0 - cosine)), 12)


def _build_edge_candidates(
    windows: Sequence[SpeakerEmbeddingWindow],
    config: SpeakerChangeDetectionConfig,
) -> tuple[_EdgeCandidate, ...]:
    output: list[_EdgeCandidate] = []
    for left, right in zip(windows, windows[1:]):
        change_score = _cosine_change_score(left.embedding, right.embedding)
        if change_score < config.min_review_change_score:
            continue
        center_gap = right.center_ms - left.center_ms
        acoustic_boundary = (left.center_ms + right.center_ms) // 2
        embedding_confidence = min(left.confidence, right.confidence)
        acoustic_confidence = round(
            0.75 * change_score + 0.25 * embedding_confidence,
            12,
        )
        output.append(
            _EdgeCandidate(
                acoustic_boundary_ms=acoustic_boundary,
                boundary_uncertainty_ms=max(1, math.ceil(center_gap / 2)),
                center_gap_ms=center_gap,
                left_window_id=left.window_id,
                right_window_id=right.window_id,
                support_window_ids=(left.window_id, right.window_id),
                change_score=change_score,
                acoustic_confidence=acoustic_confidence,
                embedding_confidence=embedding_confidence,
                overlap_risk=left.overlap_risk or right.overlap_risk,
            )
        )
    return tuple(output)


def _merge_nearby_peaks(
    candidates: Sequence[_EdgeCandidate],
    config: SpeakerChangeDetectionConfig,
) -> tuple[tuple[_EdgeCandidate, ...], int]:
    if not candidates:
        return (), 0
    groups: list[list[_EdgeCandidate]] = []
    for candidate in candidates:
        if (
            groups
            and candidate.acoustic_boundary_ms
            - groups[-1][-1].acoustic_boundary_ms
            <= config.peak_merge_radius_ms
        ):
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    merged: list[_EdgeCandidate] = []
    for group in groups:
        winner = min(
            group,
            key=lambda item: (
                -item.change_score,
                -item.acoustic_confidence,
                item.boundary_uncertainty_ms,
                item.acoustic_boundary_ms,
                item.left_window_id,
                item.right_window_id,
            ),
        )
        support_ids = _unique_ordered(
            tuple(
                window_id
                for item in group
                for window_id in item.support_window_ids
            )
        )
        merged.append(
            replace(
                winner,
                support_window_ids=support_ids,
                embedding_confidence=min(
                    item.embedding_confidence for item in group
                ),
                overlap_risk=any(item.overlap_risk for item in group),
                merged_peak_count=len(group),
            )
        )
    return tuple(merged), len(candidates) - len(merged)


def _choose_boundary(
    candidate: _EdgeCandidate,
    *,
    vad_start_ms: int,
    vad_end_ms: int,
    energy_valleys: Sequence[EnergyValley],
    asr_boundaries: Sequence[AsrBoundary],
    config: SpeakerChangeDetectionConfig,
) -> _BoundaryChoice:
    search_radius = min(
        config.boundary_search_radius_ms,
        max(
            config.min_boundary_search_radius_ms,
            candidate.boundary_uncertainty_ms,
        ),
    )
    ranked: list[tuple[tuple[float, int, int, int, str], _BoundaryChoice]] = []

    def consider(
        timestamp_ms: int,
        confidence: float,
        marker_id: str | None,
        source: str,
        source_priority: int,
    ) -> None:
        if not (vad_start_ms < timestamp_ms < vad_end_ms):
            return
        if confidence < config.min_boundary_marker_confidence:
            return
        distance = abs(timestamp_ms - candidate.acoustic_boundary_ms)
        if distance > search_radius:
            return
        proximity = 1.0 - distance / search_radius
        localization_score = round(0.75 * proximity + 0.25 * confidence, 12)
        stable_id = marker_id or ""
        rank = (
            -localization_score,
            distance,
            -source_priority,
            timestamp_ms,
            stable_id,
        )
        ranked.append(
            (
                rank,
                _BoundaryChoice(
                    timestamp_ms=timestamp_ms,
                    source=source,
                    marker_confidence=confidence,
                    marker_id=marker_id,
                ),
            )
        )

    for marker in energy_valleys:
        consider(
            marker.timestamp_ms,
            marker.confidence,
            marker.marker_id,
            "ENERGY_VALLEY",
            2,
        )
    for marker in asr_boundaries:
        consider(
            marker.timestamp_ms,
            marker.confidence,
            marker.marker_id,
            "ASR_BOUNDARY",
            1,
        )
    if not ranked:
        return _BoundaryChoice(
            timestamp_ms=candidate.acoustic_boundary_ms,
            source="ACOUSTIC_MIDPOINT",
            marker_confidence=None,
            marker_id=None,
        )
    return min(ranked, key=lambda item: item[0])[1]


def _initial_proposal(
    candidate: _EdgeCandidate,
    boundary: _BoundaryChoice,
    config: SpeakerChangeDetectionConfig,
) -> SplitProposal:
    review_reasons: list[str] = []
    evidence_codes = ["CAM_PLUS_ADJACENT_CHANGE"]
    if candidate.merged_peak_count > 1:
        evidence_codes.append("NEARBY_CHANGE_PEAKS_MERGED")
    if boundary.source != "ACOUSTIC_MIDPOINT":
        evidence_codes.append(f"{boundary.source}_LOCALIZATION")
    if candidate.change_score < config.min_auto_change_score:
        review_reasons.append("LOW_CHANGE_SCORE")
    if candidate.embedding_confidence < config.min_embedding_confidence:
        review_reasons.append("LOW_EMBEDDING_CONFIDENCE")
    if candidate.acoustic_confidence < config.min_auto_acoustic_confidence:
        review_reasons.append("LOW_ACOUSTIC_CONFIDENCE")
    if candidate.center_gap_ms > config.max_auto_center_gap_ms:
        review_reasons.append("SPARSE_ACOUSTIC_COVERAGE")
    if (
        boundary.source == "ACOUSTIC_MIDPOINT"
        and candidate.boundary_uncertainty_ms
        > config.max_auto_boundary_uncertainty_ms
    ):
        review_reasons.append("BOUNDARY_LOCALIZATION_UNCERTAIN")
    if candidate.overlap_risk:
        review_reasons.append("OVERLAP_RISK_NOT_EVALUATED")

    reasons = _unique_ordered(tuple(review_reasons))
    automatic = not reasons
    return SplitProposal(
        proposal_id=(
            f"change:{candidate.left_window_id}:"
            f"{candidate.right_window_id}:{boundary.timestamp_ms}"
        ),
        split_ms=boundary.timestamp_ms,
        acoustic_boundary_ms=candidate.acoustic_boundary_ms,
        boundary_uncertainty_ms=candidate.boundary_uncertainty_ms,
        left_window_id=candidate.left_window_id,
        right_window_id=candidate.right_window_id,
        support_window_ids=candidate.support_window_ids,
        change_score=candidate.change_score,
        acoustic_confidence=candidate.acoustic_confidence,
        boundary_source=boundary.source,
        boundary_marker_confidence=boundary.marker_confidence,
        boundary_marker_id=boundary.marker_id,
        review_status=NOT_REQUIRED if automatic else REVIEW_REQUIRED,
        review_reasons=reasons,
        evidence_codes=tuple(evidence_codes),
        apply_automatically=automatic,
        overlap_risk=candidate.overlap_risk,
        overlap_detection_status=(
            OVERLAP_RISK_NOT_EVALUATED
            if candidate.overlap_risk
            else OVERLAP_NOT_EVALUATED
        ),
    )


def _mark_short_resulting_intervals(
    proposals: Sequence[SplitProposal],
    *,
    vad_start_ms: int,
    vad_end_ms: int,
    config: SpeakerChangeDetectionConfig,
) -> tuple[SplitProposal, ...]:
    if not proposals:
        return ()
    ordered = tuple(
        sorted(
            proposals,
            key=lambda item: (
                item.split_ms,
                item.acoustic_boundary_ms,
                item.proposal_id,
            ),
        )
    )
    boundaries = (
        vad_start_ms,
        *(proposal.split_ms for proposal in ordered),
        vad_end_ms,
    )
    short_gaps = {
        index
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]))
        if right - left < config.min_resulting_interval_ms
    }
    output: list[SplitProposal] = []
    for index, proposal in enumerate(ordered):
        if index not in short_gaps and index + 1 not in short_gaps:
            output.append(proposal)
            continue
        reasons = _unique_ordered(
            (*proposal.review_reasons, "SHORT_RESULTING_INTERVAL")
        )
        output.append(
            replace(
                proposal,
                review_status=REVIEW_REQUIRED,
                review_reasons=reasons,
                apply_automatically=False,
            )
        )
    return tuple(output)


def _validate_inputs(
    *,
    vad_start_ms: int,
    vad_end_ms: int,
    windows: Sequence[SpeakerEmbeddingWindow],
    pcm_timeline: Pcm16kMonoTimeline | None,
) -> None:
    if (
        not _is_int(vad_start_ms)
        or not _is_int(vad_end_ms)
        or vad_start_ms < 0
        or vad_end_ms <= vad_start_ms
    ):
        raise ValueError("VAD interval must have a valid time range")
    if pcm_timeline is not None and not isinstance(
        pcm_timeline,
        Pcm16kMonoTimeline,
    ):
        raise ValueError("pcm_timeline must be Pcm16kMonoTimeline")
    if pcm_timeline is not None and vad_end_ms > pcm_timeline.duration_ms:
        raise ValueError("VAD interval exceeds shared PCM duration")
    if not windows:
        return
    if any(not isinstance(window, SpeakerEmbeddingWindow) for window in windows):
        raise ValueError("windows must contain SpeakerEmbeddingWindow values")
    dimensions = {len(window.embedding) for window in windows}
    if len(dimensions) != 1:
        raise ValueError("all speaker embeddings must have the same dimension")
    ids = [window.window_id for window in windows]
    if len(set(ids)) != len(ids):
        raise ValueError("embedding window IDs must be unique")
    previous_start = -1
    previous_center = -1
    for window in windows:
        if window.start_ms < vad_start_ms or window.end_ms > vad_end_ms:
            raise ValueError("embedding window lies outside the VAD interval")
        if window.start_ms <= previous_start or window.center_ms <= previous_center:
            raise ValueError(
                "embedding windows must be strictly time-sorted"
            )
        previous_start = window.start_ms
        previous_center = window.center_ms


def plan_speaker_changes(
    *,
    vad_start_ms: int,
    vad_end_ms: int,
    windows: Sequence[SpeakerEmbeddingWindow],
    energy_valleys: Sequence[EnergyValley] = (),
    asr_boundaries: Sequence[AsrBoundary] = (),
    pcm_timeline: Pcm16kMonoTimeline | None = None,
    config: SpeakerChangeDetectionConfig | None = None,
) -> SpeakerChangePlan:
    """Plan deterministic speaker-change splits inside one VAD interval.

    Only adjacent embedding change scores can create candidates.  Energy and
    ASR markers are considered after candidate creation and only select a
    nearby timestamp.  Review blockers always disable automatic application.
    """

    if config is not None and not isinstance(
        config,
        SpeakerChangeDetectionConfig,
    ):
        raise ValueError("config must be SpeakerChangeDetectionConfig")
    active_config = config or SpeakerChangeDetectionConfig()
    immutable_windows = tuple(windows)
    immutable_energy = tuple(energy_valleys)
    immutable_asr = tuple(asr_boundaries)
    if any(not isinstance(marker, EnergyValley) for marker in immutable_energy):
        raise ValueError("energy_valleys must contain EnergyValley values")
    if any(not isinstance(marker, AsrBoundary) for marker in immutable_asr):
        raise ValueError("asr_boundaries must contain AsrBoundary values")
    _validate_inputs(
        vad_start_ms=vad_start_ms,
        vad_end_ms=vad_end_ms,
        windows=immutable_windows,
        pcm_timeline=pcm_timeline,
    )

    candidates = _build_edge_candidates(immutable_windows, active_config)
    merged_candidates, suppressed_count = _merge_nearby_peaks(
        candidates,
        active_config,
    )
    proposals = tuple(
        _initial_proposal(
            candidate,
            _choose_boundary(
                candidate,
                vad_start_ms=vad_start_ms,
                vad_end_ms=vad_end_ms,
                energy_valleys=immutable_energy,
                asr_boundaries=immutable_asr,
                config=active_config,
            ),
            active_config,
        )
        for candidate in merged_candidates
    )
    proposals = _mark_short_resulting_intervals(
        proposals,
        vad_start_ms=vad_start_ms,
        vad_end_ms=vad_end_ms,
        config=active_config,
    )
    return SpeakerChangePlan(
        vad_start_ms=vad_start_ms,
        vad_end_ms=vad_end_ms,
        proposals=proposals,
        evaluated_edge_count=max(0, len(immutable_windows) - 1),
        candidate_edge_count=len(candidates),
        suppressed_peak_count=suppressed_count,
        pcm_buffer_id=(
            pcm_timeline.buffer_id if pcm_timeline is not None else None
        ),
    )
