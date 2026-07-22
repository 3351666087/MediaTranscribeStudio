from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


CARDINALITY_CHANGE_REVIEW_REQUIRED = (
    "CARDINALITY_CHANGE_REVIEW_REQUIRED"
)
_MAX_TRANSITION_SCORE_DIFFERENTIAL = 0.05
_SCORE_FLOOR = -2.0
_TIE_EPSILON = 1e-12


@dataclass(frozen=True)
class SpeakerEmission:
    speaker_id: str
    score: float


@dataclass(frozen=True)
class SequenceSegment:
    segment_id: str
    start_ms: int
    end_ms: int
    original_speaker_id: str
    speaker_scores: tuple[SpeakerEmission, ...]
    human_locked_speaker_id: str | None = None
    strong_acoustic_anchor: bool = False


@dataclass(frozen=True)
class SequenceDecoderConfig:
    """Configuration for the independent acoustic-first sequence decoder.

    Acoustic emission scores are added directly to the path objective.
    Continuity can change a transition by at most 0.05 relative to a switch,
    so it remains a weak tie-breaker rather than a substitute for voiceprints.
    """

    top_m: int = 3
    continuity_bonus: float = 0.022
    switch_penalty: float = 0.012
    gap_decay_ms: int = 1800
    auto_anchor_min_score: float = 0.82
    auto_anchor_min_margin: float = 0.18
    high_confidence_support_min_score: float = 0.65
    high_confidence_support_min_margin: float = 0.08
    preserve_original_cardinality: bool = False

    def validate(self) -> None:
        if not isinstance(self.preserve_original_cardinality, bool):
            raise ValueError(
                "preserve_original_cardinality must be a boolean"
            )
        if (
            isinstance(self.top_m, bool)
            or not isinstance(self.top_m, int)
            or self.top_m < 1
        ):
            raise ValueError("top_m must be a positive integer")
        if (
            not math.isfinite(self.continuity_bonus)
            or self.continuity_bonus < 0.0
        ):
            raise ValueError("continuity_bonus must be finite and non-negative")
        if (
            not math.isfinite(self.switch_penalty)
            or self.switch_penalty < 0.0
        ):
            raise ValueError("switch_penalty must be finite and non-negative")
        if (
            self.continuity_bonus + self.switch_penalty
            > _MAX_TRANSITION_SCORE_DIFFERENTIAL
        ):
            raise ValueError(
                "continuity and gap priors must remain weak: their combined "
                "transition differential cannot exceed 0.05"
            )
        if (
            isinstance(self.gap_decay_ms, bool)
            or not isinstance(self.gap_decay_ms, int)
            or self.gap_decay_ms < 1
        ):
            raise ValueError("gap_decay_ms must be a positive integer")
        for name, value in (
            ("auto_anchor_min_score", self.auto_anchor_min_score),
            ("auto_anchor_min_margin", self.auto_anchor_min_margin),
            (
                "high_confidence_support_min_score",
                self.high_confidence_support_min_score,
            ),
            (
                "high_confidence_support_min_margin",
                self.high_confidence_support_min_margin,
            ),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not -2.0 <= self.auto_anchor_min_score <= 2.0:
            raise ValueError("auto_anchor_min_score must be between -2 and 2")
        if not 0.0 <= self.auto_anchor_min_margin <= 4.0:
            raise ValueError("auto_anchor_min_margin must be between 0 and 4")
        if not -2.0 <= self.high_confidence_support_min_score <= 2.0:
            raise ValueError(
                "high_confidence_support_min_score must be between -2 and 2"
            )
        if not 0.0 <= self.high_confidence_support_min_margin <= 4.0:
            raise ValueError(
                "high_confidence_support_min_margin must be between 0 and 4"
            )


@dataclass(frozen=True)
class DecodedSpeakerAssignment:
    segment_id: str
    original_speaker_id: str
    speaker_id: str
    changed: bool
    acoustic_score: float
    review_status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class SequenceDecodeResult:
    assignments: tuple[DecodedSpeakerAssignment, ...]
    total_score: float
    review_required: bool
    method: str = "acoustic-topm-viterbi-v1"


@dataclass(frozen=True)
class _PreparedSegment:
    segment: SequenceSegment
    score_by_speaker: dict[str, float]
    ranked_scores: tuple[SpeakerEmission, ...]
    candidates: tuple[str, ...]
    anchor: bool


def decode_speaker_sequence(
    segments: Sequence[SequenceSegment],
    config: SequenceDecoderConfig | None = None,
) -> SequenceDecodeResult:
    """Decode a deterministic global speaker path without mutating input.

    It uses local top-M acoustic scores as emissions and applies only a small,
    gap-decayed continuity prior. Human locks and strong acoustic anchors are
    hard constraints. A post-decode guard preserves at least one original
    high-confidence support segment for every evidenced speaker.
    """

    resolved_config = config or SequenceDecoderConfig()
    resolved_config.validate()
    if not segments:
        return SequenceDecodeResult(
            assignments=(),
            total_score=0.0,
            review_required=False,
        )

    original_order = tuple(segments)
    _validate_segments(original_order)
    chronological = tuple(
        sorted(
            original_order,
            key=lambda item: (
                item.start_ms,
                item.end_ms,
                item.segment_id,
            ),
        )
    )
    prepared = tuple(
        _prepare_segment(segment, resolved_config)
        for segment in chronological
    )
    decoded_path = list(_viterbi_decode(prepared, resolved_config))
    reasons_by_id: dict[str, list[str]] = {
        item.segment.segment_id: [] for item in prepared
    }

    for item, speaker_id in zip(prepared, decoded_path):
        segment = item.segment
        reasons = reasons_by_id[segment.segment_id]
        if segment.human_locked_speaker_id is not None:
            reasons.append("HUMAN_LOCKED")
        if item.anchor:
            reasons.append("STRONG_ACOUSTIC_ANCHOR")
        if speaker_id != segment.original_speaker_id:
            reasons.append("SEQUENCE_REASSIGNED")

    if resolved_config.preserve_original_cardinality:
        _protect_speaker_cardinality(
            prepared,
            decoded_path,
            reasons_by_id,
            resolved_config,
        )
    else:
        _protect_last_high_confidence_support(
            prepared,
            decoded_path,
            reasons_by_id,
            resolved_config,
        )

    decoded_by_id = {
        item.segment.segment_id: speaker_id
        for item, speaker_id in zip(prepared, decoded_path)
    }
    prepared_by_id = {
        item.segment.segment_id: item for item in prepared
    }
    assignments: list[DecodedSpeakerAssignment] = []
    for segment in original_order:
        item = prepared_by_id[segment.segment_id]
        speaker_id = decoded_by_id[segment.segment_id]
        reason_codes = tuple(reasons_by_id[segment.segment_id])
        review_required = (
            CARDINALITY_CHANGE_REVIEW_REQUIRED in reason_codes
        )
        assignments.append(
            DecodedSpeakerAssignment(
                segment_id=segment.segment_id,
                original_speaker_id=segment.original_speaker_id,
                speaker_id=speaker_id,
                changed=speaker_id != segment.original_speaker_id,
                acoustic_score=item.score_by_speaker.get(
                    speaker_id,
                    _SCORE_FLOOR,
                ),
                review_status=(
                    "REVIEW_REQUIRED"
                    if review_required
                    else "NOT_REQUIRED"
                ),
                reason_codes=reason_codes,
            )
        )

    total_score = _score_path(prepared, decoded_path, resolved_config)
    return SequenceDecodeResult(
        assignments=tuple(assignments),
        total_score=total_score,
        review_required=any(
            assignment.review_status == "REVIEW_REQUIRED"
            for assignment in assignments
        ),
    )


def _validate_segments(segments: Sequence[SequenceSegment]) -> None:
    seen_ids: set[str] = set()
    for segment in segments:
        if not segment.segment_id.strip():
            raise ValueError("segment_id must not be empty")
        if segment.segment_id in seen_ids:
            raise ValueError(
                f"duplicate segment_id: {segment.segment_id}"
            )
        seen_ids.add(segment.segment_id)
        if (
            isinstance(segment.start_ms, bool)
            or isinstance(segment.end_ms, bool)
            or not isinstance(segment.start_ms, int)
            or not isinstance(segment.end_ms, int)
            or segment.start_ms < 0
            or segment.end_ms <= segment.start_ms
        ):
            raise ValueError(
                f"{segment.segment_id} has invalid start_ms/end_ms"
            )
        if not segment.original_speaker_id.strip():
            raise ValueError(
                f"{segment.segment_id} original_speaker_id must not be empty"
            )
        if (
            segment.human_locked_speaker_id is not None
            and not segment.human_locked_speaker_id.strip()
        ):
            raise ValueError(
                f"{segment.segment_id} human lock must not be empty"
            )
        if (
            segment.strong_acoustic_anchor
            and segment.human_locked_speaker_id is not None
            and segment.human_locked_speaker_id
            != segment.original_speaker_id
        ):
            raise ValueError(
                f"{segment.segment_id} has conflicting human lock and "
                "strong acoustic anchor"
            )
        if not segment.speaker_scores:
            raise ValueError(
                f"{segment.segment_id} must contain speaker_scores"
            )


def _prepare_segment(
    segment: SequenceSegment,
    config: SequenceDecoderConfig,
) -> _PreparedSegment:
    score_by_speaker: dict[str, float] = {}
    for item in segment.speaker_scores:
        speaker_id = item.speaker_id.strip()
        if not speaker_id:
            raise ValueError(
                f"{segment.segment_id} contains an empty speaker_id"
            )
        if not math.isfinite(item.score) or not -2.0 <= item.score <= 2.0:
            raise ValueError(
                f"{segment.segment_id} score for {speaker_id} must be finite "
                "and between -2 and 2"
            )
        score_by_speaker[speaker_id] = max(
            item.score,
            score_by_speaker.get(speaker_id, _SCORE_FLOOR),
        )

    ranked_scores = tuple(
        SpeakerEmission(speaker_id=speaker_id, score=score)
        for speaker_id, score in sorted(
            score_by_speaker.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    if (
        segment.strong_acoustic_anchor
        and segment.original_speaker_id not in score_by_speaker
    ):
        raise ValueError(
            f"{segment.segment_id} strong acoustic anchor must include an "
            "emission for its original speaker"
        )
    automatic_anchor = _is_automatic_anchor(
        segment,
        ranked_scores,
        config,
    )
    anchor = segment.strong_acoustic_anchor or automatic_anchor

    if segment.human_locked_speaker_id is not None:
        candidates = (segment.human_locked_speaker_id,)
    elif anchor:
        candidates = (segment.original_speaker_id,)
    else:
        top_speakers = [
            item.speaker_id for item in ranked_scores[: config.top_m]
        ]
        if segment.original_speaker_id not in top_speakers:
            top_speakers.append(segment.original_speaker_id)
        candidates = tuple(sorted(set(top_speakers)))

    return _PreparedSegment(
        segment=segment,
        score_by_speaker=score_by_speaker,
        ranked_scores=ranked_scores,
        candidates=candidates,
        anchor=anchor,
    )


def _is_automatic_anchor(
    segment: SequenceSegment,
    ranked_scores: Sequence[SpeakerEmission],
    config: SequenceDecoderConfig,
) -> bool:
    top = ranked_scores[0]
    runner_up_score = (
        ranked_scores[1].score
        if len(ranked_scores) > 1
        else _SCORE_FLOOR
    )
    return (
        segment.human_locked_speaker_id is None
        and top.speaker_id == segment.original_speaker_id
        and top.score >= config.auto_anchor_min_score
        and top.score - runner_up_score >= config.auto_anchor_min_margin
    )


def _is_high_confidence_support(
    item: _PreparedSegment,
    config: SequenceDecoderConfig,
) -> bool:
    segment = item.segment
    if (
        segment.human_locked_speaker_id is not None
        and segment.human_locked_speaker_id
        != segment.original_speaker_id
    ):
        return False
    top = item.ranked_scores[0]
    runner_up_score = (
        item.ranked_scores[1].score
        if len(item.ranked_scores) > 1
        else _SCORE_FLOOR
    )
    return (
        top.speaker_id == segment.original_speaker_id
        and top.score >= config.high_confidence_support_min_score
        and top.score - runner_up_score
        >= config.high_confidence_support_min_margin
    )


def _transition_score(
    previous: _PreparedSegment,
    previous_speaker: str,
    current: _PreparedSegment,
    current_speaker: str,
    config: SequenceDecoderConfig,
) -> float:
    gap_ms = max(
        0,
        current.segment.start_ms - previous.segment.end_ms,
    )
    proximity = math.exp(-gap_ms / config.gap_decay_ms)
    if previous_speaker == current_speaker:
        return config.continuity_bonus * proximity
    return -config.switch_penalty * proximity


def _viterbi_decode(
    prepared: Sequence[_PreparedSegment],
    config: SequenceDecoderConfig,
) -> tuple[str, ...]:
    scores: dict[str, float] = {
        speaker_id: prepared[0].score_by_speaker.get(
            speaker_id,
            _SCORE_FLOOR,
        )
        for speaker_id in prepared[0].candidates
    }
    backpointers: list[dict[str, str | None]] = [
        {speaker_id: None for speaker_id in prepared[0].candidates}
    ]

    for index in range(1, len(prepared)):
        previous = prepared[index - 1]
        current = prepared[index]
        next_scores: dict[str, float] = {}
        current_backpointers: dict[str, str | None] = {}
        for current_speaker in sorted(current.candidates):
            emission = current.score_by_speaker.get(
                current_speaker,
                _SCORE_FLOOR,
            )
            best_score = -math.inf
            best_previous: str | None = None
            for previous_speaker in sorted(scores):
                candidate_score = (
                    scores[previous_speaker]
                    + _transition_score(
                        previous,
                        previous_speaker,
                        current,
                        current_speaker,
                        config,
                    )
                    + emission
                )
                if (
                    candidate_score > best_score + _TIE_EPSILON
                    or (
                        abs(candidate_score - best_score) <= _TIE_EPSILON
                        and (
                            best_previous is None
                            or previous_speaker < best_previous
                        )
                    )
                ):
                    best_score = candidate_score
                    best_previous = previous_speaker
            next_scores[current_speaker] = best_score
            current_backpointers[current_speaker] = best_previous
        scores = next_scores
        backpointers.append(current_backpointers)

    final_speaker = min(
        scores,
        key=lambda speaker_id: (-scores[speaker_id], speaker_id),
    )
    path = [final_speaker]
    for index in range(len(prepared) - 1, 0, -1):
        previous_speaker = backpointers[index][path[-1]]
        if previous_speaker is None:
            raise RuntimeError("sequence decoder backpointer is missing")
        path.append(previous_speaker)
    path.reverse()
    return tuple(path)


def _collect_support_indices(
    prepared: Sequence[_PreparedSegment],
    config: SequenceDecoderConfig,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    support_indices: dict[str, list[int]] = {}
    high_confidence_indices: dict[str, list[int]] = {}
    for index, item in enumerate(prepared):
        speaker_id = item.segment.original_speaker_id
        if (
            speaker_id not in item.score_by_speaker
            or (
                item.segment.human_locked_speaker_id is not None
                and item.segment.human_locked_speaker_id != speaker_id
            )
        ):
            continue
        support_indices.setdefault(speaker_id, []).append(index)
        if _is_high_confidence_support(item, config):
            high_confidence_indices.setdefault(speaker_id, []).append(index)
    return support_indices, high_confidence_indices


def _restore_missing_speakers(
    prepared: Sequence[_PreparedSegment],
    decoded_path: list[str],
    reasons_by_id: dict[str, list[str]],
    support_indices: dict[str, list[int]],
    *,
    preferred_indices: dict[str, list[int]],
) -> None:
    for speaker_id in sorted(support_indices):
        indices = support_indices[speaker_id]
        if speaker_id in decoded_path:
            continue
        candidate_indices = preferred_indices.get(
            speaker_id,
            indices,
        )
        restore_index = min(
            candidate_indices,
            key=lambda index: (
                -prepared[index].score_by_speaker.get(
                    speaker_id,
                    _SCORE_FLOOR,
                ),
                -_speaker_margin(prepared[index]),
                prepared[index].segment.start_ms,
                prepared[index].segment.segment_id,
            ),
        )
        segment = prepared[restore_index].segment
        decoded_path[restore_index] = segment.original_speaker_id
        reasons = reasons_by_id[segment.segment_id]
        if "SEQUENCE_REASSIGNED" in reasons:
            reasons.remove("SEQUENCE_REASSIGNED")
        reasons.append(CARDINALITY_CHANGE_REVIEW_REQUIRED)


def _protect_last_high_confidence_support(
    prepared: Sequence[_PreparedSegment],
    decoded_path: list[str],
    reasons_by_id: dict[str, list[str]],
    config: SequenceDecoderConfig,
) -> None:
    _, high_confidence_indices = _collect_support_indices(
        prepared,
        config,
    )
    _restore_missing_speakers(
        prepared,
        decoded_path,
        reasons_by_id,
        high_confidence_indices,
        preferred_indices=high_confidence_indices,
    )


def _protect_speaker_cardinality(
    prepared: Sequence[_PreparedSegment],
    decoded_path: list[str],
    reasons_by_id: dict[str, list[str]],
    config: SequenceDecoderConfig,
) -> None:
    support_indices, high_confidence_indices = _collect_support_indices(
        prepared,
        config,
    )
    _restore_missing_speakers(
        prepared,
        decoded_path,
        reasons_by_id,
        support_indices,
        preferred_indices=high_confidence_indices,
    )


def _speaker_margin(item: _PreparedSegment) -> float:
    top = item.ranked_scores[0].score
    runner_up = (
        item.ranked_scores[1].score
        if len(item.ranked_scores) > 1
        else _SCORE_FLOOR
    )
    return top - runner_up


def _score_path(
    prepared: Sequence[_PreparedSegment],
    path: Sequence[str],
    config: SequenceDecoderConfig,
) -> float:
    total = 0.0
    for index, (item, speaker_id) in enumerate(zip(prepared, path)):
        total += item.score_by_speaker.get(speaker_id, _SCORE_FLOOR)
        if index:
            total += _transition_score(
                prepared[index - 1],
                path[index - 1],
                item,
                speaker_id,
                config,
            )
    return total
