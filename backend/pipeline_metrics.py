"""Runtime and reference-quality metrics for the offline speaker pipeline.

The module is deliberately standard-library only.  Runtime metrics are always
available, while reference quality metrics are emitted only when real reference
annotations are supplied by the caller.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .errors import WorkerError
from .speaker_timeline import speaker_timeline_turns


METRICS_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ReferenceTurn:
    """A reference interval with one or more simultaneously active speakers."""

    start_ms: int
    end_ms: int
    speaker_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_ms, bool)
            or isinstance(self.end_ms, bool)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValueError("reference turn boundaries are invalid")
        if not self.speaker_ids or any(
            not isinstance(item, str) or not item.strip()
            for item in self.speaker_ids
        ):
            raise ValueError("reference turn must contain speaker IDs")
        if len(set(self.speaker_ids)) != len(self.speaker_ids):
            raise ValueError("reference turn speaker IDs must be unique")


def _percentile(samples: Sequence[float], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(float(item) for item in samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class PipelineMetricsCollector:
    """Collect stage, cache, routing, and optional reference metrics."""

    def __init__(
        self,
        *,
        job_id: str,
        duration_ms: int | None = None,
        started_at: float | None = None,
    ) -> None:
        if duration_ms is not None and duration_ms < 1:
            raise ValueError("duration_ms must be positive")
        self.job_id = str(job_id)
        self.duration_ms = int(duration_ms) if duration_ms is not None else None
        self.started_at = (
            float(started_at) if started_at is not None else time.perf_counter()
        )
        self._stage_samples: dict[str, list[float]] = defaultdict(list)
        self._cache: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "requests": 0,
                "hits": 0,
                "misses": 0,
                "recomputations": 0,
            }
        )
        self._route_segments = 0
        self._escalated = 0
        self._protected = 0
        self._reason_counts: dict[str, int] = defaultdict(int)
        self._quality: dict[str, float] | None = None
        self._escalations: list[dict[str, Any]] = []
        self._cascade: list[dict[str, Any]] = []
        self._peak_ram_mb = 0.0
        self._peak_vram_mb = 0.0
        self._policy: dict[str, Any] = {
            "primaryVoiceprint": "CAM++",
            "secondaryVoiceprint": "unconfigured",
            "secondaryScope": "difficult-segments-only",
            "fullCorpusSecondaryRunAllowed": False,
            "pyannoteMode": "disabled",
            "pyannoteTelemetryEnabled": False,
            "localLlmModel": "qwen3.5:27b-q4_K_M",
            "localLlmMode": "disabled",
            "localLlmAutoApply": False,
        }

    def set_duration_ms(self, duration_ms: int) -> None:
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
            raise ValueError("duration_ms must be a positive integer")
        if duration_ms < 1:
            raise ValueError("duration_ms must be positive")
        if self.duration_ms is not None and self.duration_ms != duration_ms:
            raise ValueError("duration_ms cannot change after it is set")
        self.duration_ms = duration_ms

    def set_policy(self, **values: Any) -> None:
        for key, value in values.items():
            normalized = str(key or "").strip()
            if not normalized:
                raise ValueError("policy keys must be non-empty")
            if not isinstance(value, (str, bool, int, float)) and value is not None:
                raise ValueError("policy values must be scalar JSON values")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("policy values must be finite")
            self._policy[normalized] = value

    def record_resource_peak(
        self,
        *,
        ram_mb: float = 0.0,
        vram_mb: float = 0.0,
    ) -> None:
        ram = float(ram_mb)
        vram = float(vram_mb)
        if (
            not math.isfinite(ram)
            or not math.isfinite(vram)
            or ram < 0.0
            or vram < 0.0
        ):
            raise ValueError("resource peaks must be finite and non-negative")
        self._peak_ram_mb = max(self._peak_ram_mb, ram)
        self._peak_vram_mb = max(self._peak_vram_mb, vram)

    def record_escalation(
        self,
        *,
        stage: str,
        segment_id: str,
        reasons: Sequence[str],
        cache_key: str,
        cache_hit: bool,
        latency_ms: float,
        resource: Mapping[str, Any],
        confidence: float,
        exit_reason: str,
        provider: str,
        applied: bool,
    ) -> None:
        normalized_stage = str(stage or "").strip()
        normalized_id = str(segment_id or "").strip()
        normalized_reasons = tuple(str(item or "").strip() for item in reasons)
        normalized_key = str(cache_key or "").strip()
        normalized_exit = str(exit_reason or "").strip()
        normalized_provider = str(provider or "").strip()
        latency = float(latency_ms)
        probability = float(confidence)
        if (
            not normalized_stage
            or not normalized_id
            or not normalized_reasons
            or any(not item for item in normalized_reasons)
            or not normalized_key
            or not normalized_exit
            or not normalized_provider
            or not math.isfinite(latency)
            or latency < 0.0
            or not math.isfinite(probability)
            or probability < 0.0
            or probability > 1.0
            or not isinstance(cache_hit, bool)
            or not isinstance(applied, bool)
        ):
            raise ValueError("escalation telemetry is incomplete or invalid")
        normalized_resource: dict[str, float | None] = {
            "ramMb": None,
            "vramMb": None,
        }
        for key, raw in resource.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("resource keys must be non-empty strings")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("resource values must be finite and non-negative")
            normalized_resource[key] = value
        self.record_resource_peak(
            ram_mb=normalized_resource.get("ramMb") or 0.0,
            vram_mb=normalized_resource.get("vramMb") or 0.0,
        )
        self._escalations.append(
            {
                "stage": normalized_stage,
                "segmentId": normalized_id,
                "triggerReason": normalized_reasons[0],
                "reasons": list(normalized_reasons),
                "cacheKey": normalized_key,
                "cacheHit": cache_hit,
                "latencyMs": round(latency, 6),
                "resource": normalized_resource,
                "confidence": round(probability, 9),
                "exitReason": normalized_exit,
                "provider": normalized_provider,
                "applied": applied,
            }
        )

    def record_cascade_stage(
        self,
        *,
        stage: str,
        provider: str,
        trigger_reason: str,
        candidate_ids: Sequence[str],
        candidate_scope: str,
        source_count: int,
        max_candidates: int | None,
        candidate_start_ms: int | None,
        candidate_end_ms: int | None,
        invoked: bool,
        cache_stage: str,
        latency_ms: float,
        resource: Mapping[str, Any] | None,
        confidence: float | None,
        exit_reason: str,
    ) -> None:
        """Record one stable-schema audit event for a cascade level."""

        normalized_stage = str(stage or "").strip()
        normalized_provider = str(provider or "").strip()
        normalized_trigger = str(trigger_reason or "").strip()
        normalized_scope = str(candidate_scope or "").strip()
        normalized_cache_stage = str(cache_stage or "").strip()
        normalized_exit = str(exit_reason or "").strip()
        normalized_ids = tuple(str(item or "").strip() for item in candidate_ids)
        latency = float(latency_ms)
        if (
            not normalized_stage
            or not normalized_provider
            or not normalized_trigger
            or not normalized_scope
            or not normalized_cache_stage
            or not normalized_exit
            or any(not item for item in normalized_ids)
            or len(set(normalized_ids)) != len(normalized_ids)
            or isinstance(source_count, bool)
            or int(source_count) < len(normalized_ids)
            or not isinstance(invoked, bool)
            or not math.isfinite(latency)
            or latency < 0.0
        ):
            raise ValueError("cascade stage telemetry is incomplete or invalid")
        if any(item["stage"] == normalized_stage for item in self._cascade):
            raise ValueError(f"cascade stage {normalized_stage} was recorded twice")
        if max_candidates is not None and (
            isinstance(max_candidates, bool)
            or int(max_candidates) < len(normalized_ids)
        ):
            raise ValueError("cascade max_candidates is below the candidate count")
        if (candidate_start_ms is None) != (candidate_end_ms is None):
            raise ValueError("cascade candidate bounds must be both present or absent")
        if candidate_start_ms is not None and (
            isinstance(candidate_start_ms, bool)
            or isinstance(candidate_end_ms, bool)
            or int(candidate_start_ms) < 0
            or int(candidate_end_ms) <= int(candidate_start_ms)
        ):
            raise ValueError("cascade candidate bounds are invalid")

        normalized_confidence: float | None = None
        if confidence is not None:
            normalized_confidence = float(confidence)
            if (
                not math.isfinite(normalized_confidence)
                or normalized_confidence < 0.0
                or normalized_confidence > 1.0
            ):
                raise ValueError("cascade confidence must be between 0 and 1")

        normalized_resource: dict[str, float | None] = {
            "ramMb": None,
            "vramMb": None,
        }
        for key, raw in (resource or {}).items():
            if key not in normalized_resource:
                continue
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("cascade resource values must be non-negative")
            normalized_resource[key] = value
        self.record_resource_peak(
            ram_mb=normalized_resource["ramMb"] or 0.0,
            vram_mb=normalized_resource["vramMb"] or 0.0,
        )

        cache = self._cache.get(
            normalized_cache_stage,
            {
                "requests": 0,
                "hits": 0,
                "misses": 0,
                "recomputations": 0,
            },
        )
        self._cascade.append(
            {
                "stage": normalized_stage,
                "provider": normalized_provider,
                "triggerReason": normalized_trigger,
                "candidateRange": {
                    "scope": normalized_scope,
                    "sourceCount": int(source_count),
                    "candidateCount": len(normalized_ids),
                    "maxCandidates": (
                        int(max_candidates)
                        if max_candidates is not None
                        else None
                    ),
                    "segmentIds": list(normalized_ids),
                    "startMs": (
                        int(candidate_start_ms)
                        if candidate_start_ms is not None
                        else None
                    ),
                    "endMs": (
                        int(candidate_end_ms)
                        if candidate_end_ms is not None
                        else None
                    ),
                },
                "cache": {
                    "stage": normalized_cache_stage,
                    "requests": int(cache["requests"]),
                    "hits": int(cache["hits"]),
                    "misses": int(cache["misses"]),
                    "recomputations": int(cache["recomputations"]),
                },
                "latencyMs": round(latency, 6),
                "resource": normalized_resource,
                "confidence": (
                    round(normalized_confidence, 9)
                    if normalized_confidence is not None
                    else None
                ),
                "invoked": invoked,
                "exitReason": normalized_exit,
            }
        )

    @contextmanager
    def stage(self, stage_name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_stage(stage_name, (time.perf_counter() - started) * 1000.0)

    def record_stage(self, stage_name: str, elapsed_ms: float) -> None:
        normalized = str(stage_name or "").strip()
        value = float(elapsed_ms)
        if not normalized or not math.isfinite(value) or value < 0.0:
            raise ValueError("stage sample must have a name and finite duration")
        self._stage_samples[normalized].append(value)

    def record_adapter_stage_samples(
        self,
        samples: Mapping[str, float] | None,
    ) -> None:
        """Record fused preparation sub-stage timings supplied by an adapter."""

        for name, value in (samples or {}).items():
            self.record_stage(str(name), float(value))

    def record_cache(
        self,
        stage_name: str,
        *,
        requests: int,
        hits: int,
        misses: int,
        recomputations: int = 0,
    ) -> None:
        values = (requests, hits, misses, recomputations)
        if any(isinstance(item, bool) or int(item) < 0 for item in values):
            raise ValueError("cache counters must be non-negative integers")
        if hits + misses != requests:
            raise ValueError("cache hits plus misses must equal requests")
        if recomputations > misses:
            raise ValueError("cache recomputations cannot exceed misses")
        entry = self._cache[str(stage_name)]
        entry["requests"] += int(requests)
        entry["hits"] += int(hits)
        entry["misses"] += int(misses)
        entry["recomputations"] += int(recomputations)

    def record_routing(
        self,
        *,
        segments: int,
        escalated: int,
        protected: int,
        reason_counts: Mapping[str, int],
    ) -> None:
        if segments < 0 or escalated < 0 or protected < 0:
            raise ValueError("routing counters must be non-negative")
        if escalated > segments or protected > segments:
            raise ValueError("routing counters cannot exceed segment count")
        self._route_segments += int(segments)
        self._escalated += int(escalated)
        self._protected += int(protected)
        for reason, count in reason_counts.items():
            normalized = str(reason or "").strip()
            if not normalized or int(count) < 0:
                raise ValueError("invalid routing reason counter")
            self._reason_counts[normalized] += int(count)

    def set_reference_quality(self, quality: Mapping[str, float]) -> None:
        required = {"der", "jer", "speakerConfusion", "overlapF1"}
        if set(quality) != required:
            raise ValueError("reference quality metrics are incomplete")
        normalized: dict[str, float] = {}
        for key, raw in quality.items():
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be finite and non-negative")
            normalized[key] = value
        self._quality = normalized

    def as_dict(self) -> dict[str, Any]:
        if self.duration_ms is None:
            raise ValueError("duration_ms must be set before metrics are emitted")
        elapsed_ms = max(0.0, (time.perf_counter() - self.started_at) * 1000.0)
        stages: dict[str, dict[str, float | int]] = {}
        for name, samples in sorted(self._stage_samples.items()):
            stages[name] = {
                "count": len(samples),
                "totalMs": round(sum(samples), 6),
                "p50Ms": round(_percentile(samples, 0.50), 6),
                "p95Ms": round(_percentile(samples, 0.95), 6),
            }

        aggregate = {
            key: sum(stage[key] for stage in self._cache.values())
            for key in ("requests", "hits", "misses", "recomputations")
        }
        requests = aggregate["requests"]
        segments = self._route_segments
        value: dict[str, Any] = {
            "schemaVersion": METRICS_SCHEMA_VERSION,
            "jobId": self.job_id,
            "offline": True,
            "durationMs": self.duration_ms,
            "runtime": {
                "elapsedMs": round(elapsed_ms, 6),
                "rtf": round(elapsed_ms / self.duration_ms, 9),
                "stages": stages,
            },
            "cache": {
                **aggregate,
                "hitRate": round(
                    aggregate["hits"] / requests if requests else 0.0,
                    9,
                ),
                "recomputationRate": round(
                    aggregate["recomputations"] / requests if requests else 0.0,
                    9,
                ),
                "byStage": {
                    name: {
                        **entry,
                        "hitRate": round(
                            entry["hits"] / entry["requests"]
                            if entry["requests"]
                            else 0.0,
                            9,
                        ),
                        "recomputationRate": round(
                            entry["recomputations"] / entry["requests"]
                            if entry["requests"]
                            else 0.0,
                            9,
                        ),
                    }
                    for name, entry in sorted(self._cache.items())
                },
            },
            "routing": {
                "segments": segments,
                "escalated": self._escalated,
                "escalationRate": round(
                    self._escalated / segments if segments else 0.0,
                    9,
                ),
                "protected": self._protected,
                "reasonCounts": dict(sorted(self._reason_counts.items())),
            },
            "escalations": list(self._escalations),
            "cascade": list(self._cascade),
            "resources": {
                "peakRamMb": round(self._peak_ram_mb, 6),
                "peakVramMb": round(self._peak_vram_mb, 6),
            },
            "policy": dict(sorted(self._policy.items())),
            "referenceEvaluation": {"available": self._quality is not None},
        }
        if self._quality is not None:
            value["quality"] = {
                key: round(metric, 9)
                for key, metric in self._quality.items()
            }
        return value


def maximum_weight_assignment(
    weights: Sequence[Sequence[float]],
) -> list[tuple[int, int]]:
    """Return a maximum-weight one-to-one assignment using Hungarian O(n^3)."""

    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    if not row_count or not column_count:
        return []
    size = max(row_count, column_count)
    maximum = max(
        (float(item) for row in weights for item in row),
        default=0.0,
    )
    # Classic 1-indexed Hungarian minimization over a padded square matrix.
    cost = [
        [maximum for _ in range(size)]
        for _ in range(size)
    ]
    for row_index, row in enumerate(weights):
        for column_index, weight in enumerate(row):
            cost[row_index][column_index] = maximum - float(weight)

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    output: list[tuple[int, int]] = []
    for column in range(1, size + 1):
        row = p[column] - 1
        col = column - 1
        if row < row_count and col < column_count:
            output.append((row, col))
    return output


def _active_reference(
    turns: Sequence[ReferenceTurn],
    start_ms: int,
    end_ms: int,
) -> set[str]:
    return {
        speaker_id
        for turn in turns
        if turn.start_ms < end_ms and turn.end_ms > start_ms
        for speaker_id in turn.speaker_ids
    }


def _active_prediction(
    segments: Sequence[Any],
    start_ms: int,
    end_ms: int,
) -> set[str]:
    active: set[str] = set()
    for segment in segments:
        if (
            int(getattr(segment, "start_ms")) >= end_ms
            or int(getattr(segment, "end_ms")) <= start_ms
        ):
            continue
        canonical_turns = _canonical_speaker_turns(segment)
        if canonical_turns is not None:
            active.update(
                speaker_id
                for turn_start, turn_end, speaker_id in canonical_turns
                if turn_start < end_ms and turn_end > start_ms
            )
            continue
        active.add(str(getattr(segment, "speaker_id", "")))
    return active


def _active_timeline_prediction(
    turns: Sequence[tuple[int, int, str]],
    start_ms: int,
    end_ms: int,
) -> set[str]:
    return {
        speaker_id
        for turn_start, turn_end, speaker_id in turns
        if turn_start < end_ms and turn_end > start_ms
    }


def _regular_timeline_turns(
    speaker_timeline: Mapping[str, Any],
) -> tuple[tuple[int, int, str], ...]:
    turns: list[tuple[int, int, str]] = []
    for index, raw in enumerate(
        speaker_timeline_turns(speaker_timeline, mode="regular")
    ):
        start_ms = raw.get("startMs")
        end_ms = raw.get("endMs")
        speaker_id = raw.get("speakerId")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
            or not isinstance(speaker_id, str)
            or not speaker_id.strip()
        ):
            raise WorkerError(
                "SPEAKER_TIMELINE_INVALID",
                "regular speaker timeline contains a malformed turn",
                details={"turnIndex": index},
            )
        turns.append((start_ms, end_ms, speaker_id.strip()))
    if not turns:
        raise WorkerError(
            "SPEAKER_TIMELINE_INVALID",
            "regular speaker timeline must contain at least one turn",
        )
    return tuple(turns)


def _canonical_speaker_turns(
    segment: Any,
) -> tuple[tuple[int, int, str], ...] | None:
    evidence = getattr(segment, "evidence", None)
    if not isinstance(evidence, Mapping):
        return None
    overlap = evidence.get("overlap")
    if not isinstance(overlap, Mapping) or "canonicalSpeakerTurns" not in overlap:
        return None
    raw_turns = overlap.get("canonicalSpeakerTurns")
    if not isinstance(raw_turns, Sequence) or isinstance(
        raw_turns,
        (str, bytes, bytearray),
    ):
        return None
    turns: list[tuple[int, int, str]] = []
    segment_start = int(getattr(segment, "start_ms"))
    segment_end = int(getattr(segment, "end_ms"))
    for raw in raw_turns:
        if not isinstance(raw, Mapping):
            return None
        start_ms = raw.get("startMs")
        end_ms = raw.get("endMs")
        speaker_id = raw.get("speakerId")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or not isinstance(speaker_id, str)
            or not speaker_id.strip()
            or start_ms < segment_start
            or end_ms > segment_end
            or end_ms <= start_ms
        ):
            return None
        turns.append((start_ms, end_ms, speaker_id.strip()))
    return tuple(turns)


def _exact_overlap_intervals(
    segment: Any,
) -> tuple[tuple[int, int], ...] | None:
    evidence = getattr(segment, "evidence", None)
    if not isinstance(evidence, Mapping):
        return None
    overlap = evidence.get("overlap")
    if not isinstance(overlap, Mapping) or "overlapIntervals" not in overlap:
        return None
    raw_intervals = overlap.get("overlapIntervals")
    if not isinstance(raw_intervals, Sequence) or isinstance(
        raw_intervals,
        (str, bytes, bytearray),
    ):
        return None
    intervals: list[tuple[int, int]] = []
    segment_start = int(getattr(segment, "start_ms"))
    segment_end = int(getattr(segment, "end_ms"))
    for raw in raw_intervals:
        if not isinstance(raw, Mapping):
            return None
        start_ms = raw.get("startMs")
        end_ms = raw.get("endMs")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < segment_start
            or end_ms > segment_end
            or end_ms <= start_ms
        ):
            return None
        intervals.append((start_ms, end_ms))
    return tuple(intervals)


def _predicted_overlap(
    segments: Sequence[Any],
    start_ms: int,
    end_ms: int,
    predicted_speakers: set[str],
) -> bool:
    for segment in segments:
        if (
            int(getattr(segment, "start_ms")) >= end_ms
            or int(getattr(segment, "end_ms")) <= start_ms
        ):
            continue
        exact = _exact_overlap_intervals(segment)
        if exact is not None:
            if any(
                overlap_start < end_ms and overlap_end > start_ms
                for overlap_start, overlap_end in exact
            ):
                return True
            continue
        if bool(getattr(segment, "overlapping", False)):
            return True
    return len(predicted_speakers) > 1


def evaluate_reference_quality(
    segments: Sequence[Any],
    reference_turns: Sequence[ReferenceTurn],
    *,
    speaker_timeline: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Compute time-weighted DER/JER/confusion/overlap F1.

    No value is returned or fabricated when reference annotations are absent;
    callers must only invoke this function with real non-empty annotations.
    """

    if not reference_turns:
        raise WorkerError(
            "REFERENCE_LABELS_INVALID",
            "reference metrics require at least one labelled interval",
        )
    timeline_turns = (
        _regular_timeline_turns(speaker_timeline)
        if speaker_timeline is not None
        else None
    )
    boundaries = {
        boundary
        for turn in reference_turns
        for boundary in (turn.start_ms, turn.end_ms)
    }
    if timeline_turns is not None:
        boundaries.update(
            boundary
            for turn in timeline_turns
            for boundary in turn[:2]
        )
    else:
        boundaries.update(
            boundary
            for segment in segments
            for boundary in (
                int(getattr(segment, "start_ms")),
                int(getattr(segment, "end_ms")),
            )
        )
        boundaries.update(
            boundary
            for segment in segments
            for interval in (_exact_overlap_intervals(segment) or ())
            for boundary in interval
        )
        boundaries.update(
            boundary
            for segment in segments
            for turn in (_canonical_speaker_turns(segment) or ())
            for boundary in turn[:2]
        )
    ordered = sorted(boundaries)
    intervals: list[tuple[int, int, set[str], set[str], bool]] = []
    reference_speakers: set[str] = set()
    predicted_speakers: set[str] = set()
    overlap_by_interval: list[bool] = []
    for start_ms, end_ms in zip(ordered, ordered[1:]):
        if end_ms <= start_ms:
            continue
        reference = _active_reference(reference_turns, start_ms, end_ms)
        predicted = (
            _active_timeline_prediction(
                timeline_turns,
                start_ms,
                end_ms,
            )
            if timeline_turns is not None
            else _active_prediction(segments, start_ms, end_ms)
        )
        predicted_overlap = (
            len(predicted) > 1
            if timeline_turns is not None
            else _predicted_overlap(
                segments,
                start_ms,
                end_ms,
                predicted,
            )
        )
        intervals.append(
            (start_ms, end_ms, reference, predicted, predicted_overlap)
        )
        reference_speakers.update(reference)
        predicted_speakers.update(predicted)
        overlap_by_interval.append(predicted_overlap)

    reference_ids = sorted(reference_speakers)
    predicted_ids = sorted(predicted_speakers)
    ref_index = {speaker: index for index, speaker in enumerate(reference_ids)}
    pred_index = {speaker: index for index, speaker in enumerate(predicted_ids)}
    weights = [
        [0.0 for _ in predicted_ids]
        for _ in reference_ids
    ]
    for start_ms, end_ms, reference, predicted, _ in intervals:
        duration = float(end_ms - start_ms)
        for reference_id in reference:
            for predicted_id in predicted:
                weights[ref_index[reference_id]][pred_index[predicted_id]] += duration
    assignment = maximum_weight_assignment(weights)
    predicted_to_reference = {
        predicted_ids[predicted_index]: reference_ids[reference_index]
        for reference_index, predicted_index in assignment
        if weights[reference_index][predicted_index] > 0.0
    }

    reference_time = 0.0
    confusion = 0.0
    misses = 0.0
    false_alarm = 0.0
    overlap_true_positive = 0.0
    overlap_false_positive = 0.0
    overlap_false_negative = 0.0
    intersections = defaultdict(float)
    unions = defaultdict(float)
    for start_ms, end_ms, reference, predicted, predicted_overlap in intervals:
        duration = float(end_ms - start_ms)
        mapped_prediction = {
            predicted_to_reference.get(speaker_id, f"__unmapped__:{speaker_id}")
            for speaker_id in predicted
        }
        correct = len(reference & mapped_prediction)
        reference_time += duration * len(reference)
        confusion += duration * (min(len(reference), len(predicted)) - correct)
        misses += duration * max(0, len(reference) - len(predicted))
        false_alarm += duration * max(0, len(predicted) - len(reference))
        for speaker_id in reference_ids:
            reference_active = speaker_id in reference
            predicted_active = speaker_id in mapped_prediction
            if reference_active and predicted_active:
                intersections[speaker_id] += duration
            if reference_active or predicted_active:
                unions[speaker_id] += duration
        reference_overlap = len(reference) > 1
        if reference_overlap and predicted_overlap:
            overlap_true_positive += duration
        elif predicted_overlap:
            overlap_false_positive += duration
        elif reference_overlap:
            overlap_false_negative += duration

    if reference_time <= 0.0:
        raise WorkerError(
            "REFERENCE_LABELS_INVALID",
            "reference labels contain no scored speaker time",
        )
    jer_components = [
        1.0 - intersections[speaker_id] / unions[speaker_id]
        for speaker_id in reference_ids
        if unions[speaker_id] > 0.0
    ]
    precision_denominator = overlap_true_positive + overlap_false_positive
    recall_denominator = overlap_true_positive + overlap_false_negative
    precision = (
        overlap_true_positive / precision_denominator
        if precision_denominator
        else 1.0
    )
    recall = (
        overlap_true_positive / recall_denominator
        if recall_denominator
        else 1.0
    )
    overlap_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "der": (misses + false_alarm + confusion) / reference_time,
        "jer": statistics.fmean(jer_components) if jer_components else 0.0,
        "speakerConfusion": confusion / reference_time,
        "overlapF1": overlap_f1,
    }
