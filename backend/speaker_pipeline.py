"""Offline-first, dynamic-cardinality speaker pipeline orchestration.

The implementation is intentionally standard-library only.  Heavy model
loading belongs behind injected adapters, which keeps orchestration testable,
prevents implicit downloads, and lets deployments choose CAM++, Qwen3-ASR,
pyannote, or equivalent local engines without changing pipeline invariants.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .adapters import AdapterContext
from .asr_evidence import (
    ASR_CANDIDATE_SET_KEYS,
    AsrEvidenceError,
    build_asr_candidate_set,
    project_asr_candidate_set,
    validate_asr_candidate_set,
)
from .errors import WorkerError
from .language import reconcile_detected_languages
from .models import (
    Revision,
    SpeakerCountEstimate,
    SpeakerCountMode,
    SpeakerScore,
    StartJobRequest,
    TranscriptSegment,
    TranscriptionResult,
)
from .pipeline_metrics import (
    PipelineMetricsCollector,
    ReferenceTurn,
    evaluate_reference_quality,
    maximum_weight_assignment,
)
from .speaker_sequence_decoder import (
    CARDINALITY_CHANGE_REVIEW_REQUIRED,
    SequenceDecoderConfig,
    SequenceSegment,
    SpeakerEmission,
    decode_speaker_sequence,
)
from .speaker_timeline import build_speaker_timeline
from .voice_activity import (
    build_voice_activity,
    with_voice_activity_classification,
)


_PREPARATION_STAGES = ("decode", "normalize", "vad", "boundary")
_REVIEW_REASONS = (
    "TEMPORAL_DISCONTINUITY",
    "SHORT_SEGMENT",
    "LOW_MARGIN",
    "BOUNDARY_CONFLICT",
    "OUTLIER",
    "COUNT_UNCERTAINTY",
    "OVERLAP",
    "OVERLAP_DETECTOR_UNAVAILABLE",
    "SPEAKER_CHANGE_REFINEMENT_REVIEW_REQUIRED",
)
_ERES_RESOLVED_EXIT_REASONS = {
    "VERIFIED_NO_CHANGE",
    "VERIFIED_SPEAKER_CHANGE",
}
_OVERLAP_DETECTOR_UNAVAILABLE_REASON = "OVERLAP_DETECTOR_UNAVAILABLE"
_SPEAKER_CHANGE_REFINEMENT_STAGE = "speaker-change-refinement"
_SPEAKER_COUNT_PARTITION_STAGE = "speaker-count-partition"
_MIN_SPEAKER_COUNT_PARTITION_MS = 700
_AUTO_SPEAKER_EVIDENCE_WINDOW_MS = 1_000
_SPEAKER_CHANGE_REFINEMENT_REVIEW_REASON = (
    "SPEAKER_CHANGE_REFINEMENT_REVIEW_REQUIRED"
)
_SECONDARY_REVIEW_EXCLUSION_REASONS = frozenset(
    {
        _OVERLAP_DETECTOR_UNAVAILABLE_REASON,
        _SPEAKER_CHANGE_REFINEMENT_REVIEW_REASON,
    }
)
_CLUSTER_SELECTION_METHOD = "dynamic-n-adaptive-resample-stability-v12"
_STABILITY_MASK_ALGORITHM = "sha256-ranked-retained-mask-v1"
_PARTITION_DEGENERACY_MIN_STABILITY = 0.70
_PARTITION_DEGENERACY_MIN_BOOTSTRAP_SUPPORT = 0.80
_PARTITION_DEGENERACY_MIN_UNIQUE_RESAMPLE_RUNS = 2
_REQUIRED_STABILITY_COMPONENTS = frozenset(
    {
        "adjustedRand",
        "pairwiseJaccard",
        "coassociationAgreement",
        "alignedAccuracy",
        "coverage",
    }
)
_SEARCH_TRUNCATION_REASONS = frozenset(
    {
        "resource-limit",
        "adaptive-budget-unresolved-local-bracket",
    }
)
_SPEAKER_COUNT_ESTIMATE_METHOD = "constrained-spherical-multik-v6"
_PYANNOTE_COUNT_PRIOR_OBJECTIVE_TOLERANCE = 0.12
_PYANNOTE_COUNT_PRIOR_MIN_STABILITY = 0.60
_PYANNOTE_COUNT_PRIOR_MIN_BOOTSTRAP_SUPPORT = 0.60
_ASR_NON_LEXICAL_DISPOSITION = "rejected-non-lexical"


def _normalized_overlap_evidence(
    *,
    overlapping: bool,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(evidence)
    detector_status = str(normalized.get("detectorStatus") or "").upper()
    unavailable = (
        detector_status == "UNAVAILABLE"
        or normalized.get("overlapDetectorRun") is False
        or normalized.get("reasonCode")
        == _OVERLAP_DETECTOR_UNAVAILABLE_REASON
    )
    if unavailable:
        normalized.update(
            {
                "detectorStatus": "UNAVAILABLE",
                "overlapDetectorRun": False,
                "reviewStatus": "REVIEW_REQUIRED",
                "reasonCode": _OVERLAP_DETECTOR_UNAVAILABLE_REASON,
            }
        )
        return normalized
    normalized.setdefault("detectorStatus", "EVALUATED")
    normalized.setdefault("overlapDetectorRun", True)
    normalized.setdefault(
        "reviewStatus",
        "REVIEW_REQUIRED" if overlapping else "NOT_REQUIRED",
    )
    return normalized


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, context: AdapterContext | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if context is not None:
                context.raise_if_cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_identity(adapter: Any) -> dict[str, str]:
    return {
        "id": str(getattr(adapter, "adapter_id", type(adapter).__name__)),
        "version": str(getattr(adapter, "version", "0")),
    }


def _release_adapter_resources(
    adapter: Any,
    *,
    suppress_errors: bool = False,
) -> None:
    """Release optional heavyweight adapter state at a stage boundary.

    Production adapters use this seam to keep only one large accelerator
    model resident at a time.  Lightweight and injected test adapters remain
    compatible because the capability is optional.
    """

    release = getattr(adapter, "release_resources", None)
    if release is None:
        return
    if not callable(release):
        error = WorkerError(
            "PIPELINE_ADAPTER_RESOURCE_RELEASE_INVALID",
            "adapter resource release capability must be callable",
            details={"adapter": _adapter_identity(adapter)},
        )
        if suppress_errors:
            return
        raise error
    try:
        release()
    except Exception as exc:
        if suppress_errors:
            return
        if isinstance(exc, WorkerError):
            raise
        raise WorkerError(
            "PIPELINE_ADAPTER_RESOURCE_RELEASE_FAILED",
            "adapter resources could not be released at the stage boundary",
            details={
                "adapter": _adapter_identity(adapter),
                "exceptionType": type(exc).__name__,
            },
        ) from exc


def _cache_identity_value(value: Any) -> Any:
    """Return a deterministic JSON-safe representation for adapter identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cache identity numbers must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _cache_identity_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _cache_identity_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_cache_identity_value(item) for item in value]
        return sorted(normalized, key=_stable_json)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_cache_identity_value(item) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _cache_identity_value(as_dict())
    raise ValueError(
        f"unsupported cache identity value: {type(value).__name__}"
    )


def _invoke_refinement_identity(
    provider: Callable[..., Any],
    prepared: "PreparedAudio",
) -> Any:
    """Call an optional identity/config provider with its supported arity."""

    try:
        signature = inspect.signature(provider)
    except (TypeError, ValueError):
        try:
            return provider(prepared)
        except TypeError:
            return provider()
    try:
        signature.bind(prepared)
    except TypeError:
        try:
            signature.bind()
        except TypeError as exc:
            raise ValueError(
                "refinement identity provider must accept zero arguments "
                "or the prepared audio"
            ) from exc
        return provider()
    return provider(prepared)


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{field_name} must be a finite number",
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{field_name} must be a finite number",
        ) from exc
    if not math.isfinite(result):
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{field_name} must be a finite number",
        )
    return result


def _probability(value: Any, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result < 0.0 or result > 1.0:
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{field_name} must be between 0 and 1",
        )
    return result


def _non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{field_name} must be non-empty text",
        )
    return value.strip()


@dataclass(frozen=True)
class SpeechWindow:
    window_id: str
    start_ms: int
    end_ms: int
    boundary_conflict: bool = False
    locked_speaker_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_id, str)
            or not self.window_id.strip()
            or isinstance(self.start_ms, bool)
            or isinstance(self.end_ms, bool)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValueError("speech window is invalid")
        if self.locked_speaker_id is not None and not _speaker_number(
            self.locked_speaker_id
        ):
            raise ValueError("locked_speaker_id must be speaker-N")

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.window_id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "boundaryConflict": self.boundary_conflict,
            "metadata": dict(self.metadata),
        }
        if self.locked_speaker_id:
            value["lockedSpeakerId"] = self.locked_speaker_id
        return value

    @classmethod
    def from_mapping(cls, value: Any) -> "SpeechWindow":
        if not isinstance(value, Mapping):
            raise ValueError("speech window cache entry must be an object")
        return cls(
            window_id=str(value.get("id") or ""),
            start_ms=int(value.get("startMs")),
            end_ms=int(value.get("endMs")),
            boundary_conflict=bool(value.get("boundaryConflict", False)),
            locked_speaker_id=(
                str(value["lockedSpeakerId"])
                if value.get("lockedSpeakerId") is not None
                else None
            ),
            metadata=(
                dict(value.get("metadata", {}))
                if isinstance(value.get("metadata", {}), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class SpeakerIdentityWindow:
    """Reusable contextual voiceprint evidence independent of output turns."""

    window_id: str
    source_window_id: str
    start_ms: int
    end_ms: int
    vector: tuple[float, ...]
    confidence: float = 1.0
    resolution: str = "context"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_id, str)
            or not self.window_id.strip()
            or not isinstance(self.source_window_id, str)
            or not self.source_window_id.strip()
            or isinstance(self.start_ms, bool)
            or isinstance(self.end_ms, bool)
            or self.start_ms < 0
            or self.end_ms <= self.start_ms
        ):
            raise ValueError("speaker identity window is invalid")
        normalized = tuple(
            _finite_float(value, "speakerIdentityWindow.vector")
            for value in self.vector
        )
        if not normalized or math.sqrt(
            sum(value * value for value in normalized)
        ) <= 1e-12:
            raise ValueError(
                "speaker identity window vector must be non-zero"
            )
        object.__setattr__(self, "vector", normalized)
        object.__setattr__(
            self,
            "confidence",
            _probability(
                self.confidence,
                "speakerIdentityWindow.confidence",
            ),
        )
        if self.resolution not in {"fine", "context"}:
            raise ValueError(
                "speaker identity window resolution must be fine or context"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.window_id,
            "sourceWindowId": self.source_window_id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "vector": list(self.vector),
            "confidence": self.confidence,
            "resolution": self.resolution,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "SpeakerIdentityWindow":
        if not isinstance(value, Mapping):
            raise ValueError(
                "speaker identity window cache entry must be an object"
            )
        raw_vector = value.get("vector")
        if not isinstance(raw_vector, list):
            raise ValueError(
                "speaker identity window vector must be an array"
            )
        return cls(
            window_id=str(value.get("id") or ""),
            source_window_id=str(value.get("sourceWindowId") or ""),
            start_ms=int(value.get("startMs")),
            end_ms=int(value.get("endMs")),
            vector=tuple(raw_vector),
            confidence=value.get("confidence", 1.0),
            resolution=str(value.get("resolution") or ""),
        )


@dataclass(frozen=True)
class PreparedAudio:
    duration_ms: int
    source_fingerprint: str
    normalization_profile: str
    windows: tuple[SpeechWindow, ...]
    stage_durations_ms: Mapping[str, float]
    reference_turns: tuple[ReferenceTurn, ...] = ()
    audio_path: str | None = None
    speaker_identity_windows: tuple[SpeakerIdentityWindow, ...] = ()

    def __post_init__(self) -> None:
        if self.duration_ms < 1 or not self.windows:
            raise ValueError("prepared audio must contain bounded speech windows")
        if any(window.end_ms > self.duration_ms for window in self.windows):
            raise ValueError("speech window exceeds prepared audio duration")
        if len({window.window_id for window in self.windows}) != len(self.windows):
            raise ValueError("speech window IDs must be unique")
        previous = -1
        for window in self.windows:
            if window.start_ms < previous:
                raise ValueError("speech windows must be sorted")
            previous = window.start_ms
        for stage in _PREPARATION_STAGES:
            if stage not in self.stage_durations_ms:
                raise ValueError(f"preparation timing is missing {stage}")
            value = float(self.stage_durations_ms[stage])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("preparation timings must be finite")
        if self.audio_path is not None:
            if not isinstance(self.audio_path, str) or not self.audio_path.strip():
                raise ValueError("prepared audio_path must be non-empty text")
        previous_identity = -1
        seen_identity_ids: set[str] = set()
        for identity_window in self.speaker_identity_windows:
            if (
                identity_window.window_id in seen_identity_ids
                or identity_window.start_ms < previous_identity
                or identity_window.end_ms > self.duration_ms
            ):
                raise ValueError(
                    "speaker identity windows must be unique, sorted, and bounded"
                )
            seen_identity_ids.add(identity_window.window_id)
            previous_identity = identity_window.start_ms

    def as_dict(self) -> dict[str, Any]:
        value = {
            "durationMs": self.duration_ms,
            "sourceFingerprint": self.source_fingerprint,
            "normalizationProfile": self.normalization_profile,
            "windows": [window.as_dict() for window in self.windows],
            "stageDurationsMs": {
                key: float(value)
                for key, value in self.stage_durations_ms.items()
            },
            "referenceTurns": [
                {
                    "startMs": turn.start_ms,
                    "endMs": turn.end_ms,
                    "speakerIds": list(turn.speaker_ids),
                }
                for turn in self.reference_turns
            ],
            "speakerIdentityWindows": [
                window.as_dict()
                for window in self.speaker_identity_windows
            ],
        }
        if self.audio_path is not None:
            value["audioPath"] = self.audio_path
        return value

    @classmethod
    def from_mapping(cls, value: Any) -> "PreparedAudio":
        if not isinstance(value, Mapping):
            raise ValueError("prepared audio cache entry must be an object")
        raw_windows = value.get("windows")
        raw_timings = value.get("stageDurationsMs")
        raw_reference = value.get("referenceTurns", [])
        raw_identity_windows = value.get("speakerIdentityWindows", [])
        if (
            not isinstance(raw_windows, list)
            or not isinstance(raw_timings, Mapping)
            or not isinstance(raw_reference, list)
            or not isinstance(raw_identity_windows, list)
        ):
            raise ValueError("prepared audio cache entry is malformed")
        return cls(
            duration_ms=int(value.get("durationMs")),
            source_fingerprint=str(value.get("sourceFingerprint") or ""),
            normalization_profile=str(value.get("normalizationProfile") or ""),
            windows=tuple(SpeechWindow.from_mapping(item) for item in raw_windows),
            stage_durations_ms={
                str(key): float(item) for key, item in raw_timings.items()
            },
            reference_turns=tuple(
                ReferenceTurn(
                    start_ms=int(item["startMs"]),
                    end_ms=int(item["endMs"]),
                    speaker_ids=tuple(str(speaker) for speaker in item["speakerIds"]),
                )
                for item in raw_reference
                if isinstance(item, Mapping)
            ),
            audio_path=(
                str(value["audioPath"]).strip()
                if value.get("audioPath") is not None
                else None
            ),
            speaker_identity_windows=tuple(
                SpeakerIdentityWindow.from_mapping(item)
                for item in raw_identity_windows
            ),
        )


@dataclass(frozen=True)
class AsrHypothesis:
    window_id: str
    text: str
    confidence: float
    normalized_text: str | None = None
    display_text: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "windowId": self.window_id,
            "text": self.text,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }
        if self.normalized_text is not None:
            value["normalizedText"] = self.normalized_text
        if self.display_text is not None:
            value["displayText"] = self.display_text
        return value

    @classmethod
    def from_mapping(cls, value: Any) -> "AsrHypothesis":
        if not isinstance(value, Mapping):
            raise ValueError("ASR result must be an object")
        evidence = value.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise ValueError("ASR evidence must be an object")
        raw_text = value.get("text")
        if not isinstance(raw_text, str):
            raise ValueError("asr.text must be text")
        text = raw_text.strip()
        non_lexical_rejection = (
            evidence.get("disposition") == _ASR_NON_LEXICAL_DISPOSITION
        )
        if not text and not non_lexical_rejection:
            raise ValueError("asr.text must be non-empty")
        if non_lexical_rejection and (
            text
            or value.get("normalizedText") is not None
            or value.get("displayText") is not None
        ):
            raise ValueError(
                "non-lexical ASR rejection must not invent transcript text"
            )
        if "candidateSetSchemaVersion" in evidence:
            try:
                validate_asr_candidate_set(
                    evidence,
                    expected_text=text,
                )
            except AsrEvidenceError as exc:
                raise ValueError(
                    "ASR candidate evidence is not immutable or traceable"
                ) from exc
        return cls(
            window_id=_non_empty_text(value.get("windowId"), "asr.windowId"),
            text=text,
            confidence=_probability(value.get("confidence"), "asr.confidence"),
            normalized_text=(
                _non_empty_text(value.get("normalizedText"), "asr.normalizedText")
                if value.get("normalizedText") is not None
                else None
            ),
            display_text=(
                _non_empty_text(value.get("displayText"), "asr.displayText")
                if value.get("displayText") is not None
                else None
            ),
            evidence=dict(evidence),
        )


@dataclass(frozen=True)
class EmbeddingRecord:
    window_id: str
    vector: tuple[float, ...]
    confidence: float = 1.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "windowId": self.window_id,
            "vector": list(self.vector),
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "EmbeddingRecord":
        if not isinstance(value, Mapping):
            raise ValueError("embedding result must be an object")
        raw_vector = value.get("vector")
        evidence = value.get("evidence", {})
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError("embedding vector must be non-empty")
        if not isinstance(evidence, Mapping):
            raise ValueError("embedding evidence must be an object")
        vector = tuple(_finite_float(item, "embedding.vector") for item in raw_vector)
        if math.sqrt(sum(item * item for item in vector)) <= 1e-12:
            raise ValueError("embedding vector must have non-zero norm")
        return cls(
            window_id=_non_empty_text(
                value.get("windowId"), "embedding.windowId"
            ),
            vector=vector,
            confidence=_probability(
                value.get("confidence", 1.0), "embedding.confidence"
            ),
            evidence=dict(evidence),
        )


@dataclass(frozen=True)
class OverlapDecision:
    window_id: str
    overlapping: bool
    confidence: float = 1.0
    secondary_speaker_hint: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _normalized_overlap_evidence(
            overlapping=self.overlapping,
            evidence=self.evidence,
        )
        object.__setattr__(self, "evidence", normalized)
        if normalized["detectorStatus"] == "UNAVAILABLE":
            object.__setattr__(self, "overlapping", False)
            object.__setattr__(self, "confidence", 0.0)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "windowId": self.window_id,
            "overlapping": self.overlapping,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }
        if self.secondary_speaker_hint:
            value["secondarySpeakerHint"] = self.secondary_speaker_hint
        return value

    @classmethod
    def from_mapping(cls, value: Any) -> "OverlapDecision":
        if not isinstance(value, Mapping):
            raise ValueError("overlap result must be an object")
        overlapping = value.get("overlapping")
        evidence = value.get("evidence", {})
        if not isinstance(overlapping, bool) or not isinstance(evidence, Mapping):
            raise ValueError("overlap result is malformed")
        return cls(
            window_id=_non_empty_text(value.get("windowId"), "overlap.windowId"),
            overlapping=overlapping,
            confidence=_probability(
                value.get("confidence", 1.0), "overlap.confidence"
            ),
            secondary_speaker_hint=(
                str(value["secondarySpeakerHint"])
                if value.get("secondarySpeakerHint") is not None
                else None
            ),
            evidence=dict(evidence),
        )


@dataclass(frozen=True)
class OverlapRecoveryInterval:
    interval_id: str
    detected_start_ms: int
    detected_end_ms: int
    context_start_ms: int
    context_end_ms: int
    local_speakers: tuple[str, str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.interval_id, str)
            or not self.interval_id.strip()
            or isinstance(self.detected_start_ms, bool)
            or isinstance(self.detected_end_ms, bool)
            or isinstance(self.context_start_ms, bool)
            or isinstance(self.context_end_ms, bool)
            or self.context_start_ms < 0
            or self.detected_start_ms < self.context_start_ms
            or self.detected_end_ms <= self.detected_start_ms
            or self.context_end_ms < self.detected_end_ms
            or len(self.local_speakers) != 2
            or len(set(self.local_speakers)) != 2
            or any(not str(item).strip() for item in self.local_speakers)
        ):
            raise ValueError("overlap recovery interval is invalid")


@dataclass(frozen=True)
class SeparatedSpeechChannel:
    candidate_id: str
    interval_id: str
    channel_index: int
    audio_path: str
    audio_sha256: str
    duration_ms: int
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id.strip()
            or not isinstance(self.interval_id, str)
            or not self.interval_id.strip()
            or isinstance(self.channel_index, bool)
            or self.channel_index not in {1, 2}
            or not isinstance(self.audio_path, str)
            or not self.audio_path.strip()
            or not re.fullmatch(r"[0-9a-f]{64}", self.audio_sha256)
            or isinstance(self.duration_ms, bool)
            or self.duration_ms < 1
            or not isinstance(self.evidence, Mapping)
        ):
            raise ValueError("separated speech channel is invalid")


@dataclass(frozen=True)
class ReviewCandidate:
    segment_id: str
    reasons: tuple[str, ...]
    protected: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "segmentId": self.segment_id,
            "reasons": list(self.reasons),
            "protected": self.protected,
        }


@dataclass(frozen=True)
class ReviewProposal:
    segment_id: str
    source: str
    speaker_id: str | None = None
    normalized_text: str | None = None
    display_text: str | None = None
    overlapping: bool | None = None
    reason_code: str = "SELECTIVE_REVIEW"
    evidence_refs: tuple[str, ...] = ()
    confidence: float = 0.0
    exit_reason: str = "VERIFIED_NO_CHANGE"
    resource: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "segmentId": self.segment_id,
            "source": self.source,
            "reasonCode": self.reason_code,
            "evidenceRefs": list(self.evidence_refs),
            "confidence": self.confidence,
            "exitReason": self.exit_reason,
            "resource": dict(self.resource),
        }
        if self.speaker_id is not None:
            value["speakerId"] = self.speaker_id
        if self.normalized_text is not None:
            value["normalizedText"] = self.normalized_text
        if self.display_text is not None:
            value["displayText"] = self.display_text
        if self.overlapping is not None:
            value["overlapping"] = self.overlapping
        return value

    @classmethod
    def from_mapping(cls, value: Any) -> "ReviewProposal":
        if not isinstance(value, Mapping):
            raise ValueError("review proposal must be an object")
        source = str(value.get("source") or "").strip()
        if source not in {"acoustic", "deterministic", "llm", "manual"}:
            raise ValueError("review proposal source is unsupported")
        refs = value.get("evidenceRefs", [])
        if not isinstance(refs, list) or any(
            not isinstance(item, str) or not item.strip() for item in refs
        ):
            raise ValueError("review evidenceRefs must contain strings")
        overlapping = value.get("overlapping")
        if overlapping is not None and not isinstance(overlapping, bool):
            raise ValueError("review overlapping must be a boolean")
        resource = value.get("resource", {})
        if not isinstance(resource, Mapping):
            raise ValueError("review resource must be an object")
        normalized_resource: dict[str, float] = {}
        for key, raw in resource.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("review resource keys must be non-empty strings")
            number = _finite_float(raw, f"review.resource.{key}")
            if number < 0.0:
                raise ValueError("review resource values must not be negative")
            normalized_resource[key] = number
        return cls(
            segment_id=_non_empty_text(
                value.get("segmentId"), "review.segmentId"
            ),
            source=source,
            speaker_id=(
                str(value["speakerId"]).strip()
                if value.get("speakerId") is not None
                else None
            ),
            normalized_text=(
                _non_empty_text(
                    value.get("normalizedText"), "review.normalizedText"
                )
                if value.get("normalizedText") is not None
                else None
            ),
            display_text=(
                _non_empty_text(value.get("displayText"), "review.displayText")
                if value.get("displayText") is not None
                else None
            ),
            overlapping=overlapping,
            reason_code=_non_empty_text(
                value.get("reasonCode", "SELECTIVE_REVIEW"),
                "review.reasonCode",
            ),
            evidence_refs=tuple(item.strip() for item in refs),
            confidence=_probability(
                value.get("confidence", 0.0), "review.confidence"
            ),
            exit_reason=_non_empty_text(
                value.get("exitReason", "VERIFIED_NO_CHANGE"),
                "review.exitReason",
            ),
            resource=normalized_resource,
        )


@runtime_checkable
class AudioPreparationAdapter(Protocol):
    adapter_id: str
    version: str

    def prepare(
        self,
        source_path: Path,
        *,
        normalization_profile: str,
        context: AdapterContext,
    ) -> PreparedAudio | Mapping[str, Any]:
        """Decode, normalize, run VAD, and create boundaries exactly once."""


@runtime_checkable
class BatchAsrAdapter(Protocol):
    adapter_id: str
    version: str

    def transcribe_batch(
        self,
        prepared: PreparedAudio,
        windows: Sequence[SpeechWindow],
        context: AdapterContext,
        *,
        requested_language: str,
        max_generated_tokens: int | None = None,
    ) -> Sequence[AsrHypothesis | Mapping[str, Any]]:
        """Transcribe all cache misses in one model batch."""


@runtime_checkable
class BatchEmbeddingAdapter(Protocol):
    adapter_id: str
    version: str

    def embed_batch(
        self,
        prepared: PreparedAudio,
        windows: Sequence[SpeechWindow],
        context: AdapterContext,
    ) -> Sequence[EmbeddingRecord | Mapping[str, Any]]:
        """Embed all cache misses in one model batch."""


@runtime_checkable
class OverlapDetectionAdapter(Protocol):
    adapter_id: str
    version: str

    def detect_batch(
        self,
        prepared: PreparedAudio,
        windows: Sequence[SpeechWindow],
        context: AdapterContext,
        *,
        speaker_count_constraints: Mapping[str, int] | None = None,
    ) -> Sequence[OverlapDecision | Mapping[str, Any]]:
        """Detect overlap only for cache misses."""


@runtime_checkable
class SpeechSeparationAdapter(Protocol):
    adapter_id: str
    version: str

    def separate_batch(
        self,
        prepared: PreparedAudio,
        intervals: Sequence[OverlapRecoveryInterval],
        context: AdapterContext,
    ) -> Sequence[SeparatedSpeechChannel]:
        """Separate bounded two-speaker overlap intervals without ASR."""


@runtime_checkable
class EscalationAdapter(Protocol):
    adapter_id: str
    version: str

    def review_batch(
        self,
        candidates: Sequence[ReviewCandidate],
        segments: Mapping[str, TranscriptSegment],
        context: AdapterContext,
    ) -> Sequence[ReviewProposal | Mapping[str, Any]]:
        """Review selected difficult segments; never receives easy segments."""


class _InjectedRunner:
    def __init__(
        self,
        runner: Callable[..., Any],
        *,
        adapter_id: str,
        version: str,
    ) -> None:
        if not callable(runner):
            raise ValueError("an explicit local runner is required")
        self.runner = runner
        self.adapter_id = str(adapter_id)
        self.version = str(version)


class Qwen3AsrAdapter(_InjectedRunner):
    """Thin adapter for an explicitly supplied local Qwen3-ASR runner."""

    def transcribe_batch(
        self,
        prepared,
        windows,
        context,
        *,
        requested_language,
    ):
        context.raise_if_cancelled()
        return self.runner(
            prepared,
            tuple(windows),
            context,
            requested_language=requested_language,
        )


class CamPlusEmbeddingAdapter(_InjectedRunner):
    """Thin adapter for an explicitly supplied local CAM++ runner."""

    def embed_batch(self, prepared, windows, context):
        context.raise_if_cancelled()
        return self.runner(prepared, tuple(windows), context)


class ERes2NetV2ReviewAdapter(_InjectedRunner):
    """Selective secondary verifier; it must never receive the full corpus."""

    def review_batch(self, candidates, segments, context):
        context.raise_if_cancelled()
        return self.runner(tuple(candidates), dict(segments), context)


class PyannoteReviewAdapter(_InjectedRunner):
    """Offline pyannote community-1 audit/fallback adapter.

    Telemetry is permanently disabled at this boundary.  The pipeline defaults
    to ``disabled`` and only invokes this adapter when explicitly configured
    for ``audit`` or ``fallback``.
    """

    telemetry_enabled = False

    def review_batch(self, candidates, segments, context):
        context.raise_if_cancelled()
        return self.runner(tuple(candidates), dict(segments), context)


class UnavailableOverlapAdapter:
    """Explicitly report that no overlap detector is configured or executed."""

    adapter_id = "overlap-unavailable"
    version = "2"

    def detect_batch(
        self,
        prepared,
        windows,
        context,
        *,
        speaker_count_constraints=None,
    ):
        context.raise_if_cancelled()
        return [
            OverlapDecision(
                window_id=window.window_id,
                overlapping=False,
                confidence=0.0,
                evidence={
                    "detectorStatus": "UNAVAILABLE",
                    "overlapDetectorRun": False,
                    "reviewStatus": "REVIEW_REQUIRED",
                    "reasonCode": _OVERLAP_DETECTOR_UNAVAILABLE_REASON,
                },
            )
            for window in windows
        ]


class NoOverlapAdapter(UnavailableOverlapAdapter):
    """Compatibility alias that no longer claims overlap was evaluated."""

    adapter_id = "no-overlap"
    version = "2"


@dataclass(frozen=True)
class CacheRead:
    hit: bool
    value: Any = None
    corrupted: bool = False


@runtime_checkable
class StageCache(Protocol):
    def read(self, stage: str, key: str) -> CacheRead:
        ...

    def write(self, stage: str, key: str, value: Any) -> None:
        ...


class InMemoryStageCache:
    """Thread-safe process cache useful for tests and short-lived workers."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], Any] = {}
        self._lock = threading.RLock()

    def read(self, stage: str, key: str) -> CacheRead:
        with self._lock:
            cache_key = (str(stage), str(key))
            if cache_key not in self._values:
                return CacheRead(False)
            return CacheRead(True, self._values[cache_key])

    def write(self, stage: str, key: str, value: Any) -> None:
        with self._lock:
            self._values[(str(stage), str(key))] = value

    def set_raw(self, stage: str, key: str, value: Any) -> None:
        """Testing/repair hook for simulating a semantically corrupt entry."""

        self.write(stage, key, value)

    def keys(self, stage: str | None = None) -> tuple[tuple[str, str], ...]:
        with self._lock:
            return tuple(
                key for key in self._values if stage is None or key[0] == stage
            )


class JsonStageCache:
    """Atomic persistent JSON cache with per-stage isolation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, stage: str, key: str) -> Path:
        safe_stage = "".join(
            character
            for character in str(stage)
            if character.isalnum() or character in {"-", "_"}
        )
        if not safe_stage:
            raise ValueError("cache stage is invalid")
        return self.root / safe_stage / f"{_digest(str(key))}.json"

    def read(self, stage: str, key: str) -> CacheRead:
        path = self._path(stage, key)
        with self._lock:
            if not path.is_file():
                return CacheRead(False)
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return CacheRead(True, json.load(handle))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return CacheRead(False, corrupted=True)

    def write(self, stage: str, key: str, value: Any) -> None:
        path = self._path(stage, key)
        temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
        payload = (_stable_json(value) + "\n").encode("utf-8")
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with temporary.open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


@dataclass(frozen=True)
class SpeakerPipelineConfig:
    normalization_profile: str = "mono-16khz-f32-v1"
    cluster_similarity_threshold: float = 0.72
    low_margin_threshold: float = 0.18
    high_margin_threshold: float = 0.35
    outlier_score_threshold: float = 0.30
    # ``None`` deliberately means "no static cardinality ceiling".  Runtime
    # safety is enforced independently through max_clustering_windows and
    # max_clustering_work_items, so a deployment never silently folds speaker
    # 65+ into an earlier identity.
    max_auto_speakers: int | None = None
    max_clustering_windows: int = 100_000
    max_clustering_work_items: int = 5_000_000
    kmeans_iterations: int = 30
    max_batch_size: int = 32
    max_secondary_fraction: float = 0.25
    max_count_uncertainty_candidates: int = 8
    auto_count_confidence_threshold: float = 0.75
    count_stability_runs: int = 3
    eigengap_landmark_limit: int = 256
    temporal_short_segment_ms: int = 1_500
    temporal_max_gap_ms: int = 750
    pyannote_mapping_margin_threshold: float = 0.05
    pyannote_primary_dominance_threshold: float = 0.60
    pyannote_mode: str = "disabled"
    overlap_recovery_mode: str = "disabled"
    overlap_recovery_margin_threshold: float = 0.18
    overlap_recovery_padding_ms: int = 600
    overlap_recovery_max_intervals: int = 4
    overlap_recovery_max_interval_ms: int = 12_000
    overlap_recovery_asr_max_new_tokens: int = 96
    local_llm_mode: str = "disabled"
    local_llm_model: str = "qwen3.5:9b"
    model_residency: str = "stage"

    def __post_init__(self) -> None:
        if not self.normalization_profile.strip():
            raise ValueError("normalization_profile must not be empty")
        if not -1.0 <= self.cluster_similarity_threshold <= 1.0:
            raise ValueError("cluster_similarity_threshold is invalid")
        if not 0.0 <= self.low_margin_threshold <= self.high_margin_threshold:
            raise ValueError("speaker margin thresholds are invalid")
        if not -1.0 <= self.outlier_score_threshold <= 1.0:
            raise ValueError("outlier_score_threshold is invalid")
        if self.max_auto_speakers is not None and self.max_auto_speakers < 1:
            raise ValueError("max_auto_speakers must be positive when provided")
        if self.max_clustering_windows < 1:
            raise ValueError("max_clustering_windows must be positive")
        if self.max_clustering_work_items < 1:
            raise ValueError("max_clustering_work_items must be positive")
        if self.kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if not 0.0 < self.max_secondary_fraction < 1.0:
            raise ValueError("max_secondary_fraction must be between 0 and 1")
        if self.max_count_uncertainty_candidates < 1:
            raise ValueError("max_count_uncertainty_candidates must be positive")
        if not 0.0 <= self.auto_count_confidence_threshold <= 1.0:
            raise ValueError("auto_count_confidence_threshold is invalid")
        if self.count_stability_runs < 1:
            raise ValueError("count_stability_runs must be positive")
        if self.eigengap_landmark_limit < 2:
            raise ValueError("eigengap_landmark_limit must be at least two")
        if self.temporal_short_segment_ms < 1:
            raise ValueError("temporal_short_segment_ms must be positive")
        if self.temporal_max_gap_ms < 0:
            raise ValueError("temporal_max_gap_ms must not be negative")
        if not 0.0 <= self.pyannote_mapping_margin_threshold <= 1.0:
            raise ValueError("pyannote_mapping_margin_threshold is invalid")
        if not 0.5 <= self.pyannote_primary_dominance_threshold <= 1.0:
            raise ValueError("pyannote_primary_dominance_threshold is invalid")
        if self.pyannote_mode not in {"disabled", "fallback"}:
            raise ValueError("pyannote_mode must be disabled or fallback")
        if self.overlap_recovery_mode not in {"disabled", "guarded"}:
            raise ValueError(
                "overlap_recovery_mode must be disabled or guarded"
            )
        if not 0.0 <= self.overlap_recovery_margin_threshold <= 1.0:
            raise ValueError(
                "overlap_recovery_margin_threshold is invalid"
            )
        if self.overlap_recovery_padding_ms < 0:
            raise ValueError(
                "overlap_recovery_padding_ms must not be negative"
            )
        if self.overlap_recovery_max_intervals < 1:
            raise ValueError(
                "overlap_recovery_max_intervals must be positive"
            )
        if self.overlap_recovery_max_interval_ms < 1:
            raise ValueError(
                "overlap_recovery_max_interval_ms must be positive"
            )
        if not 24 <= self.overlap_recovery_asr_max_new_tokens <= 512:
            raise ValueError(
                "overlap_recovery_asr_max_new_tokens must be between 24 and 512"
            )
        if self.local_llm_mode not in {"disabled", "suggestion-only"}:
            raise ValueError(
                "local_llm_mode must be disabled or suggestion-only; auto_apply is forbidden"
            )
        if self.local_llm_model not in {"qwen3.5:4b", "qwen3.5:9b"}:
            raise ValueError(
                "local_llm_model must identify qwen3.5:9b or qwen3.5:4b"
            )
        if self.model_residency not in {"stage", "worker"}:
            raise ValueError("model_residency must be stage or worker")

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalizationProfile": self.normalization_profile,
            "clusterSimilarityThreshold": self.cluster_similarity_threshold,
            "lowMarginThreshold": self.low_margin_threshold,
            "highMarginThreshold": self.high_margin_threshold,
            "outlierScoreThreshold": self.outlier_score_threshold,
            "maxAutoSpeakers": self.max_auto_speakers,
            "effectiveMaxAutoSpeakers": self.max_auto_speakers,
            "maxClusteringWindows": self.max_clustering_windows,
            "maxClusteringWorkItems": self.max_clustering_work_items,
            "kmeansIterations": self.kmeans_iterations,
            "maxBatchSize": self.max_batch_size,
            "maxSecondaryFraction": self.max_secondary_fraction,
            "maxCountUncertaintyCandidates": self.max_count_uncertainty_candidates,
            "autoCountConfidenceThreshold": self.auto_count_confidence_threshold,
            "countStabilityRuns": self.count_stability_runs,
            "eigengapLandmarkLimit": self.eigengap_landmark_limit,
            "temporalShortSegmentMs": self.temporal_short_segment_ms,
            "temporalMaxGapMs": self.temporal_max_gap_ms,
            "pyannoteMappingMarginThreshold": (
                self.pyannote_mapping_margin_threshold
            ),
            "pyannotePrimaryDominanceThreshold": (
                self.pyannote_primary_dominance_threshold
            ),
            "pyannoteMode": self.pyannote_mode,
            "pyannoteTelemetryEnabled": False,
            "overlapRecoveryMode": self.overlap_recovery_mode,
            "overlapRecoveryMarginThreshold": (
                self.overlap_recovery_margin_threshold
            ),
            "overlapRecoveryPaddingMs": self.overlap_recovery_padding_ms,
            "overlapRecoveryMaxIntervals": (
                self.overlap_recovery_max_intervals
            ),
            "overlapRecoveryMaxIntervalMs": (
                self.overlap_recovery_max_interval_ms
            ),
            "overlapRecoveryAsrMaxNewTokens": (
                self.overlap_recovery_asr_max_new_tokens
            ),
            "localLlmMode": self.local_llm_mode,
            "localLlmModel": self.local_llm_model,
            "localLlmAutoApply": False,
            "modelResidency": self.model_residency,
        }


@dataclass(frozen=True)
class _ClusterCandidateScore:
    count: int
    objective: float
    compactness: float
    separation: float
    approximate_silhouette: float
    calinski_harabasz: float
    calinski_harabasz_utility: float
    davies_bouldin: float
    davies_bouldin_utility: float
    eigengap: float
    eigengap_utility: float
    stability: float
    bootstrap_support: float
    residual_dispersion: float
    singleton_count: int
    singleton_fraction: float
    tiny_cluster_count: int
    tiny_cluster_fraction: float
    minimum_cluster_size: int
    fragmentation: float
    complexity_penalty: float
    under_split_risk: float
    over_split_risk: float
    outlier_risk: float
    imbalance_penalty: float
    consensus_support: float
    metric_votes: int
    weighted_contributions: Mapping[str, float]
    correction_reasons: tuple[str, ...]
    prior_distance: int | None
    work_items: int
    stability_components: Mapping[str, float] = field(default_factory=dict)
    resample_objectives: tuple[float, ...] = ()
    stability_requested_runs: int = 0
    stability_effective_unique_runs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "objective": self.objective,
            "compactness": self.compactness,
            "separation": self.separation,
            "approximateSilhouette": self.approximate_silhouette,
            "silhouette": self.approximate_silhouette,
            "calinskiHarabasz": self.calinski_harabasz,
            "calinskiHarabaszUtility": self.calinski_harabasz_utility,
            "daviesBouldin": self.davies_bouldin,
            "daviesBouldinUtility": self.davies_bouldin_utility,
            "eigengap": self.eigengap,
            "eigengapUtility": self.eigengap_utility,
            "stability": self.stability,
            "bootstrapSupport": self.bootstrap_support,
            "residualDispersion": self.residual_dispersion,
            "singletonCount": self.singleton_count,
            "singletonFraction": self.singleton_fraction,
            "tinyClusterCount": self.tiny_cluster_count,
            "tinyClusterFraction": self.tiny_cluster_fraction,
            "minimumClusterSize": self.minimum_cluster_size,
            "fragmentation": self.fragmentation,
            "complexityPenalty": self.complexity_penalty,
            "underSplitRisk": self.under_split_risk,
            "overSplitRisk": self.over_split_risk,
            "outlierRisk": self.outlier_risk,
            "imbalancePenalty": self.imbalance_penalty,
            "consensusSupport": self.consensus_support,
            "metricVotes": self.metric_votes,
            "weightedContributions": dict(self.weighted_contributions),
            "correctionReasons": list(self.correction_reasons),
            "priorDistance": self.prior_distance,
            "workItems": self.work_items,
            "stabilityComponents": dict(self.stability_components),
            "resampleObjectives": list(self.resample_objectives),
            "stabilityRequestedRuns": self.stability_requested_runs,
            "stabilityEffectiveUniqueRuns": (
                self.stability_effective_unique_runs
            ),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "_ClusterCandidateScore":
        if not isinstance(value, Mapping):
            raise ValueError("cluster count candidate must be an object")
        count = int(value.get("count"))
        work_items = int(value.get("workItems"))
        prior_distance_value = value.get("priorDistance")
        prior_distance = (
            None
            if prior_distance_value is None
            else int(prior_distance_value)
        )
        if (
            count < 1
            or work_items < 0
            or (prior_distance is not None and prior_distance < 0)
        ):
            raise ValueError("cluster count candidate is invalid")
        raw_contributions = value.get("weightedContributions", {})
        if not isinstance(raw_contributions, Mapping):
            raise ValueError("cluster candidate weighted contributions are invalid")
        weighted_contributions = {
            str(key): _finite_float(
                item, f"cluster.candidate.weightedContributions.{key}"
            )
            for key, item in raw_contributions.items()
        }
        raw_corrections = value.get("correctionReasons", ())
        if not isinstance(raw_corrections, Sequence) or isinstance(
            raw_corrections, (str, bytes, bytearray)
        ):
            raise ValueError("cluster candidate correction reasons are invalid")
        correction_reasons = tuple(str(item) for item in raw_corrections)
        metric_votes = int(value.get("metricVotes", 0))
        if metric_votes < 0:
            raise ValueError("cluster candidate metric votes are invalid")
        raw_stability_components = value.get("stabilityComponents", {})
        if not isinstance(raw_stability_components, Mapping):
            raise ValueError("cluster candidate stability components are invalid")
        stability_components = {
            str(key): _probability(
                item,
                f"cluster.candidate.stabilityComponents.{key}",
            )
            for key, item in raw_stability_components.items()
        }
        if set(stability_components) != _REQUIRED_STABILITY_COMPONENTS:
            raise ValueError(
                "cluster candidate stability components are incomplete"
            )
        raw_resample_objectives = value.get("resampleObjectives", ())
        if not isinstance(raw_resample_objectives, Sequence) or isinstance(
            raw_resample_objectives,
            (str, bytes, bytearray),
        ):
            raise ValueError("cluster candidate resample objectives are invalid")
        resample_objectives = tuple(
            _finite_float(
                item,
                "cluster.candidate.resampleObjectives",
            )
            for item in raw_resample_objectives
        )
        stability_requested_runs = int(value.get("stabilityRequestedRuns"))
        stability_effective_unique_runs = int(
            value.get("stabilityEffectiveUniqueRuns")
        )
        if (
            stability_requested_runs < 1
            or stability_effective_unique_runs < 1
            or stability_effective_unique_runs > stability_requested_runs
            or len(resample_objectives) != stability_requested_runs
        ):
            raise ValueError("cluster candidate stability run audit is invalid")
        compactness = _probability(
            value.get("compactness"), "cluster.candidate.compactness"
        )
        singleton_fraction = _probability(
            value.get("singletonFraction"),
            "cluster.candidate.singletonFraction",
        )
        tiny_cluster_fraction = _probability(
            value.get("tinyClusterFraction"),
            "cluster.candidate.tinyClusterFraction",
        )
        singleton_count = int(
            value.get(
                "singletonCount",
                round(singleton_fraction * count),
            )
        )
        tiny_cluster_count = int(
            value.get(
                "tinyClusterCount",
                round(tiny_cluster_fraction * count),
            )
        )
        minimum_cluster_size = int(value.get("minimumClusterSize", 0))
        if (
            singleton_count < 0
            or singleton_count > count
            or tiny_cluster_count < singleton_count
            or tiny_cluster_count > count
            or minimum_cluster_size < 0
        ):
            raise ValueError("cluster candidate cardinality audit is invalid")
        return cls(
            count=count,
            objective=_finite_float(
                value.get("objective"), "cluster.candidate.objective"
            ),
            compactness=compactness,
            separation=_probability(
                value.get("separation"), "cluster.candidate.separation"
            ),
            approximate_silhouette=_probability(
                value.get(
                    "approximateSilhouette",
                    value.get("silhouette", 0.0),
                ),
                "cluster.candidate.approximateSilhouette",
            ),
            calinski_harabasz=max(
                0.0,
                _finite_float(
                    value.get("calinskiHarabasz", 0.0),
                    "cluster.candidate.calinskiHarabasz",
                ),
            ),
            calinski_harabasz_utility=_probability(
                value.get("calinskiHarabaszUtility", 0.0),
                "cluster.candidate.calinskiHarabaszUtility",
            ),
            davies_bouldin=max(
                0.0,
                _finite_float(
                    value.get("daviesBouldin", 0.0),
                    "cluster.candidate.daviesBouldin",
                ),
            ),
            davies_bouldin_utility=_probability(
                value.get("daviesBouldinUtility", 0.0),
                "cluster.candidate.daviesBouldinUtility",
            ),
            eigengap=_probability(
                value.get("eigengap", 0.0),
                "cluster.candidate.eigengap",
            ),
            eigengap_utility=_probability(
                value.get("eigengapUtility", value.get("eigengap", 0.0)),
                "cluster.candidate.eigengapUtility",
            ),
            stability=_probability(
                value.get("stability", 0.0),
                "cluster.candidate.stability",
            ),
            bootstrap_support=_probability(
                value.get("bootstrapSupport", 0.0),
                "cluster.candidate.bootstrapSupport",
            ),
            residual_dispersion=max(
                0.0,
                _finite_float(
                    value.get(
                        "residualDispersion",
                        max(0.0, 1.0 - compactness),
                    ),
                    "cluster.candidate.residualDispersion",
                ),
            ),
            singleton_count=singleton_count,
            singleton_fraction=singleton_fraction,
            tiny_cluster_count=tiny_cluster_count,
            tiny_cluster_fraction=tiny_cluster_fraction,
            minimum_cluster_size=minimum_cluster_size,
            fragmentation=_probability(
                value.get("fragmentation"),
                "cluster.candidate.fragmentation",
            ),
            complexity_penalty=max(
                0.0,
                _finite_float(
                    value.get("complexityPenalty"),
                    "cluster.candidate.complexityPenalty",
                ),
            ),
            under_split_risk=_probability(
                value.get("underSplitRisk", 0.0),
                "cluster.candidate.underSplitRisk",
            ),
            over_split_risk=_probability(
                value.get("overSplitRisk", 0.0),
                "cluster.candidate.overSplitRisk",
            ),
            outlier_risk=_probability(
                value.get("outlierRisk", 0.0),
                "cluster.candidate.outlierRisk",
            ),
            imbalance_penalty=max(
                0.0,
                _finite_float(
                    value.get("imbalancePenalty", 0.0),
                    "cluster.candidate.imbalancePenalty",
                ),
            ),
            consensus_support=_probability(
                value.get("consensusSupport", 0.0),
                "cluster.candidate.consensusSupport",
            ),
            metric_votes=metric_votes,
            weighted_contributions=weighted_contributions,
            correction_reasons=correction_reasons,
            prior_distance=prior_distance,
            work_items=work_items,
            stability_components=stability_components,
            resample_objectives=resample_objectives,
            stability_requested_runs=stability_requested_runs,
            stability_effective_unique_runs=stability_effective_unique_runs,
        )


@dataclass(frozen=True)
class _KMeansFit:
    count: int
    assignments: tuple[int, ...]
    scores: tuple[tuple[float, ...], ...]
    centroids: tuple[tuple[float, ...], ...]
    iterations: int
    work_items: int


@dataclass(frozen=True)
class _StabilityProfile:
    stability: float
    coverage: float
    components: Mapping[str, float]
    resample_compactness: tuple[float, ...]
    resample_stability: tuple[float, ...]
    requested_runs: int
    effective_unique_runs: int


@dataclass(frozen=True)
class _EigengapProfile:
    scores: Mapping[int, float]
    estimate: int | None
    method: str
    landmark_count: int
    work_items: int


@dataclass(frozen=True)
class _ClusterResult:
    count: int
    confidence: float
    candidate_min: int
    candidate_max: int
    assignments: tuple[int, ...]
    scores: tuple[tuple[float, ...], ...]
    count_candidates: tuple[_ClusterCandidateScore, ...] = ()
    count_search_truncated: bool = False
    leader_estimate: int | None = None
    spectral_estimate: int | None = None
    eigengap_method: str = "not-evaluated"
    leader_count_work_items: int = 0
    total_work_items: int = 0
    selection_method: str = _CLUSTER_SELECTION_METHOD
    confidence_reasons: tuple[str, ...] = ()
    correction_path: tuple[str, ...] = ()
    under_split_detected: bool = False
    over_split_detected: bool = False
    low_confidence_fail_closed: bool = False
    legal_min: int | None = None
    legal_max: int | None = None
    evaluated_counts: tuple[int, ...] = ()
    planned_counts: tuple[int, ...] = ()
    search_truncation_reason: str | None = None
    resample_objective_winner_frequency: Mapping[int, float] = field(
        default_factory=dict
    )
    resample_objective_winner_support: float = 0.0
    requested_resample_runs: int = 0
    effective_unique_resample_runs: int = 0
    search_exhaustive: bool = False
    decision_locally_bracketed: bool = False
    adaptive_budget_limited: bool = False
    resource_truncated: bool = False
    resource_skipped_counts: tuple[int, ...] = ()
    minimum_required_work_items: int = 0
    resource_affordable_max_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        legal_min = self.candidate_min if self.legal_min is None else self.legal_min
        legal_max = self.candidate_max if self.legal_max is None else self.legal_max
        evaluated_counts = (
            self.evaluated_counts
            or tuple(candidate.count for candidate in self.count_candidates)
            or (self.count,)
        )
        planned_counts = self.planned_counts or evaluated_counts
        return {
            "count": self.count,
            "confidence": self.confidence,
            "candidateMin": self.candidate_min,
            "candidateMax": self.candidate_max,
            "assignments": list(self.assignments),
            "scores": [list(row) for row in self.scores],
            "countCandidates": [
                candidate.as_dict() for candidate in self.count_candidates
            ],
            "countSearchTruncated": self.count_search_truncated,
            "leaderEstimate": self.leader_estimate,
            "spectralEstimate": self.spectral_estimate,
            "eigengapMethod": self.eigengap_method,
            "leaderCountWorkItems": self.leader_count_work_items,
            "totalWorkItems": self.total_work_items,
            "selectionMethod": self.selection_method,
            "confidenceKind": "bounded-decision-score-uncalibrated-v1",
            "confidenceInterval": {
                "min": self.candidate_min,
                "max": self.candidate_max,
            },
            "confidenceIntervalKind": (
                "plausible-count-range-not-calibrated-credible-interval"
            ),
            "legalRange": {
                "min": legal_min,
                "max": legal_max,
            },
            "evaluatedCounts": list(evaluated_counts),
            "evaluatedRange": {
                "min": min(evaluated_counts),
                "max": max(evaluated_counts),
            },
            "plannedCounts": list(planned_counts),
            "searchTruncationReason": self.search_truncation_reason,
            "resampleObjectiveWinnerFrequency": {
                str(count): support
                for count, support in sorted(
                    self.resample_objective_winner_frequency.items()
                )
            },
            "resampleObjectiveWinnerSupport": (
                self.resample_objective_winner_support
            ),
            "requestedResampleRuns": self.requested_resample_runs,
            "effectiveUniqueResampleRuns": (
                self.effective_unique_resample_runs
            ),
            "searchExhaustive": self.search_exhaustive,
            "decisionLocallyBracketed": self.decision_locally_bracketed,
            "adaptiveBudgetLimited": self.adaptive_budget_limited,
            "resourceTruncated": self.resource_truncated,
            "resourceSkippedCounts": list(self.resource_skipped_counts),
            "minimumRequiredWorkItems": self.minimum_required_work_items,
            "resourceAffordableMaxCount": self.resource_affordable_max_count,
            "confidenceReasons": list(self.confidence_reasons),
            "correctionPath": list(self.correction_path),
            "underSplitDetected": self.under_split_detected,
            "overSplitDetected": self.over_split_detected,
            "lowConfidenceFailClosed": self.low_confidence_fail_closed,
        }

    @classmethod
    def from_mapping(cls, value: Any, *, expected_rows: int) -> "_ClusterResult":
        if not isinstance(value, Mapping):
            raise ValueError("cluster cache entry must be an object")
        if value.get("selectionMethod") != _CLUSTER_SELECTION_METHOD:
            raise ValueError("cluster cache selection method is incompatible")

        def required_bool(field_name: str) -> bool:
            parsed = value.get(field_name)
            if not isinstance(parsed, bool):
                raise ValueError(
                    f"cluster cache {field_name} must be a boolean"
                )
            return parsed

        count = int(value.get("count"))
        assignments = tuple(int(item) for item in value.get("assignments", []))
        scores = tuple(
            tuple(_finite_float(item, "cluster.score") for item in row)
            for row in value.get("scores", [])
        )
        if (
            count < 1
            or len(assignments) != expected_rows
            or len(scores) != expected_rows
            or any(len(row) != count for row in scores)
            or any(item < 0 or item >= count for item in assignments)
        ):
            raise ValueError("cluster cache dimensions are invalid")
        candidate_min = int(value.get("candidateMin"))
        candidate_max = int(value.get("candidateMax"))
        if (
            candidate_min < 1
            or candidate_max < candidate_min
            or not candidate_min <= count <= candidate_max
        ):
            raise ValueError("cluster cache candidate range is invalid")
        raw_candidates = value.get("countCandidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(
            raw_candidates, (str, bytes, bytearray)
        ) or not raw_candidates:
            raise ValueError("cluster count candidates must be an array")
        count_candidates = tuple(
            _ClusterCandidateScore.from_mapping(item)
            for item in raw_candidates
        )
        candidate_counts = tuple(item.count for item in count_candidates)
        if (
            len(set(candidate_counts)) != len(candidate_counts)
            or count not in candidate_counts
        ):
            raise ValueError("selected count is absent from count candidates")
        leader_estimate_value = value.get("leaderEstimate")
        leader_estimate = (
            None
            if leader_estimate_value is None
            else int(leader_estimate_value)
        )
        spectral_estimate_value = value.get("spectralEstimate")
        spectral_estimate = (
            None
            if spectral_estimate_value is None
            else int(spectral_estimate_value)
        )
        leader_count_work_items = int(value.get("leaderCountWorkItems", 0))
        total_work_items = int(value.get("totalWorkItems", 0))
        if (
            (leader_estimate is not None and leader_estimate < 1)
            or (spectral_estimate is not None and spectral_estimate < 1)
            or leader_count_work_items < 0
            or total_work_items < leader_count_work_items
        ):
            raise ValueError("cluster work-item audit is invalid")
        raw_confidence_reasons = value.get("confidenceReasons", ())
        raw_correction_path = value.get("correctionPath", ())
        if (
            not isinstance(raw_confidence_reasons, Sequence)
            or isinstance(raw_confidence_reasons, (str, bytes, bytearray))
            or not isinstance(raw_correction_path, Sequence)
            or isinstance(raw_correction_path, (str, bytes, bytearray))
        ):
            raise ValueError("cluster confidence audit is invalid")
        raw_legal_range = value.get("legalRange")
        if not isinstance(raw_legal_range, Mapping):
            raise ValueError("cluster legal range is invalid")
        legal_min = int(raw_legal_range.get("min"))
        legal_max = int(raw_legal_range.get("max"))
        if (
            legal_min < 1
            or legal_max < legal_min
            or not legal_min <= count <= legal_max
            or not legal_min <= candidate_min <= candidate_max <= legal_max
        ):
            raise ValueError("cluster legal range is invalid")
        raw_evaluated_counts = value.get("evaluatedCounts")
        if not isinstance(raw_evaluated_counts, Sequence) or isinstance(
            raw_evaluated_counts,
            (str, bytes, bytearray),
        ):
            raise ValueError("cluster evaluated counts are invalid")
        evaluated_counts = tuple(int(item) for item in raw_evaluated_counts)
        if (
            not evaluated_counts
            or len(set(evaluated_counts)) != len(evaluated_counts)
            or any(item < legal_min or item > legal_max for item in evaluated_counts)
            or count not in evaluated_counts
            or set(evaluated_counts) != set(candidate_counts)
        ):
            raise ValueError("cluster evaluated counts are invalid")
        raw_planned_counts = value.get("plannedCounts")
        if not isinstance(raw_planned_counts, Sequence) or isinstance(
            raw_planned_counts,
            (str, bytes, bytearray),
        ):
            raise ValueError("cluster planned counts are invalid")
        planned_counts = tuple(int(item) for item in raw_planned_counts)
        if (
            len(set(planned_counts)) != len(planned_counts)
            or any(item < legal_min or item > legal_max for item in planned_counts)
            or any(item not in planned_counts for item in evaluated_counts)
        ):
            raise ValueError("cluster planned counts are invalid")
        raw_resample_frequency = value.get(
            "resampleObjectiveWinnerFrequency"
        )
        if not isinstance(raw_resample_frequency, Mapping):
            raise ValueError(
                "cluster resample objective winner frequency is invalid"
            )
        resample_objective_winner_frequency = {
            int(key): _probability(
                item,
                f"cluster.resampleObjectiveWinnerFrequency.{key}",
            )
            for key, item in raw_resample_frequency.items()
        }
        requested_resample_runs = int(value.get("requestedResampleRuns"))
        effective_unique_resample_runs = int(
            value.get("effectiveUniqueResampleRuns")
        )
        candidate_requested_runs = {
            item.stability_requested_runs for item in count_candidates
        }
        candidate_effective_runs = {
            item.stability_effective_unique_runs for item in count_candidates
        }
        if (
            requested_resample_runs < 1
            or effective_unique_resample_runs < 1
            or effective_unique_resample_runs > requested_resample_runs
            or candidate_requested_runs != {requested_resample_runs}
            or effective_unique_resample_runs != min(candidate_effective_runs)
            or any(
                item not in evaluated_counts
                for item in resample_objective_winner_frequency
            )
            or not resample_objective_winner_frequency
            or not math.isclose(
                sum(resample_objective_winner_frequency.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "cluster resample objective winner audit is invalid"
            )
        resample_objective_winner_support = _probability(
            value.get("resampleObjectiveWinnerSupport"),
            "cluster.resampleObjectiveWinnerSupport",
        )
        if not math.isclose(
            resample_objective_winner_support,
            resample_objective_winner_frequency.get(count, 0.0),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "cluster resample objective winner support is inconsistent"
            )

        search_exhaustive = required_bool("searchExhaustive")
        decision_locally_bracketed = required_bool(
            "decisionLocallyBracketed"
        )
        adaptive_budget_limited = required_bool("adaptiveBudgetLimited")
        resource_truncated = required_bool("resourceTruncated")
        count_search_truncated = required_bool("countSearchTruncated")
        under_split_detected = required_bool("underSplitDetected")
        over_split_detected = required_bool("overSplitDetected")
        low_confidence_fail_closed = required_bool(
            "lowConfidenceFailClosed"
        )
        expected_exhaustive = set(evaluated_counts) == set(
            range(legal_min, legal_max + 1)
        )
        expected_local_bracket = (
            (count == legal_min or count - 1 in evaluated_counts)
            and (count == legal_max or count + 1 in evaluated_counts)
        )
        raw_resource_skipped_counts = value.get("resourceSkippedCounts")
        if not isinstance(
            raw_resource_skipped_counts,
            Sequence,
        ) or isinstance(raw_resource_skipped_counts, (str, bytes, bytearray)):
            raise ValueError("cluster resource skipped counts are invalid")
        resource_skipped_counts = tuple(
            int(item) for item in raw_resource_skipped_counts
        )
        required_neighbor_counts = {
            neighbor
            for neighbor in (count - 1, count + 1)
            if legal_min <= neighbor <= legal_max
        }
        expected_resource_truncated = bool(
            required_neighbor_counts.intersection(resource_skipped_counts)
        )
        if (
            len(set(resource_skipped_counts)) != len(resource_skipped_counts)
            or any(
                item < legal_min
                or item > legal_max
                or item in evaluated_counts
                or item not in planned_counts
                for item in resource_skipped_counts
            )
            or resource_truncated != expected_resource_truncated
            or search_exhaustive != expected_exhaustive
            or decision_locally_bracketed != expected_local_bracket
            or count_search_truncated
            != (resource_truncated or adaptive_budget_limited)
            or (
                search_exhaustive
                and (adaptive_budget_limited or resource_truncated)
            )
        ):
            raise ValueError("cluster search audit is inconsistent")
        minimum_required_work_items = int(
            value.get("minimumRequiredWorkItems")
        )
        affordable_value = value.get("resourceAffordableMaxCount")
        resource_affordable_max_count = (
            None if affordable_value is None else int(affordable_value)
        )
        if (
            minimum_required_work_items < 0
            or (
                resource_affordable_max_count is not None
                and not legal_min
                <= resource_affordable_max_count
                <= legal_max
            )
        ):
            raise ValueError("cluster resource affordability audit is invalid")
        search_truncation_reason_value = value.get("searchTruncationReason")
        search_truncation_reason = (
            None
            if search_truncation_reason_value is None
            else str(search_truncation_reason_value)
        )
        if (
            (
                count_search_truncated
                and search_truncation_reason not in _SEARCH_TRUNCATION_REASONS
            )
            or (
                not count_search_truncated
                and search_truncation_reason is not None
            )
            or (
                resource_truncated
                and search_truncation_reason != "resource-limit"
            )
            or (
                adaptive_budget_limited
                and not resource_truncated
                and search_truncation_reason
                != "adaptive-budget-unresolved-local-bracket"
            )
        ):
            raise ValueError("cluster search truncation reason is inconsistent")
        return cls(
            count=count,
            confidence=_probability(
                value.get("confidence"), "cluster.confidence"
            ),
            candidate_min=candidate_min,
            candidate_max=candidate_max,
            assignments=assignments,
            scores=scores,
            count_candidates=count_candidates,
            count_search_truncated=count_search_truncated,
            leader_estimate=leader_estimate,
            spectral_estimate=spectral_estimate,
            eigengap_method=str(
                value.get("eigengapMethod", "not-evaluated")
            ),
            leader_count_work_items=leader_count_work_items,
            total_work_items=total_work_items,
            selection_method=_CLUSTER_SELECTION_METHOD,
            confidence_reasons=tuple(
                str(item) for item in raw_confidence_reasons
            ),
            correction_path=tuple(str(item) for item in raw_correction_path),
            under_split_detected=under_split_detected,
            over_split_detected=over_split_detected,
            low_confidence_fail_closed=low_confidence_fail_closed,
            legal_min=legal_min,
            legal_max=legal_max,
            evaluated_counts=evaluated_counts,
            planned_counts=planned_counts,
            search_truncation_reason=search_truncation_reason,
            resample_objective_winner_frequency=(
                resample_objective_winner_frequency
            ),
            resample_objective_winner_support=(
                resample_objective_winner_support
            ),
            requested_resample_runs=requested_resample_runs,
            effective_unique_resample_runs=effective_unique_resample_runs,
            search_exhaustive=search_exhaustive,
            decision_locally_bracketed=decision_locally_bracketed,
            adaptive_budget_limited=adaptive_budget_limited,
            resource_truncated=resource_truncated,
            resource_skipped_counts=resource_skipped_counts,
            minimum_required_work_items=minimum_required_work_items,
            resource_affordable_max_count=resource_affordable_max_count,
        )


def _speaker_number(value: str | None) -> int | None:
    if not isinstance(value, str) or not value.startswith("speaker-"):
        return None
    suffix = value[8:]
    if not suffix.isdigit() or int(suffix) < 1:
        return None
    return int(suffix)


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 1e-12:
        raise WorkerError(
            "EMBEDDING_INVALID",
            "embedding vectors must have non-zero norm",
        )
    return tuple(float(item) / norm for item in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _mean_vector(
    vectors: Sequence[Sequence[float]],
    dimension: int,
) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("cannot average an empty vector set")
    return _normalize(
        [
            sum(vector[index] for vector in vectors)
            for index in range(dimension)
        ]
    )


def _leader_count(
    vectors: Sequence[tuple[float, ...]],
    *,
    threshold: float,
    maximum: int,
    work_item_limit: int,
) -> tuple[int, int]:
    leaders: list[tuple[float, ...]] = []
    members: list[list[tuple[float, ...]]] = []
    work_items = 0
    for vector in vectors:
        if not leaders:
            leaders.append(vector)
            members.append([vector])
            continue
        pending_work = len(leaders)
        if work_items + pending_work > work_item_limit:
            raise WorkerError(
                "CLUSTERING_RESOURCE_LIMIT_EXCEEDED",
                "automatic speaker counting exceeded the configured work limit",
                details={
                    "phase": "leader-count",
                    "workItems": work_items + pending_work,
                    "maxWorkItems": work_item_limit,
                },
            )
        work_items += pending_work
        similarities = [_dot(vector, leader) for leader in leaders]
        best = max(range(len(leaders)), key=lambda index: similarities[index])
        if similarities[best] < threshold:
            if len(leaders) >= maximum:
                raise WorkerError(
                    "AUTO_SPEAKER_LIMIT_EXCEEDED",
                    "automatic speaker counting requires more speakers than allowed",
                    details={
                        "resolvedAtLeast": len(leaders) + 1,
                        "maxAutoSpeakers": maximum,
                        "similarity": similarities[best],
                        "threshold": threshold,
                    },
                )
            leaders.append(vector)
            members.append([vector])
        else:
            members[best].append(vector)
            leaders[best] = _mean_vector(members[best], len(vector))
    return max(1, len(leaders)), work_items


def _resolve_cluster_count(
    vectors: Sequence[tuple[float, ...]],
    request: StartJobRequest,
    config: SpeakerPipelineConfig,
    *,
    minimum_locked_count: int = 1,
) -> tuple[int, int, int, int]:
    available = len(vectors)
    policy = request.speaker_policy
    effective_maximum = min(
        available,
        (
            config.max_auto_speakers
            if config.max_auto_speakers is not None
            else available
        ),
    )
    if minimum_locked_count > available:
        raise WorkerError(
            "HUMAN_LOCK_OUT_OF_RANGE",
            "human speaker lock exceeds the number of speech windows",
            details={
                "minimumLockedSpeakerCount": minimum_locked_count,
                "speechWindows": available,
            },
        )
    if policy.mode is SpeakerCountMode.MANUAL:
        assert policy.manual_count is not None
        if policy.manual_count > available:
            raise WorkerError(
                "SPEAKER_COUNT_EXCEEDS_SPEECH_WINDOWS",
                "manual speaker count cannot exceed the number of speech windows",
                details={
                    "speakerCount": policy.manual_count,
                    "speechWindows": available,
                },
            )
        if policy.manual_count < minimum_locked_count:
            raise WorkerError(
                "HUMAN_LOCK_OUT_OF_RANGE",
                "human speaker lock is outside the manual speaker count",
                details={
                    "minimumLockedSpeakerCount": minimum_locked_count,
                    "speakerCount": policy.manual_count,
                },
            )
        return (
            policy.manual_count,
            policy.manual_count,
            policy.manual_count,
            0,
        )

    if minimum_locked_count > effective_maximum:
        raise WorkerError(
            "HUMAN_LOCK_OUT_OF_RANGE",
            "human speaker lock is outside the configured automatic range",
            details={
                "minimumLockedSpeakerCount": minimum_locked_count,
                "maxAutoSpeakers": config.max_auto_speakers,
                "speechWindows": available,
            },
        )

    lower = minimum_locked_count
    upper = effective_maximum
    if policy.mode is SpeakerCountMode.HYBRID:
        assert policy.minimum is not None and policy.maximum is not None
        if policy.minimum > available:
            raise WorkerError(
                "SPEAKER_COUNT_EXCEEDS_SPEECH_WINDOWS",
                "hybrid minimum cannot exceed the number of speech windows",
                details={
                    "speakerCountMinimum": policy.minimum,
                    "speechWindows": available,
                },
            )
        lower = max(lower, policy.minimum)
        upper = min(upper, policy.maximum)
    if lower > upper:
        raise WorkerError(
            "SPEAKER_COUNT_RANGE_UNSATISFIABLE",
            "speaker-count constraints do not leave a legal cluster count",
            details={
                "minimum": lower,
                "maximum": upper,
                "speechWindows": available,
            },
        )

    leader_estimate, work_items = _leader_count(
        vectors,
        threshold=config.cluster_similarity_threshold,
        maximum=effective_maximum,
        work_item_limit=config.max_clustering_work_items,
    )
    leader_estimate = max(lower, min(upper, leader_estimate))
    return lower, upper, leader_estimate, work_items


def _candidate_count_order(
    *,
    lower: int,
    upper: int,
    leader_estimate: int,
    prior: int | None,
    maximum_candidates: int,
    spectral_estimate: int | None = None,
) -> tuple[int, ...]:
    limit = min(maximum_candidates, upper - lower + 1)
    if limit < 1:
        return ()
    anchors = [
        value
        for value in (leader_estimate, spectral_estimate, prior)
        if value is not None and lower <= value <= upper
    ]
    if upper == lower:
        return (lower,)
    if upper - lower + 1 <= limit:
        complete = list(dict.fromkeys((*anchors, lower, upper)))
        complete.extend(
            count
            for count in range(lower, upper + 1)
            if count not in complete
        )
        return tuple(complete)
    if limit == 1:
        # A one-probe budget must preserve the primary acoustic estimate.
        return (leader_estimate,)
    if limit == 2:
        # Two probes are too scarce to spend exclusively on legal boundaries:
        # preserve the acoustic leader and use the second slot for an
        # independent spectral/prior anchor or, failing that, a local neighbor.
        second = next(
            (
                value
                for value in (spectral_estimate, prior)
                if value is not None
                and lower <= value <= upper
                and value != leader_estimate
            ),
            None,
        )
        if second is None:
            local_neighbors = tuple(
                value
                for value in (leader_estimate - 1, leader_estimate + 1)
                if lower <= value <= upper
            )
            second = (
                local_neighbors[0]
                if local_neighbors
                else (lower if leader_estimate != lower else upper)
            )
        return (leader_estimate, second)

    # Reserve roughly one third of the evaluation budget for adaptive
    # refinement after the coarse pass.  Acoustic anchors, their local
    # neighbours, and coarse quantiles come before legal boundaries.  A large
    # legal upper bound is a constraint, not evidence that the most expensive K
    # should be materialized before the acoustic neighbourhood is bracketed.
    seed_limit = min(
        limit,
        max(3, math.ceil(limit * 2.0 / 3.0)),
    )
    span = upper - lower
    quantiles = tuple(
        lower + round(span * fraction)
        for fraction in (0.5, 0.25, 0.75, 0.125, 0.875)
    )
    logarithmic: list[int] = []
    offset = 1
    while lower + offset < upper:
        logarithmic.extend((lower + offset, upper - offset))
        offset *= 2
    anchor_neighbors = tuple(
        neighbor
        for anchor in anchors
        for neighbor in (anchor - 1, anchor + 1)
        if lower <= neighbor <= upper
    )
    scheduled = list(
        dict.fromkeys(
            (
                *anchors,
                *anchor_neighbors,
                quantiles[0],
                *quantiles,
                *logarithmic,
                lower,
                upper,
            )
        )
    )
    return tuple(scheduled[:seed_limit])


def _adaptive_refinement_order(
    *,
    lower: int,
    upper: int,
    candidates: Sequence[_ClusterCandidateScore],
    attempted: set[int],
) -> tuple[int, ...]:
    if not candidates:
        return ()
    ordered = sorted(candidates, key=lambda item: item.count)
    best = max(ordered, key=lambda item: (item.objective, -item.count))
    best_index = ordered.index(best)
    proposals: list[int] = []

    adjacent_intervals: list[tuple[float, int, int, int]] = []
    for neighbor_index in (best_index - 1, best_index + 1):
        if not 0 <= neighbor_index < len(ordered):
            continue
        neighbor = ordered[neighbor_index]
        direction = -1 if neighbor.count < best.count else 1
        adjacent_intervals.append(
            (
                neighbor.objective,
                abs(neighbor.count - best.count),
                direction,
                neighbor.count,
            )
        )

    # First take one-count hill-climbing steps toward the adjacent interval
    # whose evaluated endpoint has the stronger objective.  This lets a coarse
    # probe that lands near a narrow optimum reach it within a small remaining
    # budget instead of repeatedly bisecting the opposite, weaker side.
    adjacent_intervals.sort(
        key=lambda item: (-item[0], -item[1], -item[2])
    )
    proposals.extend(
        best.count + direction
        for _, _, direction, _ in adjacent_intervals
    )

    # Then bisect the same intervals.  Bisection remains the fast fallback for
    # a distant missed optimum when neither immediate neighbour improves.
    for _, _, _, neighbor_count in adjacent_intervals:
        neighbor = next(
            item for item in ordered if item.count == neighbor_count
        )
        midpoint = (best.count + neighbor.count) // 2
        if midpoint in {best.count, neighbor.count}:
            continue
        proposals.append(midpoint)

    if best_index == 0 and best.count > lower:
        proposals.append((lower + best.count) // 2)
    if best_index == len(ordered) - 1 and best.count < upper:
        proposals.append((best.count + upper + 1) // 2)
    proposals.extend((best.count - 1, best.count + 1))

    # If the local intervals are exhausted, split the largest remaining
    # unevaluated gaps.  Endpoint objective is a deterministic priority signal
    # and gap width is the tie-breaker.
    gap_proposals: list[tuple[float, int, int]] = []
    objective_by_count = {
        item.count: item.objective
        for item in ordered
    }
    sorted_counts = sorted({lower, upper, *objective_by_count})
    for left_count, right_count in zip(sorted_counts, sorted_counts[1:]):
        if right_count - left_count <= 1:
            continue
        midpoint = (left_count + right_count) // 2
        endpoint_score = max(
            objective_by_count.get(left_count, -math.inf),
            objective_by_count.get(right_count, -math.inf),
        )
        gap_proposals.append(
            (
                endpoint_score,
                right_count - left_count,
                midpoint,
            )
        )
    proposals.extend(
        midpoint
        for _, _, midpoint in sorted(
            gap_proposals,
            key=lambda item: (-item[0], -item[1], item[2]),
        )
    )
    return tuple(
        dict.fromkeys(
            count
            for count in proposals
            if lower <= count <= upper and count not in attempted
        )
    )


def _candidate_work_items(
    *,
    sample_count: int,
    count: int,
    iterations: int,
    stability_runs: int = 3,
) -> int:
    fit_work = sample_count * count * iterations
    score_work = (
        sample_count * count
        + count * (count - 1) // 2
        + min(sample_count, count) * count
    )
    # Each stability replicate performs a new spherical-k-means fit on a
    # deterministic subsample, scores every full-set observation, aligns labels
    # through a cubic assignment solver, and computes pairwise partition
    # agreement.  This is a conservative upper bound used before any work is
    # started, so the runtime never silently exceeds its configured budget.
    replicate_work = stability_runs * (
        sample_count * count * iterations
        + sample_count * count
        + count**3
        + sample_count
        + count * count
    )
    return (
        fit_work
        + score_work
        + replicate_work
    )


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _eigengap_work_items(
    *,
    sample_count: int,
    landmark_limit: int,
) -> int:
    landmark_count = min(sample_count, landmark_limit)
    return landmark_count * (landmark_count - 1) // 2


def _graph_eigengap_profile(
    vectors: Sequence[tuple[float, ...]],
    *,
    landmark_limit: int,
) -> _EigengapProfile:
    """Estimate graph eigengaps through single-link connectivity persistence.

    For a thresholded cosine-affinity graph, the multiplicity of the zero
    eigenvalue of the normalized Laplacian equals the number of connected
    components.  Tracking component persistence while lowering the threshold
    is therefore a deterministic, bounded-cost eigengap equivalent.  It avoids
    materializing or diagonalizing an unbounded N x N matrix.
    """

    sample_count = len(vectors)
    if sample_count == 1:
        return _EigengapProfile(
            scores={1: 1.0},
            estimate=1,
            method="normalized-laplacian-connectivity-persistence-v1",
            landmark_count=1,
            work_items=0,
        )

    landmark_count = min(sample_count, landmark_limit)
    if landmark_count == sample_count:
        landmark_indexes = tuple(range(sample_count))
    else:
        landmark_indexes = tuple(
            dict.fromkeys(
                round(index * (sample_count - 1) / (landmark_count - 1))
                for index in range(landmark_count)
            )
        )
        landmark_count = len(landmark_indexes)
    landmarks = [vectors[index] for index in landmark_indexes]
    edges = sorted(
        (
            (
                max(-1.0, min(1.0, _dot(landmarks[left], landmarks[right]))),
                left,
                right,
            )
            for left in range(landmark_count)
            for right in range(left + 1, landmark_count)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    parents = list(range(landmark_count))
    ranks = [0] * landmark_count

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if ranks[left_root] < ranks[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        if ranks[left_root] == ranks[right_root]:
            ranks[left_root] += 1
        return True

    components = landmark_count
    upper_similarity = 1.0
    raw_scores: dict[int, float] = {}
    for similarity, left, right in edges:
        if not union(left, right):
            continue
        raw_scores[components] = _clamp_probability(
            (upper_similarity - similarity) / 2.0
        )
        upper_similarity = similarity
        components -= 1
        if components == 1:
            break
    raw_scores[1] = _clamp_probability((upper_similarity + 1.0) / 2.0)

    estimate = max(
        raw_scores,
        key=lambda count: (raw_scores[count], count),
    )
    return _EigengapProfile(
        scores=raw_scores,
        estimate=estimate,
        method=(
            "normalized-laplacian-connectivity-persistence-v1"
            if landmark_count == sample_count
            else "normalized-laplacian-landmark-connectivity-persistence-v1"
        ),
        landmark_count=landmark_count,
        work_items=_eigengap_work_items(
            sample_count=sample_count,
            landmark_limit=landmark_limit,
        ),
    )


def _fit_spherical_kmeans(
    vectors: Sequence[tuple[float, ...]],
    windows: Sequence[SpeechWindow],
    *,
    count: int,
    locks: Mapping[int, int],
    iterations: int,
    work_items: int,
) -> _KMeansFit:
    dimension = len(vectors[0])
    centroids: list[tuple[float, ...] | None] = [None] * count
    for cluster_index in sorted(set(locks.values())):
        locked_vectors = [
            vectors[index]
            for index, target in locks.items()
            if target == cluster_index
        ]
        centroids[cluster_index] = _mean_vector(locked_vectors, dimension)

    selected = set(locks)
    existing = [item for item in centroids if item is not None]
    nearest_similarity = [
        (
            max(_dot(vector, centroid) for centroid in existing)
            if existing
            else -math.inf
        )
        for vector in vectors
    ]
    for cluster_index in range(count):
        if centroids[cluster_index] is not None:
            continue
        candidates = [
            index for index in range(len(vectors)) if index not in selected
        ]
        if not candidates:
            candidates = list(range(len(vectors)))
        chosen = (
            candidates[0]
            if not existing
            else min(
                candidates,
                key=lambda index: (
                    nearest_similarity[index],
                    index,
                ),
            )
        )
        chosen_centroid = vectors[chosen]
        centroids[cluster_index] = chosen_centroid
        existing.append(chosen_centroid)
        selected.add(chosen)
        for index, vector in enumerate(vectors):
            nearest_similarity[index] = max(
                nearest_similarity[index],
                _dot(vector, chosen_centroid),
            )

    concrete = [centroid for centroid in centroids if centroid is not None]
    if len(concrete) != count:
        raise WorkerError("CLUSTERING_FAILED", "failed to initialize centroids")

    assignments = [0] * len(vectors)
    actual_iterations = 0
    for iteration in range(iterations):
        actual_iterations = iteration + 1
        changed = False
        for index, vector in enumerate(vectors):
            target = locks.get(index)
            if target is None:
                target = max(
                    range(count),
                    key=lambda cluster_index: (
                        _dot(vector, concrete[cluster_index]),
                        -cluster_index,
                    ),
                )
            if assignments[index] != target:
                assignments[index] = target
                changed = True

        members = [
            [
                index
                for index, target in enumerate(assignments)
                if target == cluster
            ]
            for cluster in range(count)
        ]
        for empty_cluster, cluster_members in enumerate(members):
            if cluster_members:
                continue
            movable = [
                index
                for index, current in enumerate(assignments)
                if index not in locks and len(members[current]) > 1
            ]
            if not movable:
                raise WorkerError(
                    "CLUSTERING_CONSTRAINT_UNSATISFIABLE",
                    "speaker constraints leave an empty cluster",
                )
            chosen = min(
                movable,
                key=lambda index: (
                    _dot(vectors[index], concrete[assignments[index]]),
                    index,
                ),
            )
            members[assignments[chosen]].remove(chosen)
            assignments[chosen] = empty_cluster
            members[empty_cluster].append(chosen)
            changed = True

        updated = [
            _mean_vector(
                [vectors[index] for index in members[cluster_index]],
                dimension,
            )
            for cluster_index in range(count)
        ]
        movement = max(
            1.0 - _dot(old, new) for old, new in zip(concrete, updated)
        )
        concrete = updated
        if not changed or movement < 1e-8:
            break

    locked_clusters = set(locks.values())
    earliest = {
        cluster: min(
            windows[index].start_ms
            for index, assigned in enumerate(assignments)
            if assigned == cluster
        )
        for cluster in range(count)
    }
    free_old = sorted(
        (cluster for cluster in range(count) if cluster not in locked_clusters),
        key=lambda cluster: (earliest[cluster], cluster),
    )
    free_new = [
        cluster for cluster in range(count) if cluster not in locked_clusters
    ]
    canonical_map = {cluster: cluster for cluster in locked_clusters}
    canonical_map.update(dict(zip(free_old, free_new)))
    canonical_centroids: list[tuple[float, ...]] = [vectors[0]] * count
    for old, new in canonical_map.items():
        canonical_centroids[new] = concrete[old]

    canonical_assignments = tuple(canonical_map[item] for item in assignments)
    score_rows = tuple(
        tuple(
            max(-1.0, min(1.0, _dot(vector, centroid)))
            for centroid in canonical_centroids
        )
        for vector in vectors
    )
    return _KMeansFit(
        count=count,
        assignments=canonical_assignments,
        scores=score_rows,
        centroids=tuple(canonical_centroids),
        iterations=actual_iterations,
        work_items=work_items,
    )


def _maximum_weight_label_map(
    reference_centroids: Sequence[tuple[float, ...]],
    replicate_centroids: Sequence[tuple[float, ...]],
) -> dict[int, int]:
    """Align replicate labels to reference labels with deterministic Hungarian.

    Cluster labels are arbitrary.  Comparing integer labels directly therefore
    overstates instability whenever a resample merely permutes labels.  This
    solver maximizes total centroid cosine similarity and returns
    ``replicate_label -> reference_label``.
    """

    count = len(reference_centroids)
    if count != len(replicate_centroids) or count < 1:
        raise ValueError("centroid sets must have the same positive size")
    maximum_weight = 1.0
    costs = [
        [
            maximum_weight
            - _dot(reference_centroids[row], replicate_centroids[column])
            + (row * count + column) * 1e-12
            for column in range(count)
        ]
        for row in range(count)
    ]

    # Standard O(K^3) Hungarian minimization, using one-based working arrays.
    potentials_left = [0.0] * (count + 1)
    potentials_right = [0.0] * (count + 1)
    matched_left = [0] * (count + 1)
    previous = [0] * (count + 1)
    for left in range(1, count + 1):
        matched_left[0] = left
        column = 0
        minimum = [math.inf] * (count + 1)
        used = [False] * (count + 1)
        while True:
            used[column] = True
            active_left = matched_left[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, count + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    costs[active_left - 1][candidate_column - 1]
                    - potentials_left[active_left]
                    - potentials_right[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    previous[candidate_column] = column
                if (
                    minimum[candidate_column] < delta
                    or (
                        abs(minimum[candidate_column] - delta) <= 1e-15
                        and candidate_column < next_column
                    )
                ):
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(count + 1):
                if used[candidate_column]:
                    potentials_left[matched_left[candidate_column]] += delta
                    potentials_right[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_left[column] == 0:
                break
        while True:
            prior_column = previous[column]
            matched_left[column] = matched_left[prior_column]
            column = prior_column
            if column == 0:
                break

    reference_to_replicate = {
        matched_left[column] - 1: column - 1
        for column in range(1, count + 1)
    }
    return {
        replicate: reference
        for reference, replicate in reference_to_replicate.items()
    }


def _partition_agreement(
    reference: Sequence[int],
    replicate: Sequence[int],
    *,
    label_map: Mapping[int, int],
) -> dict[str, float]:
    if len(reference) != len(replicate) or not reference:
        raise ValueError("partition vectors must have the same positive size")
    contingency: dict[tuple[int, int], int] = {}
    reference_sizes: dict[int, int] = {}
    replicate_sizes: dict[int, int] = {}
    aligned_matches = 0
    for reference_label, replicate_label in zip(reference, replicate):
        contingency[(reference_label, replicate_label)] = (
            contingency.get((reference_label, replicate_label), 0) + 1
        )
        reference_sizes[reference_label] = (
            reference_sizes.get(reference_label, 0) + 1
        )
        replicate_sizes[replicate_label] = (
            replicate_sizes.get(replicate_label, 0) + 1
        )
        aligned_matches += label_map.get(replicate_label) == reference_label

    def choose_two(value: int) -> int:
        return value * (value - 1) // 2

    same_both = sum(choose_two(value) for value in contingency.values())
    same_reference = sum(
        choose_two(value) for value in reference_sizes.values()
    )
    same_replicate = sum(
        choose_two(value) for value in replicate_sizes.values()
    )
    total_pairs = choose_two(len(reference))
    if total_pairs == 0:
        adjusted_rand = 1.0
        pairwise_jaccard = 1.0
        coassociation = 1.0
    else:
        expected = same_reference * same_replicate / total_pairs
        maximum = 0.5 * (same_reference + same_replicate)
        denominator = maximum - expected
        adjusted_rand = (
            1.0
            if abs(denominator) <= 1e-12
            and same_both == same_reference == same_replicate
            else (
                0.0
                if abs(denominator) <= 1e-12
                else _clamp_probability((same_both - expected) / denominator)
            )
        )
        union = same_reference + same_replicate - same_both
        pairwise_jaccard = 1.0 if union == 0 else same_both / union
        disagreements = (
            (same_reference - same_both)
            + (same_replicate - same_both)
        )
        coassociation = _clamp_probability(
            1.0 - disagreements / total_pairs
        )
    aligned_accuracy = aligned_matches / len(reference)
    return {
        "adjustedRand": _clamp_probability(adjusted_rand),
        "pairwiseJaccard": _clamp_probability(pairwise_jaccard),
        "coassociationAgreement": _clamp_probability(coassociation),
        "alignedAccuracy": _clamp_probability(aligned_accuracy),
    }


def _stability_retained_indices(
    *,
    windows: Sequence[SpeechWindow],
    count: int,
    locks: Mapping[int, int],
    run_index: int,
) -> tuple[int, ...]:
    sample_count = len(windows)
    if sample_count <= count:
        return tuple(range(sample_count))
    target = min(
        sample_count,
        max(count, math.ceil(sample_count * 0.80), len(locks)),
    )
    locked = set(locks)
    ranked = sorted(
        (
            (
                hashlib.sha256(
                    json.dumps(
                        {
                            "algorithm": _STABILITY_MASK_ALGORITHM,
                            "runIndex": run_index,
                            "sampleCount": sample_count,
                            "candidateCount": count,
                            "windowId": window.window_id,
                            "startMs": window.start_ms,
                            "endMs": window.end_ms,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).digest(),
                window.window_id,
                window.start_ms,
                window.end_ms,
                index,
            )
            for index, window in enumerate(windows)
            if index not in locked
        )
    )
    retained = locked | {
        item[-1] for item in ranked[: max(0, target - len(locked))]
    }
    return tuple(sorted(retained))


def _recluster_stability_profile(
    fit: _KMeansFit,
    windows: Sequence[SpeechWindow],
    *,
    vectors: Sequence[tuple[float, ...]],
    locks: Mapping[int, int],
    stability_runs: int,
    iterations: int,
) -> _StabilityProfile:
    component_rows: list[Mapping[str, float]] = []
    coverage_rows: list[float] = []
    compactness_rows: list[float] = []
    stability_rows: list[float] = []
    retained_masks: set[tuple[str, ...]] = set()
    for run_index in range(stability_runs):
        retained = _stability_retained_indices(
            windows=windows,
            count=fit.count,
            locks=locks,
            run_index=run_index,
        )
        retained_masks.add(
            tuple(sorted(windows[index].window_id for index in retained))
        )
        index_map = {
            original_index: subset_index
            for subset_index, original_index in enumerate(retained)
        }
        subset_locks = {
            index_map[index]: target
            for index, target in locks.items()
            if index in index_map
        }
        subset_vectors = [vectors[index] for index in retained]
        subset_windows = [windows[index] for index in retained]
        represented = len(
            {
                fit.assignments[index]
                for index in retained
            }
        )
        coverage_rows.append(represented / fit.count)
        try:
            replicate_fit = _fit_spherical_kmeans(
                subset_vectors,
                subset_windows,
                count=fit.count,
                locks=subset_locks,
                iterations=iterations,
                work_items=0,
            )
        except WorkerError:
            component_rows.append(
                {
                    "adjustedRand": 0.0,
                    "pairwiseJaccard": 0.0,
                    "coassociationAgreement": 0.0,
                    "alignedAccuracy": 0.0,
                }
            )
            compactness_rows.append(0.0)
            stability_rows.append(0.0)
            continue

        replicate_assignments: list[int] = []
        replicate_compactness: list[float] = []
        for index, vector in enumerate(vectors):
            target = locks.get(index)
            if target is None:
                target = max(
                    range(fit.count),
                    key=lambda cluster_index: (
                        _dot(vector, replicate_fit.centroids[cluster_index]),
                        -cluster_index,
                    ),
                )
            replicate_assignments.append(target)
            replicate_compactness.append(
                max(
                    0.0,
                    _dot(vector, replicate_fit.centroids[target]),
                )
            )
        label_map = _maximum_weight_label_map(
            fit.centroids,
            replicate_fit.centroids,
        )
        components = _partition_agreement(
            fit.assignments,
            replicate_assignments,
            label_map=label_map,
        )
        run_stability = (
            0.35 * components["adjustedRand"]
            + 0.25 * components["pairwiseJaccard"]
            + 0.20 * components["coassociationAgreement"]
            + 0.20 * components["alignedAccuracy"]
        )
        component_rows.append(components)
        compactness_rows.append(
            sum(replicate_compactness) / len(replicate_compactness)
        )
        stability_rows.append(_clamp_probability(run_stability))

    averaged_components = {
        key: sum(row[key] for row in component_rows) / len(component_rows)
        for key in (
            "adjustedRand",
            "pairwiseJaccard",
            "coassociationAgreement",
            "alignedAccuracy",
        )
    }
    averaged_components["coverage"] = (
        sum(coverage_rows) / len(coverage_rows)
    )
    return _StabilityProfile(
        stability=sum(stability_rows) / len(stability_rows),
        coverage=averaged_components["coverage"],
        components=averaged_components,
        resample_compactness=tuple(compactness_rows),
        resample_stability=tuple(stability_rows),
        requested_runs=stability_runs,
        effective_unique_runs=len(retained_masks),
    )


def _score_cluster_candidate(
    fit: _KMeansFit,
    windows: Sequence[SpeechWindow],
    *,
    vectors: Sequence[tuple[float, ...]],
    locks: Mapping[int, int],
    stability_runs: int,
    iterations: int,
    eigengap: float,
    prior: int | None,
) -> _ClusterCandidateScore:
    sample_count = len(fit.assignments)
    sizes = [
        fit.assignments.count(cluster_index)
        for cluster_index in range(fit.count)
    ]
    assigned_scores = [
        max(0.0, fit.scores[index][assignment])
        for index, assignment in enumerate(fit.assignments)
    ]
    compactness = sum(assigned_scores) / sample_count

    if fit.count == 1:
        separation = 0.0
        approximate_silhouette = 0.0
        maximum_similarity = -1.0
        centroid_similarities: list[float] = []
    else:
        centroid_similarities = [
            _dot(fit.centroids[left], fit.centroids[right])
            for left in range(fit.count)
            for right in range(left + 1, fit.count)
        ]
        maximum_similarity = max(centroid_similarities)
        separation = _clamp_probability(1.0 - maximum_similarity)
        silhouette_rows = []
        for index, assignment in enumerate(fit.assignments):
            assigned_distance = max(
                0.0,
                1.0 - fit.scores[index][assignment],
            )
            nearest_other_distance = min(
                max(0.0, 1.0 - score)
                for cluster_index, score in enumerate(fit.scores[index])
                if cluster_index != assignment
            )
            denominator = max(
                assigned_distance,
                nearest_other_distance,
                1e-12,
            )
            silhouette_rows.append(
                _clamp_probability(
                    (nearest_other_distance - assigned_distance) / denominator
                )
            )
        approximate_silhouette = sum(silhouette_rows) / sample_count

    singleton_count = sum(size == 1 for size in sizes)
    singleton_fraction = singleton_count / fit.count
    expected_cluster_size = sample_count / fit.count
    tiny_limit = max(1, math.floor(expected_cluster_size * 0.25))
    tiny_cluster_count = sum(size <= tiny_limit for size in sizes)
    tiny_cluster_fraction = tiny_cluster_count / fit.count
    minimum_cluster_size = min(sizes)

    chronological = sorted(
        range(sample_count),
        key=lambda index: (
            windows[index].start_ms,
            windows[index].end_ms,
            windows[index].window_id,
        ),
    )
    switches = sum(
        fit.assignments[left] != fit.assignments[right]
        for left, right in zip(chronological, chronological[1:])
    )
    fragmentation = switches / max(1, sample_count - 1)

    dimension = len(vectors[0])
    overall_sum = [
        sum(vector[axis] for vector in vectors)
        for axis in range(dimension)
    ]
    overall_norm = math.sqrt(sum(value * value for value in overall_sum))
    overall_centroid = (
        tuple(value / overall_norm for value in overall_sum)
        if overall_norm > 1e-12
        else fit.centroids[0]
    )
    within_dispersion = sum(
        max(0.0, 1.0 - fit.scores[index][assignment])
        for index, assignment in enumerate(fit.assignments)
    )
    residual_dispersion = within_dispersion / sample_count
    if fit.count == 1:
        calinski_harabasz = 0.0
        calinski_harabasz_utility = 0.0
        davies_bouldin = 1.0
        davies_bouldin_utility = 0.5
    else:
        between_dispersion = sum(
            sizes[cluster_index]
            * max(
                0.0,
                1.0
                - _dot(fit.centroids[cluster_index], overall_centroid),
            )
            for cluster_index in range(fit.count)
        )
        numerator = between_dispersion / max(1, fit.count - 1)
        denominator = max(
            1e-12,
            within_dispersion / max(1, sample_count - fit.count),
        )
        calinski_harabasz = min(1_000_000_000.0, numerator / denominator)
        calinski_harabasz_utility = _clamp_probability(
            calinski_harabasz
            / (calinski_harabasz + max(1.0, float(sample_count)))
        )

        scatter = []
        for cluster_index in range(fit.count):
            members = [
                index
                for index, assignment in enumerate(fit.assignments)
                if assignment == cluster_index
            ]
            scatter.append(
                sum(
                    max(
                        0.0,
                        1.0
                        - _dot(vectors[index], fit.centroids[cluster_index]),
                    )
                    for index in members
                )
                / max(1, len(members))
            )
        cluster_ratios = []
        for left in range(fit.count):
            worst = 0.0
            for right in range(fit.count):
                if left == right:
                    continue
                centroid_distance = max(
                    1e-9,
                    1.0 - _dot(fit.centroids[left], fit.centroids[right]),
                )
                worst = max(
                    worst,
                    (scatter[left] + scatter[right]) / centroid_distance,
                )
            cluster_ratios.append(min(1_000_000.0, worst))
        davies_bouldin = sum(cluster_ratios) / fit.count
        davies_bouldin_utility = _clamp_probability(
            1.0 / (1.0 + davies_bouldin)
        )

    stability_profile = _recluster_stability_profile(
        fit,
        windows,
        vectors=vectors,
        locks=locks,
        stability_runs=stability_runs,
        iterations=iterations,
    )
    stability = stability_profile.stability
    bootstrap_support = stability_profile.coverage

    close_centroid_risk = (
        0.0
        if fit.count == 1
        else _clamp_probability((maximum_similarity - 0.94) / 0.06)
    )
    low_assignment_fraction = sum(
        score < 0.65 for score in assigned_scores
    ) / sample_count
    under_split_risk = _clamp_probability(
        0.72 * _clamp_probability((0.94 - compactness) / 0.20)
        + 0.28 * _clamp_probability((0.12 - eigengap) / 0.12)
    )
    over_split_risk = _clamp_probability(
        0.36 * singleton_fraction
        + 0.14 * tiny_cluster_fraction
        + 0.28 * close_centroid_risk
        + 0.12 * (1.0 - stability)
        + 0.10 * (1.0 - bootstrap_support)
    )
    outlier_risk = _clamp_probability(
        0.75 * singleton_fraction + 0.25 * low_assignment_fraction
    )
    imbalanced_mass = sum(
        size
        for size in sizes
        if size < max(1.0, expected_cluster_size * 0.25)
    ) / sample_count
    imbalance_penalty = 0.06 * imbalanced_mass
    complexity_penalty = (
        0.006 * (fit.count - 1) / math.sqrt(sample_count)
    )
    weights = {
        "compactness": 0.18,
        "silhouette": 0.16,
        "calinskiHarabasz": 0.11,
        "daviesBouldin": 0.09,
        "eigengap": 0.16,
        "stability": 0.18,
        "separation": 0.12,
    }
    weighted_contributions = {
        "compactness": weights["compactness"] * compactness,
        "silhouette": weights["silhouette"] * approximate_silhouette,
        "calinskiHarabasz": (
            weights["calinskiHarabasz"] * calinski_harabasz_utility
        ),
        "daviesBouldin": (
            weights["daviesBouldin"] * davies_bouldin_utility
        ),
        "eigengap": weights["eigengap"] * eigengap,
        "stability": weights["stability"] * stability,
        "separation": weights["separation"] * separation,
        "complexityPenalty": -complexity_penalty,
        "underSplitPenalty": -0.10 * under_split_risk,
        "overSplitPenalty": -0.13 * over_split_risk,
        "outlierPenalty": -0.08 * outlier_risk,
        "imbalancePenalty": -imbalance_penalty,
        "fragmentationPenalty": -0.003 * fragmentation,
    }
    objective = sum(weighted_contributions.values())
    resample_objectives = tuple(
        objective
        + 0.22 * (replicate_compactness - compactness)
        + 0.12 * (replicate_stability - stability)
        for replicate_compactness, replicate_stability in zip(
            stability_profile.resample_compactness,
            stability_profile.resample_stability,
        )
    )
    correction_reasons: list[str] = []
    if under_split_risk >= 0.30:
        correction_reasons.append("UNDER_SPLIT_RISK")
    if over_split_risk >= 0.22:
        correction_reasons.append("OVER_SPLIT_RISK")
    if outlier_risk >= 0.15:
        correction_reasons.append("OUTLIER_CLUSTER_RISK")
    if bootstrap_support < 0.85:
        correction_reasons.append("LOW_BOOTSTRAP_SUPPORT")
    return _ClusterCandidateScore(
        count=fit.count,
        objective=objective,
        compactness=compactness,
        separation=separation,
        approximate_silhouette=approximate_silhouette,
        calinski_harabasz=calinski_harabasz,
        calinski_harabasz_utility=calinski_harabasz_utility,
        davies_bouldin=davies_bouldin,
        davies_bouldin_utility=davies_bouldin_utility,
        eigengap=eigengap,
        eigengap_utility=eigengap,
        stability=stability,
        bootstrap_support=bootstrap_support,
        residual_dispersion=residual_dispersion,
        singleton_count=singleton_count,
        singleton_fraction=singleton_fraction,
        tiny_cluster_count=tiny_cluster_count,
        tiny_cluster_fraction=tiny_cluster_fraction,
        minimum_cluster_size=minimum_cluster_size,
        fragmentation=fragmentation,
        complexity_penalty=complexity_penalty,
        under_split_risk=under_split_risk,
        over_split_risk=over_split_risk,
        outlier_risk=outlier_risk,
        imbalance_penalty=imbalance_penalty,
        consensus_support=0.0,
        metric_votes=0,
        weighted_contributions=weighted_contributions,
        correction_reasons=tuple(correction_reasons),
        prior_distance=None if prior is None else abs(prior - fit.count),
        work_items=fit.work_items,
        stability_components=stability_profile.components,
        resample_objectives=resample_objectives,
        stability_requested_runs=stability_profile.requested_runs,
        stability_effective_unique_runs=(
            stability_profile.effective_unique_runs
        ),
    )


def _finalize_cluster_candidate_scores(
    candidates: Sequence[_ClusterCandidateScore],
) -> list[_ClusterCandidateScore]:
    metric_specs = (
        ("compactness", 0.18, 0.015),
        ("approximate_silhouette", 0.16, 0.05),
        ("calinski_harabasz_utility", 0.11, 0.05),
        ("davies_bouldin_utility", 0.09, 0.05),
        ("eigengap_utility", 0.16, 0.05),
        ("stability", 0.18, 0.03),
        ("separation", 0.12, 0.04),
    )
    maxima = {
        name: max(float(getattr(candidate, name)) for candidate in candidates)
        for name, _, _ in metric_specs
    }
    finalized: list[_ClusterCandidateScore] = []
    for candidate in candidates:
        consensus = 0.0
        votes = 0
        for name, weight, tolerance in metric_specs:
            value = float(getattr(candidate, name))
            maximum = maxima[name]
            consensus += weight * (
                1.0 if maximum <= 1e-12 else _clamp_probability(value / maximum)
            )
            if maximum - value <= tolerance:
                votes += 1
        consensus = _clamp_probability(consensus)
        contributions = dict(candidate.weighted_contributions)
        contributions["consensusSupport"] = 0.06 * consensus
        finalized.append(
            replace(
                candidate,
                objective=candidate.objective + 0.06 * consensus,
                consensus_support=consensus,
                metric_votes=votes,
                weighted_contributions=contributions,
                resample_objectives=tuple(
                    value + 0.06 * consensus
                    for value in candidate.resample_objectives
                ),
            )
        )
    return finalized


def _is_persistent_singleton_outlier_step(
    persistent: _ClusterCandidateScore,
    with_singleton: _ClusterCandidateScore,
) -> bool:
    """Recognize one uncorroborated cluster without hiding the ambiguity.

    A single speech window is not enough evidence to silently promote a new
    persistent speaker.  We therefore prefer the adjacent persistent count
    only when the lower-count fit remains coherent and both counts stay in the
    reported confidence interval for mandatory human review.
    """

    return (
        with_singleton.count == persistent.count + 1
        and with_singleton.singleton_count == 1
        and with_singleton.tiny_cluster_count == 1
        and with_singleton.minimum_cluster_size == 1
        and persistent.singleton_count == 0
        and persistent.tiny_cluster_count == 0
        and persistent.minimum_cluster_size >= 2
        and persistent.compactness >= 0.75
        and persistent.residual_dispersion <= 0.25
        and persistent.stability >= 0.90
        and persistent.bootstrap_support >= 0.90
        and with_singleton.stability >= persistent.stability - 0.10
        and with_singleton.bootstrap_support >= 0.80
        and with_singleton.bootstrap_support
        >= persistent.bootstrap_support - 0.20
        and with_singleton.objective - persistent.objective <= 0.52
    )


def _is_resolvable_close_voice_step(
    merged: _ClusterCandidateScore,
    separated: _ClusterCandidateScore,
) -> bool:
    """Detect a stable adjacent split whose residual collapses dramatically.

    Raw centroid separation is intentionally not treated as a veto here:
    closely related voices can have low inter-centroid distance even when
    repeated observations form two exceptionally compact, stable clusters.
    Singleton/tiny-cluster guards keep this correction from rewarding ordinary
    overfitting.
    """

    residual_before = max(1e-12, merged.residual_dispersion)
    residual_after = max(0.0, separated.residual_dispersion)
    residual_reduction = (residual_before - residual_after) / residual_before
    return (
        separated.count == merged.count + 1
        and merged.singleton_count == 0
        and merged.tiny_cluster_count == 0
        and separated.singleton_count == 0
        and separated.tiny_cluster_count == 0
        and separated.minimum_cluster_size >= 2
        and residual_before >= 1e-5
        and residual_reduction >= 0.80
        and residual_after <= residual_before * 0.20
        and separated.compactness > merged.compactness
        and separated.approximate_silhouette
        >= merged.approximate_silhouette
        and separated.calinski_harabasz_utility
        >= merged.calinski_harabasz_utility + 0.02
        and separated.davies_bouldin_utility
        >= merged.davies_bouldin_utility - 0.02
        and separated.stability >= 0.90
        and separated.stability >= merged.stability - 0.08
        and separated.bootstrap_support >= 0.88
        and separated.bootstrap_support
        >= merged.bootstrap_support - 0.10
        and separated.separation <= 0.12
        and separated.objective >= merged.objective - 0.23
    )


def _uses_only_unconfirmed_count_partitions(
    windows: Sequence[SpeechWindow],
) -> bool:
    """Return whether every identity sample is review-only cardinality evidence.

    A confirmed change point proves that at least one transition exists, but it
    does not make every uniform cardinality partition an independently
    confirmed identity. Human locks and non-partition windows remain the
    evidence that can make an all-singleton solution legitimate.
    """

    if not windows or any(window.locked_speaker_id for window in windows):
        return False
    for window in windows:
        partition = window.metadata.get("speakerCountPartition")
        if (
            not isinstance(partition, Mapping)
            or partition.get("method")
            not in {
                "auto-acoustic-contiguous-partition-v1",
                "policy-minimum-contiguous-partition-v1",
            }
            or partition.get("reviewRequired") is not True
            or not isinstance(partition.get("sourceWindowId"), str)
            or not str(partition["sourceWindowId"]).strip()
        ):
            return False
    return True


def _strongest_partition_degeneracy_alternative(
    candidates: Sequence[_ClusterCandidateScore],
    *,
    selected: _ClusterCandidateScore,
    sample_count: int,
) -> _ClusterCandidateScore | None:
    """Find a repeatable fit that does not make every evidence window unique."""

    if (
        selected.count != sample_count
        or selected.singleton_count != selected.count
        or selected.minimum_cluster_size != 1
    ):
        return None
    alternatives = [
        candidate
        for candidate in candidates
        if (
            candidate.singleton_count < candidate.count
            and candidate.stability >= _PARTITION_DEGENERACY_MIN_STABILITY
            and candidate.bootstrap_support
            >= _PARTITION_DEGENERACY_MIN_BOOTSTRAP_SUPPORT
            and candidate.stability_effective_unique_runs
            >= _PARTITION_DEGENERACY_MIN_UNIQUE_RESAMPLE_RUNS
        )
    ]
    if not alternatives:
        return None
    return max(
        alternatives,
        key=lambda candidate: (candidate.objective, -candidate.count),
    )


def _cluster(
    embeddings: Sequence[EmbeddingRecord],
    windows: Sequence[SpeechWindow],
    request: StartJobRequest,
    config: SpeakerPipelineConfig,
    *,
    pyannote_count_prior: int | None = None,
) -> _ClusterResult:
    if len(embeddings) != len(windows) or not embeddings:
        raise WorkerError(
            "EMBEDDING_INVALID",
            "one embedding is required for every speech window",
        )
    if len(windows) > config.max_clustering_windows:
        raise WorkerError(
            "CLUSTERING_RESOURCE_LIMIT_EXCEEDED",
            "speech-window count exceeds the configured clustering limit",
            details={
                "speechWindows": len(windows),
                "maxClusteringWindows": config.max_clustering_windows,
            },
        )
    dimensions = {len(record.vector) for record in embeddings}
    if len(dimensions) != 1:
        raise WorkerError(
            "EMBEDDING_DIMENSION_MISMATCH",
            "all embedding vectors must have the same dimension",
        )
    original_vectors = [_normalize(record.vector) for record in embeddings]
    stable_order = sorted(
        range(len(original_vectors)),
        key=lambda index: (
            original_vectors[index],
            windows[index].start_ms,
            windows[index].end_ms,
            windows[index].window_id,
            embeddings[index].window_id,
        ),
    )
    vectors = [original_vectors[index] for index in stable_order]
    ordered_windows = [windows[index] for index in stable_order]

    lock_numbers = [
        number
        for number in (
            _speaker_number(window.locked_speaker_id)
            for window in ordered_windows
        )
        if number is not None
    ]
    minimum_locked_count = max(lock_numbers, default=1)
    lower, upper, leader_estimate, leader_work_items = _resolve_cluster_count(
        vectors,
        request,
        config,
        minimum_locked_count=minimum_locked_count,
    )
    policy = request.speaker_policy
    policy_prior = (
        policy.prior
        if policy.mode is SpeakerCountMode.HYBRID
        else None
    )
    external_prior = (
        pyannote_count_prior
        if policy.mode is SpeakerCountMode.AUTO
        and pyannote_count_prior is not None
        and lower <= pyannote_count_prior <= upper
        else None
    )
    search_prior = policy_prior if policy_prior is not None else external_prior
    if policy.mode is SpeakerCountMode.MANUAL:
        eigengap_profile = _EigengapProfile(
            scores={leader_estimate: 1.0},
            estimate=leader_estimate,
            method="manual-policy-exact",
            landmark_count=0,
            work_items=0,
        )
    else:
        eigengap_work_items = _eigengap_work_items(
            sample_count=len(vectors),
            landmark_limit=config.eigengap_landmark_limit,
        )
        if (
            leader_work_items + eigengap_work_items
            > config.max_clustering_work_items
        ):
            raise WorkerError(
                "CLUSTERING_RESOURCE_LIMIT_EXCEEDED",
                "speaker eigengap analysis exceeds the configured work limit",
                details={
                    "phase": "eigengap-precheck",
                    "leaderCountWorkItems": leader_work_items,
                    "eigengapWorkItems": eigengap_work_items,
                    "workItems": leader_work_items + eigengap_work_items,
                    "maxWorkItems": config.max_clustering_work_items,
                },
            )
        eigengap_profile = _graph_eigengap_profile(
            vectors,
            landmark_limit=config.eigengap_landmark_limit,
        )
        if (
            leader_work_items + eigengap_profile.work_items
            > config.max_clustering_work_items
        ):
            raise WorkerError(
                "CLUSTERING_RESOURCE_LIMIT_EXCEEDED",
                "speaker eigengap analysis exceeds the configured work limit",
                details={
                    "phase": "eigengap",
                    "leaderCountWorkItems": leader_work_items,
                    "eigengapWorkItems": eigengap_profile.work_items,
                    "workItems": (
                        leader_work_items + eigengap_profile.work_items
                    ),
                    "maxWorkItems": config.max_clustering_work_items,
                },
            )
    spectral_estimate = (
        None
        if eigengap_profile.estimate is None
        else max(lower, min(upper, eigengap_profile.estimate))
    )
    base_work_items = leader_work_items + eigengap_profile.work_items
    minimum_required_work_items = base_work_items + _candidate_work_items(
        sample_count=len(vectors),
        count=lower,
        iterations=config.kmeans_iterations,
        stability_runs=config.count_stability_runs,
    )
    affordable_counts = tuple(
        count
        for count in range(lower, upper + 1)
        if (
            base_work_items
            + _candidate_work_items(
                sample_count=len(vectors),
                count=count,
                iterations=config.kmeans_iterations,
                stability_runs=config.count_stability_runs,
            )
            <= config.max_clustering_work_items
        )
    )
    resource_affordable_max_count = (
        max(affordable_counts) if affordable_counts else None
    )
    planning_upper = (
        upper
        if resource_affordable_max_count is None
        else resource_affordable_max_count
    )
    candidate_counts = _candidate_count_order(
        lower=lower,
        upper=planning_upper,
        leader_estimate=leader_estimate,
        spectral_estimate=spectral_estimate,
        prior=search_prior,
        maximum_candidates=(
            1
            if policy.mode is SpeakerCountMode.MANUAL
            else config.max_count_uncertainty_candidates
        ),
    )

    ordered_locks: dict[int, int] = {}
    for index, window in enumerate(ordered_windows):
        speaker_number = _speaker_number(window.locked_speaker_id)
        if speaker_number is None:
            continue
        ordered_locks[index] = speaker_number - 1

    fits: dict[int, _KMeansFit] = {}
    candidate_scores: list[_ClusterCandidateScore] = []
    total_work_items = base_work_items
    evaluation_limit = (
        1
        if policy.mode is SpeakerCountMode.MANUAL
        else min(
            config.max_count_uncertainty_candidates,
            planning_upper - lower + 1,
        )
    )
    attempt_limit = min(
        planning_upper - lower + 1,
        max(evaluation_limit * 3, evaluation_limit + 4),
    )
    queue = list(candidate_counts)
    planned_counts = list(candidate_counts)
    attempted_counts: set[int] = set()
    resource_skipped_counts: list[int] = []
    minimum_resource_failure: tuple[int, int] | None = None
    while (
        len(candidate_scores) < evaluation_limit
        and len(attempted_counts) < attempt_limit
    ):
        if not queue:
            refinements = _adaptive_refinement_order(
                lower=lower,
                upper=planning_upper,
                candidates=_finalize_cluster_candidate_scores(
                    candidate_scores
                ),
                attempted=attempted_counts,
            )
            if not refinements:
                break
            # Evaluate one refinement at a time, then recompute the next
            # proposal from the newly observed objective surface.  Enqueuing
            # the whole stale proposal list would make the "adaptive" phase a
            # second static sweep and could spend the remaining budget far
            # away from the updated optimum.
            count = refinements[0]
            if count not in planned_counts:
                planned_counts.append(count)
            queue.append(count)
        count = queue.pop(0)
        if count in attempted_counts:
            continue
        attempted_counts.add(count)
        candidate_work_items = _candidate_work_items(
            sample_count=len(vectors),
            count=count,
            iterations=config.kmeans_iterations,
            stability_runs=config.count_stability_runs,
        )
        required_work_items = total_work_items + candidate_work_items
        if required_work_items > config.max_clustering_work_items:
            resource_skipped_counts.append(count)
            resource_failure = (required_work_items, count)
            if (
                minimum_resource_failure is None
                or resource_failure < minimum_resource_failure
            ):
                minimum_resource_failure = resource_failure
            continue
        fit = _fit_spherical_kmeans(
            vectors,
            ordered_windows,
            count=count,
            locks=ordered_locks,
            iterations=config.kmeans_iterations,
            work_items=candidate_work_items,
        )
        fits[count] = fit
        candidate_scores.append(
            _score_cluster_candidate(
                fit,
                ordered_windows,
                vectors=vectors,
                locks=ordered_locks,
                stability_runs=config.count_stability_runs,
                iterations=config.kmeans_iterations,
                eigengap=_clamp_probability(
                    eigengap_profile.scores.get(count, 0.0)
                ),
                prior=policy_prior,
            )
        )
        total_work_items = required_work_items

    if not candidate_scores:
        if minimum_resource_failure is not None:
            required_work_items, failed_count = minimum_resource_failure
            raise WorkerError(
                "CLUSTERING_RESOURCE_LIMIT_EXCEEDED",
                "speaker clustering exceeds the configured work limit",
                details={
                    "phase": "multi-k-candidate",
                    "candidateCount": failed_count,
                    "leaderCountWorkItems": leader_work_items,
                    "candidateWorkItems": (
                        required_work_items - total_work_items
                    ),
                    "workItems": required_work_items,
                    "maxWorkItems": config.max_clustering_work_items,
                    "attemptedCounts": sorted(attempted_counts),
                },
            )
        raise WorkerError(
            "CLUSTERING_FAILED",
            "speaker clustering did not evaluate a legal count candidate",
        )

    candidate_scores = _finalize_cluster_candidate_scores(candidate_scores)
    scores_by_count = {item.count: item for item in candidate_scores}
    acoustic_best = max(
        candidate_scores,
        key=lambda item: (item.objective, -item.count),
    )
    selected_score = acoustic_best
    correction_path: list[str] = []
    under_split_detected = False
    over_split_detected = False
    partition_degeneracy_detected = False

    if (
        policy.mode is not SpeakerCountMode.MANUAL
        and _uses_only_unconfirmed_count_partitions(ordered_windows)
    ):
        partition_alternative = _strongest_partition_degeneracy_alternative(
            candidate_scores,
            selected=selected_score,
            sample_count=len(vectors),
        )
        if partition_alternative is not None:
            correction_path.extend(
                (
                    (
                        f"OVER_SPLIT_CORRECTION:"
                        f"{selected_score.count}->{partition_alternative.count}"
                    ),
                    "ALL_SINGLETON_PARTITION_DEGENERACY",
                )
            )
            over_split_detected = True
            partition_degeneracy_detected = True
            selected_score = partition_alternative

    lower_score = scores_by_count.get(selected_score.count - 1)
    if (
        not partition_degeneracy_detected
        and lower_score is not None
        and _is_persistent_singleton_outlier_step(
            lower_score,
            selected_score,
        )
    ):
        correction_path.extend(
            (
                (
                    f"OVER_SPLIT_CORRECTION:"
                    f"{selected_score.count}->{lower_score.count}"
                ),
                "ABSOLUTE_SINGLETON_OUTLIER_AMBIGUITY",
            )
        )
        over_split_detected = True
        selected_score = lower_score
    elif (
        not partition_degeneracy_detected
        and lower_score is not None
        and (
            selected_score.over_split_risk >= 0.22
            or selected_score.outlier_risk >= 0.15
        )
        and selected_score.objective - lower_score.objective <= 0.14
        and lower_score.stability >= selected_score.stability - 0.10
    ):
        correction_path.append(
            f"OVER_SPLIT_CORRECTION:{selected_score.count}->{lower_score.count}"
        )
        over_split_detected = True
        selected_score = lower_score

    upper_score = scores_by_count.get(selected_score.count + 1)
    if upper_score is not None and not over_split_detected:
        generic_under_split = False
        if _is_resolvable_close_voice_step(selected_score, upper_score):
            correction_path.extend(
                (
                    (
                        f"UNDER_SPLIT_CORRECTION:"
                        f"{selected_score.count}->{upper_score.count}"
                    ),
                    "CLOSE_VOICE_RESIDUAL_COLLAPSE",
                )
            )
            under_split_detected = True
            selected_score = upper_score
        else:
            improvements = sum(
                (
                    upper_score.approximate_silhouette
                    > selected_score.approximate_silhouette + 0.03,
                    upper_score.calinski_harabasz_utility
                    > selected_score.calinski_harabasz_utility + 0.03,
                    upper_score.davies_bouldin_utility
                    > selected_score.davies_bouldin_utility + 0.03,
                    upper_score.compactness
                    > selected_score.compactness + 0.015,
                    upper_score.eigengap_utility
                    > selected_score.eigengap_utility + 0.05,
                )
            )
            generic_under_split = (
                improvements >= 3
                and upper_score.objective
                >= selected_score.objective - 0.03
                and upper_score.stability
                >= selected_score.stability - 0.08
                and upper_score.over_split_risk < 0.38
            )
        if not under_split_detected and generic_under_split:
            correction_path.append(
                f"UNDER_SPLIT_CORRECTION:{selected_score.count}->{upper_score.count}"
            )
            under_split_detected = True
            selected_score = upper_score

    if policy_prior is not None:
        prior_score = next(
            (item for item in candidate_scores if item.count == policy_prior),
            None,
        )
        close_voice_ambiguity = (
            prior_score is not None
            and (
                min(selected_score.separation, prior_score.separation) < 0.08
                or (
                    {selected_score.count, prior_score.count} == {1, 2}
                    and max(
                        selected_score.separation,
                        prior_score.separation,
                    )
                    <= 0.08
                    and min(
                        selected_score.compactness,
                        prior_score.compactness,
                    )
                    >= 0.95
                )
            )
        )
        prior_tolerance = 0.15 if close_voice_ambiguity else 0.035
        if (
            prior_score is not None
            and selected_score.objective - prior_score.objective
            <= prior_tolerance
        ):
            if prior_score.count != selected_score.count:
                correction_path.append(
                    f"HYBRID_PRIOR_TIE_BREAK:{selected_score.count}->{prior_score.count}"
                )
            selected_score = prior_score

    acoustic_selected_score = selected_score
    pyannote_prior_score = (
        scores_by_count.get(external_prior)
        if external_prior is not None
        else None
    )
    pyannote_prior_applied = False
    if (
        pyannote_prior_score is not None
        and pyannote_prior_score.count != selected_score.count
    ):
        objective_gap = selected_score.objective - pyannote_prior_score.objective
        prior_has_repeatable_support = (
            pyannote_prior_score.stability
            >= _PYANNOTE_COUNT_PRIOR_MIN_STABILITY
            and pyannote_prior_score.bootstrap_support
            >= _PYANNOTE_COUNT_PRIOR_MIN_BOOTSTRAP_SUPPORT
        )
        if (
            not partition_degeneracy_detected
            and prior_has_repeatable_support
            and objective_gap <= _PYANNOTE_COUNT_PRIOR_OBJECTIVE_TOLERANCE
        ):
            correction_path.append(
                "PYANNOTE_FULL_TIMELINE_PRIOR:"
                f"{selected_score.count}->{pyannote_prior_score.count}"
            )
            selected_score = pyannote_prior_score
            pyannote_prior_applied = True

    requested_resample_runs = config.count_stability_runs
    effective_unique_resample_runs = min(
        candidate.stability_effective_unique_runs
        for candidate in candidate_scores
    )
    resample_objective_winner_counts: dict[int, int] = {}
    for run_index in range(requested_resample_runs):
        winner = max(
            candidate_scores,
            key=lambda item: (
                item.resample_objectives[run_index],
                -item.count,
            ),
        )
        resample_objective_winner_counts[winner.count] = (
            resample_objective_winner_counts.get(winner.count, 0) + 1
        )
    resample_objective_winner_frequency = {
        count: selected_runs / requested_resample_runs
        for count, selected_runs in resample_objective_winner_counts.items()
    }
    resample_objective_winner_support = (
        resample_objective_winner_frequency.get(selected_score.count, 0.0)
    )

    evaluated_counts = tuple(sorted(fits))
    search_exhaustive = set(evaluated_counts) == set(
        range(lower, upper + 1)
    )
    has_direct_lower_bracket = (
        selected_score.count == lower
        or selected_score.count - 1 in evaluated_counts
    )
    has_direct_upper_bracket = (
        selected_score.count == upper
        or selected_score.count + 1 in evaluated_counts
    )
    decision_locally_bracketed = (
        has_direct_lower_bracket and has_direct_upper_bracket
    )
    required_neighbor_counts = {
        neighbor
        for neighbor in (selected_score.count - 1, selected_score.count + 1)
        if lower <= neighbor <= upper
    }
    resource_truncated = bool(
        required_neighbor_counts.intersection(resource_skipped_counts)
    )
    adaptive_budget_limited = (
        policy.mode is not SpeakerCountMode.MANUAL
        and not search_exhaustive
        and not decision_locally_bracketed
        and not resource_truncated
    )
    count_search_truncated = bool(
        resource_truncated or adaptive_budget_limited
    )
    if resource_truncated:
        search_truncation_reason = "resource-limit"
    elif adaptive_budget_limited:
        search_truncation_reason = (
            "adaptive-budget-unresolved-local-bracket"
        )
    else:
        search_truncation_reason = None

    selected_fit = fits[selected_score.count]
    acoustic_tie_tolerance = (
        0.12
        if selected_score.separation < 0.08
        or selected_score.over_split_risk >= 0.20
        else 0.045
    )
    plausible = [
        item
        for item in candidate_scores
        if (
            selected_score.objective - item.objective <= acoustic_tie_tolerance
            or acoustic_best.objective - item.objective <= acoustic_tie_tolerance
            or (
                abs(item.count - selected_score.count) == 1
                and item.singleton_fraction > 0.0
            )
            or (
                {item.count, selected_score.count} == {1, 2}
                and max(item.separation, selected_score.separation) <= 0.08
                and min(item.compactness, selected_score.compactness) >= 0.95
            )
            or any(
                path.startswith(
                    (
                        f"OVER_SPLIT_CORRECTION:{item.count}->",
                        f"UNDER_SPLIT_CORRECTION:{item.count}->",
                    )
                )
                or path.endswith(f"->{item.count}")
                for path in correction_path
            )
        )
    ]
    if selected_score not in plausible:
        plausible.append(selected_score)
    if (
        pyannote_prior_score is not None
        and pyannote_prior_score not in plausible
        and pyannote_prior_score.count != acoustic_selected_score.count
    ):
        plausible.append(pyannote_prior_score)
    if (
        pyannote_prior_applied
        and acoustic_selected_score not in plausible
    ):
        plausible.append(acoustic_selected_score)

    intrinsic_confidence = (
        0.15 * selected_score.compactness
        + 0.13 * selected_score.approximate_silhouette
        + 0.08 * selected_score.separation
        + 0.17 * selected_score.stability
        + 0.12 * selected_score.bootstrap_support
        + 0.15 * selected_score.consensus_support
        + 0.10 * selected_score.davies_bouldin_utility
        + 0.10 * selected_score.eigengap_utility
    )
    sorted_objectives = sorted(
        (item.objective for item in candidate_scores),
        reverse=True,
    )
    objective_margin = (
        1.0
        if len(sorted_objectives) == 1
        else max(0.0, sorted_objectives[0] - sorted_objectives[1])
    )
    confidence = max(
        0.0,
        min(
            1.0,
            0.80 * intrinsic_confidence
            + 0.20 * _clamp_probability(objective_margin / 0.10),
        ),
    )
    confidence_reasons: list[str] = []
    if len(plausible) > 1:
        confidence = min(confidence, 0.79)
        confidence_reasons.append("MULTIPLE_PLAUSIBLE_COUNTS")
    if selected_score.count > 1 and selected_score.separation < 0.08:
        confidence = min(confidence, 0.78)
        confidence_reasons.append("CLOSE_CENTROID_AMBIGUITY")
    if any(item.singleton_fraction > 0.0 for item in plausible):
        confidence = min(confidence, 0.72)
        confidence_reasons.append("SINGLETON_OUTLIER_AMBIGUITY")
    if "ABSOLUTE_SINGLETON_OUTLIER_AMBIGUITY" in correction_path:
        confidence = min(confidence, 0.68)
        confidence_reasons.append(
            "PERSISTENT_COUNT_SELECTED_SINGLETON_REVIEW_REQUIRED"
        )
    if "CLOSE_VOICE_RESIDUAL_COLLAPSE" in correction_path:
        confidence = min(confidence, 0.74)
        confidence_reasons.append("CLOSE_VOICE_RESIDUAL_CORRECTION")
    if partition_degeneracy_detected:
        confidence = min(confidence, 0.60)
        confidence_reasons.append(
            "PARTITION_DERIVED_ALL_SINGLETON_REVIEW_REQUIRED"
        )
    if selected_score.bootstrap_support < 0.85:
        confidence = min(confidence, 0.70)
        confidence_reasons.append("LOW_BOOTSTRAP_SUPPORT")
    if (
        resample_objective_winner_support < 2.0 / 3.0
    ):
        confidence = min(confidence, 0.69)
        confidence_reasons.append(
            "LOW_RESAMPLE_OBJECTIVE_WINNER_SUPPORT"
        )
    if (
        spectral_estimate is not None
        and spectral_estimate != leader_estimate
    ):
        confidence = min(confidence, 0.79)
        confidence_reasons.append("LEADER_EIGENGAP_DISAGREEMENT")
    if correction_path:
        confidence = min(confidence, 0.78)
        confidence_reasons.append("COUNT_CORRECTION_APPLIED")
    if pyannote_prior_applied:
        confidence = min(confidence, 0.70)
        confidence_reasons.append("PYANNOTE_COUNT_PRIOR_APPLIED_WITH_REVIEW")
    elif (
        pyannote_prior_score is not None
        and pyannote_prior_score.count != selected_score.count
    ):
        confidence = min(confidence, 0.65)
        confidence_reasons.append("PYANNOTE_COUNT_PRIOR_CONFLICT")
    if count_search_truncated:
        confidence = min(confidence, 0.50)
        confidence_reasons.append("RESOURCE_BOUNDED_SEARCH_TRUNCATED")

    if policy.mode is SpeakerCountMode.MANUAL:
        confidence = 1.0
        candidate_min = candidate_max = selected_score.count
        confidence_reasons = ["MANUAL_COUNT_EXACT"]
    elif count_search_truncated:
        candidate_min = lower
        candidate_max = upper
    else:
        candidate_min = min(item.count for item in plausible)
        candidate_max = max(item.count for item in plausible)
    low_confidence_fail_closed = (
        policy.mode is not SpeakerCountMode.MANUAL
        and (
            confidence < config.auto_count_confidence_threshold
            or candidate_min != candidate_max
            or count_search_truncated
        )
    )
    if low_confidence_fail_closed:
        confidence_reasons.append("FAIL_CLOSED_REVIEW_REQUIRED")

    assignments_by_input = [0] * len(vectors)
    score_rows_by_input: list[tuple[float, ...]] = [tuple()] * len(vectors)
    for ordered_index, input_index in enumerate(stable_order):
        assignments_by_input[input_index] = selected_fit.assignments[
            ordered_index
        ]
        score_rows_by_input[input_index] = selected_fit.scores[ordered_index]
    return _ClusterResult(
        count=selected_score.count,
        confidence=confidence,
        candidate_min=candidate_min,
        candidate_max=candidate_max,
        assignments=tuple(assignments_by_input),
        scores=tuple(score_rows_by_input),
        count_candidates=tuple(candidate_scores),
        count_search_truncated=count_search_truncated,
        leader_estimate=leader_estimate,
        spectral_estimate=spectral_estimate,
        eigengap_method=eigengap_profile.method,
        leader_count_work_items=leader_work_items,
        total_work_items=total_work_items,
        selection_method=_CLUSTER_SELECTION_METHOD,
        confidence_reasons=tuple(dict.fromkeys(confidence_reasons)),
        correction_path=tuple(correction_path),
        under_split_detected=under_split_detected,
        over_split_detected=over_split_detected,
        low_confidence_fail_closed=low_confidence_fail_closed,
        legal_min=lower,
        legal_max=upper,
        evaluated_counts=evaluated_counts,
        planned_counts=tuple(dict.fromkeys(planned_counts)),
        search_truncation_reason=search_truncation_reason,
        resample_objective_winner_frequency=(
            resample_objective_winner_frequency
        ),
        resample_objective_winner_support=(
            resample_objective_winner_support
        ),
        requested_resample_runs=requested_resample_runs,
        effective_unique_resample_runs=effective_unique_resample_runs,
        search_exhaustive=search_exhaustive,
        decision_locally_bracketed=decision_locally_bracketed,
        adaptive_budget_limited=adaptive_budget_limited,
        resource_truncated=resource_truncated,
        resource_skipped_counts=tuple(sorted(set(resource_skipped_counts))),
        minimum_required_work_items=minimum_required_work_items,
        resource_affordable_max_count=resource_affordable_max_count,
    )


def _coerce_batch(
    values: Any,
    *,
    expected_ids: set[str],
    converter: Callable[[Any], Any],
    accepted_type: type,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
    ):
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{label} adapter must return an array, not {type(values).__name__}",
        )
    try:
        converted = []
        for index, item in enumerate(values):
            if isinstance(item, accepted_type):
                converted.append(item)
            elif isinstance(item, Mapping):
                converted.append(converter(item))
            else:
                raise TypeError(
                    f"{label}[{index}] has invalid type {type(item).__name__}"
                )
    except (TypeError, ValueError, WorkerError) as exc:
        if isinstance(exc, WorkerError):
            raise
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{label} adapter returned malformed data",
            details={"exceptionType": type(exc).__name__},
        ) from exc
    by_id: dict[str, Any] = {}
    for item in converted:
        item_id = str(getattr(item, "window_id", ""))
        if not item_id:
            raise WorkerError(
                "PIPELINE_ADAPTER_RESULT_INVALID",
                f"{label} adapter returned an item without a window id",
            )
        if item_id in by_id:
            raise WorkerError(
                "PIPELINE_ADAPTER_RESULT_INVALID",
                f"{label} adapter returned a duplicate window",
            )
        by_id[item_id] = item
    if set(by_id) != expected_ids:
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{label} adapter must return exactly one result per requested window",
            details={
                "missingCount": len(expected_ids - set(by_id)),
                "unexpectedCount": len(set(by_id) - expected_ids),
            },
        )
    return by_id


def _coerce_review_batch(
    values: Any,
    *,
    expected_ids: set[str],
    label: str,
) -> dict[str, ReviewProposal]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
    ):
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{label} adapter must return an array, not {type(values).__name__}",
        )
    output: dict[str, ReviewProposal] = {}
    try:
        for index, item in enumerate(values):
            if isinstance(item, ReviewProposal):
                proposal = item
            elif isinstance(item, Mapping):
                proposal = ReviewProposal.from_mapping(item)
            else:
                raise TypeError(
                    f"{label}[{index}] has invalid type {type(item).__name__}"
                )
            if proposal.segment_id in output:
                raise ValueError("duplicate review proposal")
            output[proposal.segment_id] = proposal
    except (TypeError, ValueError, WorkerError) as exc:
        if isinstance(exc, WorkerError):
            raise
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{label} adapter returned malformed data",
            details={"exceptionType": type(exc).__name__},
        ) from exc
    if set(output) != expected_ids:
        raise WorkerError(
            "PIPELINE_ADAPTER_RESULT_INVALID",
            f"{label} adapter must return exactly one result per requested segment",
            details={
                "missingCount": len(expected_ids - set(output)),
                "unexpectedCount": len(set(output) - expected_ids),
            },
        )
    return output


class SpeakerPipeline:
    """High-throughput cascade with quality-preserving selective escalation."""

    adapter_id = "offline-dynamic-speaker-cascade"
    version = "2.15.0"

    def __init__(
        self,
        *,
        preparation_adapter: AudioPreparationAdapter,
        asr_adapter: BatchAsrAdapter,
        embedding_adapter: BatchEmbeddingAdapter,
        overlap_adapter: OverlapDetectionAdapter | None = None,
        separation_adapter: SpeechSeparationAdapter | None = None,
        secondary_adapter: EscalationAdapter | None = None,
        review_adapter: EscalationAdapter | None = None,
        pyannote_adapter: EscalationAdapter | None = None,
        cache: StageCache | None = None,
        config: SpeakerPipelineConfig | None = None,
    ) -> None:
        if secondary_adapter is not None and review_adapter is not None:
            raise ValueError(
                "provide secondary_adapter only; review_adapter is a compatibility alias"
            )
        self.preparation_adapter = preparation_adapter
        self.asr_adapter = asr_adapter
        self.embedding_adapter = embedding_adapter
        self.overlap_adapter = overlap_adapter or UnavailableOverlapAdapter()
        self.separation_adapter = separation_adapter
        self.secondary_adapter = secondary_adapter or review_adapter
        self.pyannote_adapter = pyannote_adapter
        self.cache = cache or InMemoryStageCache()
        self.config = config or SpeakerPipelineConfig()
        if self.config.pyannote_mode == "fallback" and self.pyannote_adapter is None:
            raise ValueError(
                "pyannote_adapter is required when pyannote_mode is fallback"
            )
        if (
            self.config.pyannote_mode == "fallback"
            and self.secondary_adapter is None
        ):
            raise ValueError(
                "secondary_adapter is required when pyannote_mode is fallback"
            )
        if (
            self.config.overlap_recovery_mode == "guarded"
            and self.separation_adapter is None
        ):
            raise ValueError(
                "separation_adapter is required when overlap recovery is guarded"
            )
        if self.pyannote_adapter is not None and getattr(
            self.pyannote_adapter, "telemetry_enabled", False
        ) is not False:
            raise ValueError("pyannote telemetry must remain disabled")

    def _release_after_success(self, adapter: Any) -> None:
        if self.config.model_residency == "stage":
            _release_adapter_resources(adapter)

    def release_resources(self) -> None:
        """Release every unique adapter owned by this worker."""

        adapters = (
            self.preparation_adapter,
            self.asr_adapter,
            self.embedding_adapter,
            self.overlap_adapter,
            self.separation_adapter,
            self.secondary_adapter,
            self.pyannote_adapter,
        )
        seen: set[int] = set()
        first_error: Exception | None = None
        for adapter in adapters:
            if adapter is None or id(adapter) in seen:
                continue
            seen.add(id(adapter))
            try:
                _release_adapter_resources(adapter)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def _candidate_time_bounds(
        candidate_ids: Sequence[str],
        items_by_id: Mapping[str, Any],
    ) -> tuple[int | None, int | None]:
        if not candidate_ids:
            return None, None
        items = [items_by_id[candidate_id] for candidate_id in candidate_ids]
        return (
            min(int(item.start_ms) for item in items),
            max(int(item.end_ms) for item in items),
        )

    @staticmethod
    def _review_stage_summary(
        proposals: Mapping[str, ReviewProposal],
    ) -> tuple[dict[str, float], float | None]:
        if not proposals:
            return {}, None
        resource: dict[str, float] = {}
        for key in ("ramMb", "vramMb"):
            values = [
                float(proposal.resource[key])
                for proposal in proposals.values()
                if key in proposal.resource
            ]
            if values:
                resource[key] = max(values)
        confidence = sum(
            proposal.confidence for proposal in proposals.values()
        ) / len(proposals)
        return resource, confidence

    @staticmethod
    def _assert_speaker_cardinality(
        *,
        segments: Sequence[TranscriptSegment],
        clusters: _ClusterResult,
        request: StartJobRequest,
        stage: str,
        expected_ids: set[str] | None = None,
    ) -> set[str]:
        policy = request.speaker_policy
        policy_valid = True
        if policy.mode is SpeakerCountMode.MANUAL:
            policy_valid = clusters.count == policy.manual_count
        elif policy.mode is SpeakerCountMode.HYBRID:
            assert policy.minimum is not None and policy.maximum is not None
            policy_valid = policy.minimum <= clusters.count <= policy.maximum
        else:
            policy_valid = 1 <= clusters.count <= len(segments)

        resolved_ids = {
            f"speaker-{index}" for index in range(1, clusters.count + 1)
        }
        required_ids = set(expected_ids or resolved_ids)
        actual_ids = {segment.speaker_id for segment in segments}
        if (
            not policy_valid
            or required_ids != resolved_ids
            or actual_ids != required_ids
            or len(actual_ids) != clusters.count
        ):
            raise WorkerError(
                "SPEAKER_CARDINALITY_INVARIANT_VIOLATION",
                "speaker cardinality changed outside the resolved count policy",
                details={
                    "stage": stage,
                    "mode": policy.mode.value,
                    "resolvedCount": clusters.count,
                    "requiredSpeakerIds": sorted(required_ids),
                    "actualSpeakerIds": sorted(actual_ids),
                },
            )
        return actual_ids

    def _cache_item(
        self,
        stage: str,
        key: str,
        converter: Callable[[Any], Any],
    ) -> tuple[Any | None, bool, bool]:
        read = self.cache.read(stage, key)
        if not read.hit:
            return None, False, read.corrupted
        try:
            return converter(read.value), True, False
        except (TypeError, ValueError, WorkerError):
            return None, False, True

    def _prepare(
        self,
        request: StartJobRequest,
        context: AdapterContext,
        source_fingerprint: str,
    ) -> tuple[PreparedAudio, dict[str, dict[str, int]], float]:
        identity = _adapter_identity(self.preparation_adapter)
        normalize_key = _digest(
            {
                "stage": "normalize",
                "source": source_fingerprint,
                "adapter": identity,
                "profile": self.config.normalization_profile,
            }
        )
        vad_key = _digest(
            {
                "stage": "vad",
                "normalizationKey": normalize_key,
                "adapter": identity,
            }
        )
        boundary_key = _digest(
            {
                "stage": "boundary",
                "vadKey": vad_key,
                "adapter": identity,
            }
        )
        started = time.perf_counter()

        def normalization_from_mapping(value: Any) -> dict[str, Any]:
            if not isinstance(value, Mapping):
                raise ValueError("normalization cache entry must be an object")
            fingerprint = str(value.get("sourceFingerprint") or "")
            profile = str(value.get("normalizationProfile") or "")
            duration_ms = int(value.get("durationMs"))
            content_key = str(value.get("contentKey") or "")
            audio_sha256 = value.get("audioSha256")
            if (
                fingerprint != source_fingerprint
                or profile != self.config.normalization_profile
                or duration_ms < 1
                or len(content_key) != 64
                or (
                    audio_sha256 is not None
                    and (
                        not isinstance(audio_sha256, str)
                        or len(audio_sha256) != 64
                    )
                )
            ):
                raise ValueError("normalization cache entry is inconsistent")
            return {
                "sourceFingerprint": fingerprint,
                "normalizationProfile": profile,
                "durationMs": duration_ms,
                "contentKey": content_key,
                **(
                    {"audioSha256": audio_sha256}
                    if audio_sha256 is not None
                    else {}
                ),
            }

        def vad_from_mapping(value: Any) -> dict[str, Any]:
            if not isinstance(value, Mapping):
                raise ValueError("VAD cache entry must be an object")
            if value.get("normalizationKey") != normalize_key:
                raise ValueError("VAD cache normalization key is inconsistent")
            raw_windows = value.get("windows")
            if not isinstance(raw_windows, list) or not raw_windows:
                raise ValueError("VAD cache must contain windows")
            windows = tuple(SpeechWindow.from_mapping(item) for item in raw_windows)
            if len({item.window_id for item in windows}) != len(windows):
                raise ValueError("VAD cache window IDs must be unique")
            content_key = str(value.get("contentKey") or "")
            if len(content_key) != 64:
                raise ValueError("VAD cache content key is invalid")
            return {
                "normalizationKey": normalize_key,
                "windows": [item.as_dict() for item in windows],
                "contentKey": content_key,
            }

        normalization, normalize_hit, normalize_corrupt = self._cache_item(
            "normalize", normalize_key, normalization_from_mapping
        )
        vad, vad_hit, vad_corrupt = self._cache_item(
            "vad", vad_key, vad_from_mapping
        )
        prepared, boundary_hit, boundary_corrupt = self._cache_item(
            "boundary", boundary_key, PreparedAudio.from_mapping
        )
        if boundary_hit and (
            prepared.source_fingerprint != source_fingerprint
            or prepared.normalization_profile != self.config.normalization_profile
            or (
                prepared.audio_path is not None
                and not Path(prepared.audio_path).is_file()
            )
        ):
            prepared = None
            boundary_hit = False
            boundary_corrupt = True

        if prepared is None:
            context.raise_if_cancelled()
            raw = self.preparation_adapter.prepare(
                request.source_path,
                normalization_profile=self.config.normalization_profile,
                context=context,
            )
            try:
                prepared = (
                    raw
                    if isinstance(raw, PreparedAudio)
                    else PreparedAudio.from_mapping(raw)
                )
            except (TypeError, ValueError, WorkerError) as exc:
                if isinstance(exc, WorkerError):
                    raise
                raise WorkerError(
                    "PIPELINE_ADAPTER_RESULT_INVALID",
                    "preparation adapter returned malformed data",
                    details={"exceptionType": type(exc).__name__},
                ) from exc
            if (
                prepared.source_fingerprint != source_fingerprint
                or prepared.normalization_profile
                != self.config.normalization_profile
            ):
                raise WorkerError(
                    "PREPARATION_FINGERPRINT_MISMATCH",
                    "prepared audio does not match the requested source/profile",
                )
            self.cache.write("boundary", boundary_key, prepared.as_dict())
        assert prepared is not None

        expected_normalization = {
            "sourceFingerprint": prepared.source_fingerprint,
            "normalizationProfile": prepared.normalization_profile,
            "durationMs": prepared.duration_ms,
            "contentKey": _digest(
                {
                    "source": prepared.source_fingerprint,
                    "profile": prepared.normalization_profile,
                    "durationMs": prepared.duration_ms,
                }
            ),
        }
        if prepared.audio_path is not None:
            expected_normalization["audioSha256"] = _sha256_file(
                Path(prepared.audio_path), context
            )
        if normalization != expected_normalization:
            if normalize_hit:
                normalize_hit = False
                normalize_corrupt = True
            self.cache.write("normalize", normalize_key, expected_normalization)

        expected_vad = {
            "normalizationKey": normalize_key,
            "windows": [
                {
                    "id": window.window_id,
                    "startMs": window.start_ms,
                    "endMs": window.end_ms,
                    "boundaryConflict": False,
                    "metadata": {},
                }
                for window in prepared.windows
            ],
        }
        expected_vad["contentKey"] = _digest(expected_vad)
        if vad != expected_vad:
            if vad_hit:
                vad_hit = False
                vad_corrupt = True
            self.cache.write("vad", vad_key, expected_vad)

        assert prepared is not None
        return (
            prepared,
            {
                "normalize": {
                    "requests": 1,
                    "hits": int(normalize_hit),
                    "misses": int(not normalize_hit),
                    "recomputations": int(normalize_corrupt),
                },
                "vad": {
                    "requests": 1,
                    "hits": int(vad_hit),
                    "misses": int(not vad_hit),
                    "recomputations": int(vad_corrupt),
                },
                "boundary": {
                    "requests": 1,
                    "hits": int(boundary_hit),
                    "misses": int(not boundary_hit),
                    "recomputations": int(boundary_corrupt),
                },
            },
            (time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _validate_refined_audio(
        original: PreparedAudio,
        refined: PreparedAudio,
    ) -> PreparedAudio:
        """Fail closed unless refinement is a lock-preserving exact partition."""

        immutable_fields = {
            "sourceFingerprint": (
                original.source_fingerprint,
                refined.source_fingerprint,
            ),
            "normalizationProfile": (
                original.normalization_profile,
                refined.normalization_profile,
            ),
            "durationMs": (original.duration_ms, refined.duration_ms),
            "audioPath": (original.audio_path, refined.audio_path),
            "referenceTurns": (
                original.reference_turns,
                refined.reference_turns,
            ),
        }
        changed = [
            name
            for name, (before, after) in immutable_fields.items()
            if before != after
        ]
        if changed:
            raise WorkerError(
                "PIPELINE_ADAPTER_RESULT_INVALID",
                "speaker-change refinement changed immutable prepared audio fields",
                details={"changedFields": changed},
            )

        refined_index = 0
        for source in original.windows:
            cursor = source.start_ms
            child_count = 0
            while (
                refined_index < len(refined.windows)
                and refined.windows[refined_index].start_ms < source.end_ms
            ):
                child = refined.windows[refined_index]
                if (
                    child.start_ms != cursor
                    or child.end_ms > source.end_ms
                    or child.locked_speaker_id != source.locked_speaker_id
                    or child.boundary_conflict != source.boundary_conflict
                ):
                    raise WorkerError(
                        "PIPELINE_ADAPTER_RESULT_INVALID",
                        "speaker-change refinement must preserve source "
                        "coverage, boundary evidence, and human locks",
                        details={
                            "sourceWindowId": source.window_id,
                            "refinedWindowId": child.window_id,
                        },
                    )
                cursor = child.end_ms
                child_count += 1
                refined_index += 1
            if child_count < 1 or cursor != source.end_ms:
                raise WorkerError(
                    "PIPELINE_ADAPTER_RESULT_INVALID",
                    "speaker-change refinement must exactly partition every "
                    "original speech window",
                    details={"sourceWindowId": source.window_id},
                )
        if refined_index != len(refined.windows):
            raise WorkerError(
                "PIPELINE_ADAPTER_RESULT_INVALID",
                "speaker-change refinement produced windows outside the "
                "original speech timeline",
            )
        original_by_id = {
            window.window_id: window for window in original.windows
        }
        for identity_window in refined.speaker_identity_windows:
            source = original_by_id.get(identity_window.source_window_id)
            if (
                source is None
                or identity_window.start_ms < source.start_ms
                or identity_window.end_ms > source.end_ms
            ):
                raise WorkerError(
                    "PIPELINE_ADAPTER_RESULT_INVALID",
                    "contextual speaker identity evidence must remain inside "
                    "its source VAD window",
                    details={
                        "identityWindowId": identity_window.window_id,
                        "sourceWindowId": identity_window.source_window_id,
                    },
                )
        return refined

    def _refinement_cache_identity(
        self,
        prepared: PreparedAudio,
    ) -> dict[str, Any]:
        adapter = self.embedding_adapter
        identity_provider = getattr(adapter, "refinement_identity", None)
        config_provider = getattr(adapter, "refinement_config", None)
        try:
            refinement_identity = (
                _invoke_refinement_identity(identity_provider, prepared)
                if callable(identity_provider)
                else identity_provider
            )
            refinement_config = (
                _invoke_refinement_identity(config_provider, prepared)
                if callable(config_provider)
                else config_provider
            )
            return {
                "identity": _cache_identity_value(refinement_identity),
                "config": _cache_identity_value(refinement_config),
            }
        except (TypeError, ValueError) as exc:
            raise WorkerError(
                "PIPELINE_ADAPTER_RESULT_INVALID",
                "speaker-change refinement cache identity is invalid",
                details={"adapter": _adapter_identity(adapter)},
            ) from exc

    def _refinement_stage(
        self,
        prepared: PreparedAudio,
        context: AdapterContext,
        metrics: PipelineMetricsCollector,
    ) -> PreparedAudio:
        """Run optional speaker-change refinement behind an independent cache."""

        started = time.perf_counter()
        refine = getattr(self.embedding_adapter, "refine_windows", None)
        if not callable(refine):
            metrics.record_cache(
                _SPEAKER_CHANGE_REFINEMENT_STAGE,
                requests=0,
                hits=0,
                misses=0,
                recomputations=0,
            )
            metrics.record_stage(
                _SPEAKER_CHANGE_REFINEMENT_STAGE,
                (time.perf_counter() - started) * 1000.0,
            )
            return prepared

        key = _digest(
            {
                "stage": _SPEAKER_CHANGE_REFINEMENT_STAGE,
                "source": prepared.source_fingerprint,
                "normalizationProfile": prepared.normalization_profile,
                "originalWindows": [
                    window.as_dict() for window in prepared.windows
                ],
                "adapter": _adapter_identity(self.embedding_adapter),
                "refinement": self._refinement_cache_identity(prepared),
            }
        )

        def from_cache(value: Any) -> PreparedAudio:
            cached = PreparedAudio.from_mapping(value)
            return self._validate_refined_audio(prepared, cached)

        refined, hit, corrupted = self._cache_item(
            _SPEAKER_CHANGE_REFINEMENT_STAGE,
            key,
            from_cache,
        )
        if not hit:
            context.raise_if_cancelled()
            try:
                raw = refine(prepared, context)
                computed = (
                    raw
                    if isinstance(raw, PreparedAudio)
                    else PreparedAudio.from_mapping(raw)
                )
            except (TypeError, ValueError) as exc:
                raise WorkerError(
                    "PIPELINE_ADAPTER_RESULT_INVALID",
                    "speaker-change refinement returned invalid prepared audio",
                    details={
                        "adapter": _adapter_identity(
                            self.embedding_adapter
                        )
                    },
                ) from exc
            refined = self._validate_refined_audio(prepared, computed)
            self.cache.write(
                _SPEAKER_CHANGE_REFINEMENT_STAGE,
                key,
                refined.as_dict(),
            )
        metrics.record_cache(
            _SPEAKER_CHANGE_REFINEMENT_STAGE,
            requests=1,
            hits=int(hit),
            misses=int(not hit),
            recomputations=int(corrupted),
        )
        metrics.record_stage(
            _SPEAKER_CHANGE_REFINEMENT_STAGE,
            (time.perf_counter() - started) * 1000.0,
        )
        assert refined is not None
        return refined

    def _partition_for_speaker_count_policy(
        self,
        prepared: PreparedAudio,
        request: StartJobRequest,
        metrics: PipelineMetricsCollector,
    ) -> PreparedAudio:
        """Create enough contiguous evidence windows for speaker counting.

        A VAD window is a continuous speech region, not a speaker turn. Manual
        and hybrid policies therefore cannot use the number of VAD windows as
        an upper bound on the number of speakers. This deterministic partition
        only creates acoustic evidence windows; it preserves the source turn
        identifier and does not assert speaker-change boundaries.
        """

        started = time.perf_counter()
        policy = request.speaker_policy
        constrained_minimum: int | None = None
        if policy.mode is SpeakerCountMode.MANUAL:
            constrained_minimum = policy.manual_count
        elif policy.mode is SpeakerCountMode.HYBRID:
            constrained_minimum = policy.minimum
        if constrained_minimum is None:
            desired = sum(
                max(
                    1,
                    math.ceil(
                        (window.end_ms - window.start_ms)
                        / _AUTO_SPEAKER_EVIDENCE_WINDOW_MS
                    ),
                )
                for window in prepared.windows
            )
            required = min(self.config.max_clustering_windows, desired)
        else:
            required = constrained_minimum
        if len(prepared.windows) >= required:
            metrics.record_stage(
                _SPEAKER_COUNT_PARTITION_STAGE,
                (time.perf_counter() - started) * 1000.0,
            )
            return prepared

        allocations = [1] * len(prepared.windows)
        durations = [
            window.end_ms - window.start_ms for window in prepared.windows
        ]
        capacity_limited = False
        while sum(allocations) < required:
            candidates = [
                index
                for index, duration in enumerate(durations)
                if duration
                >= (allocations[index] + 1)
                * _MIN_SPEAKER_COUNT_PARTITION_MS
            ]
            if not candidates:
                if constrained_minimum is None:
                    capacity_limited = True
                    break
                raise WorkerError(
                    "SPEAKER_COUNT_AUDIO_TOO_SHORT",
                    "speech duration is too short to create independent "
                    "speaker evidence windows",
                    details={
                        "speakerCountMinimum": constrained_minimum,
                        "targetEvidenceWindowCount": required,
                        "speechWindows": len(prepared.windows),
                        "speechDurationMs": sum(durations),
                        "minimumPartitionMs": (
                            _MIN_SPEAKER_COUNT_PARTITION_MS
                        ),
                    },
                )
            selected = max(
                candidates,
                key=lambda index: (
                    durations[index] / (allocations[index] + 1),
                    -prepared.windows[index].start_ms,
                    prepared.windows[index].window_id,
                ),
            )
            allocations[selected] += 1

        partitioned: list[SpeechWindow] = []
        for source, partition_count in zip(prepared.windows, allocations):
            if partition_count == 1:
                metadata = dict(source.metadata)
                metadata["speakerCountPartition"] = {
                    "method": (
                        "auto-acoustic-contiguous-partition-v1"
                        if constrained_minimum is None
                        else "policy-minimum-contiguous-partition-v1"
                    ),
                    "sourceWindowId": source.window_id,
                    "requestedMinimum": constrained_minimum,
                    "targetEvidenceWindowCount": required,
                    "achievedEvidenceWindowCount": sum(allocations),
                    "partitionCount": 1,
                    "capacityLimited": capacity_limited,
                    "reviewRequired": True,
                    "reasonCode": (
                        "AUTO_COUNT_EVIDENCE_CAPACITY_LIMITED"
                        if capacity_limited
                        else "AUTO_COUNT_REQUIRES_SUBWINDOW_EVIDENCE"
                        if constrained_minimum is None
                        else "SPEAKER_COUNT_REQUIRES_SUBWINDOW_EVIDENCE"
                    ),
                }
                partitioned.append(replace(source, metadata=metadata))
                continue
            boundaries, boundary_selection = (
                self._speaker_count_partition_boundaries(
                    source,
                    partition_count=partition_count,
                )
            )
            source_turn_id = (
                str(source.metadata["turnId"]).strip()
                if isinstance(source.metadata.get("turnId"), str)
                and str(source.metadata["turnId"]).strip()
                else f"turn:{source.window_id}:{source.start_ms}-{source.end_ms}"
            )
            for index, (start_ms, end_ms) in enumerate(
                zip(boundaries, boundaries[1:]),
                start=1,
            ):
                metadata = dict(source.metadata)
                metadata.update(
                    {
                        "sourceVadWindowId": metadata.get(
                            "sourceVadWindowId",
                            source.window_id,
                        ),
                        "turnId": source_turn_id,
                        "speakerCountPartition": {
                            "method": (
                                "auto-acoustic-contiguous-partition-v1"
                                if constrained_minimum is None
                                else "policy-minimum-contiguous-partition-v1"
                            ),
                            "sourceWindowId": source.window_id,
                            "requestedMinimum": constrained_minimum,
                            "targetEvidenceWindowCount": required,
                            "achievedEvidenceWindowCount": sum(allocations),
                            "partitionCount": partition_count,
                            "capacityLimited": capacity_limited,
                            "boundarySelectionMethod": boundary_selection[
                                "method"
                            ],
                            "selectedAcousticBoundaryCount": (
                                boundary_selection[
                                    "selectedAcousticBoundaryCount"
                                ]
                            ),
                            "selectedProposalIds": boundary_selection[
                                "selectedProposalIds"
                            ],
                            "reviewRequired": True,
                            "reasonCode": (
                                "AUTO_COUNT_EVIDENCE_CAPACITY_LIMITED"
                                if capacity_limited
                                else "AUTO_COUNT_REQUIRES_SUBWINDOW_EVIDENCE"
                                if constrained_minimum is None
                                else (
                                    "SPEAKER_COUNT_REQUIRES_"
                                    "SUBWINDOW_EVIDENCE"
                                )
                            ),
                        },
                    }
                )
                partitioned.append(
                    SpeechWindow(
                        window_id=(
                            f"{source.window_id}.cardinality-{index:02d}"
                        ),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        boundary_conflict=source.boundary_conflict,
                        locked_speaker_id=source.locked_speaker_id,
                        metadata=metadata,
                    )
                )

        metrics.record_stage(
            _SPEAKER_COUNT_PARTITION_STAGE,
            (time.perf_counter() - started) * 1000.0,
        )
        metrics.set_policy(
            speakerCountPartitionApplied=(
                len(partitioned) > len(prepared.windows)
            ),
            speakerCountPartitionMode=policy.mode.value,
            speakerCountPartitionSourceWindows=len(prepared.windows),
            speakerCountPartitionTargetEvidenceWindows=required,
            speakerCountPartitionEvidenceWindows=len(partitioned),
            speakerCountPartitionCapacityLimited=capacity_limited,
        )
        return replace(prepared, windows=tuple(partitioned))

    @staticmethod
    def _speaker_count_partition_boundaries(
        source: SpeechWindow,
        *,
        partition_count: int,
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        """Prefer reviewable acoustic change points over arbitrary equal cuts.

        These boundaries only define independent embedding samples. They keep
        the source turn ID and remain review-required, so a change proposal is
        never promoted to a confirmed speaker turn by this sampling step.
        """

        duration = source.end_ms - source.start_ms
        uniform = [
            source.start_ms + duration * index // partition_count
            for index in range(partition_count + 1)
        ]
        refinement = source.metadata.get("speakerChangeRefinement")
        candidates: list[dict[str, Any]] = []
        if isinstance(refinement, Mapping):
            plans = refinement.get("plans")
            if isinstance(plans, Mapping):
                for resolution, raw_plan in sorted(plans.items()):
                    if not isinstance(raw_plan, Mapping):
                        continue
                    proposals = raw_plan.get("proposals")
                    if not isinstance(proposals, Sequence) or isinstance(
                        proposals,
                        (str, bytes, bytearray),
                    ):
                        continue
                    for proposal in proposals:
                        if not isinstance(proposal, Mapping):
                            continue
                        split_ms = proposal.get("splitMs")
                        if (
                            isinstance(split_ms, bool)
                            or not isinstance(split_ms, int)
                            or not source.start_ms < split_ms < source.end_ms
                            or proposal.get("overlapRisk") is True
                        ):
                            continue

                        def score(name: str) -> float:
                            value = proposal.get(name)
                            if (
                                isinstance(value, bool)
                                or not isinstance(value, (int, float))
                                or not math.isfinite(float(value))
                            ):
                                return 0.0
                            return float(value)

                        candidates.append(
                            {
                                "splitMs": split_ms,
                                "proposalId": str(
                                    proposal.get("proposalId") or ""
                                ),
                                "resolution": str(resolution),
                                "automatic": (
                                    proposal.get("applyAutomatically") is True
                                ),
                                "changeScore": score("changeScore"),
                                "acousticConfidence": score(
                                    "acousticConfidence"
                                ),
                                "boundaryMarkerConfidence": score(
                                    "boundaryMarkerConfidence"
                                ),
                            }
                        )

        selected_ids: list[str] = []
        selected_acoustic = 0
        boundaries = [source.start_ms]
        used: set[tuple[str, str]] = set()
        for index in range(1, partition_count):
            ideal = uniform[index]
            cell_start = (uniform[index - 1] + ideal) // 2
            cell_end = (ideal + uniform[index + 1]) // 2
            minimum = max(
                boundaries[-1] + _MIN_SPEAKER_COUNT_PARTITION_MS,
                cell_start,
            )
            maximum = min(
                source.end_ms
                - (partition_count - index)
                * _MIN_SPEAKER_COUNT_PARTITION_MS,
                cell_end,
            )
            eligible = [
                candidate
                for candidate in candidates
                if (
                    minimum <= candidate["splitMs"] <= maximum
                    and (
                        candidate["resolution"],
                        candidate["proposalId"],
                    )
                    not in used
                )
            ]
            if eligible:
                selected = max(
                    eligible,
                    key=lambda candidate: (
                        candidate["automatic"],
                        candidate["acousticConfidence"],
                        candidate["changeScore"],
                        candidate["boundaryMarkerConfidence"],
                        -abs(candidate["splitMs"] - ideal),
                        candidate["resolution"] == "context",
                        candidate["proposalId"],
                    ),
                )
                boundary = int(selected["splitMs"])
                used.add(
                    (selected["resolution"], selected["proposalId"])
                )
                if selected["proposalId"]:
                    selected_ids.append(selected["proposalId"])
                selected_acoustic += 1
            else:
                boundary = max(minimum, min(ideal, maximum))
            boundaries.append(boundary)
        boundaries.append(source.end_ms)
        if selected_acoustic == partition_count - 1:
            method = "speaker-change-proposal-guided-v1"
        elif selected_acoustic:
            method = "speaker-change-proposal-mixed-v1"
        else:
            method = "uniform-duration-fallback-v1"
        return tuple(boundaries), {
            "method": method,
            "selectedAcousticBoundaryCount": selected_acoustic,
            "selectedProposalIds": selected_ids,
        }

    @staticmethod
    def _join_aligned_token_text(tokens: Sequence[Mapping[str, Any]]) -> str:
        text = " ".join(
            str(token.get("text") or "").strip()
            for token in tokens
            if str(token.get("text") or "").strip()
        )
        text = re.sub(r"\s+([,.;:!?，。！？；：])", r"\1", text)
        return re.sub(
            r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])",
            "",
            text,
        ).strip()

    def _project_asr_to_speaker_windows(
        self,
        *,
        source_windows: Sequence[SpeechWindow],
        source_hypotheses: Sequence[AsrHypothesis],
        target_windows: Sequence[SpeechWindow],
        metrics: PipelineMetricsCollector,
    ) -> list[AsrHypothesis]:
        """Project one bounded ASR pass onto finer acoustic evidence windows."""

        started = time.perf_counter()
        if len(source_windows) != len(source_hypotheses):
            raise WorkerError(
                "ASR_PROJECTION_INVALID",
                "ASR projection requires one source result per language window",
            )
        source_by_id = {
            window.window_id: (window, hypothesis)
            for window, hypothesis in zip(source_windows, source_hypotheses)
        }
        projected: list[AsrHypothesis] = []
        for target in target_windows:
            partition = target.metadata.get("speakerCountPartition")
            if not isinstance(partition, Mapping):
                source = source_by_id.get(target.window_id)
                if source is None:
                    raise WorkerError(
                        "ASR_PROJECTION_INVALID",
                        "unpartitioned speaker window has no ASR source",
                        details={"windowId": target.window_id},
                    )
                projected.append(source[1])
                continue
            source_id = partition.get("sourceWindowId")
            if not isinstance(source_id, str) or source_id not in source_by_id:
                raise WorkerError(
                    "ASR_PROJECTION_INVALID",
                    "speaker evidence window has no matching ASR source",
                    details={"windowId": target.window_id},
                )
            source_window, source_hypothesis = source_by_id[source_id]
            raw_timestamps = source_hypothesis.evidence.get("timestamps")
            if not isinstance(raw_timestamps, list) or not raw_timestamps:
                raise WorkerError(
                    "ASR_TIMESTAMPS_REQUIRED_FOR_SPEAKER_PARTITION",
                    "speaker evidence partition requires forced-alignment timestamps",
                    details={"sourceWindowId": source_id},
                )
            timestamps: list[dict[str, Any]] = []
            for item_index, item in enumerate(raw_timestamps):
                if (
                    not isinstance(item, Mapping)
                    or isinstance(item.get("startMs"), bool)
                    or isinstance(item.get("endMs"), bool)
                    or not isinstance(item.get("startMs"), int)
                    or not isinstance(item.get("endMs"), int)
                    or item["endMs"] < item["startMs"]
                    or item["startMs"] < source_window.start_ms
                    or item["endMs"] > source_window.end_ms
                ):
                    raise WorkerError(
                        "ASR_PROJECTION_INVALID",
                        "forced-alignment timestamp is outside its source window",
                        details={
                            "sourceWindowId": source_id,
                            "itemIndex": item_index,
                        },
                    )
                midpoint = (item["startMs"] + item["endMs"]) // 2
                if (
                    target.start_ms <= midpoint < target.end_ms
                    or midpoint == target.end_ms == source_window.end_ms
                ):
                    projected_item = dict(item)
                    clipped_start = max(target.start_ms, item["startMs"])
                    clipped_end = min(target.end_ms, item["endMs"])
                    if (
                        clipped_start != item["startMs"]
                        or clipped_end != item["endMs"]
                    ):
                        projected_item.update(
                            {
                                "sourceStartMs": item["startMs"],
                                "sourceEndMs": item["endMs"],
                                "timingProjection": (
                                    "clipped-to-target-window-v1"
                                ),
                            }
                        )
                    projected_item["startMs"] = clipped_start
                    projected_item["endMs"] = clipped_end
                    timestamps.append(projected_item)
            text = self._join_aligned_token_text(timestamps)
            evidence = dict(source_hypothesis.evidence)
            evidence.update(
                {
                    "timestamps": timestamps,
                    "asrProjection": {
                        "method": "forced-alignment-midpoint-v1",
                        "sourceWindowId": source_id,
                        "sourceStartMs": source_window.start_ms,
                        "sourceEndMs": source_window.end_ms,
                        "targetStartMs": target.start_ms,
                        "targetEndMs": target.end_ms,
                        "tokenCount": len(timestamps),
                        "clippedTokenCount": sum(
                            item.get("timingProjection")
                            == "clipped-to-target-window-v1"
                            for item in timestamps
                        ),
                        "sourceTextSha256": hashlib.sha256(
                            source_hypothesis.text.encode("utf-8")
                        ).hexdigest(),
                    },
                }
            )
            if "candidateSetSchemaVersion" in evidence:
                if text:
                    try:
                        evidence.update(
                            project_asr_candidate_set(
                                source_hypothesis.evidence,
                                source_text=source_hypothesis.text,
                                target_text=text,
                                target_tokens=timestamps,
                                target_window_id=target.window_id,
                                target_start_ms=target.start_ms,
                                target_end_ms=target.end_ms,
                            )
                        )
                    except AsrEvidenceError as exc:
                        raise WorkerError(
                            "ASR_CANDIDATE_EVIDENCE_INVALID",
                            "ASR candidate evidence could not be projected safely",
                            details={
                                "sourceWindowId": source_id,
                                "targetWindowId": target.window_id,
                                "reason": str(exc)[:240],
                            },
                        ) from exc
                else:
                    for key in ASR_CANDIDATE_SET_KEYS:
                        evidence.pop(key, None)
            if not text:
                evidence.update(
                    {
                        "disposition": _ASR_NON_LEXICAL_DISPOSITION,
                        "rejectionReason": (
                            "NO_ALIGNED_LEXICAL_TOKENS_IN_EVIDENCE_WINDOW"
                        ),
                    }
                )
                projected.append(
                    AsrHypothesis(
                        window_id=target.window_id,
                        text="",
                        confidence=0.0,
                        evidence=evidence,
                    )
                )
                continue
            evidence.pop("disposition", None)
            evidence.pop("rejectionReason", None)
            projected.append(
                AsrHypothesis(
                    window_id=target.window_id,
                    text=text,
                    normalized_text=text,
                    display_text=text,
                    confidence=source_hypothesis.confidence,
                    evidence=evidence,
                )
            )
        metrics.record_stage(
            "asr-projection",
            (time.perf_counter() - started) * 1000.0,
        )
        return projected

    @staticmethod
    def _clip_overlap_evidence(
        decision: OverlapDecision,
        *,
        target: SpeechWindow,
    ) -> OverlapDecision:
        evidence = dict(decision.evidence)

        def clipped_items(
            name: str,
            *,
            require_local_speaker: bool = False,
        ) -> list[dict[str, Any]] | None:
            raw_items = evidence.get(name)
            if not isinstance(raw_items, list):
                return None
            output: list[dict[str, Any]] = []
            for raw in raw_items:
                if not isinstance(raw, Mapping):
                    return None
                start_ms = raw.get("startMs")
                end_ms = raw.get("endMs")
                if (
                    isinstance(start_ms, bool)
                    or not isinstance(start_ms, int)
                    or isinstance(end_ms, bool)
                    or not isinstance(end_ms, int)
                    or end_ms <= start_ms
                ):
                    return None
                if require_local_speaker and (
                    not isinstance(raw.get("localSpeaker"), str)
                    or not str(raw["localSpeaker"]).strip()
                ):
                    return None
                clipped_start = max(target.start_ms, start_ms)
                clipped_end = min(target.end_ms, end_ms)
                if clipped_end <= clipped_start:
                    continue
                item = dict(raw)
                item["startMs"] = clipped_start
                item["endMs"] = clipped_end
                output.append(item)
            return output

        clipped_overlap = clipped_items("overlapIntervals")
        clipped_regular = clipped_items(
            "speakerTurns",
            require_local_speaker=True,
        )
        clipped_exclusive = clipped_items(
            "exclusiveSpeakerTurns",
            require_local_speaker=True,
        )
        if clipped_overlap is not None:
            evidence["overlapIntervals"] = clipped_overlap
            overlapping = bool(clipped_overlap)
        else:
            overlapping = decision.overlapping
        if clipped_regular is not None:
            evidence["speakerTurns"] = clipped_regular
        if clipped_exclusive is not None:
            evidence["exclusiveSpeakerTurns"] = clipped_exclusive
        evidence["turnProjection"] = {
            "method": "contextual-voiceprint-output-turn-v1",
            "sourceWindowId": decision.window_id,
            "targetWindowId": target.window_id,
            "targetStartMs": target.start_ms,
            "targetEndMs": target.end_ms,
        }
        return OverlapDecision(
            window_id=target.window_id,
            overlapping=overlapping,
            confidence=decision.confidence,
            secondary_speaker_hint=decision.secondary_speaker_hint,
            evidence=evidence,
        )

    def _project_contextual_speaker_turns(
        self,
        *,
        prepared: PreparedAudio,
        asr: Sequence[AsrHypothesis],
        embeddings: Sequence[EmbeddingRecord],
        overlap: Sequence[OverlapDecision],
        clusters: _ClusterResult,
        metrics: PipelineMetricsCollector,
    ) -> tuple[
        PreparedAudio,
        list[AsrHypothesis],
        list[EmbeddingRecord],
        list[OverlapDecision],
        _ClusterResult,
    ] | None:
        """Project cached multi-resolution voiceprints onto reviewable turns.

        Clustering still uses the original evidence windows.  This stage only
        reuses already-computed CAM++ identity embeddings and reviewable
        acoustic change proposals to derive output turns.
        """

        started = time.perf_counter()
        identity_windows = prepared.speaker_identity_windows
        if (
            not identity_windows
            or len(prepared.windows) != len(asr)
            or len(asr) != len(embeddings)
            or len(embeddings) != len(overlap)
            or len(overlap) != len(clusters.assignments)
        ):
            metrics.set_policy(
                contextualTurnProjectionApplied=False,
                contextualTurnProjectionReason="CONTEXT_IDENTITY_EVIDENCE_UNAVAILABLE",
            )
            return None

        normalized_embeddings = [
            _normalize(item.vector) for item in embeddings
        ]
        dimension = len(normalized_embeddings[0])
        if any(len(item) != dimension for item in normalized_embeddings):
            raise WorkerError(
                "CONTEXT_IDENTITY_EVIDENCE_INVALID",
                "speaker evidence embeddings have inconsistent dimensions",
            )
        members: list[list[tuple[float, ...]]] = [
            [] for _ in range(clusters.count)
        ]
        for vector, assignment in zip(
            normalized_embeddings,
            clusters.assignments,
        ):
            members[assignment].append(vector)
        if any(not cluster_members for cluster_members in members):
            raise WorkerError(
                "CONTEXT_IDENTITY_EVIDENCE_INVALID",
                "canonical speaker profile has no acoustic evidence",
            )
        centroids = tuple(
            _mean_vector(cluster_members, dimension)
            for cluster_members in members
        )

        contextual_scores: dict[str, tuple[float, ...]] = {}
        contextual_assignments: dict[str, int] = {}
        contextual_margins: dict[str, float] = {}
        identity_by_id: dict[str, SpeakerIdentityWindow] = {}
        for identity in identity_windows:
            if len(identity.vector) != dimension:
                raise WorkerError(
                    "CONTEXT_IDENTITY_EVIDENCE_INVALID",
                    "contextual and clustering embeddings have different dimensions",
                    details={"identityWindowId": identity.window_id},
                )
            vector = _normalize(identity.vector)
            scores = tuple(
                max(-1.0, min(1.0, _dot(vector, centroid)))
                for centroid in centroids
            )
            ranked = sorted(
                enumerate(scores),
                key=lambda item: (-item[1], item[0]),
            )
            contextual_scores[identity.window_id] = scores
            contextual_assignments[identity.window_id] = ranked[0][0]
            contextual_margins[identity.window_id] = (
                ranked[0][1] - ranked[1][1]
                if len(ranked) > 1
                else 2.0
            )
            identity_by_id[identity.window_id] = identity

        proposals_by_split: dict[int, dict[str, Any]] = {}
        seen_refinements: set[str] = set()
        for window in prepared.windows:
            refinement = window.metadata.get("speakerChangeRefinement")
            if not isinstance(refinement, Mapping):
                continue
            refinement_key = _digest(refinement)
            if refinement_key in seen_refinements:
                continue
            seen_refinements.add(refinement_key)
            plans = refinement.get("plans")
            if not isinstance(plans, Mapping):
                continue
            for resolution in ("fine", "context"):
                plan = plans.get(resolution)
                raw_proposals = (
                    plan.get("proposals")
                    if isinstance(plan, Mapping)
                    else None
                )
                if not isinstance(raw_proposals, Sequence) or isinstance(
                    raw_proposals,
                    (str, bytes, bytearray),
                ):
                    continue
                for raw in raw_proposals:
                    if not isinstance(raw, Mapping):
                        continue
                    split_ms = raw.get("splitMs")
                    support = raw.get("supportWindowIds")
                    if (
                        isinstance(split_ms, bool)
                        or not isinstance(split_ms, int)
                        or raw.get("overlapRisk") is True
                        or not isinstance(support, Sequence)
                        or isinstance(support, (str, bytes, bytearray))
                        or len(support) != 2
                        or any(
                            not isinstance(item, str)
                            or item not in identity_by_id
                            or identity_by_id[item].resolution != resolution
                            for item in support
                        )
                    ):
                        continue
                    left_id, right_id = str(support[0]), str(support[1])
                    if (
                        contextual_assignments[left_id]
                        == contextual_assignments[right_id]
                        or contextual_margins[left_id]
                        < self.config.low_margin_threshold
                        or contextual_margins[right_id]
                        < self.config.low_margin_threshold
                    ):
                        continue
                    change_score = raw.get("changeScore", 0.0)
                    acoustic_confidence = raw.get(
                        "acousticConfidence",
                        0.0,
                    )
                    if (
                        isinstance(change_score, bool)
                        or not isinstance(change_score, (int, float))
                        or not math.isfinite(float(change_score))
                        or isinstance(acoustic_confidence, bool)
                        or not isinstance(
                            acoustic_confidence,
                            (int, float),
                        )
                        or not math.isfinite(float(acoustic_confidence))
                    ):
                        continue
                    candidate = {
                        "splitMs": split_ms,
                        "proposalId": str(raw.get("proposalId") or ""),
                        "resolution": resolution,
                        "leftIdentityWindowId": left_id,
                        "rightIdentityWindowId": right_id,
                        "leftSpeakerId": (
                            f"speaker-{contextual_assignments[left_id] + 1}"
                        ),
                        "rightSpeakerId": (
                            f"speaker-{contextual_assignments[right_id] + 1}"
                        ),
                        "leftMargin": contextual_margins[left_id],
                        "rightMargin": contextual_margins[right_id],
                        "changeScore": float(change_score),
                        "acousticConfidence": float(acoustic_confidence),
                        "applyAutomatically": (
                            raw.get("applyAutomatically") is True
                        ),
                        "reviewStatus": str(
                            raw.get("reviewStatus") or "REVIEW_REQUIRED"
                        ),
                    }
                    current = proposals_by_split.get(split_ms)
                    if current is None or (
                        candidate["applyAutomatically"],
                        candidate["acousticConfidence"],
                        candidate["changeScore"],
                        candidate["resolution"] == "context",
                        candidate["proposalId"],
                    ) > (
                        current["applyAutomatically"],
                        current["acousticConfidence"],
                        current["changeScore"],
                        current["resolution"] == "context",
                        current["proposalId"],
                    ):
                        proposals_by_split[split_ms] = candidate

        if not proposals_by_split:
            metrics.set_policy(
                contextualTurnProjectionApplied=False,
                contextualTurnProjectionReason="NO_IDENTITY_CHANGING_CONTEXT_PROPOSALS",
            )
            return None

        source_hypotheses = {
            window.window_id: hypothesis
            for window, hypothesis in zip(prepared.windows, asr)
        }
        output_windows: list[SpeechWindow] = []
        output_scores: list[tuple[float, ...]] = []
        output_assignments: list[int] = []
        output_embeddings: list[EmbeddingRecord] = []
        output_overlap: list[OverlapDecision] = []
        selected_proposal_ids: list[str] = []
        added_boundary_count = 0

        for source_index, (
            source,
            source_embedding,
            source_overlap,
        ) in enumerate(zip(prepared.windows, embeddings, overlap)):
            hypothesis = source_hypotheses[source.window_id]
            raw_timestamps = hypothesis.evidence.get("timestamps")
            token_midpoints = sorted(
                (
                    (int(item["startMs"]) + int(item["endMs"])) // 2
                    for item in raw_timestamps
                    if isinstance(item, Mapping)
                    and isinstance(item.get("startMs"), int)
                    and not isinstance(item.get("startMs"), bool)
                    and isinstance(item.get("endMs"), int)
                    and not isinstance(item.get("endMs"), bool)
                )
                if isinstance(raw_timestamps, list)
                else ()
            )
            candidates = [
                proposal
                for split_ms, proposal in sorted(proposals_by_split.items())
                if (
                    source.start_ms
                    + _MIN_SPEAKER_COUNT_PARTITION_MS
                    <= split_ms
                    <= source.end_ms
                    - _MIN_SPEAKER_COUNT_PARTITION_MS
                )
            ]
            accepted: list[dict[str, Any]] = []
            cursor = source.start_ms
            if not source.locked_speaker_id:
                for proposal in candidates:
                    split_ms = int(proposal["splitMs"])
                    if (
                        split_ms - cursor
                        < _MIN_SPEAKER_COUNT_PARTITION_MS
                        or not any(
                            cursor <= midpoint < split_ms
                            for midpoint in token_midpoints
                        )
                        or not any(
                            split_ms <= midpoint <= source.end_ms
                            for midpoint in token_midpoints
                        )
                    ):
                        continue
                    accepted.append(proposal)
                    cursor = split_ms
            boundaries = (
                source.start_ms,
                *(int(item["splitMs"]) for item in accepted),
                source.end_ms,
            )
            added_boundary_count += len(accepted)
            selected_proposal_ids.extend(
                str(item["proposalId"])
                for item in accepted
                if item["proposalId"]
            )

            for child_index, (start_ms, end_ms) in enumerate(
                zip(boundaries, boundaries[1:]),
                start=1,
            ):
                child_id = (
                    source.window_id
                    if len(boundaries) == 2
                    else f"{source.window_id}.turn-{child_index:02d}"
                )
                weighted_scores_by_resolution = {
                    "fine": [0.0] * clusters.count,
                    "context": [0.0] * clusters.count,
                }
                total_weight_by_resolution = {
                    "fine": 0.0,
                    "context": 0.0,
                }
                supporting_ids: list[str] = []
                for identity in identity_windows:
                    overlap_ms = min(end_ms, identity.end_ms) - max(
                        start_ms,
                        identity.start_ms,
                    )
                    if overlap_ms <= 0:
                        continue
                    weight = overlap_ms * identity.confidence
                    total_weight_by_resolution[identity.resolution] += weight
                    supporting_ids.append(identity.window_id)
                    for score_index, score in enumerate(
                        contextual_scores[identity.window_id]
                    ):
                        weighted_scores_by_resolution[
                            identity.resolution
                        ][score_index] += weight * score
                resolution_scores = {
                    resolution: tuple(
                        value / total_weight_by_resolution[resolution]
                        for value in weighted_scores
                    )
                    for resolution, weighted_scores
                    in weighted_scores_by_resolution.items()
                    if total_weight_by_resolution[resolution] > 0.0
                }
                if resolution_scores:
                    score_row = tuple(
                        sum(
                            scores[score_index]
                            for scores in resolution_scores.values()
                        )
                        / len(resolution_scores)
                        for score_index in range(clusters.count)
                    )
                else:
                    score_row = clusters.scores[source_index]
                ranked = sorted(
                    enumerate(score_row),
                    key=lambda item: (-item[1], item[0]),
                )
                contextual_assignment = ranked[0][0]
                contextual_margin = (
                    ranked[0][1] - ranked[1][1]
                    if len(ranked) > 1
                    else 2.0
                )
                inherited = (
                    source.locked_speaker_id is not None
                    or contextual_margin < self.config.low_margin_threshold
                )
                assignment = (
                    clusters.assignments[source_index]
                    if inherited
                    else contextual_assignment
                )
                source_partition = source.metadata.get(
                    "speakerCountPartition"
                )
                partition = (
                    dict(source_partition)
                    if isinstance(source_partition, Mapping)
                    else {}
                )
                original_partition_source = partition.get("sourceWindowId")
                partition.update(
                    {
                        "sourceWindowId": source.window_id,
                        "identityEvidenceSourceWindowId": (
                            original_partition_source
                        ),
                    }
                )
                left_proposal = (
                    accepted[child_index - 2]
                    if child_index > 1
                    else None
                )
                right_proposal = (
                    accepted[child_index - 1]
                    if child_index <= len(accepted)
                    else None
                )
                projection = {
                    "method": "multiresolution-voiceprint-output-turn-v2",
                    "sourceWindowId": source.window_id,
                    "sourceSpeakerId": (
                        f"speaker-{clusters.assignments[source_index] + 1}"
                    ),
                    "projectedSpeakerId": f"speaker-{assignment + 1}",
                    "supportingIdentityWindowIds": supporting_ids,
                    "contextualScores": {
                        f"speaker-{index + 1}": round(score, 9)
                        for index, score in enumerate(score_row)
                    },
                    "resolutionScores": {
                        resolution: {
                            f"speaker-{index + 1}": round(score, 9)
                            for index, score in enumerate(scores)
                        }
                        for resolution, scores in sorted(
                            resolution_scores.items()
                        )
                    },
                    "contextualMargin": round(contextual_margin, 9),
                    "minimumMargin": self.config.low_margin_threshold,
                    "assignmentInherited": inherited,
                    "leftBoundaryProposalId": (
                        left_proposal["proposalId"]
                        if left_proposal is not None
                        else None
                    ),
                    "rightBoundaryProposalId": (
                        right_proposal["proposalId"]
                        if right_proposal is not None
                        else None
                    ),
                    "reviewStatus": "REVIEW_REQUIRED",
                    "sourceTextMutable": False,
                }
                metadata = {
                    **dict(source.metadata),
                    "turnId": (
                        f"turn:context:{source.window_id}:"
                        f"{start_ms}-{end_ms}"
                    ),
                    "speakerCountPartition": partition,
                    "speakerTurnProjection": projection,
                }
                child = SpeechWindow(
                    window_id=child_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    boundary_conflict=source.boundary_conflict,
                    locked_speaker_id=source.locked_speaker_id,
                    metadata=metadata,
                )
                output_windows.append(child)
                output_scores.append(score_row)
                output_assignments.append(assignment)
                output_embeddings.append(
                    EmbeddingRecord(
                        window_id=child_id,
                        vector=source_embedding.vector,
                        confidence=min(
                            source_embedding.confidence,
                            max(
                                identity_by_id[item].confidence
                                for item in supporting_ids
                            )
                            if supporting_ids
                            else source_embedding.confidence,
                        ),
                        evidence={
                            **dict(source_embedding.evidence),
                            "turnProjection": projection,
                        },
                    )
                )
                output_overlap.append(
                    self._clip_overlap_evidence(
                        source_overlap,
                        target=child,
                    )
                )

        if added_boundary_count < 1:
            metrics.set_policy(
                contextualTurnProjectionApplied=False,
                contextualTurnProjectionReason="NO_TOKEN_SAFE_CONTEXT_BOUNDARIES",
            )
            return None
        expected_assignments = set(clusters.assignments)
        if set(output_assignments) != expected_assignments:
            metrics.set_policy(
                contextualTurnProjectionApplied=False,
                contextualTurnProjectionReason="CARDINALITY_PRESERVATION_BLOCKED",
                contextualTurnProjectionCandidateBoundaries=added_boundary_count,
            )
            return None

        output_prepared = replace(
            prepared,
            windows=tuple(output_windows),
        )
        output_asr = self._project_asr_to_speaker_windows(
            source_windows=prepared.windows,
            source_hypotheses=asr,
            target_windows=output_windows,
            metrics=metrics,
        )
        if any(
            hypothesis.evidence.get("disposition")
            == _ASR_NON_LEXICAL_DISPOSITION
            for hypothesis in output_asr
        ):
            metrics.set_policy(
                contextualTurnProjectionApplied=False,
                contextualTurnProjectionReason="TOKEN_PROJECTION_EMPTY_CHILD",
            )
            return None

        metrics.record_stage(
            "contextual-turn-projection",
            (time.perf_counter() - started) * 1000.0,
        )
        metrics.set_policy(
            contextualTurnProjectionApplied=True,
            contextualTurnProjectionReason=(
                "ACOUSTIC_MULTIRESOLUTION_IDENTITY_CHANGE"
            ),
            contextualTurnProjectionSourceWindows=len(prepared.windows),
            contextualTurnProjectionOutputWindows=len(output_windows),
            contextualTurnProjectionAddedBoundaries=added_boundary_count,
            contextualTurnProjectionSelectedProposalIds="|".join(
                sorted(set(selected_proposal_ids))
            ),
        )
        return (
            output_prepared,
            output_asr,
            output_embeddings,
            output_overlap,
            replace(
                clusters,
                assignments=tuple(output_assignments),
                scores=tuple(output_scores),
            ),
        )

    def _window_stage(
        self,
        *,
        stage: str,
        prepared: PreparedAudio,
        windows: Sequence[SpeechWindow],
        adapter: Any,
        invoke: Callable[[Sequence[SpeechWindow]], Sequence[Any]],
        converter: Callable[[Any], Any],
        accepted_type: type,
        context: AdapterContext,
        metrics: PipelineMetricsCollector,
        cache_identity_material: Any = None,
    ) -> list[Any]:
        started = time.perf_counter()
        identity = _adapter_identity(adapter)
        values: dict[str, Any] = {}
        misses: list[SpeechWindow] = []
        keys: dict[str, str] = {}
        recomputations = 0
        for window in windows:
            key_material: dict[str, Any] = {
                "source": prepared.source_fingerprint,
                "profile": prepared.normalization_profile,
                "adapter": identity,
                "window": {
                    "id": window.window_id,
                    "startMs": window.start_ms,
                    "endMs": window.end_ms,
                },
            }
            if cache_identity_material is not None:
                key_material["request"] = _cache_identity_value(
                    cache_identity_material
                )
            key = _digest(key_material)
            keys[window.window_id] = key
            value, hit, corrupted = self._cache_item(stage, key, converter)
            if hit:
                values[window.window_id] = value
            else:
                misses.append(window)
                recomputations += int(corrupted)
        if misses:
            for offset in range(0, len(misses), self.config.max_batch_size):
                context.raise_if_cancelled()
                batch = misses[offset : offset + self.config.max_batch_size]
                raw_values = invoke(batch)
                computed = _coerce_batch(
                    raw_values,
                    expected_ids={window.window_id for window in batch},
                    converter=converter,
                    accepted_type=accepted_type,
                    label=stage,
                )
                for window_id, value in computed.items():
                    values[window_id] = value
                    self.cache.write(stage, keys[window_id], value.as_dict())
        metrics.record_cache(
            stage,
            requests=len(windows),
            hits=len(windows) - len(misses),
            misses=len(misses),
            recomputations=recomputations,
        )
        metrics.record_stage(
            stage, (time.perf_counter() - started) * 1000.0
        )
        return [values[window.window_id] for window in windows]

    def _clustering_stage(
        self,
        prepared: PreparedAudio,
        embeddings: Sequence[EmbeddingRecord],
        overlap: Sequence[OverlapDecision],
        request: StartJobRequest,
        metrics: PipelineMetricsCollector,
    ) -> _ClusterResult:
        started = time.perf_counter()
        pyannote_count_prior, pyannote_prior_audit = (
            self._derive_pyannote_count_prior(
                prepared=prepared,
                overlap=overlap,
                request=request,
            )
        )
        metrics.set_policy(
            pyannoteSpeakerCountPriorStatus=pyannote_prior_audit["status"],
            pyannoteSpeakerCountPriorObserved=(
                pyannote_prior_audit.get("observedCount")
            ),
            pyannoteSpeakerCountPriorUsed=pyannote_count_prior,
            pyannoteFullTimelineTurnsSha256=(
                pyannote_prior_audit.get("speakerTurnsSha256")
            ),
        )
        key = _digest(
            {
                "algorithm": _CLUSTER_SELECTION_METHOD,
                "source": prepared.source_fingerprint,
                "policy": request.speaker_policy.as_dict(),
                "config": self.config.as_dict(),
                "windows": [window.as_dict() for window in prepared.windows],
                "embeddings": [
                    {
                        "windowId": item.window_id,
                        "vectorHash": _digest(list(item.vector)),
                    }
                    for item in embeddings
                ],
                "pyannoteCountPrior": pyannote_prior_audit,
            }
        )
        value, hit, corrupted = self._cache_item(
            "clustering",
            key,
            lambda item: _ClusterResult.from_mapping(
                item, expected_rows=len(prepared.windows)
            ),
        )
        if not hit:
            value = _cluster(
                embeddings,
                prepared.windows,
                request,
                self.config,
                pyannote_count_prior=pyannote_count_prior,
            )
            self.cache.write("clustering", key, value.as_dict())
        metrics.record_cache(
            "clustering",
            requests=1,
            hits=int(hit),
            misses=int(not hit),
            recomputations=int(corrupted),
        )
        metrics.record_stage(
            "clustering", (time.perf_counter() - started) * 1000.0
        )
        assert value is not None
        prior_applied = any(
            item.startswith("PYANNOTE_FULL_TIMELINE_PRIOR:")
            for item in value.correction_path
        )
        observed_prior = pyannote_prior_audit.get("observedCount")
        prior_conflict = (
            isinstance(observed_prior, int)
            and not isinstance(observed_prior, bool)
            and value.count != observed_prior
        )
        metrics.set_policy(
            pyannoteSpeakerCountPriorApplied=prior_applied,
            pyannoteSpeakerCountPriorConflict=prior_conflict,
            speakerCountCorrectionPath="|".join(value.correction_path),
            speakerCountConfidenceReasons="|".join(
                value.confidence_reasons
            ),
        )
        return value

    def _derive_pyannote_count_prior(
        self,
        *,
        prepared: PreparedAudio,
        overlap: Sequence[OverlapDecision],
        request: StartJobRequest,
    ) -> tuple[int | None, dict[str, Any]]:
        """Validate one full-timeline pyannote observation for auto count."""

        audit: dict[str, Any] = {
            "status": "not-applicable",
            "provider": _adapter_identity(self.overlap_adapter),
        }
        if (
            self.config.pyannote_mode != "fallback"
            or audit["provider"]["id"] != "pyannote-community-1"
        ):
            return None, audit
        if request.speaker_policy.mode is not SpeakerCountMode.AUTO:
            audit["status"] = "ignored-non-auto-policy"
            return None, audit
        if len(overlap) != len(prepared.windows) or not overlap:
            audit["status"] = "invalid-window-coverage"
            return None, audit

        snapshots: list[tuple[Any, ...]] = []
        for window, decision in zip(prepared.windows, overlap):
            evidence = decision.evidence
            full = evidence.get("fullTimelineInference")
            if not isinstance(full, Mapping):
                audit["status"] = "missing-full-timeline-evidence"
                return None, audit
            start_ms = full.get("startMs")
            end_ms = full.get("endMs")
            turn_count = full.get("turnCount")
            local_count = full.get("localSpeakerCount")
            local_speakers = full.get("localSpeakers")
            turns_sha256 = full.get("speakerTurnsSha256")
            if (
                full.get("scope") != "full-normalized-timeline"
                or isinstance(start_ms, bool)
                or start_ms != 0
                or isinstance(end_ms, bool)
                or end_ms != prepared.duration_ms
                or isinstance(turn_count, bool)
                or not isinstance(turn_count, int)
                or isinstance(local_count, bool)
                or not isinstance(local_count, int)
                or not isinstance(local_speakers, list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in local_speakers
                )
                or local_speakers != sorted(set(local_speakers))
                or local_count != len(local_speakers)
                or local_count < 1
                or turn_count < local_count
                or not isinstance(turns_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", turns_sha256) is None
            ):
                audit["status"] = "invalid-full-timeline-evidence"
                return None, audit
            window_turns = evidence.get("speakerTurns")
            if not isinstance(window_turns, list):
                audit["status"] = "invalid-window-turn-evidence"
                return None, audit
            for turn in window_turns:
                if (
                    not isinstance(turn, Mapping)
                    or turn.get("localSpeaker") not in local_speakers
                    or isinstance(turn.get("startMs"), bool)
                    or not isinstance(turn.get("startMs"), int)
                    or isinstance(turn.get("endMs"), bool)
                    or not isinstance(turn.get("endMs"), int)
                    or turn["startMs"] < window.start_ms
                    or turn["endMs"] > window.end_ms
                    or turn["endMs"] <= turn["startMs"]
                ):
                    audit["status"] = "invalid-window-turn-evidence"
                    return None, audit
            snapshots.append(
                (
                    start_ms,
                    end_ms,
                    turn_count,
                    local_count,
                    tuple(local_speakers),
                    turns_sha256,
                )
            )

        if len(set(snapshots)) != 1:
            audit["status"] = "inconsistent-full-timeline-evidence"
            return None, audit
        snapshot = snapshots[0]
        observed_count = int(snapshot[3])
        audit.update(
            {
                "observedCount": observed_count,
                "turnCount": int(snapshot[2]),
                "speakerTurnsSha256": str(snapshot[5]),
            }
        )
        if observed_count > len(prepared.windows):
            audit["status"] = "outside-acoustic-evidence-range"
            return None, audit
        if observed_count == 1:
            audit["status"] = "single-track-not-independent-count-evidence"
            return None, audit
        audit["status"] = "eligible"
        return observed_count, audit

    def _initial_segments(
        self,
        prepared: PreparedAudio,
        asr: Sequence[AsrHypothesis],
        embeddings: Sequence[EmbeddingRecord],
        overlap: Sequence[OverlapDecision],
        clusters: _ClusterResult,
        *,
        requested_language: str,
    ) -> tuple[TranscriptSegment, ...]:
        segments: list[TranscriptSegment] = []
        for index, (window, hypothesis, embedding, overlap_item) in enumerate(
            zip(prepared.windows, asr, embeddings, overlap)
        ):
            raw_text = hypothesis.text
            normalized_text = hypothesis.normalized_text or raw_text
            display_text = hypothesis.display_text or normalized_text
            segment_language = reconcile_detected_languages(
                [
                    {
                        "languageCandidates": hypothesis.evidence.get(
                            "languageCandidates",
                            hypothesis.evidence.get(
                                "language",
                                hypothesis.evidence.get("rawLanguage"),
                            ),
                        ),
                        "speechDurationMs": window.end_ms - window.start_ms,
                    }
                ],
                requested_language=requested_language,
            )
            scores = tuple(
                SpeakerScore(f"speaker-{speaker + 1}", score)
                for speaker, score in enumerate(clusters.scores[index])
            )
            ranked = sorted(scores, key=lambda item: item.score, reverse=True)
            margin = (
                ranked[0].score - ranked[1].score
                if clusters.count > 1
                else 2.0
            )
            assigned = clusters.assignments[index]
            # Locked labels are constraints, so make the acoustic evidence
            # consistent with the constrained assignment without hiding it.
            if window.locked_speaker_id:
                assigned = _speaker_number(window.locked_speaker_id) - 1  # type: ignore[operator]
                forced = [
                    SpeakerScore(
                        score.speaker_id,
                        1.0 if score.speaker_id == window.locked_speaker_id
                        else min(score.score, 1.0 - self.config.high_margin_threshold),
                    )
                    for score in scores
                ]
                scores = tuple(forced)
                ranked = sorted(scores, key=lambda item: item.score, reverse=True)
                margin = (
                    ranked[0].score - ranked[1].score
                    if clusters.count > 1
                    else 2.0
                )
            refinement_evidence = window.metadata.get(
                "speakerChangeRefinement"
            )
            count_partition_evidence = window.metadata.get(
                "speakerCountPartition"
            )
            turn_projection_evidence = window.metadata.get(
                "speakerTurnProjection"
            )
            revisions: list[Revision] = []
            if normalized_text != raw_text:
                revisions.append(
                    Revision(
                        revision_id=f"{window.window_id}:text:1",
                        revision_type="text",
                        source="deterministic",
                        before=raw_text,
                        after=normalized_text,
                        reason_code="ASR_SOURCE_NORMALIZATION",
                        confidence=hypothesis.confidence,
                        evidence_refs=(f"asr:{window.window_id}",),
                    )
                )
            if display_text != normalized_text:
                revisions.append(
                    Revision(
                        revision_id=f"{window.window_id}:text:{len(revisions) + 1}",
                        revision_type="text",
                        source="deterministic",
                        before=normalized_text,
                        after=display_text,
                        reason_code="DISPLAY_PUNCTUATION",
                        confidence=hypothesis.confidence,
                        evidence_refs=(f"asr:{window.window_id}",),
                    )
                )
            segments.append(
                TranscriptSegment(
                    segment_id=window.window_id,
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    speaker_id=f"speaker-{assigned + 1}",
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    display_text=display_text,
                    confidence=hypothesis.confidence,
                    speaker_scores=scores,
                    speaker_margin=margin,
                    overlapping=overlap_item.overlapping,
                    human_locked=window.locked_speaker_id is not None,
                    revisions=tuple(revisions),
                    language=segment_language,
                    evidence={
                        "preparation": {
                            "provider": _adapter_identity(
                                self.preparation_adapter
                            ),
                            **(
                                {"audioPath": prepared.audio_path}
                                if prepared.audio_path is not None
                                else {}
                            ),
                        },
                        "boundary": {
                            "provider": _adapter_identity(
                                self.preparation_adapter
                            ),
                            "conflict": window.boundary_conflict,
                        },
                        **(
                            {
                                "speakerChangeRefinement": {
                                    **dict(refinement_evidence),
                                    "provider": _adapter_identity(
                                        self.embedding_adapter
                                    ),
                                }
                            }
                            if isinstance(refinement_evidence, Mapping)
                            else {}
                        ),
                        **(
                            {
                                "speakerCountPartition": dict(
                                    count_partition_evidence
                                )
                            }
                            if isinstance(
                                count_partition_evidence,
                                Mapping,
                            )
                            else {}
                        ),
                        **(
                            {
                                "speakerTurnProjection": dict(
                                    turn_projection_evidence
                                )
                            }
                            if isinstance(
                                turn_projection_evidence,
                                Mapping,
                            )
                            else {}
                        ),
                        "asr": {
                            "provider": _adapter_identity(self.asr_adapter),
                            **dict(hypothesis.evidence),
                        },
                        "voiceprint": {
                            "provider": _adapter_identity(
                                self.embedding_adapter
                            ),
                            "confidence": embedding.confidence,
                            **dict(embedding.evidence),
                        },
                        "overlap": {
                            "provider": _adapter_identity(
                                self.overlap_adapter
                            ),
                            "confidence": overlap_item.confidence,
                            **_normalized_overlap_evidence(
                                overlapping=overlap_item.overlapping,
                                evidence=overlap_item.evidence,
                            ),
                        },
                    },
                    turn_id=(
                        str(window.metadata["turnId"]).strip()
                        if isinstance(window.metadata.get("turnId"), str)
                        and str(window.metadata["turnId"]).strip()
                        else None
                    ),
                )
            )
        return tuple(segments)

    @staticmethod
    def _partition_source_id(segment: TranscriptSegment) -> str | None:
        partition = segment.evidence.get("speakerCountPartition")
        if not isinstance(partition, Mapping):
            return None
        source_id = partition.get("sourceWindowId")
        if not isinstance(source_id, str) or not source_id.strip():
            return None
        return source_id.strip()

    def _merge_partition_segment_run(
        self,
        run: Sequence[TranscriptSegment],
        *,
        run_index: int,
    ) -> TranscriptSegment | None:
        if len(run) < 2:
            return None
        first = run[0]
        source_id = self._partition_source_id(first)
        if source_id is None:
            return None
        if any(
            segment.human_locked
            or segment.revisions
            or segment.normalized_text != segment.raw_text
            or segment.display_text != segment.raw_text
            or self._partition_source_id(segment) != source_id
            or segment.speaker_id != first.speaker_id
            or segment.language != first.language
            or segment.overlapping != first.overlapping
            or segment.turn_id != first.turn_id
            for segment in run
        ):
            return None
        if any(
            left.end_ms != right.start_ms
            for left, right in zip(run, run[1:])
        ):
            return None

        validated_sets: list[dict[str, Any]] = []
        for segment in run:
            evidence = segment.evidence.get("asr")
            if (
                not isinstance(evidence, Mapping)
                or "candidateSetSchemaVersion" not in evidence
            ):
                return None
            try:
                validated_sets.append(
                    validate_asr_candidate_set(
                        evidence,
                        expected_text=segment.raw_text,
                        expected_start_ms=segment.start_ms,
                        expected_end_ms=segment.end_ms,
                    )
                )
            except AsrEvidenceError:
                return None

        identity_keys = (
            "modelId",
            "modelRevision",
            "modelManifestSha256",
            "modelIdentityStatus",
            "sourceAudioSha256",
            "normalizationProfile",
        )
        identity = tuple(validated_sets[0][key] for key in identity_keys)
        if any(
            tuple(candidate_set[key] for key in identity_keys) != identity
            for candidate_set in validated_sets[1:]
        ):
            return None
        candidates = [candidate_set["nBest"][0] for candidate_set in validated_sets]
        parent_ids = {candidate.get("parentCandidateId") for candidate in candidates}
        if len(parent_ids) != 1 or None in parent_ids:
            return None
        languages = {candidate["language"] for candidate in candidates}
        if len(languages) != 1:
            return None
        tokens = [
            {
                "text": token["text"],
                "startMs": token["startMs"],
                "endMs": token["endMs"],
            }
            for candidate in candidates
            for token in candidate["tokens"]
        ]
        if not tokens:
            return None
        merged_text = self._join_aligned_token_text(tokens)
        if not merged_text:
            return None

        segment_id = f"{source_id}.speaker-run-{run_index:02d}"
        if len(segment_id) > 120:
            segment_id = "segment-" + hashlib.sha256(
                segment_id.encode("utf-8")
            ).hexdigest()[:24]
        try:
            candidate_set = build_asr_candidate_set(
                model_id=validated_sets[0]["modelId"],
                model_revision=validated_sets[0]["modelRevision"],
                model_manifest_sha256=validated_sets[0]["modelManifestSha256"],
                model_identity_status=validated_sets[0]["modelIdentityStatus"],
                source_audio_sha256=validated_sets[0]["sourceAudioSha256"],
                normalization_profile=validated_sets[0]["normalizationProfile"],
                source_window_id=segment_id,
                start_ms=first.start_ms,
                end_ms=run[-1].end_ms,
                hypotheses=[
                    {
                        "text": merged_text,
                        "language": candidates[0]["language"],
                        "tokens": tokens,
                        "acousticScore": None,
                        "acousticScoreStatus": "projection-derived",
                        "decodeScore": None,
                        "decodeScoreStatus": "projection-derived",
                        "parentCandidateId": next(iter(parent_ids)),
                    }
                ],
                candidate_set_type="projection-derived-top1",
            )
        except AsrEvidenceError:
            return None

        durations = [segment.end_ms - segment.start_ms for segment in run]
        total_duration = sum(durations)
        score_ids = tuple(score.speaker_id for score in first.speaker_scores)
        if any(
            tuple(score.speaker_id for score in segment.speaker_scores) != score_ids
            for segment in run[1:]
        ):
            return None
        scores = tuple(
            SpeakerScore(
                speaker_id,
                sum(
                    segment.speaker_scores[index].score * duration
                    for segment, duration in zip(run, durations)
                )
                / total_duration,
            )
            for index, speaker_id in enumerate(score_ids)
        )
        ranked = sorted(scores, key=lambda item: item.score, reverse=True)
        margin = ranked[0].score - ranked[1].score if len(ranked) > 1 else 2.0

        evidence = dict(first.evidence)
        asr_evidence = dict(evidence["asr"])
        for key in ASR_CANDIDATE_SET_KEYS:
            asr_evidence.pop(key, None)
        asr_evidence.update(candidate_set)
        asr_evidence["timestamps"] = tokens
        asr_evidence["asrProjection"] = {
            "method": "coalesced-speaker-evidence-v1",
            "sourceWindowId": source_id,
            "targetWindowId": segment_id,
            "targetStartMs": first.start_ms,
            "targetEndMs": run[-1].end_ms,
            "tokenCount": len(tokens),
            "constituentCandidateSetSha256": [
                candidate_set["candidateSetSha256"]
                for candidate_set in validated_sets
            ],
        }
        evidence["asr"] = asr_evidence
        partition = dict(evidence["speakerCountPartition"])
        partition.update(
            {
                "coalescingMethod": "same-speaker-contiguous-v1",
                "constituentWindowIds": [segment.segment_id for segment in run],
                "constituentWindowCount": len(run),
            }
        )
        evidence["speakerCountPartition"] = partition
        evidence["boundary"] = {
            **dict(evidence["boundary"]),
            "conflict": any(
                bool(segment.evidence.get("boundary", {}).get("conflict"))
                for segment in run
                if isinstance(segment.evidence.get("boundary"), Mapping)
            ),
        }
        evidence["speakerEvidenceCoalescing"] = {
            "method": "same-source-speaker-run-v1",
            "sourceWindowId": source_id,
            "constituentSegmentIds": [segment.segment_id for segment in run],
            "constituentSegmentCount": len(run),
        }
        return replace(
            first,
            segment_id=segment_id,
            end_ms=run[-1].end_ms,
            raw_text=merged_text,
            normalized_text=merged_text,
            display_text=merged_text,
            confidence=min(segment.confidence for segment in run),
            speaker_scores=scores,
            speaker_margin=margin,
            revisions=(),
            evidence=evidence,
        )

    def _coalesce_speaker_evidence_segments(
        self,
        segments: Sequence[TranscriptSegment],
        metrics: PipelineMetricsCollector,
    ) -> tuple[TranscriptSegment, ...]:
        output: list[TranscriptSegment] = []
        run: list[TranscriptSegment] = []
        source_run_counts: Counter[str] = Counter()

        def flush() -> None:
            if not run:
                return
            source_id = self._partition_source_id(run[0])
            if source_id is None:
                output.extend(run)
                run.clear()
                return
            source_run_counts[source_id] += 1
            merged = self._merge_partition_segment_run(
                run,
                run_index=source_run_counts[source_id],
            )
            if merged is None:
                output.extend(run)
            else:
                output.append(merged)
            run.clear()

        for segment in segments:
            source_id = self._partition_source_id(segment)
            previous = run[-1] if run else None
            if (
                previous is not None
                and source_id is not None
                and self._partition_source_id(previous) == source_id
                and previous.end_ms == segment.start_ms
                and previous.speaker_id == segment.speaker_id
                and previous.language == segment.language
                and previous.overlapping == segment.overlapping
                and previous.turn_id == segment.turn_id
            ):
                run.append(segment)
                continue
            flush()
            run.append(segment)
        flush()
        metrics.set_policy(
            speakerEvidenceCoalescingApplied=len(output) < len(segments),
            speakerEvidenceInputSegments=len(segments),
            speakerEvidenceOutputSegments=len(output),
            speakerEvidenceMergedSegments=len(segments) - len(output),
        )
        return tuple(output)

    def _stabilize_temporal_assignments(
        self,
        segments: Sequence[TranscriptSegment],
    ) -> tuple[TranscriptSegment, ...]:
        """Resolve only provably safe short A-B-A speaker islands.

        Decisions are made from one immutable chronological snapshot so a
        correction cannot cascade into a second correction.  Cardinality is
        guarded incrementally: the last supporting segment for a speaker is
        never reassigned, even when every other continuity condition passes.
        All ambiguous islands are retained unchanged and explicitly marked for
        review.
        """

        chronological = sorted(
            segments,
            key=lambda segment: (
                segment.start_ms,
                segment.end_ms,
                segment.segment_id,
            ),
        )
        support_counts: dict[str, int] = {}
        for segment in segments:
            support_counts[segment.speaker_id] = (
                support_counts.get(segment.speaker_id, 0) + 1
            )
        updated_by_id = {
            segment.segment_id: segment for segment in segments
        }
        island_center_indices = {
            index
            for index in range(1, len(chronological) - 1)
            if (
                chronological[index - 1].speaker_id
                == chronological[index + 1].speaker_id
                and chronological[index].speaker_id
                != chronological[index - 1].speaker_id
            )
        }

        for index in range(1, len(chronological) - 1):
            left = chronological[index - 1]
            center = chronological[index]
            right = chronological[index + 1]
            if index not in island_center_indices:
                continue

            target_speaker = left.speaker_id
            duration_ms = center.end_ms - center.start_ms
            left_gap_ms = max(0, center.start_ms - left.end_ms)
            right_gap_ms = max(0, right.start_ms - center.end_ms)
            has_time_overlap = (
                left.end_ms > center.start_ms
                or center.end_ms > right.start_ms
            )
            boundary = center.evidence.get("boundary")
            boundary_conflict = (
                isinstance(boundary, Mapping)
                and boundary.get("conflict") is True
            )
            ranked = sorted(
                center.speaker_scores,
                key=lambda item: (-item.score, item.speaker_id),
            )
            top_two = {
                item.speaker_id for item in ranked[:2]
            }

            temporal_evidence: dict[str, Any] = {
                "beforeSpeakerId": center.speaker_id,
                "candidateSpeakerId": target_speaker,
                "neighborSegmentIds": [
                    left.segment_id,
                    right.segment_id,
                ],
                "durationMs": duration_ms,
                "leftGapMs": left_gap_ms,
                "rightGapMs": right_gap_ms,
            }
            if center.speaker_margin >= self.config.low_margin_threshold:
                current = updated_by_id[center.segment_id]
                updated_by_id[center.segment_id] = replace(
                    current,
                    evidence={
                        **dict(current.evidence),
                        "temporalStabilization": {
                            **temporal_evidence,
                            "reviewStatus": "NOT_REQUIRED",
                            "reasonCode": (
                                "SKIPPED_STRONG_ACOUSTIC_EVIDENCE"
                            ),
                            "blockers": [],
                            "applied": False,
                        },
                    },
                )
                continue

            blockers: list[str] = []
            if duration_ms > self.config.temporal_short_segment_ms:
                blockers.append("DURATION_EXCEEDS_LIMIT")
            if has_time_overlap:
                blockers.append("TEMPORAL_OVERLAP")
            if (
                left_gap_ms > self.config.temporal_max_gap_ms
                or right_gap_ms > self.config.temporal_max_gap_ms
            ):
                blockers.append("GAP_EXCEEDS_LIMIT")
            if (
                index - 1 in island_center_indices
                or index + 1 in island_center_indices
            ):
                blockers.append("ADJACENT_ISLAND_AMBIGUITY")
            if center.human_locked:
                blockers.append("HUMAN_LOCKED")
            if center.overlapping:
                blockers.append("OVERLAP_PROTECTED")
            if left.overlapping or right.overlapping:
                blockers.append("NEIGHBOR_OVERLAP")
            if boundary_conflict:
                blockers.append("BOUNDARY_CONFLICT")
            if target_speaker not in top_two:
                blockers.append("NEIGHBOR_SPEAKER_OUTSIDE_TOP2")
            if support_counts.get(center.speaker_id, 0) <= 1:
                blockers.append("CARDINALITY_CHANGE_REVIEW_REQUIRED")

            if blockers:
                updated_by_id[center.segment_id] = replace(
                    updated_by_id[center.segment_id],
                    evidence={
                        **dict(updated_by_id[center.segment_id].evidence),
                        "temporalStabilization": {
                            **temporal_evidence,
                            "reviewStatus": "REVIEW_REQUIRED",
                            "reasonCode": "TEMPORAL_CONTINUITY_UNRESOLVED",
                            "blockers": blockers,
                            "applied": False,
                        },
                    },
                )
                continue

            current = updated_by_id[center.segment_id]
            target_score = next(
                item.score
                for item in ranked
                if item.speaker_id == target_speaker
            )
            confidence = max(0.0, min(1.0, (target_score + 1.0) / 2.0))
            revision = Revision(
                revision_id=(
                    f"{center.segment_id}:speaker:{len(current.revisions) + 1}"
                ),
                revision_type="speaker",
                source="deterministic",
                before=center.speaker_id,
                after=target_speaker,
                reason_code="TEMPORAL_CONTINUITY_SHORT_SEGMENT",
                confidence=confidence,
                evidence_refs=(
                    f"temporal:{left.segment_id}:{center.segment_id}:"
                    f"{right.segment_id}",
                ),
            )
            updated_by_id[center.segment_id] = replace(
                current,
                speaker_id=target_speaker,
                revisions=(*current.revisions, revision),
                evidence={
                    **dict(current.evidence),
                    "temporalStabilization": {
                        **temporal_evidence,
                        "afterSpeakerId": target_speaker,
                        "reviewStatus": "RESOLVED",
                        "reasonCode": "TEMPORAL_CONTINUITY_SHORT_SEGMENT",
                        "blockers": [],
                        "applied": True,
                    },
                },
            )
            support_counts[center.speaker_id] -= 1
            support_counts[target_speaker] = (
                support_counts.get(target_speaker, 0) + 1
            )

        return tuple(
            updated_by_id[segment.segment_id] for segment in segments
        )

    def _decode_global_speaker_sequence(
        self,
        segments: Sequence[TranscriptSegment],
    ) -> tuple[TranscriptSegment, ...]:
        """Decode one acoustic-first global path while preserving cardinality.

        Every candidate emission comes from the segment's voiceprint score
        inventory. Human locks and overlap-protected segments remain hard
        constraints, and the sequence decoder derives strong anchors only
        from acoustic score and margin. Semantic context is deliberately
        excluded from this production path.
        """

        result = decode_speaker_sequence(
            tuple(
                SequenceSegment(
                    segment_id=segment.segment_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    original_speaker_id=segment.speaker_id,
                    speaker_scores=tuple(
                        SpeakerEmission(
                            speaker_id=item.speaker_id,
                            score=item.score,
                        )
                        for item in segment.speaker_scores
                    ),
                    human_locked_speaker_id=(
                        segment.speaker_id
                        if segment.human_locked or segment.overlapping
                        else None
                    ),
                )
                for segment in segments
            ),
            SequenceDecoderConfig(
                top_m=2,
                preserve_original_cardinality=True,
            ),
        )
        assignments_by_id = {
            assignment.segment_id: assignment
            for assignment in result.assignments
        }
        decoded: list[TranscriptSegment] = []
        for segment in segments:
            assignment = assignments_by_id[segment.segment_id]
            reason_codes = list(assignment.reason_codes)
            review_status = assignment.review_status
            if segment.overlapping:
                reason_codes = [
                    code
                    for code in reason_codes
                    if code != "HUMAN_LOCKED" or segment.human_locked
                ]
                if "OVERLAP_PROTECTED" not in reason_codes:
                    reason_codes.append("OVERLAP_PROTECTED")
                review_status = "REVIEW_REQUIRED"
            sequence_evidence: dict[str, Any] = {
                "method": result.method,
                "beforeSpeakerId": assignment.original_speaker_id,
                "afterSpeakerId": assignment.speaker_id,
                "acousticScore": assignment.acoustic_score,
                "reviewStatus": review_status,
                "reasonCodes": reason_codes,
                "applied": assignment.changed,
            }
            revisions = segment.revisions
            if assignment.changed:
                confidence = min(
                    1.0,
                    max(0.0, (assignment.acoustic_score + 1.0) / 2.0),
                )
                revisions = (
                    *revisions,
                    Revision(
                        revision_id=(
                            f"{segment.segment_id}:speaker:"
                            f"{len(segment.revisions) + 1}"
                        ),
                        revision_type="speaker",
                        source="deterministic",
                        before=assignment.original_speaker_id,
                        after=assignment.speaker_id,
                        reason_code="GLOBAL_ACOUSTIC_SEQUENCE_DECODE",
                        confidence=confidence,
                        evidence_refs=(
                            f"speaker-sequence:{segment.segment_id}",
                        ),
                    ),
                )
            decoded.append(
                replace(
                    segment,
                    speaker_id=assignment.speaker_id,
                    revisions=revisions,
                    evidence={
                        **dict(segment.evidence),
                        "speakerSequenceDecode": sequence_evidence,
                        },
                )
            )
        return tuple(decoded)

    @staticmethod
    def _assignment_score(
        weights: Sequence[Sequence[float]],
        assignment: Sequence[tuple[int, int]],
    ) -> float:
        return sum(weights[row][column] for row, column in assignment)

    def _apply_pyannote_canonical_mapping(
        self,
        segments: Sequence[TranscriptSegment],
    ) -> tuple[TranscriptSegment, ...]:
        """Map anonymous pyannote tracks to canonical acoustic speakers."""

        if self.config.pyannote_mode != "fallback" or not segments:
            return tuple(segments)
        canonical_speakers = sorted(
            {
                score.speaker_id
                for segment in segments
                for score in segment.speaker_scores
            },
            key=lambda speaker_id: (_speaker_number(speaker_id) or math.inf),
        )
        if not canonical_speakers:
            return tuple(segments)

        turns_by_segment: dict[str, list[Mapping[str, Any]]] = {}
        exclusive_turns_by_segment: dict[str, list[Mapping[str, Any]]] = {}
        native_exclusive_available = True
        local_speakers: set[str] = set()
        for segment in segments:
            overlap = segment.evidence.get("overlap")
            provider = (
                overlap.get("provider")
                if isinstance(overlap, Mapping)
                else None
            )
            raw_turns = (
                overlap.get("speakerTurns")
                if isinstance(overlap, Mapping)
                and isinstance(provider, Mapping)
                and provider.get("id") == "pyannote-community-1"
                else None
            )
            if not isinstance(raw_turns, list):
                return tuple(segments)
            raw_exclusive_turns = (
                overlap.get("exclusiveSpeakerTurns")
                if isinstance(overlap, Mapping)
                else None
            )
            if not isinstance(raw_exclusive_turns, list):
                native_exclusive_available = False
                raw_exclusive_turns = []
            validated: list[Mapping[str, Any]] = []
            for raw in raw_turns:
                if not isinstance(raw, Mapping):
                    return tuple(segments)
                start_ms = raw.get("startMs")
                end_ms = raw.get("endMs")
                local_speaker = raw.get("localSpeaker")
                if (
                    isinstance(start_ms, bool)
                    or not isinstance(start_ms, int)
                    or isinstance(end_ms, bool)
                    or not isinstance(end_ms, int)
                    or not isinstance(local_speaker, str)
                    or not local_speaker.strip()
                    or start_ms < segment.start_ms
                    or end_ms > segment.end_ms
                    or end_ms <= start_ms
                ):
                    return tuple(segments)
                validated.append(raw)
                local_speakers.add(local_speaker.strip())
            turns_by_segment[segment.segment_id] = validated
            validated_exclusive: list[Mapping[str, Any]] = []
            for raw in raw_exclusive_turns:
                if not isinstance(raw, Mapping):
                    raise WorkerError(
                        "SPEAKER_TIMELINE_INVALID",
                        "pyannote exclusive speaker turn is malformed",
                    )
                start_ms = raw.get("startMs")
                end_ms = raw.get("endMs")
                local_speaker = raw.get("localSpeaker")
                if (
                    isinstance(start_ms, bool)
                    or not isinstance(start_ms, int)
                    or isinstance(end_ms, bool)
                    or not isinstance(end_ms, int)
                    or not isinstance(local_speaker, str)
                    or not local_speaker.strip()
                    or start_ms < segment.start_ms
                    or end_ms > segment.end_ms
                    or end_ms <= start_ms
                ):
                    raise WorkerError(
                        "SPEAKER_TIMELINE_INVALID",
                        "pyannote exclusive speaker turn is out of range",
                    )
                validated_exclusive.append(raw)
            exclusive_turns_by_segment[segment.segment_id] = (
                validated_exclusive
            )

        ordered_local = sorted(local_speakers)
        if len(ordered_local) > len(canonical_speakers):
            return tuple(segments)
        complete_cardinality = len(ordered_local) == len(canonical_speakers)
        local_index = {
            speaker_id: index for index, speaker_id in enumerate(ordered_local)
        }
        canonical_index = {
            speaker_id: index
            for index, speaker_id in enumerate(canonical_speakers)
        }
        weights = [
            [0.0 for _ in canonical_speakers]
            for _ in ordered_local
        ]
        total_track_ms = 0
        for segment in segments:
            scores = {
                score.speaker_id: max(-1.0, min(1.0, score.score))
                for score in segment.speaker_scores
            }
            if set(scores) != set(canonical_speakers):
                return tuple(segments)
            for turn in turns_by_segment[segment.segment_id]:
                duration_ms = int(turn["endMs"]) - int(turn["startMs"])
                total_track_ms += duration_ms
                row = local_index[str(turn["localSpeaker"]).strip()]
                for speaker_id, score in scores.items():
                    weights[row][canonical_index[speaker_id]] += (
                        duration_ms * score
                    )
        if total_track_ms <= 0:
            return tuple(segments)

        assignment = maximum_weight_assignment(weights)
        if len(assignment) != len(ordered_local):
            return tuple(segments)
        optimal_score = self._assignment_score(weights, assignment)
        alternative_score = -math.inf
        minimum_weight = min(
            value for row in weights for value in row
        )
        for forbidden_row, forbidden_column in assignment:
            alternative = [list(row) for row in weights]
            alternative[forbidden_row][forbidden_column] = (
                minimum_weight - abs(optimal_score) - total_track_ms - 1.0
            )
            candidate = maximum_weight_assignment(alternative)
            if (
                len(candidate) == len(ordered_local)
                and (forbidden_row, forbidden_column) not in candidate
            ):
                alternative_score = max(
                    alternative_score,
                    self._assignment_score(weights, candidate),
                )
        if len(ordered_local) == len(canonical_speakers) == 1:
            mapping_margin = 1.0
            alternative_score_value: float | None = None
        else:
            alternative_score_value = (
                alternative_score if math.isfinite(alternative_score) else None
            )
            mapping_margin = (
                max(0.0, optimal_score - alternative_score) / total_track_ms
                if alternative_score_value is not None
                else 0.0
            )
        mapping = {
            ordered_local[row]: canonical_speakers[column]
            for row, column in assignment
        }
        margin_accepted = (
            mapping_margin
            >= self.config.pyannote_mapping_margin_threshold
        )
        mapping_accepted = complete_cardinality and margin_accepted
        partial_mapping_accepted = (
            not complete_cardinality and margin_accepted
        )
        unmatched_canonical = sorted(
            set(canonical_speakers) - set(mapping.values()),
            key=lambda speaker_id: (
                _speaker_number(speaker_id) or math.inf
            ),
        )
        mapping_evidence = {
            "provider": {
                "id": "pyannote-community-1",
                "version": str(
                    getattr(self.pyannote_adapter, "version", "unknown")
                ),
            },
            "method": (
                "global-duration-weighted-acoustic-hungarian-v1"
                if complete_cardinality
                else (
                    "global-duration-weighted-acoustic-"
                    "rectangular-hungarian-v1"
                )
            ),
            "mapping": dict(sorted(mapping.items())),
            "weights": {
                ordered_local[row]: {
                    canonical_speakers[column]: round(value, 6)
                    for column, value in enumerate(weight_row)
                }
                for row, weight_row in enumerate(weights)
            },
            "optimalScore": round(optimal_score, 6),
            "alternativeScore": (
                round(alternative_score_value, 6)
                if alternative_score_value is not None
                else None
            ),
            "totalTrackMs": total_track_ms,
            "mappingMargin": round(mapping_margin, 9),
            "mappingMarginThreshold": (
                self.config.pyannote_mapping_margin_threshold
            ),
            "primaryDominanceThreshold": (
                self.config.pyannote_primary_dominance_threshold
            ),
            "accepted": mapping_accepted,
            "partialAccepted": partial_mapping_accepted,
            "completeCanonicalBijection": complete_cardinality,
            "observedLocalSpeakerCount": len(ordered_local),
            "canonicalSpeakerCount": len(canonical_speakers),
            "unmatchedCanonicalSpeakerIds": unmatched_canonical,
            "authoritativeTimelineEligible": mapping_accepted,
        }

        proposed: list[TranscriptSegment] = []
        for segment in segments:
            local_durations: dict[str, int] = {}
            exclusive_local_durations: dict[str, int] = {}
            canonical_turns: list[dict[str, Any]] = []
            canonical_exclusive_turns: list[dict[str, Any]] = []
            partial_canonical_turns: list[dict[str, Any]] = []
            partial_canonical_exclusive_turns: list[dict[str, Any]] = []
            for turn in turns_by_segment[segment.segment_id]:
                local_speaker = str(turn["localSpeaker"]).strip()
                duration_ms = int(turn["endMs"]) - int(turn["startMs"])
                local_durations[local_speaker] = (
                    local_durations.get(local_speaker, 0) + duration_ms
                )
                if mapping_accepted:
                    canonical_turns.append(
                        {
                            "startMs": int(turn["startMs"]),
                            "endMs": int(turn["endMs"]),
                            "speakerId": mapping[local_speaker],
                            "localSpeaker": local_speaker,
                        }
                    )
                elif partial_mapping_accepted:
                    partial_canonical_turns.append(
                        {
                            "startMs": int(turn["startMs"]),
                            "endMs": int(turn["endMs"]),
                            "speakerId": mapping[local_speaker],
                            "localSpeaker": local_speaker,
                        }
                    )
            if (
                (mapping_accepted or partial_mapping_accepted)
                and native_exclusive_available
            ):
                for turn in exclusive_turns_by_segment[segment.segment_id]:
                    local_speaker = str(turn["localSpeaker"]).strip()
                    if local_speaker not in mapping:
                        raise WorkerError(
                            "SPEAKER_TIMELINE_INVALID",
                            "pyannote exclusive timeline contains an unmapped speaker",
                        )
                    destination = (
                        canonical_exclusive_turns
                        if mapping_accepted
                        else partial_canonical_exclusive_turns
                    )
                    destination.append(
                        {
                            "startMs": int(turn["startMs"]),
                            "endMs": int(turn["endMs"]),
                            "speakerId": mapping[local_speaker],
                            "localSpeaker": local_speaker,
                        }
                    )
                    exclusive_local_durations[local_speaker] = (
                        exclusive_local_durations.get(local_speaker, 0)
                        + int(turn["endMs"])
                        - int(turn["startMs"])
                    )
            attribution_durations = (
                exclusive_local_durations
                if exclusive_local_durations
                else local_durations
            )
            ranked_local = sorted(
                attribution_durations.items(),
                key=lambda item: (-item[1], item[0]),
            )
            tracked_ms = sum(attribution_durations.values())
            dominant_local = ranked_local[0][0] if ranked_local else None
            dominance = (
                ranked_local[0][1] / tracked_ms
                if ranked_local and tracked_ms > 0
                else 0.0
            )
            target_speaker = (
                mapping[dominant_local]
                if mapping_accepted and dominant_local is not None
                else segment.speaker_id
            )
            blockers: list[str] = []
            if not margin_accepted:
                blockers.append("PYANNOTE_MAPPING_MARGIN_BELOW_THRESHOLD")
            if not complete_cardinality:
                blockers.append(
                    "PYANNOTE_CANONICAL_CARDINALITY_MISMATCH"
                )
            if dominant_local is None:
                blockers.append("PYANNOTE_NO_LOCAL_SPEECH")
            if (
                target_speaker != segment.speaker_id
                and dominance
                < self.config.pyannote_primary_dominance_threshold
            ):
                blockers.append("PYANNOTE_PRIMARY_DOMINANCE_BELOW_THRESHOLD")
            if segment.human_locked and target_speaker != segment.speaker_id:
                blockers.append("HUMAN_LOCKED")
            applied = target_speaker != segment.speaker_id and not blockers
            revisions = segment.revisions
            if applied:
                revisions = (
                    *revisions,
                    Revision(
                        revision_id=(
                            f"{segment.segment_id}:speaker:"
                            f"{len(segment.revisions) + 1}"
                        ),
                        revision_type="speaker",
                        source="acoustic",
                        before=segment.speaker_id,
                        after=target_speaker,
                        reason_code="PYANNOTE_CANONICAL_TRACK_MAPPING",
                        confidence=min(1.0, mapping_margin),
                        evidence_refs=(
                            f"pyannote-mapping:{segment.segment_id}",
                        ),
                    ),
                )
            overlap = dict(segment.evidence.get("overlap", {}))
            if mapping_accepted:
                overlap["canonicalSpeakerTurns"] = canonical_turns
                if native_exclusive_available:
                    overlap["canonicalExclusiveSpeakerTurns"] = (
                        canonical_exclusive_turns
                    )
            if partial_mapping_accepted:
                overlap["partialCanonicalSpeakerTurns"] = (
                    partial_canonical_turns
                )
                if native_exclusive_available:
                    overlap["partialCanonicalExclusiveSpeakerTurns"] = (
                        partial_canonical_exclusive_turns
                    )
            proposed.append(
                replace(
                    segment,
                    speaker_id=target_speaker if applied else segment.speaker_id,
                    revisions=revisions,
                    evidence={
                        **dict(segment.evidence),
                        "overlap": overlap,
                        "pyannoteCanonicalMapping": {
                            **mapping_evidence,
                            "localDurationsMs": dict(
                                sorted(local_durations.items())
                            ),
                            "exclusiveLocalDurationsMs": dict(
                                sorted(exclusive_local_durations.items())
                            ),
                            "dominanceSource": (
                                "native-exclusive-speaker-diarization"
                                if exclusive_local_durations
                                else "regular-speaker-diarization"
                            ),
                            "dominantLocalSpeaker": dominant_local,
                            "dominance": round(dominance, 9),
                            "partialCanonicalSpeakerTurns": (
                                partial_canonical_turns
                            ),
                            "partialCanonicalExclusiveSpeakerTurns": (
                                partial_canonical_exclusive_turns
                            ),
                            "beforeSpeakerId": segment.speaker_id,
                            "afterSpeakerId": (
                                target_speaker if applied else segment.speaker_id
                            ),
                            "blockers": blockers,
                            "applied": applied,
                            "reviewStatus": (
                                "REVIEW_REQUIRED" if blockers else "RESOLVED"
                            ),
                        },
                    },
                )
            )

        before_ids = {segment.speaker_id for segment in segments}
        after_ids = {segment.speaker_id for segment in proposed}
        if after_ids == before_ids:
            return tuple(proposed)
        reverted: list[TranscriptSegment] = []
        before_by_id = {
            segment.segment_id: segment for segment in segments
        }
        for segment in proposed:
            before = before_by_id[segment.segment_id]
            evidence = dict(segment.evidence)
            mapping_item = dict(evidence["pyannoteCanonicalMapping"])
            mapping_item.update(
                {
                    "afterSpeakerId": before.speaker_id,
                    "blockers": sorted(
                        {
                            *mapping_item["blockers"],
                            "CARDINALITY_CHANGE_REVIEW_REQUIRED",
                        }
                    ),
                    "applied": False,
                    "reviewStatus": "REVIEW_REQUIRED",
                }
            )
            evidence["pyannoteCanonicalMapping"] = mapping_item
            reverted.append(
                replace(
                    before,
                    evidence=evidence,
                )
            )
        return tuple(reverted)

    @staticmethod
    def _build_pyannote_speaker_timeline(
        segments: Sequence[TranscriptSegment],
        *,
        duration_ms: int,
    ) -> Mapping[str, Any] | None:
        if not segments:
            return None
        canonical_ids = tuple(
            sorted(
                {
                    score.speaker_id
                    for segment in segments
                    for score in segment.speaker_scores
                },
                key=lambda item: _speaker_number(item) or math.inf,
            )
        )
        full_regular_snapshot: list[Mapping[str, Any]] | None = None
        full_exclusive_snapshot: list[Mapping[str, Any]] | None = None
        full_regular_sha256: str | None = None
        full_exclusive_sha256: str | None = None
        mapping_snapshot: dict[str, str] | None = None
        provider_version: str | None = None
        mapping_margin: float | None = None
        for segment in segments:
            mapping_evidence = segment.evidence.get(
                "pyannoteCanonicalMapping"
            )
            overlap = segment.evidence.get("overlap")
            if (
                not isinstance(mapping_evidence, Mapping)
                or mapping_evidence.get("accepted") is not True
                or not isinstance(overlap, Mapping)
            ):
                return None
            if "CARDINALITY_CHANGE_REVIEW_REQUIRED" in set(
                mapping_evidence.get("blockers", ())
            ):
                return None
            raw_mapping = mapping_evidence.get("mapping")
            provider = mapping_evidence.get("provider")
            margin = mapping_evidence.get("mappingMargin")
            raw_regular = overlap.get("canonicalSpeakerTurns")
            raw_exclusive = overlap.get("canonicalExclusiveSpeakerTurns")
            full_timeline = overlap.get("fullTimelineInference")
            if (
                not isinstance(raw_mapping, Mapping)
                or not isinstance(provider, Mapping)
                or provider.get("id") != "pyannote-community-1"
                or not isinstance(provider.get("version"), str)
                or isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or not isinstance(raw_regular, list)
                or not isinstance(raw_exclusive, list)
                or not isinstance(full_timeline, Mapping)
            ):
                return None
            full_regular = full_timeline.get("speakerTurns")
            full_exclusive = full_timeline.get("exclusiveSpeakerTurns")
            current_regular_sha256 = full_timeline.get(
                "speakerTurnsSha256"
            )
            current_exclusive_sha256 = full_timeline.get(
                "exclusiveSpeakerTurnsSha256"
            )
            if (
                full_timeline.get("scope") != "full-normalized-timeline"
                or full_timeline.get("startMs") != 0
                or full_timeline.get("endMs") != duration_ms
                or full_timeline.get("exclusiveNative") is not True
                or not isinstance(full_regular, list)
                or not isinstance(full_exclusive, list)
                or any(
                    not isinstance(turn, Mapping)
                    for turn in (*full_regular, *full_exclusive)
                )
                or full_timeline.get("turnCount") != len(full_regular)
                or full_timeline.get("exclusiveTurnCount")
                != len(full_exclusive)
                or not isinstance(current_regular_sha256, str)
                or not isinstance(current_exclusive_sha256, str)
            ):
                return None
            serialized_regular = json.dumps(
                full_regular,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            serialized_exclusive = json.dumps(
                full_exclusive,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if (
                hashlib.sha256(serialized_regular).hexdigest()
                != current_regular_sha256
                or hashlib.sha256(serialized_exclusive).hexdigest()
                != current_exclusive_sha256
            ):
                raise WorkerError(
                    "SPEAKER_TIMELINE_INVALID",
                    "pyannote full timeline hashes do not bind their turns",
                )
            current_mapping = {
                str(local): str(speaker)
                for local, speaker in raw_mapping.items()
            }
            current_provider_version = str(provider["version"])
            current_margin = float(margin)
            if mapping_snapshot is None:
                mapping_snapshot = current_mapping
                provider_version = current_provider_version
                mapping_margin = current_margin
                full_regular_snapshot = [
                    dict(turn) for turn in full_regular
                ]
                full_exclusive_snapshot = [
                    dict(turn) for turn in full_exclusive
                ]
                full_regular_sha256 = current_regular_sha256
                full_exclusive_sha256 = current_exclusive_sha256
            elif (
                current_mapping != mapping_snapshot
                or current_provider_version != provider_version
                or current_margin != mapping_margin
                or current_regular_sha256 != full_regular_sha256
                or current_exclusive_sha256 != full_exclusive_sha256
                or full_regular != full_regular_snapshot
                or full_exclusive != full_exclusive_snapshot
            ):
                raise WorkerError(
                    "SPEAKER_TIMELINE_INVALID",
                    "pyannote canonical mapping evidence is inconsistent",
                )

            if segment.human_locked:
                durations: dict[str, int] = {}
                for turn in raw_exclusive:
                    start_ms = max(segment.start_ms, int(turn["startMs"]))
                    end_ms = min(segment.end_ms, int(turn["endMs"]))
                    if end_ms > start_ms:
                        speaker_id = str(turn["speakerId"])
                        durations[speaker_id] = (
                            durations.get(speaker_id, 0)
                            + end_ms
                            - start_ms
                        )
                if not durations:
                    return None
                dominant = min(
                    durations,
                    key=lambda speaker_id: (
                        -durations[speaker_id],
                        _speaker_number(speaker_id) or math.inf,
                    ),
                )
                if dominant != segment.speaker_id:
                    return None
        if (
            mapping_snapshot is None
            or provider_version is None
            or mapping_margin is None
            or not full_regular_snapshot
            or not full_exclusive_snapshot
        ):
            return None
        regular_turns: list[dict[str, Any]] = []
        exclusive_turns: list[dict[str, Any]] = []
        for raw_turns, destination in (
            (full_regular_snapshot, regular_turns),
            (full_exclusive_snapshot, exclusive_turns),
        ):
            for raw in raw_turns:
                start_ms = raw.get("startMs")
                end_ms = raw.get("endMs")
                local_speaker = raw.get("localSpeaker")
                if (
                    isinstance(start_ms, bool)
                    or not isinstance(start_ms, int)
                    or isinstance(end_ms, bool)
                    or not isinstance(end_ms, int)
                    or not isinstance(local_speaker, str)
                    or local_speaker not in mapping_snapshot
                    or start_ms < 0
                    or end_ms <= start_ms
                    or end_ms > duration_ms
                ):
                    raise WorkerError(
                        "SPEAKER_TIMELINE_INVALID",
                        "pyannote full timeline contains a malformed turn",
                    )
                destination.append(
                    {
                        "startMs": start_ms,
                        "endMs": end_ms,
                        "speakerId": mapping_snapshot[local_speaker],
                        "localSpeaker": local_speaker,
                    }
                )
        return build_speaker_timeline(
            provider_version=provider_version,
            local_to_canonical=mapping_snapshot,
            mapping_margin=mapping_margin,
            regular_turns=regular_turns,
            exclusive_turns=exclusive_turns,
            duration_ms=duration_ms,
            canonical_speaker_ids=canonical_ids,
        )

    def _select_candidates(
        self,
        segments: Sequence[TranscriptSegment],
        windows: Sequence[SpeechWindow],
        clusters: _ClusterResult,
    ) -> list[ReviewCandidate]:
        reasons_by_segment: dict[str, list[str]] = {}
        protected_by_segment: dict[str, bool] = {}
        windows_by_id = {window.window_id: window for window in windows}
        for segment in segments:
            window = windows_by_id.get(segment.segment_id)
            reasons: list[str] = []
            temporal = segment.evidence.get("temporalStabilization")
            if (
                isinstance(temporal, Mapping)
                and temporal.get("reviewStatus") == "REVIEW_REQUIRED"
            ):
                reasons.extend(("TEMPORAL_DISCONTINUITY", "SHORT_SEGMENT"))
            sequence_decode = segment.evidence.get(
                "speakerSequenceDecode"
            )
            if (
                isinstance(sequence_decode, Mapping)
                and sequence_decode.get("reviewStatus")
                == "REVIEW_REQUIRED"
            ):
                reasons.append("TEMPORAL_DISCONTINUITY")
                reason_codes = sequence_decode.get("reasonCodes", ())
                if (
                    isinstance(reason_codes, Sequence)
                    and not isinstance(reason_codes, (str, bytes))
                    and CARDINALITY_CHANGE_REVIEW_REQUIRED
                    in reason_codes
                ):
                    reasons.append("COUNT_UNCERTAINTY")
            if segment.overlapping:
                reasons.append("OVERLAP")
            pyannote_mapping = segment.evidence.get(
                "pyannoteCanonicalMapping"
            )
            if (
                isinstance(pyannote_mapping, Mapping)
                and pyannote_mapping.get("partialAccepted") is True
                and pyannote_mapping.get("accepted") is not True
            ):
                reasons.append("PYANNOTE_PARTIAL_MAPPING")
            refinement_evidence = segment.evidence.get(
                "speakerChangeRefinement"
            )
            if (
                isinstance(refinement_evidence, Mapping)
                and (
                    refinement_evidence.get("reviewRequired") is True
                    or refinement_evidence.get("reviewStatus")
                    == "REVIEW_REQUIRED"
                )
            ):
                reasons.append(
                    _SPEAKER_CHANGE_REFINEMENT_REVIEW_REASON
                )
            count_partition_evidence = segment.evidence.get(
                "speakerCountPartition"
            )
            if (
                isinstance(count_partition_evidence, Mapping)
                and count_partition_evidence.get("reviewRequired") is True
            ):
                reasons.append("COUNT_UNCERTAINTY")
            overlap_evidence = segment.evidence.get("overlap")
            if (
                isinstance(overlap_evidence, Mapping)
                and overlap_evidence.get("detectorStatus") == "UNAVAILABLE"
                and overlap_evidence.get("overlapDetectorRun") is False
                and overlap_evidence.get("reviewStatus") == "REVIEW_REQUIRED"
                and overlap_evidence.get("reasonCode")
                == _OVERLAP_DETECTOR_UNAVAILABLE_REASON
            ):
                reasons.append(_OVERLAP_DETECTOR_UNAVAILABLE_REASON)
            if segment.speaker_margin < self.config.low_margin_threshold:
                reasons.append("LOW_MARGIN")
            boundary_evidence = segment.evidence.get("boundary")
            if (
                (window is not None and window.boundary_conflict)
                or (
                    isinstance(boundary_evidence, Mapping)
                    and boundary_evidence.get("conflict") is True
                )
            ):
                reasons.append("BOUNDARY_CONFLICT")
            if max(score.score for score in segment.speaker_scores) < (
                self.config.outlier_score_threshold
            ):
                reasons.append("OUTLIER")
            if reasons:
                reasons_by_segment[segment.segment_id] = reasons
                protected_by_segment[segment.segment_id] = (
                    segment.human_locked
                    or segment.overlapping
                    or segment.speaker_margin
                    >= self.config.high_margin_threshold
                )

        count_uncertain = (
            clusters.confidence < self.config.auto_count_confidence_threshold
            or clusters.candidate_min != clusters.candidate_max
            or clusters.low_confidence_fail_closed
        )
        if count_uncertain:
            uncertainty_budget = min(
                self.config.max_count_uncertainty_candidates,
                len(segments),
            )
            ranked_for_count = sorted(
                segments,
                key=lambda segment: (
                    segment.human_locked or segment.overlapping,
                    segment.speaker_margin,
                    -max(score.score for score in segment.speaker_scores),
                    segment.start_ms,
                ),
            )
            for segment in ranked_for_count[:uncertainty_budget]:
                reasons = reasons_by_segment.setdefault(segment.segment_id, [])
                if "COUNT_UNCERTAINTY" not in reasons:
                    reasons.append("COUNT_UNCERTAINTY")
                protected_by_segment[segment.segment_id] = (
                    segment.human_locked
                    or segment.overlapping
                    or segment.speaker_margin
                    >= self.config.high_margin_threshold
                )

        return [
            ReviewCandidate(
                segment_id=segment.segment_id,
                reasons=tuple(reasons_by_segment[segment.segment_id]),
                protected=protected_by_segment[segment.segment_id],
            )
            for segment in segments
            if segment.segment_id in reasons_by_segment
        ]

    def _run_review_adapter(
        self,
        *,
        stage: str,
        adapter: EscalationAdapter,
        candidates: Sequence[ReviewCandidate],
        segments: Sequence[TranscriptSegment],
        metrics: PipelineMetricsCollector,
        context: AdapterContext,
    ) -> tuple[
        dict[str, ReviewProposal],
        dict[str, str],
        dict[str, bool],
        dict[str, float],
        dict[str, str],
    ]:
        started = time.perf_counter()
        identity = _adapter_identity(adapter)
        by_segment = {segment.segment_id: segment for segment in segments}
        proposals: dict[str, ReviewProposal] = {}
        misses: list[ReviewCandidate] = []
        keys: dict[str, str] = {}
        cache_hits: dict[str, bool] = {}
        latencies: dict[str, float] = {}
        recomputations = 0
        for candidate in candidates:
            segment = by_segment[candidate.segment_id]
            cache_material_provider = getattr(adapter, "cache_material", None)
            cache_material = (
                cache_material_provider(candidate, by_segment)
                if callable(cache_material_provider)
                else None
            )
            key = _digest(
                {
                    "adapter": identity,
                    "stage": stage,
                    "candidate": candidate.as_dict(),
                    "segment": segment.as_dict(),
                    "adapterCacheMaterial": cache_material,
                }
            )
            keys[candidate.segment_id] = key
            proposal, hit, corrupted = self._cache_item(
                stage, key, ReviewProposal.from_mapping
            )
            if hit:
                proposals[candidate.segment_id] = proposal
                cache_hits[candidate.segment_id] = True
                latencies[candidate.segment_id] = 0.0
            else:
                misses.append(candidate)
                cache_hits[candidate.segment_id] = False
                recomputations += int(corrupted)
        for offset in range(0, len(misses), self.config.max_batch_size):
            context.raise_if_cancelled()
            batch = misses[offset : offset + self.config.max_batch_size]
            batch_started = time.perf_counter()
            raw = adapter.review_batch(batch, by_segment, context)
            batch_elapsed = (time.perf_counter() - batch_started) * 1000.0
            converted = _coerce_review_batch(
                raw,
                expected_ids={candidate.segment_id for candidate in batch},
                label=stage,
            )
            per_item_latency = batch_elapsed / len(batch)
            for candidate in batch:
                proposal = converted[candidate.segment_id]
                proposals[candidate.segment_id] = proposal
                latencies[candidate.segment_id] = per_item_latency
                self.cache.write(
                    stage,
                    keys[candidate.segment_id],
                    proposal.as_dict(),
                )
        metrics.record_cache(
            stage,
            requests=len(candidates),
            hits=len(candidates) - len(misses),
            misses=len(misses),
            recomputations=recomputations,
        )
        metrics.record_stage(
            stage, (time.perf_counter() - started) * 1000.0
        )
        return proposals, keys, cache_hits, latencies, identity

    def _attach_overlap_review_evidence(
        self,
        segments: Sequence[TranscriptSegment],
        candidates: Sequence[ReviewCandidate],
    ) -> tuple[TranscriptSegment, ...]:
        overlap_candidates = {
            candidate.segment_id: candidate
            for candidate in candidates
            if "OVERLAP" in candidate.reasons
            or _OVERLAP_DETECTOR_UNAVAILABLE_REASON in candidate.reasons
        }
        output: list[TranscriptSegment] = []
        for segment in segments:
            candidate = overlap_candidates.get(segment.segment_id)
            if candidate is None:
                output.append(segment)
                continue
            detector_unavailable = (
                _OVERLAP_DETECTOR_UNAVAILABLE_REASON in candidate.reasons
                and "OVERLAP" not in candidate.reasons
            )
            reason_code = (
                _OVERLAP_DETECTOR_UNAVAILABLE_REASON
                if detector_unavailable
                else "OVERLAP_REQUIRES_HUMAN_REVIEW"
            )
            exit_reason = (
                _OVERLAP_DETECTOR_UNAVAILABLE_REASON
                if detector_unavailable
                else "HUMAN_REVIEW_REQUIRED"
            )
            evidence_prefix = (
                "overlap-detector" if detector_unavailable else "overlap"
            )
            output.append(
                replace(
                    segment,
                    evidence={
                        **dict(segment.evidence),
                        "selectiveReview": {
                            "provider": {
                                "id": "overlap-review-policy",
                                "version": "2",
                            },
                            "reasonCode": reason_code,
                            "speakerProtected": candidate.protected,
                            "reasons": list(candidate.reasons),
                            "evidenceRefs": [
                                f"{evidence_prefix}:{segment.segment_id}"
                            ],
                            "reviewStatus": "REVIEW_REQUIRED",
                            "exitReason": exit_reason,
                            "applied": False,
                        },
                    },
                )
            )
        return tuple(output)

    def _attach_unresolved_review_evidence(
        self,
        segments: Sequence[TranscriptSegment],
        candidates: Sequence[ReviewCandidate],
        *,
        provider_id: str,
        provider_version: str,
        reason_code: str,
        exit_reason: str,
    ) -> tuple[TranscriptSegment, ...]:
        """Persist routing failures without mutating transcript state.

        Every difficult segment must either receive an actual verifier proposal
        or carry explicit, auditable evidence explaining why verification could
        not be performed.  This is deliberately fail-closed: the absence of a
        secondary model, a protected speaker decision, or an exhausted budget
        can never look like a successful verification.
        """

        candidate_by_id = {
            candidate.segment_id: candidate for candidate in candidates
        }
        output: list[TranscriptSegment] = []
        for segment in segments:
            candidate = candidate_by_id.get(segment.segment_id)
            if candidate is None:
                output.append(segment)
                continue
            existing = segment.evidence.get("selectiveReview")
            if isinstance(existing, Mapping):
                output.append(segment)
                continue
            output.append(
                replace(
                    segment,
                    evidence={
                        **dict(segment.evidence),
                        "selectiveReview": {
                            "provider": {
                                "id": provider_id,
                                "version": provider_version,
                            },
                            "reasonCode": reason_code,
                            "speakerProtected": candidate.protected,
                            "reasons": list(candidate.reasons),
                            "evidenceRefs": [
                                f"review-policy:{segment.segment_id}"
                            ],
                            "reviewStatus": "REVIEW_REQUIRED",
                            "exitReason": exit_reason,
                            "applied": False,
                        },
                    },
                )
            )
        return tuple(output)

    def _apply_secondary_review(
        self,
        *,
        segments: Sequence[TranscriptSegment],
        candidates: Sequence[ReviewCandidate],
        proposals: Mapping[str, ReviewProposal],
        keys: Mapping[str, str],
        cache_hits: Mapping[str, bool],
        latencies: Mapping[str, float],
        identity: Mapping[str, str],
        metrics: PipelineMetricsCollector,
    ) -> tuple[tuple[TranscriptSegment, ...], list[ReviewCandidate]]:
        by_segment = {
            segment.segment_id: segment for segment in segments
        }
        speaker_support: dict[str, int] = {}
        for segment in segments:
            speaker_support[segment.speaker_id] = (
                speaker_support.get(segment.speaker_id, 0) + 1
            )
        updated_by_id = dict(by_segment)
        unresolved_ids: set[str] = set()
        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: (
                by_segment[candidate.segment_id].start_ms,
                by_segment[candidate.segment_id].end_ms,
                candidate.segment_id,
            ),
        )
        for candidate in ordered_candidates:
            segment = updated_by_id[candidate.segment_id]
            proposal = proposals.get(candidate.segment_id)
            if proposal is None:
                unresolved_ids.add(candidate.segment_id)
                exit_reason = "SECONDARY_PROPOSAL_MISSING"
                cache_key = keys.get(candidate.segment_id) or _digest(
                    {
                        "stage": "secondary-review",
                        "segmentId": candidate.segment_id,
                        "missingProposal": True,
                    }
                )
                cache_hit = bool(cache_hits.get(candidate.segment_id, False))
                latency_ms = float(latencies.get(candidate.segment_id, 0.0))
                provider = str(identity.get("id") or "ERes2NetV2")
                metrics.record_escalation(
                    stage="secondary-review",
                    segment_id=candidate.segment_id,
                    reasons=candidate.reasons,
                    cache_key=cache_key,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    resource={},
                    confidence=0.0,
                    exit_reason=exit_reason,
                    provider=provider,
                    applied=False,
                )
                updated_by_id[candidate.segment_id] = replace(
                    segment,
                    evidence={
                        **dict(segment.evidence),
                        "selectiveReview": {
                            "provider": {
                                "id": provider,
                                "version": str(
                                    identity.get("version") or "unknown"
                                ),
                            },
                            "reasonCode": exit_reason,
                            "speakerProtected": candidate.protected,
                            "reasons": list(candidate.reasons),
                            "evidenceRefs": [
                                f"secondary-missing:{candidate.segment_id}"
                            ],
                            "cacheKey": cache_key,
                            "cacheHit": cache_hit,
                            "latencyMs": round(latency_ms, 6),
                            "resource": {},
                            "confidence": 0.0,
                            "reviewStatus": "REVIEW_REQUIRED",
                            "exitReason": exit_reason,
                            "applied": False,
                        },
                    },
                )
                continue
            if proposal.source not in {"acoustic", "deterministic"}:
                raise WorkerError(
                    "SECONDARY_VERIFIER_MUTATION_FORBIDDEN",
                    "secondary verifier proposals must be acoustic/deterministic",
                    details={"segmentId": segment.segment_id},
                )
            if (
                proposal.normalized_text is not None
                or proposal.display_text is not None
                or proposal.overlapping is not None
            ):
                raise WorkerError(
                    "SECONDARY_VERIFIER_MUTATION_FORBIDDEN",
                    "secondary verifier cannot modify text, boundary, turn, or overlap",
                    details={"segmentId": segment.segment_id},
                )
            if not proposal.evidence_refs:
                raise WorkerError(
                    "SECONDARY_VERIFIER_EVIDENCE_REQUIRED",
                    "secondary verifier proposals require evidence references",
                    details={"segmentId": segment.segment_id},
                )

            revisions = list(segment.revisions)
            speaker_id = segment.speaker_id
            protected_speaker = (
                candidate.protected
                or segment.human_locked
                or segment.overlapping
                or segment.speaker_margin >= self.config.high_margin_threshold
            )
            applied = False
            review_status = "REVIEW_REQUIRED"
            exit_reason = proposal.exit_reason
            proposed_speaker = proposal.speaker_id

            if proposal.exit_reason not in _ERES_RESOLVED_EXIT_REASONS:
                unresolved_ids.add(segment.segment_id)
            elif proposal.exit_reason == "VERIFIED_NO_CHANGE":
                if proposed_speaker is None or proposed_speaker == speaker_id:
                    review_status = "RESOLVED"
                else:
                    unresolved_ids.add(segment.segment_id)
                    exit_reason = (
                        "SECONDARY_SEMANTIC_MISMATCH_REVIEW_REQUIRED"
                    )
            elif (
                proposed_speaker is None
                or proposed_speaker == speaker_id
            ):
                unresolved_ids.add(segment.segment_id)
                exit_reason = "SECONDARY_SEMANTIC_MISMATCH_REVIEW_REQUIRED"
            else:
                if protected_speaker:
                    unresolved_ids.add(segment.segment_id)
                    exit_reason = "PROTECTED_SPEAKER_REVIEW_REQUIRED"
                else:
                    ranked = sorted(
                        segment.speaker_scores,
                        key=lambda item: (-item.score, item.speaker_id),
                    )
                    top_two = {
                        item.speaker_id for item in ranked[:2]
                    }
                    if proposed_speaker not in top_two:
                        unresolved_ids.add(segment.segment_id)
                        exit_reason = "TOP2_CONSTRAINT_REVIEW_REQUIRED"
                    elif speaker_support.get(speaker_id, 0) <= 1:
                        unresolved_ids.add(segment.segment_id)
                        exit_reason = (
                            "CARDINALITY_CHANGE_REVIEW_REQUIRED"
                        )
                    else:
                        revisions.append(
                            Revision(
                                revision_id=(
                                    f"{segment.segment_id}:speaker:"
                                    f"{len(revisions) + 1}"
                                ),
                                revision_type="speaker",
                                source=proposal.source,
                                before=speaker_id,
                                after=proposed_speaker,
                                reason_code=proposal.reason_code,
                                confidence=proposal.confidence,
                                evidence_refs=proposal.evidence_refs,
                            )
                        )
                        speaker_support[speaker_id] -= 1
                        speaker_support[proposed_speaker] = (
                            speaker_support.get(proposed_speaker, 0) + 1
                        )
                        speaker_id = proposed_speaker
                        applied = True
                        review_status = "RESOLVED"

            cache_key = keys.get(segment.segment_id) or _digest(
                {
                    "stage": "secondary-review",
                    "segmentId": segment.segment_id,
                    "proposal": proposal.as_dict(),
                }
            )
            cache_hit = bool(cache_hits.get(segment.segment_id, False))
            latency_ms = float(latencies.get(segment.segment_id, 0.0))
            provider = str(identity.get("id") or "ERes2NetV2")
            metrics.record_escalation(
                stage="secondary-review",
                segment_id=segment.segment_id,
                reasons=candidate.reasons,
                cache_key=cache_key,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                resource=proposal.resource,
                confidence=proposal.confidence,
                exit_reason=exit_reason,
                provider=provider,
                applied=applied,
            )
            updated_by_id[segment.segment_id] = replace(
                segment,
                speaker_id=speaker_id,
                revisions=tuple(revisions),
                evidence={
                    **dict(segment.evidence),
                    "selectiveReview": {
                        "provider": {
                            "id": provider,
                            "version": str(
                                identity.get("version") or "unknown"
                            ),
                        },
                        "reasonCode": proposal.reason_code,
                        "speakerProtected": protected_speaker,
                        "reasons": list(candidate.reasons),
                        "evidenceRefs": list(proposal.evidence_refs),
                        "cacheKey": cache_key,
                        "cacheHit": cache_hit,
                        "latencyMs": round(latency_ms, 6),
                        "resource": dict(proposal.resource),
                        "confidence": proposal.confidence,
                        "reviewStatus": review_status,
                        "proposalExitReason": proposal.exit_reason,
                        "exitReason": exit_reason,
                        "applied": applied,
                    },
                },
            )

        unresolved = [
            candidate
            for candidate in candidates
            if candidate.segment_id in unresolved_ids
        ]
        return (
            tuple(
                updated_by_id[segment.segment_id] for segment in segments
            ),
            unresolved,
        )

    def _attach_pyannote_review(
        self,
        *,
        stage: str,
        segments: Sequence[TranscriptSegment],
        candidates: Sequence[ReviewCandidate],
        proposals: Mapping[str, ReviewProposal],
        keys: Mapping[str, str],
        cache_hits: Mapping[str, bool],
        latencies: Mapping[str, float],
        identity: Mapping[str, str],
        metrics: PipelineMetricsCollector,
    ) -> tuple[TranscriptSegment, ...]:
        by_candidate = {
            candidate.segment_id: candidate for candidate in candidates
        }
        evidence_key = "pyannoteFallback"
        output: list[TranscriptSegment] = []
        for segment in segments:
            proposal = proposals.get(segment.segment_id)
            if proposal is None:
                output.append(segment)
                continue
            if (
                proposal.speaker_id is not None
                or proposal.normalized_text is not None
                or proposal.display_text is not None
                or proposal.overlapping is not None
            ):
                raise WorkerError(
                    "PYANNOTE_AUDIT_MUTATION_FORBIDDEN",
                    "pyannote review cannot mutate transcript state",
                    details={"segmentId": segment.segment_id, "stage": stage},
                )
            if not proposal.evidence_refs:
                raise WorkerError(
                    "PYANNOTE_EVIDENCE_REQUIRED",
                    "pyannote review must retain auditable evidence",
                    details={"segmentId": segment.segment_id, "stage": stage},
                )
            candidate = by_candidate[segment.segment_id]
            cache_key = keys.get(segment.segment_id) or _digest(
                {
                    "stage": stage,
                    "segmentId": segment.segment_id,
                    "proposal": proposal.as_dict(),
                }
            )
            cache_hit = bool(cache_hits.get(segment.segment_id, False))
            latency_ms = float(latencies.get(segment.segment_id, 0.0))
            provider = str(identity.get("id") or "pyannote-community-1")
            metrics.record_escalation(
                stage=stage,
                segment_id=segment.segment_id,
                reasons=candidate.reasons,
                cache_key=cache_key,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                resource=proposal.resource,
                confidence=proposal.confidence,
                exit_reason=proposal.exit_reason,
                provider=provider,
                applied=False,
            )
            output.append(
                replace(
                    segment,
                    evidence={
                        **dict(segment.evidence),
                        evidence_key: {
                            "provider": dict(identity),
                            "reasonCode": proposal.reason_code,
                            "reasons": list(candidate.reasons),
                            "evidenceRefs": list(proposal.evidence_refs),
                            "cacheKey": cache_key,
                            "cacheHit": cache_hit,
                            "latencyMs": round(latency_ms, 6),
                            "resource": dict(proposal.resource),
                            "confidence": proposal.confidence,
                            "reviewStatus": "REVIEW_REQUIRED",
                            "exitReason": proposal.exit_reason,
                            "applied": False,
                        },
                    },
                )
            )
        return tuple(output)

    def _review_stage(
        self,
        segments: tuple[TranscriptSegment, ...],
        candidates: Sequence[ReviewCandidate],
        metrics: PipelineMetricsCollector,
        context: AdapterContext,
    ) -> tuple[TranscriptSegment, ...]:
        started = time.perf_counter()
        reason_counts = {
            reason: sum(reason in candidate.reasons for candidate in candidates)
            for reason in _REVIEW_REASONS
        }
        protected = sum(candidate.protected for candidate in candidates)
        output = self._attach_overlap_review_evidence(segments, candidates)
        refinement_only_candidates = [
            candidate
            for candidate in candidates
            if (
                _SPEAKER_CHANGE_REFINEMENT_REVIEW_REASON
                in candidate.reasons
                and not any(
                    reason not in _SECONDARY_REVIEW_EXCLUSION_REASONS
                    for reason in candidate.reasons
                )
            )
        ]
        output = self._attach_unresolved_review_evidence(
            output,
            refinement_only_candidates,
            provider_id="speaker-change-refinement-review-policy",
            provider_version="1",
            reason_code=_SPEAKER_CHANGE_REFINEMENT_REVIEW_REASON,
            exit_reason=_SPEAKER_CHANGE_REFINEMENT_REVIEW_REASON,
        )
        by_segment = {segment.segment_id: segment for segment in output}
        secondary_candidates = [
            candidate
            for candidate in candidates
            if any(
                reason not in _SECONDARY_REVIEW_EXCLUSION_REASONS
                for reason in candidate.reasons
            )
        ]
        unprotected = [
            candidate
            for candidate in secondary_candidates
            if not candidate.protected
        ]
        maximum_secondary = math.floor(
            len(segments) * self.config.max_secondary_fraction
        )
        ranked_eligible = sorted(
            unprotected,
            key=lambda candidate: (
                0
                if "TEMPORAL_DISCONTINUITY" in candidate.reasons
                else 1,
                0 if "BOUNDARY_CONFLICT" in candidate.reasons else 1,
                0 if "SHORT_SEGMENT" in candidate.reasons else 1,
                0 if "LOW_MARGIN" in candidate.reasons else 1,
                by_segment[candidate.segment_id].speaker_margin,
                by_segment[candidate.segment_id].start_ms,
                candidate.segment_id,
            ),
        )
        eligible = ranked_eligible[:maximum_secondary]
        overflow = ranked_eligible[maximum_secondary:]
        output = self._attach_unresolved_review_evidence(
            output,
            [
                candidate
                for candidate in candidates
                if candidate.protected
                and "OVERLAP" not in candidate.reasons
                and _OVERLAP_DETECTOR_UNAVAILABLE_REASON
                not in candidate.reasons
            ],
            provider_id="review-routing-policy",
            provider_version="1",
            reason_code="PROTECTED_REQUIRES_HUMAN_REVIEW",
            exit_reason="PROTECTED_REQUIRES_HUMAN_REVIEW",
        )
        output = self._attach_unresolved_review_evidence(
            output,
            overflow,
            provider_id="review-routing-policy",
            provider_version="1",
            reason_code="SECONDARY_BUDGET_EXHAUSTED",
            exit_reason="SECONDARY_BUDGET_EXHAUSTED",
        )
        metrics.record_routing(
            segments=len(segments),
            escalated=len(eligible),
            protected=protected,
            reason_counts=reason_counts,
        )

        secondary_started = time.perf_counter()
        secondary_proposals: Mapping[str, ReviewProposal] = {}
        secondary_identity = (
            _adapter_identity(self.secondary_adapter)
            if self.secondary_adapter is not None
            else {"id": "ERes2NetV2", "version": "unavailable"}
        )
        secondary_invoked = False
        unresolved: list[ReviewCandidate] = []
        if self.secondary_adapter is not None and eligible:
            try:
                (
                    proposals,
                    keys,
                    cache_hits,
                    latencies,
                    identity,
                ) = self._run_review_adapter(
                    stage="secondary-review",
                    adapter=self.secondary_adapter,
                    candidates=eligible,
                    segments=output,
                    metrics=metrics,
                    context=context,
                )
                secondary_proposals = proposals
                secondary_identity = identity
                secondary_invoked = True
                output, unresolved = self._apply_secondary_review(
                    segments=output,
                    candidates=eligible,
                    proposals=proposals,
                    keys=keys,
                    cache_hits=cache_hits,
                    latencies=latencies,
                    identity=identity,
                    metrics=metrics,
                )
            except Exception:
                _release_adapter_resources(
                    self.secondary_adapter,
                    suppress_errors=True,
                )
                raise
            self._release_after_success(self.secondary_adapter)
        elif eligible:
            output = self._attach_unresolved_review_evidence(
                output,
                eligible,
                provider_id=secondary_identity["id"],
                provider_version=secondary_identity["version"],
                reason_code="SECONDARY_ADAPTER_DISABLED",
                exit_reason="SECONDARY_ADAPTER_DISABLED",
            )
        secondary_elapsed = (
            time.perf_counter() - secondary_started
        ) * 1000.0
        secondary_ids = [candidate.segment_id for candidate in eligible]
        secondary_start, secondary_end = self._candidate_time_bounds(
            secondary_ids, by_segment
        )
        secondary_resource, secondary_confidence = self._review_stage_summary(
            secondary_proposals
        )
        if secondary_invoked:
            secondary_trigger = "CAMPP_DIFFICULT_SEGMENTS"
            secondary_exit = (
                "COMPLETED_WITH_UNRESOLVED" if unresolved else "COMPLETED"
            )
        elif maximum_secondary == 0 and unprotected:
            secondary_trigger = "MAX_SECONDARY_FRACTION"
            secondary_exit = "SECONDARY_FRACTION_BUDGET_ZERO"
        elif not secondary_candidates and candidates:
            secondary_trigger = "HUMAN_REVIEW_ONLY_EVIDENCE"
            secondary_exit = "NO_ACOUSTIC_ESCALATION_REQUIRED"
        elif not unprotected and candidates:
            secondary_trigger = "PROTECTED_SEGMENTS_EXCLUDED"
            secondary_exit = "ALL_DIFFICULT_CANDIDATES_PROTECTED"
        elif not unprotected:
            secondary_trigger = "CAMPP_DIFFICULT_SEGMENTS"
            secondary_exit = "NO_DIFFICULT_CANDIDATES"
        else:
            secondary_trigger = "CAMPP_DIFFICULT_SEGMENTS"
            secondary_exit = "ADAPTER_DISABLED"
        metrics.record_cascade_stage(
            stage="secondary-review",
            provider=secondary_identity["id"],
            trigger_reason=secondary_trigger,
            candidate_ids=secondary_ids,
            candidate_scope="campp-difficult-segments-only",
            source_count=len(segments),
            max_candidates=maximum_secondary,
            candidate_start_ms=secondary_start,
            candidate_end_ms=secondary_end,
            invoked=secondary_invoked,
            cache_stage="secondary-review",
            latency_ms=secondary_elapsed,
            resource=secondary_resource,
            confidence=secondary_confidence,
            exit_reason=secondary_exit,
        )

        pyannote_candidates: Sequence[ReviewCandidate] = ()
        pyannote_stage = "pyannote-fallback"
        pyannote_trigger = "PYANNOTE_MODE_DISABLED"
        pyannote_exit = "DISABLED"
        if self.config.pyannote_mode == "fallback":
            if self.secondary_adapter is None:
                raise WorkerError(
                    "SPEAKER_CASCADE_ORDER_VIOLATION",
                    "pyannote fallback cannot bypass ERes2NetV2",
                )
            eligible_order = [
                candidate.segment_id for candidate in eligible
            ]
            unresolved_order = [
                candidate.segment_id for candidate in unresolved
            ]
            if len(set(eligible_order)) != len(eligible_order):
                raise WorkerError(
                    "SPEAKER_CASCADE_ORDER_VIOLATION",
                    "ERes2NetV2 candidate sequence contains duplicates",
                    details={
                        "eresCandidateIds": eligible_order,
                    },
                )
            if len(set(unresolved_order)) != len(unresolved_order):
                raise WorkerError(
                    "SPEAKER_CASCADE_ORDER_VIOLATION",
                    "unresolved ERes2NetV2 sequence contains duplicates",
                    details={
                        "unresolvedIds": unresolved_order,
                    },
                )
            unresolved_set = set(unresolved_order)
            expected_unresolved_order = [
                segment_id
                for segment_id in eligible_order
                if segment_id in unresolved_set
            ]
            if unresolved_order != expected_unresolved_order:
                raise WorkerError(
                    "SPEAKER_CASCADE_ORDER_VIOLATION",
                    "pyannote fallback must receive the exact ordered unresolved ERes subsequence",
                    details={
                        "eresCandidateIds": eligible_order,
                        "unresolvedIds": unresolved_order,
                        "expectedUnresolvedIds": expected_unresolved_order,
                    },
                )
            pyannote_candidates = unresolved
            pyannote_trigger = "ERES_UNRESOLVED_ONLY"
            pyannote_exit = (
                "PENDING"
                if pyannote_candidates
                else "NO_UNRESOLVED_AFTER_ERES"
            )
        pyannote_started = time.perf_counter()
        pyannote_proposals: Mapping[str, ReviewProposal] = {}
        pyannote_identity = (
            _adapter_identity(self.pyannote_adapter)
            if self.pyannote_adapter is not None
            else {"id": "pyannote-community-1", "version": "unavailable"}
        )
        pyannote_invoked = False
        if (
            pyannote_candidates
            and self.pyannote_adapter is not None
            and pyannote_stage
        ):
            try:
                (
                    proposals,
                    keys,
                    cache_hits,
                    latencies,
                    identity,
                ) = self._run_review_adapter(
                    stage=pyannote_stage,
                    adapter=self.pyannote_adapter,
                    candidates=pyannote_candidates,
                    segments=output,
                    metrics=metrics,
                    context=context,
                )
                pyannote_proposals = proposals
                pyannote_identity = identity
                pyannote_invoked = True
                output = self._attach_pyannote_review(
                    stage=pyannote_stage,
                    segments=output,
                    candidates=pyannote_candidates,
                    proposals=proposals,
                    keys=keys,
                    cache_hits=cache_hits,
                    latencies=latencies,
                    identity=identity,
                    metrics=metrics,
                )
                pyannote_exit = "COMPLETED"
            except Exception:
                _release_adapter_resources(
                    self.pyannote_adapter,
                    suppress_errors=True,
                )
                raise
            self._release_after_success(self.pyannote_adapter)
        elif pyannote_candidates and self.pyannote_adapter is None:
            pyannote_exit = "ADAPTER_DISABLED"
        pyannote_elapsed = (
            time.perf_counter() - pyannote_started
        ) * 1000.0
        pyannote_ids = [
            candidate.segment_id for candidate in pyannote_candidates
        ]
        pyannote_start, pyannote_end = self._candidate_time_bounds(
            pyannote_ids, by_segment
        )
        pyannote_resource, pyannote_confidence = self._review_stage_summary(
            pyannote_proposals
        )
        metrics.record_cascade_stage(
            stage=pyannote_stage,
            provider=pyannote_identity["id"],
            trigger_reason=pyannote_trigger,
            candidate_ids=pyannote_ids,
            candidate_scope=(
                "eres-unresolved-only"
                if self.config.pyannote_mode == "fallback"
                else "disabled"
            ),
            source_count=len(eligible) if self.config.pyannote_mode == "fallback" else 0,
            max_candidates=len(eligible) if self.config.pyannote_mode == "fallback" else 0,
            candidate_start_ms=pyannote_start,
            candidate_end_ms=pyannote_end,
            invoked=pyannote_invoked,
            cache_stage=pyannote_stage,
            latency_ms=pyannote_elapsed,
            resource=pyannote_resource,
            confidence=pyannote_confidence,
            exit_reason=pyannote_exit,
        )
        metrics.record_stage(
            "review", (time.perf_counter() - started) * 1000.0
        )
        return tuple(output)

    def _overlap_recovery_intervals(
        self,
        prepared: PreparedAudio,
        overlap: Sequence[OverlapDecision],
    ) -> tuple[OverlapRecoveryInterval, ...]:
        observed: dict[
            tuple[int, int, tuple[str, str]],
            OverlapRecoveryInterval,
        ] = {}
        for decision in overlap:
            raw_intervals = decision.evidence.get("overlapIntervals")
            if not isinstance(raw_intervals, Sequence) or isinstance(
                raw_intervals,
                (str, bytes, bytearray),
            ):
                continue
            for raw in raw_intervals:
                if not isinstance(raw, Mapping):
                    continue
                start_ms = raw.get("startMs")
                end_ms = raw.get("endMs")
                raw_speakers = raw.get("localSpeakers")
                if (
                    isinstance(start_ms, bool)
                    or not isinstance(start_ms, int)
                    or isinstance(end_ms, bool)
                    or not isinstance(end_ms, int)
                    or start_ms < 0
                    or end_ms <= start_ms
                    or end_ms > prepared.duration_ms
                    or end_ms - start_ms
                    > self.config.overlap_recovery_max_interval_ms
                    or not isinstance(raw_speakers, Sequence)
                    or isinstance(raw_speakers, (str, bytes, bytearray))
                ):
                    continue
                local_speakers = tuple(
                    sorted(
                        {
                            str(item).strip()
                            for item in raw_speakers
                            if str(item).strip()
                        }
                    )
                )
                if len(local_speakers) != 2:
                    continue
                pair = (local_speakers[0], local_speakers[1])
                key = (start_ms, end_ms, pair)
                padding = self.config.overlap_recovery_padding_ms
                observed[key] = OverlapRecoveryInterval(
                    interval_id=(
                        f"overlap-{start_ms:08d}-{end_ms:08d}"
                    ),
                    detected_start_ms=start_ms,
                    detected_end_ms=end_ms,
                    context_start_ms=max(0, start_ms - padding),
                    context_end_ms=min(
                        prepared.duration_ms,
                        end_ms + padding,
                    ),
                    local_speakers=pair,
                )
        return tuple(
            sorted(
                observed.values(),
                key=lambda item: (
                    item.detected_start_ms,
                    item.detected_end_ms,
                    item.local_speakers,
                ),
            )[: self.config.overlap_recovery_max_intervals]
        )

    @staticmethod
    def _canonical_centroids(
        embeddings: Sequence[EmbeddingRecord],
        clusters: _ClusterResult,
    ) -> tuple[tuple[float, ...], ...]:
        if (
            not embeddings
            or len(embeddings) != len(clusters.assignments)
            or clusters.count < 1
        ):
            raise WorkerError(
                "OVERLAP_RECOVERY_IDENTITY_INVALID",
                "overlap recovery requires complete canonical voiceprints",
            )
        vectors = tuple(_normalize(item.vector) for item in embeddings)
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise WorkerError(
                "OVERLAP_RECOVERY_IDENTITY_INVALID",
                "canonical voiceprints have inconsistent dimensions",
            )
        members: list[list[tuple[float, ...]]] = [
            [] for _ in range(clusters.count)
        ]
        for vector, assignment in zip(vectors, clusters.assignments):
            members[assignment].append(vector)
        if any(not items for items in members):
            raise WorkerError(
                "OVERLAP_RECOVERY_IDENTITY_INVALID",
                "a canonical speaker has no enrollment voiceprint",
            )
        return tuple(
            _mean_vector(items, dimension)
            for items in members
        )

    @staticmethod
    def _separated_prepared_audio(
        channel: SeparatedSpeechChannel,
    ) -> tuple[PreparedAudio, SpeechWindow]:
        window = SpeechWindow(
            window_id=f"{channel.candidate_id}.source",
            start_ms=0,
            end_ms=channel.duration_ms,
            metadata={"overlapRecoveryCandidateId": channel.candidate_id},
        )
        return (
            PreparedAudio(
                duration_ms=channel.duration_ms,
                source_fingerprint=channel.audio_sha256,
                normalization_profile="mossformer2-separated-16khz-v1",
                windows=(window,),
                stage_durations_ms={
                    "decode": 0.0,
                    "normalize": 0.0,
                    "vad": 0.0,
                    "boundary": 0.0,
                },
                audio_path=channel.audio_path,
            ),
            window,
        )

    def _overlap_recovery_asr_token_budget(self, duration_ms: int) -> int:
        """Bound separated-channel decoding by speech duration.

        The fixed allowance covers Qwen's language metadata and short lexical
        bursts. Eight tokens per second is intentionally generous for fast
        multilingual speech while preventing non-EOS noise from consuming the
        general 512-token ASR ceiling.
        """

        duration_seconds = max(0.001, duration_ms / 1000.0)
        estimated = 16 + math.ceil(duration_seconds * 8.0)
        return min(
            self.config.overlap_recovery_asr_max_new_tokens,
            max(24, estimated),
        )

    @staticmethod
    def _shared_full_timeline_inference(
        segments: Sequence[TranscriptSegment],
    ) -> Mapping[str, Any] | None:
        serialized: str | None = None
        shared: Mapping[str, Any] | None = None
        for segment in segments:
            overlap = segment.evidence.get("overlap")
            full_timeline = (
                overlap.get("fullTimelineInference")
                if isinstance(overlap, Mapping)
                else None
            )
            if not isinstance(full_timeline, Mapping):
                return None
            current = json.dumps(
                full_timeline,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if serialized is not None and current != serialized:
                return None
            serialized = current
            shared = full_timeline
        return dict(shared) if shared is not None else None

    def _recover_overlap_speech(
        self,
        *,
        prepared: PreparedAudio,
        overlap: Sequence[OverlapDecision],
        embeddings: Sequence[EmbeddingRecord],
        clusters: _ClusterResult,
        segments: Sequence[TranscriptSegment],
        requested_language: str,
        context: AdapterContext,
        metrics: PipelineMetricsCollector,
    ) -> tuple[TranscriptSegment, ...]:
        if self.config.overlap_recovery_mode != "guarded":
            metrics.set_policy(
                overlapRecoveryMode="disabled",
                overlapRecoveryIntervalCount=0,
                overlapRecoveryPublishedCount=0,
            )
            return tuple(segments)
        if self.separation_adapter is None:
            raise WorkerError(
                "OVERLAP_RECOVERY_ADAPTER_MISSING",
                "guarded overlap recovery requires a separation adapter",
            )
        intervals = self._overlap_recovery_intervals(prepared, overlap)
        metrics.set_policy(
            overlapRecoveryMode="guarded",
            overlapRecoveryIntervalCount=len(intervals),
            overlapRecoveryMarginThreshold=(
                self.config.overlap_recovery_margin_threshold
            ),
        )
        if not intervals:
            metrics.set_policy(overlapRecoveryPublishedCount=0)
            return tuple(segments)

        started = time.perf_counter()
        try:
            raw_channels = self.separation_adapter.separate_batch(
                prepared,
                intervals,
                context,
            )
        except Exception:
            _release_adapter_resources(
                self.separation_adapter,
                suppress_errors=True,
            )
            raise
        self._release_after_success(self.separation_adapter)
        channels = tuple(raw_channels)
        expected = {
            (interval.interval_id, channel_index)
            for interval in intervals
            for channel_index in (1, 2)
        }
        observed = {
            (channel.interval_id, channel.channel_index)
            for channel in channels
            if isinstance(channel, SeparatedSpeechChannel)
        }
        if (
            len(channels) != len(expected)
            or any(
                not isinstance(channel, SeparatedSpeechChannel)
                for channel in channels
            )
            or observed != expected
        ):
            raise WorkerError(
                "OVERLAP_RECOVERY_RESULT_INVALID",
                "separator must return exactly two channels per interval",
            )
        separation_elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0
        metrics.record_stage(
            "overlap-recovery-separation",
            separation_elapsed_ms,
        )
        metrics.record_cascade_stage(
            stage="overlap-recovery-separation",
            provider=_adapter_identity(self.separation_adapter)["id"],
            trigger_reason="EXACT_TWO_SPEAKER_OVERLAP",
            candidate_ids=[interval.interval_id for interval in intervals],
            candidate_scope="pyannote-exact-two-speaker-overlap",
            source_count=len(intervals),
            max_candidates=self.config.overlap_recovery_max_intervals,
            candidate_start_ms=min(
                item.detected_start_ms for item in intervals
            ),
            candidate_end_ms=max(
                item.detected_end_ms for item in intervals
            ),
            invoked=True,
            cache_stage="overlap-recovery-separation",
            latency_ms=separation_elapsed_ms,
            resource=None,
            confidence=None,
            exit_reason="COMPLETED",
        )

        interval_by_id = {
            interval.interval_id: interval for interval in intervals
        }
        full_timeline_inference = self._shared_full_timeline_inference(
            segments
        )
        centroids = self._canonical_centroids(embeddings, clusters)
        gated: list[
            tuple[
                SeparatedSpeechChannel,
                OverlapRecoveryInterval,
                PreparedAudio,
                SpeechWindow,
                EmbeddingRecord,
                tuple[float, ...],
                int,
                float,
            ]
        ] = []
        voiceprint_started = time.perf_counter()
        try:
            for channel in channels:
                context.raise_if_cancelled()
                separated, window = self._separated_prepared_audio(channel)
                raw = self.embedding_adapter.embed_batch(
                    separated,
                    (window,),
                    context,
                )
                if len(raw) != 1:
                    raise WorkerError(
                        "OVERLAP_RECOVERY_IDENTITY_INVALID",
                        "CAM++ must return one voiceprint per separated channel",
                    )
                embedding = (
                    raw[0]
                    if isinstance(raw[0], EmbeddingRecord)
                    else EmbeddingRecord.from_mapping(raw[0])
                )
                vector = _normalize(embedding.vector)
                if any(len(vector) != len(item) for item in centroids):
                    raise WorkerError(
                        "OVERLAP_RECOVERY_IDENTITY_INVALID",
                        "separated and canonical voiceprints have different dimensions",
                    )
                scores = tuple(
                    max(-1.0, min(1.0, _dot(vector, centroid)))
                    for centroid in centroids
                )
                ranked = sorted(
                    enumerate(scores),
                    key=lambda item: (-item[1], item[0]),
                )
                assigned = ranked[0][0]
                margin = (
                    ranked[0][1] - ranked[1][1]
                    if len(ranked) > 1
                    else 2.0
                )
                interval = interval_by_id[channel.interval_id]
                active_primary = {
                    segment.speaker_id
                    for segment in segments
                    if segment.start_ms < interval.detected_end_ms
                    and segment.end_ms > interval.detected_start_ms
                }
                speaker_id = f"speaker-{assigned + 1}"
                if (
                    margin
                    < self.config.overlap_recovery_margin_threshold
                    or speaker_id in active_primary
                ):
                    continue
                gated.append(
                    (
                        channel,
                        interval,
                        separated,
                        window,
                        embedding,
                        scores,
                        assigned,
                        margin,
                    )
                )
        finally:
            self._release_after_success(self.embedding_adapter)
        voiceprint_elapsed_ms = (
            time.perf_counter() - voiceprint_started
        ) * 1000.0
        metrics.record_stage(
            "overlap-recovery-voiceprint",
            voiceprint_elapsed_ms,
        )

        token_budgets = {
            channel.candidate_id: self._overlap_recovery_asr_token_budget(
                channel.duration_ms
            )
            for channel, *_rest in gated
        }
        metrics.set_policy(
            overlapRecoveryAsrCandidateCount=len(gated),
            overlapRecoveryAsrMaxNewTokens=(
                self.config.overlap_recovery_asr_max_new_tokens
            ),
            overlapRecoveryAsrMinTokenBudgetApplied=(
                min(token_budgets.values()) if token_budgets else None
            ),
            overlapRecoveryAsrMaxTokenBudgetApplied=(
                max(token_budgets.values()) if token_budgets else None
            ),
        )
        recovered: list[TranscriptSegment] = []
        asr_started = time.perf_counter()
        try:
            for (
                channel,
                interval,
                separated,
                window,
                embedding,
                scores,
                assigned,
                margin,
            ) in gated:
                context.raise_if_cancelled()
                raw_hypotheses = self.asr_adapter.transcribe_batch(
                    separated,
                    (window,),
                    context,
                    requested_language=requested_language,
                    max_generated_tokens=token_budgets[
                        channel.candidate_id
                    ],
                )
                if len(raw_hypotheses) != 1:
                    raise WorkerError(
                        "OVERLAP_RECOVERY_ASR_INVALID",
                        "Qwen ASR must return one result per separated channel",
                    )
                hypothesis = (
                    raw_hypotheses[0]
                    if isinstance(raw_hypotheses[0], AsrHypothesis)
                    else AsrHypothesis.from_mapping(raw_hypotheses[0])
                )
                raw_timestamps = hypothesis.evidence.get("timestamps")
                if not isinstance(raw_timestamps, Sequence) or isinstance(
                    raw_timestamps,
                    (str, bytes, bytearray),
                ):
                    continue
                local_start = (
                    interval.detected_start_ms
                    - interval.context_start_ms
                )
                local_end = (
                    interval.detected_end_ms
                    - interval.context_start_ms
                )
                timestamps: list[dict[str, Any]] = []
                for item in raw_timestamps:
                    if (
                        not isinstance(item, Mapping)
                        or isinstance(item.get("startMs"), bool)
                        or not isinstance(item.get("startMs"), int)
                        or isinstance(item.get("endMs"), bool)
                        or not isinstance(item.get("endMs"), int)
                        or item["endMs"] < item["startMs"]
                    ):
                        continue
                    midpoint = (item["startMs"] + item["endMs"]) // 2
                    if not local_start <= midpoint < local_end:
                        continue
                    timestamps.append(
                        {
                            "text": item.get("text"),
                            "startMs": max(
                                interval.detected_start_ms,
                                interval.context_start_ms
                                + item["startMs"],
                            ),
                            "endMs": min(
                                interval.detected_end_ms,
                                interval.context_start_ms
                                + item["endMs"],
                            ),
                        }
                    )
                text = self._join_aligned_token_text(timestamps)
                if not text or not timestamps:
                    continue
                try:
                    candidate_set = project_asr_candidate_set(
                        hypothesis.evidence,
                        source_text=hypothesis.text,
                        target_text=text,
                        target_tokens=timestamps,
                        target_window_id=channel.candidate_id,
                        target_start_ms=interval.detected_start_ms,
                        target_end_ms=interval.detected_end_ms,
                    )
                except AsrEvidenceError:
                    continue
                speaker_id = f"speaker-{assigned + 1}"
                recovered.append(
                    TranscriptSegment(
                        segment_id=channel.candidate_id,
                        start_ms=interval.detected_start_ms,
                        end_ms=interval.detected_end_ms,
                        speaker_id=speaker_id,
                        raw_text=text,
                        normalized_text=text,
                        display_text=text,
                        confidence=hypothesis.confidence,
                        speaker_scores=tuple(
                            SpeakerScore(
                                f"speaker-{index + 1}",
                                score,
                            )
                            for index, score in enumerate(scores)
                        ),
                        speaker_margin=margin,
                        overlapping=True,
                        evidence={
                            "preparation": {
                                "provider": _adapter_identity(
                                    self.separation_adapter
                                ),
                                "audioPath": channel.audio_path,
                            },
                            "boundary": {
                                "provider": _adapter_identity(
                                    self.overlap_adapter
                                ),
                                "conflict": False,
                            },
                            "asr": {
                                "provider": _adapter_identity(
                                    self.asr_adapter
                                ),
                                **dict(hypothesis.evidence),
                                "timestamps": timestamps,
                                **candidate_set,
                            },
                            "voiceprint": {
                                "provider": _adapter_identity(
                                    self.embedding_adapter
                                ),
                                "confidence": embedding.confidence,
                                **dict(embedding.evidence),
                            },
                            "overlap": {
                                "provider": _adapter_identity(
                                    self.overlap_adapter
                                ),
                                "detectorStatus": "EVALUATED",
                                "overlapDetectorRun": True,
                                "reviewStatus": "REVIEW_REQUIRED",
                                "reasonCode": (
                                    "SEPARATED_SECONDARY_SPEECH_RECOVERED"
                                ),
                                "overlapIntervals": [
                                    {
                                        "startMs": (
                                            interval.detected_start_ms
                                        ),
                                        "endMs": interval.detected_end_ms,
                                        "localSpeakers": list(
                                            interval.local_speakers
                                        ),
                                    }
                                ],
                                **(
                                    {
                                        "fullTimelineInference": (
                                            full_timeline_inference
                                        )
                                    }
                                    if full_timeline_inference is not None
                                    else {}
                                ),
                            },
                            "overlapRecovery": {
                                "provider": _adapter_identity(
                                    self.separation_adapter
                                ),
                                "candidateId": channel.candidate_id,
                                "channelIndex": channel.channel_index,
                                "audioSha256": channel.audio_sha256,
                                "contextStartMs": interval.context_start_ms,
                                "contextEndMs": interval.context_end_ms,
                                "detectedStartMs": (
                                    interval.detected_start_ms
                                ),
                                "detectedEndMs": interval.detected_end_ms,
                                "contextPublishedAsSpeech": False,
                                "speakerMargin": margin,
                                "marginThreshold": (
                                    self.config
                                    .overlap_recovery_margin_threshold
                                ),
                                "tokenProjection": (
                                    "forced-alignment-midpoint-v1"
                                ),
                                "asrMaxNewTokens": token_budgets[
                                    channel.candidate_id
                                ],
                                "asrTokenBudgetPolicy": (
                                    "duration-8tps-plus16-v1"
                                ),
                                **dict(channel.evidence),
                            },
                        },
                        language=(
                            str(hypothesis.evidence["language"])
                            if isinstance(
                                hypothesis.evidence.get("language"),
                                str,
                            )
                            else None
                        ),
                    )
                )
        finally:
            self._release_after_success(self.asr_adapter)
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0
        metrics.record_stage("overlap-recovery-asr", asr_elapsed_ms)
        metrics.set_policy(
            overlapRecoverySeparatedChannelCount=len(channels),
            overlapRecoveryVoiceprintQualifiedCount=len(gated),
            overlapRecoveryPublishedCount=len(recovered),
            overlapRecoveryRejectedCount=len(channels) - len(recovered),
        )
        metrics.record_cascade_stage(
            stage="overlap-recovery-publish",
            provider=_adapter_identity(self.asr_adapter)["id"],
            trigger_reason="CAMPP_SECONDARY_MARGIN_AND_TOKEN_ALIGNMENT",
            candidate_ids=[item.segment_id for item in recovered],
            candidate_scope="separated-secondary-speech-only",
            source_count=len(channels),
            max_candidates=len(channels),
            candidate_start_ms=(
                min(item.start_ms for item in recovered)
                if recovered
                else None
            ),
            candidate_end_ms=(
                max(item.end_ms for item in recovered)
                if recovered
                else None
            ),
            invoked=bool(gated),
            cache_stage="overlap-recovery-asr",
            latency_ms=asr_elapsed_ms,
            resource=None,
            confidence=(
                min(item.speaker_margin for item in recovered)
                if recovered
                else None
            ),
            exit_reason=(
                "PUBLISHED_REVIEW_REQUIRED"
                if recovered
                else "NO_TOKEN_ALIGNED_SECONDARY_SPEECH"
            ),
        )
        return tuple(
            sorted(
                [*segments, *recovered],
                key=lambda item: (
                    item.start_ms,
                    item.end_ms,
                    item.segment_id,
                ),
            )
        )

    def transcribe(
        self,
        request: StartJobRequest,
        context: AdapterContext,
    ) -> TranscriptionResult:
        """Run an offline job without loading or downloading any model."""

        pipeline_started = time.perf_counter()
        metrics = PipelineMetricsCollector(
            job_id=request.job_id,
            started_at=pipeline_started,
        )
        metrics.set_policy(
            pyannoteMode=self.config.pyannote_mode,
            pyannoteTelemetryEnabled=False,
            localLlmModel=request.local_llm_model,
            localLlmMode=request.local_llm_mode,
            localLlmAutoApply=False,
            maxSecondaryFraction=self.config.max_secondary_fraction,
            speakerCountMode=request.speaker_policy.mode.value,
            overlapRecoveryMode=self.config.overlap_recovery_mode,
        )
        context.raise_if_cancelled()
        language_validator = getattr(
            self.asr_adapter,
            "validate_requested_language",
            None,
        )
        if language_validator is not None:
            if not callable(language_validator):
                raise WorkerError(
                    "ASR_LANGUAGE_VALIDATOR_INVALID",
                    "ASR language validation capability must be callable",
                )
            language_validator(request.language)
        context.raise_if_cancelled()
        hashing_started = time.perf_counter()
        source_fingerprint = _sha256_file(request.source_path, context)
        metrics.record_stage(
            "source-hashing",
            (time.perf_counter() - hashing_started) * 1000.0,
        )
        prepared, prepare_cache, prepare_elapsed = self._prepare(
            request, context, source_fingerprint
        )
        voice_activity = build_voice_activity(
            job_id=request.job_id,
            source_sha256=prepared.source_fingerprint,
            media_duration_ms=prepared.duration_ms,
            normalization_profile=prepared.normalization_profile,
            provider=_adapter_identity(self.preparation_adapter),
            windows=tuple(
                {
                    "id": window.window_id,
                    "startMs": window.start_ms,
                    "endMs": window.end_ms,
                }
                for window in prepared.windows
            ),
            minimum_window_ms=max(
                1,
                int(
                    getattr(
                        self.preparation_adapter,
                        "minimum_window_ms",
                        1,
                    )
                ),
            ),
            classification="transcribable-speech-detected",
            has_transcribable_speech=True,
        )
        metrics.set_duration_ms(prepared.duration_ms)
        for stage, cache_stats in prepare_cache.items():
            metrics.record_cache(stage, **cache_stats)
        metrics.record_stage("prepare-orchestration", prepare_elapsed)
        for stage in _PREPARATION_STAGES:
            cache_stage = "normalize" if stage in {"decode", "normalize"} else stage
            if prepare_cache[cache_stage]["hits"]:
                metrics.record_stage(stage, 0.0)
            else:
                metrics.record_stage(
                    stage,
                    float(prepared.stage_durations_ms.get(stage, 0.0)),
                )

        try:
            prepared = self._refinement_stage(prepared, context, metrics)
        except Exception:
            _release_adapter_resources(
                self.embedding_adapter,
                suppress_errors=True,
            )
            raise
        self._release_after_success(self.embedding_adapter)

        evidence_cache_identity = getattr(
            self.asr_adapter,
            "evidence_cache_identity",
            None,
        )
        try:
            asr = self._window_stage(
                stage="asr",
                prepared=prepared,
                windows=prepared.windows,
                adapter=self.asr_adapter,
                invoke=lambda windows: self.asr_adapter.transcribe_batch(
                    prepared,
                    windows,
                    context,
                    requested_language=request.language,
                ),
                converter=AsrHypothesis.from_mapping,
                accepted_type=AsrHypothesis,
                context=context,
                metrics=metrics,
                cache_identity_material={
                    "requestedLanguage": request.language,
                    "evidenceIdentity": (
                        evidence_cache_identity()
                        if callable(evidence_cache_identity)
                        else None
                    ),
                },
            )
        except Exception:
            _release_adapter_resources(
                self.asr_adapter,
                suppress_errors=True,
            )
            raise
        self._release_after_success(self.asr_adapter)
        rejected_asr = tuple(
            (window, hypothesis)
            for window, hypothesis in zip(prepared.windows, asr)
            if hypothesis.evidence.get("disposition")
            == _ASR_NON_LEXICAL_DISPOSITION
        )
        if rejected_asr:
            rejected_ids = [window.window_id for window, _ in rejected_asr]
            metrics.record_cascade_stage(
                stage="asr-non-lexical-rejection",
                provider=_adapter_identity(self.asr_adapter)["id"],
                trigger_reason="EMPTY_AFTER_INDIVIDUAL_RETRY",
                candidate_ids=rejected_ids,
                candidate_scope="vad-windows-without-lexical-asr",
                source_count=len(prepared.windows),
                max_candidates=len(prepared.windows),
                candidate_start_ms=min(
                    window.start_ms for window, _ in rejected_asr
                ),
                candidate_end_ms=max(
                    window.end_ms for window, _ in rejected_asr
                ),
                invoked=True,
                cache_stage="asr",
                latency_ms=0.0,
                resource=None,
                confidence=1.0,
                exit_reason="REJECTED_WITHOUT_FABRICATED_TEXT",
            )
        lexical_pairs = tuple(
            (window, hypothesis)
            for window, hypothesis in zip(prepared.windows, asr)
            if hypothesis.evidence.get("disposition")
            != _ASR_NON_LEXICAL_DISPOSITION
        )
        metrics.set_policy(
            asrSourceLexicalWindowCount=len(lexical_pairs),
            asrSourceRejectedNonLexicalWindowCount=len(rejected_asr),
        )
        if not lexical_pairs:
            no_lexical_voice_activity = (
                with_voice_activity_classification(
                    voice_activity,
                    classification="no-lexical-speech-detected",
                    has_transcribable_speech=False,
                )
            )
            raise WorkerError(
                "NO_TRANSCRIBABLE_SPEECH",
                "No lexical speech remained after auditable ASR retries",
                details={
                    "sourceWindowCount": len(prepared.windows),
                    "rejectedWindowCount": len(rejected_asr),
                    "voiceActivity": no_lexical_voice_activity,
                },
            )
        if rejected_asr:
            prepared = replace(
                prepared,
                windows=tuple(window for window, _ in lexical_pairs),
            )
            asr = [hypothesis for _, hypothesis in lexical_pairs]
        source_windows = prepared.windows
        source_asr = tuple(asr)
        prepared = self._partition_for_speaker_count_policy(
            prepared,
            request,
            metrics,
        )
        asr = self._project_asr_to_speaker_windows(
            source_windows=source_windows,
            source_hypotheses=source_asr,
            target_windows=prepared.windows,
            metrics=metrics,
        )
        projected_rejected_asr = tuple(
            (window, hypothesis)
            for window, hypothesis in zip(prepared.windows, asr)
            if hypothesis.evidence.get("disposition")
            == _ASR_NON_LEXICAL_DISPOSITION
        )
        if projected_rejected_asr:
            rejected_ids = [
                window.window_id for window, _ in projected_rejected_asr
            ]
            metrics.record_cascade_stage(
                stage="asr-projection-non-lexical-rejection",
                provider=_adapter_identity(self.asr_adapter)["id"],
                trigger_reason="NO_ALIGNED_LEXICAL_TOKENS",
                candidate_ids=rejected_ids,
                candidate_scope="speaker-evidence-windows-without-aligned-text",
                source_count=len(prepared.windows),
                max_candidates=len(prepared.windows),
                candidate_start_ms=min(
                    window.start_ms for window, _ in projected_rejected_asr
                ),
                candidate_end_ms=max(
                    window.end_ms for window, _ in projected_rejected_asr
                ),
                invoked=True,
                cache_stage="asr",
                latency_ms=0.0,
                resource=None,
                confidence=1.0,
                exit_reason="REJECTED_WITHOUT_FABRICATED_TEXT",
            )
        projected_lexical_pairs = tuple(
            (window, hypothesis)
            for window, hypothesis in zip(prepared.windows, asr)
            if hypothesis.evidence.get("disposition")
            != _ASR_NON_LEXICAL_DISPOSITION
        )
        metrics.set_policy(
            asrLexicalWindowCount=len(projected_lexical_pairs),
            asrRejectedNonLexicalWindowCount=(
                len(rejected_asr) + len(projected_rejected_asr)
            ),
        )
        if not projected_lexical_pairs:
            no_lexical_voice_activity = with_voice_activity_classification(
                voice_activity,
                classification="no-lexical-speech-detected",
                has_transcribable_speech=False,
            )
            raise WorkerError(
                "NO_TRANSCRIBABLE_SPEECH",
                "No aligned lexical speech remained for speaker evidence",
                details={
                    "sourceWindowCount": len(source_windows),
                    "speakerEvidenceWindowCount": len(prepared.windows),
                    "rejectedWindowCount": len(projected_rejected_asr),
                    "voiceActivity": no_lexical_voice_activity,
                },
            )
        if projected_rejected_asr:
            prepared = replace(
                prepared,
                windows=tuple(
                    window for window, _ in projected_lexical_pairs
                ),
            )
            asr = [
                hypothesis for _, hypothesis in projected_lexical_pairs
            ]
        campp_started = time.perf_counter()
        try:
            embeddings = self._window_stage(
                stage="campp-embedding",
                prepared=prepared,
                windows=prepared.windows,
                adapter=self.embedding_adapter,
                invoke=lambda windows: self.embedding_adapter.embed_batch(
                    prepared, windows, context
                ),
                converter=EmbeddingRecord.from_mapping,
                accepted_type=EmbeddingRecord,
                context=context,
                metrics=metrics,
            )
        except Exception:
            _release_adapter_resources(
                self.embedding_adapter,
                suppress_errors=True,
            )
            raise
        self._release_after_success(self.embedding_adapter)
        campp_elapsed = (time.perf_counter() - campp_started) * 1000.0
        campp_candidate_ids = [
            window.window_id for window in prepared.windows
        ]
        campp_start, campp_end = self._candidate_time_bounds(
            campp_candidate_ids,
            {
                window.window_id: window
                for window in prepared.windows
            },
        )
        metrics.record_cascade_stage(
            stage="campp-embedding",
            provider=_adapter_identity(self.embedding_adapter)["id"],
            trigger_reason="ALL_SPEECH_WINDOWS",
            candidate_ids=campp_candidate_ids,
            candidate_scope="all-speech-windows",
            source_count=len(prepared.windows),
            max_candidates=len(prepared.windows),
            candidate_start_ms=campp_start,
            candidate_end_ms=campp_end,
            invoked=True,
            cache_stage="campp-embedding",
            latency_ms=campp_elapsed,
            resource=None,
            confidence=(
                sum(item.confidence for item in embeddings)
                / len(embeddings)
                if embeddings
                else None
            ),
            exit_reason="COMPLETED",
        )
        try:
            speaker_count_constraints: dict[str, int] | None = None
            if request.speaker_policy.mode is SpeakerCountMode.MANUAL:
                assert request.speaker_policy.manual_count is not None
                speaker_count_constraints = {
                    "numSpeakers": request.speaker_policy.manual_count,
                }
            elif request.speaker_policy.mode is SpeakerCountMode.HYBRID:
                assert request.speaker_policy.minimum is not None
                assert request.speaker_policy.maximum is not None
                speaker_count_constraints = {
                    "minSpeakers": request.speaker_policy.minimum,
                    "maxSpeakers": request.speaker_policy.maximum,
                }
            overlap = self._window_stage(
                stage="overlap",
                prepared=prepared,
                windows=prepared.windows,
                adapter=self.overlap_adapter,
                invoke=lambda windows: self.overlap_adapter.detect_batch(
                    prepared,
                    windows,
                    context,
                    speaker_count_constraints=speaker_count_constraints,
                ),
                converter=OverlapDecision.from_mapping,
                accepted_type=OverlapDecision,
                context=context,
                metrics=metrics,
                cache_identity_material={
                    "speakerCountConstraints": speaker_count_constraints,
                },
            )
        except Exception:
            _release_adapter_resources(
                self.overlap_adapter,
                suppress_errors=True,
            )
            raise
        self._release_after_success(self.overlap_adapter)
        clusters = self._clustering_stage(
            prepared, embeddings, overlap, request, metrics
        )
        metrics.set_policy(resolvedSpeakerCount=clusters.count)
        contextual_projection = self._project_contextual_speaker_turns(
            prepared=prepared,
            asr=asr,
            embeddings=embeddings,
            overlap=overlap,
            clusters=clusters,
            metrics=metrics,
        )
        if contextual_projection is not None:
            (
                prepared,
                asr,
                embeddings,
                overlap,
                clusters,
            ) = contextual_projection
        segments = self._initial_segments(
            prepared,
            asr,
            embeddings,
            overlap,
            clusters,
            requested_language=request.language,
        )
        baseline_speaker_ids = self._assert_speaker_cardinality(
            segments=segments,
            clusters=clusters,
            request=request,
            stage="post-clustering",
        )
        segments = self._decode_global_speaker_sequence(segments)
        segments = self._apply_pyannote_canonical_mapping(segments)
        segments = self._recover_overlap_speech(
            prepared=prepared,
            overlap=overlap,
            embeddings=embeddings,
            clusters=clusters,
            segments=segments,
            requested_language=request.language,
            context=context,
            metrics=metrics,
        )
        self._assert_speaker_cardinality(
            segments=segments,
            clusters=clusters,
            request=request,
            stage="post-global-speaker-sequence-decode",
            expected_ids=baseline_speaker_ids,
        )
        candidates = self._select_candidates(
            segments, prepared.windows, clusters
        )
        segments = self._review_stage(
            segments, candidates, metrics, context
        )
        self._assert_speaker_cardinality(
            segments=segments,
            clusters=clusters,
            request=request,
            stage="post-review",
            expected_ids=baseline_speaker_ids,
        )
        segments = self._coalesce_speaker_evidence_segments(segments, metrics)
        self._assert_speaker_cardinality(
            segments=segments,
            clusters=clusters,
            request=request,
            stage="post-speaker-evidence-coalescing",
            expected_ids=baseline_speaker_ids,
        )
        speaker_timeline = self._build_pyannote_speaker_timeline(
            segments,
            duration_ms=prepared.duration_ms,
        )
        if prepared.reference_turns:
            metrics.set_reference_quality(
                evaluate_reference_quality(
                    segments,
                    prepared.reference_turns,
                    speaker_timeline=speaker_timeline,
                )
            )

        estimate = SpeakerCountEstimate(
            estimated_count=clusters.count,
            confidence=clusters.confidence,
            candidate_min=clusters.candidate_min,
            candidate_max=clusters.candidate_max,
            method=_SPEAKER_COUNT_ESTIMATE_METHOD,
        )
        models: list[Mapping[str, Any]] = [
            {
                "role": "preparation",
                "name": _adapter_identity(self.preparation_adapter)["id"],
                "version": _adapter_identity(self.preparation_adapter)["version"],
                "offline": True,
            },
            {
                "role": "asr",
                "name": _adapter_identity(self.asr_adapter)["id"],
                "version": _adapter_identity(self.asr_adapter)["version"],
                "offline": True,
            },
            {
                "role": "voiceprint",
                "name": _adapter_identity(self.embedding_adapter)["id"],
                "version": _adapter_identity(self.embedding_adapter)["version"],
                "offline": True,
            },
            {
                "role": "overlap",
                "name": _adapter_identity(self.overlap_adapter)["id"],
                "version": _adapter_identity(self.overlap_adapter)["version"],
                "offline": True,
            },
        ]
        if self.secondary_adapter is not None:
            models.append(
                {
                    "role": "secondary-voiceprint",
                    "name": _adapter_identity(self.secondary_adapter)["id"],
                    "version": _adapter_identity(self.secondary_adapter)["version"],
                    "scope": "difficult-segments-only",
                    "fullCorpusRun": False,
                    "offline": True,
                }
            )
        if self.separation_adapter is not None:
            models.append(
                {
                    "role": "overlap-separation",
                    "name": _adapter_identity(
                        self.separation_adapter
                    )["id"],
                    "version": _adapter_identity(
                        self.separation_adapter
                    )["version"],
                    "mode": self.config.overlap_recovery_mode,
                    "scope": "exact-two-speaker-overlap-only",
                    "offline": True,
                }
            )
        models.append(
            {
                "role": "semantic",
                "name": request.local_llm_model,
                "mode": request.local_llm_mode,
                "status": "reject_for_production",
                "autoApply": False,
                "calls": 0,
                "offline": True,
            }
        )
        models.append(
            {
                "role": "diarization-audit",
                "name": "pyannote-community-1",
                "mode": self.config.pyannote_mode,
                "telemetryEnabled": False,
                "offline": True,
            }
        )
        resolved_language = reconcile_detected_languages(
            (
                {
                    "languageCandidates": hypothesis.evidence.get(
                        "languageCandidates",
                        hypothesis.evidence.get(
                            "language",
                            hypothesis.evidence.get("rawLanguage"),
                        ),
                    ),
                    "speechDurationMs": window.end_ms - window.start_ms,
                }
                for window, hypothesis in zip(prepared.windows, asr)
            ),
            requested_language=request.language,
        )
        return TranscriptionResult(
            segments=segments,
            duration_ms=prepared.duration_ms,
            language=resolved_language,
            speaker_count_estimate=estimate,
            models=tuple(models),
            pipeline_metrics=metrics.as_dict(),
            speaker_timeline=speaker_timeline,
            voice_activity=with_voice_activity_classification(
                voice_activity,
                classification="transcribable-speech-detected",
                has_transcribable_speech=True,
            ),
        )


__all__ = [
    "AsrHypothesis",
    "AudioPreparationAdapter",
    "BatchAsrAdapter",
    "BatchEmbeddingAdapter",
    "CacheRead",
    "CamPlusEmbeddingAdapter",
    "ERes2NetV2ReviewAdapter",
    "EmbeddingRecord",
    "EscalationAdapter",
    "InMemoryStageCache",
    "JsonStageCache",
    "NoOverlapAdapter",
    "OverlapDecision",
    "OverlapRecoveryInterval",
    "OverlapDetectionAdapter",
    "PreparedAudio",
    "PyannoteReviewAdapter",
    "Qwen3AsrAdapter",
    "ReviewCandidate",
    "ReviewProposal",
    "SeparatedSpeechChannel",
    "SpeechSeparationAdapter",
    "SpeakerPipeline",
    "SpeakerPipelineConfig",
    "SpeechWindow",
    "StageCache",
    "UnavailableOverlapAdapter",
]
