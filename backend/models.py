"""Typed command, speaker-policy, transcript, and renderer models."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .business_processing import BusinessProcessingConfig
from .errors import WorkerError, invalid_request
from .language import normalize_language_tag
from .output_recipe import OutputRecipe
from .speaker_timeline import validate_speaker_timeline
from .voice_activity import validate_voice_activity


PROTOCOL_VERSION = "1.0.0"
TRANSCRIPT_SCHEMA_VERSION = "2.0.0"
CHECKPOINT_SCHEMA_VERSION = "2.0.0"
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SEGMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")
_CANONICAL_SPEAKER = re.compile(r"^speaker-([1-9][0-9]*)$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_REVISION_TYPES = frozenset({"text", "speaker", "boundary", "split", "merge"})
_REVISION_SOURCES = frozenset({"acoustic", "deterministic", "llm", "manual"})


def _positive_int(
    value: Any,
    field_name: str,
    *,
    error_code: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerError(error_code, f"{field_name} must be a positive integer")
    if value < 1:
        raise WorkerError(error_code, f"{field_name} must be a positive integer")
    return value


def _request_positive_int(value: Any, field_name: str) -> int:
    return _positive_int(value, field_name, error_code="INVALID_REQUEST")


def _adapter_positive_int(value: Any, field_name: str) -> int:
    return _positive_int(value, field_name, error_code="ADAPTER_RESULT_INVALID")


def validate_job_id(value: Any) -> str:
    if not isinstance(value, str):
        raise invalid_request("jobId is missing or invalid")
    job_id = value.strip()
    if not _JOB_ID.fullmatch(job_id):
        raise invalid_request("jobId is missing or invalid")
    return job_id


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise WorkerError("ADAPTER_RESULT_INVALID", f"{field_name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerError(
            "ADAPTER_RESULT_INVALID", f"{field_name} must be numeric"
        ) from exc
    if not math.isfinite(number):
        raise WorkerError(
            "ADAPTER_RESULT_INVALID", f"{field_name} must be finite"
        )
    return number


def _probability(value: Any, field_name: str) -> float:
    number = _finite_number(value, field_name)
    if number < 0.0 or number > 1.0:
        raise WorkerError(
            "ADAPTER_RESULT_INVALID",
            f"{field_name} must be between 0 and 1",
        )
    return number


def canonical_speaker_ids(count: int) -> tuple[str, ...]:
    normalized = _request_positive_int(count, "speakerCount")
    return tuple(f"speaker-{index}" for index in range(1, normalized + 1))


class SpeakerCountMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    HYBRID = "hybrid"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SpeakerCountPolicy:
    mode: SpeakerCountMode
    manual_count: int | None = None
    roles: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    prior: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SpeakerCountPolicy":
        raw_mode = payload.get("speakerCountMode", "auto")
        try:
            mode = SpeakerCountMode(str(raw_mode))
        except ValueError as exc:
            raise invalid_request(
                "speakerCountMode must be auto, manual, or hybrid"
            ) from exc

        roles_provided = "speakerRoles" in payload
        raw_roles = payload.get("speakerRoles")
        roles: tuple[str, ...] = ()
        if roles_provided:
            if (
                not isinstance(raw_roles, list)
                or any(not isinstance(item, str) or not item.strip() for item in raw_roles)
            ):
                raise invalid_request(
                    "speakerRoles must be an array of non-empty strings"
                )
            roles = tuple(item.strip() for item in raw_roles)
            if len(set(roles)) != len(roles):
                raise invalid_request("speakerRoles must not contain duplicates")

        if mode is SpeakerCountMode.MANUAL:
            count = _request_positive_int(payload.get("speakerCount"), "speakerCount")
            if roles_provided and len(roles) != count:
                raise invalid_request(
                    "speakerRoles length must equal speakerCount",
                    speakerCount=count,
                    roleCount=len(roles),
                )
            if payload.get("speakerCountBounds") is not None:
                raise invalid_request(
                    "manual mode does not accept speakerCountBounds"
                )
            if payload.get("speakerCountPrior") is not None:
                raise invalid_request(
                    "manual mode does not accept speakerCountPrior"
                )
            return cls(mode=mode, manual_count=count, roles=roles)

        if payload.get("speakerCount") is not None:
            raise invalid_request(
                f"{mode.value} mode does not accept speakerCount"
            )
        if roles_provided:
            raise invalid_request(
                "speakerRoles is only valid in manual mode"
            )

        if mode is SpeakerCountMode.AUTO:
            if payload.get("speakerCountBounds") is not None:
                raise invalid_request(
                    "auto mode does not accept speakerCountBounds; use hybrid"
                )
            if payload.get("speakerCountPrior") is not None:
                raise invalid_request(
                    "auto mode does not accept speakerCountPrior; use hybrid"
                )
            return cls(mode=mode)

        bounds = payload.get("speakerCountBounds")
        if not isinstance(bounds, Mapping):
            raise invalid_request(
                "hybrid mode requires speakerCountBounds with min and max"
            )
        if set(bounds) != {"min", "max"}:
            raise invalid_request(
                "speakerCountBounds must contain exactly min and max"
            )
        minimum = _request_positive_int(
            bounds.get("min"), "speakerCountBounds.min"
        )
        maximum = _request_positive_int(
            bounds.get("max"), "speakerCountBounds.max"
        )
        if minimum > maximum:
            raise invalid_request(
                "speakerCountBounds.min cannot exceed speakerCountBounds.max"
            )
        raw_prior = payload.get("speakerCountPrior")
        prior = (
            _request_positive_int(raw_prior, "speakerCountPrior")
            if raw_prior is not None
            else None
        )
        if prior is not None and not minimum <= prior <= maximum:
            raise invalid_request(
                "speakerCountPrior must be within speakerCountBounds"
            )
        return cls(
            mode=mode,
            minimum=minimum,
            maximum=maximum,
            prior=prior,
        )

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"mode": self.mode.value}
        if self.manual_count is not None:
            value["manualCount"] = self.manual_count
        if self.roles:
            value["roles"] = list(self.roles)
        if self.minimum is not None and self.maximum is not None:
            value["bounds"] = {"min": self.minimum, "max": self.maximum}
        if self.prior is not None:
            value["prior"] = self.prior
        return value


@dataclass(frozen=True)
class SpeakerCountEstimate:
    estimated_count: int
    confidence: float
    candidate_min: int
    candidate_max: int
    method: str

    @classmethod
    def from_mapping(cls, value: Any) -> "SpeakerCountEstimate":
        if not isinstance(value, Mapping):
            raise WorkerError(
                "SPEAKER_COUNT_ESTIMATE_MISSING",
                "auto and hybrid modes require a speakerCountEstimate",
            )
        estimated = _adapter_positive_int(
            value.get("estimatedCount"), "speakerCountEstimate.estimatedCount"
        )
        confidence = _probability(
            value.get("confidence"), "speakerCountEstimate.confidence"
        )
        candidate = value.get("candidateRange")
        if not isinstance(candidate, Mapping):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "speakerCountEstimate.candidateRange must contain min and max",
            )
        candidate_min = _adapter_positive_int(
            candidate.get("min"), "speakerCountEstimate.candidateRange.min"
        )
        candidate_max = _adapter_positive_int(
            candidate.get("max"), "speakerCountEstimate.candidateRange.max"
        )
        if candidate_min > candidate_max:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "speakerCountEstimate candidate min cannot exceed max",
            )
        if not candidate_min <= estimated <= candidate_max:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "speakerCountEstimate candidate range must contain estimatedCount",
            )
        method = str(value.get("method") or "").strip()
        if not method:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "speakerCountEstimate.method must not be empty",
            )
        return cls(estimated, confidence, candidate_min, candidate_max, method)

    @classmethod
    def manual(cls, count: int) -> "SpeakerCountEstimate":
        return cls(count, 1.0, count, count, "manual")

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimatedCount": self.estimated_count,
            "confidence": self.confidence,
            "candidateRange": {
                "min": self.candidate_min,
                "max": self.candidate_max,
            },
            "method": self.method,
        }


@dataclass(frozen=True)
class SpeakerScore:
    speaker_id: str
    score: float

    @classmethod
    def from_mapping(cls, value: Any, field_name: str) -> "SpeakerScore":
        if not isinstance(value, Mapping):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name} must be an object"
            )
        speaker_id = str(value.get("speakerId") or "").strip()
        if not _CANONICAL_SPEAKER.fullmatch(speaker_id):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.speakerId must be speaker-N",
            )
        score = _finite_number(value.get("score"), f"{field_name}.score")
        if score < -2.0 or score > 2.0:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.score must be between -2 and 2",
            )
        return cls(speaker_id, score)

    def as_dict(self) -> dict[str, Any]:
        return {"speakerId": self.speaker_id, "score": self.score}


@dataclass(frozen=True)
class Revision:
    revision_id: str
    revision_type: str
    source: str
    before: Any
    after: Any
    reason_code: str
    confidence: float
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any, field_name: str) -> "Revision":
        if not isinstance(value, Mapping):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name} must be an object"
            )
        revision_id = str(value.get("id") or "").strip()
        revision_type = str(value.get("type") or "").strip()
        source = str(value.get("source") or "").strip()
        reason_code = str(value.get("reasonCode") or "").strip()
        refs = value.get("evidenceRefs", [])
        if not revision_id:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name}.id must not be empty"
            )
        if revision_type not in _REVISION_TYPES:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name}.type is unsupported"
            )
        if source not in _REVISION_SOURCES:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name}.source is unsupported"
            )
        if source == "llm":
            raise WorkerError(
                "LLM_AUTO_APPLY_FORBIDDEN",
                f"{field_name} cannot persist an LLM-authored revision",
            )
        if not reason_code:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.reasonCode must not be empty",
            )
        if not isinstance(refs, list) or not refs or any(
            not isinstance(item, str) or not item.strip() for item in refs
        ):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.evidenceRefs must be a non-empty array of strings",
            )
        return cls(
            revision_id=revision_id,
            revision_type=revision_type,
            source=source,
            before=value.get("before"),
            after=value.get("after"),
            reason_code=reason_code,
            confidence=_probability(
                value.get("confidence"), f"{field_name}.confidence"
            ),
            evidence_refs=tuple(item.strip() for item in refs),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.revision_id,
            "type": self.revision_type,
            "source": self.source,
            "before": self.before,
            "after": self.after,
            "reasonCode": self.reason_code,
            "confidence": self.confidence,
            "evidenceRefs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    start_ms: int
    end_ms: int
    speaker_id: str
    raw_text: str
    normalized_text: str
    display_text: str
    confidence: float
    speaker_scores: tuple[SpeakerScore, ...]
    speaker_margin: float
    overlapping: bool = False
    human_locked: bool = False
    revisions: tuple[Revision, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        candidate = self.language
        if candidate is not None:
            try:
                normalized = normalize_language_tag(candidate, allow_auto=False)
            except ValueError as exc:
                raise WorkerError(
                    "ADAPTER_RESULT_INVALID",
                    "segment.language must be a valid persisted BCP-47 language tag",
                ) from exc
            object.__setattr__(self, "language", normalized)
            return

        asr_evidence = self.evidence.get("asr")
        if not isinstance(asr_evidence, Mapping):
            return
        evidence_language = asr_evidence.get("language")
        try:
            normalized = normalize_language_tag(
                evidence_language,
                allow_auto=False,
            )
        except ValueError:
            return
        object.__setattr__(self, "language", normalized)

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "TranscriptSegment":
        field_name = f"segments[{index}]"
        if not isinstance(value, Mapping):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name} must be an object"
            )
        segment_id = str(value.get("id") or "").strip()
        if not _SEGMENT_ID.fullmatch(segment_id):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name}.id is invalid"
            )
        start_ms = value.get("startMs")
        end_ms = value.get("endMs")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or start_ms < 0
            or end_ms <= start_ms
        ):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name} has invalid startMs/endMs",
            )
        speaker_id = str(value.get("speakerId") or "").strip()
        if not _CANONICAL_SPEAKER.fullmatch(speaker_id):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.speakerId must be speaker-N",
            )
        texts: list[str] = []
        for key in ("rawText", "normalizedText", "displayText"):
            text = value.get(key)
            if not isinstance(text, str) or not text.strip():
                raise WorkerError(
                    "ADAPTER_RESULT_INVALID",
                    f"{field_name}.{key} must be non-empty source text",
                )
            texts.append(text.strip())
        scores_raw = value.get("speakerScores")
        if not isinstance(scores_raw, list) or not scores_raw:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.speakerScores must be a non-empty array",
            )
        scores = tuple(
            SpeakerScore.from_mapping(item, f"{field_name}.speakerScores[{score_index}]")
            for score_index, item in enumerate(scores_raw)
        )
        revisions_raw = value.get("revisions", [])
        if not isinstance(revisions_raw, list):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.revisions must be an array",
            )
        revisions = tuple(
            Revision.from_mapping(item, f"{field_name}.revisions[{revision_index}]")
            for revision_index, item in enumerate(revisions_raw)
        )
        evidence = value.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", f"{field_name}.evidence must be an object"
            )
        overlapping = value.get("overlapping", False)
        human_locked = value.get("humanLocked", False)
        if not isinstance(overlapping, bool):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.overlapping must be a boolean",
            )
        if not isinstance(human_locked, bool):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                f"{field_name}.humanLocked must be a boolean",
            )
        turn_id_raw = value.get("turnId")
        turn_id = None
        if turn_id_raw is not None:
            if not isinstance(turn_id_raw, str) or not turn_id_raw.strip():
                raise WorkerError(
                    "ADAPTER_RESULT_INVALID",
                    f"{field_name}.turnId must be non-empty text",
                )
            turn_id = turn_id_raw.strip()
        language = None
        if "language" in value and value.get("language") is not None:
            try:
                language = normalize_language_tag(
                    value.get("language"),
                    allow_auto=False,
                )
            except ValueError as exc:
                raise WorkerError(
                    "ADAPTER_RESULT_INVALID",
                    f"{field_name}.language must be a valid persisted BCP-47 language tag",
                ) from exc
        return cls(
            segment_id=segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            speaker_id=speaker_id,
            raw_text=texts[0],
            normalized_text=texts[1],
            display_text=texts[2],
            confidence=_probability(value.get("confidence"), f"{field_name}.confidence"),
            speaker_scores=scores,
            speaker_margin=_finite_number(
                value.get("speakerMargin"), f"{field_name}.speakerMargin"
            ),
            overlapping=overlapping,
            human_locked=human_locked,
            revisions=revisions,
            evidence=dict(evidence),
            turn_id=turn_id,
            language=language,
        )

    def as_dict(self) -> dict[str, Any]:
        value = {
            "id": self.segment_id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "speakerId": self.speaker_id,
            "rawText": self.raw_text,
            "normalizedText": self.normalized_text,
            "displayText": self.display_text,
            "confidence": self.confidence,
            "speakerScores": [item.as_dict() for item in self.speaker_scores],
            "speakerMargin": self.speaker_margin,
            "overlapping": self.overlapping,
            "humanLocked": self.human_locked,
            "revisions": [item.as_dict() for item in self.revisions],
            "evidence": dict(self.evidence),
        }
        if self.turn_id is not None:
            value["turnId"] = self.turn_id
        if self.language is not None:
            value["language"] = self.language
        return value


@dataclass(frozen=True)
class TranscriptionResult:
    segments: tuple[TranscriptSegment, ...]
    duration_ms: int
    language: str = "und"
    speaker_count_estimate: SpeakerCountEstimate | None = None
    models: tuple[Mapping[str, Any], ...] = ()
    pipeline_metrics: Mapping[str, Any] | None = None
    voice_activity: Mapping[str, Any] | None = None
    speaker_timeline: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            normalized = normalize_language_tag(
                self.language,
                allow_auto=False,
            )
        except ValueError as exc:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "transcription result language must be a valid persisted BCP-47 language tag",
            ) from exc
        object.__setattr__(self, "language", normalized)
        if self.voice_activity is not None:
            object.__setattr__(
                self,
                "voice_activity",
                validate_voice_activity(self.voice_activity),
            )
        if self.speaker_timeline is not None:
            canonical_ids = tuple(
                sorted(
                    {
                        score.speaker_id
                        for segment in self.segments
                        for score in segment.speaker_scores
                    },
                    key=lambda item: int(item.removeprefix("speaker-")),
                )
            )
            object.__setattr__(
                self,
                "speaker_timeline",
                validate_speaker_timeline(
                    self.speaker_timeline,
                    duration_ms=self.duration_ms,
                    canonical_speaker_ids=canonical_ids,
                ),
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "TranscriptionResult":
        if not isinstance(value, Mapping):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", "transcription result must be an object"
            )
        raw_segments = value.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "transcription result must contain at least one segment",
            )
        duration_ms = value.get("durationMs")
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or duration_ms < 1
        ):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", "durationMs must be a positive integer"
            )
        estimate_raw = value.get("speakerCountEstimate")
        estimate = (
            SpeakerCountEstimate.from_mapping(estimate_raw)
            if estimate_raw is not None
            else None
        )
        models = value.get("models", [])
        if not isinstance(models, list) or any(
            not isinstance(item, Mapping) for item in models
        ):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID", "models must be an array of objects"
            )
        pipeline_metrics = value.get("pipelineMetrics")
        if pipeline_metrics is not None and not isinstance(
            pipeline_metrics, Mapping
        ):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "pipelineMetrics must be an object when provided",
            )
        voice_activity = value.get("voiceActivity")
        if voice_activity is not None and not isinstance(
            voice_activity,
            Mapping,
        ):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "voiceActivity must be an object when provided",
            )
        speaker_timeline = value.get("speakerTimeline")
        if speaker_timeline is not None and not isinstance(
            speaker_timeline,
            Mapping,
        ):
            raise WorkerError(
                "ADAPTER_RESULT_INVALID",
                "speakerTimeline must be an object when provided",
            )
        segments = tuple(
            TranscriptSegment.from_mapping(item, index)
            for index, item in enumerate(raw_segments)
        )
        if "language" in value:
            try:
                language = normalize_language_tag(
                    value.get("language"),
                    allow_auto=False,
                )
            except ValueError as exc:
                raise WorkerError(
                    "ADAPTER_RESULT_INVALID",
                    "language must be a valid persisted BCP-47 language tag",
                ) from exc
        else:
            detected_languages = {
                segment.language
                for segment in segments
                if segment.language not in {None, "und"}
            }
            if not detected_languages:
                language = "und"
            elif len(detected_languages) == 1:
                language = next(iter(detected_languages))
            else:
                language = "mul"
        return cls(
            segments=segments,
            duration_ms=duration_ms,
            language=language,
            speaker_count_estimate=estimate,
            models=tuple(dict(item) for item in models),
            pipeline_metrics=(
                dict(pipeline_metrics) if pipeline_metrics is not None else None
            ),
            voice_activity=(
                validate_voice_activity(voice_activity)
                if voice_activity is not None
                else None
            ),
            speaker_timeline=(
                dict(speaker_timeline)
                if speaker_timeline is not None
                else None
            ),
        )


@dataclass(frozen=True)
class StartJobRequest:
    job_id: str
    source_path: Path
    output_directory: Path
    speaker_policy: SpeakerCountPolicy
    render_pdf: bool = False
    title: str | None = None
    language: str = "auto"
    local_llm_mode: str = "disabled"
    local_llm_model: str = "qwen3.5:9b"
    local_llm_endpoint: str = "http://127.0.0.1:11434"
    business_config: BusinessProcessingConfig = field(
        default_factory=BusinessProcessingConfig
    )
    output_recipe: OutputRecipe | None = None


@dataclass(frozen=True)
class RenderArtifact:
    artifact_type: str
    path: Path

    def validate(self) -> None:
        if (
            not isinstance(self.artifact_type, str)
            or not self.artifact_type.strip()
            or len(self.artifact_type) > 160
        ):
            raise WorkerError(
                "RENDER_RESULT_INVALID",
                "renderer artifactType must be a non-empty bounded string",
            )
        if not isinstance(self.path, Path):
            raise WorkerError(
                "RENDER_RESULT_INVALID",
                "renderer artifact paths must be pathlib.Path values",
            )


@dataclass(frozen=True)
class RenderResult:
    template_hash: str
    renderer_version: str
    quality_status: str
    quality_report_path: Path
    render_manifest_path: Path
    artifact_paths: tuple[Path, ...]
    artifacts: tuple[RenderArtifact, ...] = ()

    def validate(self) -> None:
        if (
            not isinstance(self.template_hash, str)
            or not _SHA256.fullmatch(self.template_hash)
        ):
            raise WorkerError(
                "RENDER_RESULT_INVALID", "renderer templateHash must be SHA-256"
            )
        if (
            not isinstance(self.renderer_version, str)
            or not self.renderer_version.strip()
        ):
            raise WorkerError(
                "RENDER_RESULT_INVALID", "rendererVersion must not be empty"
            )
        if self.quality_status != "passed":
            raise WorkerError(
                "PDF_QUALITY_FAILED",
                "renderer did not return a passed quality status",
                details={"qualityStatus": self.quality_status},
            )
        if (
            not isinstance(self.artifact_paths, tuple)
            or not self.artifact_paths
            or any(not isinstance(path, Path) for path in self.artifact_paths)
            or not any(path.suffix.casefold() == ".pdf" for path in self.artifact_paths)
        ):
            raise WorkerError(
                "RENDER_RESULT_INVALID",
                "renderer must return at least one PDF artifact",
            )
        if not isinstance(self.quality_report_path, Path) or not isinstance(
            self.render_manifest_path, Path
        ):
            raise WorkerError(
                "RENDER_RESULT_INVALID",
                "renderer report and manifest paths must be pathlib.Path values",
            )
        if not isinstance(self.artifacts, tuple):
            raise WorkerError(
                "RENDER_RESULT_INVALID",
                "renderer artifacts must be a tuple",
            )
        typed_paths: set[Path] = set()
        typed_names: set[str] = set()
        for artifact in self.artifacts:
            if not isinstance(artifact, RenderArtifact):
                raise WorkerError(
                    "RENDER_RESULT_INVALID",
                    "renderer artifacts must contain RenderArtifact values",
                )
            artifact.validate()
            if artifact.path in typed_paths:
                raise WorkerError(
                    "RENDER_RESULT_INVALID",
                    "renderer artifact paths must be unique",
                    details={"path": str(artifact.path)},
                )
            if artifact.artifact_type in typed_names:
                raise WorkerError(
                    "RENDER_RESULT_INVALID",
                    "renderer artifact types must be unique",
                    details={"artifactType": artifact.artifact_type},
                )
            typed_paths.add(artifact.path)
            typed_names.add(artifact.artifact_type)
