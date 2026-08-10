"""Build the immutable report contract consumed by the Java PDF renderer.

This module is intentionally standard-library only.  The media worker can use
it in a locked-down, offline environment without importing the ASR stack or a
JSON Schema runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from contracts.pyannote_evidence import is_verified_pyannote_speaker_revision
from contracts.validate_contracts import ContractError, validate_report_document


SCHEMA_VERSION = "1.0.0"
_CANONICAL_SPEAKER_PATTERN = re.compile(r"^speaker-([1-9][0-9]*)$")
_UNKNOWN_SPEAKER_LABELS = frozenset(
    {
        "",
        "unknown",
        "unk",
        "none",
        "null",
        "n/a",
        "speaker_unknown",
        "speaker-unknown",
        "未知",
        "未知说话人",
    }
)
_DOCUMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
_SEGMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,120}$")
_REASON_CODE_PATTERN = re.compile(r"^[A-Z0-9_:-]+$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_REVISION_SOURCES = frozenset({"acoustic", "deterministic", "llm", "manual"})
_REVISION_TYPES = frozenset({"text", "speaker", "boundary", "split", "merge"})
_REVIEW_STATUSES = frozenset(
    {"accepted", "review-required", "manually-reviewed", "locked"}
)
_SPEAKER_COUNT_MODES = frozenset({"auto", "manual", "hybrid"})
_SPEAKER_DISPLAY_LABELS = {
    "de": "Sprecher",
    "en": "Speaker",
    "es": "Hablante",
    "fr": "Intervenant",
    "ja": "話者",
    "ko": "화자",
    "pt": "Falante",
    "zh": "角色",
}


class ReportAssemblyError(ValueError):
    """Raised when legacy data cannot be represented without inventing facts."""


def _normalize_report_language(value: Any, field: str) -> str:
    # Keep this import lazy: importing backend at module load time creates a
    # cycle through backend.composition -> reporting.
    from backend.language import normalize_language_tag

    try:
        return normalize_language_tag(value, allow_auto=False)
    except ValueError as exc:
        raise ReportAssemblyError(
            f"{field} must be a valid persisted BCP-47 language tag"
        ) from exc


def canonical_speaker_ids(count: int) -> tuple[str, ...]:
    """Return the contiguous canonical speaker namespace for a resolved count."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise ReportAssemblyError("speaker count must be an integer")
    normalized = count
    if normalized < 1:
        raise ReportAssemblyError("speaker count must be at least 1")
    return tuple(f"speaker-{index}" for index in range(1, normalized + 1))


def _canonical_speaker_index(value: str) -> Optional[int]:
    match = _CANONICAL_SPEAKER_PATTERN.fullmatch(value)
    return int(match.group(1)) if match else None


def _read_value(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if not isinstance(value, Mapping) and hasattr(value, name):
            return getattr(value, name)
    return default


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _default_speaker_display_name(index: int, locale: str) -> str:
    language = locale.casefold().split("-", 1)[0]
    label = _SPEAKER_DISPLAY_LABELS.get(language, _SPEAKER_DISPLAY_LABELS["en"])
    return f"{label} {index}"


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ReportAssemblyError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportAssemblyError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ReportAssemblyError(f"{field} must be finite")
    return number


def _probability(value: Any, *, field: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    number = _finite_number(value, field=field)
    if number < 0.0 or number > 1.0:
        raise ReportAssemblyError(f"{field} must be between 0 and 1")
    return number


def _speaker_score(value: Any, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if number < -2.0 or number > 2.0:
        raise ReportAssemblyError(f"{field} must be between -2 and 2")
    return number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_iso(value: Optional[str]) -> str:
    if value:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReportAssemblyError("generated_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ReportAssemblyError("generated_at must include a timezone")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ReportDocumentAssembler:
    """Normalize legacy transcription output into a fail-closed report contract."""

    def __init__(
        self,
        *,
        pipeline_version: str = "2.0.0",
        low_confidence_threshold: float = 0.75,
        semantic_margin_threshold: float = 0.18,
        allow_neutral_compatibility_evidence: bool = False,
    ) -> None:
        self.pipeline_version = _clean_text(pipeline_version)
        if not self.pipeline_version:
            raise ReportAssemblyError("pipeline_version must not be empty")
        self.low_confidence_threshold = _probability(
            low_confidence_threshold,
            field="low_confidence_threshold",
        )
        self.semantic_margin_threshold = _finite_number(
            semantic_margin_threshold,
            field="semantic_margin_threshold",
        )
        if self.semantic_margin_threshold < 0.0 or self.semantic_margin_threshold > 2.0:
            raise ReportAssemblyError(
                "semantic_margin_threshold must be between 0 and 2"
            )
        self.allow_neutral_compatibility_evidence = bool(
            allow_neutral_compatibility_evidence
        )

    def assemble(
        self,
        segments: Sequence[Any],
        *,
        source_path: Optional[str | Path] = None,
        source_file_name: Optional[str] = None,
        media_type: Optional[str] = None,
        duration_ms: Optional[int] = None,
        duration_sec: Optional[float] = None,
        document_id: Optional[str] = None,
        generated_at: Optional[str] = None,
        title: Optional[str] = None,
        language: str = "und",
        report_locale: Optional[str] = None,
        speaker_count_mode: str = "auto",
        speaker_count: Optional[int] = None,
        minimum_speaker_count: Optional[int] = None,
        maximum_speaker_count: Optional[int] = None,
        speaker_count_confidence: Optional[float] = None,
        speaker_count_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
        speaker_mapping: Optional[Mapping[str, str]] = None,
        speaker_profiles: Optional[Mapping[str, Mapping[str, Any]] | Sequence[Any]] = None,
        models: Optional[Sequence[Mapping[str, Any]]] = None,
        config: Any = None,
        privacy: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        materialized = list(segments or [])
        if not materialized:
            raise ReportAssemblyError("at least one transcription segment is required")

        normalized_language = _normalize_report_language(language, "language")
        normalized_report_locale = (
            None
            if report_locale is None
            else _normalize_report_language(report_locale, "report_locale")
        )

        source = self._build_source(
            source_path=source_path,
            source_file_name=source_file_name,
            media_type=media_type,
            duration_ms=duration_ms,
            duration_sec=duration_sec,
        )
        (
            mapping,
            speaker_ids,
            speaker_policy,
        ) = self._build_speaker_mapping(
            materialized,
            speaker_mapping,
            mode=speaker_count_mode,
            requested_count=speaker_count,
            minimum_count=minimum_speaker_count,
            maximum_count=maximum_speaker_count,
            confidence=speaker_count_confidence,
            candidates=speaker_count_candidates,
        )
        report_segments = self._build_segments(
            materialized,
            mapping=mapping,
            speaker_ids=speaker_ids,
            duration_ms=source["durationMs"],
            language=normalized_language,
        )
        report_models = self._build_models(models, report_segments)
        resolved_document_id = self._document_id(
            document_id,
            source=source,
            segments=report_segments,
        )

        document: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "documentId": resolved_document_id,
            "generatedAt": _utc_iso(generated_at),
            "language": normalized_language,
            "source": source,
            "speakerPolicy": speaker_policy,
            "speakers": self._build_speakers(
                mapping,
                speaker_ids,
                speaker_profiles,
                display_locale=normalized_report_locale or normalized_language,
            ),
            "segments": report_segments,
            "provenance": {
                "pipelineVersion": self.pipeline_version,
                "models": report_models,
                "offline": True,
            },
        }
        if normalized_report_locale is not None:
            document["reportLocale"] = normalized_report_locale
        clean_title = _clean_text(title)
        if clean_title:
            if len(clean_title) > 240:
                raise ReportAssemblyError("title exceeds 240 characters")
            document["title"] = clean_title
        if config is not None:
            document["provenance"]["configSha256"] = _canonical_json_sha256(config)
        if privacy is not None:
            document["privacy"] = self._build_privacy(privacy)

        try:
            validate_report_document(document)
        except (ContractError, TypeError, ValueError) as exc:
            raise ReportAssemblyError(f"assembled ReportDocument is invalid: {exc}") from exc
        self._validate_deep_invariants(document)
        return document

    @staticmethod
    def review_queue(document: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            dict(segment)
            for segment in document.get("segments", [])
            if isinstance(segment, Mapping)
            and segment.get("reviewStatus") == "review-required"
        ]

    def _build_source(
        self,
        *,
        source_path: Optional[str | Path],
        source_file_name: Optional[str],
        media_type: Optional[str],
        duration_ms: Optional[int],
        duration_sec: Optional[float],
    ) -> dict[str, Any]:
        path: Optional[Path] = None
        if source_path is not None:
            path = Path(source_path).expanduser()
            if not path.is_file():
                raise ReportAssemblyError(f"source media does not exist: {path}")
            path = path.resolve()

        if duration_ms is None:
            if duration_sec is None:
                raise ReportAssemblyError("duration_ms or duration_sec is required")
            duration_value = int(round(_finite_number(duration_sec, field="duration_sec") * 1000))
        else:
            try:
                duration_value = int(duration_ms)
            except (TypeError, ValueError) as exc:
                raise ReportAssemblyError("duration_ms must be an integer") from exc
        if duration_value <= 0:
            raise ReportAssemblyError("media duration must be positive")

        name = _clean_text(source_file_name) or (path.name if path else "")
        if not name:
            raise ReportAssemblyError("source_file_name is required without source_path")
        if len(name) > 260:
            raise ReportAssemblyError("source_file_name exceeds 260 characters")

        detected_type = _clean_text(media_type)
        if not detected_type:
            detected_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if len(detected_type) < 3 or len(detected_type) > 120:
            raise ReportAssemblyError("invalid media_type")

        source: dict[str, Any] = {
            "fileName": name,
            "mediaType": detected_type,
            "durationMs": duration_value,
        }
        if path is not None:
            source["sha256"] = _sha256_file(path)
        return source

    def _build_speaker_mapping(
        self,
        segments: Sequence[Any],
        explicit: Optional[Mapping[str, str]],
        *,
        mode: str,
        requested_count: Optional[int],
        minimum_count: Optional[int],
        maximum_count: Optional[int],
        confidence: Optional[float],
        candidates: Optional[Sequence[Mapping[str, Any]]],
    ) -> tuple[dict[str, str], tuple[str, ...], dict[str, Any]]:
        normalized_mode = _clean_text(mode).casefold() or "auto"
        if normalized_mode not in _SPEAKER_COUNT_MODES:
            raise ReportAssemblyError(
                "speaker_count_mode must be auto, manual, or hybrid"
            )

        def normalize_optional_count(value: Optional[int], field: str) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ReportAssemblyError(f"{field} must be an integer")
            result = value
            if result < 1:
                raise ReportAssemblyError(f"{field} must be at least 1")
            return result

        requested = normalize_optional_count(requested_count, "speaker_count")
        minimum = normalize_optional_count(
            minimum_count,
            "minimum_speaker_count",
        )
        maximum = normalize_optional_count(
            maximum_count,
            "maximum_speaker_count",
        )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ReportAssemblyError(
                "minimum_speaker_count cannot exceed maximum_speaker_count"
            )
        if normalized_mode == "manual" and requested is None:
            raise ReportAssemblyError(
                "manual speaker_count_mode requires speaker_count"
            )

        observed: list[str] = []
        seen: set[str] = set()
        for index, segment in enumerate(segments):
            raw = _clean_text(
                _read_value(segment, "speakerId", "speaker_id", "speaker", default="")
            )
            if raw.casefold() in _UNKNOWN_SPEAKER_LABELS:
                raise ReportAssemblyError(
                    f"segments[{index}] has an unknown speaker label; explicit review is required"
                )
            if raw not in seen:
                seen.add(raw)
                observed.append(raw)

        normalized: dict[str, str] = {}
        if explicit is None:
            resolved_count = requested if normalized_mode == "manual" else len(observed)
            if resolved_count is None or resolved_count < 1:
                raise ReportAssemblyError("no resolved speakers are available")
            speaker_ids = canonical_speaker_ids(resolved_count)
            if len(observed) != resolved_count:
                raise ReportAssemblyError(
                    f"{normalized_mode} speaker policy resolved {resolved_count} speakers "
                    f"but the transcript contains {len(observed)} observed labels; "
                    "provide an explicit alias mapping after diarization review"
                )
            if set(observed) == set(speaker_ids):
                normalized = {speaker_id: speaker_id for speaker_id in observed}
            else:
                normalized = {
                    raw: speaker_ids[index]
                    for index, raw in enumerate(observed)
                }
        else:
            for raw_key, raw_value in explicit.items():
                key = _clean_text(raw_key)
                value = _clean_text(raw_value)
                if not key:
                    raise ReportAssemblyError(
                        "speaker_mapping contains an empty source label"
                    )
                if _canonical_speaker_index(value) is None:
                    raise ReportAssemblyError(
                        f"speaker_mapping[{key!r}] must target speaker-N where N >= 1"
                    )
                normalized[key] = value
            missing = [label for label in observed if label not in normalized]
            if missing:
                raise ReportAssemblyError(
                    f"speaker_mapping does not cover observed labels: {missing}"
                )
            mapped_indices = {
                _canonical_speaker_index(normalized[label]) for label in observed
            }
            mapped_indices.discard(None)
            inferred_count = max(mapped_indices, default=0)
            resolved_count = (
                requested
                if normalized_mode == "manual"
                else max(inferred_count, requested or 0)
            )
            if resolved_count < 1:
                raise ReportAssemblyError("speaker_mapping did not resolve any speaker")
            speaker_ids = canonical_speaker_ids(resolved_count)
            mapped_set = {normalized[label] for label in observed}
            if mapped_set != set(speaker_ids):
                raise ReportAssemblyError(
                    "speaker_mapping targets must form a contiguous, fully observed "
                    "speaker-1..speaker-N set"
                )

        resolved_count = len(speaker_ids)
        if minimum is not None and resolved_count < minimum:
            raise ReportAssemblyError(
                f"resolved speaker count {resolved_count} is below minimum {minimum}"
            )
        if maximum is not None and resolved_count > maximum:
            raise ReportAssemblyError(
                f"resolved speaker count {resolved_count} exceeds maximum {maximum}"
            )

        policy: dict[str, Any] = {
            "mode": normalized_mode,
            "resolvedCount": resolved_count,
            "requireExactSet": True,
            "speakerIds": list(speaker_ids),
            "unknownSpeakerAllowed": False,
            "speakerChangeRequiresEvidence": True,
        }
        if requested is not None:
            policy["requestedCount"] = requested
        if minimum is not None:
            policy["minimumCount"] = minimum
        if maximum is not None:
            policy["maximumCount"] = maximum
        if normalized_mode in {"auto", "hybrid"}:
            detection: dict[str, Any] = {
                "provider": "diarization-consensus",
                "estimatedCount": resolved_count,
            }
            if confidence is not None:
                detection["confidence"] = _probability(
                    confidence,
                    field="speaker_count_confidence",
                )
            if candidates is not None:
                normalized_candidates: list[dict[str, Any]] = []
                for index, candidate in enumerate(candidates):
                    if not isinstance(candidate, Mapping):
                        raise ReportAssemblyError(
                            f"speaker_count_candidates[{index}] must be an object"
                        )
                    count = normalize_optional_count(
                        candidate.get("count"),
                        f"speaker_count_candidates[{index}].count",
                    )
                    probability = _probability(
                        candidate.get("confidence"),
                        field=f"speaker_count_candidates[{index}].confidence",
                    )
                    normalized_candidates.append(
                        {"count": count, "confidence": probability}
                    )
                if normalized_candidates:
                    detection["candidates"] = normalized_candidates
            policy["detection"] = detection
        return normalized, speaker_ids, policy

    def _build_speakers(
        self,
        mapping: Mapping[str, str],
        speaker_ids: Sequence[str],
        profiles: Optional[Mapping[str, Mapping[str, Any]] | Sequence[Any]],
        *,
        display_locale: str,
    ) -> list[dict[str, Any]]:
        speaker_set = frozenset(speaker_ids)
        aliases: dict[str, list[str]] = {speaker_id: [] for speaker_id in speaker_ids}
        for raw, canonical in mapping.items():
            if raw != canonical and raw not in aliases[canonical]:
                aliases[canonical].append(raw)

        profile_by_id: dict[str, Mapping[str, Any]] = {}
        if isinstance(profiles, Mapping):
            for key, value in profiles.items():
                canonical = mapping.get(_clean_text(key), _clean_text(key))
                if canonical in speaker_set and isinstance(value, Mapping):
                    profile_by_id[canonical] = value
        elif isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
            for value in profiles:
                if not isinstance(value, Mapping):
                    continue
                canonical = _clean_text(value.get("id"))
                if canonical in speaker_set:
                    profile_by_id[canonical] = value

        output: list[dict[str, Any]] = []
        for index, speaker_id in enumerate(speaker_ids, start=1):
            profile = profile_by_id.get(speaker_id, {})
            display_name = _clean_text(
                profile.get("displayName", profile.get("display_name"))
            ) or _default_speaker_display_name(index, display_locale)
            short_label = _clean_text(
                profile.get("shortLabel", profile.get("short_label"))
            ) or f"S{index}"
            if len(display_name) > 80 or len(short_label) > 16:
                raise ReportAssemblyError(f"{speaker_id} profile exceeds contract limits")
            entry: dict[str, Any] = {
                "id": speaker_id,
                "order": index,
                "displayName": display_name,
                "shortLabel": short_label,
                "colorToken": f"speaker.{index}",
            }
            role = _clean_text(profile.get("role"))
            if role:
                if len(role) > 80:
                    raise ReportAssemblyError(f"{speaker_id} role exceeds 80 characters")
                entry["role"] = role
            explicit_aliases = profile.get("aliases")
            if isinstance(explicit_aliases, Sequence) and not isinstance(
                explicit_aliases, (str, bytes)
            ):
                for alias in explicit_aliases:
                    text = _clean_text(alias)
                    if text and text not in aliases[speaker_id]:
                        aliases[speaker_id].append(text)
            if aliases[speaker_id]:
                entry["aliases"] = aliases[speaker_id]
            output.append(entry)
        return output

    def _build_segments(
        self,
        segments: Sequence[Any],
        *,
        mapping: Mapping[str, str],
        speaker_ids: Sequence[str],
        duration_ms: int,
        language: str,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        used_revision_ids: set[str] = set()
        previous_start = -1

        for index, source_segment in enumerate(segments, start=1):
            segment_id = _clean_text(
                _read_value(source_segment, "id", "segmentId", "segment_id", default="")
            ) or f"segment-{index:06d}"
            if not _SEGMENT_ID_PATTERN.fullmatch(segment_id):
                raise ReportAssemblyError(f"invalid segment id: {segment_id!r}")
            if segment_id in seen_ids:
                raise ReportAssemblyError(f"duplicate segment id: {segment_id}")
            seen_ids.add(segment_id)

            start_ms = self._segment_ms(source_segment, "start", "startMs", "start_ms")
            end_ms = self._segment_ms(source_segment, "end", "endMs", "end_ms")
            if start_ms < previous_start:
                raise ReportAssemblyError(f"{segment_id}: start time is not monotonic")
            if end_ms <= start_ms:
                raise ReportAssemblyError(f"{segment_id}: end time must be after start time")
            if end_ms > duration_ms:
                raise ReportAssemblyError(
                    f"{segment_id}: end time {end_ms} exceeds media duration {duration_ms}"
                )
            previous_start = start_ms

            raw_speaker = _clean_text(
                _read_value(
                    source_segment,
                    "speakerId",
                    "speaker_id",
                    "speaker",
                    default="",
                )
            )
            if raw_speaker not in mapping:
                raise ReportAssemblyError(
                    f"{segment_id}: speaker {raw_speaker!r} is not mapped"
                )
            speaker_id = mapping[raw_speaker]

            fallback_text = _clean_text(_read_value(source_segment, "text", default=""))
            raw_text = _clean_text(
                _read_value(
                    source_segment,
                    "rawText",
                    "raw_text",
                    "originalText",
                    "original_text",
                    "asrText",
                    "asr_text",
                    default=fallback_text,
                )
            )
            normalized_text = _clean_text(
                _read_value(
                    source_segment,
                    "normalizedText",
                    "normalized_text",
                    "semanticText",
                    "semantic_text",
                    default=fallback_text or raw_text,
                )
            )
            display_text = _clean_text(
                _read_value(
                    source_segment,
                    "displayText",
                    "display_text",
                    default=normalized_text,
                )
            )
            if not raw_text or not normalized_text or not display_text:
                raise ReportAssemblyError(
                    f"{segment_id}: raw, normalized, and display text must be non-empty"
                )

            revisions = self._normalize_revisions(
                _read_value(source_segment, "revisions", default=[]),
                mapping=mapping,
                speaker_ids=speaker_ids,
                segment_id=segment_id,
                used_revision_ids=used_revision_ids,
            )
            revision_source = _clean_text(
                _read_value(
                    source_segment,
                    "normalizationSource",
                    "normalization_source",
                    "textRevisionSource",
                    "text_revision_source",
                    default="deterministic",
                )
            )
            if revision_source not in _REVISION_SOURCES:
                revision_source = "deterministic"
            if revision_source == "llm":
                raise ReportAssemblyError(
                    f"{segment_id}: the local semantic model is "
                    "suggestion-only/disabled and "
                    "cannot author text revisions"
                )
            normalization_reason = _clean_text(
                _read_value(
                    source_segment,
                    "normalizationReasonCode",
                    "normalization_reason_code",
                    default="SEMANTIC_CORRECTION"
                    if revision_source == "llm"
                    else "TEXT_NORMALIZATION",
                )
            )
            if not _REASON_CODE_PATTERN.fullmatch(normalization_reason):
                raise ReportAssemblyError(
                    f"{segment_id}: invalid normalization reason code"
                )
            if raw_text != normalized_text and not self._has_text_trace(
                revisions, raw_text, normalized_text
            ):
                revisions.append(
                    self._new_text_revision(
                        segment_id=segment_id,
                        used_revision_ids=used_revision_ids,
                        source=revision_source,
                        reason_code=normalization_reason,
                        before=raw_text,
                        after=normalized_text,
                        model=_clean_text(
                            _read_value(
                                source_segment,
                                "normalizationModel",
                                "normalization_model",
                                default="",
                            )
                        ),
                    )
                )
            if normalized_text != display_text and not self._has_text_trace(
                revisions, normalized_text, display_text
            ):
                revisions.append(
                    self._new_text_revision(
                        segment_id=segment_id,
                        used_revision_ids=used_revision_ids,
                        source="deterministic",
                        reason_code="DISPLAY_TEXT_FORMATTING",
                        before=normalized_text,
                        after=display_text,
                    )
                )

            confidence = _probability(
                _read_value(source_segment, "confidence", default=0.0),
                field=f"{segment_id}.confidence",
            )
            evidence, incomplete_speaker_evidence = self._build_evidence(
                source_segment,
                mapping=mapping,
                speaker_ids=speaker_ids,
                segment_id=segment_id,
                speaker_id=speaker_id,
                confidence=confidence,
                revisions=revisions,
                word_count=len(
                    _read_value(source_segment, "words", default=[]) or []
                ),
            )
            review_status = self._review_status(
                source_segment,
                confidence=confidence,
                evidence=evidence,
                incomplete_speaker_evidence=incomplete_speaker_evidence,
            )

            segment_language_raw = _read_value(
                source_segment,
                "language",
                default=None,
            )
            segment_language = (
                language
                if segment_language_raw is None
                else _normalize_report_language(
                    segment_language_raw,
                    f"{segment_id}.language",
                )
            )
            entry: dict[str, Any] = {
                "id": segment_id,
                "startMs": start_ms,
                "endMs": end_ms,
                "speakerId": speaker_id,
                "rawText": raw_text,
                "normalizedText": normalized_text,
                "displayText": display_text,
                "language": segment_language,
                "confidence": confidence,
                "reviewStatus": review_status,
                "evidence": evidence,
                "revisions": revisions,
            }
            overlap_group = _clean_text(
                _read_value(
                    source_segment,
                    "overlapGroupId",
                    "overlap_group_id",
                    default="",
                )
            )
            if overlap_group:
                entry["overlapGroupId"] = overlap_group[:120]
            source_ids = _read_value(
                source_segment,
                "sourceSegmentIds",
                "source_segment_ids",
                default=None,
            )
            if isinstance(source_ids, Sequence) and not isinstance(
                source_ids, (str, bytes)
            ):
                normalized_ids = []
                for value in source_ids:
                    text = _clean_text(value)
                    if text and text not in normalized_ids:
                        normalized_ids.append(text[:120])
                if normalized_ids:
                    entry["sourceSegmentIds"] = normalized_ids
            output.append(entry)
        return output

    @staticmethod
    def _segment_ms(
        segment: Any,
        seconds_name: str,
        milliseconds_name: str,
        milliseconds_snake_name: str,
    ) -> int:
        explicit = _read_value(
            segment,
            milliseconds_name,
            milliseconds_snake_name,
            default=None,
        )
        if explicit is not None:
            number = _finite_number(explicit, field=milliseconds_name)
            rounded = int(round(number))
        else:
            seconds = _read_value(segment, seconds_name, default=None)
            if seconds is None:
                raise ReportAssemblyError(
                    f"segment is missing {seconds_name}/{milliseconds_name}"
                )
            rounded = int(round(_finite_number(seconds, field=seconds_name) * 1000))
        if rounded < 0:
            raise ReportAssemblyError(f"{milliseconds_name} must not be negative")
        return rounded

    def _normalize_revisions(
        self,
        values: Any,
        *,
        mapping: Mapping[str, str],
        speaker_ids: Sequence[str],
        segment_id: str,
        used_revision_ids: set[str],
    ) -> list[dict[str, Any]]:
        if values is None:
            return []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ReportAssemblyError(f"{segment_id}.revisions must be an array")
        output: list[dict[str, Any]] = []
        for index, value in enumerate(values, start=1):
            if not isinstance(value, Mapping):
                raise ReportAssemblyError(
                    f"{segment_id}.revisions[{index - 1}] must be an object"
                )
            revision_id = _clean_text(
                value.get(
                    "revisionId",
                    value.get("revision_id", value.get("id")),
                )
            )
            if not revision_id:
                revision_id = self._allocate_revision_id(
                    f"{segment_id}-revision-{index:03d}", used_revision_ids
                )
            elif len(revision_id) > 120 or revision_id in used_revision_ids:
                raise ReportAssemblyError(
                    f"{segment_id}: invalid or duplicate revision id {revision_id!r}"
                )
            used_revision_ids.add(revision_id)
            revision_type = _clean_text(value.get("type"))
            source = _clean_text(value.get("source"))
            reason = _clean_text(
                value.get("reasonCode", value.get("reason_code"))
            )
            if revision_type not in _REVISION_TYPES:
                raise ReportAssemblyError(
                    f"{segment_id}: unsupported revision type {revision_type!r}"
                )
            if source not in _REVISION_SOURCES:
                raise ReportAssemblyError(
                    f"{segment_id}: unsupported revision source {source!r}"
                )
            if source == "llm":
                raise ReportAssemblyError(
                    f"{segment_id}: LLM revisions are forbidden in production output"
                )
            if source != "manual" and revision_type in {
                "boundary",
                "split",
                "merge",
            }:
                raise ReportAssemblyError(
                    f"{segment_id}: automatic boundary/turn revisions are forbidden"
                )
            if not _REASON_CODE_PATTERN.fullmatch(reason):
                raise ReportAssemblyError(
                    f"{segment_id}: invalid revision reason code {reason!r}"
                )
            before = value.get("before")
            after = value.get("after")
            if before == after:
                raise ReportAssemblyError(
                    f"{segment_id}: revisions must record a real change"
                )
            if revision_type == "speaker":
                before = mapping.get(_clean_text(before), _clean_text(before))
                after = mapping.get(_clean_text(after), _clean_text(after))
                if before not in speaker_ids:
                    raise ReportAssemblyError(
                        f"{segment_id}: speaker revision has invalid before value"
                    )
                if after not in speaker_ids:
                    raise ReportAssemblyError(
                        f"{segment_id}: speaker revision has invalid after value"
                    )
            entry: dict[str, Any] = {
                "revisionId": revision_id,
                "type": revision_type,
                "source": source,
                "reasonCode": reason,
                "before": before,
                "after": after,
            }
            if "confidence" in value:
                entry["confidence"] = _probability(
                    value.get("confidence"),
                    field=f"{segment_id}.revisions[{index - 1}].confidence",
                )
            raw_refs = value.get(
                "evidenceRefs",
                value.get("evidence_refs"),
            )
            if raw_refs is not None:
                if not isinstance(raw_refs, Sequence) or isinstance(
                    raw_refs,
                    (str, bytes),
                ):
                    raise ReportAssemblyError(
                        f"{segment_id}.revisions[{index - 1}].evidenceRefs "
                        "must be an array"
                    )
                refs: list[str] = []
                for raw_ref in raw_refs:
                    ref = _clean_text(raw_ref)
                    if not ref or len(ref) > 240 or ref in refs:
                        raise ReportAssemblyError(
                            f"{segment_id}.revisions[{index - 1}].evidenceRefs "
                            "contains an invalid or duplicate reference"
                        )
                    refs.append(ref)
                if not refs:
                    raise ReportAssemblyError(
                        f"{segment_id}.revisions[{index - 1}].evidenceRefs "
                        "must not be empty"
                    )
                entry["evidenceRefs"] = refs
            for source_key, target_key in (
                ("model", "model"),
                ("actor", "actor"),
                ("occurredAt", "occurredAt"),
                ("occurred_at", "occurredAt"),
            ):
                text = _clean_text(value.get(source_key))
                if text and target_key not in entry:
                    entry[target_key] = text
            output.append(entry)
        return output

    @staticmethod
    def _allocate_revision_id(preferred: str, used: set[str]) -> str:
        candidate = preferred[:120]
        suffix = 1
        while candidate in used:
            tail = f"-{suffix}"
            candidate = f"{preferred[: 120 - len(tail)]}{tail}"
            suffix += 1
        used.add(candidate)
        return candidate

    def _new_text_revision(
        self,
        *,
        segment_id: str,
        used_revision_ids: set[str],
        source: str,
        reason_code: str,
        before: str,
        after: str,
        model: str = "",
    ) -> dict[str, Any]:
        revision_id = self._allocate_revision_id(
            f"{segment_id}-text-{len(used_revision_ids) + 1:04d}",
            used_revision_ids,
        )
        entry: dict[str, Any] = {
            "revisionId": revision_id,
            "type": "text",
            "source": source,
            "reasonCode": reason_code,
            "before": before,
            "after": after,
        }
        if model:
            entry["model"] = model[:160]
        return entry

    @staticmethod
    def _has_text_trace(
        revisions: Sequence[Mapping[str, Any]],
        before: str,
        after: str,
    ) -> bool:
        text_revisions = [
            revision for revision in revisions if revision.get("type") == "text"
        ]
        if not text_revisions:
            return False
        current = before
        progressed = False
        for revision in text_revisions:
            if revision.get("before") == current:
                current = revision.get("after")
                progressed = True
                if current == after:
                    return True
        return progressed and current == after

    def _build_evidence(
        self,
        segment: Any,
        *,
        mapping: Mapping[str, str],
        speaker_ids: Sequence[str],
        segment_id: str,
        speaker_id: str,
        confidence: float,
        revisions: Sequence[Mapping[str, Any]],
        word_count: int,
    ) -> tuple[dict[str, Any], bool]:
        raw_evidence = _read_value(segment, "evidence", default={})
        if not isinstance(raw_evidence, Mapping):
            raw_evidence = {}

        asr_source = raw_evidence.get("asr")
        if not isinstance(asr_source, Mapping):
            asr_source = {}
        asr: dict[str, Any] = {
            "provider": _clean_text(
                asr_source.get(
                    "provider",
                    _read_value(segment, "asrProvider", "asr_provider", default="qwen-asr"),
                )
            )
            or "qwen-asr",
            "model": _clean_text(
                asr_source.get(
                    "model",
                    _read_value(
                        segment,
                        "asrModel",
                        "asr_model",
                        default="Qwen3-ASR-1.7B",
                    ),
                )
            )
            or "Qwen3-ASR-1.7B",
            "confidence": _probability(
                asr_source.get("confidence", confidence),
                field=f"{segment_id}.evidence.asr.confidence",
            ),
        }
        if "confidenceAvailable" in asr_source or "confidence_available" in asr_source:
            confidence_available = asr_source.get(
                "confidenceAvailable",
                asr_source.get("confidence_available"),
            )
            if not isinstance(confidence_available, bool):
                raise ReportAssemblyError(
                    f"{segment_id}.evidence.asr.confidenceAvailable must be a boolean"
                )
            asr["confidenceAvailable"] = confidence_available
        model_revision = _clean_text(
            asr_source.get("modelRevision", asr_source.get("model_revision"))
        )
        if model_revision:
            asr["modelRevision"] = model_revision[:160]
        source_word_count = asr_source.get("wordCount", asr_source.get("word_count"))
        if source_word_count is not None:
            try:
                normalized_word_count = int(source_word_count)
            except (TypeError, ValueError) as exc:
                raise ReportAssemblyError(
                    f"{segment_id}.evidence.asr.wordCount must be an integer"
                ) from exc
            if normalized_word_count < 0:
                raise ReportAssemblyError(
                    f"{segment_id}.evidence.asr.wordCount must not be negative"
                )
            asr["wordCount"] = normalized_word_count
        elif word_count:
            asr["wordCount"] = word_count

        boundary_source = raw_evidence.get("boundary")
        if not isinstance(boundary_source, Mapping):
            boundary_source = {}
        overlap_detected = boundary_source.get(
            "overlapDetected",
            boundary_source.get(
                "overlap_detected",
                _read_value(
                    segment,
                    "overlapDetected",
                    "overlap_detected",
                    "overlapping",
                    default=False,
                ),
            ),
        )
        if not isinstance(overlap_detected, bool):
            raise ReportAssemblyError(
                f"{segment_id}.evidence.boundary.overlapDetected must be boolean"
            )
        boundary: dict[str, Any] = {
            "provider": _clean_text(
                boundary_source.get(
                    "provider",
                    _read_value(
                        segment,
                        "boundaryProvider",
                        "boundary_provider",
                        default="funasr",
                    ),
                )
            )
            or "funasr",
            "confidence": _probability(
                boundary_source.get(
                    "confidence",
                    _read_value(
                        segment,
                        "boundaryConfidence",
                        "boundary_confidence",
                        default=0.0,
                    ),
                ),
                field=f"{segment_id}.evidence.boundary.confidence",
            ),
            "overlapDetected": overlap_detected,
        }
        boundary_model = _clean_text(
            boundary_source.get(
                "model",
                _read_value(
                    segment,
                    "boundaryModel",
                    "boundary_model",
                    default="FunASR",
                ),
            )
        )
        if boundary_model:
            boundary["model"] = boundary_model[:160]

        speaker, incomplete = self._speaker_evidence(
            segment,
            raw_evidence=raw_evidence,
            mapping=mapping,
            speaker_ids=speaker_ids,
            segment_id=segment_id,
            speaker_id=speaker_id,
        )
        self._validate_speaker_decision(
            segment_id=segment_id,
            speaker_id=speaker_id,
            speaker_evidence=speaker,
            speaker_ids=speaker_ids,
            revisions=revisions,
            raw_evidence=raw_evidence,
        )

        evidence: dict[str, Any] = {
            "asr": asr,
            "boundary": boundary,
            "speaker": speaker,
        }
        semantic = self._semantic_evidence(
            segment,
            raw_evidence=raw_evidence,
            revisions=revisions,
            incomplete_speaker_evidence=incomplete,
        )
        if semantic is not None:
            evidence["semantic"] = semantic
        audio_review = self._audio_review(
            raw_evidence=raw_evidence,
            requires_review=incomplete
            or (
                semantic is not None
                and semantic.get("decision") == "review-required"
            ),
        )
        if audio_review is not None:
            evidence["audioReview"] = audio_review
        verified_mapping_revisions = [
            revision
            for revision in revisions
            if is_verified_pyannote_speaker_revision(
                revision,
                raw_evidence,
                set(speaker_ids),
            )
        ]
        if verified_mapping_revisions:
            proof = raw_evidence["pyannoteCanonicalMapping"]
            overlap = raw_evidence["overlap"]
            evidence["speakerMapping"] = {
                **dict(proof),
                "evidenceRefs": sorted(
                    {
                        ref
                        for revision in verified_mapping_revisions
                        for ref in revision["evidenceRefs"]
                    }
                ),
                "canonicalSpeakerTurns": [
                    dict(turn)
                    for turn in overlap["canonicalSpeakerTurns"]
                ],
            }
        return evidence, incomplete

    def _speaker_evidence(
        self,
        segment: Any,
        *,
        raw_evidence: Mapping[str, Any],
        mapping: Mapping[str, str],
        speaker_ids: Sequence[str],
        segment_id: str,
        speaker_id: str,
    ) -> tuple[dict[str, Any], bool]:
        source = raw_evidence.get("speaker")
        if not isinstance(source, Mapping):
            source = {}

        score_map: dict[str, float] = {}
        raw_scores = source.get(
            "scores",
            _read_value(segment, "speakerScores", "speaker_scores", default=None),
        )
        if isinstance(raw_scores, Mapping):
            iterable = [
                {"speaker": key, "score": value} for key, value in raw_scores.items()
            ]
        elif isinstance(raw_scores, Sequence) and not isinstance(
            raw_scores, (str, bytes)
        ):
            iterable = list(raw_scores)
        else:
            raise ReportAssemblyError(
                f"{segment_id}: speaker scores must be an exact array/object vector"
            )
        for index, value in enumerate(iterable):
            if not isinstance(value, Mapping):
                raise ReportAssemblyError(
                    f"{segment_id}.speakerScores[{index}] must be an object"
                )
            raw_speaker = _clean_text(
                value.get(
                    "speakerId",
                    value.get("speaker_id", value.get("speaker")),
                )
            )
            canonical = mapping.get(raw_speaker, raw_speaker)
            if canonical not in speaker_ids:
                raise ReportAssemblyError(
                    f"{segment_id}: speaker score targets unrequested speaker "
                    f"{raw_speaker!r}"
                )
            if canonical in score_map:
                raise ReportAssemblyError(
                    f"{segment_id}: duplicate score for {canonical}"
                )
            if "score" in value:
                raw_score = value["score"]
            elif "confidence" in value:
                raw_score = value["confidence"]
            else:
                raise ReportAssemblyError(
                    f"{segment_id}.speakerScores[{index}] is missing score"
                )
            score_map[canonical] = _speaker_score(
                raw_score,
                field=f"{segment_id}.speakerScore[{index}]",
            )

        missing = [
            canonical for canonical in speaker_ids if canonical not in score_map
        ]
        if missing or len(score_map) != len(speaker_ids):
            raise ReportAssemblyError(
                f"{segment_id}: speaker evidence is missing scores for {missing}; "
                "partial vectors always fail closed"
            )
        incomplete = False

        averaged_scores = dict(score_map)
        ranked = sorted(
            averaged_scores.items(),
            key=lambda item: (-item[1], speaker_ids.index(item[0])),
        )
        computed_margin = (
            ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 2.0
        )
        margin_value = source.get(
            "margin",
            _read_value(
                segment,
                "speakerMargin",
                "speaker_margin",
                default=computed_margin,
            ),
        )
        margin = _finite_number(
            margin_value,
            field=f"{segment_id}.evidence.speaker.margin",
        )
        if margin < -2.0 or margin > 2.0:
            raise ReportAssemblyError(
                f"{segment_id}.evidence.speaker.margin must be between -2 and 2"
            )
        if abs(margin - computed_margin) > 1e-6:
            raise ReportAssemblyError(
                f"{segment_id}.evidence.speaker.margin does not match the score vector"
            )

        locked = bool(
            source.get(
                "locked",
                _read_value(
                    segment,
                    "speakerLocked",
                    "speaker_locked",
                    "humanLocked",
                    "locked",
                    default=False,
                ),
            )
        )
        provider = _clean_text(
            source.get(
                "provider",
                _read_value(
                    segment,
                    "speakerProvider",
                    "speaker_provider",
                    default="camp-plus",
                ),
            )
        ) or "camp-plus"
        model = _clean_text(
            source.get(
                "model",
                _read_value(
                    segment,
                    "speakerModel",
                    "speaker_model",
                    default="CAM++",
                ),
            )
        )
        evidence: dict[str, Any] = {
            "provider": provider[:80],
            "assignment": speaker_id,
            "locked": locked,
            "margin": margin,
            "scores": [
                {
                    "speakerId": canonical,
                    "score": averaged_scores[canonical],
                }
                for canonical in speaker_ids
            ],
        }
        if model:
            evidence["model"] = model[:160]
        return evidence, incomplete

    @staticmethod
    def _collect_scores(
        values: Sequence[Any],
        *,
        score_map: dict[str, list[float]],
        mapping: Mapping[str, str],
        segment_id: str,
        value_names: Sequence[str],
    ) -> None:
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                continue
            raw_speaker = _clean_text(
                value.get("speakerId", value.get("speaker_id", value.get("speaker")))
            )
            canonical = mapping.get(raw_speaker, raw_speaker)
            if canonical not in score_map:
                continue
            score_value = None
            for name in value_names:
                if name in value:
                    score_value = value[name]
                    break
            if score_value is None:
                continue
            score_map[canonical].append(
                _speaker_score(
                    score_value,
                    field=f"{segment_id}.speakerScore[{index}]",
                )
            )

    def _validate_speaker_decision(
        self,
        *,
        segment_id: str,
        speaker_id: str,
        speaker_evidence: Mapping[str, Any],
        speaker_ids: Sequence[str],
        revisions: Sequence[Mapping[str, Any]],
        raw_evidence: Mapping[str, Any],
    ) -> None:
        scores = {
            item["speakerId"]: float(item["score"])
            for item in speaker_evidence["scores"]
        }
        ranked = sorted(
            scores,
            key=lambda candidate: (
                -scores[candidate],
                speaker_ids.index(candidate),
            ),
        )
        acoustic_top = ranked[0]
        margin = float(speaker_evidence.get("margin", 0.0))
        speaker_revisions = [
            revision for revision in revisions if revision.get("type") == "speaker"
        ]
        non_manual_revisions = [
            revision
            for revision in speaker_revisions
            if revision.get("source") != "manual"
        ]
        verified_pyannote_revisions = {
            str(revision.get("revisionId") or "")
            for revision in speaker_revisions
            if is_verified_pyannote_speaker_revision(
                revision,
                raw_evidence,
                set(speaker_ids),
            )
        }
        unsafe_non_manual_revisions = [
            revision
            for revision in non_manual_revisions
            if str(revision.get("revisionId") or "")
            not in verified_pyannote_revisions
        ]
        locked = bool(speaker_evidence.get("locked"))
        semantic = raw_evidence.get("semantic")
        if not isinstance(semantic, Mapping):
            semantic = {}
        boundary = raw_evidence.get("boundary")
        overlap_detected = (
            isinstance(boundary, Mapping)
            and boundary.get(
                "overlapDetected",
                boundary.get("overlap_detected"),
            )
            is True
        )

        if locked and (
            non_manual_revisions
            or (
                semantic.get("decision") == "suggest-speaker"
                and bool(semantic.get("autoApplied", semantic.get("auto_applied")))
            )
        ):
            raise ReportAssemblyError(
                f"{segment_id}: a locked speaker cannot be overridden by semantic logic"
            )
        if overlap_detected and unsafe_non_manual_revisions:
            raise ReportAssemblyError(
                f"{segment_id}: overlap/串话 speaker decisions require manual review"
            )
        if (
            unsafe_non_manual_revisions
            and margin >= self.semantic_margin_threshold
        ):
            raise ReportAssemblyError(
                f"{segment_id}: automatic speaker override is forbidden at acoustic "
                f"margin {margin:.3f}"
            )

        canonical = set(speaker_ids)
        top_two = set(ranked[:2])
        effective_speaker = acoustic_top
        for revision in speaker_revisions:
            before = revision.get("before")
            after = revision.get("after")
            source = revision.get("source")
            if before not in canonical or after not in canonical:
                raise ReportAssemblyError(
                    f"{segment_id}: speaker revisions must use canonical speaker IDs"
                )
            if before != effective_speaker:
                raise ReportAssemblyError(
                    f"{segment_id}: speaker revision chain is not contiguous"
                )
            if before == after:
                raise ReportAssemblyError(
                    f"{segment_id}: speaker revisions must record a change"
                )
            if (
                source != "manual"
                and after not in top_two
                and str(revision.get("revisionId") or "")
                not in verified_pyannote_revisions
            ):
                raise ReportAssemblyError(
                    f"{segment_id}: automatic speaker override must remain inside "
                    "the acoustic top-2"
                )
            effective_speaker = after

        if speaker_revisions and effective_speaker != speaker_id:
            raise ReportAssemblyError(
                f"{segment_id}: final speaker revision must match the assigned speaker"
            )
        if speaker_id != acoustic_top:
            if not speaker_revisions or effective_speaker != speaker_id:
                raise ReportAssemblyError(
                    f"{segment_id}: non-top acoustic assignment requires an auditable "
                    "speaker revision chain"
                )

    def _semantic_evidence(
        self,
        segment: Any,
        *,
        raw_evidence: Mapping[str, Any],
        revisions: Sequence[Mapping[str, Any]],
        incomplete_speaker_evidence: bool,
    ) -> Optional[dict[str, Any]]:
        source = raw_evidence.get("semantic")
        if not isinstance(source, Mapping):
            source = {}
        has_llm_text_revision = any(
            revision.get("type") == "text" and revision.get("source") == "llm"
            for revision in revisions
        )
        if not source and not has_llm_text_revision and not incomplete_speaker_evidence:
            return None

        decision = _clean_text(source.get("decision"))
        if incomplete_speaker_evidence:
            decision = "review-required"
        elif not decision:
            decision = "normalize" if has_llm_text_revision else "keep"
        allowed_decisions = {
            "keep",
            "normalize",
            "suggest-speaker",
            "review-required",
            "reject",
        }
        if decision not in allowed_decisions:
            raise ReportAssemblyError(f"unsupported semantic decision {decision!r}")
        auto_applied = bool(
            source.get(
                "autoApplied",
                source.get("auto_applied", has_llm_text_revision),
            )
        )
        if auto_applied:
            raise ReportAssemblyError(
                "the local semantic model is not production-eligible; "
                "semantic auto-apply is forbidden"
            )
        mode = _clean_text(source.get("mode", source.get("applyMode")))
        if mode and mode not in {"disabled", "suggestion-only"}:
            raise ReportAssemblyError(
                "semantic mode must be disabled or suggestion-only"
            )
        if incomplete_speaker_evidence:
            auto_applied = False
        output: dict[str, Any] = {
            "provider": _clean_text(source.get("provider")) or "local-llm",
            "decision": decision,
            "autoApplied": auto_applied,
        }
        model = _clean_text(source.get("model"))
        if model:
            output["model"] = model[:160]
        reasons = source.get("reasonCodes", source.get("reason_codes", []))
        normalized_reasons: list[str] = []
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
            for reason in reasons:
                text = _clean_text(reason)
                if not text:
                    continue
                if not _REASON_CODE_PATTERN.fullmatch(text):
                    raise ReportAssemblyError(
                        f"invalid semantic reason code {text!r}"
                    )
                if text not in normalized_reasons:
                    normalized_reasons.append(text)
        if incomplete_speaker_evidence:
            normalized_reasons.append("INCOMPLETE_ACOUSTIC_EVIDENCE")
        if normalized_reasons:
            output["reasonCodes"] = normalized_reasons
        return output

    @staticmethod
    def _audio_review(
        *,
        raw_evidence: Mapping[str, Any],
        requires_review: bool,
    ) -> Optional[dict[str, Any]]:
        source = raw_evidence.get("audioReview")
        if not isinstance(source, Mapping):
            source = raw_evidence.get("audio_review")
        if not isinstance(source, Mapping):
            if not requires_review:
                return None
            return {"status": "queued"}
        status = _clean_text(source.get("status")) or (
            "queued" if requires_review else "not-needed"
        )
        if status not in {
            "not-needed",
            "queued",
            "model-reviewed",
            "human-reviewed",
        }:
            raise ReportAssemblyError(f"unsupported audio review status {status!r}")
        output: dict[str, Any] = {"status": status}
        reviewer = _clean_text(source.get("reviewer"))
        notes = _clean_text(source.get("notes"))
        if reviewer:
            output["reviewer"] = reviewer[:120]
        if notes:
            output["notes"] = notes[:1000]
        return output

    def _review_status(
        self,
        segment: Any,
        *,
        confidence: float,
        evidence: Mapping[str, Any],
        incomplete_speaker_evidence: bool,
    ) -> str:
        explicit = _clean_text(
            _read_value(
                segment,
                "reviewStatus",
                "review_status",
                default="",
            )
        )
        if explicit and explicit not in _REVIEW_STATUSES:
            raise ReportAssemblyError(f"unsupported review status {explicit!r}")
        speaker = evidence["speaker"]
        semantic = evidence.get("semantic", {})
        audio = evidence.get("audioReview", {})
        must_review = (
            incomplete_speaker_evidence
            or confidence < self.low_confidence_threshold
            or abs(float(speaker.get("margin", 0.0))) < self.semantic_margin_threshold
            or semantic.get("decision") in {"review-required", "reject"}
            or audio.get("status") == "queued"
        )
        if bool(speaker.get("locked")):
            return explicit if explicit in {"locked", "manually-reviewed"} else "locked"
        if explicit == "manually-reviewed" or audio.get("status") == "human-reviewed":
            return "manually-reviewed"
        if must_review:
            return "review-required"
        return explicit or "accepted"

    def _build_models(
        self,
        values: Optional[Sequence[Mapping[str, Any]]],
        segments: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if values is None:
            values = [
                {"role": "asr", "name": "Qwen3-ASR-1.7B"},
                {"role": "boundary", "name": "FunASR"},
                {"role": "speaker", "name": "CAM++"},
            ]
            if any(
                isinstance(segment.get("evidence", {}).get("semantic"), Mapping)
                for segment in segments
            ):
                values = [*values, {"role": "semantic", "name": "local-llm"}]
        allowed_roles = {"asr", "vad", "boundary", "speaker", "semantic", "overlap"}
        output: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ReportAssemblyError(f"models[{index}] must be an object")
            role = _clean_text(value.get("role"))
            name = _clean_text(value.get("name"))
            if role not in allowed_roles or not name:
                raise ReportAssemblyError(f"models[{index}] has an invalid role or name")
            entry: dict[str, Any] = {"role": role, "name": name[:160]}
            revision = _clean_text(value.get("revision"))
            quantization = _clean_text(value.get("quantization"))
            if revision:
                entry["revision"] = revision[:160]
            if quantization:
                entry["quantization"] = quantization[:80]
            output.append(entry)
        if not output:
            raise ReportAssemblyError("at least one model provenance record is required")
        return output

    @staticmethod
    def _document_id(
        requested: Optional[str],
        *,
        source: Mapping[str, Any],
        segments: Sequence[Mapping[str, Any]],
    ) -> str:
        if requested:
            value = _clean_text(requested)
            if not _DOCUMENT_ID_PATTERN.fullmatch(value):
                raise ReportAssemblyError(
                    "document_id must be 8-160 ASCII letters, digits, dot, colon, "
                    "underscore, or hyphen"
                )
            return value
        seed = {
            "fileName": source["fileName"],
            "durationMs": source["durationMs"],
            "sha256": source.get("sha256"),
            "segments": [
                (
                    segment["id"],
                    segment["startMs"],
                    segment["endMs"],
                    segment["speakerId"],
                    segment["rawText"],
                )
                for segment in segments
            ],
        }
        return f"mts-{_canonical_json_sha256(seed)[:24]}"

    @staticmethod
    def _build_privacy(value: Mapping[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key in ("containsRealMeetingText", "exportApproved"):
            snake = (
                "contains_real_meeting_text"
                if key == "containsRealMeetingText"
                else "export_approved"
            )
            if key in value or snake in value:
                output[key] = bool(value.get(key, value.get(snake)))
        notes = _clean_text(value.get("redactionNotes", value.get("redaction_notes")))
        if notes:
            output["redactionNotes"] = notes[:1000]
        return output

    def _validate_deep_invariants(self, document: Mapping[str, Any]) -> None:
        speaker_ids = list(document["speakerPolicy"]["speakerIds"])
        for segment in document["segments"]:
            revisions = segment["revisions"]
            if segment["normalizedText"] != segment["displayText"] and not self._has_text_trace(
                revisions,
                segment["normalizedText"],
                segment["displayText"],
            ):
                raise ReportAssemblyError(
                    f"{segment['id']}: display text changed without a revision"
                )
            scores = segment["evidence"]["speaker"]["scores"]
            if [item["speakerId"] for item in scores] != speaker_ids:
                raise ReportAssemblyError(
                    f"{segment['id']}: speaker scores must preserve canonical ordering"
                )
            if len({item["speakerId"] for item in scores}) != len(speaker_ids):
                raise ReportAssemblyError(
                    f"{segment['id']}: speaker scores must be unique"
                )
            if segment["reviewStatus"] == "accepted":
                semantic = segment["evidence"].get("semantic", {})
                if semantic.get("decision") in {"review-required", "reject"}:
                    raise ReportAssemblyError(
                        f"{segment['id']}: unresolved semantic evidence cannot be accepted"
                    )
