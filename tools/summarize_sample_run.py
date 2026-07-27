"""Build a content-free audit summary for one production sample-library run."""

from __future__ import annotations

import argparse
import json
import math
import stat
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _path_is_dataless(path: Path) -> bool:
    """Detect a macOS cloud placeholder without triggering a download."""

    try:
        flags = path.stat(follow_symlinks=False).st_flags
    except (AttributeError, OSError):
        return False
    return bool(flags & getattr(stat, "SF_DATALESS", 0))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"required artifact is missing: {path}") from exc
    except OSError as exc:
        raise ValueError(f"required artifact is unavailable: {path}") from exc
    if _path_is_dataless(path):
        raise ValueError(
            f"required artifact is a dataless cloud placeholder: {path}"
        )
    try:
        payload = path.read_bytes()
        if len(payload) != metadata.st_size:
            raise ValueError(
                f"required artifact byte count changed during read: {path}"
            )
        value = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"artifact is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"required artifact became unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field=field)


def _optional_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field=field)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _sum_present_numbers(
    values: Sequence[Mapping[str, Any]],
    key: str,
    *,
    digits: int | None = None,
) -> int | float | None:
    present = [value[key] for value in values if value.get(key) is not None]
    if not present:
        return None
    total = sum(present)
    return round(float(total), digits) if digits is not None else total


def _job_id(value: Mapping[str, Any], *, field: str) -> str:
    job_id = value.get("jobId", value.get("job_id"))
    if not isinstance(job_id, str) or not job_id:
        raise ValueError(f"{field} is missing jobId")
    return job_id


def _assert_job_id(
    value: Mapping[str, Any],
    *,
    expected: str,
    field: str,
) -> None:
    actual = _job_id(value, field=field)
    if actual != expected:
        raise ValueError(
            f"{field} jobId mismatch: expected {expected!r}, got {actual!r}"
        )


def _language_window_summary(
    transcript: Mapping[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    segments = _list(transcript.get("segments"), field=f"{case_id}.segments")
    languages: Counter[str] = Counter()
    speaker_ids: set[str] = set()
    durations: list[int] = []
    configured_limits: set[int] = set()
    speaker_splits: set[int] = set()
    language_splits: set[int] = set()
    applied_splits: set[int] = set()

    for index, raw_segment in enumerate(segments):
        segment = _mapping(
            raw_segment,
            field=f"{case_id}.segments[{index}]",
        )
        start_ms = _nonnegative_int(
            segment.get("startMs"),
            field=f"{case_id}.segments[{index}].startMs",
        )
        end_ms = _positive_int(
            segment.get("endMs"),
            field=f"{case_id}.segments[{index}].endMs",
        )
        if end_ms <= start_ms:
            raise ValueError(
                f"{case_id}.segments[{index}] has a non-positive duration"
            )
        durations.append(end_ms - start_ms)
        speaker_id = segment.get("speakerId")
        if not isinstance(speaker_id, str) or not speaker_id:
            raise ValueError(
                f"{case_id}.segments[{index}].speakerId must be non-empty"
            )
        speaker_ids.add(speaker_id)

        evidence = _mapping(
            segment.get("evidence"),
            field=f"{case_id}.segments[{index}].evidence",
        )
        asr = _mapping(
            evidence.get("asr"),
            field=f"{case_id}.segments[{index}].evidence.asr",
        )
        language = asr.get("language")
        languages[
            language if isinstance(language, str) and language else "und"
        ] += 1

        raw_refinement = evidence.get("speakerChangeRefinement")
        if raw_refinement is None:
            # Review-only overlap recovery segments inherit the source ASR
            # evidence but are not produced by speaker-change refinement.
            continue
        refinement = _mapping(
            raw_refinement,
            field=(
                f"{case_id}.segments[{index}]"
                ".evidence.speakerChangeRefinement"
            ),
        )
        configured_limits.add(
            _positive_int(
                refinement.get("maxLanguageWindowMs"),
                field=(
                    f"{case_id}.segments[{index}]"
                    ".evidence.speakerChangeRefinement.maxLanguageWindowMs"
                ),
            )
        )
        for field, target in (
            ("speakerChangeSplitsMs", speaker_splits),
            ("languageDurationSplitsMs", language_splits),
            ("appliedSplitsMs", applied_splits),
        ):
            for raw_split in _list(
                refinement.get(field),
                field=(
                    f"{case_id}.segments[{index}]"
                    f".evidence.speakerChangeRefinement.{field}"
                ),
            ):
                target.add(_positive_int(raw_split, field=f"{case_id}.{field}"))

    if not segments:
        raise ValueError(f"{case_id} transcribable transcript has no segments")
    if len(configured_limits) != 1:
        raise ValueError(
            f"{case_id} has inconsistent maxLanguageWindowMs values: "
            f"{sorted(configured_limits)}"
        )
    configured_limit = next(iter(configured_limits))
    maximum_duration = max(durations)
    if maximum_duration > configured_limit:
        raise ValueError(
            f"{case_id} language window exceeds configured limit: "
            f"{maximum_duration} > {configured_limit}"
        )
    return {
        "documentLanguage": transcript.get("language"),
        "segmentCount": len(segments),
        "speakerCount": len(speaker_ids),
        "segmentLanguageCounts": dict(sorted(languages.items())),
        "maxObservedLanguageWindowMs": maximum_duration,
        "maxConfiguredLanguageWindowMs": configured_limit,
        "languageWindowLimitPassed": True,
        "speakerChangeSplitCount": len(speaker_splits),
        "languageDurationSplitCount": len(language_splits),
        "appliedSplitCount": len(applied_splits),
    }


def _review_summary(
    review: Mapping[str, Any] | None,
    *,
    case_id: str,
) -> dict[str, Any]:
    if review is None:
        return {
            "openCount": 0,
            "reasonCounts": {},
            "speakerCountEstimate": None,
        }
    raw_items = _list(review.get("items"), field=f"{case_id}.review.items")
    open_items = [
        _mapping(item, field=f"{case_id}.review.items")
        for item in raw_items
        if isinstance(item, Mapping) and item.get("status") == "open"
    ]
    open_count = review.get("openCount")
    if (
        not isinstance(open_count, int)
        or isinstance(open_count, bool)
        or open_count < 0
        or open_count != len(open_items)
    ):
        raise ValueError(
            f"{case_id}.review.openCount does not match open review items"
        )
    reasons: Counter[str] = Counter()
    for item in open_items:
        reason = item.get("reasonCode")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{case_id} has an open review item without reasonCode")
        reasons[reason] += 1
    estimate = review.get("speakerCountEstimate")
    if estimate is not None:
        estimate = _mapping(
            estimate,
            field=f"{case_id}.review.speakerCountEstimate",
        )
        candidate_range = _mapping(
            estimate.get("candidateRange"),
            field=f"{case_id}.review.speakerCountEstimate.candidateRange",
        )
        estimate = {
            "estimatedCount": _positive_int(
                estimate.get("estimatedCount"),
                field=f"{case_id}.review.speakerCountEstimate.estimatedCount",
            ),
            "confidence": _finite_number(
                estimate.get("confidence"),
                field=f"{case_id}.review.speakerCountEstimate.confidence",
            ),
            "candidateRange": {
                "min": _positive_int(
                    candidate_range.get("min"),
                    field=(
                        f"{case_id}.review.speakerCountEstimate"
                        ".candidateRange.min"
                    ),
                ),
                "max": _positive_int(
                    candidate_range.get("max"),
                    field=(
                        f"{case_id}.review.speakerCountEstimate"
                        ".candidateRange.max"
                    ),
                ),
            },
            "method": estimate.get("method"),
        }
    return {
        "openCount": open_count,
        "reasonCounts": dict(sorted(reasons.items())),
        "speakerCountEstimate": estimate,
    }


def _metrics_summary(
    metrics: Mapping[str, Any] | None,
    *,
    case_id: str,
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    runtime = _mapping(metrics.get("runtime"), field=f"{case_id}.metrics.runtime")
    resources = _mapping(
        metrics.get("resources"),
        field=f"{case_id}.metrics.resources",
    )
    cache = _mapping(metrics.get("cache"), field=f"{case_id}.metrics.cache")
    routing = _mapping(metrics.get("routing"), field=f"{case_id}.metrics.routing")
    policy = _mapping(metrics.get("policy"), field=f"{case_id}.metrics.policy")
    return {
        "pipelineElapsedMs": _finite_number(
            runtime.get("elapsedMs"),
            field=f"{case_id}.metrics.runtime.elapsedMs",
        ),
        "pipelineRtf": _finite_number(
            runtime.get("rtf"),
            field=f"{case_id}.metrics.runtime.rtf",
        ),
        "peakRamMb": _optional_number(
            resources.get("peakRamMb"),
            field=f"{case_id}.metrics.resources.peakRamMb",
        ),
        "peakVramMb": _optional_number(
            resources.get("peakVramMb"),
            field=f"{case_id}.metrics.resources.peakVramMb",
        ),
        "cacheRequests": _positive_int(
            cache.get("requests"),
            field=f"{case_id}.metrics.cache.requests",
        ),
        "cacheHitRate": _finite_number(
            cache.get("hitRate"),
            field=f"{case_id}.metrics.cache.hitRate",
        ),
        "cacheRecomputationRate": _finite_number(
            cache.get("recomputationRate"),
            field=f"{case_id}.metrics.cache.recomputationRate",
        ),
        "escalationRate": _finite_number(
            routing.get("escalationRate"),
            field=f"{case_id}.metrics.routing.escalationRate",
        ),
        "resolvedSpeakerCount": _positive_int(
            policy.get("resolvedSpeakerCount"),
            field=f"{case_id}.metrics.policy.resolvedSpeakerCount",
        ),
        "speakerCountMode": policy.get("speakerCountMode"),
        "offline": metrics.get("offline"),
        "referenceEvaluationAvailable": bool(
            _mapping(
                metrics.get("referenceEvaluation"),
                field=f"{case_id}.metrics.referenceEvaluation",
            ).get("available")
        ),
    }


def _semantic_summary(
    semantic: Mapping[str, Any] | None,
    *,
    case_id: str,
) -> dict[str, Any] | None:
    if semantic is None:
        return None
    metrics = _mapping(
        semantic.get("metrics"),
        field=f"{case_id}.semantic.metrics",
    )
    provider = _mapping(
        semantic.get("provider"),
        field=f"{case_id}.semantic.provider",
    )

    def count(name: str) -> int:
        return _nonnegative_int(
            metrics.get(name),
            field=f"{case_id}.semantic.metrics.{name}",
        )

    def optional_count(name: str) -> int | None:
        return _optional_nonnegative_int(
            metrics.get(name),
            field=f"{case_id}.semantic.metrics.{name}",
        )

    def duration_seconds(name: str) -> float | None:
        value = optional_count(name)
        return round(value / 1_000_000_000, 9) if value is not None else None

    status = semantic.get("status")
    model = semantic.get("model")
    if not isinstance(status, str) or not status:
        raise ValueError(f"{case_id}.semantic.status must be non-empty text")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{case_id}.semantic.model must be non-empty text")
    return {
        "status": status,
        "model": model,
        "promptVersion": semantic.get("promptVersion"),
        "applicationPolicy": semantic.get("applicationPolicy"),
        "requiresHumanApproval": semantic.get("requiresHumanApproval"),
        "provider": {
            "id": provider.get("id"),
            "version": provider.get("version"),
            "networkPolicy": provider.get("networkPolicy"),
        },
        "segmentsEvaluated": count("segmentsEvaluated"),
        "providerCalls": count("providerCalls"),
        "gateProviderCalls": count("gateProviderCalls"),
        "proposalProviderCalls": count("proposalProviderCalls"),
        "acceptedResultCount": count("acceptedResultCount"),
        "abstentionCount": count("abstentionCount"),
        "suggestionCount": count("suggestionCount"),
        "speakerSuggestionCount": count("speakerSuggestionCount"),
        "textSuggestionCount": count("textSuggestionCount"),
        "rejectionCount": count("rejectionCount"),
        "failureCount": count("failureCount"),
        "unresolvedSegmentCount": count("unresolvedSegmentCount"),
        "autoAppliedCount": count("autoAppliedCount"),
        "providerMetrics": {
            "completedCalls": optional_count("providerCompletedCalls"),
            "totalDurationSeconds": duration_seconds(
                "providerTotalDurationNanoseconds"
            ),
            "loadDurationSeconds": duration_seconds(
                "providerLoadDurationNanoseconds"
            ),
            "promptEvalTokens": optional_count("providerPromptEvalTokens"),
            "promptEvalDurationSeconds": duration_seconds(
                "providerPromptEvalDurationNanoseconds"
            ),
            "outputTokens": optional_count("providerOutputTokens"),
            "outputEvalDurationSeconds": duration_seconds(
                "providerOutputEvalDurationNanoseconds"
            ),
        },
    }


def _case_summary(
    case: Mapping[str, Any],
    *,
    results_root: Path,
    outputs_root: Path,
) -> dict[str, Any]:
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("manifest case id must be a non-empty string")
    result = _read_object(results_root / f"{case_id}-result.json")
    output = outputs_root / case_id
    voice = _read_object(output / "voice-activity.v1.json")
    expected_job_id = _job_id(result, field=f"{case_id}.result")
    _assert_job_id(voice, expected=expected_job_id, field=f"{case_id}.voice")

    source_sha = case.get("sha256")
    if not isinstance(source_sha, str) or voice.get("sourceSha256") != source_sha:
        raise ValueError(f"{case_id} source SHA-256 does not match voice evidence")

    terminal_type = result.get("terminal_type")
    if not isinstance(terminal_type, str) or not terminal_type:
        raise ValueError(f"{case_id}.result is missing terminal_type")
    forced_cleanup = _list(
        result.get("forced_cleanup_pids"),
        field=f"{case_id}.result.forced_cleanup_pids",
    )
    classification = voice.get("classification")
    if not isinstance(classification, str) or not classification:
        raise ValueError(f"{case_id}.voice is missing classification")

    transcript_path = output / "transcript-document.v2.json"
    review_path = output / "review" / "review-queue.json"
    metrics_path = output / "pipeline-metrics.v1.json"
    semantic_path = output / "semantic" / "semantic-suggestions.v1.json"
    transcript = _read_object(transcript_path) if transcript_path.exists() else None
    review = _read_object(review_path) if review_path.exists() else None
    metrics = _read_object(metrics_path) if metrics_path.exists() else None
    semantic = _read_object(semantic_path) if semantic_path.exists() else None

    transcribable = classification == "transcribable-speech-detected"
    if transcribable and transcript is None:
        raise ValueError(f"{case_id} is transcribable but has no transcript")
    if not transcribable and transcript is not None:
        raise ValueError(f"{case_id} is no-speech but has a transcript")
    if not transcribable and semantic is not None:
        raise ValueError(f"{case_id} is no-speech but has semantic output")
    for field, artifact in (
        ("transcript", transcript),
        ("review", review),
        ("metrics", metrics),
        ("semantic", semantic),
    ):
        if artifact is not None:
            _assert_job_id(
                artifact,
                expected=expected_job_id,
                field=f"{case_id}.{field}",
            )

    selection = _mapping(
        case.get("windowSelection", case.get("selection")),
        field=f"{case_id}.selection",
    )
    truth = _mapping(
        case.get("truthEligibility"),
        field=f"{case_id}.truthEligibility",
    )
    transcription = (
        _language_window_summary(transcript, case_id=case_id)
        if transcript is not None
        else None
    )
    return {
        "caseId": case_id,
        "sourceId": case.get("sourceId"),
        "expectedLexicalSpeech": case.get("expectedLexicalSpeech"),
        "windowSelection": {
            "reason": selection.get("reason"),
            "startMs": selection.get(
                "startMs",
                (
                    round(float(selection["startSeconds"]) * 1000)
                    if isinstance(selection.get("startSeconds"), (int, float))
                    else None
                ),
            ),
            "endMs": selection.get(
                "endMs",
                (
                    round(
                        (
                            float(selection["startSeconds"])
                            + float(selection["durationSeconds"])
                        )
                        * 1000
                    )
                    if isinstance(selection.get("startSeconds"), (int, float))
                    and isinstance(
                        selection.get("durationSeconds"),
                        (int, float),
                    )
                    else None
                ),
            ),
            "durationMs": selection.get(
                "durationMs",
                (
                    round(float(selection["durationSeconds"]) * 1000)
                    if isinstance(
                        selection.get("durationSeconds"),
                        (int, float),
                    )
                    else None
                ),
            ),
            "audioActivityRatio": selection.get("audioActivityRatio"),
        },
        "truthEligibility": dict(truth),
        "runner": {
            "status": result.get("status"),
            "terminalType": terminal_type,
            "exitCode": result.get("exit_code"),
            "elapsedSeconds": _finite_number(
                result.get("elapsed_seconds"),
                field=f"{case_id}.result.elapsed_seconds",
            ),
            "shutdownAcknowledged": result.get("shutdown_acknowledged"),
            "forcedCleanupProcessCount": len(forced_cleanup),
        },
        "voiceActivity": {
            "classification": classification,
            "mediaDurationMs": voice.get("mediaDurationMs"),
            "speechWindowCount": voice.get("speechWindowCount"),
            "speechDurationMs": voice.get("speechDurationMs"),
            "speechRatio": voice.get("speechRatio"),
        },
        "transcription": transcription,
        "review": _review_summary(review, case_id=case_id),
        "metrics": _metrics_summary(metrics, case_id=case_id),
        "semantic": _semantic_summary(semantic, case_id=case_id),
    }


def _required_stratum_is_covered(
    required: str,
    observed_reasons: set[str],
) -> bool:
    return any(
        reason == required or reason.startswith(f"{required}-")
        for reason in observed_reasons
    )


def _source_summaries(
    manifest: Mapping[str, Any],
    *,
    cases: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_sources = manifest.get("sources", [])
    sources = _list(raw_sources, field="manifest.sources")
    source_records: dict[str, Mapping[str, Any]] = {}
    for index, raw_source in enumerate(sources):
        source = _mapping(raw_source, field=f"manifest.sources[{index}]")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"manifest.sources[{index}].id must be non-empty")
        if source_id in source_records:
            raise ValueError(f"duplicate manifest source id: {source_id}")
        source_records[source_id] = source

    case_records: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("manifest case id must be a non-empty string")
        case_records[case_id] = case

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        source_id = summary.get("sourceId")
        if not isinstance(source_id, str) or not source_id:
            continue
        grouped.setdefault(source_id, []).append(summary)

    selection_policy = manifest.get("selectionPolicy", {})
    if not isinstance(selection_policy, Mapping):
        raise ValueError("manifest.selectionPolicy must be an object")
    required_strata = selection_policy.get("requiredStrata", [])
    if not isinstance(required_strata, list) or any(
        not isinstance(value, str) or not value for value in required_strata
    ):
        raise ValueError("manifest.selectionPolicy.requiredStrata must be text")

    results: list[dict[str, Any]] = []
    for source_id in sorted(grouped):
        rows = grouped[source_id]
        source = source_records.get(source_id, {})
        raw_analysis = source.get("analysis")
        analysis = (
            _mapping(raw_analysis, field=f"manifest.sources[{source_id}].analysis")
            if raw_analysis is not None
            else None
        )
        duration_ms = source.get("durationMs")
        frame_count = analysis.get("frameCount") if analysis is not None else None
        frame_duration_ms = (
            analysis.get("frameDurationMs") if analysis is not None else None
        )
        analyzed_duration_ms: int | None = None
        acoustic_coverage_ratio: float | None = None
        if analysis is not None:
            frame_count = _positive_int(
                frame_count,
                field=f"manifest.sources[{source_id}].analysis.frameCount",
            )
            frame_duration_ms = _positive_int(
                frame_duration_ms,
                field=f"manifest.sources[{source_id}].analysis.frameDurationMs",
            )
            duration_ms = _positive_int(
                duration_ms,
                field=f"manifest.sources[{source_id}].durationMs",
            )
            analyzed_duration_ms = min(duration_ms, frame_count * frame_duration_ms)
            acoustic_coverage_ratio = round(analyzed_duration_ms / duration_ms, 9)

        observed_reasons = {
            str(row["windowSelection"]["reason"])
            for row in rows
            if row["windowSelection"].get("reason")
        }
        strata_coverage = {
            stratum: _required_stratum_is_covered(stratum, observed_reasons)
            for stratum in required_strata
        }
        speaker_counts = [
            int(row["transcription"]["speakerCount"])
            for row in rows
            if row.get("transcription") is not None
        ]
        speaker_distribution = Counter(str(value) for value in speaker_counts)
        modal_frequency = max(speaker_distribution.values(), default=0)
        modal_counts = sorted(
            int(value)
            for value, frequency in speaker_distribution.items()
            if frequency == modal_frequency
        )
        segment_counts = [
            int(row["transcription"]["segmentCount"])
            for row in rows
            if row.get("transcription") is not None
        ]
        split_counts = [
            int(row["transcription"]["appliedSplitCount"])
            for row in rows
            if row.get("transcription") is not None
        ]
        review_counts = [int(row["review"]["openCount"]) for row in rows]
        metrics = [row["metrics"] for row in rows if row.get("metrics") is not None]
        semantic_rows = [
            row["semantic"] for row in rows if row.get("semantic") is not None
        ]
        semantic_statuses = Counter(row["status"] for row in semantic_rows)
        semantic_models = Counter(row["model"] for row in semantic_rows)
        semantic_provider_ids = Counter(
            str(row["provider"]["id"]) for row in semantic_rows
        )
        semantic_provider_metrics = [
            row["providerMetrics"] for row in semantic_rows
        ]
        rtfs = [float(value["pipelineRtf"]) for value in metrics]
        peak_ram = [
            float(value["peakRamMb"])
            for value in metrics
            if value.get("peakRamMb") is not None
        ]
        peak_vram = [
            float(value["peakVramMb"])
            for value in metrics
            if value.get("peakVramMb") is not None
        ]
        truth_counts: Counter[str] = Counter()
        for row in rows:
            for metric, eligible in row["truthEligibility"].items():
                if eligible is True:
                    truth_counts[metric] += 1
        manual_five_eligible_cases = 0
        for row in rows:
            case = case_records[str(row["caseId"])]
            selection = case.get("windowSelection", case.get("selection", {}))
            speaker_set = (
                selection.get("speakerSet")
                if isinstance(selection, Mapping)
                else None
            )
            if (
                row["truthEligibility"].get("speakerCount") is True
                and isinstance(speaker_set, list)
                and all(
                    isinstance(speaker_id, str) and speaker_id
                    for speaker_id in speaker_set
                )
                and len(set(speaker_set)) == 5
            ):
                manual_five_eligible_cases += 1

        expected_window_count = source.get("windowCount")
        window_set_complete = (
            isinstance(expected_window_count, int)
            and not isinstance(expected_window_count, bool)
            and expected_window_count == len(rows)
        )
        speaker_count_stable = bool(speaker_counts) and len(set(speaker_counts)) == 1

        all_technical = all(
            row["runner"]["status"] == "observed"
            and row["runner"]["exitCode"] == 0
            and row["runner"]["shutdownAcknowledged"] is True
            and row["runner"]["forcedCleanupProcessCount"] == 0
            for row in rows
        )
        all_review_required = all(
            row["runner"]["terminalType"] == "review.required" for row in rows
        )
        reference_scored = any(
            row.get("metrics") is not None
            and row["metrics"]["referenceEvaluationAvailable"]
            for row in rows
        )
        source_sha = source.get("sha256")
        results.append(
            {
                "sourceId": source_id,
                "sourceSha256": (
                    source_sha if isinstance(source_sha, str) else None
                ),
                "durationMs": duration_ms,
                "fullTimelineAcousticScan": {
                    "available": analysis is not None,
                    "algorithm": analysis.get("algorithm") if analysis else None,
                    "frameCount": frame_count,
                    "frameDurationMs": frame_duration_ms,
                    "analyzedDurationMs": analyzed_duration_ms,
                    "coverageRatio": acoustic_coverage_ratio,
                    "activeFrameRatio": (
                        analysis.get("activeFrameRatio") if analysis else None
                    ),
                    "activityThresholdDb": (
                        analysis.get("activityThresholdDb") if analysis else None
                    ),
                    "rmsDbPercentiles": (
                        analysis.get("rmsDbPercentiles") if analysis else None
                    ),
                    "isSpeechClassification": False,
                },
                "stratifiedWindows": {
                    "expectedCount": expected_window_count,
                    "observedCount": len(rows),
                    "windowSetComplete": window_set_complete,
                    "coverageRatio": source.get("windowCoverageRatio"),
                    "observedReasons": sorted(observed_reasons),
                    "requiredStrataCovered": strata_coverage,
                    "selectionUsesModelScores": selection_policy.get(
                        "modelScoresUsed"
                    ),
                },
                "speakerCountStability": {
                    "observedWindowCount": len(speaker_counts),
                    "distribution": dict(
                        sorted(
                            speaker_distribution.items(),
                            key=lambda item: int(item[0]),
                        )
                    ),
                    "minimum": min(speaker_counts) if speaker_counts else None,
                    "maximum": max(speaker_counts) if speaker_counts else None,
                    "modalCounts": modal_counts,
                    "modalAgreementRate": (
                        round(modal_frequency / len(speaker_counts), 9)
                        if speaker_counts
                        else None
                    ),
                    "stableAcrossWindows": speaker_count_stable,
                },
                "turnGranularity": {
                    "segmentCountTotal": sum(segment_counts),
                    "segmentCountMin": min(segment_counts) if segment_counts else None,
                    "segmentCountMax": max(segment_counts) if segment_counts else None,
                    "appliedSplitCountTotal": sum(split_counts),
                },
                "review": {
                    "openCountTotal": sum(review_counts),
                    "openCountMin": min(review_counts),
                    "openCountMax": max(review_counts),
                    "allWindowsRequireReview": all_review_required,
                },
                "semantic": {
                    "evidenceWindowCount": len(semantic_rows),
                    "statusCounts": dict(sorted(semantic_statuses.items())),
                    "modelCounts": dict(sorted(semantic_models.items())),
                    "providerIdCounts": dict(sorted(semantic_provider_ids.items())),
                    "segmentsEvaluatedTotal": sum(
                        row["segmentsEvaluated"] for row in semantic_rows
                    ),
                    "providerCallsTotal": sum(
                        row["providerCalls"] for row in semantic_rows
                    ),
                    "acceptedResultCount": sum(
                        row["acceptedResultCount"] for row in semantic_rows
                    ),
                    "abstentionCount": sum(
                        row["abstentionCount"] for row in semantic_rows
                    ),
                    "suggestionCount": sum(
                        row["suggestionCount"] for row in semantic_rows
                    ),
                    "rejectionCount": sum(
                        row["rejectionCount"] for row in semantic_rows
                    ),
                    "failureCount": sum(
                        row["failureCount"] for row in semantic_rows
                    ),
                    "autoAppliedCount": sum(
                        row["autoAppliedCount"] for row in semantic_rows
                    ),
                    "providerTotalDurationSeconds": _sum_present_numbers(
                        semantic_provider_metrics,
                        "totalDurationSeconds",
                        digits=9,
                    ),
                    "providerLoadDurationSeconds": _sum_present_numbers(
                        semantic_provider_metrics,
                        "loadDurationSeconds",
                        digits=9,
                    ),
                    "providerPromptEvalDurationSeconds": _sum_present_numbers(
                        semantic_provider_metrics,
                        "promptEvalDurationSeconds",
                        digits=9,
                    ),
                    "providerOutputEvalDurationSeconds": _sum_present_numbers(
                        semantic_provider_metrics,
                        "outputEvalDurationSeconds",
                        digits=9,
                    ),
                    "providerPromptEvalTokens": _sum_present_numbers(
                        semantic_provider_metrics,
                        "promptEvalTokens",
                    ),
                    "providerOutputTokens": _sum_present_numbers(
                        semantic_provider_metrics,
                        "outputTokens",
                    ),
                },
                "performance": {
                    "pipelineRtfMin": min(rtfs) if rtfs else None,
                    "pipelineRtfMedian": (
                        round(statistics.median(rtfs), 9) if rtfs else None
                    ),
                    "pipelineRtfMax": max(rtfs) if rtfs else None,
                    "peakRamMbMax": max(peak_ram) if peak_ram else None,
                    "peakVramMbMax": max(peak_vram) if peak_vram else None,
                },
                "truthEligibility": {
                    "eligibleWindowCounts": dict(sorted(truth_counts.items())),
                    "referenceQualityScored": reference_scored,
                    "manualFiveEligibleWindowCount": manual_five_eligible_cases,
                },
                "terminal": {
                    "windowTechnicalExecutionPassed": all_technical,
                    "mandatorySemanticEvidenceComplete": (
                        len(semantic_rows) == len(speaker_counts)
                        and all(
                            row["status"] == "completed"
                            and row["autoAppliedCount"] == 0
                            for row in semantic_rows
                        )
                    ),
                    "fullTimelineAcousticScanPassed": (
                        acoustic_coverage_ratio is not None
                        and acoustic_coverage_ratio >= 0.999
                    ),
                    "requiredStrataCovered": (
                        all(strata_coverage.values()) if strata_coverage else None
                    ),
                    "windowSetComplete": window_set_complete,
                    "completeSourceProductionRunObserved": False,
                    "speakerCountStableAcrossWindows": speaker_count_stable,
                    "referenceQualityScored": reference_scored,
                    "manualFiveQualityEligible": manual_five_eligible_cases > 0,
                    "releaseApproved": False,
                    "qualityConclusion": (
                        "quality_scored_review_required"
                        if reference_scored
                        else "not_scored_missing_reference_truth"
                    ),
                    "disposition": "review.required",
                },
            }
        )
    return results


def summarize_run(
    *,
    manifest_path: Path,
    results_root: Path,
    outputs_root: Path,
) -> dict[str, Any]:
    manifest = _read_object(manifest_path)
    cases = _list(manifest.get("cases"), field="manifest.cases")
    if not cases:
        raise ValueError("manifest.cases must not be empty")
    summaries: list[dict[str, Any]] = []
    case_records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw_case in cases:
        case = _mapping(raw_case, field="manifest.cases")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("manifest case id must be a non-empty string")
        if case_id in seen:
            raise ValueError(f"duplicate manifest case id: {case_id}")
        seen.add(case_id)
        case_records.append(case)
        summaries.append(
            _case_summary(
                case,
                results_root=results_root,
                outputs_root=outputs_root,
            )
        )

    terminal_types = Counter(row["runner"]["terminalType"] for row in summaries)
    classifications = Counter(
        row["voiceActivity"]["classification"] for row in summaries
    )
    review_counts = [row["review"]["openCount"] for row in summaries]
    reason_counts: Counter[str] = Counter()
    document_languages: Counter[str] = Counter()
    segment_languages: Counter[str] = Counter()
    estimated_speakers: Counter[str] = Counter()
    runner_elapsed = 0.0
    pipeline_rtfs: list[float] = []
    cache_requests = 0
    cache_hits = 0.0
    max_language_window = 0
    semantic_rows: list[Mapping[str, Any]] = []
    for row in summaries:
        runner_elapsed += row["runner"]["elapsedSeconds"]
        reason_counts.update(row["review"]["reasonCounts"])
        estimate = row["review"]["speakerCountEstimate"]
        if estimate is not None:
            estimated_speakers[str(estimate["estimatedCount"])] += 1
        transcription = row["transcription"]
        if transcription is not None:
            language = transcription.get("documentLanguage")
            document_languages[
                language if isinstance(language, str) and language else "und"
            ] += 1
            segment_languages.update(transcription["segmentLanguageCounts"])
            max_language_window = max(
                max_language_window,
                transcription["maxObservedLanguageWindowMs"],
            )
        metrics = row["metrics"]
        if metrics is not None:
            pipeline_rtfs.append(metrics["pipelineRtf"])
            cache_requests += metrics["cacheRequests"]
            cache_hits += metrics["cacheRequests"] * metrics["cacheHitRate"]
        semantic = row["semantic"]
        if semantic is not None:
            semantic_rows.append(semantic)

    truth_eligible_counts: Counter[str] = Counter()
    for row in summaries:
        for metric, eligible in row["truthEligibility"].items():
            if eligible is True:
                truth_eligible_counts[metric] += 1
    lexical_negative_rows = [
        row
        for row in summaries
        if row["expectedLexicalSpeech"] is False
    ]
    lexical_false_positives = sum(
        row["voiceActivity"]["classification"]
        == "transcribable-speech-detected"
        for row in lexical_negative_rows
    )

    all_observed = all(row["runner"]["status"] == "observed" for row in summaries)
    all_zero_exit = all(row["runner"]["exitCode"] == 0 for row in summaries)
    all_shutdown = all(
        row["runner"]["shutdownAcknowledged"] is True for row in summaries
    )
    no_forced_cleanup = all(
        row["runner"]["forcedCleanupProcessCount"] == 0 for row in summaries
    )
    all_review_required = (
        terminal_types == Counter({"review.required": len(summaries)})
    )
    truth_available = any(truth_eligible_counts.values())
    quality_scored = any(
        row["metrics"] is not None
        and row["metrics"]["referenceEvaluationAvailable"]
        for row in summaries
    )
    transcribable_rows = [
        row for row in summaries if row["transcription"] is not None
    ]
    semantic_statuses = Counter(row["status"] for row in semantic_rows)
    semantic_models = Counter(row["model"] for row in semantic_rows)
    semantic_provider_metrics = [
        row["providerMetrics"] for row in semantic_rows
    ]
    mandatory_semantic_complete = (
        all(
            row["semantic"] is not None
            and row["semantic"]["status"] == "completed"
            for row in transcribable_rows
        )
        if transcribable_rows
        else None
    )
    semantic_suggestion_only = (
        all(
            row["applicationPolicy"] == "suggestion-only"
            and row["requiresHumanApproval"] is True
            and row["autoAppliedCount"] == 0
            for row in semantic_rows
        )
        if semantic_rows
        else None
    )
    source_summaries = _source_summaries(
        manifest,
        cases=case_records,
        summaries=summaries,
    )
    return {
        "schemaVersion": "1.2.0",
        "artifactType": "sample-run-audit-summary",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "libraryId": manifest.get("libraryId"),
        "runId": results_root.name,
        "privacy": {
            "containsTranscriptText": False,
            "containsSourceMediaPaths": False,
        },
        "aggregate": {
            "expectedCases": len(cases),
            "observedCases": len(summaries),
            "failedCases": sum(
                1
                for row in summaries
                if row["runner"]["status"] != "observed"
                or row["runner"]["exitCode"] != 0
            ),
            "terminalTypeCounts": dict(sorted(terminal_types.items())),
            "voiceClassificationCounts": dict(sorted(classifications.items())),
            "runnerElapsedSeconds": round(runner_elapsed, 6),
            "allShutdownAcknowledged": all_shutdown,
            "forcedCleanupProcessCount": sum(
                row["runner"]["forcedCleanupProcessCount"] for row in summaries
            ),
            "reviewOpenCount": sum(review_counts),
            "reviewOpenAverage": round(sum(review_counts) / len(review_counts), 6),
            "reviewOpenMin": min(review_counts),
            "reviewOpenMax": max(review_counts),
            "reviewReasonCounts": dict(sorted(reason_counts.items())),
            "documentLanguageCounts": dict(sorted(document_languages.items())),
            "segmentLanguageCounts": dict(sorted(segment_languages.items())),
            "estimatedSpeakerCountDistribution": dict(
                sorted(estimated_speakers.items(), key=lambda item: int(item[0]))
            ),
            "maxObservedLanguageWindowMs": max_language_window or None,
            "pipelineRtfMin": min(pipeline_rtfs) if pipeline_rtfs else None,
            "pipelineRtfMax": max(pipeline_rtfs) if pipeline_rtfs else None,
            "weightedCacheHitRate": (
                round(cache_hits / cache_requests, 9) if cache_requests else None
            ),
            "semantic": {
                "evidenceCaseCount": len(semantic_rows),
                "statusCounts": dict(sorted(semantic_statuses.items())),
                "modelCounts": dict(sorted(semantic_models.items())),
                "segmentsEvaluatedTotal": sum(
                    row["segmentsEvaluated"] for row in semantic_rows
                ),
                "providerCallsTotal": sum(
                    row["providerCalls"] for row in semantic_rows
                ),
                "acceptedResultCount": sum(
                    row["acceptedResultCount"] for row in semantic_rows
                ),
                "abstentionCount": sum(
                    row["abstentionCount"] for row in semantic_rows
                ),
                "suggestionCount": sum(
                    row["suggestionCount"] for row in semantic_rows
                ),
                "speakerSuggestionCount": sum(
                    row["speakerSuggestionCount"] for row in semantic_rows
                ),
                "textSuggestionCount": sum(
                    row["textSuggestionCount"] for row in semantic_rows
                ),
                "rejectionCount": sum(
                    row["rejectionCount"] for row in semantic_rows
                ),
                "failureCount": sum(
                    row["failureCount"] for row in semantic_rows
                ),
                "unresolvedSegmentCount": sum(
                    row["unresolvedSegmentCount"] for row in semantic_rows
                ),
                "autoAppliedCount": sum(
                    row["autoAppliedCount"] for row in semantic_rows
                ),
                "providerTotalDurationSeconds": _sum_present_numbers(
                    semantic_provider_metrics,
                    "totalDurationSeconds",
                    digits=9,
                ),
                "providerLoadDurationSeconds": _sum_present_numbers(
                    semantic_provider_metrics,
                    "loadDurationSeconds",
                    digits=9,
                ),
                "providerPromptEvalDurationSeconds": _sum_present_numbers(
                    semantic_provider_metrics,
                    "promptEvalDurationSeconds",
                    digits=9,
                ),
                "providerOutputEvalDurationSeconds": _sum_present_numbers(
                    semantic_provider_metrics,
                    "outputEvalDurationSeconds",
                    digits=9,
                ),
                "providerPromptEvalTokens": _sum_present_numbers(
                    semantic_provider_metrics,
                    "promptEvalTokens",
                ),
                "providerOutputTokens": _sum_present_numbers(
                    semantic_provider_metrics,
                    "outputTokens",
                ),
            },
            "truthEligibleCaseCounts": dict(sorted(truth_eligible_counts.items())),
            "lexicalSpeechNegativeCases": len(lexical_negative_rows),
            "lexicalSpeechFalsePositiveCount": lexical_false_positives,
            "lexicalSpeechFalsePositiveRate": (
                lexical_false_positives / len(lexical_negative_rows)
                if lexical_negative_rows
                else None
            ),
        },
        "gates": {
            "technicalExecutionPassed": (
                all_observed
                and all_zero_exit
                and all_shutdown
                and no_forced_cleanup
            ),
            "allCasesRequireReview": all_review_required,
            "languageWindowLimitPassed": all(
                row["transcription"] is None
                or row["transcription"]["languageWindowLimitPassed"]
                for row in summaries
            ),
            "referenceQualityScored": quality_scored,
            "mandatorySemanticCompleted": mandatory_semantic_complete,
            "semanticSuggestionOnlyPolicyPassed": semantic_suggestion_only,
            "lexicalSpeechNegativeGatePassed": (
                lexical_false_positives == 0
                if lexical_negative_rows
                else None
            ),
            "qualityConclusion": (
                (
                    "lexical_speech_negative_gate_passed"
                    if lexical_false_positives == 0
                    else "lexical_speech_negative_gate_failed"
                )
                if lexical_negative_rows
                else (
                    "quality_scored"
                    if quality_scored
                    else (
                        "not_scored_despite_reference_truth"
                        if truth_available
                        else "not_scored_missing_reference_truth"
                    )
                )
            ),
        },
        "sourceSummaries": source_summaries,
        "cases": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_run(
        manifest_path=args.manifest.resolve(),
        results_root=args.results_root.resolve(),
        outputs_root=args.outputs_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "aggregate": summary["aggregate"],
                "gates": summary["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
