"""Compile canonical output customization into executable local output plans.

This module is the narrow bridge between the public presentation contract and
the existing PDF, subtitle, and source-immutable delivery domains.  It does
not execute FFmpeg itself.  Every derived path is constrained to the job
output directory, existing outputs are rejected, and burn-in is authorized
only by hash-bound trusted SDR probe evidence.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .errors import WorkerError
from .language import normalize_language_tag
from .media_probe import (
    MediaProbeEvidenceError,
    MediaProbeResult,
    canonical_local_media_file,
    validated_media_probe_payload,
)
from .output_customization import (
    OutputCustomization,
    OutputCustomizationError,
    resolve_output_customization,
)
from .persistence import (
    PublishedJsonEvidence,
    atomic_publish_json_evidence,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from .subtitle_delivery import (
    BurnInVideoStrategy,
    SubtitleDeliveryError,
)
from .subtitles import (
    CuePolicy,
    SubtitleArrangement,
    SubtitleFormat,
    SubtitleOutputMode,
    SubtitleOutputPlan,
    SubtitleStyle,
    SubtitleTheme,
    arrange_cues_within_source_duration,
    build_subtitle_output_plan,
    export_subtitles,
    resolve_speaker_colors,
)


_INVALID_WINDOWS_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALIGNMENT = {
    "top-left": 7,
    "top-center": 8,
    "top-right": 9,
    "middle-left": 4,
    "middle-center": 5,
    "middle-right": 6,
    "bottom-left": 1,
    "bottom-center": 2,
    "bottom-right": 3,
}
_SUBTITLE_SUFFIX = {
    SubtitleFormat.SRT: ".srt",
    SubtitleFormat.WEBVTT: ".vtt",
    SubtitleFormat.ASS: ".ass",
}
_KNOWN_SDR_TRANSFERS = frozenset(
    {
        "bt709",
        "gamma22",
        "gamma28",
        "smpte170m",
        "smpte240m",
        "linear",
        "log",
        "log_sqrt",
        "iec61966-2-4",
        "bt1361e",
        "iec61966-2-1",
        "bt2020-10",
        "bt2020-12",
        "smpte428",
    }
)


@dataclass(frozen=True)
class MediaProbeArtifact:
    """Immutable, exact-byte evidence for ``media-probe.v1.json``."""

    path: Path
    size_bytes: int
    sha256: str
    source_path: Path
    source_size_bytes: int
    source_sha256: str
    probe_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sizeBytes": self.size_bytes,
            "sha256": self.sha256,
            "source": {
                "path": str(self.source_path),
                "sizeBytes": self.source_size_bytes,
                "sha256": self.source_sha256,
            },
            "probeFingerprintSha256": self.probe_fingerprint_sha256,
        }


@dataclass(frozen=True)
class OutputExecutionPlan:
    """Fully resolved, source-preserving output intent for one job."""

    customization_sha256: str
    source_path: Path
    output_directory: Path
    media_probe_artifact: MediaProbeArtifact
    report_enabled: bool
    report_config: dict[str, Any]
    exports_config: dict[str, Any]
    safety_config: dict[str, Any]
    reversibility_config: dict[str, Any]
    subtitle_enabled: bool
    subtitle_config: dict[str, Any]
    subtitle_formats: tuple[SubtitleFormat, ...]
    subtitle_style: SubtitleStyle | None
    cue_policy: CuePolicy | None
    subtitle_theme: str | None
    speaker_color_mode: str | None
    speaker_color_seed: str | None
    speaker_color_overrides: dict[str, str]
    sidecar_paths: tuple[tuple[SubtitleFormat, Path], ...]
    delivery_mode: SubtitleOutputMode
    delivery_output_path: Path | None
    subtitle_codec: str | None
    burn_in_strategy: str | None
    visual_qa_required: bool
    presentation_limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0.0",
            "customizationSha256": self.customization_sha256,
            "source": {
                "path": str(self.source_path),
                "outputDirectory": str(self.output_directory),
            },
            "mediaProbeArtifact": self.media_probe_artifact.to_dict(),
            "report": {
                "enabled": self.report_enabled,
                "config": dict(self.report_config),
            },
            "subtitle": {
                "enabled": self.subtitle_enabled,
                "formats": [value.value for value in self.subtitle_formats],
                "theme": self.subtitle_theme,
                "style": (
                    self.subtitle_style.as_dict()
                    if self.subtitle_style is not None
                    else None
                ),
                "cuePolicy": (
                    {
                        "maxCharactersPerLine": (
                            self.cue_policy.max_characters_per_line
                        ),
                        "maxLines": self.cue_policy.max_lines,
                        "maxReadingSpeed": self.cue_policy.max_reading_speed,
                        "minCueMs": self.cue_policy.min_cue_ms,
                        "maxCueMs": self.cue_policy.max_cue_ms,
                        "gapMs": self.cue_policy.gap_ms,
                        "includeSpeakerLabels": (
                            self.cue_policy.include_speaker_labels
                        ),
                        "speakerLabelTemplate": (
                            self.cue_policy.speaker_label_template
                        ),
                    }
                    if self.cue_policy is not None
                    else None
                ),
                "speakerColors": {
                    "mode": self.speaker_color_mode,
                    "seed": self.speaker_color_seed,
                    "overrides": [
                        {"speakerId": speaker_id, "color": color}
                        for speaker_id, color in sorted(
                            self.speaker_color_overrides.items()
                        )
                    ],
                },
                "sidecars": [
                    {"format": value.value, "path": str(path)}
                    for value, path in self.sidecar_paths
                ],
                "config": dict(self.subtitle_config),
            },
            "delivery": {
                "mode": self.delivery_mode.value,
                "outputPath": (
                    str(self.delivery_output_path)
                    if self.delivery_output_path is not None
                    else None
                ),
                "subtitleCodec": self.subtitle_codec,
                "burnInStrategy": self.burn_in_strategy,
                "visualQaRequired": self.visual_qa_required,
            },
            "exports": dict(self.exports_config),
            "safety": dict(self.safety_config),
            "reversibility": dict(self.reversibility_config),
            "presentationLimitations": list(self.presentation_limitations),
            "sourceProtection": {
                "sourceMediaImmutable": True,
                "overwriteExistingOutputs": False,
                "derivedArtifactsOnly": True,
            },
        }

    def deterministic_hash(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def report_renderer_config(
        self,
        *,
        speaker_policy: Mapping[str, Any],
        renderer_id: str,
    ) -> dict[str, Any]:
        """Return exact report/font/DIY provenance for the Java PDF adapter."""

        if not self.report_enabled:
            raise WorkerError(
                "REPORT_OUTPUT_DISABLED",
                "the output execution plan does not enable PDF reporting",
            )
        return {
            "speakerPolicy": dict(speaker_policy),
            "offline": True,
            "renderer": renderer_id,
            "outputExecutionPlanSha256": self.deterministic_hash(),
            "mediaProbeArtifact": self.media_probe_artifact.to_dict(),
            "presentation": {
                "customizationSha256": self.customization_sha256,
                "report": dict(self.report_config),
                "exports": dict(self.exports_config),
                "safety": dict(self.safety_config),
                "reversibility": dict(self.reversibility_config),
            },
        }


@dataclass(frozen=True)
class PreparedSubtitleArtifact:
    subtitle_format: SubtitleFormat
    output_path: Path
    payload: str
    delivery_plan: SubtitleOutputPlan


@dataclass(frozen=True)
class PreparedSubtitleOutputs:
    arrangement: SubtitleArrangement
    sidecars: tuple[PreparedSubtitleArtifact, ...]
    media_delivery_plan: SubtitleOutputPlan | None
    visual_qa_required: bool
    timing_policy_evidence: dict[str, Any]

    @property
    def visual_qa_speakers(self) -> tuple[dict[str, str], ...]:
        """Exact speaker colors rendered by ASS, or empty when disabled."""

        return self.arrangement.visual_qa_speakers()


def persist_media_probe_artifact(
    media_probe: MediaProbeResult,
    *,
    source_path: Path,
    output_directory: Path,
) -> MediaProbeArtifact:
    """Validate and immutably publish hash-bound admission evidence."""

    source = canonical_local_media_file(source_path)
    output_root = _canonical_output_directory(output_directory)
    try:
        payload = validated_media_probe_payload(media_probe)
    except MediaProbeEvidenceError as exc:
        raise WorkerError(
            "MEDIA_PROBE_EVIDENCE_INVALID",
            "media admission evidence failed its trust boundary",
            details={"reason": str(exc)},
        ) from exc
    _require_probe_matches_source(media_probe, source)
    source_before = source.stat()
    if (
        not media_probe.decode_smoke_tested
        or not media_probe.decode_smoke_test_passed
        or media_probe.ffmpeg is None
    ):
        raise WorkerError(
            "MEDIA_PROBE_DECODE_EVIDENCE_REQUIRED",
            "production admission requires a successful FFmpeg decode smoke test",
        )

    artifact_path = output_root / "media-probe.v1.json"
    try:
        published = atomic_publish_json_evidence(artifact_path, payload)
    except FileExistsError as exc:
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_EXISTS",
            "media-probe.v1.json already exists and will not be replaced",
            details={"path": str(artifact_path)},
        ) from exc
    source_after = source.stat()
    if (
        source_before.st_size,
        source_before.st_mtime_ns,
        getattr(source_before, "st_dev", 0),
        getattr(source_before, "st_ino", 0),
    ) != (
        source_after.st_size,
        source_after.st_mtime_ns,
        getattr(source_after, "st_dev", 0),
        getattr(source_after, "st_ino", 0),
    ):
        raise WorkerError(
            "SOURCE_MEDIA_CHANGED",
            "source media changed while media-probe evidence was published",
            details={"path": str(source)},
        )
    return _media_probe_artifact(
        published,
        source=source,
        media_probe=media_probe,
    )


def legacy_output_customization(render_pdf: bool) -> OutputCustomization:
    """Materialize pre-customization behavior as a canonical snapshot."""

    return resolve_output_customization(
        {
            "report": {"enabled": render_pdf},
            "subtitle": {"enabled": False},
            "exports": {
                "reportFormats": ["pdf"] if render_pdf else [],
                "subtitleAlternates": [],
            },
        }
    )


def resolve_request_output_customization(
    raw: Mapping[str, Any],
    *,
    render_pdf: bool,
    render_pdf_supplied: bool,
) -> tuple[OutputCustomization, bool]:
    """Resolve a request override and preserve explicit legacy conflicts."""

    try:
        customization = resolve_output_customization(raw)
    except OutputCustomizationError as exc:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_INVALID",
            "outputCustomization violates the canonical contract",
            details={"reason": str(exc)},
        ) from exc
    payload = customization.canonical_dict()
    report = payload["report"]
    report_formats = payload["exports"]["reportFormats"]
    unsupported = [value for value in report_formats if value != "pdf"]
    if unsupported:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_UNSUPPORTED_EXPORT",
            "the current production renderer supports PDF report output only",
            details={"reportFormats": unsupported},
        )
    resolved_render_pdf = bool(report["enabled"] and "pdf" in report_formats)
    if render_pdf_supplied and render_pdf != resolved_render_pdf:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_CONFLICT",
            "renderPdf conflicts with outputCustomization report settings",
            details={
                "renderPdf": render_pdf,
                "resolvedRenderPdf": resolved_render_pdf,
            },
        )
    if payload["subtitle"]["karaoke"]["mode"] != "off":
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_KARAOKE_UNSUPPORTED",
            "word-progress karaoke generation is not wired to verified word timings",
        )
    return customization, resolved_render_pdf


def compile_output_execution_plan(
    customization: OutputCustomization,
    *,
    source_path: Path,
    output_directory: Path,
    media_probe: MediaProbeResult,
    media_probe_artifact: MediaProbeArtifact,
    language: str,
    speaker_count: int,
    generated_date: str,
) -> OutputExecutionPlan:
    """Compile canonical settings after the trusted media probe has passed."""

    if not isinstance(customization, OutputCustomization):
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_INVALID",
            "job request is missing a canonical output customization snapshot",
        )
    source = canonical_local_media_file(source_path)
    output_root = _canonical_output_directory(output_directory)
    _validated_speaker_count(speaker_count)
    canonical_language = _validated_language(language)
    canonical_date = _validated_generated_date(generated_date)
    _validate_media_probe_artifact(
        media_probe_artifact,
        media_probe=media_probe,
        source=source,
        output_directory=output_root,
    )
    try:
        validated_media_probe_payload(media_probe)
    except MediaProbeEvidenceError as exc:
        raise WorkerError(
            "MEDIA_PROBE_EVIDENCE_INVALID",
            "output planning rejected untrusted media probe evidence",
            details={"reason": str(exc)},
        ) from exc
    _require_probe_matches_source(media_probe, source)

    payload = customization.canonical_dict()
    subtitle = payload["subtitle"]
    delivery = payload["delivery"]
    exports = payload["exports"]
    mode = SubtitleOutputMode(delivery["mode"])
    target = delivery["outputTarget"]
    bound_source: Path | None = None
    bound_output: Path | None = None
    if target["binding"] == "bound":
        bound_source = _canonical_bound_source(
            target["sourcePath"],
            expected_source=source,
        )
        bound_output = _canonical_derived_output(
            target["outputPath"],
            output_directory=output_root,
            source_path=source,
        )
    elif mode is not SubtitleOutputMode.SIDECAR and subtitle["enabled"]:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_DELIVERY_UNBOUND",
            "soft-mux and burn-in delivery require bound source and output paths",
            details={"mode": mode.value},
        )

    del bound_source
    if mode is SubtitleOutputMode.BURN_IN and subtitle["enabled"]:
        _validate_burn_in_authorization(
            delivery,
            media_probe=media_probe,
            media_probe_artifact_sha256=media_probe_artifact.sha256,
        )

    formats: tuple[SubtitleFormat, ...] = ()
    style: SubtitleStyle | None = None
    cue_policy: CuePolicy | None = None
    sidecars: tuple[tuple[SubtitleFormat, Path], ...] = ()
    limitations: list[str] = []
    if subtitle["enabled"]:
        formats = tuple(
            SubtitleFormat(value)
            for value in (
                subtitle["format"],
                *exports["subtitleAlternates"],
            )
        )
        style = _compile_subtitle_style(subtitle)
        cue_policy = _compile_cue_policy(subtitle)
        sidecars = _resolve_sidecar_paths(
            formats,
            source_path=source,
            output_directory=output_root,
            file_name_template=exports["fileNameTemplate"],
            language=canonical_language,
            speaker_count=speaker_count,
            generated_date=canonical_date,
            bound_primary=(
                bound_output if mode is SubtitleOutputMode.SIDECAR else None
            ),
        )
        if (
            mode is SubtitleOutputMode.SIDECAR
            and bound_output is not None
            and bound_output.suffix.lower()
            != _SUBTITLE_SUFFIX[formats[0]]
        ):
            raise WorkerError(
                "OUTPUT_CUSTOMIZATION_SUBTITLE_SUFFIX_MISMATCH",
                "bound sidecar output suffix does not match its subtitle format",
                details={
                    "path": str(bound_output),
                    "format": formats[0].value,
                },
            )
        if subtitle["lineHeight"] != 1.0:
            limitations.append(
                "ASS/SRT/WebVTT exporters do not encode arbitrary line-height"
            )
        if subtitle["background"]["radiusPx"] != 0:
            limitations.append(
                "ASS rectangular backgrounds do not encode corner radius"
            )
        if subtitle["shadow"]["color"] != "#000000":
            limitations.append(
                "the current ASS style model records shadow depth, not a separate color"
            )

    codec = delivery["softMux"]["subtitleCodec"]
    plan = OutputExecutionPlan(
        customization_sha256=customization.deterministic_hash(),
        source_path=source,
        output_directory=output_root,
        media_probe_artifact=media_probe_artifact,
        report_enabled=bool(payload["report"]["enabled"]),
        report_config=dict(payload["report"]),
        exports_config=dict(exports),
        safety_config=dict(payload["safety"]),
        reversibility_config=dict(payload["reversibility"]),
        subtitle_enabled=bool(subtitle["enabled"]),
        subtitle_config=dict(subtitle),
        subtitle_formats=formats,
        subtitle_style=style,
        cue_policy=cue_policy,
        subtitle_theme=subtitle["theme"] if subtitle["enabled"] else None,
        speaker_color_mode=(
            subtitle["speakerColors"]["mode"] if subtitle["enabled"] else None
        ),
        speaker_color_seed=(
            subtitle["speakerColors"]["seed"] if subtitle["enabled"] else None
        ),
        speaker_color_overrides=(
            {
                entry["speakerId"]: entry["color"]
                for entry in subtitle["speakerColors"]["overrides"]
            }
            if subtitle["enabled"]
            else {}
        ),
        sidecar_paths=sidecars,
        delivery_mode=mode,
        delivery_output_path=(
            bound_output if mode is not SubtitleOutputMode.SIDECAR else None
        ),
        subtitle_codec=(None if codec == "probe-selected" else codec),
        burn_in_strategy=delivery["burnIn"]["strategy"],
        visual_qa_required=bool(delivery["burnIn"]["requireVisualQa"]),
        presentation_limitations=tuple(limitations),
    )
    _validate_distinct_output_paths(plan)
    return plan


def transcript_subtitle_segments(
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract authoritative presentation text without mutating transcript data."""

    raw_speakers = document.get("speakers")
    if raw_speakers is None:
        raw_speakers = []
    if not isinstance(raw_speakers, list):
        raise WorkerError(
            "SUBTITLE_GENERATION_FAILED",
            "transcript document speakers must be an array when present",
        )
    speaker_names = {
        str(item.get("id")): str(
            item.get("displayName") or item.get("name") or item.get("id")
        )
        for item in raw_speakers
        if isinstance(item, Mapping) and item.get("id")
    }
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise WorkerError(
            "SUBTITLE_GENERATION_FAILED",
            "transcript document contains no subtitle source segments",
        )
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise WorkerError(
                "SUBTITLE_GENERATION_FAILED",
                "transcript subtitle source segment is malformed",
                details={"segmentIndex": index},
            )
        text = (
            raw.get("displayText")
            or raw.get("normalizedText")
            or raw.get("rawText")
        )
        speaker_id = str(raw.get("speakerId") or "").strip()
        if not isinstance(text, str) or not text:
            raise WorkerError(
                "SUBTITLE_GENERATION_FAILED",
                "transcript subtitle source text is empty",
                details={"segmentIndex": index},
            )
        result.append(
            {
                "startMs": raw.get("startMs"),
                "endMs": raw.get("endMs"),
                "text": text,
                "speaker": speaker_names.get(speaker_id, speaker_id or None),
                "speakerId": speaker_id or None,
            }
        )
    return result


def prepare_subtitle_outputs(
    plan: OutputExecutionPlan,
    document: Mapping[str, Any],
    *,
    title: str = "MediaTranscribeStudio subtitles",
) -> PreparedSubtitleOutputs:
    """Use the existing cue/export/plan modules without writing source media."""

    if not isinstance(plan, OutputExecutionPlan) or not plan.subtitle_enabled:
        raise WorkerError(
            "SUBTITLE_OUTPUT_DISABLED",
            "the output execution plan does not enable subtitle generation",
        )
    if plan.cue_policy is None or plan.subtitle_style is None:
        raise WorkerError(
            "SUBTITLE_GENERATION_FAILED",
            "the subtitle execution plan is incomplete",
        )
    try:
        source_duration_ms = document.get("source", {}).get("durationMs")
        source_bound = arrange_cues_within_source_duration(
            transcript_subtitle_segments(document),
            source_duration_ms=source_duration_ms,
            policy=plan.cue_policy,
        )
        arrangement = source_bound.arrangement
        speaker_color_config = plan.subtitle_config["speakerColors"]
        assignments = resolve_speaker_colors(
            arrangement.cues,
            mode=plan.speaker_color_mode,
            seed=plan.speaker_color_seed,
            overrides=plan.speaker_color_overrides,
            algorithm=speaker_color_config["algorithm"],
        )
        arrangement = SubtitleArrangement(
            cues=arrangement.cues,
            qa=arrangement.qa,
            speaker_colors=assignments,
        )
        sidecars: list[PreparedSubtitleArtifact] = []
        for subtitle_format, output_path in plan.sidecar_paths:
            payload = export_subtitles(
                arrangement,
                subtitle_format,
                theme=(
                    SubtitleTheme.CUSTOM
                    if subtitle_format is SubtitleFormat.ASS
                    else SubtitleTheme.YOUTUBE_CLEAN
                ),
                custom_style=(
                    plan.subtitle_style
                    if subtitle_format is SubtitleFormat.ASS
                    else None
                ),
                title=title,
                speaker_color_mode=plan.speaker_color_mode,
                speaker_color_seed=plan.speaker_color_seed,
                speaker_color_overrides=plan.speaker_color_overrides,
                speaker_color_algorithm=speaker_color_config["algorithm"],
                speaker_color_assignments=(
                    assignments
                    if subtitle_format is SubtitleFormat.ASS
                    else None
                ),
            )
            delivery_plan = build_subtitle_output_plan(
                source_path=plan.source_path,
                output_path=output_path,
                subtitle_format=subtitle_format,
                mode=SubtitleOutputMode.SIDECAR,
            )
            sidecars.append(
                PreparedSubtitleArtifact(
                    subtitle_format=subtitle_format,
                    output_path=output_path,
                    payload=payload,
                    delivery_plan=delivery_plan,
                )
            )

        media_delivery_plan: SubtitleOutputPlan | None = None
        if plan.delivery_mode is not SubtitleOutputMode.SIDECAR:
            if plan.delivery_output_path is None or not sidecars:
                raise WorkerError(
                    "SUBTITLE_GENERATION_FAILED",
                    "media subtitle delivery requires a primary sidecar and output path",
                )
            media_delivery_plan = build_subtitle_output_plan(
                source_path=plan.source_path,
                output_path=plan.delivery_output_path,
                subtitle_format=sidecars[0].subtitle_format,
                mode=plan.delivery_mode,
                subtitle_path=sidecars[0].output_path,
                subtitle_codec=plan.subtitle_codec,
                video_encoder=None,
            )
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(
            "SUBTITLE_GENERATION_FAILED",
            "existing subtitle modules rejected the canonical output plan",
            details={"reason": str(exc), "exceptionType": type(exc).__name__},
        ) from exc
    return PreparedSubtitleOutputs(
        arrangement=arrangement,
        sidecars=tuple(sidecars),
        media_delivery_plan=media_delivery_plan,
        visual_qa_required=media_delivery_plan is not None,
        timing_policy_evidence=source_bound.timing_policy_evidence(),
    )


def execute_prepared_subtitle_outputs(
    prepared: PreparedSubtitleOutputs,
    *,
    executor: Any,
    subtitle_language: str,
    subtitle_title: str,
    make_subtitle_default: bool = False,
    burn_in_strategy: BurnInVideoStrategy | str | None = None,
    visual_qa_hook: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Publish media only after QA passes against a quarantined render."""

    if prepared.visual_qa_required and visual_qa_hook is None:
        raise WorkerError(
            "SUBTITLE_VISUAL_QA_REQUIRED",
            "media subtitle delivery requires a representative-frame visual-QA hook",
        )
    language = _validated_language(subtitle_language)
    sidecar_receipts: list[Any] = []
    staged_media: Any = None
    try:
        for artifact in prepared.sidecars:
            sidecar_receipts.append(
                executor.deliver(
                    artifact.delivery_plan,
                    sidecar_payload=artifact.payload,
                    subtitle_language=language,
                    subtitle_title=subtitle_title,
                    make_subtitle_default=make_subtitle_default,
                )
            )
        media_receipt = None
        media_publication = None
        visual_qa_result = None
        if prepared.media_delivery_plan is not None:
            staged_media = executor.stage_media_delivery(
                prepared.media_delivery_plan,
                subtitle_language=language,
                subtitle_title=subtitle_title,
                make_subtitle_default=make_subtitle_default,
                burn_in_strategy=burn_in_strategy,
            )
            assert visual_qa_hook is not None
            try:
                visual_qa_result = visual_qa_hook(
                    source_path=Path(prepared.media_delivery_plan.source_path),
                    rendered_path=Path(staged_media.quarantine_path),
                    arrangement=prepared.arrangement,
                    delivery_receipt=staged_media.receipt,
                )
                visual_qa_result = _passing_visual_qa_evidence(
                    visual_qa_result
                )
            except Exception as exc:
                rollback = _rollback_staged_media(
                    executor,
                    staged_media,
                    reason="visual-qa-failed",
                )
                if isinstance(exc, WorkerError):
                    details = dict(exc.details)
                    details["rollback"] = rollback
                    raise WorkerError(
                        exc.code,
                        exc.message,
                        details=details,
                        retryable=exc.retryable,
                    ) from exc
                raise WorkerError(
                    "SUBTITLE_VISUAL_QA_FAILED",
                    "representative-frame subtitle visual QA failed closed",
                    details={
                        "reason": str(exc),
                        "exceptionType": type(exc).__name__,
                        "rollback": rollback,
                    },
                ) from exc
            publication = executor.publish_staged_media(
                staged_media,
                visual_qa_evidence=visual_qa_result,
            )
            media_receipt = publication.receipt
            media_publication = publication.to_dict()
    except WorkerError:
        raise
    except SubtitleDeliveryError as exc:
        details: dict[str, Any] = {
            "reason": str(exc),
            "deliveryErrorCode": exc.code.value,
        }
        if exc.detail:
            details["deliveryErrorDetail"] = exc.detail
        if exc.evidence:
            details.update(exc.evidence)
        elif staged_media is not None:
            details["rollback"] = _rollback_staged_media(
                executor,
                staged_media,
                reason=f"delivery-{exc.code.value}",
            )
        raise WorkerError(
            "SUBTITLE_DELIVERY_FAILED",
            "subtitle delivery failed closed",
            details=details,
        ) from exc
    except Exception as exc:
        details: dict[str, Any] = {
            "reason": str(exc),
            "exceptionType": type(exc).__name__,
        }
        if staged_media is not None:
            details["rollback"] = _rollback_staged_media(
                executor,
                staged_media,
                reason="delivery-unexpected-failure",
            )
        raise WorkerError(
            "SUBTITLE_DELIVERY_FAILED",
            "subtitle delivery or visual QA failed closed",
            details=details,
        ) from exc
    return {
        "sidecarReceipts": tuple(sidecar_receipts),
        "mediaReceipt": media_receipt,
        "mediaPublication": media_publication,
        "visualQa": visual_qa_result,
    }


def _passing_visual_qa_evidence(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        payload = dict(result)
    else:
        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict):
            raise WorkerError(
                "SUBTITLE_VISUAL_QA_FAILED",
                "the visual-QA hook returned no structured evidence",
            )
        payload = to_dict()
    if not isinstance(payload, Mapping) or payload.get("passed") is not True:
        raise WorkerError(
            "SUBTITLE_VISUAL_QA_FAILED",
            "the visual-QA evidence did not explicitly pass",
        )
    return dict(payload)


def _rollback_staged_media(
    executor: Any,
    staged_media: Any,
    *,
    reason: str,
) -> dict[str, Any]:
    try:
        evidence = executor.rollback_staged_media(
            staged_media,
            reason=reason,
        )
        to_dict = getattr(evidence, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
        elif isinstance(evidence, Mapping):
            payload = dict(evidence)
        else:
            raise TypeError("rollback returned no structured evidence")
        return dict(payload)
    except Exception as exc:
        return {
            "status": "rollback-evidence-unavailable",
            "reason": reason,
            "cleanupError": str(exc),
            "exceptionType": type(exc).__name__,
        }


def render_planned_report(
    plan: OutputExecutionPlan,
    *,
    renderer: Any,
    document: Mapping[str, Any],
    request: Any,
    context: Any,
) -> Any:
    """Invoke the Java PDF adapter with exact canonical presentation config."""

    if not plan.report_enabled:
        return None
    try:
        return renderer.render(
            document,
            request,
            context,
            output_plan=plan,
        )
    except WorkerError:
        raise
    except TypeError as exc:
        raise WorkerError(
            "REPORT_RENDERER_UNPLANNED",
            "configured PDF renderer does not accept output execution plans",
        ) from exc


def _compile_subtitle_style(subtitle: Mapping[str, Any]) -> SubtitleStyle:
    font = subtitle["fontPack"]
    fallbacks: list[str] = []
    for script in ("latin", "cjk", "rtl", "symbols"):
        for value in font["fallbacks"][script]:
            if value != font["primary"] and value not in fallbacks:
                fallbacks.append(value)
    safe_area = subtitle["safeArea"]
    alignment = subtitle["alignment"]
    if alignment.startswith("top-"):
        vertical_percent = safe_area["topPercent"]
    elif alignment.startswith("bottom-"):
        vertical_percent = safe_area["bottomPercent"]
    else:
        vertical_percent = max(
            safe_area["topPercent"],
            safe_area["bottomPercent"],
        )
    shadow = subtitle["shadow"]
    shadow_depth = min(
        12.0,
        max(
            abs(shadow["offsetXPx"]),
            abs(shadow["offsetYPx"]),
            shadow["blurPx"] / 2,
        )
        * shadow["opacity"],
    )
    return SubtitleStyle(
        font_family=font["primary"],
        font_fallbacks=tuple(fallbacks),
        font_size=subtitle["fontSizePx"],
        font_weight=subtitle["fontWeight"],
        italic=subtitle["italic"],
        primary_color=subtitle["foregroundColor"],
        active_word_color=subtitle["activeWordColor"],
        outline_color=subtitle["outline"]["color"],
        outline_width=subtitle["outline"]["widthPx"],
        shadow_depth=shadow_depth,
        background_color=subtitle["background"]["color"],
        background_opacity=subtitle["background"]["opacity"],
        margin_horizontal=round(1920 * safe_area["horizontalPercent"] / 100),
        margin_vertical=round(1080 * vertical_percent / 100),
        alignment=_ALIGNMENT[alignment],
    )


def _compile_cue_policy(subtitle: Mapping[str, Any]) -> CuePolicy:
    source = subtitle["cuePolicy"]
    labels = subtitle["speakerLabels"]
    template = labels["template"]
    if labels["position"] == "inline":
        template = template + ("" if template.endswith((" ", "\t")) else " ")
    elif labels["position"] == "line-above":
        template = template + "\n"
    else:
        template = f"[{template}] "
    return CuePolicy(
        max_characters_per_line=source["maxCharactersPerLine"],
        max_lines=source["maxLines"],
        max_reading_speed=source["maxReadingSpeed"],
        min_cue_ms=source["minCueMs"],
        max_cue_ms=source["maxCueMs"],
        gap_ms=source["gapMs"],
        include_speaker_labels=labels["enabled"],
        speaker_label_template=template,
    )


def _resolve_sidecar_paths(
    formats: Sequence[SubtitleFormat],
    *,
    source_path: Path,
    output_directory: Path,
    file_name_template: str,
    language: str,
    speaker_count: int,
    generated_date: str,
    bound_primary: Path | None,
) -> tuple[tuple[SubtitleFormat, Path], ...]:
    paths: list[tuple[SubtitleFormat, Path]] = []
    seen: set[str] = set()
    for index, subtitle_format in enumerate(formats):
        if index == 0 and bound_primary is not None:
            candidate = bound_primary
        else:
            artifact = (
                "subtitle"
                if index == 0
                else f"subtitle-{subtitle_format.value}"
            )
            filename = _render_filename(
                file_name_template,
                source_stem=source_path.stem,
                artifact=artifact,
                language=language,
                generated_date=generated_date,
                speaker_count=speaker_count,
            )
            candidate = output_directory / (
                filename + _SUBTITLE_SUFFIX[subtitle_format]
            )
        resolved = _canonical_derived_output(
            candidate,
            output_directory=output_directory,
            source_path=source_path,
        )
        key = os.path.normcase(str(resolved))
        if key in seen:
            raise WorkerError(
                "OUTPUT_CUSTOMIZATION_OUTPUT_COLLISION",
                "subtitle output settings resolve multiple formats to one path",
                details={"path": str(resolved)},
            )
        seen.add(key)
        paths.append((subtitle_format, resolved))
    return tuple(paths)


def _render_filename(
    template: str,
    *,
    source_stem: str,
    artifact: str,
    language: str,
    generated_date: str,
    speaker_count: int,
) -> str:
    replacements = {
        "sourceStem": source_stem,
        "artifact": artifact,
        "language": language,
        "date": generated_date,
        "speakerCount": str(speaker_count),
    }
    value = template
    for token, replacement in replacements.items():
        value = value.replace(f"{{{token}}}", replacement)
    if not value or value in {".", ".."}:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_INVALID_FILENAME",
            "fileNameTemplate resolved to an invalid empty filename",
        )
    if (
        _INVALID_WINDOWS_FILENAME.search(value)
        or value.endswith((" ", "."))
        or value.upper().split(".", 1)[0]
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
    ):
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_INVALID_FILENAME",
            "fileNameTemplate resolved to an unsafe platform filename",
            details={"filename": value},
        )
    return value


def _media_probe_artifact(
    published: PublishedJsonEvidence,
    *,
    source: Path,
    media_probe: MediaProbeResult,
) -> MediaProbeArtifact:
    return MediaProbeArtifact(
        path=published.path,
        size_bytes=published.size_bytes,
        sha256=published.sha256,
        source_path=source,
        source_size_bytes=media_probe.source_size_bytes,
        source_sha256=media_probe.source_sha256,
        probe_fingerprint_sha256=media_probe.probe_fingerprint_sha256,
    )


def _canonical_output_directory(value: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise WorkerError(
            "OUTPUT_DIRECTORY_INVALID",
            "job output directory must be an absolute existing directory",
        )
    try:
        canonical = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkerError(
            "OUTPUT_DIRECTORY_INVALID",
            "job output directory cannot be resolved",
            details={"path": str(candidate)},
        ) from exc
    if not canonical.is_dir():
        raise WorkerError(
            "OUTPUT_DIRECTORY_INVALID",
            "job output directory must be an existing directory",
            details={"path": str(canonical)},
        )
    return canonical


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(
            str(right.resolve())
        )


def _require_probe_matches_source(
    media_probe: MediaProbeResult,
    source: Path,
) -> None:
    try:
        probe_source = Path(media_probe.source_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkerError(
            "MEDIA_PROBE_SOURCE_MISMATCH",
            "media probe source path cannot be resolved",
        ) from exc
    if not _same_file(probe_source, source):
        raise WorkerError(
            "MEDIA_PROBE_SOURCE_MISMATCH",
            "media probe evidence does not describe the trusted job source",
        )
    stat = source.stat()
    if stat.st_size != media_probe.source_size_bytes:
        raise WorkerError(
            "SOURCE_MEDIA_CHANGED",
            "source media size no longer matches admission evidence",
            details={"path": str(source)},
        )
    if sha256_file(source) != media_probe.source_sha256:
        raise WorkerError(
            "SOURCE_MEDIA_CHANGED",
            "source media content no longer matches admission evidence",
            details={"path": str(source)},
        )


def _validate_media_probe_artifact(
    artifact: MediaProbeArtifact,
    *,
    media_probe: MediaProbeResult,
    source: Path,
    output_directory: Path,
) -> None:
    if not isinstance(artifact, MediaProbeArtifact):
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "output planning requires immutable media-probe artifact evidence",
        )
    expected_path = output_directory / "media-probe.v1.json"
    try:
        artifact_path = artifact.path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe artifact cannot be resolved",
        ) from exc
    if artifact_path != expected_path or not artifact_path.is_file():
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe artifact must use the fixed job evidence path",
            details={"path": str(artifact_path)},
        )
    if _same_file(artifact_path, source):
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe evidence cannot alias source media",
        )
    if not _SHA256.fullmatch(artifact.sha256):
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe artifact SHA-256 is malformed",
        )
    stat = artifact_path.stat()
    if stat.st_size != artifact.size_bytes or sha256_file(
        artifact_path
    ) != artifact.sha256:
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe artifact bytes do not match their integrity evidence",
        )
    if (
        artifact.source_size_bytes != media_probe.source_size_bytes
        or artifact.source_sha256 != media_probe.source_sha256
        or artifact.probe_fingerprint_sha256
        != media_probe.probe_fingerprint_sha256
        or not _same_file(artifact.source_path, source)
    ):
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe artifact binding does not match admission evidence",
        )
    try:
        persisted_payload = read_json_strict(artifact_path)
    except (OSError, ValueError, WorkerError) as exc:
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe artifact is not strict JSON evidence",
        ) from exc
    if persisted_payload != media_probe.to_dict():
        raise WorkerError(
            "MEDIA_PROBE_ARTIFACT_INVALID",
            "media-probe artifact payload does not match admission evidence",
        )


def _validated_language(value: str) -> str:
    try:
        canonical = normalize_language_tag(value, allow_auto=False)
    except ValueError as exc:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_LANGUAGE_INVALID",
            "output language must be a persisted BCP-47 language tag",
            details={"language": value},
        ) from exc
    return canonical


def _validated_speaker_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_SPEAKER_COUNT_INVALID",
            "speaker_count must be a positive integer",
        )
    return value


def _validated_generated_date(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
    ):
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_DATE_INVALID",
            "generated_date must use YYYY-MM-DD",
        )
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_DATE_INVALID",
            "generated_date is not a valid calendar date",
        ) from exc
    return value


def _validate_distinct_output_paths(plan: OutputExecutionPlan) -> None:
    values = [path for _, path in plan.sidecar_paths]
    if plan.delivery_output_path is not None:
        values.append(plan.delivery_output_path)
    keys = [os.path.normcase(str(path)) for path in values]
    if len(keys) != len(set(keys)):
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_OUTPUT_COLLISION",
            "derived output settings resolve multiple artifacts to one path",
        )


def _canonical_bound_source(value: Any, *, expected_source: Path) -> Path:
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_SOURCE_MISMATCH",
            "bound delivery sourcePath must be the absolute trusted job source",
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_SOURCE_MISMATCH",
            "bound delivery sourcePath cannot be resolved",
        ) from exc
    expected = expected_source.resolve(strict=True)
    try:
        matches = os.path.samefile(resolved, expected)
    except OSError:
        matches = os.path.normcase(str(resolved)) == os.path.normcase(
            str(expected)
        )
    if not matches:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_SOURCE_MISMATCH",
            "bound delivery sourcePath does not match the trusted job source",
        )
    return resolved


def _canonical_derived_output(
    value: Any,
    *,
    output_directory: Path,
    source_path: Path,
) -> Path:
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = output_directory / candidate
    if not candidate.name:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_INVALID_OUTPUT_PATH",
            "derived output path must name a file",
        )
    root = output_directory.resolve(strict=True)
    try:
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_INVALID_OUTPUT_PATH",
            "derived output parent must already exist",
            details={"path": str(candidate)},
        ) from exc
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_OUTPUT_OUTSIDE_JOB",
            "derived outputs must remain inside the job output directory",
            details={"path": str(candidate)},
        ) from exc
    resolved = parent / candidate.name
    source = source_path.resolve(strict=True)
    if os.path.normcase(str(resolved)) == os.path.normcase(str(source)):
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_SOURCE_ALIAS",
            "derived output cannot alias or overwrite source media",
        )
    if os.path.lexists(resolved):
        try:
            aliases_source = os.path.samefile(resolved, source)
        except OSError:
            aliases_source = False
        if aliases_source:
            raise WorkerError(
                "OUTPUT_CUSTOMIZATION_SOURCE_ALIAS",
                "derived output cannot alias or overwrite source media",
            )
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_OUTPUT_EXISTS",
            "derived output already exists and will not be overwritten",
            details={"path": str(resolved)},
        )
    return resolved


def _validate_burn_in_authorization(
    delivery: Mapping[str, Any],
    *,
    media_probe: MediaProbeResult,
    media_probe_artifact_sha256: str,
) -> None:
    if not media_probe.video_stream_indexes:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_HDR_UNSAFE",
            "burn-in requires a trusted non-attached video stream",
            details={"reason": "audio-only"},
        )
    if media_probe.has_hdr_video:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_HDR_UNSAFE",
            "HDR subtitle burn-in is rejected by the production pipeline",
            details={"reason": "hdr-source"},
        )
    video_streams = [
        stream
        for stream in media_probe.streams
        if stream.index in media_probe.video_stream_indexes
    ]
    untrusted_transfers = sorted(
        {
            (
                stream.color_transfer.strip().lower()
                if stream.color_transfer
                else "<missing>"
            )
            for stream in video_streams
            if (
                not stream.color_transfer
                or stream.color_transfer.strip().lower()
                not in _KNOWN_SDR_TRANSFERS
            )
        }
    )
    if untrusted_transfers:
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_HDR_UNSAFE",
            "burn-in rejects sources whose dynamic range is unknown",
            details={
                "reason": "unknown-dynamic-range",
                "colorTransfers": untrusted_transfers,
            },
        )
    evidence = delivery["burnIn"]["dynamicRangeEvidence"]
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("verified") is not True
        or evidence.get("probeSha256") != media_probe_artifact_sha256
    ):
        raise WorkerError(
            "OUTPUT_CUSTOMIZATION_HDR_UNSAFE",
            "burn-in SDR evidence is not bound to media-probe.v1.json",
            details={"reason": "untrusted-dynamic-range-evidence"},
        )
