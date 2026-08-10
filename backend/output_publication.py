"""Transactional publication for multi-mode subtitle and media outputs.

The lower-level subtitle modules intentionally execute one output plan at a
time.  This module owns the higher-level transaction needed when one native
recipe requests public subtitle sidecars, soft-mux media, and burn-in media at
the same time.

The transaction has four important properties:

* Public SRT, WebVTT, and ASS payloads are derived from the original recipe
  and each requested format is published exactly once.
* Media delivery always reads a high-fidelity ASS carrier.  When ASS was not
  requested as a customer artifact, the carrier is created under an
  unpredictable private directory and removed before the transaction returns.
* Every media artifact is rendered into quarantine and receives explicit
  representative-frame visual-QA approval before any media output is
  published.
* Failures and cooperative cancellation roll back transaction-owned customer
  outputs, quarantined media, and private carriers without deleting files that
  no longer match the evidence captured by this transaction.

The returned manifest deliberately excludes low-level commands, temporary
paths, quarantine names, and private carrier paths.  Its normalized receipts
are deterministic for the same recipe, source, customer outputs, and QA
evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import JobCancelled, WorkerError
from .media_probe import canonical_local_media_file
from .output_orchestration import (
    OutputExecutionPlan,
    PreparedSubtitleArtifact,
    PreparedSubtitleOutputs,
    prepare_subtitle_outputs,
)
from .output_recipe import OutputRecipe, OutputRecipeError, parse_output_recipe
from .persistence import canonical_json_sha256
from .subtitle_delivery import SubtitleDeliveryError
from .subtitles import (
    SubtitleFormat,
    SubtitleOutputMode,
    SubtitleTheme,
    build_subtitle_output_plan,
    export_subtitles,
)


OUTPUT_PUBLICATION_SCHEMA_VERSION = "1.0.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORMAT_ORDER = {
    SubtitleFormat.SRT: 0,
    SubtitleFormat.WEBVTT: 1,
    SubtitleFormat.ASS: 2,
}
_MODE_ORDER = {
    SubtitleOutputMode.SIDECAR: 0,
    SubtitleOutputMode.SOFT_MUX: 1,
    SubtitleOutputMode.BURN_IN: 2,
}


class OutputPublicationError(WorkerError):
    """Structured fail-closed output-publication failure."""


class OutputPublicationCancelled(JobCancelled):
    """Cancellation carrying sanitized rollback and cleanup evidence."""

    def __init__(self, *, evidence: Mapping[str, Any]) -> None:
        super().__init__()
        self.details["outputPublication"] = dict(evidence)


@dataclass(frozen=True)
class CustomerArtifactReceipt:
    """Stable customer-visible artifact evidence."""

    artifact_type: str
    path: Path
    size_bytes: int
    sha256: str
    source_sha256: str
    subtitle_format: SubtitleFormat | None = None
    delivery_mode: SubtitleOutputMode | None = None
    visual_qa_evidence_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifactType": self.artifact_type,
            "path": str(self.path),
            "sizeBytes": self.size_bytes,
            "sha256": self.sha256,
            "subtitleFormat": (
                self.subtitle_format.value
                if self.subtitle_format is not None
                else None
            ),
            "deliveryMode": (
                self.delivery_mode.value
                if self.delivery_mode is not None
                else None
            ),
            "visualQaEvidenceSha256": self.visual_qa_evidence_sha256,
            "sourceIntegrity": {
                "unchanged": True,
                "sourceSha256": self.source_sha256,
            },
            "publication": {
                "atomic": True,
                "noReplace": True,
                "sourceMediaImmutable": True,
            },
        }


@dataclass(frozen=True)
class OutputPublicationManifest:
    """Deterministic customer receipts and separately scoped internal evidence."""

    recipe_sha256: str
    source_path: Path
    source_size_bytes: int
    source_sha256: str
    plan_sha256: tuple[tuple[str, str, str], ...]
    customer_artifacts: tuple[CustomerArtifactReceipt, ...]
    internal_evidence: dict[str, Any]

    def _body(self) -> dict[str, Any]:
        return {
            "schemaVersion": OUTPUT_PUBLICATION_SCHEMA_VERSION,
            "status": "published",
            "recipeSha256": self.recipe_sha256,
            "source": {
                "path": str(self.source_path),
                "sizeBytes": self.source_size_bytes,
                "sha256": self.source_sha256,
                "unchanged": True,
            },
            "plans": [
                {
                    "deliveryMode": mode,
                    "customizationSha256": customization_digest,
                    "executionPlanSha256": plan_digest,
                }
                for mode, customization_digest, plan_digest in self.plan_sha256
            ],
            "customerArtifacts": [
                receipt.to_dict() for receipt in self.customer_artifacts
            ],
            "internalEvidence": dict(self.internal_evidence),
            "transaction": {
                "allConflictsCheckedBeforeWrites": True,
                "mediaQuarantinedBeforePublication": True,
                "allMediaQaPassedBeforePublication": True,
                "rollbackSupported": True,
                "privatePathsExcluded": True,
                "sourceMediaImmutable": True,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        return {
            **body,
            "manifestSha256": canonical_json_sha256(body),
        }

    @property
    def manifest_sha256(self) -> str:
        return canonical_json_sha256(self._body())


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    size_bytes: int
    sha256: str
    device: int
    inode: int
    modified_ns: int


@dataclass(frozen=True)
class _PreparedPlan:
    plan: OutputExecutionPlan
    prepared: PreparedSubtitleOutputs


@dataclass(frozen=True)
class _PublicSidecar:
    subtitle_format: SubtitleFormat
    output_path: Path
    payload: str
    artifact: PreparedSubtitleArtifact


@dataclass(frozen=True)
class _MediaOutput:
    mode: SubtitleOutputMode
    plan: OutputExecutionPlan
    prepared: PreparedSubtitleOutputs
    output_path: Path


@dataclass
class _PublishedSidecar:
    candidate: _PublicSidecar
    evidence: _FileSnapshot


@dataclass
class _StagedMedia:
    candidate: _MediaOutput
    staged: Any
    visual_qa: dict[str, Any] | None = None
    visual_qa_sha256: str | None = None
    publication: Any = None
    evidence: _FileSnapshot | None = None


@dataclass
class _TransactionState:
    source_before: _FileSnapshot
    public_sidecars: list[_PublishedSidecar] = field(default_factory=list)
    staged_media: list[_StagedMedia] = field(default_factory=list)
    private_root: Path | None = None
    private_carrier: Path | None = None
    private_payload_sha256: str | None = None
    private_payload_size: int | None = None
    private_evidence: _FileSnapshot | None = None

    @property
    def has_side_effects(self) -> bool:
        return bool(
            self.public_sidecars
            or self.staged_media
            or self.private_root is not None
        )


@dataclass(frozen=True)
class _Preflight:
    recipe: OutputRecipe
    source: Path
    output_root: Path
    prepared_plans: tuple[_PreparedPlan, ...]
    public_sidecars: tuple[_PublicSidecar, ...]
    media_outputs: tuple[_MediaOutput, ...]
    ass_payload: str | None


def publish_output_plans(
    recipe: OutputRecipe | Mapping[str, Any],
    plans: Sequence[OutputExecutionPlan],
    document: Mapping[str, Any],
    *,
    executor: Any,
    visual_qa_hook: Callable[..., Any] | None,
    subtitle_language: str = "und",
    subtitle_title: str = "MediaTranscribeStudio subtitles",
    make_subtitle_default: bool = False,
    cancellation_check: Callable[[], None] | Any | None = None,
) -> OutputPublicationManifest:
    """Publish one canonical recipe's subtitle and media plans transactionally.

    ``visual_qa_hook`` is mandatory whenever soft-mux or burn-in output is
    requested.  It is invoked once for every quarantined media artifact and
    must return structured evidence containing the exact boolean
    ``{"passed": true}``.
    """

    preflight = _preflight_publication(
        recipe,
        plans,
        document,
        visual_qa_hook=visual_qa_hook,
        subtitle_title=subtitle_title,
        cancellation_check=cancellation_check,
    )
    source_before = _capture_snapshot(preflight.source)
    state = _TransactionState(source_before=source_before)
    public_by_format = {
        item.subtitle_format: item for item in preflight.public_sidecars
    }
    public_ass = public_by_format.get(SubtitleFormat.ASS)

    try:
        _check_cancelled(cancellation_check)

        carrier_path: Path | None = None
        if preflight.media_outputs:
            if preflight.ass_payload is None:
                raise OutputPublicationError(
                    "OUTPUT_PUBLICATION_ASS_REQUIRED",
                    "media subtitle delivery requires a deterministic ASS payload",
                )
            if public_ass is not None:
                published_ass = _publish_public_sidecar(
                    public_ass,
                    executor=executor,
                    source_before=source_before,
                    subtitle_language=subtitle_language,
                    subtitle_title=subtitle_title,
                    make_subtitle_default=make_subtitle_default,
                )
                state.public_sidecars.append(published_ass)
                carrier_path = public_ass.output_path
            else:
                carrier_path = _publish_private_ass_carrier(
                    preflight,
                    state=state,
                    executor=executor,
                    subtitle_language=subtitle_language,
                    subtitle_title=subtitle_title,
                    make_subtitle_default=make_subtitle_default,
                )
            _require_source_unchanged(source_before)

            for candidate in preflight.media_outputs:
                _check_cancelled(cancellation_check)
                delivery_plan = build_subtitle_output_plan(
                    source_path=preflight.source,
                    output_path=candidate.output_path,
                    subtitle_format=SubtitleFormat.ASS,
                    mode=candidate.mode,
                    subtitle_path=carrier_path,
                    subtitle_codec=candidate.plan.subtitle_codec,
                    video_encoder=None,
                )
                staged = executor.stage_media_delivery(
                    delivery_plan,
                    subtitle_language=subtitle_language,
                    subtitle_title=subtitle_title,
                    make_subtitle_default=make_subtitle_default,
                    burn_in_strategy=(
                        candidate.plan.burn_in_strategy
                        if candidate.mode is SubtitleOutputMode.BURN_IN
                        else None
                    ),
                )
                _validate_staged_media(
                    staged,
                    candidate=candidate,
                    output_root=preflight.output_root,
                )
                state.staged_media.append(
                    _StagedMedia(candidate=candidate, staged=staged)
                )
                _require_source_unchanged(source_before)
                _check_cancelled(cancellation_check)

            assert visual_qa_hook is not None
            for item in state.staged_media:
                _check_cancelled(cancellation_check)
                evidence = visual_qa_hook(
                    source_path=preflight.source,
                    rendered_path=Path(item.staged.quarantine_path),
                    arrangement=item.candidate.prepared.arrangement,
                    delivery_receipt=item.staged.receipt,
                    execution_plan=item.candidate.plan,
                )
                item.visual_qa = _passing_visual_qa(evidence)
                item.visual_qa_sha256 = canonical_json_sha256(item.visual_qa)
                _require_source_unchanged(source_before)
                _check_cancelled(cancellation_check)

            # No media customer path is created until every quarantined render
            # has independently and explicitly passed representative-frame QA.
            for item in state.staged_media:
                assert item.visual_qa is not None
                _check_cancelled(cancellation_check)
                item.publication = executor.publish_staged_media(
                    item.staged,
                    visual_qa_evidence=item.visual_qa,
                )
                item.evidence = _validate_published_media(item)
                _require_source_unchanged(source_before)
                _check_cancelled(cancellation_check)

        for candidate in preflight.public_sidecars:
            if (
                preflight.media_outputs
                and candidate.subtitle_format is SubtitleFormat.ASS
                and public_ass
            ):
                continue
            _check_cancelled(cancellation_check)
            published = _publish_public_sidecar(
                candidate,
                executor=executor,
                source_before=source_before,
                subtitle_language=subtitle_language,
                subtitle_title=subtitle_title,
                make_subtitle_default=make_subtitle_default,
            )
            state.public_sidecars.append(published)
            _require_source_unchanged(source_before)
            _check_cancelled(cancellation_check)

        private_cleanup = _cleanup_private_carrier(state)
        if private_cleanup["status"] not in {"removed", "not-created"}:
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_PRIVATE_CLEANUP_FAILED",
                "the private ASS carrier could not be safely removed",
                details={"privateCarrier": private_cleanup},
            )
        _require_no_quarantine(state)
        source_after = _require_source_unchanged(source_before)
        return _build_manifest(
            preflight,
            state=state,
            source_after=source_after,
            private_cleanup=private_cleanup,
        )
    except JobCancelled as exc:
        rollback = _rollback_transaction(
            state,
            executor=executor,
            reason="cancelled",
        )
        raise OutputPublicationCancelled(evidence=rollback) from exc
    except Exception as exc:
        rollback = _rollback_transaction(
            state,
            executor=executor,
            reason=_failure_reason(exc),
        )
        if isinstance(exc, OutputPublicationError) and not state.has_side_effects:
            raise
        details: dict[str, Any] = {
            "phase": _failure_phase(state),
            "reason": str(exc),
            "exceptionType": type(exc).__name__,
            "rollback": rollback,
        }
        if isinstance(exc, WorkerError):
            details["causeCode"] = exc.code
            if exc.details:
                details["causeDetails"] = _sanitize_cause_details(exc.details)
        elif isinstance(exc, SubtitleDeliveryError):
            details["deliveryErrorCode"] = exc.code.value
            if exc.detail:
                details["deliveryErrorDetail"] = exc.detail
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_FAILED",
            "multi-mode subtitle/media publication failed closed",
            details=details,
        ) from exc


def _preflight_publication(
    recipe_value: OutputRecipe | Mapping[str, Any],
    plans: Sequence[OutputExecutionPlan],
    document: Mapping[str, Any],
    *,
    visual_qa_hook: Callable[..., Any] | None,
    subtitle_title: str,
    cancellation_check: Callable[[], None] | Any | None,
) -> _Preflight:
    recipe = _coerce_recipe(recipe_value)
    if not isinstance(document, Mapping):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_DOCUMENT_INVALID",
            "subtitle publication requires a transcript document mapping",
        )
    _validate_cancellation_check(cancellation_check)
    plan_values = tuple(plans) if isinstance(plans, Sequence) else ()
    if not plan_values:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PLANS_REQUIRED",
            "at least one OutputExecutionPlan is required",
        )
    if any(not isinstance(plan, OutputExecutionPlan) for plan in plan_values):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PLAN_INVALID",
            "every publication plan must be an OutputExecutionPlan",
        )

    payload = recipe.canonical_dict()
    subtitles_enabled = payload["subtitles"]["enabled"]
    requested_modes = tuple(
        SubtitleOutputMode(value)
        for value in payload["delivery"]["subtitleModes"]
    )
    explicit_formats = tuple(
        SubtitleFormat(value)
        for value in payload["delivery"]["formats"]
        if value in {"srt", "webvtt", "ass"}
    )
    if subtitles_enabled and (
        SubtitleOutputMode.SIDECAR in requested_modes
        and not explicit_formats
    ):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PUBLIC_FORMAT_REQUIRED",
            "sidecar mode requires an explicitly requested SRT, WebVTT, or ASS format",
        )

    source = _canonical_source(plan_values[0].source_path)
    output_root = _canonical_output_root(plan_values[0].output_directory)
    mode_to_plan: dict[SubtitleOutputMode, OutputExecutionPlan] = {}
    for plan in plan_values:
        _validate_plan_invariants(
            plan,
            source=source,
            output_root=output_root,
            subtitles_enabled=subtitles_enabled,
        )
        if not plan.subtitle_enabled:
            continue
        if plan.delivery_mode in mode_to_plan:
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_DUPLICATE_MODE",
                "the plan collection contains a duplicate delivery mode",
                details={"deliveryMode": plan.delivery_mode.value},
            )
        mode_to_plan[plan.delivery_mode] = plan

    expected_modes = set(requested_modes) if subtitles_enabled else set()
    actual_modes = set(mode_to_plan)
    if actual_modes != expected_modes:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_MODE_CONFLICT",
            "the plan collection does not exactly match the canonical recipe modes",
            details={
                "recipeModes": [
                    mode.value
                    for mode in sorted(requested_modes, key=_mode_sort_key)
                ],
                "planModes": [
                    mode.value
                    for mode in sorted(actual_modes, key=_mode_sort_key)
                ],
            },
        )
    media_modes = expected_modes - {SubtitleOutputMode.SIDECAR}
    if media_modes and not callable(visual_qa_hook):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_VISUAL_QA_REQUIRED",
            "soft-mux and burn-in publication require a visual-QA hook",
        )

    if not subtitles_enabled:
        return _Preflight(
            recipe=recipe,
            source=source,
            output_root=output_root,
            prepared_plans=(),
            public_sidecars=(),
            media_outputs=(),
            ass_payload=None,
        )

    prepared_plans: list[_PreparedPlan] = []
    by_format: dict[
        SubtitleFormat,
        list[tuple[SubtitleOutputMode, PreparedSubtitleArtifact]],
    ] = {}
    for mode in requested_modes:
        plan = mode_to_plan[mode]
        prepared = prepare_subtitle_outputs(
            plan,
            document,
            title=subtitle_title,
        )
        prepared_plans.append(_PreparedPlan(plan=plan, prepared=prepared))
        present_formats = {
            artifact.subtitle_format for artifact in prepared.sidecars
        }
        missing = [
            value.value
            for value in explicit_formats
            if value not in present_formats
        ]
        if missing:
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_FORMAT_MISSING",
                "a canonical plan omitted an explicitly requested public subtitle format",
                details={"deliveryMode": mode.value, "formats": missing},
            )
        for artifact in prepared.sidecars:
            by_format.setdefault(artifact.subtitle_format, []).append(
                (mode, artifact)
            )

    _validate_payload_consistency(by_format)
    public_sidecars = _derive_public_sidecars(
        explicit_formats,
        by_format=by_format,
        output_root=output_root,
        source=source,
    )
    media_outputs = _derive_media_outputs(
        prepared_plans,
        output_root=output_root,
        source=source,
    )
    _validate_customer_path_uniqueness(public_sidecars, media_outputs)
    ass_payload = _derive_ass_payload(
        prepared_plans,
        by_format=by_format,
        title=subtitle_title,
    )
    return _Preflight(
        recipe=recipe,
        source=source,
        output_root=output_root,
        prepared_plans=tuple(prepared_plans),
        public_sidecars=public_sidecars,
        media_outputs=media_outputs,
        ass_payload=ass_payload,
    )


def _coerce_recipe(value: OutputRecipe | Mapping[str, Any]) -> OutputRecipe:
    if isinstance(value, OutputRecipe):
        return value
    if not isinstance(value, Mapping):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_RECIPE_INVALID",
            "recipe must be an OutputRecipe or canonical recipe mapping",
        )
    try:
        return parse_output_recipe(value)
    except OutputRecipeError as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_RECIPE_INVALID",
            "the output recipe failed strict canonical validation",
            details={"reason": str(exc)},
        ) from exc


def _canonical_source(path: Path) -> Path:
    try:
        return canonical_local_media_file(path)
    except Exception as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_SOURCE_INVALID",
            "the publication source is not a canonical local media file",
            details={"reason": str(exc)},
        ) from exc


def _canonical_output_root(path: Path) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_OUTPUT_ROOT_INVALID",
            "the publication output directory cannot be resolved",
            details={"reason": str(exc)},
        ) from exc
    if not resolved.is_dir():
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_OUTPUT_ROOT_INVALID",
            "the publication output root must be a directory",
        )
    return resolved


def _validate_plan_invariants(
    plan: OutputExecutionPlan,
    *,
    source: Path,
    output_root: Path,
    subtitles_enabled: bool,
) -> None:
    if _canonical_source(plan.source_path) != source:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_SOURCE_CONFLICT",
            "all output plans must reference the same immutable source",
        )
    if _canonical_output_root(plan.output_directory) != output_root:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_OUTPUT_ROOT_CONFLICT",
            "all output plans must share one canonical output directory",
        )
    if plan.subtitle_enabled is not subtitles_enabled:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_SUBTITLE_STATE_CONFLICT",
            "plan subtitle state conflicts with the canonical recipe",
            details={"deliveryMode": plan.delivery_mode.value},
        )
    safety = plan.safety_config
    reversibility = plan.reversibility_config
    if (
        safety.get("preserveSourceMedia") is not True
        or safety.get("overwriteSourceMedia") is not False
        or reversibility.get("sourceMediaImmutable") is not True
        or reversibility.get("derivedArtifactOnly") is not True
    ):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_SOURCE_PROTECTION_INVALID",
            "an output plan does not preserve immutable-source invariants",
            details={"deliveryMode": plan.delivery_mode.value},
        )
    if not _SHA256.fullmatch(plan.customization_sha256):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PLAN_HASH_INVALID",
            "an output plan has no trustworthy customization SHA-256",
            details={"deliveryMode": plan.delivery_mode.value},
        )


def _validate_payload_consistency(
    by_format: Mapping[
        SubtitleFormat,
        Sequence[tuple[SubtitleOutputMode, PreparedSubtitleArtifact]],
    ],
) -> None:
    for subtitle_format, variants in by_format.items():
        digests = {
            hashlib.sha256(artifact.payload.encode("utf-8")).hexdigest()
            for _, artifact in variants
        }
        if len(digests) > 1:
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_PAYLOAD_CONFLICT",
                "the same subtitle format produced conflicting payloads across plans",
                details={
                    "format": subtitle_format.value,
                    "deliveryModes": sorted(
                        {mode.value for mode, _ in variants}
                    ),
                },
            )


def _derive_public_sidecars(
    explicit_formats: Sequence[SubtitleFormat],
    *,
    by_format: Mapping[
        SubtitleFormat,
        Sequence[tuple[SubtitleOutputMode, PreparedSubtitleArtifact]],
    ],
    output_root: Path,
    source: Path,
) -> tuple[_PublicSidecar, ...]:
    result: list[_PublicSidecar] = []
    for subtitle_format in explicit_formats:
        variants = tuple(by_format.get(subtitle_format, ()))
        if not variants:
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_FORMAT_MISSING",
                "an explicitly requested subtitle format has no prepared payload",
                details={"format": subtitle_format.value},
            )
        sidecar_variants = tuple(
            item
            for item in variants
            if item[0] is SubtitleOutputMode.SIDECAR
        )
        # Media plans may bind the same payload to plan-private carrier names
        # because ASS is primary there. An explicit sidecar plan is the sole
        # authority for customer-visible subtitle paths.
        target_variants = sidecar_variants or variants
        targets = {
            _canonical_customer_output(
                artifact.output_path,
                output_root=output_root,
                source=source,
            )
            for _, artifact in target_variants
        }
        if len(targets) != 1:
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_TARGET_CONFLICT",
                "the same public subtitle format resolves to conflicting output paths",
                details={"format": subtitle_format.value},
            )
        selected = min(
            target_variants,
            key=lambda item: _mode_sort_key(item[0]),
        )[1]
        result.append(
            _PublicSidecar(
                subtitle_format=subtitle_format,
                output_path=next(iter(targets)),
                payload=selected.payload,
                artifact=selected,
            )
        )
    return tuple(sorted(result, key=lambda item: _format_sort_key(item.subtitle_format)))


def _derive_media_outputs(
    prepared_plans: Sequence[_PreparedPlan],
    *,
    output_root: Path,
    source: Path,
) -> tuple[_MediaOutput, ...]:
    result: list[_MediaOutput] = []
    for item in prepared_plans:
        if item.plan.delivery_mode is SubtitleOutputMode.SIDECAR:
            if item.prepared.media_delivery_plan is not None:
                raise OutputPublicationError(
                    "OUTPUT_PUBLICATION_MEDIA_PLAN_CONFLICT",
                    "sidecar mode unexpectedly contains a media delivery plan",
                )
            continue
        if (
            item.plan.delivery_output_path is None
            or item.prepared.media_delivery_plan is None
        ):
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_MEDIA_PLAN_MISSING",
                "a media delivery mode is missing its bound output plan",
                details={"deliveryMode": item.plan.delivery_mode.value},
            )
        output = _canonical_customer_output(
            item.plan.delivery_output_path,
            output_root=output_root,
            source=source,
        )
        result.append(
            _MediaOutput(
                mode=item.plan.delivery_mode,
                plan=item.plan,
                prepared=item.prepared,
                output_path=output,
            )
        )
    return tuple(sorted(result, key=lambda item: _mode_sort_key(item.mode)))


def _derive_ass_payload(
    prepared_plans: Sequence[_PreparedPlan],
    *,
    by_format: Mapping[
        SubtitleFormat,
        Sequence[tuple[SubtitleOutputMode, PreparedSubtitleArtifact]],
    ],
    title: str,
) -> str | None:
    media = [
        item
        for item in prepared_plans
        if item.plan.delivery_mode is not SubtitleOutputMode.SIDECAR
    ]
    if not media:
        return None
    variants = by_format.get(SubtitleFormat.ASS, ())
    payloads = [artifact.payload for _, artifact in variants]
    if not payloads:
        for item in media:
            if item.plan.subtitle_style is None:
                raise OutputPublicationError(
                    "OUTPUT_PUBLICATION_ASS_REQUIRED",
                    "a media plan cannot generate its required ASS carrier",
                    details={"deliveryMode": item.plan.delivery_mode.value},
                )
            payloads.append(
                export_subtitles(
                    item.prepared.arrangement,
                    SubtitleFormat.ASS,
                    theme=SubtitleTheme.CUSTOM,
                    custom_style=item.plan.subtitle_style,
                    title=title,
                )
            )
    digests = {
        hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for payload in payloads
    }
    if len(digests) != 1:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PAYLOAD_CONFLICT",
            "media plans produced conflicting ASS carrier payloads",
            details={"format": SubtitleFormat.ASS.value},
        )
    return payloads[0]


def _validate_customer_path_uniqueness(
    sidecars: Sequence[_PublicSidecar],
    media_outputs: Sequence[_MediaOutput],
) -> None:
    owners: dict[str, str] = {}
    for item in sidecars:
        _claim_path(
            owners,
            item.output_path,
            f"sidecar:{item.subtitle_format.value}",
        )
    for item in media_outputs:
        _claim_path(owners, item.output_path, f"media:{item.mode.value}")


def _claim_path(owners: dict[str, str], path: Path, owner: str) -> None:
    key = _path_key(path)
    previous = owners.get(key)
    if previous is not None:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_DUPLICATE_OUTPUT_PATH",
            "two customer artifacts resolve to the same output path",
            details={"first": previous, "second": owner, "path": str(path)},
        )
    owners[key] = owner


def _canonical_customer_output(
    path: Path,
    *,
    output_root: Path,
    source: Path,
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.name:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_OUTPUT_PATH_INVALID",
            "customer output paths must be absolute local file paths",
        )
    try:
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_OUTPUT_PATH_INVALID",
            "a customer output parent cannot be resolved",
            details={"reason": str(exc)},
        ) from exc
    resolved = parent / candidate.name
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_OUTPUT_ESCAPE",
            "customer outputs must remain inside the job output directory",
            details={"path": str(resolved)},
        ) from exc
    if _path_key(resolved) == _path_key(source):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_SOURCE_ALIAS",
            "a customer output path aliases the immutable source media",
        )
    if os.path.lexists(resolved):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_OUTPUT_EXISTS",
            "output publication never replaces an existing artifact",
            details={"path": str(resolved)},
        )
    return resolved


def _publish_private_ass_carrier(
    preflight: _Preflight,
    *,
    state: _TransactionState,
    executor: Any,
    subtitle_language: str,
    subtitle_title: str,
    make_subtitle_default: bool,
) -> Path:
    assert preflight.ass_payload is not None
    try:
        root = Path(
            tempfile.mkdtemp(
                prefix=".mts-private-subtitle-",
                dir=preflight.output_root,
            )
        ).resolve(strict=True)
        root.chmod(0o700)
    except OSError as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PRIVATE_CARRIER_FAILED",
            "an unpredictable private subtitle directory could not be created",
            details={"reason": str(exc)},
        ) from exc
    state.private_root = root
    carrier = root / f"carrier-{secrets.token_hex(24)}.ass"
    state.private_carrier = carrier
    payload_bytes = preflight.ass_payload.encode("utf-8")
    state.private_payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    state.private_payload_size = len(payload_bytes)
    delivery_plan = build_subtitle_output_plan(
        source_path=preflight.source,
        output_path=carrier,
        subtitle_format=SubtitleFormat.ASS,
        mode=SubtitleOutputMode.SIDECAR,
    )
    receipt = executor.deliver(
        delivery_plan,
        sidecar_payload=preflight.ass_payload,
        subtitle_language=subtitle_language,
        subtitle_title=subtitle_title,
        make_subtitle_default=make_subtitle_default,
    )
    evidence = _capture_snapshot(carrier)
    if (
        evidence.size_bytes != state.private_payload_size
        or evidence.sha256 != state.private_payload_sha256
    ):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PRIVATE_CARRIER_MISMATCH",
            "the private ASS carrier differs from the prepared payload",
        )
    _require_receipt_matches(receipt, evidence)
    state.private_evidence = evidence
    return carrier


def _publish_public_sidecar(
    candidate: _PublicSidecar,
    *,
    executor: Any,
    source_before: _FileSnapshot,
    subtitle_language: str,
    subtitle_title: str,
    make_subtitle_default: bool,
) -> _PublishedSidecar:
    receipt = executor.deliver(
        candidate.artifact.delivery_plan,
        sidecar_payload=candidate.payload,
        subtitle_language=subtitle_language,
        subtitle_title=subtitle_title,
        make_subtitle_default=make_subtitle_default,
    )
    evidence = _capture_snapshot(candidate.output_path)
    expected = candidate.payload.encode("utf-8")
    if (
        evidence.size_bytes != len(expected)
        or evidence.sha256 != hashlib.sha256(expected).hexdigest()
    ):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_SIDECAR_MISMATCH",
            "a published sidecar differs from its prepared payload",
            details={"format": candidate.subtitle_format.value},
        )
    _require_receipt_matches(receipt, evidence)
    _require_source_unchanged(source_before)
    return _PublishedSidecar(candidate=candidate, evidence=evidence)


def _validate_staged_media(
    staged: Any,
    *,
    candidate: _MediaOutput,
    output_root: Path,
) -> None:
    quarantine_value = getattr(staged, "quarantine_path", None)
    customer_value = getattr(staged, "customer_output_path", None)
    receipt = getattr(staged, "receipt", None)
    if quarantine_value is None or customer_value is None or receipt is None:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_QUARANTINE_INVALID",
            "the media executor returned no structured quarantine evidence",
            details={"deliveryMode": candidate.mode.value},
        )
    quarantine = Path(quarantine_value)
    customer = Path(customer_value)
    if _path_key(customer) != _path_key(candidate.output_path):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_QUARANTINE_INVALID",
            "the staged customer path differs from the canonical media target",
            details={"deliveryMode": candidate.mode.value},
        )
    if not quarantine.is_absolute() or not os.path.lexists(quarantine):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_QUARANTINE_INVALID",
            "the executor did not create a private quarantined media artifact",
            details={"deliveryMode": candidate.mode.value},
        )
    try:
        quarantine.parent.resolve(strict=True).relative_to(output_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_QUARANTINE_INVALID",
            "the quarantined media path escaped the output directory",
            details={"deliveryMode": candidate.mode.value},
        ) from exc
    if os.path.lexists(customer):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_PREMATURE_MEDIA_PUBLICATION",
            "customer media appeared before representative-frame QA",
            details={"deliveryMode": candidate.mode.value},
        )


def _passing_visual_qa(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_VISUAL_QA_FAILED",
                "visual QA returned no structured evidence",
            )
        payload = to_dict()
    if not isinstance(payload, Mapping) or payload.get("passed") is not True:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_VISUAL_QA_FAILED",
            "representative-frame visual QA did not explicitly pass",
        )
    normalized = dict(payload)
    try:
        canonical_json_sha256(normalized)
    except (TypeError, ValueError) as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_VISUAL_QA_FAILED",
            "visual QA evidence is not deterministic strict JSON",
            details={"reason": str(exc)},
        ) from exc
    return normalized


def _validate_published_media(item: _StagedMedia) -> _FileSnapshot:
    output = item.candidate.output_path
    if not os.path.lexists(output):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_MEDIA_MISSING",
            "QA-approved media was not published to its customer path",
            details={"deliveryMode": item.candidate.mode.value},
        )
    if os.path.lexists(Path(item.staged.quarantine_path)):
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_QUARANTINE_CLEANUP_FAILED",
            "published media retained its private quarantine artifact",
            details={"deliveryMode": item.candidate.mode.value},
        )
    evidence = _capture_snapshot(output)
    published_evidence = getattr(item.publication, "published_evidence", None)
    if published_evidence is not None:
        _require_external_evidence_matches(published_evidence, evidence)
    return evidence


def _require_receipt_matches(receipt: Any, evidence: _FileSnapshot) -> None:
    external = getattr(receipt, "output_evidence", None)
    if external is not None:
        _require_external_evidence_matches(external, evidence)


def _require_external_evidence_matches(
    external: Any,
    evidence: _FileSnapshot,
) -> None:
    size = getattr(external, "size_bytes", None)
    digest = getattr(external, "sha256", None)
    if size != evidence.size_bytes or digest != evidence.sha256:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_RECEIPT_MISMATCH",
            "low-level receipt evidence differs from the published artifact",
        )


def _build_manifest(
    preflight: _Preflight,
    *,
    state: _TransactionState,
    source_after: _FileSnapshot,
    private_cleanup: Mapping[str, Any],
) -> OutputPublicationManifest:
    receipts: list[CustomerArtifactReceipt] = []
    for item in state.public_sidecars:
        receipts.append(
            CustomerArtifactReceipt(
                artifact_type="subtitle-sidecar",
                path=item.evidence.path,
                size_bytes=item.evidence.size_bytes,
                sha256=item.evidence.sha256,
                source_sha256=source_after.sha256,
                subtitle_format=item.candidate.subtitle_format,
                delivery_mode=SubtitleOutputMode.SIDECAR,
            )
        )
    quarantine_evidence: list[dict[str, Any]] = []
    for item in state.staged_media:
        if item.evidence is None or item.visual_qa_sha256 is None:
            raise OutputPublicationError(
                "OUTPUT_PUBLICATION_MANIFEST_INCOMPLETE",
                "published media is missing deterministic receipt evidence",
            )
        receipts.append(
            CustomerArtifactReceipt(
                artifact_type="subtitled-media",
                path=item.evidence.path,
                size_bytes=item.evidence.size_bytes,
                sha256=item.evidence.sha256,
                source_sha256=source_after.sha256,
                subtitle_format=SubtitleFormat.ASS,
                delivery_mode=item.candidate.mode,
                visual_qa_evidence_sha256=item.visual_qa_sha256,
            )
        )
        quarantine_evidence.append(
            {
                "deliveryMode": item.candidate.mode.value,
                "visualQaPassed": True,
                "visualQaEvidenceSha256": item.visual_qa_sha256,
                "publishedAtomically": True,
                "quarantineCleanup": "removed",
            }
        )
    receipts.sort(key=_receipt_sort_key)
    plan_hashes = tuple(
        (
            item.plan.delivery_mode.value,
            item.plan.customization_sha256,
            item.plan.deterministic_hash(),
        )
        for item in sorted(
            preflight.prepared_plans,
            key=lambda value: _mode_sort_key(value.plan.delivery_mode),
        )
    )
    private_created = state.private_payload_sha256 is not None
    timing_policy_evidence = (
        dict(preflight.prepared_plans[0].prepared.timing_policy_evidence)
        if preflight.prepared_plans
        else None
    )
    internal = {
        "privateAssCarrier": {
            "created": private_created,
            "customerArtifact": False,
            "pathDisclosed": False,
            "unpredictableName": private_created,
            "payloadSha256": (
                state.private_payload_sha256 if private_created else None
            ),
            "sizeBytes": (
                state.private_payload_size if private_created else None
            ),
            "cleanup": private_cleanup["status"],
            "reason": (
                "public-ass-reused"
                if preflight.media_outputs and not private_created
                else (
                    "internal-media-carrier"
                    if private_created
                    else "media-not-requested"
                )
            ),
        },
        "mediaQuarantine": quarantine_evidence,
        "subtitleTimingPolicy": timing_policy_evidence,
        "customerAndInternalEvidenceSeparated": True,
        "privatePathsExcluded": True,
    }
    return OutputPublicationManifest(
        recipe_sha256=preflight.recipe.deterministic_hash(),
        source_path=source_after.path,
        source_size_bytes=source_after.size_bytes,
        source_sha256=source_after.sha256,
        plan_sha256=plan_hashes,
        customer_artifacts=tuple(receipts),
        internal_evidence=internal,
    )


def _rollback_transaction(
    state: _TransactionState,
    *,
    executor: Any,
    reason: str,
) -> dict[str, Any]:
    media_results: list[dict[str, Any]] = []
    for item in reversed(state.staged_media):
        published_evidence = getattr(
            item.publication,
            "published_evidence",
            None,
        )
        try:
            rollback = executor.rollback_staged_media(
                item.staged,
                reason=reason,
                published_evidence=published_evidence,
            )
            media_results.append(
                {
                    "deliveryMode": item.candidate.mode.value,
                    **_sanitize_media_rollback(rollback),
                }
            )
        except Exception as exc:
            media_results.append(
                {
                    "deliveryMode": item.candidate.mode.value,
                    "status": "rollback-evidence-unavailable",
                    "cleanupError": str(exc),
                    "exceptionType": type(exc).__name__,
                }
            )

    sidecar_results: list[dict[str, Any]] = []
    for item in reversed(state.public_sidecars):
        removal = _remove_matching_file(
            item.evidence.path,
            expected=item.evidence,
        )
        sidecar_results.append(
            {
                "format": item.candidate.subtitle_format.value,
                **removal,
            }
        )

    private_cleanup = _cleanup_private_carrier(state)
    try:
        source_after = _capture_snapshot(state.source_before.path)
        source_unchanged = source_after == state.source_before
    except Exception as exc:
        source_unchanged = False
        source_error = str(exc)
    else:
        source_error = None
    complete = (
        all(
            item.get("customerOutputState")
            in {None, "absent", "rolled-back"}
            and item.get("quarantineState")
            in {None, "absent", "removed"}
            and not item.get("cleanupError")
            for item in media_results
        )
        and all(
            item["status"] in {"removed", "absent"}
            for item in sidecar_results
        )
        and private_cleanup["status"] in {"removed", "not-created"}
    )
    return {
        "status": "rolled-back" if complete else "rollback-incomplete",
        "reason": reason,
        "media": list(reversed(media_results)),
        "publicSidecars": list(reversed(sidecar_results)),
        "privateAssCarrier": private_cleanup,
        "sourceMediaImmutable": source_unchanged,
        "sourceInspectionError": source_error,
        "privatePathsExcluded": True,
    }


def _sanitize_media_rollback(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        payload = to_dict() if callable(to_dict) else {}
    quarantine = payload.get("quarantine")
    if not isinstance(quarantine, Mapping):
        quarantine = {}
    return {
        "status": str(payload.get("status") or "rollback-evidence-unavailable"),
        "customerOutputState": payload.get("customerOutputState"),
        "quarantineState": quarantine.get("state"),
        "sourceMediaImmutable": payload.get("sourceMediaImmutable"),
        "cleanupError": payload.get("cleanupError"),
    }


def _cleanup_private_carrier(state: _TransactionState) -> dict[str, Any]:
    root = state.private_root
    carrier = state.private_carrier
    if root is None and carrier is None:
        return {
            "status": "not-created",
            "customerArtifact": False,
            "pathDisclosed": False,
        }
    carrier_status = "absent"
    cleanup_error: str | None = None
    if carrier is not None and os.path.lexists(carrier):
        expected = state.private_evidence
        if expected is None:
            try:
                current = _capture_snapshot(carrier)
            except Exception as exc:
                carrier_status = "inspection-failed-retained"
                cleanup_error = str(exc)
            else:
                if (
                    current.sha256 == state.private_payload_sha256
                    and current.size_bytes == state.private_payload_size
                ):
                    try:
                        carrier.unlink()
                        carrier_status = "removed"
                    except OSError as exc:
                        carrier_status = "removal-failed-retained"
                        cleanup_error = str(exc)
                else:
                    carrier_status = "changed-retained"
        else:
            removal = _remove_matching_file(carrier, expected=expected)
            carrier_status = removal["status"]
            cleanup_error = removal.get("cleanupError")

    root_status = "absent"
    if root is not None and os.path.lexists(root):
        try:
            root.rmdir()
            root_status = "removed"
        except OSError as exc:
            root_status = "not-empty-or-removal-failed"
            cleanup_error = (
                f"{cleanup_error}; {exc}" if cleanup_error else str(exc)
            )
    if carrier_status in {"removed", "absent"} and root_status in {
        "removed",
        "absent",
    }:
        status = "removed"
    else:
        status = "cleanup-incomplete"
    state.private_root = None if root_status in {"removed", "absent"} else root
    state.private_carrier = (
        None if carrier_status in {"removed", "absent"} else carrier
    )
    return {
        "status": status,
        "carrierState": carrier_status,
        "directoryState": root_status,
        "customerArtifact": False,
        "pathDisclosed": False,
        "cleanupError": cleanup_error,
    }


def _remove_matching_file(
    path: Path,
    *,
    expected: _FileSnapshot,
) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {"status": "absent", "cleanupError": None}
    try:
        current = _capture_snapshot(path)
    except Exception as exc:
        return {
            "status": "inspection-failed-retained",
            "cleanupError": str(exc),
        }
    if (
        current.size_bytes != expected.size_bytes
        or current.sha256 != expected.sha256
        or current.device != expected.device
        or current.inode != expected.inode
    ):
        return {"status": "changed-retained", "cleanupError": None}
    try:
        path.unlink()
    except OSError as exc:
        return {
            "status": "removal-failed-retained",
            "cleanupError": str(exc),
        }
    return {"status": "removed", "cleanupError": None}


def _require_no_quarantine(state: _TransactionState) -> None:
    retained = [
        item.candidate.mode.value
        for item in state.staged_media
        if os.path.lexists(Path(item.staged.quarantine_path))
    ]
    if retained:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_QUARANTINE_CLEANUP_FAILED",
            "one or more published media quarantine artifacts remain",
            details={"deliveryModes": retained},
        )


def _capture_snapshot(path: Path) -> _FileSnapshot:
    candidate = Path(path)
    try:
        before = candidate.stat()
        if not candidate.is_file():
            raise OSError("path is not a regular file")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = candidate.stat()
    except OSError as exc:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_EVIDENCE_FAILED",
            "artifact evidence could not be captured",
            details={"path": str(candidate), "reason": str(exc)},
        ) from exc
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_EVIDENCE_UNSTABLE",
            "artifact changed while its integrity evidence was captured",
            details={"path": str(candidate)},
        )
    return _FileSnapshot(
        path=candidate.resolve(strict=True),
        size_bytes=int(after.st_size),
        sha256=digest.hexdigest(),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        modified_ns=int(after.st_mtime_ns),
    )


def _require_source_unchanged(before: _FileSnapshot) -> _FileSnapshot:
    after = _capture_snapshot(before.path)
    if after != before:
        raise OutputPublicationError(
            "OUTPUT_PUBLICATION_SOURCE_CHANGED",
            "the immutable source media changed during output publication",
            details={
                "sourceSha256Before": before.sha256,
                "sourceSha256After": after.sha256,
            },
        )
    return after


def _validate_cancellation_check(value: Any) -> None:
    if value is None or callable(value):
        return
    if callable(getattr(value, "is_set", None)):
        return
    raise OutputPublicationError(
        "OUTPUT_PUBLICATION_CANCELLATION_INVALID",
        "cancellation_check must be callable or expose is_set()",
    )


def _check_cancelled(value: Any) -> None:
    if value is None:
        return
    if callable(value):
        value()
        return
    if value.is_set():
        raise JobCancelled()


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, WorkerError):
        return f"failure-{exc.code.lower()}"
    if isinstance(exc, SubtitleDeliveryError):
        return f"delivery-{exc.code.value}"
    return f"failure-{type(exc).__name__.lower()}"


def _failure_phase(state: _TransactionState) -> str:
    if any(item.publication is not None for item in state.staged_media):
        return "customer-publication"
    if state.staged_media:
        return "quarantine-or-visual-qa"
    if state.public_sidecars:
        return "sidecar-publication"
    if state.private_root is not None:
        return "private-carrier"
    return "preflight"


def _sanitize_cause_details(value: Mapping[str, Any]) -> dict[str, Any]:
    # Error details may originate from a low-level delivery boundary and can
    # contain private quarantine paths.  Preserve only non-path diagnostics.
    result: dict[str, Any] = {}
    for key, item in value.items():
        lowered = key.casefold()
        if "path" in lowered or "quarantine" in lowered:
            continue
        if isinstance(item, (str, bool, int, float)) or item is None:
            result[str(key)] = item
    return result


def _receipt_sort_key(receipt: CustomerArtifactReceipt) -> tuple[int, int, str]:
    if receipt.artifact_type == "subtitle-sidecar":
        assert receipt.subtitle_format is not None
        return (0, _format_sort_key(receipt.subtitle_format), str(receipt.path))
    assert receipt.delivery_mode is not None
    return (1, _mode_sort_key(receipt.delivery_mode), str(receipt.path))


def _format_sort_key(value: SubtitleFormat) -> int:
    return _FORMAT_ORDER[value]


def _mode_sort_key(value: SubtitleOutputMode) -> int:
    return _MODE_ORDER[value]


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path)).casefold()


__all__ = [
    "CustomerArtifactReceipt",
    "OUTPUT_PUBLICATION_SCHEMA_VERSION",
    "OutputPublicationCancelled",
    "OutputPublicationError",
    "OutputPublicationManifest",
    "publish_output_plans",
]
