"""Pluggable model and renderer interfaces; defaults deliberately fail closed."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .errors import JobCancelled, WorkerError
from .language import UNDETERMINED_LANGUAGE, normalize_language_tag
from .models import RenderArtifact, RenderResult, StartJobRequest, TranscriptionResult


_JAVA_PDF_ARTIFACT_TYPES = {
    "reportDocumentPath": "pdf-report-document-v1",
    "htmlPath": "pdf-canonical-xhtml",
    "pdfPath": "pdf",
    "manifestPath": "pdf-render-manifest-v1",
    "qualityReportPath": "pdf-quality-report-v1",
    "repairQueuePath": "pdf-repair-queue-v1",
    "contactSheetPath": "pdf-contact-sheet",
}


def _provider_id(value: Any, default: str) -> str:
    if isinstance(value, Mapping):
        for key in ("id", "name", "provider"):
            candidate = str(value.get(key) or "").strip()
            if candidate:
                return candidate
        return default
    candidate = str(value or "").strip()
    return candidate or default


def _provider_version(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    candidate = str(value.get("version") or value.get("revision") or "").strip()
    return candidate or None


@dataclass(frozen=True)
class AdapterContext:
    job_id: str
    output_directory: Path
    cancellation: threading.Event

    def raise_if_cancelled(self) -> None:
        if self.cancellation.is_set():
            raise JobCancelled()


@runtime_checkable
class TranscriptionAdapter(Protocol):
    adapter_id: str
    version: str

    def transcribe(
        self,
        request: StartJobRequest,
        context: AdapterContext,
    ) -> TranscriptionResult | Mapping[str, Any]:
        """Return typed transcript evidence without inventing missing results."""


@runtime_checkable
class ReportRendererAdapter(Protocol):
    adapter_id: str
    version: str

    def render(
        self,
        document: Mapping[str, Any],
        request: StartJobRequest,
        context: AdapterContext,
        *,
        output_plan: Any | None = None,
    ) -> RenderResult:
        """Render and verify a dynamic-cardinality transcript document."""


class UnavailableTranscriptionAdapter:
    adapter_id = "unavailable"
    version = "0"

    def transcribe(
        self,
        request: StartJobRequest,
        context: AdapterContext,
    ) -> TranscriptionResult:
        context.raise_if_cancelled()
        raise WorkerError(
            "ADAPTER_UNAVAILABLE",
            "no offline transcription adapter has been configured",
        )


class UnavailableRendererAdapter:
    adapter_id = "unavailable"
    version = "0"

    def render(
        self,
        document: Mapping[str, Any],
        request: StartJobRequest,
        context: AdapterContext,
        *,
        output_plan: Any | None = None,
    ) -> RenderResult:
        context.raise_if_cancelled()
        raise WorkerError(
            "RENDERER_UNAVAILABLE",
            "PDF rendering was requested but no renderer adapter is configured",
        )


class JavaPdfRendererAdapter:
    """Dynamic ReportDocument -> Java OpenHTMLtoPDF/PDFBox bridge."""

    adapter_id = "java-openhtmltopdf-pdfbox"
    version = "2"

    def __init__(self, *, assembler: Any, java_client: Any) -> None:
        self.assembler = assembler
        self.java_client = java_client

    def render(
        self,
        document: Mapping[str, Any],
        request: StartJobRequest,
        context: AdapterContext,
        *,
        output_plan: Any | None = None,
    ) -> RenderResult:
        context.raise_if_cancelled()
        policy = document.get("speakerPolicy")
        if not isinstance(policy, Mapping):
            raise WorkerError(
                "REPORT_DOCUMENT_INVALID",
                "transcript document is missing speakerPolicy",
            )
        count = policy.get("resolvedCount")
        speaker_ids = policy.get("speakerIds")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or not isinstance(speaker_ids, list)
            or speaker_ids
            != [f"speaker-{index}" for index in range(1, count + 1)]
        ):
            raise WorkerError(
                "REPORT_DOCUMENT_INVALID",
                "speakerPolicy must expose a contiguous dynamic speaker namespace",
            )
        source = document.get("source")
        segments = document.get("segments")
        if not isinstance(source, Mapping) or not isinstance(segments, list):
            raise WorkerError(
                "REPORT_DOCUMENT_INVALID",
                "transcript document source/segments are malformed",
            )

        report_segments: list[dict[str, Any]] = []
        for index, raw_segment in enumerate(segments):
            if not isinstance(raw_segment, Mapping):
                raise WorkerError(
                    "REPORT_DOCUMENT_INVALID",
                    f"segments[{index}] must be an object",
                )
            segment = dict(raw_segment)
            scores = segment.get("speakerScores")
            if not isinstance(scores, list):
                raise WorkerError(
                    "REPORT_DOCUMENT_INVALID",
                    f"segments[{index}].speakerScores must be an array",
                )
            raw_evidence = segment.get("evidence")
            raw_evidence = (
                dict(raw_evidence) if isinstance(raw_evidence, Mapping) else {}
            )
            voiceprint = raw_evidence.get("voiceprint")
            if not isinstance(voiceprint, Mapping):
                voiceprint = {}
            asr = raw_evidence.get("asr")
            if not isinstance(asr, Mapping):
                asr = {}
            boundary = raw_evidence.get("boundary")
            if not isinstance(boundary, Mapping):
                boundary = {}
            overlap = raw_evidence.get("overlap")
            if not isinstance(overlap, Mapping):
                overlap = {}
            asr_provider = asr.get("provider")
            confidence_available = asr.get("confidenceAvailable", True)
            if not isinstance(confidence_available, bool):
                raise WorkerError(
                    "REPORT_DOCUMENT_INVALID",
                    f"segments[{index}].evidence.asr.confidenceAvailable must be a boolean",
                )
            segment["speaker_locked"] = bool(segment.get("humanLocked", False))
            segment["overlapDetected"] = bool(segment.get("overlapping", False))
            report_evidence: dict[str, Any] = {
                "asr": {
                    "provider": _provider_id(asr_provider, "qwen-asr"),
                    "model": str(asr.get("model") or "Qwen3-ASR-1.7B"),
                    "confidence": float(
                        asr.get("confidence", segment.get("confidence", 0.0))
                    ),
                    "confidenceAvailable": confidence_available,
                },
                "boundary": {
                    "provider": _provider_id(
                        boundary.get("provider"), "funasr-forced-aligner"
                    ),
                    "model": str(
                        boundary.get("model") or "FunASR/Qwen3-ForcedAligner"
                    ),
                    "confidence": float(boundary.get("confidence", 1.0)),
                    "overlapDetected": bool(
                        segment.get("overlapping", False)
                        or overlap.get("overlapping", False)
                    ),
                },
                "speaker": {
                    "provider": _provider_id(
                        voiceprint.get("provider"), "camp-plus"
                    ),
                    "model": str(voiceprint.get("model") or "CAM++"),
                    "assignment": str(segment.get("speakerId") or ""),
                    "locked": bool(segment.get("humanLocked", False)),
                    "margin": float(segment.get("speakerMargin", 0.0)),
                    "scores": list(scores),
                },
            }
            asr_model_revision = str(
                asr.get("modelRevision")
                or asr.get("model_revision")
                or _provider_version(asr_provider)
                or ""
            ).strip()
            if asr_model_revision:
                report_evidence["asr"]["modelRevision"] = asr_model_revision
            for key in ("overlap", "pyannoteCanonicalMapping"):
                value = raw_evidence.get(key)
                if isinstance(value, Mapping):
                    report_evidence[key] = dict(value)
            segment["evidence"] = report_evidence
            revisions = segment.get("revisions", [])
            if isinstance(revisions, list):
                segment["revisions"] = [
                    {
                        "revisionId": str(
                            revision.get("revisionId")
                            or revision.get("id")
                            or ""
                        ),
                        "type": revision.get("type"),
                        "source": revision.get("source"),
                        "reasonCode": revision.get("reasonCode"),
                        "before": revision.get("before"),
                        "after": revision.get("after"),
                        **(
                            {"confidence": revision.get("confidence")}
                            if "confidence" in revision
                            else {}
                        ),
                        **(
                            {"evidenceRefs": revision.get("evidenceRefs")}
                            if "evidenceRefs" in revision
                            else {}
                        ),
                        **(
                            {"model": revision.get("model")}
                            if "model" in revision
                            else {}
                        ),
                        **(
                            {"actor": revision.get("actor")}
                            if "actor" in revision
                            else {}
                        ),
                        **(
                            {"occurredAt": revision.get("occurredAt")}
                            if "occurredAt" in revision
                            else {}
                        ),
                    }
                    for revision in revisions
                    if isinstance(revision, Mapping)
                ]
            report_segments.append(segment)

        provenance = document.get("provenance")
        raw_models = (
            provenance.get("models", [])
            if isinstance(provenance, Mapping)
            else []
        )
        role_map = {
            "preparation": "vad",
            "asr": "asr",
            "voiceprint": "speaker",
            "secondary-voiceprint": "speaker",
            "overlap": "overlap",
            "semantic": "semantic",
        }
        report_models: list[dict[str, str]] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, Mapping):
                    continue
                role = role_map.get(str(item.get("role") or ""))
                name = str(item.get("name") or "").strip()
                if role and name:
                    report_models.append({"role": role, "name": name})
        if not report_models:
            report_models = [
                {"role": "asr", "name": "Qwen3-ASR-1.7B"},
                {"role": "boundary", "name": "FunASR"},
                {"role": "speaker", "name": "CAM++"},
            ]

        mode = str(policy.get("mode") or "")
        estimate = policy.get("estimate")
        estimate = estimate if isinstance(estimate, Mapping) else {}
        bounds = policy.get("bounds")
        bounds = bounds if isinstance(bounds, Mapping) else {}
        speaker_profiles = {
            str(item.get("id")): dict(item)
            for item in document.get("speakers", [])
            if isinstance(item, Mapping) and item.get("id") in speaker_ids
        }
        try:
            report_language = normalize_language_tag(
                document.get("language") or UNDETERMINED_LANGUAGE,
                allow_auto=False,
            )
        except ValueError as exc:
            raise WorkerError(
                "REPORT_DOCUMENT_INVALID",
                "transcript document language must be a persisted BCP-47 tag",
                details={"language": document.get("language")},
            ) from exc
        try:
            renderer_config: dict[str, Any] = {
                "speakerPolicy": dict(policy),
                "offline": True,
                "renderer": self.adapter_id,
            }
            if output_plan is not None:
                from .output_orchestration import OutputExecutionPlan

                if not isinstance(output_plan, OutputExecutionPlan):
                    raise WorkerError(
                        "OUTPUT_EXECUTION_PLAN_INVALID",
                        "Java PDF rendering requires an OutputExecutionPlan",
                    )
                renderer_config = output_plan.report_renderer_config(
                    speaker_policy=policy,
                    renderer_id=self.adapter_id,
                )
            report_document = self.assembler.assemble(
                report_segments,
                source_path=request.source_path,
                duration_ms=source["durationMs"],
                document_id=document.get("documentId"),
                generated_at=document.get("generatedAt"),
                title=document.get("title"),
                language=report_language,
                speaker_count_mode=mode,
                speaker_count=count if mode in {"manual", "hybrid"} else None,
                minimum_speaker_count=bounds.get("min"),
                maximum_speaker_count=bounds.get("max"),
                speaker_count_confidence=estimate.get("confidence"),
                speaker_count_candidates=(
                    [
                        {
                            "count": estimate.get("estimatedCount"),
                            "confidence": estimate.get("confidence"),
                        }
                    ]
                    if mode in {"auto", "hybrid"} and estimate
                    else None
                ),
                speaker_mapping={
                    speaker_id: speaker_id for speaker_id in speaker_ids
                },
                speaker_profiles=speaker_profiles,
                models=report_models,
                config=renderer_config,
            )
            outcome = self.java_client.render(
                report_document,
                job_id=request.job_id,
                output_directory=request.output_directory,
            )
        except WorkerError:
            raise
        except Exception as exc:
            raise WorkerError(
                "REPORT_RENDER_FAILED",
                "dynamic Java PDF reporting adapter failed closed",
                details={"exceptionType": type(exc).__name__},
            ) from exc
        result = outcome.result
        artifacts = outcome.artifact_paths
        template_hash = str(
            result.get("templateHash")
            or result.get("quality", {}).get("templateHash")
            or ""
        )
        if not template_hash:
            import hashlib

            template_hash = hashlib.sha256(
                Path(artifacts["htmlPath"]).read_bytes()
            ).hexdigest()
        return RenderResult(
            template_hash=template_hash,
            renderer_version=str(result.get("rendererVersion") or ""),
            quality_status=str(result.get("quality", {}).get("status") or ""),
            quality_report_path=Path(artifacts["qualityReportPath"]),
            render_manifest_path=Path(artifacts["manifestPath"]),
            artifact_paths=tuple(
                Path(path)
                for key, path in artifacts.items()
                if key != "screenshotsDirectory"
            ),
            artifacts=tuple(
                RenderArtifact(
                    artifact_type=_JAVA_PDF_ARTIFACT_TYPES[key],
                    path=Path(path),
                )
                for key, path in artifacts.items()
                if key in _JAVA_PDF_ARTIFACT_TYPES
            ),
        )
