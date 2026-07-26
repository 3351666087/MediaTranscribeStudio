"""Explicit production dependency graph for the offline worker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reporting import JavaPdfClient, ReportDocumentAssembler

from .adapters import JavaPdfRendererAdapter
from .local_llm import LocalLLMConfig, OllamaLocalProvider
from .media_probe import MediaProbe
from .paths import PathPolicy
from .production_config import (
    ProductionConfig,
    ProductionPreflightReport,
    apply_offline_environment,
    offline_environment,
    run_production_preflight,
)
from .production_runners import (
    FfmpegFunAsrPreparationAdapter,
    LocalERes2NetV2Verifier,
    LocalFunAsrCamPlusAdapter,
    LocalPyannoteAuditAdapter,
    LocalQwen3AsrAdapter,
)
from .service import WorkerService
from .speaker_pipeline import (
    JsonStageCache,
    SpeakerPipeline,
    SpeakerPipelineConfig,
)

EventSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ProductionFactories:
    """Injection seam used only by deterministic composition tests."""

    preparation: Callable[..., Any] = FfmpegFunAsrPreparationAdapter
    asr: Callable[..., Any] = LocalQwen3AsrAdapter
    embedding: Callable[..., Any] = LocalFunAsrCamPlusAdapter
    secondary: Callable[..., Any] = LocalERes2NetV2Verifier
    pyannote: Callable[..., Any] = LocalPyannoteAuditAdapter
    cache: Callable[..., Any] = JsonStageCache
    pipeline: Callable[..., Any] = SpeakerPipeline
    assembler: Callable[..., Any] = ReportDocumentAssembler
    java_client_from_jar: Callable[..., Any] = JavaPdfClient.from_jar
    renderer: Callable[..., Any] = JavaPdfRendererAdapter
    media_probe: Callable[..., Any] = MediaProbe
    service: Callable[..., Any] = WorkerService


@dataclass(frozen=True)
class ProductionComposition:
    service: WorkerService
    preflight: ProductionPreflightReport


def build_production_composition(
    config: ProductionConfig,
    *,
    event_sink: EventSink | None = None,
    factories: ProductionFactories | None = None,
    preflight_report: ProductionPreflightReport | None = None,
    probe_runtime_imports: bool = True,
) -> ProductionComposition:
    """Build every production adapter explicitly; no unavailable fallback exists."""

    factories = factories or ProductionFactories()
    apply_offline_environment()
    preflight = preflight_report or run_production_preflight(
        config,
        probe_runtime_imports=probe_runtime_imports,
    )
    preflight.raise_if_failed()

    preparation = factories.preparation(
        vad_model_path=config.models.funasr_vad,
        ffmpeg_executable=config.executables.ffmpeg,
        device=config.runtime.vad_device,
    )
    asr = factories.asr(
        model_path=config.models.qwen3_asr,
        forced_aligner_path=config.models.qwen3_forced_aligner,
        device_map=config.runtime.asr_device,
        torch_dtype=config.runtime.asr_dtype,
    )
    embedding = factories.embedding(
        model_path=config.models.cam_plus,
        device=config.runtime.cam_plus_device,
        max_language_window_ms=config.speaker.max_language_window_ms,
        language_split_search_ms=config.speaker.language_split_search_ms,
    )
    secondary = factories.secondary(
        model_path=config.models.eres2net_v2,
        device=config.runtime.eres2net_device,
        decision_margin=config.speaker.eres2net_decision_margin,
    )
    pyannote = None
    if config.speaker.pyannote_mode != "disabled":
        assert config.models.pyannote is not None
        pyannote = factories.pyannote(
            model_path=config.models.pyannote,
            device=config.runtime.pyannote_device,
            python_executable=config.executables.pyannote_python,
        )

    pipeline_config = SpeakerPipelineConfig(
        normalization_profile=config.speaker.normalization_profile,
        cluster_similarity_threshold=(
            config.speaker.cluster_similarity_threshold
        ),
        low_margin_threshold=config.speaker.low_margin_threshold,
        high_margin_threshold=config.speaker.high_margin_threshold,
        outlier_score_threshold=config.speaker.outlier_score_threshold,
        max_auto_speakers=config.speaker.max_auto_speakers,
        max_clustering_windows=config.speaker.max_clustering_windows,
        max_clustering_work_items=config.speaker.max_clustering_work_items,
        kmeans_iterations=config.speaker.kmeans_iterations,
        max_batch_size=config.speaker.max_batch_size,
        max_secondary_fraction=config.speaker.max_secondary_fraction,
        max_count_uncertainty_candidates=(
            config.speaker.max_count_uncertainty_candidates
        ),
        auto_count_confidence_threshold=(
            config.speaker.auto_count_confidence_threshold
        ),
        count_stability_runs=config.speaker.count_stability_runs,
        eigengap_landmark_limit=config.speaker.eigengap_landmark_limit,
        pyannote_mapping_margin_threshold=(
            config.speaker.pyannote_mapping_margin_threshold
        ),
        pyannote_primary_dominance_threshold=(
            config.speaker.pyannote_primary_dominance_threshold
        ),
        pyannote_mode=config.speaker.pyannote_mode,
        local_llm_mode=config.speaker.local_llm_mode,
        local_llm_model=config.speaker.local_llm_model,
        model_residency=config.runtime.model_residency,
    )
    transcription = factories.pipeline(
        preparation_adapter=preparation,
        asr_adapter=asr,
        embedding_adapter=embedding,
        overlap_adapter=pyannote,
        secondary_adapter=secondary,
        pyannote_adapter=pyannote,
        cache=factories.cache(config.paths.cache_root),
        config=pipeline_config,
    )

    assembler = factories.assembler(
        pipeline_version=getattr(transcription, "version", "2.0.0"),
        low_confidence_threshold=config.speaker.auto_count_confidence_threshold,
        semantic_margin_threshold=config.speaker.low_margin_threshold,
        allow_neutral_compatibility_evidence=False,
    )
    java_client = factories.java_client_from_jar(
        config.executables.pdf_renderer_jar,
        allowed_output_root=config.paths.allowed_output_root,
        java_executable=config.executables.java,
        timeout_seconds=config.pdf.timeout_seconds,
        minimum_score=config.pdf.minimum_score,
        max_rounds=config.pdf.max_rounds,
        capture_dpi=config.pdf.capture_dpi,
        margin_mm=config.pdf.margin_mm,
        preferred_font=config.pdf.preferred_font,
        template_id=config.pdf.template_id,
        environment=offline_environment(),
    )
    renderer = factories.renderer(
        assembler=assembler,
        java_client=java_client,
    )
    ffmpeg_path = Path(config.executables.ffmpeg)
    ffprobe_name = "ffprobe.exe" if ffmpeg_path.suffix.casefold() == ".exe" else "ffprobe"
    sibling_ffprobe = ffmpeg_path.with_name(ffprobe_name)
    ffprobe_command = (
        str(sibling_ffprobe)
        if sibling_ffprobe.is_file()
        else "ffprobe"
    )
    media_probe = factories.media_probe(
        ffprobe_command=(ffprobe_command,),
        ffmpeg_command=(config.executables.ffmpeg,),
    )
    service = factories.service(
        path_policy=PathPolicy(
            allowed_input_roots=config.paths.allowed_input_roots,
            allowed_output_root=config.paths.allowed_output_root,
        ),
        transcription_adapter=transcription,
        renderer_adapter=renderer,
        media_probe=media_probe,
        event_sink=event_sink,
        max_workers=config.runtime.max_workers,
        max_pending_jobs=config.runtime.max_pending_jobs,
        heartbeat_interval_seconds=(
            config.runtime.heartbeat_interval_seconds
        ),
        count_confidence_threshold=(
            config.speaker.auto_count_confidence_threshold
        ),
        segment_confidence_threshold=(
            config.speaker.auto_count_confidence_threshold
        ),
        low_speaker_margin_threshold=config.speaker.low_margin_threshold,
        high_speaker_margin_threshold=config.speaker.high_margin_threshold,
        business_provider_factory=lambda request: OllamaLocalProvider(
            LocalLLMConfig(
                model=request.business_config.model,
                endpoint=request.local_llm_endpoint,
            )
        ),
        semantic_provider_factory=lambda request: OllamaLocalProvider(
            LocalLLMConfig(
                model=config.speaker.local_llm_model,
                endpoint=request.local_llm_endpoint,
            )
        ),
        semantic_required=True,
        semantic_model=config.speaker.local_llm_model,
    )
    return ProductionComposition(service=service, preflight=preflight)


__all__ = [
    "ProductionComposition",
    "ProductionFactories",
    "build_production_composition",
]
