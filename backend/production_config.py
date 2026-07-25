"""Strict, local-only configuration for the production worker composition root.

The legacy project accepted many implicit defaults and network-capable model
identifiers.  The production worker deliberately does the opposite: every
runtime artifact is an explicit local path, unknown configuration keys fail,
and diagnostics expose component state without echoing sensitive paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import WorkerError

PRODUCTION_CONFIG_SCHEMA_VERSION = "1.0.0"
PRODUCTION_MODE = "offline-production"
_MAX_CONFIG_BYTES = 1024 * 1024
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_RUNTIME_IMPORT_TIMEOUT_SECONDS = 120.0


class ProductionConfigError(WorkerError):
    """A fail-closed production configuration or preflight failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "PRODUCTION_CONFIG_INVALID",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, details=details)


def _object(
    value: Any,
    *,
    field: str,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionConfigError(f"{field} must be an object")
    output = {str(key): item for key, item in value.items()}
    unknown = sorted(set(output) - allowed)
    if unknown:
        raise ProductionConfigError(
            f"{field} contains unsupported fields",
            details={"field": field, "unsupportedFields": unknown},
        )
    missing = sorted(required - set(output))
    if missing:
        raise ProductionConfigError(
            f"{field} is missing required fields",
            details={"field": field, "missingFields": missing},
        )
    return output


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionConfigError(f"{field} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ProductionConfigError(f"{field} contains control characters")
    return value.strip()


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProductionConfigError(f"{field} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionConfigError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = (
            f" between {minimum} and {maximum}"
            if maximum is not None
            else f" at least {minimum}"
        )
        raise ProductionConfigError(f"{field} must be{suffix}")
    return value


def _number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionConfigError(f"{field} must be a number")
    output = float(value)
    lower_invalid = output <= minimum if minimum_exclusive else output < minimum
    if lower_invalid or output > maximum:
        relation = "greater than" if minimum_exclusive else "at least"
        raise ProductionConfigError(
            f"{field} must be {relation} {minimum} and at most {maximum}"
        )
    return output


def _choice(value: Any, *, field: str, choices: set[str]) -> str:
    output = _nonempty_text(value, field=field)
    if output not in choices:
        raise ProductionConfigError(
            f"{field} must be one of {', '.join(sorted(choices))}"
        )
    return output


def _reject_remote_or_unc(text: str, *, field: str) -> None:
    if _URI_PATTERN.match(text):
        raise ProductionConfigError(f"{field} must be a local filesystem path")
    if text.startswith(("\\\\", "//")):
        raise ProductionConfigError(f"{field} must not be a UNC path")


def _local_path(
    value: Any,
    *,
    field: str,
    base_directory: Path,
) -> Path:
    text = _nonempty_text(value, field=field)
    _reject_remote_or_unc(text, field=field)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return Path(os.path.abspath(candidate))


def _executable_text(
    value: Any,
    *,
    field: str,
    base_directory: Path,
) -> str:
    text = _nonempty_text(value, field=field)
    _reject_remote_or_unc(text, field=field)
    candidate = Path(text).expanduser()
    contains_separator = any(
        separator and separator in text for separator in (os.sep, os.altsep)
    )
    if candidate.is_absolute() or contains_separator:
        return str(_local_path(text, field=field, base_directory=base_directory))
    return text


def _optional_local_path(
    value: Any,
    *,
    field: str,
    base_directory: Path,
) -> Path | None:
    if value is None:
        return None
    return _local_path(value, field=field, base_directory=base_directory)


def _unique_paths(values: Sequence[Path], *, field: str) -> tuple[Path, ...]:
    output: list[Path] = []
    seen: set[str] = set()
    for path in values:
        key = os.path.normcase(str(path))
        if key in seen:
            raise ProductionConfigError(f"{field} must not contain duplicates")
        seen.add(key)
        output.append(path)
    return tuple(output)


@dataclass(frozen=True)
class ProductionPaths:
    allowed_input_roots: tuple[Path, ...]
    allowed_output_root: Path
    cache_root: Path


@dataclass(frozen=True)
class ProductionModels:
    funasr_vad: Path
    qwen3_asr: Path
    qwen3_forced_aligner: Path | None
    cam_plus: Path
    eres2net_v2: Path
    pyannote: Path | None


@dataclass(frozen=True)
class ProductionExecutables:
    ffmpeg: str
    java: str
    pdf_renderer_jar: Path
    pyannote_python: str | None = None


@dataclass(frozen=True)
class ProductionRuntime:
    max_workers: int = 1
    max_pending_jobs: int = 1
    max_line_bytes: int = 1024 * 1024
    model_residency: str = "stage"
    heartbeat_interval_seconds: float = 15.0
    vad_device: str = "cpu"
    asr_device: str = "cuda:0"
    asr_dtype: str = "bfloat16"
    cam_plus_device: str = "cuda:0"
    eres2net_device: str = "gpu"
    pyannote_device: str = "cuda"
    strict_startup_preflight: bool = True


@dataclass(frozen=True)
class ProductionSpeakerPolicy:
    normalization_profile: str = "mono-16khz-f32-v1"
    cluster_similarity_threshold: float = 0.72
    low_margin_threshold: float = 0.18
    high_margin_threshold: float = 0.35
    outlier_score_threshold: float = 0.30
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
    max_language_window_ms: int = 12_000
    language_split_search_ms: int = 1_000
    eres2net_decision_margin: float = 0.05
    pyannote_mapping_margin_threshold: float = 0.05
    pyannote_primary_dominance_threshold: float = 0.60
    pyannote_mode: str = "disabled"
    local_llm_mode: str = "disabled"
    local_llm_model: str = "qwen3.5:9b"


@dataclass(frozen=True)
class ProductionPdfPolicy:
    minimum_score: float = 85.0
    max_rounds: int = 5
    capture_dpi: int = 144
    margin_mm: float = 14.0
    preferred_font: str | None = None
    template_id: str = "mts-cute-transcript-v1"
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class ProductionConfig:
    """Fully validated production settings with local absolute paths."""

    schema_version: str
    mode: str
    offline: bool
    paths: ProductionPaths
    models: ProductionModels
    executables: ProductionExecutables
    runtime: ProductionRuntime
    speaker: ProductionSpeakerPolicy
    pdf: ProductionPdfPolicy
    source_path: Path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ProductionConfig":
        source = Path(path).expanduser()
        if not source.is_file():
            raise ProductionConfigError(
                "production configuration file is missing",
                details={"component": "production-config"},
            )
        if source.is_symlink():
            raise ProductionConfigError(
                "production configuration must not be a symbolic link"
            )
        if source.stat().st_size > _MAX_CONFIG_BYTES:
            raise ProductionConfigError("production configuration is too large")
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProductionConfigError(
                "production configuration is not valid UTF-8 JSON",
                details={"exceptionType": type(exc).__name__},
            ) from exc
        return cls.from_mapping(
            raw,
            source_path=source.resolve(strict=True),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        source_path: str | os.PathLike[str],
    ) -> "ProductionConfig":
        source = Path(source_path).expanduser()
        if not source.is_absolute():
            source = Path(os.path.abspath(source))
        base_directory = source.parent
        root = _object(
            value,
            field="config",
            allowed={
                "schemaVersion",
                "mode",
                "offline",
                "paths",
                "models",
                "executables",
                "runtime",
                "speaker",
                "pdf",
            },
            required={
                "schemaVersion",
                "mode",
                "offline",
                "paths",
                "models",
                "executables",
            },
        )
        schema_version = _nonempty_text(
            root["schemaVersion"], field="schemaVersion"
        )
        if schema_version != PRODUCTION_CONFIG_SCHEMA_VERSION:
            raise ProductionConfigError(
                "unsupported production configuration schema",
                details={
                    "expectedSchemaVersion": PRODUCTION_CONFIG_SCHEMA_VERSION,
                    "receivedSchemaVersion": schema_version,
                },
            )
        mode = _choice(
            root["mode"],
            field="mode",
            choices={PRODUCTION_MODE},
        )
        offline = _boolean(root["offline"], field="offline")
        if not offline:
            raise ProductionConfigError(
                "production mode requires offline=true",
                code="NETWORK_POLICY_INVALID",
            )

        raw_paths = _object(
            root["paths"],
            field="paths",
            allowed={
                "allowedInputRoots",
                "allowedOutputRoot",
                "cacheRoot",
            },
            required={
                "allowedInputRoots",
                "allowedOutputRoot",
                "cacheRoot",
            },
        )
        raw_input_roots = raw_paths["allowedInputRoots"]
        if (
            not isinstance(raw_input_roots, list)
            or not raw_input_roots
            or len(raw_input_roots) > 32
        ):
            raise ProductionConfigError(
                "paths.allowedInputRoots must contain between 1 and 32 paths"
            )
        input_roots = _unique_paths(
            [
                _local_path(
                    item,
                    field=f"paths.allowedInputRoots[{index}]",
                    base_directory=base_directory,
                )
                for index, item in enumerate(raw_input_roots)
            ],
            field="paths.allowedInputRoots",
        )
        paths = ProductionPaths(
            allowed_input_roots=input_roots,
            allowed_output_root=_local_path(
                raw_paths["allowedOutputRoot"],
                field="paths.allowedOutputRoot",
                base_directory=base_directory,
            ),
            cache_root=_local_path(
                raw_paths["cacheRoot"],
                field="paths.cacheRoot",
                base_directory=base_directory,
            ),
        )

        raw_models = _object(
            root["models"],
            field="models",
            allowed={
                "funasrVad",
                "qwen3Asr",
                "qwen3ForcedAligner",
                "camPlus",
                "eres2netV2",
                "pyannote",
            },
            required={
                "funasrVad",
                "qwen3Asr",
                "camPlus",
                "eres2netV2",
            },
        )
        models = ProductionModels(
            funasr_vad=_local_path(
                raw_models["funasrVad"],
                field="models.funasrVad",
                base_directory=base_directory,
            ),
            qwen3_asr=_local_path(
                raw_models["qwen3Asr"],
                field="models.qwen3Asr",
                base_directory=base_directory,
            ),
            qwen3_forced_aligner=_optional_local_path(
                raw_models.get("qwen3ForcedAligner"),
                field="models.qwen3ForcedAligner",
                base_directory=base_directory,
            ),
            cam_plus=_local_path(
                raw_models["camPlus"],
                field="models.camPlus",
                base_directory=base_directory,
            ),
            eres2net_v2=_local_path(
                raw_models["eres2netV2"],
                field="models.eres2netV2",
                base_directory=base_directory,
            ),
            pyannote=_optional_local_path(
                raw_models.get("pyannote"),
                field="models.pyannote",
                base_directory=base_directory,
            ),
        )

        raw_executables = _object(
            root["executables"],
            field="executables",
            allowed={
                "ffmpeg",
                "java",
                "pdfRendererJar",
                "pyannotePython",
            },
            required={"ffmpeg", "java", "pdfRendererJar"},
        )
        raw_pyannote_python = raw_executables.get("pyannotePython")
        executables = ProductionExecutables(
            ffmpeg=_executable_text(
                raw_executables["ffmpeg"],
                field="executables.ffmpeg",
                base_directory=base_directory,
            ),
            java=_executable_text(
                raw_executables["java"],
                field="executables.java",
                base_directory=base_directory,
            ),
            pdf_renderer_jar=_local_path(
                raw_executables["pdfRendererJar"],
                field="executables.pdfRendererJar",
                base_directory=base_directory,
            ),
            pyannote_python=(
                None
                if raw_pyannote_python is None
                else _executable_text(
                    raw_pyannote_python,
                    field="executables.pyannotePython",
                    base_directory=base_directory,
                )
            ),
        )

        raw_runtime = _object(
            root.get("runtime", {}),
            field="runtime",
            allowed={
                "maxWorkers",
                "maxPendingJobs",
                "maxLineBytes",
                "modelResidency",
                "heartbeatIntervalSeconds",
                "vadDevice",
                "asrDevice",
                "asrDtype",
                "camPlusDevice",
                "eres2netDevice",
                "pyannoteDevice",
                "strictStartupPreflight",
            },
        )
        runtime = ProductionRuntime(
            max_workers=_integer(
                raw_runtime.get("maxWorkers", 1),
                field="runtime.maxWorkers",
                minimum=1,
                maximum=16,
            ),
            max_pending_jobs=_integer(
                raw_runtime.get("maxPendingJobs", 1),
                field="runtime.maxPendingJobs",
                minimum=0,
                maximum=64,
            ),
            max_line_bytes=_integer(
                raw_runtime.get("maxLineBytes", 1024 * 1024),
                field="runtime.maxLineBytes",
                minimum=4096,
                maximum=16 * 1024 * 1024,
            ),
            model_residency=_choice(
                raw_runtime.get("modelResidency", "stage"),
                field="runtime.modelResidency",
                choices={"stage", "worker"},
            ),
            heartbeat_interval_seconds=_number(
                raw_runtime.get("heartbeatIntervalSeconds", 15.0),
                field="runtime.heartbeatIntervalSeconds",
                minimum=0.25,
                maximum=300.0,
            ),
            vad_device=_nonempty_text(
                raw_runtime.get("vadDevice", "cpu"),
                field="runtime.vadDevice",
            ),
            asr_device=_nonempty_text(
                raw_runtime.get("asrDevice", "cuda:0"),
                field="runtime.asrDevice",
            ),
            asr_dtype=_choice(
                raw_runtime.get("asrDtype", "bfloat16"),
                field="runtime.asrDtype",
                choices={"bfloat16", "float16", "float32"},
            ),
            cam_plus_device=_nonempty_text(
                raw_runtime.get("camPlusDevice", "cuda:0"),
                field="runtime.camPlusDevice",
            ),
            eres2net_device=_nonempty_text(
                raw_runtime.get("eres2netDevice", "gpu"),
                field="runtime.eres2netDevice",
            ),
            pyannote_device=_nonempty_text(
                raw_runtime.get("pyannoteDevice", "cuda"),
                field="runtime.pyannoteDevice",
            ),
            strict_startup_preflight=_boolean(
                raw_runtime.get("strictStartupPreflight", True),
                field="runtime.strictStartupPreflight",
            ),
        )

        raw_speaker = _object(
            root.get("speaker", {}),
            field="speaker",
            allowed={
                "normalizationProfile",
                "clusterSimilarityThreshold",
                "lowMarginThreshold",
                "highMarginThreshold",
                "outlierScoreThreshold",
                "maxAutoSpeakers",
                "maxClusteringWindows",
                "maxClusteringWorkItems",
                "kmeansIterations",
                "maxBatchSize",
                "maxSecondaryFraction",
                "maxCountUncertaintyCandidates",
                "autoCountConfidenceThreshold",
                "countStabilityRuns",
                "eigengapLandmarkLimit",
                "maxLanguageWindowMs",
                "languageSplitSearchMs",
                "eres2netDecisionMargin",
                "pyannoteMappingMarginThreshold",
                "pyannotePrimaryDominanceThreshold",
                "pyannoteMode",
                "localLlmMode",
                "localLlmModel",
            },
        )
        raw_max_auto_speakers = raw_speaker.get("maxAutoSpeakers")
        max_auto_speakers = (
            None
            if raw_max_auto_speakers is None
            else _integer(
                raw_max_auto_speakers,
                field="speaker.maxAutoSpeakers",
                minimum=1,
                maximum=4096,
            )
        )
        raw_pyannote_mode = _nonempty_text(
            raw_speaker.get("pyannoteMode", "disabled"),
            field="speaker.pyannoteMode",
        )
        if raw_pyannote_mode == "audit":
            raise ProductionConfigError(
                "speaker.pyannoteMode='audit' is no longer supported",
                code="PRODUCTION_CONFIG_MIGRATION_REQUIRED",
                details={
                    "field": "speaker.pyannoteMode",
                    "removedValue": "audit",
                    "supportedValues": ["disabled", "fallback"],
                    "migration": (
                        "Use 'fallback' for unresolved-segment review or "
                        "'disabled' to omit pyannote."
                    ),
                },
            )
        speaker = ProductionSpeakerPolicy(
            normalization_profile=_nonempty_text(
                raw_speaker.get(
                    "normalizationProfile", "mono-16khz-f32-v1"
                ),
                field="speaker.normalizationProfile",
            ),
            cluster_similarity_threshold=_number(
                raw_speaker.get("clusterSimilarityThreshold", 0.72),
                field="speaker.clusterSimilarityThreshold",
                minimum=-1.0,
                maximum=1.0,
            ),
            low_margin_threshold=_number(
                raw_speaker.get("lowMarginThreshold", 0.18),
                field="speaker.lowMarginThreshold",
                minimum=0.0,
                maximum=1.0,
            ),
            high_margin_threshold=_number(
                raw_speaker.get("highMarginThreshold", 0.35),
                field="speaker.highMarginThreshold",
                minimum=0.0,
                maximum=1.0,
            ),
            outlier_score_threshold=_number(
                raw_speaker.get("outlierScoreThreshold", 0.30),
                field="speaker.outlierScoreThreshold",
                minimum=-1.0,
                maximum=1.0,
            ),
            max_auto_speakers=max_auto_speakers,
            max_clustering_windows=_integer(
                raw_speaker.get("maxClusteringWindows", 100_000),
                field="speaker.maxClusteringWindows",
                minimum=1,
                maximum=10_000_000,
            ),
            max_clustering_work_items=_integer(
                raw_speaker.get("maxClusteringWorkItems", 5_000_000),
                field="speaker.maxClusteringWorkItems",
                minimum=1,
                maximum=1_000_000_000,
            ),
            kmeans_iterations=_integer(
                raw_speaker.get("kmeansIterations", 30),
                field="speaker.kmeansIterations",
                minimum=1,
                maximum=500,
            ),
            max_batch_size=_integer(
                raw_speaker.get("maxBatchSize", 32),
                field="speaker.maxBatchSize",
                minimum=1,
                maximum=512,
            ),
            max_secondary_fraction=_number(
                raw_speaker.get("maxSecondaryFraction", 0.25),
                field="speaker.maxSecondaryFraction",
                minimum=0.0,
                maximum=1.0,
                minimum_exclusive=True,
            ),
            max_count_uncertainty_candidates=_integer(
                raw_speaker.get("maxCountUncertaintyCandidates", 8),
                field="speaker.maxCountUncertaintyCandidates",
                minimum=1,
                maximum=512,
            ),
            auto_count_confidence_threshold=_number(
                raw_speaker.get("autoCountConfidenceThreshold", 0.75),
                field="speaker.autoCountConfidenceThreshold",
                minimum=0.0,
                maximum=1.0,
            ),
            count_stability_runs=_integer(
                raw_speaker.get("countStabilityRuns", 3),
                field="speaker.countStabilityRuns",
                minimum=1,
                maximum=64,
            ),
            eigengap_landmark_limit=_integer(
                raw_speaker.get("eigengapLandmarkLimit", 256),
                field="speaker.eigengapLandmarkLimit",
                minimum=2,
                maximum=4096,
            ),
            max_language_window_ms=_integer(
                raw_speaker.get("maxLanguageWindowMs", 12_000),
                field="speaker.maxLanguageWindowMs",
                minimum=1_000,
                maximum=120_000,
            ),
            language_split_search_ms=_integer(
                raw_speaker.get("languageSplitSearchMs", 1_000),
                field="speaker.languageSplitSearchMs",
                minimum=1,
                maximum=10_000,
            ),
            eres2net_decision_margin=_number(
                raw_speaker.get("eres2netDecisionMargin", 0.05),
                field="speaker.eres2netDecisionMargin",
                minimum=0.0,
                maximum=1.0,
            ),
            pyannote_mapping_margin_threshold=_number(
                raw_speaker.get("pyannoteMappingMarginThreshold", 0.05),
                field="speaker.pyannoteMappingMarginThreshold",
                minimum=0.0,
                maximum=1.0,
            ),
            pyannote_primary_dominance_threshold=_number(
                raw_speaker.get("pyannotePrimaryDominanceThreshold", 0.60),
                field="speaker.pyannotePrimaryDominanceThreshold",
                minimum=0.5,
                maximum=1.0,
            ),
            pyannote_mode=_choice(
                raw_pyannote_mode,
                field="speaker.pyannoteMode",
                choices={"disabled", "fallback"},
            ),
            local_llm_mode=_choice(
                raw_speaker.get("localLlmMode", "disabled"),
                field="speaker.localLlmMode",
                choices={"disabled", "suggestion-only"},
            ),
            local_llm_model=_choice(
                raw_speaker.get("localLlmModel", "qwen3.5:9b"),
                field="speaker.localLlmModel",
                choices={"qwen3.5:4b", "qwen3.5:9b"},
            ),
        )
        if speaker.high_margin_threshold <= speaker.low_margin_threshold:
            raise ProductionConfigError(
                "speaker.highMarginThreshold must exceed lowMarginThreshold"
            )
        if speaker.max_secondary_fraction >= 1.0:
            raise ProductionConfigError(
                "speaker.maxSecondaryFraction must be less than 1.0"
            )
        if (
            speaker.max_language_window_ms
            <= speaker.language_split_search_ms + 700
        ):
            raise ProductionConfigError(
                "speaker.maxLanguageWindowMs must exceed "
                "languageSplitSearchMs by more than 700 ms"
            )
        if speaker.pyannote_mode != "disabled" and models.pyannote is None:
            raise ProductionConfigError(
                "models.pyannote is required when pyannote is enabled"
            )
        if (
            speaker.pyannote_mode != "disabled"
            and executables.pyannote_python is None
        ):
            raise ProductionConfigError(
                "executables.pyannotePython is required when pyannote is enabled"
            )
        if speaker.pyannote_mode == "disabled" and models.pyannote is not None:
            raise ProductionConfigError(
                "models.pyannote must be null or omitted when pyannote is disabled"
            )

        raw_pdf = _object(
            root.get("pdf", {}),
            field="pdf",
            allowed={
                "minimumScore",
                "maxRounds",
                "captureDpi",
                "marginMm",
                "preferredFont",
                "templateId",
                "timeoutSeconds",
            },
        )
        raw_preferred_font = raw_pdf.get("preferredFont")
        preferred_font = (
            None
            if raw_preferred_font is None
            else _choice(
                raw_preferred_font,
                field="pdf.preferredFont",
                choices={"LXGW WenKai"},
            )
        )
        pdf = ProductionPdfPolicy(
            minimum_score=_number(
                raw_pdf.get("minimumScore", 85.0),
                field="pdf.minimumScore",
                minimum=85.0,
                maximum=100.0,
            ),
            max_rounds=_integer(
                raw_pdf.get("maxRounds", 5),
                field="pdf.maxRounds",
                minimum=1,
                maximum=5,
            ),
            capture_dpi=_integer(
                raw_pdf.get("captureDpi", 144),
                field="pdf.captureDpi",
                minimum=96,
                maximum=300,
            ),
            margin_mm=_number(
                raw_pdf.get("marginMm", 14.0),
                field="pdf.marginMm",
                minimum=8.0,
                maximum=30.0,
            ),
            preferred_font=preferred_font,
            template_id=_nonempty_text(
                raw_pdf.get("templateId", "mts-cute-transcript-v1"),
                field="pdf.templateId",
            ),
            timeout_seconds=_number(
                raw_pdf.get("timeoutSeconds", 300.0),
                field="pdf.timeoutSeconds",
                minimum=1.0,
                maximum=3600.0,
            ),
        )
        return cls(
            schema_version=schema_version,
            mode=mode,
            offline=offline,
            paths=paths,
            models=models,
            executables=executables,
            runtime=runtime,
            speaker=speaker,
            pdf=pdf,
            source_path=source,
        )

    def with_runtime_overrides(
        self,
        *,
        allowed_input_roots: Sequence[str | os.PathLike[str]] | None = None,
        allowed_output_root: str | os.PathLike[str] | None = None,
        max_workers: int | None = None,
    ) -> "ProductionConfig":
        """Apply explicit CLI roots without weakening any other policy."""

        base_directory = self.source_path.parent
        roots = (
            _unique_paths(
                [
                    _local_path(
                        item,
                        field=f"cli.inputRoot[{index}]",
                        base_directory=base_directory,
                    )
                    for index, item in enumerate(allowed_input_roots)
                ],
                field="cli.inputRoot",
            )
            if allowed_input_roots
            else self.paths.allowed_input_roots
        )
        output_root = (
            _local_path(
                allowed_output_root,
                field="cli.outputRoot",
                base_directory=base_directory,
            )
            if allowed_output_root is not None
            else self.paths.allowed_output_root
        )
        workers = (
            _integer(
                max_workers,
                field="cli.maxWorkers",
                minimum=1,
                maximum=16,
            )
            if max_workers is not None
            else self.runtime.max_workers
        )
        return ProductionConfig(
            schema_version=self.schema_version,
            mode=self.mode,
            offline=self.offline,
            paths=ProductionPaths(
                allowed_input_roots=roots,
                allowed_output_root=output_root,
                cache_root=self.paths.cache_root,
            ),
            models=self.models,
            executables=self.executables,
            runtime=ProductionRuntime(
                max_workers=workers,
                max_pending_jobs=self.runtime.max_pending_jobs,
                max_line_bytes=self.runtime.max_line_bytes,
                model_residency=self.runtime.model_residency,
                heartbeat_interval_seconds=(
                    self.runtime.heartbeat_interval_seconds
                ),
                vad_device=self.runtime.vad_device,
                asr_device=self.runtime.asr_device,
                asr_dtype=self.runtime.asr_dtype,
                cam_plus_device=self.runtime.cam_plus_device,
                eres2net_device=self.runtime.eres2net_device,
                pyannote_device=self.runtime.pyannote_device,
                strict_startup_preflight=self.runtime.strict_startup_preflight,
            ),
            speaker=self.speaker,
            pdf=self.pdf,
            source_path=self.source_path,
        )

    def fingerprint(self) -> str:
        """Return a non-reversible identifier for the effective configuration."""

        material = {
            "schemaVersion": self.schema_version,
            "mode": self.mode,
            "offline": self.offline,
            "inputRootCount": len(self.paths.allowed_input_roots),
            "pyannoteMode": self.speaker.pyannote_mode,
            "maxWorkers": self.runtime.max_workers,
            "modelResidency": self.runtime.model_residency,
            "heartbeatIntervalSeconds": (
                self.runtime.heartbeat_interval_seconds
            ),
            "maxAutoSpeakers": self.speaker.max_auto_speakers,
            "maxClusteringWindows": self.speaker.max_clustering_windows,
            "maxClusteringWorkItems": (
                self.speaker.max_clustering_work_items
            ),
            "countStabilityRuns": self.speaker.count_stability_runs,
            "eigengapLandmarkLimit": (
                self.speaker.eigengap_landmark_limit
            ),
            "maxLanguageWindowMs": self.speaker.max_language_window_ms,
            "languageSplitSearchMs": self.speaker.language_split_search_ms,
            "pyannoteMappingMarginThreshold": (
                self.speaker.pyannote_mapping_margin_threshold
            ),
            "pyannotePrimaryDominanceThreshold": (
                self.speaker.pyannote_primary_dominance_threshold
            ),
            "pdfTemplate": self.pdf.template_id,
            "pyannotePython": (
                hashlib.sha256(
                    self.executables.pyannote_python.encode("utf-8")
                ).hexdigest()
                if self.executables.pyannote_python is not None
                else None
            ),
            "paths": [
                hashlib.sha256(str(path).encode("utf-8")).hexdigest()
                for path in (
                    *self.paths.allowed_input_roots,
                    self.paths.allowed_output_root,
                    self.paths.cache_root,
                    self.models.funasr_vad,
                    self.models.qwen3_asr,
                    self.models.cam_plus,
                    self.models.eres2net_v2,
                    self.executables.pdf_renderer_jar,
                )
            ],
        }
        return hashlib.sha256(
            json.dumps(
                material,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class PreflightCheck:
    check_id: str
    category: str
    required: bool
    passed: bool
    reason_code: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "category": self.category,
            "required": self.required,
            "status": "passed" if self.passed else "failed",
            "reasonCode": self.reason_code,
        }


@dataclass(frozen=True)
class ProductionPreflightReport:
    config_fingerprint: str
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed or not check.required for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PRODUCTION_CONFIG_SCHEMA_VERSION,
            "mode": PRODUCTION_MODE,
            "offline": True,
            "status": "passed" if self.passed else "failed",
            "configFingerprint": self.config_fingerprint,
            "checks": [check.as_dict() for check in self.checks],
        }

    def raise_if_failed(self) -> None:
        failed = [
            check.check_id
            for check in self.checks
            if check.required and not check.passed
        ]
        if failed:
            raise ProductionConfigError(
                "production preflight failed",
                code="PRODUCTION_PREFLIGHT_FAILED",
                details={"failedChecks": failed},
            )


RuntimeProbe = Callable[[str], bool]


def _resolve_executable(value: str) -> str | None:
    candidate = Path(value).expanduser()
    contains_separator = any(
        separator and separator in value for separator in (os.sep, os.altsep)
    )
    if candidate.is_absolute() or contains_separator:
        return str(candidate.resolve(strict=True)) if candidate.is_file() else None
    return shutil.which(value)


def _probe_command(command: Sequence[str], *, timeout_seconds: float = 10.0) -> bool:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_seconds,
            env=offline_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _probe_runtime_import(module: str) -> bool:
    code = (
        "import importlib;"
        f"importlib.import_module({module!r});"
        "print('ok')"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_RUNTIME_IMPORT_TIMEOUT_SECONDS,
            env=offline_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == b"ok"


def _probe_python_import(python_executable: str, module: str) -> bool:
    resolved = _resolve_executable(python_executable)
    if resolved is None:
        return False
    code = (
        "import importlib;"
        f"importlib.import_module({module!r});"
        "print('ok')"
    )
    try:
        completed = subprocess.run(
            [resolved, "-I", "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_RUNTIME_IMPORT_TIMEOUT_SECONDS,
            env=offline_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and completed.stdout.strip() == b"ok"


def _jar_has_required_engines(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = tuple(name.casefold() for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return (
        any("openhtmltopdf" in name for name in names)
        and any("pdfbox" in name for name in names)
        and "meta-inf/manifest.mf" in names
    )


def _directory_writable(path: Path) -> bool:
    probe: Path | None = None
    descriptor: int | None = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".mts-write-probe-",
            dir=path,
        )
        probe = Path(name)
        handle = os.fdopen(descriptor, "wb")
        descriptor = None
        with handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
        probe.unlink()
        probe = None
    except OSError:
        return False
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if probe is not None:
            try:
                probe.unlink()
            except OSError:
                pass
    return True


def offline_environment(
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a subprocess environment that forbids implicit model downloads."""

    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYANNOTE_METRICS_ENABLED": "0",
            "DO_NOT_TRACK": "1",
            "FUNASR_DISABLE_UPDATE": "1",
        }
    )
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def apply_offline_environment() -> None:
    """Pin the current worker process to the same deny-network model policy."""

    os.environ.update(offline_environment())


def run_production_preflight(
    config: ProductionConfig,
    *,
    probe_runtime_imports: bool = True,
    runtime_probe: RuntimeProbe | None = None,
    probe_executables: bool = True,
) -> ProductionPreflightReport:
    """Validate all required local components without loading model weights."""

    runtime_probe = runtime_probe or _probe_runtime_import
    checks: list[PreflightCheck] = []

    def add(
        check_id: str,
        category: str,
        required: bool,
        passed: bool,
        reason_code: str,
    ) -> None:
        checks.append(
            PreflightCheck(
                check_id=check_id,
                category=category,
                required=required,
                passed=bool(passed),
                reason_code=reason_code,
            )
        )

    add(
        "network-policy",
        "policy",
        True,
        config.offline and config.mode == PRODUCTION_MODE,
        "OFFLINE_ENFORCED",
    )
    for index, root in enumerate(config.paths.allowed_input_roots):
        add(
            f"input-root-{index + 1}",
            "filesystem",
            True,
            root.is_dir(),
            "LOCAL_INPUT_ROOT",
        )
    add(
        "output-root",
        "filesystem",
        True,
        _directory_writable(config.paths.allowed_output_root),
        "LOCAL_OUTPUT_ROOT_WRITABLE",
    )
    add(
        "cache-root",
        "filesystem",
        True,
        _directory_writable(config.paths.cache_root),
        "LOCAL_CACHE_ROOT_WRITABLE",
    )

    model_paths: list[tuple[str, Path, bool]] = [
        ("funasr-vad-model", config.models.funasr_vad, True),
        ("qwen3-asr-model", config.models.qwen3_asr, True),
        ("cam-plus-model", config.models.cam_plus, True),
        ("eres2net-v2-model", config.models.eres2net_v2, True),
    ]
    if config.models.qwen3_forced_aligner is not None:
        model_paths.append(
            (
                "qwen3-forced-aligner-model",
                config.models.qwen3_forced_aligner,
                True,
            )
        )
    if config.speaker.pyannote_mode != "disabled":
        assert config.models.pyannote is not None
        model_paths.append(("pyannote-model", config.models.pyannote, True))
    for check_id, path, required in model_paths:
        add(
            check_id,
            "model",
            required,
            path.is_dir(),
            "LOCAL_MODEL_DIRECTORY",
        )

    ffmpeg = _resolve_executable(config.executables.ffmpeg)
    java = _resolve_executable(config.executables.java)
    pyannote_python = (
        _resolve_executable(config.executables.pyannote_python)
        if config.executables.pyannote_python is not None
        else None
    )
    add(
        "ffmpeg-executable",
        "executable",
        True,
        bool(ffmpeg)
        and (
            not probe_executables
            or _probe_command([str(ffmpeg), "-version"])
        ),
        "FFMPEG_EXECUTABLE_AVAILABLE",
    )
    add(
        "java-executable",
        "executable",
        True,
        bool(java)
        and (
            not probe_executables
            or _probe_command([str(java), "-version"])
        ),
        "JAVA_EXECUTABLE_AVAILABLE",
    )
    if config.speaker.pyannote_mode != "disabled":
        add(
            "pyannote-python-executable",
            "executable",
            True,
            bool(pyannote_python)
            and (
                not probe_executables
                or _probe_command([str(pyannote_python), "--version"])
            ),
            "ISOLATED_PYANNOTE_PYTHON_AVAILABLE",
        )
    jar = config.executables.pdf_renderer_jar
    add(
        "java-pdf-renderer",
        "renderer",
        True,
        jar.is_file()
        and jar.suffix.casefold() == ".jar"
        and _jar_has_required_engines(jar),
        "OPENHTMLTOPDF_PDFBOX_BUNDLE",
    )

    runtime_modules: list[tuple[str, str, bool]] = [
        ("runtime-funasr", "funasr", True),
        ("runtime-qwen-asr", "qwen_asr", True),
        ("runtime-modelscope", "modelscope.pipelines", True),
        ("runtime-simplejson", "simplejson", True),
        ("runtime-soundfile", "soundfile", True),
        ("runtime-numpy", "numpy", True),
    ]
    for check_id, module, required in runtime_modules:
        passed = True if not probe_runtime_imports else runtime_probe(module)
        add(
            check_id,
            "runtime",
            required,
            passed,
            "ISOLATED_IMPORT_PROBE",
        )
    if config.speaker.pyannote_mode != "disabled":
        assert config.executables.pyannote_python is not None
        passed = (
            True
            if not probe_runtime_imports
            else (
                runtime_probe("pyannote.audio")
                if runtime_probe is not _probe_runtime_import
                else _probe_python_import(
                    config.executables.pyannote_python,
                    "pyannote.audio",
                )
            )
        )
        add(
            "runtime-pyannote",
            "runtime",
            True,
            passed,
            "ISOLATED_PYANNOTE_IMPORT_PROBE",
        )
    return ProductionPreflightReport(
        config_fingerprint=config.fingerprint(),
        checks=tuple(checks),
    )


def production_diagnostics(
    config: ProductionConfig,
    report: ProductionPreflightReport,
) -> dict[str, Any]:
    """Return a path-free production topology and preflight summary."""

    stages: list[dict[str, Any]] = [
        {
            "stage": "boundary",
            "component": "FunASR VAD",
            "scope": "full-corpus-batched",
        },
        {
            "stage": "asr",
            "component": "Qwen3-ASR-1.7B",
            "scope": "speech-windows-batched",
        },
        {
            "stage": "voiceprint-primary",
            "component": "CAM++",
            "scope": "full-corpus-batched",
        },
        {
            "stage": "voiceprint-secondary",
            "component": "ERes2NetV2",
            "scope": "difficult-segments-only",
            "maximumFraction": config.speaker.max_secondary_fraction,
        },
    ]
    if config.speaker.pyannote_mode != "disabled":
        stages.append(
            {
                "stage": "diarization-fallback",
                "component": "pyannote community-1",
                "scope": "unresolved-segments-only",
                "mode": config.speaker.pyannote_mode,
                "telemetry": "disabled",
            }
        )
    return {
        "schemaVersion": PRODUCTION_CONFIG_SCHEMA_VERSION,
        "mode": PRODUCTION_MODE,
        "offline": True,
        "status": "passed" if report.passed else "failed",
        "configFingerprint": config.fingerprint(),
        "speakerCardinality": {
            "modes": ["auto", "manual", "hybrid"],
            "fixedFivePersonLimit": False,
            "configuredAutoResourceLimit": config.speaker.max_auto_speakers,
            "resourceBudgets": {
                "maxClusteringWindows": (
                    config.speaker.max_clustering_windows
                ),
                "maxClusteringWorkItems": (
                    config.speaker.max_clustering_work_items
                ),
                "countStabilityRuns": config.speaker.count_stability_runs,
                "eigengapLandmarkLimit": (
                    config.speaker.eigengap_landmark_limit
                ),
            },
        },
        "localLlm": {
            "mode": config.speaker.local_llm_mode,
            "model": config.speaker.local_llm_model,
            "autoApply": False,
        },
        "runtime": {
            "modelResidency": config.runtime.model_residency,
            "heartbeatIntervalSeconds": (
                config.runtime.heartbeat_interval_seconds
            ),
        },
        "pdf": {
            "engine": "OpenHTMLtoPDF 1.0.10 + PDFBox 2.0.30",
            "qualityMinimum": config.pdf.minimum_score,
            "maximumRepairRounds": config.pdf.max_rounds,
        },
        "stages": stages,
        "preflight": report.as_dict(),
    }


__all__ = [
    "PRODUCTION_CONFIG_SCHEMA_VERSION",
    "PRODUCTION_MODE",
    "PreflightCheck",
    "ProductionConfig",
    "ProductionConfigError",
    "ProductionExecutables",
    "ProductionModels",
    "ProductionPaths",
    "ProductionPdfPolicy",
    "ProductionPreflightReport",
    "ProductionRuntime",
    "ProductionSpeakerPolicy",
    "apply_offline_environment",
    "offline_environment",
    "production_diagnostics",
    "run_production_preflight",
]
