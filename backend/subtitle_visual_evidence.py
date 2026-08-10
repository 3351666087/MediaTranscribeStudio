"""Auditable representative-frame evidence for subtitle visual QA.

This module is an execution boundary between immutable local media and
``backend.subtitle_visual_qa``.  It extracts deterministic frame candidates
from an already-rendered subtitle artifact and its unsubtitled source,
collects pixel evidence, and emits an exact ``subtitle-visual-qa-request``
inside a canonical, SHA-256-bound evidence envelope.

Security and truthfulness rules are intentionally strict:

* FFmpeg and FFprobe executable paths are mandatory constructor arguments.
* Every process receives an argv vector and runs with ``shell=False``.
* Source and rendered media are only opened by this module for binary reads.
* Temporary files live beside the rendered media and are always cleaned.
* Process time, stdout, stderr, frame size, and analyzed pixels are bounded.
* Pillow is loaded lazily; missing image support fails closed.
* Font installation/embedding claims require real artifact paths whose hashes
  are calculated here.  Raw caller-supplied claim hashes are not accepted.
* Word timings are never generated or interpolated.  Authentic caller
  evidence is passed through byte-for-byte at the JSON value level.

The analyzer and process runner are protocols so unit tests and higher-quality
native analyzers can be injected without requiring a real FFmpeg installation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from backend.subtitle_visual_qa import (
    SUBTITLE_VISUAL_QA_REQUEST_KIND,
    SUBTITLE_VISUAL_QA_SCHEMA_VERSION,
    contrast_ratio,
    evaluate_subtitle_visual_qa,
)


SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION = "1.0.0"
SUBTITLE_RENDER_EVIDENCE_REQUEST_KIND = "subtitle-render-evidence-request"
SUBTITLE_RENDER_EVIDENCE_RESULT_KIND = "subtitle-render-evidence-result"
SUBTITLE_RENDER_EVIDENCE_STRATEGY = (
    "cue-interior-contrast-candidates-v1"
)
CONTRAST_MATTE_TRANSFORM_NAME = "canonical-ass-contrast-mattes"
CONTRAST_MATTE_TRANSFORM_VERSION = "1.0.0"
CONTRAST_BACKGROUND_STRATEGY = "background-only-ass-v1"
GLYPH_FILL_MATTE_STRATEGY = "opaque-white-fill-ass-v1"
GLYPH_CORE_COMPONENT_STRATEGY = "glyph-fill-core-component-q05-v2"
TINY_COMPONENT_MAX_PIXELS = 6
TINY_COMPONENT_MINIMUM_ALPHA = 0.50

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_CODEPOINT = re.compile(r"^U\+[0-9A-F]{4,6}$")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_AUTHENTIC_WORD_TIMING_SOURCES = frozenset(
    {
        "forced-aligner",
        "native-word-timestamps",
        "human-authored",
    }
)
_FONT_METHODS = frozenset(
    {
        "not-provided",
        "renderer-glyph-map",
        "font-cmap-and-shaping",
        "directwrite-enumeration",
        "fontconfig-scan",
        "libass-render-report",
        "manual-pixel-audit",
        "container-font-inspection",
    }
)
_POSITIVE_FONT_METHODS = _FONT_METHODS - {"not-provided"}
_INSTALLATION_STATUSES = frozenset({"verified-installed"})
_EMBEDDING_STATUSES = frozenset(
    {"verified-embedded", "verified-not-embedded"}
)


class SubtitleVisualEvidenceErrorCode(str, Enum):
    INVALID_REQUEST = "invalid-request"
    INVALID_PATH = "invalid-path"
    INPUT_MISSING = "input-missing"
    PATH_ALIAS = "path-alias"
    TOOL_UNAVAILABLE = "tool-unavailable"
    PROCESS_TIMEOUT = "process-timeout"
    PROCESS_OUTPUT_LIMIT = "process-output-limit"
    PROCESS_FAILED = "process-failed"
    PROBE_FAILED = "probe-failed"
    VIDEO_REQUIRED = "video-required"
    VIDEO_MISMATCH = "video-mismatch"
    FRAME_TIME_UNAVAILABLE = "frame-time-unavailable"
    FRAME_EXTRACTION_FAILED = "frame-extraction-failed"
    FRAME_OUTPUT_LIMIT = "frame-output-limit"
    ANALYZER_UNAVAILABLE = "analyzer-unavailable"
    ANALYSIS_INCOMPLETE = "analysis-incomplete"
    FONT_EVIDENCE_INVALID = "font-evidence-invalid"
    SOURCE_CHANGED = "source-changed"
    CLEANUP_FAILED = "cleanup-failed"


class SubtitleVisualEvidenceError(RuntimeError):
    """Structured, fail-closed collection failure."""

    def __init__(
        self,
        code: SubtitleVisualEvidenceErrorCode,
        message: str,
        *,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class EvidenceProcessLimits:
    """Strict limits for each local FFmpeg/FFprobe invocation."""

    timeout_seconds: float = 45.0
    max_stdout_bytes: int = 8 * 1024 * 1024
    max_stderr_bytes: int = 4 * 1024 * 1024
    poll_interval_seconds: float = 0.025

    def __post_init__(self) -> None:
        if not 0.05 <= self.timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be between 0.05 and 600")
        if not 1_024 <= self.max_stdout_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_stdout_bytes must be 1 KiB..64 MiB")
        if not 1_024 <= self.max_stderr_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_stderr_bytes must be 1 KiB..64 MiB")
        if not 0.005 <= self.poll_interval_seconds <= 1.0:
            raise ValueError(
                "poll_interval_seconds must be between 0.005 and 1"
            )


@dataclass(frozen=True)
class SubtitleVisualEvidencePolicy:
    """Resource and image-analysis bounds for one collection."""

    process_limits: EvidenceProcessLimits = field(
        default_factory=EvidenceProcessLimits
    )
    maximum_frame_bytes: int = 128 * 1024 * 1024
    maximum_overlay_bytes: int = 64 * 1024 * 1024
    hash_chunk_bytes: int = 4 * 1024 * 1024
    maximum_frames: int = 100_000

    def __post_init__(self) -> None:
        if not 1_024 <= self.maximum_frame_bytes <= 512 * 1024 * 1024:
            raise ValueError("maximum_frame_bytes must be 1 KiB..512 MiB")
        if not 1_024 <= self.maximum_overlay_bytes <= 256 * 1024 * 1024:
            raise ValueError(
                "maximum_overlay_bytes must be 1 KiB..256 MiB"
            )
        if not 4_096 <= self.hash_chunk_bytes <= 64 * 1024 * 1024:
            raise ValueError("hash_chunk_bytes must be 4 KiB..64 MiB")
        if not 1 <= self.maximum_frames <= 100_000:
            raise ValueError("maximum_frames must be 1..100000")


@dataclass(frozen=True)
class EvidenceProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class EvidenceRunner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        limits: EvidenceProcessLimits,
        cwd: Path,
    ) -> EvidenceProcessResult: ...


class BoundedEvidenceRunner:
    """Run a trusted argv vector with bounded capture and ``shell=False``."""

    def run(
        self,
        command: Sequence[str],
        *,
        limits: EvidenceProcessLimits,
        cwd: Path,
    ) -> EvidenceProcessResult:
        argv = tuple(str(part) for part in command)
        if not argv or any(not part or "\x00" in part for part in argv):
            raise ValueError(
                "command must contain non-empty, non-NUL argv entries"
            )
        resolved_cwd = Path(cwd).resolve(strict=True)
        if not resolved_cwd.is_dir():
            raise ValueError("cwd must be an existing directory")

        creationflags = 0
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))

        capture_root = Path(
            tempfile.mkdtemp(
                prefix=".mts-evidence-process-",
                dir=resolved_cwd,
            )
        )
        stdout_path = capture_root / "stdout.bin"
        stderr_path = capture_root / "stderr.bin"
        started = time.monotonic()
        try:
            try:
                with stdout_path.open("wb") as stdout_handle, stderr_path.open(
                    "wb"
                ) as stderr_handle:
                    process = subprocess.Popen(  # noqa: S603
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        cwd=str(resolved_cwd),
                        shell=False,
                        close_fds=True,
                        creationflags=creationflags,
                    )
                    while process.poll() is None:
                        if (
                            time.monotonic() - started
                            > limits.timeout_seconds
                        ):
                            self._stop(process)
                            raise SubtitleVisualEvidenceError(
                                SubtitleVisualEvidenceErrorCode.PROCESS_TIMEOUT,
                                "Local media evidence command timed out.",
                            )
                        if (
                            self._size(stdout_path)
                            > limits.max_stdout_bytes
                            or self._size(stderr_path)
                            > limits.max_stderr_bytes
                        ):
                            self._stop(process)
                            raise SubtitleVisualEvidenceError(
                                SubtitleVisualEvidenceErrorCode.PROCESS_OUTPUT_LIMIT,
                                "Local media evidence command exceeded its output limit.",
                            )
                        time.sleep(limits.poll_interval_seconds)
                    returncode = int(process.returncode or 0)
            except FileNotFoundError as exc:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.TOOL_UNAVAILABLE,
                    f"Local media tool is unavailable: {argv[0]}",
                ) from exc
            except OSError as exc:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.TOOL_UNAVAILABLE,
                    f"Local media tool could not be started: {argv[0]}",
                    detail=str(exc),
                ) from exc

            return EvidenceProcessResult(
                returncode=returncode,
                stdout=self._read_bounded(
                    stdout_path, limits.max_stdout_bytes
                ),
                stderr=self._read_bounded(
                    stderr_path, limits.max_stderr_bytes
                ),
            )
        finally:
            cleanup_error = _remove_tree(capture_root)
            if cleanup_error is not None:
                active_error = sys.exc_info()[1]
                if active_error is not None:
                    active_error.add_note(cleanup_error)
                else:
                    raise SubtitleVisualEvidenceError(
                        SubtitleVisualEvidenceErrorCode.CLEANUP_FAILED,
                        "Process capture directory could not be removed.",
                        detail=cleanup_error,
                    )

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    @staticmethod
    def _size(path: Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    @staticmethod
    def _read_bounded(path: Path, maximum: int) -> bytes:
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
        if len(payload) > maximum:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.PROCESS_OUTPUT_LIMIT,
                "Local media evidence command exceeded its output limit.",
            )
        return payload


@dataclass(frozen=True)
class ComponentDescriptor:
    name: str
    version: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or len(self.name) > 160
            or self.name != self.name.strip()
            or "\x00" in self.name
        ):
            raise ValueError("component name must be trimmed non-empty text")
        if (
            not isinstance(self.version, str)
            or not self.version
            or len(self.version) > 320
            or self.version != self.version.strip()
            or "\x00" in self.version
        ):
            raise ValueError(
                "component version must be trimmed non-empty text"
            )
        if (
            not isinstance(self.configuration_sha256, str)
            or not _SHA256.fullmatch(self.configuration_sha256)
        ):
            raise ValueError(
                "component configuration must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "configurationSha256": self.configuration_sha256,
        }


@dataclass(frozen=True)
class ContrastObservation:
    background_class: str
    foreground_rgb: str
    background_rgb: str
    foreground_pixel_count: int
    background_pixel_count: int


@dataclass(frozen=True)
class CueFrameObservation:
    cue_id: str
    bounds: Mapping[str, int]
    ink_bounds: Mapping[str, int]
    clipped_pixel_count: int
    edge_touching_pixel_count: int
    overflow_detected: bool
    contrast_samples: tuple[ContrastObservation, ...]
    contrast_diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FrameObservation:
    width_px: int
    height_px: int
    instances: tuple[CueFrameObservation, ...]


class FrameAnalyzer(Protocol):
    @property
    def descriptor(self) -> ComponentDescriptor: ...

    def analyze(
        self,
        *,
        rendered_frame_path: Path,
        source_frame_path: Path,
        contrast_background_frame_path: Path | None,
        glyph_fill_matte_frame_path: Path | None,
        frame_id: str,
        timestamp_ms: int,
        width_px: int,
        height_px: int,
        cues: Sequence[Mapping[str, Any]],
        contrast_policy: Mapping[str, Any],
    ) -> FrameObservation: ...


@dataclass(frozen=True)
class PillowAnalysisPolicy:
    difference_threshold: int = 24
    strong_difference_threshold: int = 48
    compression_noise_percentile: float = 0.98
    compression_noise_margin: int = 8
    strong_difference_margin: int = 12
    minimum_component_pixels: int = 4
    minimum_strong_ink_pixels: int = 4
    minimum_ink_pixels: int = 8
    minimum_paired_core_pixels: int = 32
    glyph_core_minimum_alpha: float = 0.80
    glyph_core_quantile: float = 0.75
    component_contrast_quantile: float = 0.05
    tiny_component_max_pixels: int = TINY_COMPONENT_MAX_PIXELS
    tiny_component_minimum_alpha: float = TINY_COMPONENT_MINIMUM_ALPHA
    search_margin_px: int = 8
    maximum_analyzed_pixels_per_cue: int = 20_000_000
    maximum_candidate_pixels_per_cue: int = 2_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.difference_threshold <= 255:
            raise ValueError("difference_threshold must be 1..255")
        if not 1 <= self.strong_difference_threshold <= 255:
            raise ValueError(
                "strong_difference_threshold must be 1..255"
            )
        if not 0.5 <= self.compression_noise_percentile <= 1.0:
            raise ValueError(
                "compression_noise_percentile must be 0.5..1.0"
            )
        if not 0 <= self.compression_noise_margin <= 64:
            raise ValueError("compression_noise_margin must be 0..64")
        if not 0 <= self.strong_difference_margin <= 64:
            raise ValueError("strong_difference_margin must be 0..64")
        if not 1 <= self.minimum_component_pixels <= 1_000_000:
            raise ValueError(
                "minimum_component_pixels must be 1..1000000"
            )
        if not 1 <= self.minimum_strong_ink_pixels <= 1_000_000:
            raise ValueError(
                "minimum_strong_ink_pixels must be 1..1000000"
            )
        if not 1 <= self.minimum_ink_pixels <= 1_000_000:
            raise ValueError("minimum_ink_pixels must be 1..1000000")
        if not 1 <= self.minimum_paired_core_pixels <= 1_000_000:
            raise ValueError(
                "minimum_paired_core_pixels must be 1..1000000"
            )
        if not 0.0 < self.glyph_core_minimum_alpha <= 1.0:
            raise ValueError("glyph_core_minimum_alpha must be in (0, 1]")
        if not 0.0 < self.glyph_core_quantile <= 1.0:
            raise ValueError("glyph_core_quantile must be in (0, 1]")
        if not 0.0 < self.component_contrast_quantile <= 1.0:
            raise ValueError(
                "component_contrast_quantile must be in (0, 1]"
            )
        if self.tiny_component_max_pixels != TINY_COMPONENT_MAX_PIXELS:
            raise ValueError(
                "tiny_component_max_pixels is fixed at 6 for the v2 "
                "glyph-matte strategy"
            )
        if (
            self.tiny_component_minimum_alpha
            != TINY_COMPONENT_MINIMUM_ALPHA
        ):
            raise ValueError(
                "tiny_component_minimum_alpha is fixed at 0.50 for the v2 "
                "glyph-matte strategy"
            )
        if not 0 <= self.search_margin_px <= 128:
            raise ValueError("search_margin_px must be 0..128")
        if not (
            1_024
            <= self.maximum_analyzed_pixels_per_cue
            <= 200_000_000
        ):
            raise ValueError(
                "maximum_analyzed_pixels_per_cue must be 1024..200000000"
            )
        if not (
            1_024
            <= self.maximum_candidate_pixels_per_cue
            <= self.maximum_analyzed_pixels_per_cue
        ):
            raise ValueError(
                "maximum_candidate_pixels_per_cue must be 1024.."
                "maximum_analyzed_pixels_per_cue"
            )


class PillowFrameAnalyzer:
    """Compare rendered and source PNGs without asserting font truth."""

    def __init__(
        self,
        *,
        policy: PillowAnalysisPolicy | None = None,
        image_module_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.policy = policy or PillowAnalysisPolicy()
        self._image_module_loader = (
            image_module_loader or self._default_image_loader
        )
        configuration = {
            "differenceThreshold": self.policy.difference_threshold,
            "strongDifferenceThreshold": (
                self.policy.strong_difference_threshold
            ),
            "compressionNoisePercentile": (
                self.policy.compression_noise_percentile
            ),
            "compressionNoiseMargin": (
                self.policy.compression_noise_margin
            ),
            "strongDifferenceMargin": (
                self.policy.strong_difference_margin
            ),
            "minimumComponentPixels": (
                self.policy.minimum_component_pixels
            ),
            "minimumStrongInkPixels": (
                self.policy.minimum_strong_ink_pixels
            ),
            "minimumInkPixels": self.policy.minimum_ink_pixels,
            "minimumPairedCorePixels": (
                self.policy.minimum_paired_core_pixels
            ),
            "glyphCoreMinimumAlpha": self.policy.glyph_core_minimum_alpha,
            "glyphCoreQuantile": self.policy.glyph_core_quantile,
            "componentContrastQuantile": (
                self.policy.component_contrast_quantile
            ),
            "tinyComponentMaxPixels": self.policy.tiny_component_max_pixels,
            "tinyComponentMinimumAlpha": (
                self.policy.tiny_component_minimum_alpha
            ),
            "searchMarginPx": self.policy.search_margin_px,
            "maximumAnalyzedPixelsPerCue": (
                self.policy.maximum_analyzed_pixels_per_cue
            ),
            "maximumCandidatePixelsPerCue": (
                self.policy.maximum_candidate_pixels_per_cue
            ),
        }
        self._descriptor = ComponentDescriptor(
            name="pillow-source-render-and-glyph-matte",
            version="2.1.0",
            configuration_sha256=deterministic_sha256(configuration),
        )

    @property
    def descriptor(self) -> ComponentDescriptor:
        return self._descriptor

    @staticmethod
    def _default_image_loader() -> Any:
        try:
            return importlib.import_module("PIL.Image")
        except (ImportError, ModuleNotFoundError) as exc:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYZER_UNAVAILABLE,
                "Pillow image analysis is unavailable; evidence collection cannot continue.",
            ) from exc

    def analyze(
        self,
        *,
        rendered_frame_path: Path,
        source_frame_path: Path,
        contrast_background_frame_path: Path | None = None,
        glyph_fill_matte_frame_path: Path | None = None,
        frame_id: str,
        timestamp_ms: int,
        width_px: int,
        height_px: int,
        cues: Sequence[Mapping[str, Any]],
        contrast_policy: Mapping[str, Any],
    ) -> FrameObservation:
        del timestamp_ms
        image_module = self._image_module_loader()
        try:
            with image_module.open(rendered_frame_path) as rendered_image:
                rendered_image.load()
                rendered = rendered_image.convert("RGB")
            with image_module.open(source_frame_path) as source_image:
                source_image.load()
                source = source_image.convert("RGB")
            contrast_background = None
            glyph_fill_matte = None
            if contrast_background_frame_path is not None:
                with image_module.open(
                    contrast_background_frame_path
                ) as background_image:
                    background_image.load()
                    contrast_background = background_image.convert("RGB")
            if glyph_fill_matte_frame_path is not None:
                with image_module.open(
                    glyph_fill_matte_frame_path
                ) as matte_image:
                    matte_image.load()
                    glyph_fill_matte = matte_image.convert("RGB")
        except SubtitleVisualEvidenceError:
            raise
        except Exception as exc:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                f"Representative frame image could not be decoded: {frame_id}",
                detail=str(exc),
            ) from exc

        if rendered.size != source.size:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.VIDEO_MISMATCH,
                "Rendered and source representative frames differ in size.",
            )
        if (contrast_background is None) != (glyph_fill_matte is None):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Contrast background and glyph-fill matte frames must be "
                "supplied together.",
            )
        if contrast_background is not None and (
            contrast_background.size != rendered.size
            or glyph_fill_matte is None
            or glyph_fill_matte.size != rendered.size
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.VIDEO_MISMATCH,
                "Contrast evidence frames differ from the rendered frame size.",
            )
        if rendered.size != (width_px, height_px):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.VIDEO_MISMATCH,
                "Decoded frame size contradicts FFprobe evidence.",
            )

        rendered_pixels = rendered.load()
        source_pixels = source.load()
        contrast_background_pixels = (
            contrast_background.load()
            if contrast_background is not None
            else None
        )
        glyph_fill_matte_pixels = (
            glyph_fill_matte.load() if glyph_fill_matte is not None else None
        )
        instances = tuple(
            self._analyze_cue(
                rendered_pixels=rendered_pixels,
                source_pixels=source_pixels,
                contrast_background_pixels=contrast_background_pixels,
                glyph_fill_matte_pixels=glyph_fill_matte_pixels,
                width_px=width_px,
                height_px=height_px,
                cue=cue,
                contrast_policy=contrast_policy,
            )
            for cue in cues
        )
        return FrameObservation(
            width_px=width_px,
            height_px=height_px,
            instances=instances,
        )

    def _analyze_cue(
        self,
        *,
        rendered_pixels: Any,
        source_pixels: Any,
        contrast_background_pixels: Any | None,
        glyph_fill_matte_pixels: Any | None,
        width_px: int,
        height_px: int,
        cue: Mapping[str, Any],
        contrast_policy: Mapping[str, Any],
    ) -> CueFrameObservation:
        bounds = _parse_rect(
            cue["bounds"],
            f"cue {cue['cueId']!r} bounds",
            frame_width=width_px,
            frame_height=height_px,
        )
        left = max(0, bounds["x"] - self.policy.search_margin_px)
        top = max(0, bounds["y"] - self.policy.search_margin_px)
        right = min(
            width_px,
            bounds["x"]
            + bounds["width"]
            + self.policy.search_margin_px,
        )
        bottom = min(
            height_px,
            bounds["y"]
            + bounds["height"]
            + self.policy.search_margin_px,
        )
        area = (right - left) * (bottom - top)
        if area > self.policy.maximum_analyzed_pixels_per_cue:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                f"Cue {cue['cueId']!r} exceeds the bounded analysis area.",
            )

        bound_right = bounds["x"] + bounds["width"]
        bound_bottom = bounds["y"] + bounds["height"]

        # Burn-in delivery normally re-encodes the video. Comparing that frame
        # directly with a source decode produces low-amplitude codec noise even
        # where no subtitle exists. Estimate a local noise floor from the
        # bounded margin outside the declared subtitle rectangle instead of
        # treating every changed pixel as subtitle ink.
        noise_histogram = [0] * 256
        noise_samples = 0
        for y_coord in range(top, bottom):
            for x_coord in range(left, right):
                if (
                    bounds["x"] <= x_coord < bound_right
                    and bounds["y"] <= y_coord < bound_bottom
                ):
                    continue
                rendered_rgb = rendered_pixels[x_coord, y_coord]
                source_rgb = source_pixels[x_coord, y_coord]
                difference = max(
                    abs(rendered_rgb[channel] - source_rgb[channel])
                    for channel in range(3)
                )
                noise_histogram[difference] += 1
                noise_samples += 1

        noise_floor = _histogram_percentile(
            noise_histogram,
            noise_samples,
            self.policy.compression_noise_percentile,
        )
        threshold = min(
            255,
            max(
                self.policy.difference_threshold,
                noise_floor + self.policy.compression_noise_margin,
            ),
        )
        strong_threshold = min(
            255,
            max(
                self.policy.strong_difference_threshold,
                threshold + self.policy.strong_difference_margin,
            ),
        )

        candidates: dict[
            tuple[int, int],
            tuple[
                int,
                int,
                tuple[int, int, int],
                tuple[int, int, int],
                int,
            ],
        ] = {}
        for y_coord in range(top, bottom):
            for x_coord in range(left, right):
                rendered_rgb = rendered_pixels[x_coord, y_coord]
                source_rgb = source_pixels[x_coord, y_coord]
                difference = max(
                    abs(rendered_rgb[channel] - source_rgb[channel])
                    for channel in range(3)
                )
                if difference < threshold:
                    continue
                observation = (
                    x_coord,
                    y_coord,
                    rendered_rgb,
                    source_rgb,
                    difference,
                )
                candidates[(x_coord, y_coord)] = observation
                if (
                    len(candidates)
                    > self.policy.maximum_candidate_pixels_per_cue
                ):
                    raise SubtitleVisualEvidenceError(
                        SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                        f"Cue {cue['cueId']!r} exceeds the bounded "
                        "candidate-pixel allowance.",
                    )

        changed = self._coherent_ink_components(
            candidates,
            strong_threshold=strong_threshold,
        )
        inside = [
            item
            for item in changed
            if (
                bounds["x"] <= item[0] < bound_right
                and bounds["y"] <= item[1] < bound_bottom
            )
        ]

        if len(inside) < self.policy.minimum_ink_pixels:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                f"Cue {cue['cueId']!r} has insufficient visible-ink evidence.",
            )

        x_values = [item[0] for item in changed]
        y_values = [item[1] for item in changed]
        ink_bounds = {
            "x": min(x_values),
            "y": min(y_values),
            "width": max(x_values) - min(x_values) + 1,
            "height": max(y_values) - min(y_values) + 1,
        }
        outside_count = len(changed) - len(inside)
        frame_edge_count = sum(
            1
            for x_coord, y_coord, _rendered, _source, _difference in changed
            if (
                x_coord == 0
                or y_coord == 0
                or x_coord == width_px - 1
                or y_coord == height_px - 1
            )
        )
        touches_declared_edge = any(
            (
                x_coord == bounds["x"]
                or y_coord == bounds["y"]
                or x_coord == bound_right - 1
                or y_coord == bound_bottom - 1
            )
            for x_coord, y_coord, _rendered, _source, _difference in inside
        )

        dark_maximum = _require_number(
            contrast_policy["darkMaximumLuminance"],
            "contrast.darkMaximumLuminance",
            minimum=0.0,
            maximum=1.0,
        )
        light_minimum = _require_number(
            contrast_policy["lightMinimumLuminance"],
            "contrast.lightMinimumLuminance",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            contrast_background_pixels is not None
            and glyph_fill_matte_pixels is not None
        ):
            samples, diagnostics = self._glyph_matte_contrast_samples(
                rendered_pixels=rendered_pixels,
                source_pixels=source_pixels,
                contrast_background_pixels=contrast_background_pixels,
                glyph_fill_matte_pixels=glyph_fill_matte_pixels,
                bounds=bounds,
                dark_maximum=dark_maximum,
                light_minimum=light_minimum,
                cue_id=str(cue["cueId"]),
            )
        else:
            samples = self._legacy_contrast_samples(
                inside=inside,
                dark_maximum=dark_maximum,
                light_minimum=light_minimum,
            )
            diagnostics = None

        return CueFrameObservation(
            cue_id=str(cue["cueId"]),
            bounds=bounds,
            ink_bounds=ink_bounds,
            clipped_pixel_count=frame_edge_count,
            edge_touching_pixel_count=frame_edge_count,
            overflow_detected=outside_count > 0 or touches_declared_edge,
            contrast_samples=tuple(samples),
            contrast_diagnostics=diagnostics,
        )

    def _legacy_contrast_samples(
        self,
        *,
        inside: Sequence[
            tuple[
                int,
                int,
                tuple[int, int, int],
                tuple[int, int, int],
                int,
            ]
        ],
        dark_maximum: float,
        light_minimum: float,
    ) -> list[ContrastObservation]:
        buckets: dict[
            str,
            list[
                tuple[
                    int,
                    int,
                    tuple[int, int, int],
                    tuple[int, int, int],
                    int,
                ]
            ],
        ] = {"dark": [], "light": []}
        for item in inside:
            luminance = _relative_luminance_tuple(item[3])
            if luminance <= dark_maximum:
                buckets["dark"].append(item)
            elif luminance >= light_minimum:
                buckets["light"].append(item)

        samples: list[ContrastObservation] = []
        for background_class in ("dark", "light"):
            pixels = buckets[background_class]
            if not pixels:
                continue
            worst = min(
                pixels,
                key=lambda item: (
                    contrast_ratio(
                        _rgb_hex(item[2]),
                        _rgb_hex(item[3]),
                    ),
                    item[1],
                    item[0],
                ),
            )
            samples.append(
                ContrastObservation(
                    background_class=background_class,
                    foreground_rgb=_rgb_hex(worst[2]),
                    background_rgb=_rgb_hex(worst[3]),
                    foreground_pixel_count=len(pixels),
                    background_pixel_count=len(pixels),
                )
            )

        return samples

    def _glyph_matte_contrast_samples(
        self,
        *,
        rendered_pixels: Any,
        source_pixels: Any,
        contrast_background_pixels: Any,
        glyph_fill_matte_pixels: Any,
        bounds: Mapping[str, int],
        dark_maximum: float,
        light_minimum: float,
        cue_id: str,
    ) -> tuple[list[ContrastObservation], dict[str, Any]]:
        right = bounds["x"] + bounds["width"]
        bottom = bounds["y"] + bounds["height"]
        alpha_pixels: dict[tuple[int, int], int] = {}
        for y_coord in range(bounds["y"], bottom):
            for x_coord in range(bounds["x"], right):
                matte_rgb = glyph_fill_matte_pixels[x_coord, y_coord]
                alpha_byte = max(matte_rgb)
                if alpha_byte > 0:
                    alpha_pixels[(x_coord, y_coord)] = alpha_byte
        if len(alpha_pixels) > self.policy.maximum_candidate_pixels_per_cue:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                f"Cue {cue_id!r} exceeds the bounded glyph-matte allowance.",
            )

        components = self._coordinate_components(alpha_pixels)
        meaningful = [
            component
            for component in components
            if len(component) >= self.policy.minimum_component_pixels
        ]
        if not meaningful:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                f"Cue {cue_id!r} has no meaningful glyph-fill components.",
            )

        alpha_q75 = _nearest_rank_quantile(
            tuple(alpha_pixels.values()),
            self.policy.glyph_core_quantile,
        )
        core_alpha = max(
            round(self.policy.glyph_core_minimum_alpha * 255),
            alpha_q75,
        )
        component_rows: list[dict[str, Any]] = []
        tiny_component_fallback_count = 0
        paired_by_class: dict[str, list[dict[str, Any]]] = {
            "dark": [],
            "light": [],
        }
        all_core_coordinates: list[tuple[int, int]] = []
        underlying_classes: set[str] = set()
        effective_classes: set[str] = set()

        for component_index, component in enumerate(meaningful):
            core_alpha_threshold = core_alpha
            core_coordinates = sorted(
                (
                    coordinate
                    for coordinate in component
                    if alpha_pixels[coordinate] >= core_alpha
                ),
                key=lambda coordinate: (coordinate[1], coordinate[0]),
            )
            tiny_component_fallback = False
            if not core_coordinates and (
                len(component) <= self.policy.tiny_component_max_pixels
            ):
                # Small punctuation can be entirely antialiased by libass at
                # low raster resolutions. Keep it auditable without allowing
                # a larger, low-alpha component to bypass the global core
                # threshold: use the component-local Q75 only for <=6 pixels,
                # with a minimum alpha floor of 0.50.
                component_alpha_q75 = _nearest_rank_quantile(
                    tuple(alpha_pixels[coordinate] for coordinate in component),
                    self.policy.glyph_core_quantile,
                )
                local_core_alpha = max(
                    round(
                        self.policy.tiny_component_minimum_alpha * 255
                    ),
                    component_alpha_q75,
                )
                fallback_coordinates = sorted(
                    (
                        coordinate
                        for coordinate in component
                        if alpha_pixels[coordinate] >= local_core_alpha
                    ),
                    key=lambda coordinate: (coordinate[1], coordinate[0]),
                )
                if fallback_coordinates:
                    core_coordinates = fallback_coordinates
                    core_alpha_threshold = local_core_alpha
                    tiny_component_fallback = True
                    tiny_component_fallback_count += 1
            if not core_coordinates:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                    f"Cue {cue_id!r} has an uncovered glyph component.",
                )
            all_core_coordinates.extend(core_coordinates)
            row = {
                "componentIndex": component_index,
                "pixelCount": len(component),
                "corePixelCount": len(core_coordinates),
                "coreAlphaThreshold": core_alpha_threshold,
                "tinyComponentFallback": tiny_component_fallback,
                "bounds": _coordinate_bounds(component),
            }
            component_rows.append(row)
            for x_coord, y_coord in core_coordinates:
                foreground_rgb = rendered_pixels[x_coord, y_coord]
                background_rgb = contrast_background_pixels[x_coord, y_coord]
                source_rgb = source_pixels[x_coord, y_coord]
                effective_class = _luminance_class(
                    background_rgb,
                    dark_maximum=dark_maximum,
                    light_minimum=light_minimum,
                )
                underlying_class = _luminance_class(
                    source_rgb,
                    dark_maximum=dark_maximum,
                    light_minimum=light_minimum,
                )
                if underlying_class is not None:
                    underlying_classes.add(underlying_class)
                if effective_class is None:
                    continue
                effective_classes.add(effective_class)
                paired_by_class[effective_class].append(
                    {
                        "componentIndex": component_index,
                        "x": x_coord,
                        "y": y_coord,
                        "foregroundRgb": foreground_rgb,
                        "backgroundRgb": background_rgb,
                        "ratio": contrast_ratio(
                            _rgb_hex(foreground_rgb),
                            _rgb_hex(background_rgb),
                        ),
                    }
                )

        if len(all_core_coordinates) < self.policy.minimum_paired_core_pixels:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                f"Cue {cue_id!r} has fewer than "
                f"{self.policy.minimum_paired_core_pixels} paired glyph-core "
                "pixels.",
            )

        samples: list[ContrastObservation] = []
        for background_class in ("dark", "light"):
            pixels = paired_by_class[background_class]
            if not pixels:
                continue
            by_component: dict[int, list[dict[str, Any]]] = {}
            for pixel in pixels:
                by_component.setdefault(pixel["componentIndex"], []).append(
                    pixel
                )
            component_q05 = [
                _quantile_pixel(
                    component_pixels,
                    self.policy.component_contrast_quantile,
                )
                for _index, component_pixels in sorted(by_component.items())
            ]
            representative = min(
                component_q05,
                key=lambda item: (
                    item["ratio"],
                    item["componentIndex"],
                    item["y"],
                    item["x"],
                ),
            )
            samples.append(
                ContrastObservation(
                    background_class=background_class,
                    foreground_rgb=_rgb_hex(representative["foregroundRgb"]),
                    background_rgb=_rgb_hex(representative["backgroundRgb"]),
                    foreground_pixel_count=len(pixels),
                    background_pixel_count=len(pixels),
                )
            )

        coverage_core = [
            {
                "bounds": row["bounds"],
                "corePixelCount": row["corePixelCount"],
                "pixelCount": row["pixelCount"],
                "coreAlphaThreshold": row["coreAlphaThreshold"],
                "tinyComponentFallback": row["tinyComponentFallback"],
            }
            for row in component_rows
        ]
        diagnostics = {
            "strategy": GLYPH_CORE_COMPONENT_STRATEGY,
            "glyphComponentCount": len(meaningful),
            "glyphComponentsCovered": len(component_rows),
            "pairedCorePixelCount": len(all_core_coordinates),
            "glyphCoreAlphaThreshold": round(core_alpha / 255.0, 6),
            "glyphNonzeroAlphaQuantile75": round(alpha_q75 / 255.0, 6),
            "componentContrastQuantile": (
                self.policy.component_contrast_quantile
            ),
            "tinyComponentFallbackCount": tiny_component_fallback_count,
            "tinyComponentMaxPixels": self.policy.tiny_component_max_pixels,
            "tinyComponentMinimumAlpha": (
                self.policy.tiny_component_minimum_alpha
            ),
            "minimumComponentCorePixelCount": min(
                row["corePixelCount"] for row in component_rows
            ),
            "effectiveBackgroundClasses": sorted(effective_classes),
            "underlyingSceneClasses": sorted(underlying_classes),
            "componentCoverageSha256": deterministic_sha256(coverage_core),
        }
        return samples, diagnostics

    @staticmethod
    def _coordinate_components(
        pixels: Mapping[tuple[int, int], int],
    ) -> list[list[tuple[int, int]]]:
        remaining = set(pixels)
        components: list[list[tuple[int, int]]] = []
        while remaining:
            start = min(remaining, key=lambda item: (item[1], item[0]))
            remaining.remove(start)
            stack = [start]
            component = [start]
            while stack:
                x_coord, y_coord = stack.pop()
                for y_offset in (-1, 0, 1):
                    for x_offset in (-1, 0, 1):
                        if x_offset == 0 and y_offset == 0:
                            continue
                        neighbor = (
                            x_coord + x_offset,
                            y_coord + y_offset,
                        )
                        if neighbor not in remaining:
                            continue
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
            component.sort(key=lambda item: (item[1], item[0]))
            components.append(component)
        components.sort(
            key=lambda component: (
                component[0][1],
                component[0][0],
                len(component),
            )
        )
        return components

    def _coherent_ink_components(
        self,
        candidates: Mapping[
            tuple[int, int],
            tuple[
                int,
                int,
                tuple[int, int, int],
                tuple[int, int, int],
                int,
            ],
        ],
        *,
        strong_threshold: int,
    ) -> list[
        tuple[
            int,
            int,
            tuple[int, int, int],
            tuple[int, int, int],
            int,
        ]
    ]:
        """Discard isolated/block-noise deltas that cannot prove subtitle ink."""

        remaining = dict(candidates)
        accepted: list[
            tuple[
                int,
                int,
                tuple[int, int, int],
                tuple[int, int, int],
                int,
            ]
        ] = []
        while remaining:
            start, first = remaining.popitem()
            stack = [start]
            component = [first]
            strong_count = int(first[4] >= strong_threshold)
            while stack:
                x_coord, y_coord = stack.pop()
                for y_offset in (-1, 0, 1):
                    for x_offset in (-1, 0, 1):
                        if x_offset == 0 and y_offset == 0:
                            continue
                        neighbor = (
                            x_coord + x_offset,
                            y_coord + y_offset,
                        )
                        observation = remaining.pop(neighbor, None)
                        if observation is None:
                            continue
                        component.append(observation)
                        strong_count += int(
                            observation[4] >= strong_threshold
                        )
                        stack.append(neighbor)
            if (
                len(component) >= self.policy.minimum_component_pixels
                and strong_count
                >= self.policy.minimum_strong_ink_pixels
            ):
                accepted.extend(component)
        accepted.sort(key=lambda item: (item[1], item[0]))
        return accepted


@dataclass(frozen=True)
class VerifiedFontClaim:
    """A positive font claim backed by local artifacts, not raw hashes."""

    status: str
    verification_method: str
    evidence_artifact_path: str | Path
    font_artifact_path: str | Path


@dataclass(frozen=True)
class FontEvidenceObservation:
    """Font resolution/glyph evidence returned by a trusted provider."""

    resolved_family: str | None
    resolution_verified: bool
    glyph_coverage_verified: bool
    verification_method: str
    evidence_artifact_path: str | Path | None
    covered_renderable_code_points: int
    missing_code_points: tuple[str, ...] = ()
    tofu_glyph_count: int = 0
    installation: VerifiedFontClaim | None = None
    embedding: VerifiedFontClaim | None = None


class FontEvidenceProvider(Protocol):
    @property
    def descriptor(self) -> ComponentDescriptor: ...

    def collect(
        self,
        *,
        cue: Mapping[str, Any],
        frame_id: str,
        timestamp_ms: int,
        rendered_frame_path: Path,
        source_frame_path: Path,
    ) -> FontEvidenceObservation | None: ...


class NoFontEvidenceProvider:
    """Honest default: make no font resolution or installation claims."""

    _descriptor = ComponentDescriptor(
        name="font-evidence-not-provided",
        version="1.0.0",
        configuration_sha256=hashlib.sha256(
            b'{"claimPolicy":"no-implicit-font-claims"}'
        ).hexdigest(),
    )

    @property
    def descriptor(self) -> ComponentDescriptor:
        return self._descriptor

    def collect(
        self,
        *,
        cue: Mapping[str, Any],
        frame_id: str,
        timestamp_ms: int,
        rendered_frame_path: Path,
        source_frame_path: Path,
    ) -> None:
        del cue, frame_id, timestamp_ms, rendered_frame_path, source_frame_path
        return None


@dataclass(frozen=True)
class SubtitleRenderEvidenceResult:
    """Canonical evidence envelope plus a directly consumable QA request."""

    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.payload))

    @property
    def qa_request(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.payload["qaRequest"]))

    def canonical_json(self) -> str:
        return canonical_json(self.payload)


@dataclass(frozen=True)
class _RenderEvidenceContext:
    """Path-private inputs that are bound into public evidence hashes."""

    delivery_mode: str | None
    overlay_path: Path | None
    overlay_snapshot: Mapping[str, Any] | None
    overlay_sha256: str | None
    contrast_background_ass: bytes | None
    glyph_fill_matte_ass: bytes | None
    contrast_matte_evidence: Mapping[str, Any] | None
    delivery_receipt_sha256: str | None
    effective_render_configuration_sha256: str


@dataclass(frozen=True)
class _StagedAssOverlays:
    canonical: Path
    contrast_background: Path
    glyph_fill_matte: Path


def default_subtitle_visual_evidence_sampling() -> dict[str, Any]:
    return {
        "strategy": SUBTITLE_RENDER_EVIDENCE_STRATEGY,
        "candidateFractions": [0.25, 0.5, 0.75],
    }


def canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Canonical UTF-8 JSON with sorted keys and no non-finite numbers."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
            "Value is not canonical JSON data.",
            detail=str(exc),
        ) from exc


def deterministic_sha256(
    value: Mapping[str, Any] | Sequence[Any],
) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def verify_evidence_artifact_hash(payload: Mapping[str, Any]) -> bool:
    """Verify the result self-hash over the envelope excluding that field."""

    value = dict(payload)
    claimed = value.pop("evidenceArtifactSha256", None)
    return isinstance(claimed, str) and claimed == deterministic_sha256(value)


class SubtitleVisualEvidenceCollector:
    """Collect canonical visual evidence from immutable local media."""

    def __init__(
        self,
        *,
        ffmpeg_path: str | Path,
        ffprobe_path: str | Path,
        runner: EvidenceRunner | None = None,
        analyzer: FrameAnalyzer | None = None,
        font_evidence_provider: FontEvidenceProvider | None = None,
        policy: SubtitleVisualEvidencePolicy | None = None,
    ) -> None:
        self.ffmpeg_path = _canonical_tool_path(
            ffmpeg_path, label="ffmpeg_path"
        )
        self.ffprobe_path = _canonical_tool_path(
            ffprobe_path, label="ffprobe_path"
        )
        self.runner = runner or BoundedEvidenceRunner()
        self.analyzer = analyzer or PillowFrameAnalyzer()
        self.font_evidence_provider = (
            font_evidence_provider or NoFontEvidenceProvider()
        )
        self.policy = policy or SubtitleVisualEvidencePolicy()

    def collect(
        self,
        request: Mapping[str, Any],
        *,
        canonical_ass_overlay_path: str | Path | None = None,
        canonical_ass_overlay_sha256: str | None = None,
        delivery_receipt: Mapping[str, Any] | Any | None = None,
    ) -> SubtitleRenderEvidenceResult:
        normalized = _parse_request(request)
        source_path = _canonical_media_path(
            normalized["sourceMediaPath"],
            label="sourceMediaPath",
        )
        rendered_path = _canonical_media_path(
            normalized["renderedMediaPath"],
            label="renderedMediaPath",
        )
        if os.path.samefile(source_path, rendered_path):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.PATH_ALIAS,
                "Source and rendered media must be distinct files.",
            )

        source_before = self._snapshot(source_path)
        rendered_before = self._snapshot(rendered_path)
        render_context = self._prepare_render_context(
            normalized=normalized,
            source_path=source_path,
            rendered_path=rendered_path,
            source_snapshot=source_before,
            rendered_snapshot=rendered_before,
            canonical_ass_overlay_path=canonical_ass_overlay_path,
            canonical_ass_overlay_sha256=canonical_ass_overlay_sha256,
            delivery_receipt=delivery_receipt,
        )
        request_binding = copy.deepcopy(normalized)
        request_binding["sourceMediaPath"] = str(source_path)
        request_binding["renderedMediaPath"] = str(rendered_path)
        request_binding["renderArtifact"][
            "renderConfigurationSha256"
        ] = render_context.effective_render_configuration_sha256
        request_binding["injectedTools"] = {
            "ffmpegPath": str(self.ffmpeg_path),
            "ffprobePath": str(self.ffprobe_path),
        }
        request_binding["components"] = {
            "frameAnalyzer": self.analyzer.descriptor.to_dict(),
            "fontEvidenceProvider": (
                self.font_evidence_provider.descriptor.to_dict()
            ),
        }
        request_binding["externalEvidenceBindings"] = {
            "deliveryMode": render_context.delivery_mode,
            "canonicalAssOverlaySha256": render_context.overlay_sha256,
            "deliveryReceiptSha256": (
                render_context.delivery_receipt_sha256
            ),
            "contrastMatte": render_context.contrast_matte_evidence,
            "privatePathsIncluded": False,
        }
        request_sha256 = deterministic_sha256(request_binding)

        temporary_root = self._make_same_directory_temp(rendered_path)
        try:
            staged_ass_overlays = self._stage_ass_overlay(
                render_context,
                temporary_root=temporary_root,
            )
            tools = {
                "ffmpeg": self._tool_evidence(
                    self.ffmpeg_path, temporary_root
                ),
                "ffprobe": self._tool_evidence(
                    self.ffprobe_path, temporary_root
                ),
            }
            source_video = self._probe_video(source_path, temporary_root)
            rendered_video = self._probe_video(
                rendered_path, temporary_root
            )
            self._validate_video_pair(source_video, rendered_video)

            candidates = self._resolve_candidates(
                rendered_path=rendered_path,
                cues=normalized["cues"],
                fractions=normalized["sampling"]["candidateFractions"],
                temporary_root=temporary_root,
            )
            if len({item["frameId"] for item in candidates}) > (
                self.policy.maximum_frames
            ):
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
                    "Representative-frame count exceeds the configured maximum.",
                )

            frame_payloads, frame_records, bindings = self._collect_frames(
                source_path=source_path,
                rendered_path=rendered_path,
                video=rendered_video,
                candidates=candidates,
                cues=normalized["cues"],
                contrast_policy=normalized["policy"]["contrast"],
                temporary_root=temporary_root,
                staged_ass_overlays=staged_ass_overlays,
                render_context=render_context,
            )
            selection_core = {
                "strategy": normalized["sampling"]["strategy"],
                "candidateFractions": normalized["sampling"][
                    "candidateFractions"
                ],
                "sourceMediaSha256": source_before["sha256"],
                "renderArtifactSha256": rendered_before["sha256"],
                "candidates": candidates,
                "frames": frame_records,
            }
            selection_sha256 = deterministic_sha256(selection_core)

            qa_request = {
                "kind": SUBTITLE_VISUAL_QA_REQUEST_KIND,
                "schemaVersion": SUBTITLE_VISUAL_QA_SCHEMA_VERSION,
                "analysisId": normalized["collectionId"],
                "renderArtifact": {
                    "artifactSha256": rendered_before["sha256"],
                    "renderer": normalized["renderArtifact"]["renderer"],
                    "rendererVersion": normalized["renderArtifact"][
                        "rendererVersion"
                    ],
                    "renderConfigurationSha256": (
                        render_context.effective_render_configuration_sha256
                    ),
                },
                "policy": copy.deepcopy(normalized["policy"]),
                "sampling": {
                    "strategy": normalized["sampling"]["strategy"],
                    "selectionArtifactSha256": selection_sha256,
                    "expectedFrameIds": [
                        frame["frameId"] for frame in frame_payloads
                    ],
                },
                "speakers": copy.deepcopy(normalized["speakers"]),
                "cues": [
                    _qa_cue(cue) for cue in normalized["cues"]
                ],
                "frames": frame_payloads,
            }
            try:
                evaluate_subtitle_visual_qa(qa_request)
            except Exception as exc:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                    "Collected evidence does not match the visual-QA request contract.",
                    detail=str(exc),
                ) from exc

            qa_request_sha256 = deterministic_sha256(qa_request)
            source_after = self._snapshot(source_path)
            rendered_after = self._snapshot(rendered_path)
            if source_after != source_before or rendered_after != rendered_before:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.SOURCE_CHANGED,
                    "Source or rendered media changed during evidence collection.",
                )
            self._require_overlay_unchanged(render_context)

            payload: dict[str, Any] = {
                "kind": SUBTITLE_RENDER_EVIDENCE_RESULT_KIND,
                "schemaVersion": SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION,
                "collectionId": normalized["collectionId"],
                "requestSha256": request_sha256,
                "sourceMedia": source_before,
                "renderArtifact": {
                    **rendered_before,
                    "renderer": normalized["renderArtifact"]["renderer"],
                    "rendererVersion": normalized["renderArtifact"][
                        "rendererVersion"
                    ],
                    "renderConfigurationSha256": (
                        render_context.effective_render_configuration_sha256
                    ),
                },
                "tools": tools,
                "components": {
                    "frameAnalyzer": self.analyzer.descriptor.to_dict(),
                    "fontEvidenceProvider": (
                        self.font_evidence_provider.descriptor.to_dict()
                    ),
                },
                "selection": {
                    **selection_core,
                    "selectionArtifactSha256": selection_sha256,
                },
                "bindings": bindings,
                "qaRequest": qa_request,
                "qaRequestSha256": qa_request_sha256,
            }
            if render_context.contrast_matte_evidence is not None:
                payload["contrastMatte"] = copy.deepcopy(
                    render_context.contrast_matte_evidence
                )
            payload["evidenceArtifactSha256"] = deterministic_sha256(
                payload
            )
            return SubtitleRenderEvidenceResult(payload=payload)
        finally:
            cleanup_error = _remove_tree(temporary_root)
            if cleanup_error is not None:
                active_error = sys.exc_info()[1]
                if active_error is not None:
                    active_error.add_note(cleanup_error)
                else:
                    raise SubtitleVisualEvidenceError(
                        SubtitleVisualEvidenceErrorCode.CLEANUP_FAILED,
                        "Representative-frame temporary directory could not be removed.",
                        detail=cleanup_error,
                    )

    def _prepare_render_context(
        self,
        *,
        normalized: Mapping[str, Any],
        source_path: Path,
        rendered_path: Path,
        source_snapshot: Mapping[str, Any],
        rendered_snapshot: Mapping[str, Any],
        canonical_ass_overlay_path: str | Path | None,
        canonical_ass_overlay_sha256: str | None,
        delivery_receipt: Mapping[str, Any] | Any | None,
    ) -> _RenderEvidenceContext:
        has_overlay_path = canonical_ass_overlay_path is not None
        has_overlay_hash = canonical_ass_overlay_sha256 is not None
        if has_overlay_path != has_overlay_hash:
            _invalid(
                "canonical ASS overlay path and SHA-256 must be supplied "
                "together"
            )

        receipt_payload: dict[str, Any] | None = None
        delivery_mode: str | None = None
        receipt_sha256: str | None = None
        if delivery_receipt is not None:
            receipt_payload = _canonical_delivery_receipt(delivery_receipt)
            delivery_mode = _validate_delivery_receipt(
                receipt_payload,
                source_snapshot=source_snapshot,
                rendered_snapshot=rendered_snapshot,
            )
            receipt_sha256 = deterministic_sha256(receipt_payload)

        if has_overlay_path and receipt_payload is None:
            _invalid(
                "canonical ASS overlay evidence requires a delivery receipt"
            )
        if delivery_mode in {"soft-mux", "burn-in"} and not has_overlay_path:
            _invalid(
                "media visual evidence requires the canonical private ASS "
                "carrier for glyph-matte contrast analysis"
            )
        if has_overlay_path and delivery_mode not in {"soft-mux", "burn-in"}:
            _invalid(
                "canonical ASS overlay evidence requires soft-mux or burn-in "
                "delivery"
            )
        if delivery_mode not in {None, "soft-mux", "burn-in"}:
            _invalid(
                "visual evidence only supports soft-mux or burn-in delivery"
            )

        overlay_path: Path | None = None
        overlay_snapshot: Mapping[str, Any] | None = None
        overlay_sha256: str | None = None
        contrast_background_ass: bytes | None = None
        glyph_fill_matte_ass: bytes | None = None
        contrast_matte_evidence: dict[str, Any] | None = None
        if has_overlay_path:
            assert canonical_ass_overlay_path is not None
            assert canonical_ass_overlay_sha256 is not None
            overlay_sha256 = _require_sha256(
                canonical_ass_overlay_sha256,
                "canonical_ass_overlay_sha256",
            )
            overlay_path = _canonical_private_ass_path(
                canonical_ass_overlay_path,
                maximum_bytes=self.policy.maximum_overlay_bytes,
            )
            if (
                os.path.samefile(overlay_path, source_path)
                or os.path.samefile(overlay_path, rendered_path)
            ):
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.PATH_ALIAS,
                    "The canonical ASS overlay must not alias media input.",
                )
            overlay_snapshot = self._snapshot(overlay_path)
            if overlay_snapshot["sha256"] != overlay_sha256:
                _invalid(
                    "canonical_ass_overlay_sha256 does not match the "
                    "private ASS artifact"
                )
            _validate_ass_payload(
                overlay_path,
                maximum_bytes=self.policy.maximum_overlay_bytes,
            )
            if receipt_payload is None:
                _invalid("canonical ASS carrier requires a delivery receipt")
            receipt_subtitle_value = receipt_payload.get("subtitlePath")
            if not isinstance(receipt_subtitle_value, str):
                _invalid(
                    "delivery_receipt.subtitlePath must bind the canonical "
                    "ASS carrier"
                )
            receipt_subtitle_path = _canonical_private_ass_path(
                receipt_subtitle_value,
                maximum_bytes=self.policy.maximum_overlay_bytes,
            )
            if not os.path.samefile(receipt_subtitle_path, overlay_path):
                _invalid(
                    "canonical ASS overlay does not match the delivery "
                    "receipt subtitlePath"
                )
            ass_text = _read_ass_text(
                overlay_path,
                maximum_bytes=self.policy.maximum_overlay_bytes,
            )
            (
                contrast_background_ass,
                glyph_fill_matte_ass,
                transform_descriptor,
            ) = _build_contrast_ass_variants(ass_text)
            contrast_background_sha256 = hashlib.sha256(
                contrast_background_ass
            ).hexdigest()
            glyph_fill_matte_sha256 = hashlib.sha256(
                glyph_fill_matte_ass
            ).hexdigest()
            contrast_matte_evidence = {
                "transform": transform_descriptor.to_dict(),
                "canonicalAssOverlaySha256": overlay_sha256,
                "contrastBackground": {
                    "strategy": CONTRAST_BACKGROUND_STRATEGY,
                    "assSha256": contrast_background_sha256,
                },
                "glyphFillMatte": {
                    "strategy": GLYPH_FILL_MATTE_STRATEGY,
                    "assSha256": glyph_fill_matte_sha256,
                },
                "inlineOverridePolicy": "fail-closed",
            }

        declared_configuration_sha256 = normalized["renderArtifact"][
            "renderConfigurationSha256"
        ]
        if delivery_mode is None:
            effective_configuration_sha256 = declared_configuration_sha256
        else:
            effective_configuration_sha256 = deterministic_sha256(
                {
                    "schemaVersion": "1.0.0",
                    "declaredRenderConfigurationSha256": (
                        declared_configuration_sha256
                    ),
                    "deliveryMode": delivery_mode,
                    "canonicalAssOverlaySha256": overlay_sha256,
                    "deliveryReceiptSha256": receipt_sha256,
                    "contrastMatte": contrast_matte_evidence,
                    "representativeFrameRendering": (
                        "ffmpeg-libass-single-png-absolute-pts-v1"
                        if delivery_mode == "soft-mux"
                        else "burn-in-rendered-media-delta-v1"
                    ),
                    "fullMediaPreviewRendered": False,
                    "frameAnalyzer": self.analyzer.descriptor.to_dict(),
                    "fontEvidenceProvider": (
                        self.font_evidence_provider.descriptor.to_dict()
                    ),
                }
            )
        return _RenderEvidenceContext(
            delivery_mode=delivery_mode,
            overlay_path=overlay_path,
            overlay_snapshot=overlay_snapshot,
            overlay_sha256=overlay_sha256,
            contrast_background_ass=contrast_background_ass,
            glyph_fill_matte_ass=glyph_fill_matte_ass,
            contrast_matte_evidence=contrast_matte_evidence,
            delivery_receipt_sha256=receipt_sha256,
            effective_render_configuration_sha256=(
                effective_configuration_sha256
            ),
        )

    def _stage_ass_overlay(
        self,
        context: _RenderEvidenceContext,
        *,
        temporary_root: Path,
    ) -> _StagedAssOverlays | None:
        if context.overlay_path is None:
            return None
        destination = temporary_root / "canonical-overlay.ass"
        if destination.exists():
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "Temporary canonical ASS overlay unexpectedly exists.",
            )
        try:
            with context.overlay_path.open("rb") as source_handle:
                with destination.open("xb") as destination_handle:
                    shutil.copyfileobj(
                        source_handle,
                        destination_handle,
                        length=self.policy.hash_chunk_bytes,
                    )
        except OSError as exc:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "The private ASS overlay could not be staged for "
                "representative-frame rendering.",
                detail=str(exc),
            ) from exc
        staged_sha256 = _hash_bounded_file(
            destination,
            maximum=self.policy.maximum_overlay_bytes,
            chunk_bytes=self.policy.hash_chunk_bytes,
        )
        if staged_sha256 != context.overlay_sha256:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.SOURCE_CHANGED,
                "The canonical ASS overlay changed while it was staged.",
            )
        if (
            context.contrast_background_ass is None
            or context.glyph_fill_matte_ass is None
            or context.contrast_matte_evidence is None
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Canonical ASS contrast transformations are unavailable.",
            )
        contrast_background = temporary_root / "contrast-background.ass"
        glyph_fill_matte = temporary_root / "glyph-fill-matte.ass"
        try:
            with contrast_background.open("xb") as handle:
                handle.write(context.contrast_background_ass)
            with glyph_fill_matte.open("xb") as handle:
                handle.write(context.glyph_fill_matte_ass)
        except OSError as exc:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "Derived ASS contrast evidence could not be staged.",
                detail=str(exc),
            ) from exc
        expected_background_sha = context.contrast_matte_evidence[
            "contrastBackground"
        ]["assSha256"]
        expected_matte_sha = context.contrast_matte_evidence[
            "glyphFillMatte"
        ]["assSha256"]
        if (
            _hash_bounded_file(
                contrast_background,
                maximum=self.policy.maximum_overlay_bytes,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            != expected_background_sha
            or _hash_bounded_file(
                glyph_fill_matte,
                maximum=self.policy.maximum_overlay_bytes,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            != expected_matte_sha
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.SOURCE_CHANGED,
                "Derived ASS contrast evidence changed while it was staged.",
            )
        return _StagedAssOverlays(
            canonical=destination,
            contrast_background=contrast_background,
            glyph_fill_matte=glyph_fill_matte,
        )

    def _require_overlay_unchanged(
        self,
        context: _RenderEvidenceContext,
    ) -> None:
        if context.overlay_path is None or context.overlay_snapshot is None:
            return
        if self._snapshot(context.overlay_path) != context.overlay_snapshot:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.SOURCE_CHANGED,
                "The canonical ASS overlay changed during evidence "
                "collection.",
            )

    def _snapshot(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path),
            "sizeBytes": stat.st_size,
            "mtimeNs": stat.st_mtime_ns,
            "sha256": _hash_file(
                path,
                chunk_bytes=self.policy.hash_chunk_bytes,
            ),
        }

    @staticmethod
    def _make_same_directory_temp(rendered_path: Path) -> Path:
        try:
            root = Path(
                tempfile.mkdtemp(
                    prefix=".mts-subtitle-visual-evidence-",
                    dir=rendered_path.parent,
                )
            ).resolve(strict=True)
        except OSError as exc:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "A same-directory evidence workspace could not be created.",
                detail=str(exc),
            ) from exc
        if root.parent != rendered_path.parent:
            _remove_tree(root)
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "Evidence workspace escaped the rendered-media directory.",
            )
        return root

    def _run(
        self,
        command: Sequence[str],
        temporary_root: Path,
    ) -> EvidenceProcessResult:
        result = self.runner.run(
            tuple(str(item) for item in command),
            limits=self.policy.process_limits,
            cwd=temporary_root,
        )
        if (
            len(result.stdout)
            > self.policy.process_limits.max_stdout_bytes
            or len(result.stderr)
            > self.policy.process_limits.max_stderr_bytes
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.PROCESS_OUTPUT_LIMIT,
                "Injected runner returned output beyond the configured limit.",
            )
        return result

    def _tool_evidence(
        self,
        executable: Path,
        temporary_root: Path,
    ) -> dict[str, Any]:
        result = self._run(
            (str(executable), "-hide_banner", "-version"),
            temporary_root,
        )
        if result.returncode != 0:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.PROCESS_FAILED,
                f"Tool version probe failed: {executable}",
                detail=_bounded_detail(result.stderr),
            )
        version_line = _first_nonempty_line(
            result.stdout, result.stderr
        )
        if version_line is None:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.PROCESS_FAILED,
                f"Tool version probe returned no auditable version: {executable}",
            )
        version_artifact_sha256 = hashlib.sha256(
            result.stdout + b"\x00" + result.stderr
        ).hexdigest()
        return {
            "path": str(executable),
            "executableSha256": _hash_file(
                executable,
                chunk_bytes=self.policy.hash_chunk_bytes,
            ),
            "versionLine": version_line[:320],
            "versionArtifactSha256": version_artifact_sha256,
        }

    def _probe_video(
        self,
        media_path: Path,
        temporary_root: Path,
    ) -> dict[str, Any]:
        command = (
            str(self.ffprobe_path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(media_path),
        )
        result = self._run(command, temporary_root)
        if result.returncode != 0:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.PROBE_FAILED,
                f"FFprobe failed for local media: {media_path}",
                detail=_bounded_detail(result.stderr),
            )
        payload = _decode_json_object(result.stdout, label="FFprobe video")
        streams = payload.get("streams")
        if not isinstance(streams, list) or not streams:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.VIDEO_REQUIRED,
                f"A decodable primary video stream is required: {media_path}",
            )
        stream = _require_mapping(streams[0], "FFprobe streams[0]")
        width = _require_integer(
            stream.get("width"),
            "FFprobe width",
            minimum=16,
            maximum=32_768,
        )
        height = _require_integer(
            stream.get("height"),
            "FFprobe height",
            minimum=16,
            maximum=32_768,
        )
        format_value = payload.get("format", {})
        format_item = _require_mapping(format_value, "FFprobe format")
        duration_raw = format_item.get("duration")
        duration_ms: int | None = None
        if duration_raw is not None:
            try:
                duration_ms = round(float(duration_raw) * 1_000)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.PROBE_FAILED,
                    "FFprobe returned an invalid media duration.",
                ) from exc
            if not 1 <= duration_ms <= 604_800_000:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.PROBE_FAILED,
                    "FFprobe media duration is outside supported bounds.",
                )
        return {
            "widthPx": width,
            "heightPx": height,
            "durationMs": duration_ms,
        }

    @staticmethod
    def _validate_video_pair(
        source: Mapping[str, Any],
        rendered: Mapping[str, Any],
    ) -> None:
        if (
            source["widthPx"] != rendered["widthPx"]
            or source["heightPx"] != rendered["heightPx"]
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.VIDEO_MISMATCH,
                "Source and rendered video dimensions must match for pixel evidence.",
            )
        source_duration = source["durationMs"]
        rendered_duration = rendered["durationMs"]
        if source_duration is not None and rendered_duration is not None:
            tolerance = max(750, round(source_duration * 0.005))
            if abs(source_duration - rendered_duration) > tolerance:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.VIDEO_MISMATCH,
                    "Source and rendered video durations differ beyond tolerance.",
                )

    def _resolve_candidates(
        self,
        *,
        rendered_path: Path,
        cues: Sequence[Mapping[str, Any]],
        fractions: Sequence[float],
        temporary_root: Path,
    ) -> list[dict[str, Any]]:
        raw: list[dict[str, Any]] = []
        for cue in cues:
            for fraction in fractions:
                requested_ms = _fractional_timestamp(
                    cue["startMs"],
                    cue["endMs"],
                    fraction,
                )
                decoded_ms = self._resolve_frame_timestamp(
                    rendered_path,
                    requested_ms,
                    cue_start_ms=cue["startMs"],
                    cue_end_ms=cue["endMs"],
                    temporary_root=temporary_root,
                )
                if not cue["startMs"] <= decoded_ms <= cue["endMs"]:
                    raise SubtitleVisualEvidenceError(
                        SubtitleVisualEvidenceErrorCode.FRAME_TIME_UNAVAILABLE,
                        f"Decoded frame falls outside cue {cue['cueId']!r}.",
                    )
                raw.append(
                    {
                        "cueId": cue["cueId"],
                        "fraction": fraction,
                        "requestedTimestampMs": requested_ms,
                        "decodedTimestampMs": decoded_ms,
                    }
                )

        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in raw:
            grouped.setdefault(item["decodedTimestampMs"], []).append(item)
        frame_ids: dict[int, str] = {}
        for decoded_ms, members in sorted(grouped.items()):
            identity = deterministic_sha256(
                {
                    "decodedTimestampMs": decoded_ms,
                    "members": sorted(
                        members,
                        key=lambda value: (
                            value["cueId"],
                            value["fraction"],
                            value["requestedTimestampMs"],
                        ),
                    ),
                }
            )
            frame_ids[decoded_ms] = (
                f"frame-{decoded_ms:012d}-{identity[:12]}"
            )
        return [
            {
                **item,
                "frameId": frame_ids[item["decodedTimestampMs"]],
            }
            for item in sorted(
                raw,
                key=lambda value: (
                    value["decodedTimestampMs"],
                    value["cueId"],
                    value["fraction"],
                ),
            )
        ]

    def _resolve_frame_timestamp(
        self,
        rendered_path: Path,
        requested_ms: int,
        *,
        cue_start_ms: int,
        cue_end_ms: int,
        temporary_root: Path,
    ) -> int:
        interval = (
            f"{_seconds_text(requested_ms)}%{_seconds_text(cue_end_ms)}"
        )
        command = (
            str(self.ffprobe_path),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            interval,
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,width,height",
            "-of",
            "json",
            str(rendered_path),
        )
        result = self._run(command, temporary_root)
        if result.returncode != 0:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_TIME_UNAVAILABLE,
                "FFprobe could not resolve a representative-frame timestamp.",
                detail=_bounded_detail(result.stderr),
            )
        payload = _decode_json_object(
            result.stdout, label="FFprobe frame timestamp"
        )
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_TIME_UNAVAILABLE,
                "FFprobe returned no representative video frame.",
            )
        decoded_timestamps: list[int] = []
        for index, value in enumerate(frames):
            frame = _require_mapping(value, f"FFprobe frames[{index}]")
            raw = frame.get("best_effort_timestamp_time")
            try:
                decoded_ms = round(float(raw) * 1_000)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.FRAME_TIME_UNAVAILABLE,
                    "FFprobe returned an invalid frame timestamp.",
                ) from exc
            if not 0 <= decoded_ms <= 604_800_000:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.FRAME_TIME_UNAVAILABLE,
                    "FFprobe frame timestamp is outside supported bounds.",
                )
            if cue_start_ms <= decoded_ms <= cue_end_ms:
                decoded_timestamps.append(decoded_ms)

        if not decoded_timestamps:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_TIME_UNAVAILABLE,
                "FFprobe returned no representative frame within the cue "
                "interval.",
            )
        return min(
            decoded_timestamps,
            key=lambda timestamp_ms: (
                abs(timestamp_ms - requested_ms),
                timestamp_ms < requested_ms,
                timestamp_ms,
            ),
        )

    def _collect_frames(
        self,
        *,
        source_path: Path,
        rendered_path: Path,
        video: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        cues: Sequence[Mapping[str, Any]],
        contrast_policy: Mapping[str, Any],
        temporary_root: Path,
        staged_ass_overlays: _StagedAssOverlays | None,
        render_context: _RenderEvidenceContext,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        by_timestamp: dict[int, str] = {}
        for candidate in candidates:
            by_timestamp[candidate["decodedTimestampMs"]] = candidate[
                "frameId"
            ]
        cue_by_id = {cue["cueId"]: cue for cue in cues}
        frame_payloads: list[dict[str, Any]] = []
        frame_records: list[dict[str, Any]] = []
        bindings: list[dict[str, Any]] = []

        for index, (timestamp_ms, frame_id) in enumerate(
            sorted(by_timestamp.items())
        ):
            rendered_frame = temporary_root / f"{index:06d}-rendered.png"
            source_frame = temporary_root / f"{index:06d}-source.png"
            contrast_background_frame: Path | None = None
            glyph_fill_matte_frame: Path | None = None
            self._extract_frame(
                source_path,
                timestamp_ms,
                source_frame,
                temporary_root,
            )
            if staged_ass_overlays is None:
                self._extract_frame(
                    rendered_path,
                    timestamp_ms,
                    rendered_frame,
                    temporary_root,
                )
            else:
                if render_context.delivery_mode == "soft-mux":
                    self._render_ass_overlay_frame(
                        base_frame_path=source_frame,
                        timestamp_ms=timestamp_ms,
                        output_path=rendered_frame,
                        temporary_root=temporary_root,
                        ass_overlay_path=staged_ass_overlays.canonical,
                    )
                else:
                    self._extract_frame(
                        rendered_path,
                        timestamp_ms,
                        rendered_frame,
                        temporary_root,
                    )
                contrast_background_frame = temporary_root / (
                    f"{index:06d}-contrast-background.png"
                )
                glyph_fill_matte_frame = temporary_root / (
                    f"{index:06d}-glyph-fill-matte.png"
                )
                self._render_ass_overlay_frame(
                    base_frame_path=source_frame,
                    timestamp_ms=timestamp_ms,
                    output_path=contrast_background_frame,
                    temporary_root=temporary_root,
                    ass_overlay_path=(
                        staged_ass_overlays.contrast_background
                    ),
                )
                self._render_ass_overlay_frame(
                    base_frame_path=source_frame,
                    timestamp_ms=timestamp_ms,
                    output_path=glyph_fill_matte_frame,
                    temporary_root=temporary_root,
                    ass_overlay_path=staged_ass_overlays.glyph_fill_matte,
                    black_base=True,
                )
            rendered_image_sha256 = _hash_bounded_file(
                rendered_frame,
                maximum=self.policy.maximum_frame_bytes,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            source_image_sha256 = _hash_bounded_file(
                source_frame,
                maximum=self.policy.maximum_frame_bytes,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            contrast_background_image_sha256 = (
                _hash_bounded_file(
                    contrast_background_frame,
                    maximum=self.policy.maximum_frame_bytes,
                    chunk_bytes=self.policy.hash_chunk_bytes,
                )
                if contrast_background_frame is not None
                else None
            )
            glyph_fill_matte_image_sha256 = (
                _hash_bounded_file(
                    glyph_fill_matte_frame,
                    maximum=self.policy.maximum_frame_bytes,
                    chunk_bytes=self.policy.hash_chunk_bytes,
                )
                if glyph_fill_matte_frame is not None
                else None
            )
            active_cues = [
                cue
                for cue in cues
                if cue["startMs"] <= timestamp_ms <= cue["endMs"]
            ]
            if not active_cues:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                    f"Representative frame {frame_id!r} has no active cue.",
                )
            analyzer_kwargs: dict[str, Any] = {
                "rendered_frame_path": rendered_frame,
                "source_frame_path": source_frame,
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "width_px": video["widthPx"],
                "height_px": video["heightPx"],
                "cues": active_cues,
                "contrast_policy": contrast_policy,
            }
            if contrast_background_frame is not None:
                analyzer_kwargs.update(
                    contrast_background_frame_path=(
                        contrast_background_frame
                    ),
                    glyph_fill_matte_frame_path=glyph_fill_matte_frame,
                )
            observation = self.analyzer.analyze(
                **analyzer_kwargs,
            )
            instances, contrast_analysis = self._normalize_frame_observation(
                observation=observation,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                expected_cues=active_cues,
                rendered_frame=rendered_frame,
                source_frame=source_frame,
                rendered_image_sha256=rendered_image_sha256,
                source_image_sha256=source_image_sha256,
                contrast_background_image_sha256=(
                    contrast_background_image_sha256
                ),
                glyph_fill_matte_image_sha256=(
                    glyph_fill_matte_image_sha256
                ),
                video=video,
                bindings=bindings,
                render_context=render_context,
            )
            frame_payloads.append(
                {
                    "frameId": frame_id,
                    "timestampMs": timestamp_ms,
                    "widthPx": observation.width_px,
                    "heightPx": observation.height_px,
                    "imageSha256": rendered_image_sha256,
                    "instances": instances,
                }
            )
            frame_record = {
                    "frameId": frame_id,
                    "requestedTimestampMs": min(
                        item["requestedTimestampMs"]
                        for item in candidates
                        if item["frameId"] == frame_id
                    ),
                    "decodedTimestampMs": timestamp_ms,
                    "cueIds": sorted(
                        cue["cueId"] for cue in active_cues
                    ),
                    "renderedImageSha256": rendered_image_sha256,
                    "sourceImageSha256": source_image_sha256,
                }
            if contrast_background_image_sha256 is not None:
                frame_record.update(
                    contrastBackgroundImageSha256=(
                        contrast_background_image_sha256
                    ),
                    glyphFillMatteImageSha256=(
                        glyph_fill_matte_image_sha256
                    ),
                    contrastAnalysis=contrast_analysis,
                )
            frame_records.append(frame_record)

        represented = {
            instance["cueId"]
            for frame in frame_payloads
            for instance in frame["instances"]
        }
        missing = sorted(set(cue_by_id) - represented)
        if missing:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                f"Cues lack representative-frame evidence: {missing!r}",
            )
        return frame_payloads, frame_records, bindings

    def _extract_frame(
        self,
        media_path: Path,
        timestamp_ms: int,
        output_path: Path,
        temporary_root: Path,
        *,
        ass_overlay_path: Path | None = None,
    ) -> None:
        if output_path.exists():
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "Temporary frame output unexpectedly exists.",
            )
        if ass_overlay_path is not None:
            self._extract_ass_overlay_frame(
                media_path=media_path,
                timestamp_ms=timestamp_ms,
                output_path=output_path,
                temporary_root=temporary_root,
                ass_overlay_path=ass_overlay_path,
            )
            return
        self._extract_plain_frame(
            media_path=media_path,
            timestamp_ms=timestamp_ms,
            output_path=output_path,
            temporary_root=temporary_root,
        )

    def _extract_plain_frame(
        self,
        *,
        media_path: Path,
        timestamp_ms: int,
        output_path: Path,
        temporary_root: Path,
    ) -> None:
        command = (
            str(self.ffmpeg_path),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _seconds_text(timestamp_ms),
            "-i",
            str(media_path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "png",
            "-f",
            "image2",
            str(output_path),
        )
        result = self._run(command, temporary_root)
        if result.returncode != 0:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_EXTRACTION_FAILED,
                f"FFmpeg could not extract representative frame at {timestamp_ms} ms.",
                detail=_bounded_detail(result.stderr),
            )
        if not output_path.is_file() or output_path.stat().st_size < 1:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_EXTRACTION_FAILED,
                "FFmpeg did not create a non-empty representative frame.",
            )
        if output_path.stat().st_size > self.policy.maximum_frame_bytes:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_OUTPUT_LIMIT,
                "Representative frame exceeds the configured byte limit.",
            )

    def _extract_ass_overlay_frame(
        self,
        *,
        media_path: Path,
        timestamp_ms: int,
        output_path: Path,
        temporary_root: Path,
        ass_overlay_path: Path,
    ) -> None:
        if ass_overlay_path.parent != temporary_root:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "The staged ASS overlay escaped the evidence workspace.",
            )
        if ass_overlay_path.name != "canonical-overlay.ass":
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "The staged ASS overlay name is not canonical.",
            )
        base_frame = output_path.with_name(
            f"{output_path.stem}-unsubtitled.png"
        )
        self._extract_plain_frame(
            media_path=media_path,
            timestamp_ms=timestamp_ms,
            output_path=base_frame,
            temporary_root=temporary_root,
        )
        self._render_ass_overlay_frame(
            base_frame_path=base_frame,
            timestamp_ms=timestamp_ms,
            output_path=output_path,
            temporary_root=temporary_root,
            ass_overlay_path=ass_overlay_path,
        )

    def _render_ass_overlay_frame(
        self,
        *,
        base_frame_path: Path,
        timestamp_ms: int,
        output_path: Path,
        temporary_root: Path,
        ass_overlay_path: Path,
        black_base: bool = False,
    ) -> None:
        if output_path.exists():
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "Temporary ASS-rendered frame output unexpectedly exists.",
            )
        if base_frame_path.parent != temporary_root:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "The ASS base frame escaped the evidence workspace.",
            )
        if ass_overlay_path.parent != temporary_root:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "The staged ASS overlay escaped the evidence workspace.",
            )
        if ass_overlay_path.name not in {
            "canonical-overlay.ass",
            "contrast-background.ass",
            "glyph-fill-matte.ass",
        }:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "The staged ASS overlay name is not recognized.",
            )
        filters = []
        if black_base:
            filters.append("lutrgb=r=0:g=0:b=0")
        filters.extend(
            (
                f"setpts=PTS+{_seconds_text(timestamp_ms)}/TB",
                f"ass=filename={ass_overlay_path.name}",
            )
        )
        filter_value = ",".join(filters)
        command = (
            str(self.ffmpeg_path),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(base_frame_path),
            "-map",
            "0:v:0",
            "-vf",
            filter_value,
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "png",
            "-f",
            "image2",
            str(output_path),
        )
        result = self._run(command, temporary_root)
        if result.returncode != 0:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_EXTRACTION_FAILED,
                "FFmpeg/libass could not render the canonical ASS overlay "
                f"at {timestamp_ms} ms.",
                detail=_bounded_detail(result.stderr),
            )
        if not output_path.is_file() or output_path.stat().st_size < 1:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_EXTRACTION_FAILED,
                "FFmpeg/libass did not create a non-empty representative "
                "overlay frame.",
            )
        if output_path.stat().st_size > self.policy.maximum_frame_bytes:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FRAME_OUTPUT_LIMIT,
                "Representative overlay frame exceeds the configured byte "
                "limit.",
            )

    def _normalize_frame_observation(
        self,
        *,
        observation: FrameObservation,
        frame_id: str,
        timestamp_ms: int,
        expected_cues: Sequence[Mapping[str, Any]],
        rendered_frame: Path,
        source_frame: Path,
        rendered_image_sha256: str,
        source_image_sha256: str,
        contrast_background_image_sha256: str | None,
        glyph_fill_matte_image_sha256: str | None,
        video: Mapping[str, Any],
        bindings: list[dict[str, Any]],
        render_context: _RenderEvidenceContext,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if (
            observation.width_px != video["widthPx"]
            or observation.height_px != video["heightPx"]
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.VIDEO_MISMATCH,
                "Analyzer dimensions contradict FFprobe evidence.",
            )
        expected = {cue["cueId"]: cue for cue in expected_cues}
        actual_ids = [instance.cue_id for instance in observation.instances]
        if len(actual_ids) != len(set(actual_ids)):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Analyzer returned duplicate cue observations.",
            )
        if set(actual_ids) != set(expected):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Analyzer cue observations do not match active cues.",
            )

        payloads: list[dict[str, Any]] = []
        contrast_analysis: list[dict[str, Any]] = []
        for instance in sorted(
            observation.instances, key=lambda item: item.cue_id
        ):
            cue = expected[instance.cue_id]
            bounds = _parse_rect(
                instance.bounds,
                f"{frame_id}/{instance.cue_id}/bounds",
                frame_width=observation.width_px,
                frame_height=observation.height_px,
            )
            ink_bounds = _parse_rect(
                instance.ink_bounds,
                f"{frame_id}/{instance.cue_id}/inkBounds",
                frame_width=observation.width_px,
                frame_height=observation.height_px,
            )
            clipped = _require_integer(
                instance.clipped_pixel_count,
                "clipped_pixel_count",
                minimum=0,
                maximum=1_000_000_000,
            )
            edge = _require_integer(
                instance.edge_touching_pixel_count,
                "edge_touching_pixel_count",
                minimum=0,
                maximum=1_000_000_000,
            )
            if not isinstance(instance.overflow_detected, bool):
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                    "Analyzer overflow evidence must be boolean.",
                )

            diagnostics = self._normalize_contrast_diagnostics(
                instance.contrast_diagnostics,
                cue_id=cue["cueId"],
                matte_required=(
                    contrast_background_image_sha256 is not None
                    or glyph_fill_matte_image_sha256 is not None
                ),
            )
            if diagnostics is not None:
                contrast_analysis.append(diagnostics)

            contrast_samples: list[dict[str, Any]] = []
            seen_classes: set[str] = set()
            for sample in sorted(
                instance.contrast_samples,
                key=lambda item: item.background_class,
            ):
                if sample.background_class not in {"dark", "light"}:
                    raise SubtitleVisualEvidenceError(
                        SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                        "Analyzer returned an unsupported background class.",
                    )
                if sample.background_class in seen_classes:
                    raise SubtitleVisualEvidenceError(
                        SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                        "Analyzer returned duplicate background-class samples.",
                    )
                seen_classes.add(sample.background_class)
                _require_color(
                    sample.foreground_rgb,
                    "contrast foreground_rgb",
                )
                _require_color(
                    sample.background_rgb,
                    "contrast background_rgb",
                )
                foreground_count = _require_integer(
                    sample.foreground_pixel_count,
                    "foreground_pixel_count",
                    minimum=1,
                    maximum=1_000_000_000,
                )
                background_count = _require_integer(
                    sample.background_pixel_count,
                    "background_pixel_count",
                    minimum=1,
                    maximum=1_000_000_000,
                )
                sample_core = {
                    "frameId": frame_id,
                    "cueId": cue["cueId"],
                    "backgroundClass": sample.background_class,
                    "foregroundRgb": sample.foreground_rgb.upper(),
                    "backgroundRgb": sample.background_rgb.upper(),
                    "foregroundPixelCount": foreground_count,
                    "backgroundPixelCount": background_count,
                    "renderedImageSha256": rendered_image_sha256,
                    "sourceImageSha256": source_image_sha256,
                    "contrastBackgroundImageSha256": (
                        contrast_background_image_sha256
                    ),
                    "glyphFillMatteImageSha256": (
                        glyph_fill_matte_image_sha256
                    ),
                    "contrastDiagnostics": diagnostics,
                }
                sample_sha256 = deterministic_sha256(sample_core)
                contrast_samples.append(
                    {
                        "sampleId": (
                            f"sample-{sample_sha256[:16]}-"
                            f"{sample.background_class}"
                        ),
                        "backgroundClass": sample.background_class,
                        "foregroundRgb": sample.foreground_rgb.upper(),
                        "backgroundRgb": sample.background_rgb.upper(),
                        "foregroundPixelCount": foreground_count,
                        "backgroundPixelCount": background_count,
                        "sampledFromRenderedFrame": True,
                        "sampleArtifactSha256": sample_sha256,
                    }
                )

            font_evidence = self._collect_font_evidence(
                cue=cue,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                rendered_frame=rendered_frame,
                source_frame=source_frame,
            )
            payload = {
                "cueId": cue["cueId"],
                "bounds": bounds,
                "inkBounds": ink_bounds,
                "clippedPixelCount": clipped,
                "edgeTouchingPixelCount": edge,
                "overflowDetected": instance.overflow_detected,
                "fontEvidence": font_evidence,
                "contrastSamples": contrast_samples,
            }
            binding_core = {
                "frameId": frame_id,
                "cueId": cue["cueId"],
                "speakerId": cue["speakerId"],
                "styleId": cue["styleId"],
                "cueTextSha256": hashlib.sha256(
                    cue["text"].encode("utf-8")
                ).hexdigest(),
                "renderedImageSha256": rendered_image_sha256,
                "sourceImageSha256": source_image_sha256,
                "contrastBackgroundImageSha256": (
                    contrast_background_image_sha256
                ),
                "glyphFillMatteImageSha256": (
                    glyph_fill_matte_image_sha256
                ),
                "bounds": bounds,
                "inkBounds": ink_bounds,
                "fontEvidence": font_evidence,
                "contrastSamples": contrast_samples,
                "contrastDiagnostics": diagnostics,
                "renderConfigurationSha256": (
                    render_context.effective_render_configuration_sha256
                ),
                "canonicalAssOverlaySha256": (
                    render_context.overlay_sha256
                ),
                "deliveryReceiptSha256": (
                    render_context.delivery_receipt_sha256
                ),
            }
            bindings.append(
                {
                    "frameId": frame_id,
                    "cueId": cue["cueId"],
                    "speakerId": cue["speakerId"],
                    "styleId": cue["styleId"],
                    "bindingArtifactSha256": deterministic_sha256(
                        binding_core
                    ),
                }
            )
            payloads.append(payload)
        return payloads, contrast_analysis

    def _normalize_contrast_diagnostics(
        self,
        value: Mapping[str, Any] | None,
        *,
        cue_id: str,
        matte_required: bool,
    ) -> dict[str, Any] | None:
        if value is None:
            if matte_required:
                raise SubtitleVisualEvidenceError(
                    SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                    "Glyph-matte analysis returned no component diagnostics.",
                )
            return None
        if not matte_required:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Analyzer returned glyph-matte diagnostics without matte frames.",
            )
        item = _require_mapping(value, "contrast diagnostics")
        required = {
            "strategy",
            "glyphComponentCount",
            "glyphComponentsCovered",
            "pairedCorePixelCount",
            "glyphCoreAlphaThreshold",
            "glyphNonzeroAlphaQuantile75",
            "componentContrastQuantile",
            "tinyComponentFallbackCount",
            "tinyComponentMaxPixels",
            "tinyComponentMinimumAlpha",
            "minimumComponentCorePixelCount",
            "effectiveBackgroundClasses",
            "underlyingSceneClasses",
            "componentCoverageSha256",
        }
        if set(item) != required:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Glyph-matte diagnostics contain missing or unknown fields.",
            )
        strategy = _require_text(
            item["strategy"],
            "contrast diagnostics strategy",
            maximum=160,
        )
        if strategy != GLYPH_CORE_COMPONENT_STRATEGY:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Glyph-matte diagnostics use an unsupported strategy.",
            )
        component_count = _require_integer(
            item["glyphComponentCount"],
            "glyphComponentCount",
            minimum=1,
            maximum=1_000_000,
        )
        covered_count = _require_integer(
            item["glyphComponentsCovered"],
            "glyphComponentsCovered",
            minimum=1,
            maximum=1_000_000,
        )
        if covered_count != component_count:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Not every meaningful glyph component has core-pixel coverage.",
            )
        paired_count = _require_integer(
            item["pairedCorePixelCount"],
            "pairedCorePixelCount",
            minimum=32,
            maximum=1_000_000_000,
        )
        minimum_component_core = _require_integer(
            item["minimumComponentCorePixelCount"],
            "minimumComponentCorePixelCount",
            minimum=1,
            maximum=1_000_000_000,
        )
        alpha_threshold = _require_number(
            item["glyphCoreAlphaThreshold"],
            "glyphCoreAlphaThreshold",
            minimum=0.80,
            maximum=1.0,
        )
        alpha_q75 = _require_number(
            item["glyphNonzeroAlphaQuantile75"],
            "glyphNonzeroAlphaQuantile75",
            minimum=0.0,
            maximum=1.0,
        )
        contrast_quantile = _require_number(
            item["componentContrastQuantile"],
            "componentContrastQuantile",
            minimum=0.05,
            maximum=0.05,
        )
        tiny_fallback_count = _require_integer(
            item["tinyComponentFallbackCount"],
            "tinyComponentFallbackCount",
            minimum=0,
            maximum=component_count,
        )
        tiny_max_pixels = _require_integer(
            item["tinyComponentMaxPixels"],
            "tinyComponentMaxPixels",
            minimum=1,
            maximum=1_000_000,
        )
        if tiny_max_pixels != TINY_COMPONENT_MAX_PIXELS:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Glyph-matte diagnostics use an unsupported tiny-component "
                "pixel limit.",
            )
        tiny_minimum_alpha = _require_number(
            item["tinyComponentMinimumAlpha"],
            "tinyComponentMinimumAlpha",
            minimum=0.50,
            maximum=1.0,
        )
        if tiny_minimum_alpha != TINY_COMPONENT_MINIMUM_ALPHA:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
                "Glyph-matte diagnostics use an unsupported tiny-component "
                "alpha floor.",
            )
        effective_classes = _normalize_background_classes(
            item["effectiveBackgroundClasses"],
            label="effectiveBackgroundClasses",
        )
        underlying_classes = _normalize_background_classes(
            item["underlyingSceneClasses"],
            label="underlyingSceneClasses",
        )
        coverage_sha256 = _require_sha256(
            item["componentCoverageSha256"],
            "componentCoverageSha256",
        )
        return {
            "cueId": cue_id,
            "strategy": strategy,
            "glyphComponentCount": component_count,
            "glyphComponentsCovered": covered_count,
            "pairedCorePixelCount": paired_count,
            "glyphCoreAlphaThreshold": alpha_threshold,
            "glyphNonzeroAlphaQuantile75": alpha_q75,
            "componentContrastQuantile": contrast_quantile,
            "tinyComponentFallbackCount": tiny_fallback_count,
            "tinyComponentMaxPixels": tiny_max_pixels,
            "tinyComponentMinimumAlpha": tiny_minimum_alpha,
            "minimumComponentCorePixelCount": minimum_component_core,
            "effectiveBackgroundClasses": effective_classes,
            "underlyingSceneClasses": underlying_classes,
            "componentCoverageSha256": coverage_sha256,
        }

    def _collect_font_evidence(
        self,
        *,
        cue: Mapping[str, Any],
        frame_id: str,
        timestamp_ms: int,
        rendered_frame: Path,
        source_frame: Path,
    ) -> dict[str, Any] | None:
        try:
            observation = self.font_evidence_provider.collect(
                cue=cue,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                rendered_frame_path=rendered_frame,
                source_frame_path=source_frame,
            )
        except SubtitleVisualEvidenceError:
            raise
        except Exception as exc:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FONT_EVIDENCE_INVALID,
                f"Font evidence provider failed for cue {cue['cueId']!r}.",
                detail=str(exc),
            ) from exc
        if observation is None:
            return None
        method = _require_enum(
            observation.verification_method,
            "font verification method",
            _FONT_METHODS,
        )
        resolved_family = observation.resolved_family
        if resolved_family is not None:
            resolved_family = _require_text(
                resolved_family,
                "resolved font family",
                maximum=320,
            )
        if not isinstance(observation.resolution_verified, bool) or not isinstance(
            observation.glyph_coverage_verified, bool
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FONT_EVIDENCE_INVALID,
                "Font verification flags must be boolean.",
            )
        evidence_sha256: str | None = None
        if observation.evidence_artifact_path is not None:
            artifact = _canonical_evidence_file(
                observation.evidence_artifact_path,
                label="font evidence artifact",
            )
            evidence_sha256 = _hash_file(
                artifact,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
        if (
            observation.resolution_verified
            or observation.glyph_coverage_verified
        ) and (
            method == "not-provided"
            or evidence_sha256 is None
            or resolved_family is None
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FONT_EVIDENCE_INVALID,
                "Positive font resolution/glyph evidence requires a method, resolved family, and real artifact.",
            )

        expected = _renderable_codepoint_count(cue["text"])
        covered = _require_integer(
            observation.covered_renderable_code_points,
            "covered renderable code points",
            minimum=0,
            maximum=1_000_000,
        )
        missing = tuple(observation.missing_code_points)
        if len(missing) != len(set(missing)) or any(
            not _CODEPOINT.fullmatch(value) for value in missing
        ):
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.FONT_EVIDENCE_INVALID,
                "Missing code points must be unique U+XXXX values.",
            )
        tofu = _require_integer(
            observation.tofu_glyph_count,
            "tofu glyph count",
            minimum=0,
            maximum=1_000_000,
        )
        installation = self._normalize_font_claim(
            observation.installation,
            statuses=_INSTALLATION_STATUSES,
            label="installation",
        )
        embedding = self._normalize_font_claim(
            observation.embedding,
            statuses=_EMBEDDING_STATUSES,
            label="embedding",
        )
        return {
            "requestedFamilies": list(cue["requestedFontFamilies"]),
            "resolvedFamily": resolved_family,
            "resolutionVerified": observation.resolution_verified,
            "glyphCoverageVerified": observation.glyph_coverage_verified,
            "verificationMethod": method,
            "evidenceArtifactSha256": evidence_sha256,
            "expectedRenderableCodePoints": expected,
            "coveredRenderableCodePoints": covered,
            "missingCodePoints": list(missing),
            "tofuGlyphCount": tofu,
            "installation": installation,
            "embedding": embedding,
        }

    def _normalize_font_claim(
        self,
        claim: VerifiedFontClaim | None,
        *,
        statuses: frozenset[str],
        label: str,
    ) -> dict[str, Any]:
        if claim is None:
            return {
                "status": "not-asserted",
                "verificationMethod": "not-provided",
                "evidenceArtifactSha256": None,
                "fontArtifactSha256": None,
            }
        status = _require_enum(
            claim.status, f"font {label} status", statuses
        )
        method = _require_enum(
            claim.verification_method,
            f"font {label} method",
            _POSITIVE_FONT_METHODS,
        )
        evidence_path = _canonical_evidence_file(
            claim.evidence_artifact_path,
            label=f"font {label} evidence artifact",
        )
        font_path = _canonical_evidence_file(
            claim.font_artifact_path,
            label=f"font {label} artifact",
        )
        return {
            "status": status,
            "verificationMethod": method,
            "evidenceArtifactSha256": _hash_file(
                evidence_path,
                chunk_bytes=self.policy.hash_chunk_bytes,
            ),
            "fontArtifactSha256": _hash_file(
                font_path,
                chunk_bytes=self.policy.hash_chunk_bytes,
            ),
        }


def _parse_request(request: Mapping[str, Any]) -> dict[str, Any]:
    root = _require_mapping(request, "$")
    _require_keys(
        root,
        required={
            "kind",
            "schemaVersion",
            "collectionId",
            "sourceMediaPath",
            "renderedMediaPath",
            "renderArtifact",
            "policy",
            "sampling",
            "speakers",
            "cues",
        },
        label="$",
    )
    if root["kind"] != SUBTITLE_RENDER_EVIDENCE_REQUEST_KIND:
        _invalid("$.kind is unsupported")
    if root["schemaVersion"] != SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION:
        _invalid("$.schemaVersion is unsupported")
    collection_id = _require_text(
        root["collectionId"], "$.collectionId", maximum=160
    )
    source_path = _require_text(
        root["sourceMediaPath"], "$.sourceMediaPath", maximum=32_768
    )
    rendered_path = _require_text(
        root["renderedMediaPath"], "$.renderedMediaPath", maximum=32_768
    )

    artifact = _require_mapping(root["renderArtifact"], "$.renderArtifact")
    _require_keys(
        artifact,
        required={
            "renderer",
            "rendererVersion",
            "renderConfigurationSha256",
        },
        label="$.renderArtifact",
    )
    render_artifact = {
        "renderer": _require_text(
            artifact["renderer"],
            "$.renderArtifact.renderer",
            maximum=160,
        ),
        "rendererVersion": _require_text(
            artifact["rendererVersion"],
            "$.renderArtifact.rendererVersion",
            maximum=320,
        ),
        "renderConfigurationSha256": _require_sha256(
            artifact["renderConfigurationSha256"],
            "$.renderArtifact.renderConfigurationSha256",
        ),
    }

    policy = _canonical_mapping(root["policy"], "$.policy")
    sampling_raw = _require_mapping(root["sampling"], "$.sampling")
    _require_keys(
        sampling_raw,
        required={"strategy", "candidateFractions"},
        label="$.sampling",
    )
    strategy = _require_text(
        sampling_raw["strategy"], "$.sampling.strategy", maximum=160
    )
    if strategy != SUBTITLE_RENDER_EVIDENCE_STRATEGY:
        _invalid("$.sampling.strategy is unsupported")
    fractions_raw = sampling_raw["candidateFractions"]
    if not isinstance(fractions_raw, list) or not 1 <= len(
        fractions_raw
    ) <= 9:
        _invalid("$.sampling.candidateFractions must contain 1..9 values")
    fractions = [
        _require_number(
            value,
            f"$.sampling.candidateFractions[{index}]",
            minimum=0.01,
            maximum=0.99,
        )
        for index, value in enumerate(fractions_raw)
    ]
    if fractions != sorted(set(fractions)):
        _invalid(
            "$.sampling.candidateFractions must be unique and ascending"
        )

    speakers_raw = root["speakers"]
    if not isinstance(speakers_raw, list) or not 1 <= len(
        speakers_raw
    ) <= 10_000:
        _invalid("$.speakers must contain 1..10000 values")
    speakers: list[dict[str, str]] = []
    speaker_ids: set[str] = set()
    for index, raw in enumerate(speakers_raw):
        item = _require_mapping(raw, f"$.speakers[{index}]")
        _require_keys(
            item,
            required={"speakerId", "color"},
            label=f"$.speakers[{index}]",
        )
        speaker_id = _require_text(
            item["speakerId"],
            f"$.speakers[{index}].speakerId",
            maximum=160,
        )
        if speaker_id in speaker_ids:
            _invalid(f"duplicate speakerId: {speaker_id!r}")
        speaker_ids.add(speaker_id)
        speakers.append(
            {
                "speakerId": speaker_id,
                "color": _require_color(
                    item["color"],
                    f"$.speakers[{index}].color",
                ).upper(),
            }
        )

    cues_raw = root["cues"]
    if not isinstance(cues_raw, list) or not 1 <= len(
        cues_raw
    ) <= 1_000_000:
        _invalid("$.cues must contain 1..1000000 values")
    cues: list[dict[str, Any]] = []
    cue_ids: set[str] = set()
    for index, raw in enumerate(cues_raw):
        cue = _parse_cue(raw, index=index, speaker_ids=speaker_ids)
        if cue["cueId"] in cue_ids:
            _invalid(f"duplicate cueId: {cue['cueId']!r}")
        cue_ids.add(cue["cueId"])
        cues.append(cue)

    return {
        "kind": SUBTITLE_RENDER_EVIDENCE_REQUEST_KIND,
        "schemaVersion": SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION,
        "collectionId": collection_id,
        "sourceMediaPath": source_path,
        "renderedMediaPath": rendered_path,
        "renderArtifact": render_artifact,
        "policy": policy,
        "sampling": {
            "strategy": strategy,
            "candidateFractions": fractions,
        },
        "speakers": speakers,
        "cues": cues,
    }


def _parse_cue(
    value: Any,
    *,
    index: int,
    speaker_ids: set[str],
) -> dict[str, Any]:
    path = f"$.cues[{index}]"
    item = _require_mapping(value, path)
    _require_keys(
        item,
        required={
            "cueId",
            "startMs",
            "endMs",
            "text",
            "speakerId",
            "styleId",
            "renderedLines",
            "karaokeMode",
            "wordTimingEvidence",
            "bounds",
            "requestedFontFamilies",
        },
        label=path,
    )
    cue_id = _require_text(item["cueId"], f"{path}.cueId", maximum=160)
    start_ms = _require_integer(
        item["startMs"], f"{path}.startMs", minimum=0, maximum=604_800_000
    )
    end_ms = _require_integer(
        item["endMs"], f"{path}.endMs", minimum=1, maximum=604_800_000
    )
    if end_ms <= start_ms:
        _invalid(f"{path}.endMs must be greater than startMs")
    text = _require_text(item["text"], f"{path}.text", maximum=100_000)
    speaker_id = _require_text(
        item["speakerId"], f"{path}.speakerId", maximum=160
    )
    if speaker_id not in speaker_ids:
        _invalid(f"{path}.speakerId does not reference $.speakers")
    style_id = _require_text(
        item["styleId"], f"{path}.styleId", maximum=160
    )
    lines_raw = item["renderedLines"]
    if not isinstance(lines_raw, list) or not 1 <= len(lines_raw) <= 20:
        _invalid(f"{path}.renderedLines must contain 1..20 values")
    rendered_lines = [
        _require_text(
            line,
            f"{path}.renderedLines[{line_index}]",
            maximum=10_000,
        )
        for line_index, line in enumerate(lines_raw)
    ]
    karaoke_mode = _require_enum(
        item["karaokeMode"],
        f"{path}.karaokeMode",
        frozenset({"none", "word-progress"}),
    )
    word_evidence = _parse_word_timing(
        item["wordTimingEvidence"],
        path=f"{path}.wordTimingEvidence",
        cue_text=text,
        cue_start_ms=start_ms,
        cue_end_ms=end_ms,
    )
    bounds = _parse_rect(item["bounds"], f"{path}.bounds")
    families_raw = item["requestedFontFamilies"]
    if not isinstance(families_raw, list) or not 1 <= len(
        families_raw
    ) <= 64:
        _invalid(f"{path}.requestedFontFamilies must contain 1..64 values")
    families = [
        _require_text(
            family,
            f"{path}.requestedFontFamilies[{family_index}]",
            maximum=320,
        )
        for family_index, family in enumerate(families_raw)
    ]
    if len(families) != len(set(families)):
        _invalid(f"{path}.requestedFontFamilies must be unique")
    return {
        "cueId": cue_id,
        "startMs": start_ms,
        "endMs": end_ms,
        "text": text,
        "speakerId": speaker_id,
        "styleId": style_id,
        "renderedLines": rendered_lines,
        "karaokeMode": karaoke_mode,
        "wordTimingEvidence": word_evidence,
        "bounds": bounds,
        "requestedFontFamilies": families,
    }


def _parse_word_timing(
    value: Any,
    *,
    path: str,
    cue_text: str,
    cue_start_ms: int,
    cue_end_ms: int,
) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _require_mapping(value, path)
    _require_keys(
        item,
        required={
            "source",
            "verified",
            "evidenceArtifactSha256",
            "cueTextSha256",
            "words",
        },
        label=path,
    )
    source = _require_enum(
        item["source"],
        f"{path}.source",
        _AUTHENTIC_WORD_TIMING_SOURCES,
    )
    if item["verified"] is not True:
        _invalid(f"{path}.verified must be true")
    evidence_hash = _require_sha256(
        item["evidenceArtifactSha256"],
        f"{path}.evidenceArtifactSha256",
    )
    cue_hash = _require_sha256(
        item["cueTextSha256"], f"{path}.cueTextSha256"
    )
    expected_hash = hashlib.sha256(cue_text.encode("utf-8")).hexdigest()
    if cue_hash != expected_hash:
        _invalid(f"{path}.cueTextSha256 does not bind the exact cue text")
    words_raw = item["words"]
    if not isinstance(words_raw, list) or len(words_raw) > 100_000:
        _invalid(f"{path}.words must contain at most 100000 values")
    words: list[dict[str, Any]] = []
    previous_end = cue_start_ms
    for index, raw in enumerate(words_raw):
        word_path = f"{path}.words[{index}]"
        word = _require_mapping(raw, word_path)
        _require_keys(
            word,
            required={"text", "startMs", "endMs"},
            label=word_path,
        )
        text = _require_text(
            word["text"], f"{word_path}.text", maximum=1_000
        )
        start_ms = _require_integer(
            word["startMs"],
            f"{word_path}.startMs",
            minimum=0,
            maximum=604_800_000,
        )
        end_ms = _require_integer(
            word["endMs"],
            f"{word_path}.endMs",
            minimum=1,
            maximum=604_800_000,
        )
        if (
            end_ms <= start_ms
            or start_ms < cue_start_ms
            or end_ms > cue_end_ms
            or start_ms < previous_end
        ):
            _invalid(f"{word_path} contains invalid or overlapping timing")
        previous_end = end_ms
        words.append({"text": text, "startMs": start_ms, "endMs": end_ms})
    if _without_whitespace("".join(word["text"] for word in words)) != (
        _without_whitespace(cue_text)
    ):
        _invalid(f"{path}.words do not reconstruct the cue text")
    return {
        "source": source,
        "verified": True,
        "evidenceArtifactSha256": evidence_hash,
        "cueTextSha256": cue_hash,
        "words": words,
    }


def _qa_cue(cue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cueId": cue["cueId"],
        "startMs": cue["startMs"],
        "endMs": cue["endMs"],
        "text": cue["text"],
        "speakerId": cue["speakerId"],
        "styleId": cue["styleId"],
        "renderedLines": copy.deepcopy(cue["renderedLines"]),
        "karaokeMode": cue["karaokeMode"],
        "wordTimingEvidence": copy.deepcopy(cue["wordTimingEvidence"]),
    }


def _canonical_media_path(value: str, *, label: str) -> Path:
    if "\x00" in value or _URL_SCHEME.match(value):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            f"{label} must be a local filesystem path.",
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            f"{label} must be absolute.",
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INPUT_MISSING,
            f"{label} does not resolve to an existing file.",
            detail=str(exc),
        ) from exc
    if not resolved.is_file():
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            f"{label} must be a regular file.",
        )
    return resolved


def _canonical_tool_path(value: str | Path, *, label: str) -> Path:
    raw = str(value)
    if "\x00" in raw or _URL_SCHEME.match(raw):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            f"{label} must be an explicit local executable path.",
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            f"{label} must be an absolute path.",
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.TOOL_UNAVAILABLE,
            f"{label} does not resolve to an existing file.",
            detail=str(exc),
        ) from exc
    if not resolved.is_file():
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.TOOL_UNAVAILABLE,
            f"{label} must identify a regular file.",
        )
    return resolved


def _canonical_evidence_file(value: str | Path, *, label: str) -> Path:
    return _canonical_media_path(str(value), label=label)


def _canonical_private_ass_path(
    value: str | Path,
    *,
    maximum_bytes: int,
) -> Path:
    raw = str(value)
    if (
        not raw
        or len(raw) > 32_768
        or "\x00" in raw
        or _URL_SCHEME.match(raw)
    ):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical ASS overlay must be an explicit local path.",
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical ASS overlay path must be absolute.",
        )
    try:
        if candidate.is_symlink():
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_PATH,
                "The canonical ASS overlay must not be a symbolic link.",
            )
        resolved = candidate.resolve(strict=True)
        stat = resolved.stat()
    except SubtitleVisualEvidenceError:
        raise
    except (OSError, RuntimeError) as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INPUT_MISSING,
            "The canonical ASS overlay is unavailable.",
            detail=str(exc),
        ) from exc
    if not resolved.is_file() or resolved.suffix.lower() != ".ass":
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical subtitle overlay must be a regular .ass file.",
        )
    if not 1 <= stat.st_size <= maximum_bytes:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical ASS overlay is empty or exceeds its byte limit.",
        )
    return resolved


def _validate_ass_payload(path: Path, *, maximum_bytes: int) -> None:
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical ASS overlay could not be read.",
            detail=str(exc),
        ) from exc
    if not payload or len(payload) > maximum_bytes:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical ASS overlay is empty or exceeds its byte limit.",
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
            "The canonical ASS overlay must be strict UTF-8.",
            detail=str(exc),
        ) from exc
    casefolded = text.casefold()
    if (
        "[script info]" not in casefolded
        or "[events]" not in casefolded
        or not any(
            line.lstrip().casefold().startswith("dialogue:")
            for line in text.splitlines()
        )
    ):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
            "The canonical ASS overlay lacks required Script Info, Events, "
            "or Dialogue evidence.",
        )


def _read_ass_text(path: Path, *, maximum_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical ASS overlay could not be read for contrast analysis.",
            detail=str(exc),
        ) from exc
    if not payload or len(payload) > maximum_bytes:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            "The canonical ASS overlay is empty or exceeds its byte limit.",
        )
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
            "The canonical ASS overlay must be strict UTF-8.",
            detail=str(exc),
        ) from exc


def _build_contrast_ass_variants(
    ass_text: str,
) -> tuple[bytes, bytes, ComponentDescriptor]:
    """Create deterministic box-only and glyph-fill-only ASS carriers."""

    lines = ass_text.splitlines()
    background_lines = list(lines)
    glyph_lines = list(lines)
    section: str | None = None
    seen_sections: set[str] = set()
    style_fields: list[str] | None = None
    event_fields: list[str] | None = None
    style_names: set[str] = set()
    dialogue_styles: list[str] = []
    style_count = 0
    dialogue_count = 0
    border_styles: set[int] = set()

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.casefold()
            if section in {"[v4+ styles]", "[events]"}:
                if section in seen_sections:
                    _invalid(
                        "canonical ASS contains duplicate style or event sections"
                    )
                seen_sections.add(section)
            continue

        if section == "[v4+ styles]":
            if stripped.casefold().startswith("format:"):
                if style_fields is not None:
                    _invalid("canonical ASS contains duplicate style formats")
                style_fields = _ass_format_fields(stripped, label="style")
                _require_ass_fields(
                    style_fields,
                    (
                        "name",
                        "primarycolour",
                        "secondarycolour",
                        "outlinecolour",
                        "backcolour",
                        "borderstyle",
                        "outline",
                        "shadow",
                    ),
                    label="style",
                )
                continue
            if stripped.casefold().startswith("style:"):
                if style_fields is None:
                    _invalid("canonical ASS style appears before its format")
                values = _ass_record_values(
                    stripped,
                    field_count=len(style_fields),
                    label="style",
                )
                field_index = {
                    name: position
                    for position, name in enumerate(style_fields)
                }
                style_name = values[field_index["name"]].strip()
                if not style_name or style_name in style_names:
                    _invalid("canonical ASS style names must be unique and non-empty")
                style_names.add(style_name)
                try:
                    border_style = int(
                        values[field_index["borderstyle"]].strip()
                    )
                except ValueError as exc:
                    raise SubtitleVisualEvidenceError(
                        SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
                        "canonical ASS BorderStyle must be an integer",
                    ) from exc
                if border_style not in {1, 3}:
                    _invalid(
                        "canonical ASS contrast mattes support only "
                        "BorderStyle 1 or 3"
                    )
                border_styles.add(border_style)

                background = list(values)
                background[field_index["primarycolour"]] = "&HFF000000"
                background[field_index["secondarycolour"]] = "&HFF000000"
                background[field_index["shadow"]] = "0"
                if border_style == 1:
                    background[field_index["outlinecolour"]] = "&HFF000000"
                    background[field_index["backcolour"]] = "&HFF000000"
                    background[field_index["outline"]] = "0"

                glyph = list(values)
                glyph[field_index["primarycolour"]] = "&H00FFFFFF"
                glyph[field_index["secondarycolour"]] = "&H00FFFFFF"
                glyph[field_index["outlinecolour"]] = "&HFF000000"
                glyph[field_index["backcolour"]] = "&HFF000000"
                glyph[field_index["borderstyle"]] = "1"
                glyph[field_index["outline"]] = "0"
                glyph[field_index["shadow"]] = "0"

                background_lines[index] = "Style: " + ",".join(background)
                glyph_lines[index] = "Style: " + ",".join(glyph)
                style_count += 1
                continue

        if section == "[events]":
            if stripped.casefold().startswith("format:"):
                if event_fields is not None:
                    _invalid("canonical ASS contains duplicate event formats")
                event_fields = _ass_format_fields(stripped, label="event")
                _require_ass_fields(
                    event_fields,
                    ("style", "text"),
                    label="event",
                )
                continue
            if stripped.casefold().startswith("dialogue:"):
                if event_fields is None:
                    _invalid("canonical ASS dialogue appears before its format")
                values = _ass_record_values(
                    stripped,
                    field_count=len(event_fields),
                    label="dialogue",
                )
                field_index = {
                    name: position
                    for position, name in enumerate(event_fields)
                }
                dialogue_text = values[field_index["text"]]
                if _contains_unescaped_ass_override(dialogue_text):
                    _invalid(
                        "canonical ASS inline override blocks cannot be "
                        "normalized safely for contrast mattes"
                    )
                dialogue_styles.append(
                    values[field_index["style"]].strip() or "Default"
                )
                dialogue_count += 1

    if seen_sections != {"[v4+ styles]", "[events]"}:
        _invalid("canonical ASS lacks unique V4+ Styles or Events sections")
    if style_count < 1 or dialogue_count < 1:
        _invalid("canonical ASS must contain styles and dialogue events")
    missing_styles = sorted(set(dialogue_styles) - style_names)
    if missing_styles:
        _invalid(
            "canonical ASS dialogue references undefined styles: "
            + repr(missing_styles)
        )

    background_payload = ("\n".join(background_lines) + "\n").encode("utf-8")
    glyph_payload = ("\n".join(glyph_lines) + "\n").encode("utf-8")
    transform_configuration = {
        "backgroundStrategy": CONTRAST_BACKGROUND_STRATEGY,
        "glyphFillStrategy": GLYPH_FILL_MATTE_STRATEGY,
        "supportedBorderStyles": sorted(border_styles),
        "styleCount": style_count,
        "dialogueCount": dialogue_count,
        "inlineOverridePolicy": "fail-closed",
        "backgroundRule": (
            "transparent-fill-preserve-borderstyle-3-box-remove-shadow"
        ),
        "glyphRule": "opaque-white-fill-transparent-outline-box-shadow",
    }
    descriptor = ComponentDescriptor(
        name=CONTRAST_MATTE_TRANSFORM_NAME,
        version=CONTRAST_MATTE_TRANSFORM_VERSION,
        configuration_sha256=deterministic_sha256(transform_configuration),
    )
    return background_payload, glyph_payload, descriptor


def _ass_format_fields(line: str, *, label: str) -> list[str]:
    _prefix, separator, payload = line.partition(":")
    if not separator:
        _invalid(f"canonical ASS {label} format is malformed")
    fields = [field.strip().casefold() for field in payload.split(",")]
    if not fields or any(not field for field in fields):
        _invalid(f"canonical ASS {label} format contains an empty field")
    if len(fields) != len(set(fields)):
        _invalid(f"canonical ASS {label} format contains duplicate fields")
    return fields


def _require_ass_fields(
    fields: Sequence[str],
    required: Sequence[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(required) - set(fields))
    if missing:
        _invalid(
            f"canonical ASS {label} format lacks required fields: {missing!r}"
        )


def _ass_record_values(
    line: str,
    *,
    field_count: int,
    label: str,
) -> list[str]:
    _prefix, separator, payload = line.partition(":")
    if not separator:
        _invalid(f"canonical ASS {label} record is malformed")
    values = payload.lstrip().split(",", field_count - 1)
    if len(values) != field_count:
        _invalid(
            f"canonical ASS {label} record does not match its format"
        )
    return values


def _contains_unescaped_ass_override(text: str) -> bool:
    return any(
        character in "{}" and (index == 0 or text[index - 1] != "\\")
        for index, character in enumerate(text)
    )


def _canonical_delivery_receipt(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload: Any = value
    else:
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            _invalid(
                "delivery_receipt must be an object or expose to_dict()"
            )
        try:
            payload = to_dict()
        except Exception as exc:
            raise SubtitleVisualEvidenceError(
                SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
                "delivery_receipt.to_dict() failed.",
                detail=str(exc),
            ) from exc
    canonical = _canonical_mapping(payload, "delivery_receipt")
    if len(canonical_json(canonical).encode("utf-8")) > 2 * 1024 * 1024:
        _invalid("delivery_receipt exceeds the 2 MiB evidence limit")
    return canonical


def _validate_delivery_receipt(
    receipt: Mapping[str, Any],
    *,
    source_snapshot: Mapping[str, Any],
    rendered_snapshot: Mapping[str, Any],
) -> str:
    mode = _require_enum(
        receipt.get("mode"),
        "delivery_receipt.mode",
        frozenset({"soft-mux", "burn-in"}),
    )
    output = _require_mapping(
        receipt.get("outputEvidence"),
        "delivery_receipt.outputEvidence",
    )
    output_sha256 = _require_sha256(
        output.get("sha256"),
        "delivery_receipt.outputEvidence.sha256",
    )
    output_size = _require_integer(
        output.get("sizeBytes"),
        "delivery_receipt.outputEvidence.sizeBytes",
        minimum=1,
        maximum=sys.maxsize,
    )
    if (
        output_sha256 != rendered_snapshot["sha256"]
        or output_size != rendered_snapshot["sizeBytes"]
    ):
        _invalid(
            "delivery_receipt output evidence does not bind the rendered "
            "media artifact"
        )

    integrity = _require_mapping(
        receipt.get("sourceIntegrity"),
        "delivery_receipt.sourceIntegrity",
    )
    if integrity.get("unchanged") is not True:
        _invalid("delivery_receipt must prove source integrity")
    before = _require_mapping(
        integrity.get("before"),
        "delivery_receipt.sourceIntegrity.before",
    )
    after = _require_mapping(
        integrity.get("after"),
        "delivery_receipt.sourceIntegrity.after",
    )
    for label, item in (("before", before), ("after", after)):
        sha256 = _require_sha256(
            item.get("sha256"),
            f"delivery_receipt.sourceIntegrity.{label}.sha256",
        )
        size = _require_integer(
            item.get("sizeBytes"),
            f"delivery_receipt.sourceIntegrity.{label}.sizeBytes",
            minimum=1,
            maximum=sys.maxsize,
        )
        if (
            sha256 != source_snapshot["sha256"]
            or size != source_snapshot["sizeBytes"]
        ):
            _invalid(
                "delivery_receipt source evidence does not bind the source "
                "media artifact"
            )
    qa = _require_mapping(receipt.get("qa"), "delivery_receipt.qa")
    if qa.get("passed") is not True or qa.get("outputNonEmpty") is not True:
        _invalid("delivery_receipt must contain passing delivery QA")
    if mode == "soft-mux" and qa.get("subtitleStreamVerified") is not True:
        _invalid(
            "soft-mux delivery_receipt must verify its subtitle stream"
        )
    return mode


def _hash_file(path: Path, *, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_PATH,
            f"Evidence file could not be read: {path}",
            detail=str(exc),
        ) from exc
    return digest.hexdigest()


def _hash_bounded_file(
    path: Path,
    *,
    maximum: int,
    chunk_bytes: int,
) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.FRAME_EXTRACTION_FAILED,
            f"Representative frame is unavailable: {path}",
            detail=str(exc),
        ) from exc
    if size < 1:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.FRAME_EXTRACTION_FAILED,
            "Representative frame is empty.",
        )
    if size > maximum:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.FRAME_OUTPUT_LIMIT,
            "Representative frame exceeds the configured byte limit.",
        )
    return _hash_file(path, chunk_bytes=chunk_bytes)


def _remove_tree(path: Path) -> str | None:
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            shutil.rmtree(path)
            return None
        except FileNotFoundError:
            return None
        except OSError as exc:
            last_error = exc
            time.sleep(0.025 * (attempt + 1))
    return str(last_error) if last_error is not None else "unknown cleanup error"


def _decode_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.PROBE_FAILED,
            f"{label} output is not valid UTF-8 JSON.",
            detail=str(exc),
        ) from exc
    return dict(_require_mapping(value, label))


def _first_nonempty_line(*payloads: bytes) -> str | None:
    for payload in payloads:
        text = payload.decode("utf-8", errors="replace")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def _bounded_detail(payload: bytes, maximum: int = 2_000) -> str:
    return payload[:maximum].decode("utf-8", errors="replace")


def _fractional_timestamp(
    start_ms: int,
    end_ms: int,
    fraction: float,
) -> int:
    ratio = Fraction(str(fraction))
    duration = end_ms - start_ms
    offset = (duration * ratio.numerator + ratio.denominator // 2) // (
        ratio.denominator
    )
    value = start_ms + offset
    if duration > 1:
        return min(max(value, start_ms + 1), end_ms - 1)
    return start_ms


def _seconds_text(timestamp_ms: int) -> str:
    return f"{timestamp_ms // 1000}.{timestamp_ms % 1000:03d}"


def _histogram_percentile(
    histogram: Sequence[int],
    sample_count: int,
    percentile: float,
) -> int:
    if sample_count <= 0:
        return 0
    rank = max(1, int(sample_count * percentile + 0.999999999))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= rank:
            return value
    return len(histogram) - 1


def _nearest_rank_quantile(
    values: Sequence[int],
    quantile: float,
) -> int:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * quantile + 0.999999999))
    return ordered[min(len(ordered), rank) - 1]


def _quantile_pixel(
    pixels: Sequence[Mapping[str, Any]],
    quantile: float,
) -> dict[str, Any]:
    if not pixels:
        raise ValueError("pixel quantile requires at least one value")
    ordered = sorted(
        pixels,
        key=lambda item: (
            item["ratio"],
            item["y"],
            item["x"],
        ),
    )
    rank = max(1, int(len(ordered) * quantile + 0.999999999))
    return dict(ordered[min(len(ordered), rank) - 1])


def _coordinate_bounds(
    coordinates: Sequence[tuple[int, int]],
) -> dict[str, int]:
    if not coordinates:
        raise ValueError("coordinate bounds require at least one point")
    x_values = [coordinate[0] for coordinate in coordinates]
    y_values = [coordinate[1] for coordinate in coordinates]
    return {
        "x": min(x_values),
        "y": min(y_values),
        "width": max(x_values) - min(x_values) + 1,
        "height": max(y_values) - min(y_values) + 1,
    }


def _luminance_class(
    color: tuple[int, int, int],
    *,
    dark_maximum: float,
    light_minimum: float,
) -> str | None:
    luminance = _relative_luminance_tuple(color)
    if luminance <= dark_maximum:
        return "dark"
    if luminance >= light_minimum:
        return "light"
    return None


def _relative_luminance_tuple(color: tuple[int, int, int]) -> float:
    channels: list[float] = []
    for value in color:
        component = value / 255.0
        channels.append(
            component / 12.92
            if component <= 0.04045
            else ((component + 0.055) / 1.055) ** 2.4
        )
    return (
        0.2126 * channels[0]
        + 0.7152 * channels[1]
        + 0.0722 * channels[2]
    )


def _rgb_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def _renderable_codepoint_count(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def _without_whitespace(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def _parse_rect(
    value: Any,
    label: str,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> dict[str, int]:
    item = _require_mapping(value, label)
    _require_keys(item, required={"x", "y", "width", "height"}, label=label)
    rect = {
        "x": _require_integer(
            item["x"], f"{label}.x", minimum=0, maximum=1_000_000
        ),
        "y": _require_integer(
            item["y"], f"{label}.y", minimum=0, maximum=1_000_000
        ),
        "width": _require_integer(
            item["width"],
            f"{label}.width",
            minimum=1,
            maximum=1_000_000,
        ),
        "height": _require_integer(
            item["height"],
            f"{label}.height",
            minimum=1,
            maximum=1_000_000,
        ),
    }
    if frame_width is not None and (
        rect["x"] + rect["width"] > frame_width
    ):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
            f"{label} exceeds the frame width.",
        )
    if frame_height is not None and (
        rect["y"] + rect["height"] > frame_height
    ):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
            f"{label} exceeds the frame height.",
        )
    return rect


def _canonical_mapping(value: Any, label: str) -> dict[str, Any]:
    item = _require_mapping(value, label)
    try:
        return json.loads(canonical_json(item))
    except json.JSONDecodeError as exc:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
            f"{label} is not canonical JSON.",
        ) from exc


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid(f"{label} must be an object")
    return value


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        _invalid(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _require_text(
    value: Any,
    label: str,
    *,
    maximum: int,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or value != value.strip()
        or "\x00" in value
    ):
        _invalid(f"{label} must be trimmed non-empty text")
    return value


def _require_integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        _invalid(f"{label} must be an integer in {minimum}..{maximum}")
    return value


def _require_number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= float(value) <= maximum
    ):
        _invalid(f"{label} must be a number in {minimum}..{maximum}")
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        _invalid(f"{label} must be finite")
    return numeric


def _require_enum(
    value: Any,
    label: str,
    allowed: frozenset[str],
) -> str:
    if not isinstance(value, str) or value not in allowed:
        _invalid(f"{label} must be one of {sorted(allowed)!r}")
    return value


def _normalize_background_classes(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
            f"{label} must be an array of observed background classes.",
        )
    classes = list(value)
    if any(item not in {"dark", "light"} for item in classes):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
            f"{label} contains an unsupported background class.",
        )
    if len(classes) != len(set(classes)) or classes != sorted(classes):
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE,
            f"{label} must be sorted and unique.",
        )
    return classes


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _invalid(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_color(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _COLOR.fullmatch(value):
        _invalid(f"{label} must be an RGB #RRGGBB color")
    return value


def _invalid(message: str) -> None:
    raise SubtitleVisualEvidenceError(
        SubtitleVisualEvidenceErrorCode.INVALID_REQUEST,
        message,
    )


__all__ = [
    "BoundedEvidenceRunner",
    "ComponentDescriptor",
    "ContrastObservation",
    "CueFrameObservation",
    "EvidenceProcessLimits",
    "EvidenceProcessResult",
    "EvidenceRunner",
    "FontEvidenceObservation",
    "FontEvidenceProvider",
    "FrameAnalyzer",
    "FrameObservation",
    "GLYPH_CORE_COMPONENT_STRATEGY",
    "NoFontEvidenceProvider",
    "PillowAnalysisPolicy",
    "PillowFrameAnalyzer",
    "SUBTITLE_RENDER_EVIDENCE_REQUEST_KIND",
    "SUBTITLE_RENDER_EVIDENCE_RESULT_KIND",
    "SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION",
    "SUBTITLE_RENDER_EVIDENCE_STRATEGY",
    "SubtitleRenderEvidenceResult",
    "SubtitleVisualEvidenceCollector",
    "SubtitleVisualEvidenceError",
    "SubtitleVisualEvidenceErrorCode",
    "SubtitleVisualEvidencePolicy",
    "TINY_COMPONENT_MAX_PIXELS",
    "TINY_COMPONENT_MINIMUM_ALPHA",
    "VerifiedFontClaim",
    "canonical_json",
    "default_subtitle_visual_evidence_sampling",
    "deterministic_sha256",
    "verify_evidence_artifact_hash",
]
