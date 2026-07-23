"""Source-immutable FFmpeg subtitle delivery with fail-closed publication.

The pure subtitle domain in :mod:`backend.subtitles` deliberately stops at a
reviewable plan.  This module is the execution boundary that resolves that
plan against trusted :class:`backend.media_probe.MediaProbeResult` evidence,
the exact local FFmpeg capability listing, bounded argv-only process
execution, post-write probing, and no-clobber atomic publication.

No command is executed through a shell.  Source media is never opened for
writing.  Every artifact is first created under an unpredictable temporary
name in the final output directory and is published with an atomic hard-link
create that fails if the destination appears concurrently.
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
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from backend.media_probe import (
    MediaProbe,
    MediaProbeResult,
    ProcessResult,
    canonical_local_media_file,
)
from backend.subtitles import (
    SubtitleFormat,
    SubtitleOutputMode,
    SubtitleOutputPlan,
)


SUBTITLE_DELIVERY_SCHEMA_VERSION = "1.0.0"
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SAFE_LANGUAGE = re.compile(
    r"^(?:und|[A-Za-z]{2,8})(?:-[A-Za-z0-9]{1,8})*$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KARAOKE_TAG = re.compile(
    r"\{[^}\r\n]*\\(?:k|K|kf|ko)\s*\d+[^}\r\n]*\}"
)
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


class SubtitleDeliveryErrorCode(str, Enum):
    INVALID_PLAN = "invalid-plan"
    INVALID_PATH = "invalid-path"
    INPUT_MISSING = "input-missing"
    OUTPUT_EXISTS = "output-exists"
    PATH_ALIAS = "path-alias"
    SUBTITLE_INVALID = "subtitle-invalid"
    SUBTITLE_TOO_LARGE = "subtitle-too-large"
    KARAOKE_EVIDENCE_REQUIRED = "karaoke-evidence-required"
    CONTAINER_UNSUPPORTED = "container-unsupported"
    CODEC_UNSUPPORTED = "codec-unsupported"
    VIDEO_STRATEGY_REQUIRED = "video-strategy-required"
    VIDEO_STRATEGY_UNSUPPORTED = "video-strategy-unsupported"
    HDR_BURN_IN_UNSUPPORTED = "hdr-burn-in-unsupported"
    CAPABILITY_PROBE_FAILED = "capability-probe-failed"
    TOOL_UNAVAILABLE = "tool-unavailable"
    PROCESS_TIMEOUT = "process-timeout"
    PROCESS_OUTPUT_LIMIT = "process-output-limit"
    PROCESS_FAILED = "process-failed"
    OUTPUT_MISSING = "output-missing"
    OUTPUT_EMPTY = "output-empty"
    DURATION_UNAVAILABLE = "duration-unavailable"
    QA_FAILED = "qa-failed"
    SOURCE_CHANGED = "source-changed"
    PUBLISH_FAILED = "publish-failed"
    CLEANUP_FAILED = "cleanup-failed"


class SubtitleDeliveryError(RuntimeError):
    """Structured, fail-closed delivery failure."""

    def __init__(
        self,
        code: SubtitleDeliveryErrorCode,
        message: str,
        *,
        detail: str | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail
        self.evidence = dict(evidence or {})


@dataclass(frozen=True)
class DeliveryProcessLimits:
    """Bounded execution limits suitable for long local transcodes."""

    timeout_seconds: float = 4 * 60 * 60
    max_stdout_bytes: int = 512 * 1024
    max_stderr_bytes: int = 8 * 1024 * 1024
    poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not 0.05 <= self.timeout_seconds <= 24 * 60 * 60:
            raise ValueError("timeout_seconds must be between 0.05 and 86400")
        if not 1_024 <= self.max_stdout_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_stdout_bytes must be between 1 KiB and 64 MiB")
        if not 1_024 <= self.max_stderr_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_stderr_bytes must be between 1 KiB and 64 MiB")
        if not 0.005 <= self.poll_interval_seconds <= 1.0:
            raise ValueError("poll_interval_seconds must be between 0.005 and 1")


class DeliveryRunner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        limits: DeliveryProcessLimits,
    ) -> ProcessResult: ...


class TrustedMediaProbe(Protocol):
    def probe(self, source: str | Path) -> MediaProbeResult: ...


class BoundedDeliveryRunner:
    """Run a trusted argv vector with ``shell=False`` and bounded diagnostics."""

    def run(
        self,
        command: Sequence[str],
        *,
        limits: DeliveryProcessLimits,
    ) -> ProcessResult:
        argv = tuple(str(part) for part in command)
        if not argv or any(not part or "\x00" in part for part in argv):
            raise ValueError("command must contain non-empty, non-NUL argv entries")

        creationflags = 0
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))

        started = time.monotonic()
        root = Path(tempfile.mkdtemp(prefix="mts-subtitle-delivery-"))
        stdout_path = root / "stdout.bin"
        stderr_path = root / "stderr.bin"
        try:
            try:
                with stdout_path.open("wb") as stdout_handle, stderr_path.open(
                    "wb"
                ) as stderr_handle:
                    process = subprocess.Popen(  # noqa: S603 - fixed argv boundary
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        shell=False,
                        close_fds=True,
                        creationflags=creationflags,
                    )
                    while process.poll() is None:
                        if time.monotonic() - started > limits.timeout_seconds:
                            self._stop(process)
                            raise SubtitleDeliveryError(
                                SubtitleDeliveryErrorCode.PROCESS_TIMEOUT,
                                "FFmpeg exceeded the configured delivery timeout.",
                            )
                        stdout_handle.flush()
                        stderr_handle.flush()
                        if (
                            stdout_handle.tell() > limits.max_stdout_bytes
                            or stderr_handle.tell() > limits.max_stderr_bytes
                        ):
                            self._stop(process)
                            raise SubtitleDeliveryError(
                                SubtitleDeliveryErrorCode.PROCESS_OUTPUT_LIMIT,
                                "FFmpeg exceeded the bounded diagnostic output allowance.",
                            )
                        time.sleep(limits.poll_interval_seconds)
                    returncode = int(process.returncode or 0)
            except FileNotFoundError as exc:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.TOOL_UNAVAILABLE,
                    f"The required local media tool is unavailable: {argv[0]}",
                ) from exc
            except OSError as exc:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.TOOL_UNAVAILABLE,
                    f"The required local media tool could not be started: {argv[0]}",
                    detail=str(exc),
                ) from exc

            stdout = self._read_bounded(stdout_path, limits.max_stdout_bytes)
            stderr = self._read_bounded(stderr_path, limits.max_stderr_bytes)
            return ProcessResult(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                elapsed_ms=max(0, round((time.monotonic() - started) * 1_000)),
            )
        finally:
            cleanup_error = self._remove_capture_directory(root)
            if cleanup_error is not None:
                active_error = sys.exc_info()[1]
                if active_error is not None:
                    active_error.add_note(cleanup_error)
                else:
                    raise SubtitleDeliveryError(
                        SubtitleDeliveryErrorCode.CLEANUP_FAILED,
                        "Bounded FFmpeg diagnostic files could not be removed.",
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
    def _read_bounded(path: Path, maximum: int) -> bytes:
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
        if len(payload) > maximum:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.PROCESS_OUTPUT_LIMIT,
                "FFmpeg exceeded the bounded diagnostic output allowance.",
            )
        return payload

    @staticmethod
    def _remove_capture_directory(root: Path) -> str | None:
        last_error: OSError | None = None
        for attempt in range(8):
            try:
                shutil.rmtree(root)
                return None
            except FileNotFoundError:
                return None
            except OSError as exc:
                last_error = exc
                time.sleep(0.025 * (attempt + 1))
        return str(last_error) if last_error is not None else "unknown cleanup error"


class BurnInVideoStrategy(str, Enum):
    H264_HIGH_QUALITY = "h264-high-quality"
    H265_HIGH_QUALITY = "h265-high-quality"
    VP9_HIGH_QUALITY = "vp9-high-quality"
    AV1_HIGH_QUALITY = "av1-high-quality"
    PRORES_422_HQ = "prores-422-hq"
    FFV1_LOSSLESS = "ffv1-lossless"


class KaraokeTimingSource(str, Enum):
    FORCED_ALIGNER = "forced-aligner"
    NATIVE_WORD_TIMESTAMPS = "native-word-timestamps"
    HUMAN_AUTHORED = "human-authored"


@dataclass(frozen=True)
class KaraokeTimingEvidence:
    """Evidence required before pre-timed ASS karaoke tags may be delivered."""

    source: KaraokeTimingSource | str
    subtitle_sha256: str
    word_count: int
    verified: bool = True

    def __post_init__(self) -> None:
        try:
            resolved = KaraokeTimingSource(self.source)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported karaoke timing evidence source") from exc
        object.__setattr__(self, "source", resolved)
        if not self.verified:
            raise ValueError("karaoke timing evidence must be verified")
        if not _SHA256.fullmatch(self.subtitle_sha256):
            raise ValueError("subtitle_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.word_count, bool) or self.word_count < 1:
            raise ValueError("word_count must be a positive integer")


@dataclass(frozen=True)
class SubtitleDeliveryPolicy:
    process_limits: DeliveryProcessLimits = field(
        default_factory=DeliveryProcessLimits
    )
    capability_limits: DeliveryProcessLimits = field(
        default_factory=lambda: DeliveryProcessLimits(
            timeout_seconds=30,
            max_stdout_bytes=16 * 1024 * 1024,
            max_stderr_bytes=2 * 1024 * 1024,
        )
    )
    maximum_subtitle_bytes: int = 64 * 1024 * 1024
    duration_absolute_tolerance_ms: int = 750
    duration_relative_tolerance: float = 0.005
    hash_chunk_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        if not 1_024 <= self.maximum_subtitle_bytes <= 512 * 1024 * 1024:
            raise ValueError("maximum_subtitle_bytes must be between 1 KiB and 512 MiB")
        if not 0 <= self.duration_absolute_tolerance_ms <= 10_000:
            raise ValueError("duration_absolute_tolerance_ms must be 0..10000")
        if not 0 <= self.duration_relative_tolerance <= 0.1:
            raise ValueError("duration_relative_tolerance must be 0..0.1")
        if not 64 * 1024 <= self.hash_chunk_bytes <= 64 * 1024 * 1024:
            raise ValueError("hash_chunk_bytes must be between 64 KiB and 64 MiB")


@dataclass(frozen=True)
class FileEvidence:
    path: str
    size_bytes: int
    modified_time_ns: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sizeBytes": self.size_bytes,
            "modifiedTimeNs": self.modified_time_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class FFmpegCapabilityEvidence:
    encoders: frozenset[str]
    filters: frozenset[str]
    muxers: frozenset[str]
    fingerprint_sha256: str


@dataclass(frozen=True)
class DeliveryQA:
    output_non_empty: bool
    duration_checked: bool
    source_duration_ms: int | None
    output_duration_ms: int | None
    duration_delta_ms: int | None
    duration_tolerance_ms: int | None
    source_audio_streams: int
    output_audio_streams: int | None
    source_video_streams: int
    output_video_streams: int | None
    source_subtitle_streams: int
    output_subtitle_streams: int | None
    expected_subtitle_codec: str | None
    subtitle_stream_verified: bool
    post_publish_probe_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": True,
            "outputNonEmpty": self.output_non_empty,
            "durationChecked": self.duration_checked,
            "sourceDurationMs": self.source_duration_ms,
            "outputDurationMs": self.output_duration_ms,
            "durationDeltaMs": self.duration_delta_ms,
            "durationToleranceMs": self.duration_tolerance_ms,
            "sourceAudioStreams": self.source_audio_streams,
            "outputAudioStreams": self.output_audio_streams,
            "sourceVideoStreams": self.source_video_streams,
            "outputVideoStreams": self.output_video_streams,
            "sourceSubtitleStreams": self.source_subtitle_streams,
            "outputSubtitleStreams": self.output_subtitle_streams,
            "expectedSubtitleCodec": self.expected_subtitle_codec,
            "subtitleStreamVerified": self.subtitle_stream_verified,
            "postPublishProbeVerified": self.post_publish_probe_verified,
        }


@dataclass(frozen=True)
class SubtitleDeliveryReceipt:
    mode: SubtitleOutputMode
    source_path: str
    output_path: str
    subtitle_path: str
    container: str | None
    selected_subtitle_codec: str | None
    selected_video_strategy: BurnInVideoStrategy | None
    command: tuple[str, ...] | None
    process_result: ProcessResult | None
    process_limits: DeliveryProcessLimits | None
    source_before: FileEvidence
    source_after: FileEvidence
    output_evidence: FileEvidence
    qa: DeliveryQA
    source_probe_fingerprint_sha256: str
    output_probe_fingerprint_sha256: str | None
    capability_fingerprint_sha256: str | None
    karaoke_timing_detected: bool
    karaoke_timing_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SUBTITLE_DELIVERY_SCHEMA_VERSION,
            "status": "delivered",
            "mode": self.mode.value,
            "sourcePath": self.source_path,
            "outputPath": self.output_path,
            "subtitlePath": self.subtitle_path,
            "container": self.container,
            "selectedSubtitleCodec": self.selected_subtitle_codec,
            "selectedVideoStrategy": (
                self.selected_video_strategy.value
                if self.selected_video_strategy is not None
                else None
            ),
            "command": (
                {
                    "executable": self.command[0],
                    "arguments": list(self.command[1:]),
                    "shell": False,
                    "timeoutSeconds": self.process_limits.timeout_seconds,
                    "maxStdoutBytes": self.process_limits.max_stdout_bytes,
                    "maxStderrBytes": self.process_limits.max_stderr_bytes,
                }
                if self.command is not None and self.process_limits is not None
                else None
            ),
            "process": (
                {
                    "returnCode": self.process_result.returncode,
                    "elapsedMs": self.process_result.elapsed_ms,
                    "stdoutBytes": len(self.process_result.stdout),
                    "stderrBytes": len(self.process_result.stderr),
                }
                if self.process_result is not None
                else None
            ),
            "sourceIntegrity": {
                "unchanged": self.source_before == self.source_after,
                "before": self.source_before.to_dict(),
                "after": self.source_after.to_dict(),
            },
            "outputEvidence": self.output_evidence.to_dict(),
            "qa": self.qa.to_dict(),
            "probeEvidence": {
                "sourceFingerprintSha256": (
                    self.source_probe_fingerprint_sha256
                ),
                "outputFingerprintSha256": (
                    self.output_probe_fingerprint_sha256
                ),
                "capabilityFingerprintSha256": (
                    self.capability_fingerprint_sha256
                ),
            },
            "publication": {
                "strategy": "same-directory-hard-link-no-replace",
                "temporaryFileRemoved": True,
                "sourceMediaImmutable": True,
                "outputPreexisted": False,
            },
            "karaokeTiming": {
                "detected": self.karaoke_timing_detected,
                "synthesized": False,
                "verifiedEvidence": self.karaoke_timing_verified,
            },
        }


@dataclass(frozen=True)
class StagedSubtitleMediaDelivery:
    """Validated media held outside the customer-visible output name."""

    customer_output_path: Path
    quarantine_path: Path
    quarantine_evidence: FileEvidence
    receipt: SubtitleDeliveryReceipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "quarantined",
            "mode": self.receipt.mode.value,
            "customerOutputPath": str(self.customer_output_path),
            "customerOutputAbsent": not os.path.lexists(
                self.customer_output_path
            ),
            "quarantine": {
                "path": str(self.quarantine_path),
                "privateName": True,
                "artifact": self.quarantine_evidence.to_dict(),
            },
            "sourceIntegrity": {
                "unchanged": (
                    self.receipt.source_before == self.receipt.source_after
                ),
                "before": self.receipt.source_before.to_dict(),
                "after": self.receipt.source_after.to_dict(),
            },
        }


@dataclass(frozen=True)
class SubtitleMediaPublication:
    """Evidence for QA-gated, atomic no-replace media publication."""

    receipt: SubtitleDeliveryReceipt
    quarantine_path: Path
    quarantine_evidence: FileEvidence
    published_evidence: FileEvidence
    visual_qa_evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "published",
            "customerOutputPath": self.receipt.output_path,
            "quarantinePath": str(self.quarantine_path),
            "quarantineArtifact": self.quarantine_evidence.to_dict(),
            "publishedArtifact": self.published_evidence.to_dict(),
            "visualQaEvidenceSha256": self.visual_qa_evidence_sha256,
            "strategy": "same-directory-hard-link-no-replace",
            "atomic": True,
            "noReplace": True,
            "quarantineRemoved": not os.path.lexists(
                self.quarantine_path
            ),
            "sourceMediaImmutable": (
                self.receipt.source_before == self.receipt.source_after
            ),
        }


@dataclass(frozen=True)
class SubtitleMediaRollback:
    """Best-effort cleanup and preservation evidence after a failed gate."""

    reason: str
    customer_output_path: Path
    quarantine_path: Path
    quarantine_evidence: FileEvidence
    customer_output_state: str
    quarantine_state: str
    source_unchanged: bool
    cleanup_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "rolled-back",
            "reason": self.reason,
            "customerOutputPath": str(self.customer_output_path),
            "customerOutputState": self.customer_output_state,
            "quarantine": {
                "path": str(self.quarantine_path),
                "state": self.quarantine_state,
                "artifact": self.quarantine_evidence.to_dict(),
            },
            "sourceMediaImmutable": self.source_unchanged,
            "cleanupError": self.cleanup_error,
        }


@dataclass(frozen=True)
class _ContainerProfile:
    name: str
    muxer: str
    soft_codecs: Mapping[SubtitleFormat, tuple[str, ...]]
    burn_strategies: frozenset[BurnInVideoStrategy]
    final_arguments: tuple[str, ...] = ()


_ALL_ISO_BURN = frozenset(
    {
        BurnInVideoStrategy.H264_HIGH_QUALITY,
        BurnInVideoStrategy.H265_HIGH_QUALITY,
        BurnInVideoStrategy.AV1_HIGH_QUALITY,
    }
)
_ALL_MATROSKA_BURN = frozenset(BurnInVideoStrategy)
_CONTAINER_BY_SUFFIX: dict[str, _ContainerProfile] = {
    ".mp4": _ContainerProfile(
        name="mp4",
        muxer="mp4",
        soft_codecs={
            SubtitleFormat.SRT: ("mov_text",),
            SubtitleFormat.WEBVTT: ("mov_text",),
            SubtitleFormat.ASS: ("mov_text",),
        },
        burn_strategies=_ALL_ISO_BURN,
        final_arguments=("-movflags", "+faststart"),
    ),
    ".m4v": _ContainerProfile(
        name="mp4",
        muxer="mp4",
        soft_codecs={
            SubtitleFormat.SRT: ("mov_text",),
            SubtitleFormat.WEBVTT: ("mov_text",),
            SubtitleFormat.ASS: ("mov_text",),
        },
        burn_strategies=_ALL_ISO_BURN,
        final_arguments=("-movflags", "+faststart"),
    ),
    ".mov": _ContainerProfile(
        name="mov",
        muxer="mov",
        soft_codecs={
            SubtitleFormat.SRT: ("mov_text",),
            SubtitleFormat.WEBVTT: ("mov_text",),
            SubtitleFormat.ASS: ("mov_text",),
        },
        burn_strategies=frozenset(
            {
                *_ALL_ISO_BURN,
                BurnInVideoStrategy.PRORES_422_HQ,
            }
        ),
    ),
    ".mkv": _ContainerProfile(
        name="matroska",
        muxer="matroska",
        soft_codecs={
            SubtitleFormat.SRT: ("subrip",),
            SubtitleFormat.WEBVTT: ("webvtt", "subrip"),
            SubtitleFormat.ASS: ("ass",),
        },
        burn_strategies=_ALL_MATROSKA_BURN,
    ),
    ".mka": _ContainerProfile(
        name="matroska",
        muxer="matroska",
        soft_codecs={
            SubtitleFormat.SRT: ("subrip",),
            SubtitleFormat.WEBVTT: ("webvtt", "subrip"),
            SubtitleFormat.ASS: ("ass",),
        },
        burn_strategies=frozenset(),
    ),
    ".webm": _ContainerProfile(
        name="webm",
        muxer="webm",
        soft_codecs={
            SubtitleFormat.SRT: ("webvtt",),
            SubtitleFormat.WEBVTT: ("webvtt",),
            SubtitleFormat.ASS: ("webvtt",),
        },
        burn_strategies=frozenset(
            {
                BurnInVideoStrategy.VP9_HIGH_QUALITY,
                BurnInVideoStrategy.AV1_HIGH_QUALITY,
            }
        ),
    ),
}

_VIDEO_STRATEGIES: dict[
    BurnInVideoStrategy,
    tuple[str, tuple[str, ...]],
] = {
    BurnInVideoStrategy.H264_HIGH_QUALITY: (
        "libx264",
        (
            "-preset:v:0",
            "slow",
            "-crf:v:0",
            "18",
            "-pix_fmt:v:0",
            "yuv420p",
        ),
    ),
    BurnInVideoStrategy.H265_HIGH_QUALITY: (
        "libx265",
        (
            "-preset:v:0",
            "slow",
            "-crf:v:0",
            "20",
            "-pix_fmt:v:0",
            "yuv420p",
            "-tag:v:0",
            "hvc1",
        ),
    ),
    BurnInVideoStrategy.VP9_HIGH_QUALITY: (
        "libvpx-vp9",
        (
            "-crf:v:0",
            "24",
            "-b:v:0",
            "0",
            "-row-mt:v:0",
            "1",
        ),
    ),
    BurnInVideoStrategy.AV1_HIGH_QUALITY: (
        "libsvtav1",
        (
            "-preset:v:0",
            "6",
            "-crf:v:0",
            "24",
        ),
    ),
    BurnInVideoStrategy.PRORES_422_HQ: (
        "prores_ks",
        (
            "-profile:v:0",
            "3",
            "-pix_fmt:v:0",
            "yuv422p10le",
        ),
    ),
    BurnInVideoStrategy.FFV1_LOSSLESS: (
        "ffv1",
        (
            "-level:v:0",
            "3",
            "-coder:v:0",
            "1",
            "-context:v:0",
            "1",
            "-slicecrc:v:0",
            "1",
        ),
    ),
}


class SubtitleDeliveryExecutor:
    """Resolve and execute a :class:`SubtitleOutputPlan` without source writes."""

    def __init__(
        self,
        *,
        runner: DeliveryRunner | None = None,
        probe: TrustedMediaProbe | None = None,
        policy: SubtitleDeliveryPolicy | None = None,
        ffmpeg_command: Sequence[str] = ("ffmpeg",),
    ) -> None:
        self.runner = runner or BoundedDeliveryRunner()
        self.probe = probe or MediaProbe()
        self.policy = policy or SubtitleDeliveryPolicy()
        self.ffmpeg_command = _validated_command(ffmpeg_command)

    def deliver(
        self,
        plan: SubtitleOutputPlan,
        *,
        sidecar_payload: str | bytes | None = None,
        subtitle_language: str = "und",
        subtitle_title: str = "MediaTranscribeStudio",
        make_subtitle_default: bool = False,
        burn_in_strategy: BurnInVideoStrategy | str | None = None,
        karaoke_timing_evidence: KaraokeTimingEvidence | None = None,
    ) -> SubtitleDeliveryReceipt:
        _validate_plan(plan)
        language = _validated_language(subtitle_language)
        title = _validated_title(subtitle_title)
        source = canonical_local_media_file(plan.source_path)
        output = _canonical_new_output_path(plan.output_path, source=source)
        source_before = _capture_file_evidence(
            source,
            chunk_bytes=self.policy.hash_chunk_bytes,
        )
        source_probe = self.probe.probe(source)
        _assert_probe_matches_file(source_probe, source, source_before)

        subtitle_bytes: bytes
        subtitle: Path
        if plan.mode is SubtitleOutputMode.SIDECAR:
            if sidecar_payload is None:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.SUBTITLE_INVALID,
                    "Sidecar delivery requires an explicit subtitle payload.",
                )
            subtitle_bytes = _coerce_subtitle_payload(sidecar_payload)
            subtitle = output
        else:
            if sidecar_payload is not None:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.INVALID_PLAN,
                    "Media delivery reads plan.subtitle_path; sidecar_payload is ambiguous.",
                )
            subtitle = _canonical_subtitle_input(
                plan.subtitle_path,
                source=source,
                output=output,
            )
            subtitle_bytes = _read_subtitle_bounded(
                subtitle,
                maximum=self.policy.maximum_subtitle_bytes,
            )

        if len(subtitle_bytes) > self.policy.maximum_subtitle_bytes:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.SUBTITLE_TOO_LARGE,
                "Subtitle input exceeds the configured delivery size limit.",
            )
        karaoke_detected = _validate_subtitle_payload(
            subtitle_bytes,
            subtitle_format=plan.subtitle_format,
            karaoke_timing_evidence=karaoke_timing_evidence,
        )
        karaoke_verified = _verify_karaoke_evidence(
            subtitle_bytes,
            detected=karaoke_detected,
            evidence=karaoke_timing_evidence,
        )

        if plan.mode is SubtitleOutputMode.SIDECAR:
            _validate_sidecar_suffix(output, plan.subtitle_format)
            return self._deliver_sidecar(
                plan=plan,
                source=source,
                output=output,
                payload=subtitle_bytes,
                source_before=source_before,
                source_probe=source_probe,
                karaoke_detected=karaoke_detected,
                karaoke_verified=karaoke_verified,
            )

        profile = _container_profile(output)
        capabilities = self._inspect_ffmpeg_capabilities()
        _require_capability(
            profile.muxer,
            capabilities.muxers,
            kind="muxer",
        )

        if plan.mode is SubtitleOutputMode.SOFT_MUX:
            if burn_in_strategy is not None:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.INVALID_PLAN,
                    "burn_in_strategy is valid only for burn-in delivery.",
                )
            selected_codec = _select_subtitle_codec(
                plan,
                profile=profile,
                capabilities=capabilities,
            )
            command = self._soft_mux_command(
                plan=plan,
                source=source,
                subtitle=subtitle,
                output=output,
                source_probe=source_probe,
                profile=profile,
                selected_codec=selected_codec,
                language=language,
                title=title,
                make_default=make_subtitle_default,
            )
            return self._deliver_media(
                plan=plan,
                source=source,
                output=output,
                subtitle=subtitle,
                source_before=source_before,
                source_probe=source_probe,
                profile=profile,
                capabilities=capabilities,
                selected_subtitle_codec=selected_codec,
                selected_video_strategy=None,
                command_builder=command,
                karaoke_detected=karaoke_detected,
                karaoke_verified=karaoke_verified,
            )

        strategy = _resolve_burn_strategy(burn_in_strategy)
        if not source_probe.video_stream_indexes:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.VIDEO_STRATEGY_UNSUPPORTED,
                "Burn-in delivery requires at least one non-attached video stream.",
            )
        if source_probe.has_hdr_video:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.HDR_BURN_IN_UNSUPPORTED,
                "HDR burn-in is fail-closed until a color-managed render pipeline is selected.",
            )
        if strategy not in profile.burn_strategies:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.VIDEO_STRATEGY_UNSUPPORTED,
                f"{strategy.value} is not safe for the {profile.name} container.",
            )
        encoder, _ = _VIDEO_STRATEGIES[strategy]
        _require_capability(encoder, capabilities.encoders, kind="encoder")
        _require_capability("subtitles", capabilities.filters, kind="filter")
        command = self._burn_in_command(
            source=source,
            subtitle=subtitle,
            output=output,
            source_probe=source_probe,
            profile=profile,
            strategy=strategy,
        )
        return self._deliver_media(
            plan=plan,
            source=source,
            output=output,
            subtitle=subtitle,
            source_before=source_before,
            source_probe=source_probe,
            profile=profile,
            capabilities=capabilities,
            selected_subtitle_codec=None,
            selected_video_strategy=strategy,
            command_builder=command,
            karaoke_detected=karaoke_detected,
            karaoke_verified=karaoke_verified,
        )

    def stage_media_delivery(
        self,
        plan: SubtitleOutputPlan,
        *,
        subtitle_language: str = "und",
        subtitle_title: str = "MediaTranscribeStudio",
        make_subtitle_default: bool = False,
        burn_in_strategy: BurnInVideoStrategy | str | None = None,
        karaoke_timing_evidence: KaraokeTimingEvidence | None = None,
    ) -> StagedSubtitleMediaDelivery:
        """Render and validate media under a private quarantine name.

        The customer output path is validated before expensive work starts,
        but is never created by this method.  The returned artifact remains
        quarantined until :meth:`publish_staged_media` receives passing visual
        QA evidence.
        """

        _validate_plan(plan)
        if plan.mode is SubtitleOutputMode.SIDECAR:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.INVALID_PLAN,
                "Only soft-mux and burn-in media can be quarantined.",
            )
        source = canonical_local_media_file(plan.source_path)
        customer_output = _canonical_new_output_path(
            plan.output_path,
            source=source,
        )
        quarantine = _new_quarantine_output(customer_output)
        staged_plan = replace(plan, output_path=str(quarantine))
        try:
            receipt = self.deliver(
                staged_plan,
                subtitle_language=subtitle_language,
                subtitle_title=subtitle_title,
                make_subtitle_default=make_subtitle_default,
                burn_in_strategy=burn_in_strategy,
                karaoke_timing_evidence=karaoke_timing_evidence,
            )
            quarantine_evidence = _capture_file_evidence(
                quarantine,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            if not _same_artifact(
                quarantine_evidence,
                receipt.output_evidence,
            ):
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.QA_FAILED,
                    "Quarantined media changed after delivery validation.",
                )
            staged = StagedSubtitleMediaDelivery(
                customer_output_path=customer_output,
                quarantine_path=quarantine,
                quarantine_evidence=quarantine_evidence,
                receipt=receipt,
            )
            if os.path.lexists(customer_output):
                rollback = self.rollback_staged_media(
                    staged,
                    reason="customer-output-appeared-during-staging",
                )
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.OUTPUT_EXISTS,
                    "Customer output appeared while media was quarantined.",
                    evidence={"rollback": rollback.to_dict()},
                )
            return staged
        except Exception:
            if os.path.lexists(quarantine):
                try:
                    quarantine.unlink()
                except OSError:
                    pass
            raise

    def publish_staged_media(
        self,
        staged: StagedSubtitleMediaDelivery,
        *,
        visual_qa_evidence: Mapping[str, Any],
    ) -> SubtitleMediaPublication:
        """Atomically publish a quarantined media artifact after visual QA."""

        if not isinstance(staged, StagedSubtitleMediaDelivery):
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.INVALID_PLAN,
                "Publication requires a staged subtitle media delivery.",
            )
        try:
            qa_payload = _validated_passing_visual_qa(visual_qa_evidence)
        except SubtitleDeliveryError as exc:
            rollback = self.rollback_staged_media(
                staged,
                reason="visual-qa-not-passed",
            )
            raise _with_rollback_evidence(exc, rollback) from exc

        source = canonical_local_media_file(staged.receipt.source_path)
        output = staged.customer_output_path
        quarantine = staged.quarantine_path
        published_evidence: FileEvidence | None = None
        try:
            resolved_output = _canonical_new_output_path(
                output,
                source=source,
            )
            if resolved_output != output:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.INVALID_PATH,
                    "The staged customer output path changed before publication.",
                )
            if quarantine.parent != output.parent:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.INVALID_PATH,
                    "Quarantine and customer output must share a directory.",
                )
            current_quarantine = _capture_file_evidence(
                quarantine,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            if not _same_file_evidence(
                current_quarantine,
                staged.quarantine_evidence,
            ):
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.QA_FAILED,
                    "Quarantined media changed after visual QA.",
                )
            source_before_publication = _capture_file_evidence(
                source,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            _require_unchanged_source(
                staged.receipt.source_before,
                source_before_publication,
            )

            _atomic_publish_no_replace(quarantine, output)
            published_evidence = _capture_file_evidence(
                output,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            if not _same_artifact(
                published_evidence,
                staged.quarantine_evidence,
            ):
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.PUBLISH_FAILED,
                    "Published media differs from the QA-approved quarantine artifact.",
                )

            source_probe = self.probe.probe(source)
            _assert_probe_matches_file(
                source_probe,
                source,
                source_before_publication,
            )
            final_probe = self.probe.probe(output)
            _assert_probe_matches_file(
                final_probe,
                output,
                published_evidence,
            )
            qa = _validate_media_qa(
                mode=staged.receipt.mode,
                source=source_probe,
                output=final_probe,
                expected_subtitle_codec=(
                    staged.receipt.selected_subtitle_codec
                ),
                policy=self.policy,
            )
            source_after = _capture_file_evidence(
                source,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            _require_unchanged_source(
                staged.receipt.source_before,
                source_after,
            )
            receipt = replace(
                staged.receipt,
                output_path=str(output),
                source_after=source_after,
                output_evidence=published_evidence,
                qa=qa,
                source_probe_fingerprint_sha256=(
                    source_probe.probe_fingerprint_sha256
                ),
                output_probe_fingerprint_sha256=(
                    final_probe.probe_fingerprint_sha256
                ),
            )
            return SubtitleMediaPublication(
                receipt=receipt,
                quarantine_path=quarantine,
                quarantine_evidence=staged.quarantine_evidence,
                published_evidence=published_evidence,
                visual_qa_evidence_sha256=_canonical_mapping_sha256(
                    qa_payload
                ),
            )
        except SubtitleDeliveryError as exc:
            rollback = self.rollback_staged_media(
                staged,
                reason=f"publication-{exc.code.value}",
                published_evidence=published_evidence,
            )
            raise _with_rollback_evidence(exc, rollback) from exc
        except Exception as exc:
            rollback = self.rollback_staged_media(
                staged,
                reason="publication-unexpected-failure",
                published_evidence=published_evidence,
            )
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.PUBLISH_FAILED,
                "QA-approved media could not be atomically published.",
                detail=str(exc),
                evidence={"rollback": rollback.to_dict()},
            ) from exc

    def rollback_staged_media(
        self,
        staged: StagedSubtitleMediaDelivery,
        *,
        reason: str,
        published_evidence: FileEvidence | None = None,
    ) -> SubtitleMediaRollback:
        """Remove only transaction-owned artifacts and describe the result."""

        if not isinstance(staged, StagedSubtitleMediaDelivery):
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.INVALID_PLAN,
                "Rollback requires a staged subtitle media delivery.",
            )
        cleanup_errors: list[str] = []
        customer_state = "absent"
        output = staged.customer_output_path
        if os.path.lexists(output):
            if published_evidence is None:
                customer_state = "existing-preserved"
            else:
                state, error = _remove_matching_artifact(
                    output,
                    published_evidence,
                )
                customer_state = (
                    "rolled-back" if state == "removed" else state
                )
                if error is not None:
                    cleanup_errors.append(error)

        quarantine_state = "absent"
        quarantine = staged.quarantine_path
        if os.path.lexists(quarantine):
            state, error = _remove_matching_artifact(
                quarantine,
                staged.quarantine_evidence,
            )
            quarantine_state = state
            if error is not None:
                cleanup_errors.append(error)

        source_unchanged = False
        try:
            source_after = _capture_file_evidence(
                Path(staged.receipt.source_path),
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            source_unchanged = (
                source_after == staged.receipt.source_before
            )
        except SubtitleDeliveryError as exc:
            cleanup_errors.append(str(exc))

        return SubtitleMediaRollback(
            reason=str(reason),
            customer_output_path=output,
            quarantine_path=quarantine,
            quarantine_evidence=staged.quarantine_evidence,
            customer_output_state=customer_state,
            quarantine_state=quarantine_state,
            source_unchanged=source_unchanged,
            cleanup_error=(
                "; ".join(cleanup_errors) if cleanup_errors else None
            ),
        )

    def _deliver_sidecar(
        self,
        *,
        plan: SubtitleOutputPlan,
        source: Path,
        output: Path,
        payload: bytes,
        source_before: FileEvidence,
        source_probe: MediaProbeResult,
        karaoke_detected: bool,
        karaoke_verified: bool,
    ) -> SubtitleDeliveryReceipt:
        temporary = _new_temporary_output(output)
        published = False
        published_evidence: FileEvidence | None = None
        try:
            _write_exclusive_file(temporary, payload)
            temporary_evidence = _capture_file_evidence(
                temporary,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            if temporary_evidence.size_bytes < 1:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.OUTPUT_EMPTY,
                    "The generated sidecar is empty.",
                )
            _atomic_publish_no_replace(temporary, output)
            published = True
            published_evidence = _capture_file_evidence(
                output,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            if (
                published_evidence.size_bytes != temporary_evidence.size_bytes
                or published_evidence.sha256 != temporary_evidence.sha256
            ):
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.PUBLISH_FAILED,
                    "Published sidecar evidence differs from its temporary artifact.",
                )
            source_after = _capture_file_evidence(
                source,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            _require_unchanged_source(source_before, source_after)
            qa = DeliveryQA(
                output_non_empty=True,
                duration_checked=False,
                source_duration_ms=source_probe.duration_ms,
                output_duration_ms=None,
                duration_delta_ms=None,
                duration_tolerance_ms=None,
                source_audio_streams=len(source_probe.audio_stream_indexes),
                output_audio_streams=None,
                source_video_streams=len(source_probe.video_stream_indexes),
                output_video_streams=None,
                source_subtitle_streams=len(
                    source_probe.subtitle_stream_indexes
                ),
                output_subtitle_streams=None,
                expected_subtitle_codec=None,
                subtitle_stream_verified=False,
                post_publish_probe_verified=False,
            )
            return SubtitleDeliveryReceipt(
                mode=plan.mode,
                source_path=str(source),
                output_path=str(output),
                subtitle_path=str(output),
                container=None,
                selected_subtitle_codec=None,
                selected_video_strategy=None,
                command=None,
                process_result=None,
                process_limits=None,
                source_before=source_before,
                source_after=source_after,
                output_evidence=published_evidence,
                qa=qa,
                source_probe_fingerprint_sha256=(
                    source_probe.probe_fingerprint_sha256
                ),
                output_probe_fingerprint_sha256=None,
                capability_fingerprint_sha256=None,
                karaoke_timing_detected=karaoke_detected,
                karaoke_timing_verified=karaoke_verified,
            )
        except Exception:
            if published and published_evidence is not None:
                _remove_if_unchanged(output, published_evidence)
            raise
        finally:
            _cleanup_temporary(temporary)

    def _deliver_media(
        self,
        *,
        plan: SubtitleOutputPlan,
        source: Path,
        output: Path,
        subtitle: Path,
        source_before: FileEvidence,
        source_probe: MediaProbeResult,
        profile: _ContainerProfile,
        capabilities: FFmpegCapabilityEvidence,
        selected_subtitle_codec: str | None,
        selected_video_strategy: BurnInVideoStrategy | None,
        command_builder: tuple[str, ...],
        karaoke_detected: bool,
        karaoke_verified: bool,
    ) -> SubtitleDeliveryReceipt:
        temporary = _new_temporary_output(output)
        command = tuple(
            str(temporary) if part == _OUTPUT_SENTINEL else part
            for part in command_builder
        )
        published = False
        published_evidence: FileEvidence | None = None
        process_result: ProcessResult | None = None
        try:
            try:
                process_result = self.runner.run(
                    command,
                    limits=self.policy.process_limits,
                )
            except SubtitleDeliveryError:
                raise
            except Exception as exc:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.PROCESS_FAILED,
                    "The injected FFmpeg runner failed before producing a result.",
                    detail=str(exc),
                ) from exc
            if process_result.returncode != 0:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.PROCESS_FAILED,
                    "FFmpeg subtitle delivery failed.",
                    detail=_safe_process_detail(process_result.stderr),
                )
            if not temporary.is_file():
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.OUTPUT_MISSING,
                    "FFmpeg reported success but produced no temporary output.",
                )
            temporary_evidence = _capture_file_evidence(
                temporary,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            if temporary_evidence.size_bytes < 1:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.OUTPUT_EMPTY,
                    "FFmpeg produced an empty temporary output.",
                )

            temporary_probe = self.probe.probe(temporary)
            _assert_probe_matches_file(
                temporary_probe,
                temporary,
                temporary_evidence,
            )
            _validate_media_qa(
                mode=plan.mode,
                source=source_probe,
                output=temporary_probe,
                expected_subtitle_codec=selected_subtitle_codec,
                policy=self.policy,
            )

            _atomic_publish_no_replace(temporary, output)
            published = True
            published_evidence = _capture_file_evidence(
                output,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            if (
                published_evidence.size_bytes != temporary_evidence.size_bytes
                or published_evidence.sha256 != temporary_evidence.sha256
            ):
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.PUBLISH_FAILED,
                    "Published media evidence differs from its temporary artifact.",
                )

            final_probe = self.probe.probe(output)
            _assert_probe_matches_file(final_probe, output, published_evidence)
            qa = _validate_media_qa(
                mode=plan.mode,
                source=source_probe,
                output=final_probe,
                expected_subtitle_codec=selected_subtitle_codec,
                policy=self.policy,
            )
            source_after = _capture_file_evidence(
                source,
                chunk_bytes=self.policy.hash_chunk_bytes,
            )
            _require_unchanged_source(source_before, source_after)
            return SubtitleDeliveryReceipt(
                mode=plan.mode,
                source_path=str(source),
                output_path=str(output),
                subtitle_path=str(subtitle),
                container=profile.name,
                selected_subtitle_codec=selected_subtitle_codec,
                selected_video_strategy=selected_video_strategy,
                command=command,
                process_result=process_result,
                process_limits=self.policy.process_limits,
                source_before=source_before,
                source_after=source_after,
                output_evidence=published_evidence,
                qa=qa,
                source_probe_fingerprint_sha256=(
                    source_probe.probe_fingerprint_sha256
                ),
                output_probe_fingerprint_sha256=(
                    final_probe.probe_fingerprint_sha256
                ),
                capability_fingerprint_sha256=(
                    capabilities.fingerprint_sha256
                ),
                karaoke_timing_detected=karaoke_detected,
                karaoke_timing_verified=karaoke_verified,
            )
        except Exception:
            if published and published_evidence is not None:
                _remove_if_unchanged(output, published_evidence)
            raise
        finally:
            _cleanup_temporary(temporary)

    def _inspect_ffmpeg_capabilities(self) -> FFmpegCapabilityEvidence:
        payloads: list[bytes] = []
        parsed: dict[str, frozenset[str]] = {}
        for switch, label in (
            ("-encoders", "encoders"),
            ("-filters", "filters"),
            ("-muxers", "muxers"),
        ):
            result = self.runner.run(
                (*self.ffmpeg_command, "-nostdin", "-hide_banner", switch),
                limits=self.policy.capability_limits,
            )
            if result.returncode != 0:
                raise SubtitleDeliveryError(
                    SubtitleDeliveryErrorCode.CAPABILITY_PROBE_FAILED,
                    f"FFmpeg could not report its available {label}.",
                    detail=_safe_process_detail(result.stderr),
                )
            payload = result.stdout + b"\n" + result.stderr
            payloads.append(payload)
            parsed[label] = _parse_capability_names(payload)
        fingerprint = hashlib.sha256(b"\x00".join(payloads)).hexdigest()
        return FFmpegCapabilityEvidence(
            encoders=parsed["encoders"],
            filters=parsed["filters"],
            muxers=parsed["muxers"],
            fingerprint_sha256=fingerprint,
        )

    def _soft_mux_command(
        self,
        *,
        plan: SubtitleOutputPlan,
        source: Path,
        subtitle: Path,
        output: Path,
        source_probe: MediaProbeResult,
        profile: _ContainerProfile,
        selected_codec: str,
        language: str,
        title: str,
        make_default: bool,
    ) -> tuple[str, ...]:
        del output
        subtitle_index = len(source_probe.subtitle_stream_indexes)
        format_name = _subtitle_demuxer(plan.subtitle_format)
        return (
            *self.ffmpeg_command,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-i",
            str(source),
            "-f",
            format_name,
            "-i",
            str(subtitle),
            "-map",
            "0",
            "-map",
            "1:0",
            "-c",
            "copy",
            f"-c:s:{subtitle_index}",
            selected_codec,
            f"-metadata:s:s:{subtitle_index}",
            f"language={language}",
            f"-metadata:s:s:{subtitle_index}",
            f"title={title}",
            f"-disposition:s:{subtitle_index}",
            "default" if make_default else "0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            *profile.final_arguments,
            "-n",
            _OUTPUT_SENTINEL,
        )

    def _burn_in_command(
        self,
        *,
        source: Path,
        subtitle: Path,
        output: Path,
        source_probe: MediaProbeResult,
        profile: _ContainerProfile,
        strategy: BurnInVideoStrategy,
    ) -> tuple[str, ...]:
        del output
        encoder, encoding_arguments = _VIDEO_STRATEGIES[strategy]
        mapping: list[str] = []
        for index in source_probe.video_stream_indexes:
            mapping.extend(("-map", f"0:{index}"))
        for index in source_probe.audio_stream_indexes:
            mapping.extend(("-map", f"0:{index}"))
        return (
            *self.ffmpeg_command,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-i",
            str(source),
            *mapping,
            "-filter:v:0",
            _subtitle_filter_argument(subtitle),
            "-c",
            "copy",
            "-c:v:0",
            encoder,
            *encoding_arguments,
            "-c:a",
            "copy",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            *profile.final_arguments,
            "-n",
            _OUTPUT_SENTINEL,
        )


_OUTPUT_SENTINEL = "{MTS_SUBTITLE_DELIVERY_OUTPUT}"


def _validate_plan(plan: SubtitleOutputPlan) -> None:
    if not isinstance(plan, SubtitleOutputPlan):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PLAN,
            "Delivery requires a SubtitleOutputPlan.",
        )
    if (
        not plan.source_preserved
        or plan.overwrites_source
        or not plan.creates_new_file
    ):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PLAN,
            "The plan does not satisfy immutable-source delivery invariants.",
        )
    if plan.mode is SubtitleOutputMode.SIDECAR and plan.ffmpeg is not None:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PLAN,
            "Sidecar delivery must not contain an FFmpeg plan.",
        )
    if plan.mode is not SubtitleOutputMode.SIDECAR and plan.ffmpeg is None:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PLAN,
            "Media delivery requires an FFmpeg intent plan.",
        )


def _validated_command(command: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(str(part).strip() for part in command)
    if not resolved or any(not part or "\x00" in part for part in resolved):
        raise ValueError("ffmpeg_command must contain non-empty argv entries")
    return resolved


def _validated_language(value: str) -> str:
    resolved = str(value).strip()
    if not _SAFE_LANGUAGE.fullmatch(resolved):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PLAN,
            "subtitle_language must be a valid internal language tag.",
        )
    return resolved


def _validated_title(value: str) -> str:
    resolved = str(value).strip()
    if (
        not resolved
        or len(resolved) > 256
        or _CONTROL_CHARACTER.search(resolved)
    ):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PLAN,
            "subtitle_title must be 1..256 printable characters.",
        )
    return resolved


def _canonical_new_output_path(path: str | Path, *, source: Path) -> Path:
    text = os.fspath(path).strip()
    if (
        not text
        or "\x00" in text
        or _URL_SCHEME.match(text)
        or text.replace("/", "\\").startswith("\\\\")
    ):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PATH,
            "Output must be an absolute local file path.",
        )
    candidate = Path(text).expanduser()
    if not candidate.is_absolute() or not candidate.name:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PATH,
            "Output must be an absolute local file path.",
        )
    try:
        parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PATH,
            "The output directory does not exist or cannot be resolved.",
            detail=str(exc),
        ) from exc
    if not parent.is_dir():
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PATH,
            "The output parent must be a local directory.",
        )
    resolved = parent / candidate.name
    if os.path.lexists(resolved):
        try:
            aliases_source = os.path.samefile(source, resolved)
        except OSError:
            aliases_source = False
        if aliases_source:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.PATH_ALIAS,
                "Output aliases the immutable source media.",
            )
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.OUTPUT_EXISTS,
            "Output already exists; subtitle delivery never overwrites.",
        )
    if _path_key(resolved) == _path_key(source):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.PATH_ALIAS,
            "Output aliases the immutable source media.",
        )
    return resolved


def _canonical_subtitle_input(
    path: str | Path,
    *,
    source: Path,
    output: Path,
) -> Path:
    text = os.fspath(path).strip()
    if (
        not text
        or "\x00" in text
        or _URL_SCHEME.match(text)
        or text.replace("/", "\\").startswith("\\\\")
    ):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PATH,
            "Subtitle input must be an absolute local file path.",
        )
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INVALID_PATH,
            "Subtitle input must be an absolute local file path.",
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INPUT_MISSING,
            "Subtitle input does not exist or cannot be resolved.",
            detail=str(exc),
        ) from exc
    if not resolved.is_file():
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INPUT_MISSING,
            "Subtitle input must be a regular local file.",
        )
    if _paths_alias(resolved, source):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.PATH_ALIAS,
            "Subtitle input cannot alias the immutable source media.",
        )
    if _path_key(resolved) == _path_key(output):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.PATH_ALIAS,
            "Subtitle input cannot alias the derived media output.",
        )
    return resolved


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return _path_key(left) == _path_key(right)


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(os.fspath(path))))


def _coerce_subtitle_payload(payload: str | bytes) -> bytes:
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, bytes):
        return payload
    raise SubtitleDeliveryError(
        SubtitleDeliveryErrorCode.SUBTITLE_INVALID,
        "Subtitle payload must be UTF-8 text or bytes.",
    )


def _read_subtitle_bounded(path: Path, *, maximum: int) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
    except OSError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INPUT_MISSING,
            "Subtitle input could not be read.",
            detail=str(exc),
        ) from exc
    if len(payload) > maximum:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SUBTITLE_TOO_LARGE,
            "Subtitle input exceeds the configured delivery size limit.",
        )
    return payload


def _validate_subtitle_payload(
    payload: bytes,
    *,
    subtitle_format: SubtitleFormat,
    karaoke_timing_evidence: KaraokeTimingEvidence | None,
) -> bool:
    if not payload:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SUBTITLE_INVALID,
            "Subtitle input is empty.",
        )
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SUBTITLE_INVALID,
            "Subtitle input must be valid UTF-8.",
            detail=str(exc),
        ) from exc
    if "\x00" in text:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SUBTITLE_INVALID,
            "Subtitle input contains a NUL character.",
        )
    if subtitle_format is SubtitleFormat.SRT:
        valid = bool(
            re.search(
                r"(?m)^\d{2,}:\d{2}:\d{2},\d{3}\s+-->\s+"
                r"\d{2,}:\d{2}:\d{2},\d{3}(?:\s|$)",
                text,
            )
        )
    elif subtitle_format is SubtitleFormat.WEBVTT:
        valid = text.startswith("WEBVTT")
    else:
        valid = (
            "[Script Info]" in text
            and "[Events]" in text
            and re.search(r"(?m)^Dialogue\s*:", text) is not None
        )
    if not valid:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SUBTITLE_INVALID,
            f"Subtitle content does not match {subtitle_format.value}.",
        )
    karaoke_detected = _KARAOKE_TAG.search(text) is not None
    if karaoke_detected and karaoke_timing_evidence is None:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.KARAOKE_EVIDENCE_REQUIRED,
            "ASS karaoke timing tags require verified word-level timing evidence.",
        )
    return karaoke_detected


def _verify_karaoke_evidence(
    payload: bytes,
    *,
    detected: bool,
    evidence: KaraokeTimingEvidence | None,
) -> bool:
    if not detected:
        return False
    if evidence is None or not evidence.verified:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.KARAOKE_EVIDENCE_REQUIRED,
            "Verified word-level timing evidence is required.",
        )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != evidence.subtitle_sha256:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.KARAOKE_EVIDENCE_REQUIRED,
            "Karaoke timing evidence does not match the delivered subtitle.",
        )
    return True


def _validate_sidecar_suffix(path: Path, subtitle_format: SubtitleFormat) -> None:
    suffixes = {
        SubtitleFormat.SRT: frozenset({".srt"}),
        SubtitleFormat.WEBVTT: frozenset({".vtt", ".webvtt"}),
        SubtitleFormat.ASS: frozenset({".ass"}),
    }
    if path.suffix.lower() not in suffixes[subtitle_format]:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SUBTITLE_INVALID,
            f"Output suffix does not match {subtitle_format.value}.",
        )


def _container_profile(output: Path) -> _ContainerProfile:
    profile = _CONTAINER_BY_SUFFIX.get(output.suffix.lower())
    if profile is None:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.CONTAINER_UNSUPPORTED,
            "Target container is not approved for safe subtitle delivery.",
        )
    return profile


def _requested_plan_codec(plan: SubtitleOutputPlan) -> str | None:
    if plan.ffmpeg is None:
        return None
    arguments = plan.ffmpeg.arguments
    for index, argument in enumerate(arguments[:-1]):
        if argument == "-c:s":
            value = arguments[index + 1]
            if value.startswith("<") and value.endswith(">"):
                return None
            return value
    return None


def _select_subtitle_codec(
    plan: SubtitleOutputPlan,
    *,
    profile: _ContainerProfile,
    capabilities: FFmpegCapabilityEvidence,
) -> str:
    approved = profile.soft_codecs.get(plan.subtitle_format, ())
    if not approved:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.CODEC_UNSUPPORTED,
            f"{plan.subtitle_format.value} cannot be safely muxed into {profile.name}.",
        )
    requested = _requested_plan_codec(plan)
    if requested is not None:
        if requested not in approved:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.CODEC_UNSUPPORTED,
                f"Requested subtitle codec {requested} is unsafe for {profile.name}.",
            )
        _require_capability(requested, capabilities.encoders, kind="encoder")
        return requested
    for codec in approved:
        if codec in capabilities.encoders:
            return codec
    raise SubtitleDeliveryError(
        SubtitleDeliveryErrorCode.CODEC_UNSUPPORTED,
        f"No approved subtitle encoder is available for {profile.name}.",
    )


def _resolve_burn_strategy(
    value: BurnInVideoStrategy | str | None,
) -> BurnInVideoStrategy:
    if value is None:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.VIDEO_STRATEGY_REQUIRED,
            "Burn-in delivery requires an explicit video encoding strategy.",
        )
    try:
        return BurnInVideoStrategy(value)
    except (TypeError, ValueError) as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.VIDEO_STRATEGY_UNSUPPORTED,
            "Unknown burn-in video encoding strategy.",
        ) from exc


def _require_capability(
    name: str,
    available: frozenset[str],
    *,
    kind: str,
) -> None:
    if name not in available:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.CODEC_UNSUPPORTED,
            f"The installed FFmpeg build does not provide required {kind}: {name}.",
        )


def _parse_capability_names(payload: bytes) -> frozenset[str]:
    text = payload.decode("utf-8", errors="replace")
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        flags, name = parts[0], parts[1]
        if (
            1 <= len(flags) <= 8
            and all(character.isalpha() or character == "." for character in flags)
            and re.fullmatch(r"[A-Za-z0-9_.+-]+", name)
        ):
            names.add(name)
    return frozenset(names)


def _subtitle_demuxer(subtitle_format: SubtitleFormat) -> str:
    return {
        SubtitleFormat.SRT: "srt",
        SubtitleFormat.WEBVTT: "webvtt",
        SubtitleFormat.ASS: "ass",
    }[subtitle_format]


def _subtitle_filter_argument(path: Path) -> str:
    normalized = str(path).replace("\\", "/")
    escaped = (
        normalized.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace(",", r"\,")
        .replace(";", r"\;")
    )
    return f"subtitles=filename='{escaped}'"


def _capture_file_evidence(path: Path, *, chunk_bytes: int) -> FileEvidence:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.INPUT_MISSING,
            f"File evidence could not be captured: {path}",
            detail=str(exc),
        ) from exc
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SOURCE_CHANGED,
            f"File changed while integrity evidence was captured: {path}",
        )
    return FileEvidence(
        path=str(path.resolve(strict=True)),
        size_bytes=after.st_size,
        modified_time_ns=after.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _assert_probe_matches_file(
    result: MediaProbeResult,
    path: Path,
    evidence: FileEvidence,
) -> None:
    try:
        probed = Path(result.source_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Trusted probe returned a non-canonical source path.",
            detail=str(exc),
        ) from exc
    if not _paths_alias(probed, path):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Trusted probe evidence belongs to a different file.",
        )
    if result.source_size_bytes != evidence.size_bytes:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Trusted probe size differs from file integrity evidence.",
        )


def _duration_tolerance(
    duration_ms: int,
    policy: SubtitleDeliveryPolicy,
) -> int:
    return max(
        policy.duration_absolute_tolerance_ms,
        round(duration_ms * policy.duration_relative_tolerance),
    )


def _validate_media_qa(
    *,
    mode: SubtitleOutputMode,
    source: MediaProbeResult,
    output: MediaProbeResult,
    expected_subtitle_codec: str | None,
    policy: SubtitleDeliveryPolicy,
) -> DeliveryQA:
    if source.duration_ms is None or output.duration_ms is None:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.DURATION_UNAVAILABLE,
            "Source and delivered media require measurable durations.",
        )
    tolerance = _duration_tolerance(source.duration_ms, policy)
    delta = abs(output.duration_ms - source.duration_ms)
    if delta > tolerance:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            f"Delivered duration differs by {delta} ms; tolerance is {tolerance} ms.",
        )

    source_audio = _streams_of_type(source, "audio")
    output_audio = _streams_of_type(output, "audio")
    source_video = _streams_of_type(source, "video", exclude_attached=True)
    output_video = _streams_of_type(output, "video", exclude_attached=True)
    source_subtitles = _streams_of_type(source, "subtitle")
    output_subtitles = _streams_of_type(output, "subtitle")
    if len(output_audio) < len(source_audio):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Delivered media lost one or more source audio streams.",
        )
    if len(output_video) < len(source_video):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Delivered media lost one or more source video streams.",
        )
    if tuple(stream.codec_name for stream in output_audio[: len(source_audio)]) != tuple(
        stream.codec_name for stream in source_audio
    ):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Delivered media did not preserve source audio codecs.",
        )
    if source.chapters and output.chapters < source.chapters:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Delivered media lost source chapter metadata.",
        )

    subtitle_verified = False
    if mode is SubtitleOutputMode.SOFT_MUX:
        if tuple(
            stream.codec_name for stream in output_video[: len(source_video)]
        ) != tuple(stream.codec_name for stream in source_video):
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.QA_FAILED,
                "Soft-mux delivery did not stream-copy source video codecs.",
            )
        if len(output_subtitles) != len(source_subtitles) + 1:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.QA_FAILED,
                "Soft-mux delivery did not add exactly one subtitle stream.",
            )
        if tuple(
            stream.codec_name
            for stream in output_subtitles[: len(source_subtitles)]
        ) != tuple(stream.codec_name for stream in source_subtitles):
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.QA_FAILED,
                "Soft-mux delivery did not preserve existing subtitle streams.",
            )
        if (
            expected_subtitle_codec is None
            or output_subtitles[-1].codec_name != expected_subtitle_codec
        ):
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.QA_FAILED,
                "Soft-mux output lacks the expected subtitle codec.",
            )
        if source.has_hdr_video != output.has_hdr_video:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.QA_FAILED,
                "Soft-mux delivery changed HDR stream evidence.",
            )
        if source.has_rotation_metadata != output.has_rotation_metadata:
            raise SubtitleDeliveryError(
                SubtitleDeliveryErrorCode.QA_FAILED,
                "Soft-mux delivery changed rotation metadata evidence.",
            )
        subtitle_verified = True

    return DeliveryQA(
        output_non_empty=True,
        duration_checked=True,
        source_duration_ms=source.duration_ms,
        output_duration_ms=output.duration_ms,
        duration_delta_ms=delta,
        duration_tolerance_ms=tolerance,
        source_audio_streams=len(source_audio),
        output_audio_streams=len(output_audio),
        source_video_streams=len(source_video),
        output_video_streams=len(output_video),
        source_subtitle_streams=len(source_subtitles),
        output_subtitle_streams=len(output_subtitles),
        expected_subtitle_codec=expected_subtitle_codec,
        subtitle_stream_verified=subtitle_verified,
        post_publish_probe_verified=True,
    )


def _streams_of_type(
    probe: MediaProbeResult,
    stream_type: str,
    *,
    exclude_attached: bool = False,
) -> tuple[Any, ...]:
    return tuple(
        stream
        for stream in probe.streams
        if stream.type == stream_type
        and (not exclude_attached or not stream.attached_picture)
    )


def _require_unchanged_source(
    before: FileEvidence,
    after: FileEvidence,
) -> None:
    if before != after:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.SOURCE_CHANGED,
            "Source media size, mtime, path, or SHA-256 changed during delivery.",
        )


def _validated_passing_visual_qa(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Atomic publication requires structured visual-QA evidence.",
        )
    try:
        payload = json.loads(
            json.dumps(
                dict(evidence),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Visual-QA evidence must be finite canonical JSON.",
            detail=str(exc),
        ) from exc
    if payload.get("passed") is not True:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.QA_FAILED,
            "Visual-QA evidence did not explicitly pass.",
        )
    return payload


def _canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_rollback_evidence(
    error: SubtitleDeliveryError,
    rollback: SubtitleMediaRollback,
) -> SubtitleDeliveryError:
    evidence = dict(error.evidence)
    evidence["rollback"] = rollback.to_dict()
    return SubtitleDeliveryError(
        error.code,
        str(error),
        detail=error.detail,
        evidence=evidence,
    )


def _same_artifact(left: FileEvidence, right: FileEvidence) -> bool:
    return (
        left.size_bytes == right.size_bytes
        and left.sha256 == right.sha256
    )


def _same_file_evidence(left: FileEvidence, right: FileEvidence) -> bool:
    return left == right


def _remove_matching_artifact(
    path: Path,
    evidence: FileEvidence,
) -> tuple[str, str | None]:
    try:
        current = _capture_file_evidence(path, chunk_bytes=1024 * 1024)
    except SubtitleDeliveryError as exc:
        return "inspection-failed-retained", str(exc)
    if not _same_file_evidence(current, evidence):
        return "changed-retained", None
    try:
        path.unlink()
    except OSError as exc:
        return "removal-failed-retained", str(exc)
    return "removed", None


def _new_quarantine_output(output: Path) -> Path:
    for _ in range(16):
        candidate = output.parent / (
            f".{output.stem}.mts-quarantine-{uuid.uuid4().hex}"
            f"{output.suffix}"
        )
        if not os.path.lexists(candidate):
            return candidate
    raise SubtitleDeliveryError(
        SubtitleDeliveryErrorCode.PUBLISH_FAILED,
        "Could not allocate a unique private quarantine output.",
    )


def _new_temporary_output(output: Path) -> Path:
    for _ in range(16):
        candidate = output.parent / (
            f".{output.stem}.mts-{uuid.uuid4().hex}{output.suffix}"
        )
        if not os.path.lexists(candidate):
            return candidate
    raise SubtitleDeliveryError(
        SubtitleDeliveryErrorCode.PUBLISH_FAILED,
        "Could not allocate a unique same-directory temporary output.",
    )


def _write_exclusive_file(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.PUBLISH_FAILED,
            "Temporary sidecar path appeared concurrently.",
        ) from exc
    except OSError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.PUBLISH_FAILED,
            "Temporary sidecar could not be written.",
            detail=str(exc),
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_publish_no_replace(temporary: Path, output: Path) -> None:
    if temporary.parent != output.parent:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.PUBLISH_FAILED,
            "Temporary and final outputs must share the same directory.",
        )
    try:
        os.link(temporary, output, follow_symlinks=False)
    except FileExistsError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.OUTPUT_EXISTS,
            "Output appeared concurrently; no file was overwritten.",
        ) from exc
    except OSError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.PUBLISH_FAILED,
            "Atomic no-replace publication is unavailable for this output directory.",
            detail=str(exc),
        ) from exc
    try:
        temporary.unlink()
    except OSError as exc:
        try:
            output.unlink()
        except OSError:
            pass
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.CLEANUP_FAILED,
            "Published output was rolled back because temporary cleanup failed.",
            detail=str(exc),
        ) from exc


def _remove_if_unchanged(path: Path, evidence: FileEvidence) -> None:
    if not os.path.lexists(path):
        return
    try:
        current = _capture_file_evidence(path, chunk_bytes=1024 * 1024)
    except SubtitleDeliveryError:
        return
    if (
        current.size_bytes == evidence.size_bytes
        and current.modified_time_ns == evidence.modified_time_ns
        and current.sha256 == evidence.sha256
    ):
        try:
            path.unlink()
        except OSError:
            pass


def _cleanup_temporary(path: Path) -> None:
    if not os.path.lexists(path):
        return
    try:
        path.unlink()
    except OSError as exc:
        raise SubtitleDeliveryError(
            SubtitleDeliveryErrorCode.CLEANUP_FAILED,
            "Temporary subtitle delivery artifact could not be removed.",
            detail=str(exc),
        ) from exc


def _safe_process_detail(payload: bytes) -> str | None:
    text = payload.decode("utf-8", errors="replace").strip()
    return text[-8_192:] if text else None


__all__ = [
    "BoundedDeliveryRunner",
    "BurnInVideoStrategy",
    "DeliveryProcessLimits",
    "DeliveryQA",
    "FileEvidence",
    "KaraokeTimingEvidence",
    "KaraokeTimingSource",
    "SUBTITLE_DELIVERY_SCHEMA_VERSION",
    "SubtitleDeliveryError",
    "SubtitleDeliveryErrorCode",
    "SubtitleDeliveryExecutor",
    "SubtitleDeliveryPolicy",
    "SubtitleDeliveryReceipt",
    "StagedSubtitleMediaDelivery",
    "SubtitleMediaPublication",
    "SubtitleMediaRollback",
]
