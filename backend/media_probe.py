"""Trusted local-media probing with bounded FFprobe/FFmpeg subprocesses.

File extensions are intentionally treated as UI hints only.  Admission is
based on a canonical local regular file plus machine-readable FFprobe stream
evidence.  An optional one-frame FFmpeg decode smoke test rejects content that
the installed local build can identify but cannot actually decode.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


MEDIA_PROBE_SCHEMA_VERSION = "1.0.0"
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
_HDR_SIDE_DATA_MARKERS = (
    "mastering display metadata",
    "content light level metadata",
    "dynamic hdr plus",
    "hdr dynamic metadata",
    "smpte2094",
    "dolby vision",
    "dovi configuration record",
)
_ENCRYPTION_TAG_KEYS = frozenset(
    {
        "crypt_byte_block",
        "drm",
        "enc_key_id",
        "encrypted",
        "encryption",
        "encryption_scheme",
        "key_id",
        "protection_scheme",
        "skip_byte_block",
    }
)
_ENCRYPTED_CODEC_TAGS = frozenset({"enca", "encv"})


class MediaProbeErrorCode(str, Enum):
    INVALID_PATH = "invalid-path"
    NOT_FOUND = "not-found"
    NOT_REGULAR_FILE = "not-regular-file"
    TOOL_UNAVAILABLE = "tool-unavailable"
    TOOL_TIMEOUT = "tool-timeout"
    OUTPUT_LIMIT = "output-limit"
    PROBE_FAILED = "probe-failed"
    MALFORMED_PROBE = "malformed-probe"
    NO_MEDIA_STREAM = "no-media-stream"
    UNSUPPORTED_CODEC = "unsupported-codec"
    ENCRYPTED_MEDIA = "encrypted-media"
    DECODE_FAILED = "decode-failed"
    SOURCE_CHANGED = "source-changed"


class MediaProbeError(RuntimeError):
    """Structured fail-closed error raised by the media capability boundary."""

    def __init__(
        self,
        code: MediaProbeErrorCode,
        message: str,
        *,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class MediaProbeEvidenceError(ValueError):
    """Raised when injected or persisted probe evidence is not trustworthy."""


@dataclass(frozen=True)
class ProcessLimits:
    timeout_seconds: float = 20.0
    max_stdout_bytes: int = 4 * 1024 * 1024
    max_stderr_bytes: int = 512 * 1024
    poll_interval_seconds: float = 0.02

    def __post_init__(self) -> None:
        if not 0.05 <= self.timeout_seconds <= 300.0:
            raise ValueError("timeout_seconds must be between 0.05 and 300")
        if not 1_024 <= self.max_stdout_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_stdout_bytes must be between 1 KiB and 64 MiB")
        if not 1_024 <= self.max_stderr_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_stderr_bytes must be between 1 KiB and 16 MiB")
        if not 0.005 <= self.poll_interval_seconds <= 0.5:
            raise ValueError("poll_interval_seconds must be between 0.005 and 0.5")


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    elapsed_ms: int


class ProcessRunner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        limits: ProcessLimits,
    ) -> ProcessResult: ...


class BoundedProcessRunner:
    """Execute an argv vector without a shell, bounding time and output size."""

    def run(
        self,
        command: Sequence[str],
        *,
        limits: ProcessLimits,
    ) -> ProcessResult:
        argv = tuple(str(part) for part in command)
        if not argv or any("\x00" in part for part in argv):
            raise ValueError("command must contain non-NUL argv entries")

        creationflags = 0
        if os.name == "nt":
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="mts-media-probe-") as temp_root:
            stdout_path = Path(temp_root) / "stdout.bin"
            stderr_path = Path(temp_root) / "stderr.bin"
            try:
                with stdout_path.open("wb") as stdout_handle, stderr_path.open(
                    "wb"
                ) as stderr_handle:
                    process = subprocess.Popen(  # noqa: S603 - trusted argv boundary
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        shell=False,
                        close_fds=True,
                        creationflags=creationflags,
                    )
                    while process.poll() is None:
                        elapsed = time.monotonic() - started
                        if elapsed > limits.timeout_seconds:
                            self._stop(process)
                            raise MediaProbeError(
                                MediaProbeErrorCode.TOOL_TIMEOUT,
                                "The local media tool exceeded its execution timeout.",
                            )
                        stdout_handle.flush()
                        stderr_handle.flush()
                        if (
                            stdout_handle.tell() > limits.max_stdout_bytes
                            or stderr_handle.tell() > limits.max_stderr_bytes
                        ):
                            self._stop(process)
                            raise MediaProbeError(
                                MediaProbeErrorCode.OUTPUT_LIMIT,
                                "The local media tool exceeded its bounded output allowance.",
                            )
                        time.sleep(limits.poll_interval_seconds)
                    returncode = int(process.returncode or 0)
            except FileNotFoundError as exc:
                raise MediaProbeError(
                    MediaProbeErrorCode.TOOL_UNAVAILABLE,
                    f"The required local media tool is unavailable: {argv[0]}",
                ) from exc
            except OSError as exc:
                raise MediaProbeError(
                    MediaProbeErrorCode.TOOL_UNAVAILABLE,
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

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                pass

    @staticmethod
    def _read_bounded(path: Path, maximum: int) -> bytes:
        with path.open("rb") as handle:
            payload = handle.read(maximum + 1)
        if len(payload) > maximum:
            raise MediaProbeError(
                MediaProbeErrorCode.OUTPUT_LIMIT,
                "The local media tool exceeded its bounded output allowance.",
            )
        return payload


@dataclass(frozen=True)
class MediaProbePolicy:
    ffprobe_limits: ProcessLimits = field(default_factory=ProcessLimits)
    decode_limits: ProcessLimits = field(
        default_factory=lambda: ProcessLimits(
            timeout_seconds=30.0,
            max_stdout_bytes=64 * 1024,
            max_stderr_bytes=1024 * 1024,
        )
    )
    require_decode_smoke_test: bool = True
    reject_encrypted_media: bool = True


@dataclass(frozen=True)
class ToolEvidence:
    executable: str
    version_line: str
    configuration: str | None
    fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "versionLine": self.version_line,
            "configuration": self.configuration,
            "fingerprintSha256": self.fingerprint_sha256,
        }


@dataclass(frozen=True)
class MediaStream:
    index: int
    type: str
    codec_name: str | None
    codec_long_name: str | None
    codec_tag: str | None
    profile: str | None
    duration_ms: int | None
    bit_rate: int | None
    language: str | None
    title: str | None
    default: bool
    forced: bool
    attached_picture: bool
    width: int | None = None
    height: int | None = None
    pixel_format: str | None = None
    frame_rate: str | None = None
    rotation_degrees: float | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_space: str | None = None
    color_range: str | None = None
    hdr: bool = False
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return {_camel_case(key): value for key, value in raw.items()}


@dataclass(frozen=True)
class MediaProbeResult:
    schema_version: str
    source_path: str
    source_size_bytes: int
    source_sha256: str
    format_names: tuple[str, ...]
    format_long_name: str | None
    duration_ms: int | None
    bit_rate: int | None
    programs: int
    chapters: int
    streams: tuple[MediaStream, ...]
    audio_stream_indexes: tuple[int, ...]
    video_stream_indexes: tuple[int, ...]
    subtitle_stream_indexes: tuple[int, ...]
    existing_subtitle_codecs: tuple[str, ...]
    has_hdr_video: bool
    has_rotation_metadata: bool
    decode_smoke_tested: bool
    decode_smoke_test_passed: bool
    warnings: tuple[str, ...]
    ffprobe: ToolEvidence
    ffmpeg: ToolEvidence | None
    probe_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "sourcePath": self.source_path,
            "sourceSizeBytes": self.source_size_bytes,
            "sourceSha256": self.source_sha256,
            "formatNames": list(self.format_names),
            "formatLongName": self.format_long_name,
            "durationMs": self.duration_ms,
            "bitRate": self.bit_rate,
            "programs": self.programs,
            "chapters": self.chapters,
            "streams": [stream.to_dict() for stream in self.streams],
            "audioStreamIndexes": list(self.audio_stream_indexes),
            "videoStreamIndexes": list(self.video_stream_indexes),
            "subtitleStreamIndexes": list(self.subtitle_stream_indexes),
            "existingSubtitleCodecs": list(self.existing_subtitle_codecs),
            "hasHdrVideo": self.has_hdr_video,
            "hasRotationMetadata": self.has_rotation_metadata,
            "decodeSmokeTested": self.decode_smoke_tested,
            "decodeSmokeTestPassed": self.decode_smoke_test_passed,
            "warnings": list(self.warnings),
            "toolchain": {
                "ffprobe": self.ffprobe.to_dict(),
                "ffmpeg": self.ffmpeg.to_dict() if self.ffmpeg else None,
            },
            "probeFingerprintSha256": self.probe_fingerprint_sha256,
        }


@lru_cache(maxsize=1)
def _media_probe_contract_validator() -> Draft202012Validator:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "media-probe.schema.json"
    )
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaError) as exc:
        raise MediaProbeEvidenceError(
            "the media-probe contract is unavailable or invalid"
        ) from exc
    return Draft202012Validator(payload)


def validated_media_probe_payload(value: MediaProbeResult) -> dict[str, Any]:
    """Return schema-validated probe evidence with cross-field checks.

    ``MediaProbeResult`` is intentionally injectable for deterministic tests
    and alternate local probing engines.  The worker therefore validates the
    injected value again at the orchestration boundary rather than trusting a
    dataclass-shaped object.
    """

    if not isinstance(value, MediaProbeResult):
        raise MediaProbeEvidenceError(
            "media probe must return a MediaProbeResult"
        )
    payload = value.to_dict()
    try:
        _media_probe_contract_validator().validate(payload)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "$"
        raise MediaProbeEvidenceError(
            f"media-probe evidence violates the public contract at {location}"
        ) from exc

    streams = {stream.index: stream for stream in value.streams}
    if len(streams) != len(value.streams):
        raise MediaProbeEvidenceError("media stream indexes must be unique")
    stream_indexes = tuple(stream.index for stream in value.streams)
    if stream_indexes != tuple(sorted(stream_indexes)):
        raise MediaProbeEvidenceError(
            "media streams must be ordered by ascending stream index"
        )
    if value.format_names != tuple(sorted(set(value.format_names))):
        raise MediaProbeEvidenceError(
            "formatNames must be sorted and unique"
        )
    expected_audio = tuple(
        stream.index for stream in value.streams if stream.type == "audio"
    )
    expected_video = tuple(
        stream.index
        for stream in value.streams
        if stream.type == "video" and not stream.attached_picture
    )
    expected_subtitle = tuple(
        stream.index for stream in value.streams if stream.type == "subtitle"
    )
    if value.audio_stream_indexes != expected_audio:
        raise MediaProbeEvidenceError(
            "audioStreamIndexes do not match stream evidence"
        )
    if value.video_stream_indexes != expected_video:
        raise MediaProbeEvidenceError(
            "videoStreamIndexes do not match stream evidence"
        )
    if value.subtitle_stream_indexes != expected_subtitle:
        raise MediaProbeEvidenceError(
            "subtitleStreamIndexes do not match stream evidence"
        )
    expected_subtitle_codecs = tuple(
        sorted(
            {
                stream.codec_name
                for stream in value.streams
                if stream.type == "subtitle" and stream.codec_name is not None
            }
        )
    )
    if value.existing_subtitle_codecs != expected_subtitle_codecs:
        raise MediaProbeEvidenceError(
            "existingSubtitleCodecs do not match subtitle stream evidence"
        )
    expected_hdr = any(
        stream.hdr
        for stream in value.streams
        if stream.type == "video" and not stream.attached_picture
    )
    if value.has_hdr_video is not expected_hdr:
        raise MediaProbeEvidenceError(
            "hasHdrVideo does not match video stream evidence"
        )
    expected_rotation = any(
        stream.rotation_degrees is not None
        for stream in value.streams
        if stream.type == "video" and not stream.attached_picture
    )
    if value.has_rotation_metadata is not expected_rotation:
        raise MediaProbeEvidenceError(
            "hasRotationMetadata does not match video stream evidence"
        )
    if value.decode_smoke_tested and not value.decode_smoke_test_passed:
        raise MediaProbeEvidenceError(
            "persisted probe evidence cannot claim a failed decode smoke test"
        )
    if value.decode_smoke_test_passed and not value.decode_smoke_tested:
        raise MediaProbeEvidenceError(
            "decodeSmokeTestPassed requires decodeSmokeTested"
        )
    if value.decode_smoke_tested != (value.ffmpeg is not None):
        raise MediaProbeEvidenceError(
            "FFmpeg tool evidence must exactly match decode smoke-test execution"
        )
    if not value.audio_stream_indexes and not value.video_stream_indexes:
        raise MediaProbeEvidenceError(
            "probe evidence must contain at least one audio or video stream"
        )
    admitted_streams = tuple(
        stream
        for stream in value.streams
        if stream.type == "audio"
        or (stream.type == "video" and not stream.attached_picture)
    )
    unsupported = tuple(
        stream.index
        for stream in admitted_streams
        if not stream.codec_name or stream.codec_name.lower() == "unknown"
    )
    if unsupported:
        raise MediaProbeEvidenceError(
            "admitted media streams must identify a decodable codec"
        )
    expected_fingerprint = _probe_fingerprint_sha256(
        source_path=value.source_path,
        source_size_bytes=value.source_size_bytes,
        source_sha256=value.source_sha256,
        format_names=value.format_names,
        duration_ms=value.duration_ms,
        streams=value.streams,
        ffprobe_fingerprint=value.ffprobe.fingerprint_sha256,
        ffmpeg_fingerprint=(
            value.ffmpeg.fingerprint_sha256 if value.ffmpeg else None
        ),
    )
    if value.probe_fingerprint_sha256 != expected_fingerprint:
        raise MediaProbeEvidenceError(
            "probeFingerprintSha256 does not match probe evidence"
        )
    return payload


class MediaProbe:
    """Probe local media by content and optionally prove first-frame decoding."""

    def __init__(
        self,
        *,
        ffprobe_command: Sequence[str] = ("ffprobe",),
        ffmpeg_command: Sequence[str] = ("ffmpeg",),
        runner: ProcessRunner | None = None,
        policy: MediaProbePolicy | None = None,
    ) -> None:
        self.ffprobe_command = _validated_command(ffprobe_command, "ffprobe")
        self.ffmpeg_command = _validated_command(ffmpeg_command, "ffmpeg")
        self.runner = runner or BoundedProcessRunner()
        self.policy = policy or MediaProbePolicy()
        self._tool_evidence: dict[tuple[str, ...], ToolEvidence] = {}

    def probe(self, source: str | Path) -> MediaProbeResult:
        canonical = canonical_local_media_file(source)
        source_before = canonical.stat()
        source_sha256_before = _sha256_file(canonical)
        source_after_initial_hash = canonical.stat()
        if _stat_file_identity(source_before) != _stat_file_identity(
            source_after_initial_hash
        ):
            raise MediaProbeError(
                MediaProbeErrorCode.SOURCE_CHANGED,
                "The media source changed while admission evidence was collected.",
            )
        ffprobe_evidence = self._inspect_tool(
            self.ffprobe_command,
            limits=self.policy.ffprobe_limits,
        )
        probe_result = self.runner.run(
            (
                *self.ffprobe_command,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-show_programs",
                "-show_chapters",
                str(canonical),
            ),
            limits=self.policy.ffprobe_limits,
        )
        if probe_result.returncode != 0:
            raise MediaProbeError(
                MediaProbeErrorCode.PROBE_FAILED,
                "FFprobe could not identify this file as supported local media.",
                detail=_safe_detail(probe_result.stderr),
            )
        payload = _decode_probe_json(probe_result.stdout)
        if self.policy.reject_encrypted_media and _contains_encryption(payload):
            raise MediaProbeError(
                MediaProbeErrorCode.ENCRYPTED_MEDIA,
                "Encrypted or DRM-protected media is not accepted.",
            )

        raw_streams = payload.get("streams")
        if not isinstance(raw_streams, list):
            raise MediaProbeError(
                MediaProbeErrorCode.MALFORMED_PROBE,
                "FFprobe did not return a valid streams array.",
            )
        if any(not isinstance(raw, Mapping) for raw in raw_streams):
            raise MediaProbeError(
                MediaProbeErrorCode.MALFORMED_PROBE,
                "FFprobe returned a malformed media stream entry.",
            )
        parsed_streams = tuple(
            _parse_stream(raw, fallback_index=index)
            for index, raw in enumerate(raw_streams)
        )
        if len({stream.index for stream in parsed_streams}) != len(
            parsed_streams
        ):
            raise MediaProbeError(
                MediaProbeErrorCode.MALFORMED_PROBE,
                "FFprobe returned duplicate media stream indexes.",
            )
        streams = tuple(sorted(parsed_streams, key=lambda stream: stream.index))
        media_streams = tuple(
            stream
            for stream in streams
            if stream.type == "audio"
            or (stream.type == "video" and not stream.attached_picture)
        )
        if not media_streams:
            raise MediaProbeError(
                MediaProbeErrorCode.NO_MEDIA_STREAM,
                "The file contains no audio or video stream.",
            )
        unsupported = tuple(
            stream.index
            for stream in media_streams
            if not stream.codec_name or stream.codec_name.lower() == "unknown"
        )
        if unsupported:
            joined = ", ".join(str(index) for index in unsupported)
            raise MediaProbeError(
                MediaProbeErrorCode.UNSUPPORTED_CODEC,
                f"FFprobe reported an unknown codec for media stream(s): {joined}.",
            )

        audio_indexes = tuple(
            stream.index for stream in streams if stream.type == "audio"
        )
        video_indexes = tuple(
            stream.index
            for stream in streams
            if stream.type == "video" and not stream.attached_picture
        )
        subtitle_streams = tuple(
            stream for stream in streams if stream.type == "subtitle"
        )
        warnings: list[str] = []
        format_payload = payload.get("format")
        if not isinstance(format_payload, Mapping):
            format_payload = MappingProxyType({})
            warnings.append("format-metadata-missing")
        duration_ms = _duration_ms(format_payload.get("duration"))
        if duration_ms is None:
            stream_durations = tuple(
                value
                for value in (stream.duration_ms for stream in media_streams)
                if value is not None
            )
            duration_ms = max(stream_durations, default=None)
        if duration_ms is None:
            warnings.append("duration-unknown")
        if not audio_indexes:
            warnings.append("audio-stream-missing")
        if not video_indexes:
            warnings.append("video-stream-missing")

        ffmpeg_evidence: ToolEvidence | None = None
        decode_passed = False
        if self.policy.require_decode_smoke_test:
            ffmpeg_evidence = self._inspect_tool(
                self.ffmpeg_command,
                limits=self.policy.decode_limits,
            )
            self._decode_smoke_test(
                canonical,
                audio_stream_indexes=audio_indexes,
                video_stream_indexes=video_indexes,
            )
            decode_passed = True

        format_names = tuple(
            sorted(
                {
                    value.strip()
                    for value in str(format_payload.get("format_name") or "").split(
                        ","
                    )
                    if value.strip()
                }
            )
        )
        existing_subtitle_codecs = tuple(
            sorted(
                {
                    stream.codec_name
                    for stream in subtitle_streams
                    if stream.codec_name is not None
                }
            )
        )
        source_sha256 = _sha256_file(canonical)
        source_after = canonical.stat()
        if (
            _stat_file_identity(source_before)
            != _stat_file_identity(source_after)
            or source_sha256_before != source_sha256
        ):
            raise MediaProbeError(
                MediaProbeErrorCode.SOURCE_CHANGED,
                "The media source changed while admission evidence was collected.",
            )
        fingerprint = _probe_fingerprint_sha256(
            source_path=str(canonical),
            source_size_bytes=source_after.st_size,
            source_sha256=source_sha256,
            format_names=format_names,
            duration_ms=duration_ms,
            streams=streams,
            ffprobe_fingerprint=ffprobe_evidence.fingerprint_sha256,
            ffmpeg_fingerprint=(
                ffmpeg_evidence.fingerprint_sha256 if ffmpeg_evidence else None
            ),
        )

        return MediaProbeResult(
            schema_version=MEDIA_PROBE_SCHEMA_VERSION,
            source_path=str(canonical),
            source_size_bytes=source_after.st_size,
            source_sha256=source_sha256,
            format_names=format_names,
            format_long_name=_optional_text(format_payload.get("format_long_name")),
            duration_ms=duration_ms,
            bit_rate=_optional_non_negative_int(format_payload.get("bit_rate")),
            programs=_list_length(payload.get("programs")),
            chapters=_list_length(payload.get("chapters")),
            streams=streams,
            audio_stream_indexes=audio_indexes,
            video_stream_indexes=video_indexes,
            subtitle_stream_indexes=tuple(
                stream.index for stream in subtitle_streams
            ),
            existing_subtitle_codecs=existing_subtitle_codecs,
            has_hdr_video=any(
                stream.hdr
                for stream in streams
                if stream.type == "video" and not stream.attached_picture
            ),
            has_rotation_metadata=any(
                stream.rotation_degrees is not None
                for stream in streams
                if stream.type == "video" and not stream.attached_picture
            ),
            decode_smoke_tested=self.policy.require_decode_smoke_test,
            decode_smoke_test_passed=decode_passed,
            warnings=tuple(warnings),
            ffprobe=ffprobe_evidence,
            ffmpeg=ffmpeg_evidence,
            probe_fingerprint_sha256=fingerprint,
        )

    def _inspect_tool(
        self,
        command: tuple[str, ...],
        *,
        limits: ProcessLimits,
    ) -> ToolEvidence:
        cached = self._tool_evidence.get(command)
        if cached is not None:
            return cached
        result = self.runner.run((*command, "-version"), limits=limits)
        if result.returncode != 0:
            raise MediaProbeError(
                MediaProbeErrorCode.TOOL_UNAVAILABLE,
                f"The local media tool failed its version check: {command[0]}",
                detail=_safe_detail(result.stderr),
            )
        text = result.stdout.decode("utf-8", errors="replace").strip()
        lines = tuple(line.strip() for line in text.splitlines() if line.strip())
        if not lines:
            raise MediaProbeError(
                MediaProbeErrorCode.TOOL_UNAVAILABLE,
                f"The local media tool returned no version evidence: {command[0]}",
            )
        configuration = next(
            (
                line.removeprefix("configuration:").strip()
                for line in lines
                if line.startswith("configuration:")
            ),
            None,
        )
        evidence = ToolEvidence(
            executable=command[0],
            version_line=lines[0][:1_024],
            configuration=configuration[:8_192] if configuration else None,
            fingerprint_sha256=hashlib.sha256(result.stdout).hexdigest(),
        )
        self._tool_evidence[command] = evidence
        return evidence

    def _decode_smoke_test(
        self,
        source: Path,
        *,
        audio_stream_indexes: Sequence[int],
        video_stream_indexes: Sequence[int],
    ) -> None:
        command = [
            *self.ffmpeg_command,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(source),
        ]
        if audio_stream_indexes:
            for stream_index in audio_stream_indexes:
                command.extend(("-map", f"0:{stream_index}"))
            command.extend(("-frames:a", "1"))
        else:
            command.append("-an")
        if video_stream_indexes:
            for stream_index in video_stream_indexes:
                command.extend(("-map", f"0:{stream_index}"))
            command.extend(("-frames:v", "1"))
        else:
            command.append("-vn")
        command.extend(("-sn", "-dn", "-f", "null", "-"))
        result = self.runner.run(command, limits=self.policy.decode_limits)
        if result.returncode != 0:
            raise MediaProbeError(
                MediaProbeErrorCode.DECODE_FAILED,
                "The installed FFmpeg build could not decode this media safely.",
                detail=_safe_detail(result.stderr),
            )


def canonical_local_media_file(source: str | Path) -> Path:
    """Return a canonical local regular file without consulting its extension."""

    text = os.fspath(source).strip()
    if not text or "\x00" in text or _URL_SCHEME.match(text):
        raise MediaProbeError(
            MediaProbeErrorCode.INVALID_PATH,
            "The media source must be a non-empty absolute local file path.",
        )
    normalized = text.replace("/", "\\")
    if normalized.startswith("\\\\"):
        raise MediaProbeError(
            MediaProbeErrorCode.INVALID_PATH,
            "UNC, device, and network media paths are not accepted.",
        )
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise MediaProbeError(
            MediaProbeErrorCode.INVALID_PATH,
            "The media source must use an absolute local path.",
        )
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MediaProbeError(
            MediaProbeErrorCode.NOT_FOUND,
            "The media source does not exist or cannot be resolved.",
            detail=str(exc),
        ) from exc
    if not canonical.is_file():
        raise MediaProbeError(
            MediaProbeErrorCode.NOT_REGULAR_FILE,
            "The media source must be a regular local file.",
        )
    return canonical


def _probe_fingerprint_sha256(
    *,
    source_path: str,
    source_size_bytes: int,
    source_sha256: str,
    format_names: Sequence[str],
    duration_ms: int | None,
    streams: Sequence[MediaStream],
    ffprobe_fingerprint: str,
    ffmpeg_fingerprint: str | None,
) -> str:
    payload = {
        "sourcePath": source_path,
        "sourceSizeBytes": source_size_bytes,
        "sourceSha256": source_sha256,
        "formatNames": list(format_names),
        "durationMs": duration_ms,
        "streams": [stream.to_dict() for stream in streams],
        "ffprobe": ffprobe_fingerprint,
        "ffmpeg": ffmpeg_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_size),
        int(getattr(value, "st_dev", 0)),
        int(getattr(value, "st_ino", 0)),
    )


def _validated_command(command: Sequence[str], label: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in command)
    if not normalized or any(not value or "\x00" in value for value in normalized):
        raise ValueError(f"{label}_command must contain non-empty argv entries")
    return normalized


def _decode_probe_json(payload: bytes) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        decoded = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MediaProbeError(
            MediaProbeErrorCode.MALFORMED_PROBE,
            "FFprobe returned malformed UTF-8 JSON.",
            detail=str(exc),
        ) from exc
    if not isinstance(decoded, Mapping):
        raise MediaProbeError(
            MediaProbeErrorCode.MALFORMED_PROBE,
            "FFprobe returned a non-object JSON document.",
        )
    return decoded


def _parse_stream(raw: Mapping[str, Any], *, fallback_index: int) -> MediaStream:
    stream_type = str(raw.get("codec_type") or "unknown").strip().lower()
    disposition = raw.get("disposition")
    if not isinstance(disposition, Mapping):
        disposition = MappingProxyType({})
    tags = raw.get("tags")
    if not isinstance(tags, Mapping):
        tags = MappingProxyType({})
    rotation = _rotation(raw, tags)
    color_transfer = _optional_text(raw.get("color_transfer"))
    side_data = raw.get("side_data_list")
    side_data_types = (
        tuple(
            str(value.get("side_data_type") or "").strip().lower()
            for value in side_data
            if isinstance(value, Mapping)
        )
        if isinstance(side_data, list)
        else ()
    )
    hdr = bool(
        (color_transfer and color_transfer.lower() in _HDR_TRANSFERS)
        or any(
            marker in side_type
            for side_type in side_data_types
            for marker in _HDR_SIDE_DATA_MARKERS
        )
    )
    return MediaStream(
        index=_optional_non_negative_int(raw.get("index"), default=fallback_index)
        or 0,
        type=stream_type,
        codec_name=_optional_text(raw.get("codec_name")),
        codec_long_name=_optional_text(raw.get("codec_long_name")),
        codec_tag=_optional_text(raw.get("codec_tag_string")),
        profile=_optional_text(raw.get("profile")),
        duration_ms=_duration_ms(raw.get("duration")),
        bit_rate=_optional_non_negative_int(raw.get("bit_rate")),
        language=_optional_text(tags.get("language")),
        title=_optional_text(tags.get("title")),
        default=_truthy(disposition.get("default")),
        forced=_truthy(disposition.get("forced")),
        attached_picture=_truthy(disposition.get("attached_pic")),
        width=_optional_non_negative_int(raw.get("width")),
        height=_optional_non_negative_int(raw.get("height")),
        pixel_format=_optional_text(raw.get("pix_fmt")),
        frame_rate=_optional_text(raw.get("avg_frame_rate")),
        rotation_degrees=rotation,
        color_primaries=_optional_text(raw.get("color_primaries")),
        color_transfer=color_transfer,
        color_space=_optional_text(raw.get("color_space")),
        color_range=_optional_text(raw.get("color_range")),
        hdr=hdr,
        sample_rate=_optional_non_negative_int(raw.get("sample_rate")),
        channels=_optional_non_negative_int(raw.get("channels")),
        channel_layout=_optional_text(raw.get("channel_layout")),
    )


def _rotation(
    raw: Mapping[str, Any],
    tags: Mapping[str, Any],
) -> float | None:
    tagged = _optional_float(tags.get("rotate"))
    if tagged is not None:
        return tagged
    side_data = raw.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if not isinstance(item, Mapping):
                continue
            rotation = _optional_float(item.get("rotation"))
            if rotation is not None:
                return rotation
    return None


def _contains_encryption(value: Any) -> bool:
    if isinstance(value, Mapping):
        codec_tag = str(value.get("codec_tag_string") or "").strip().lower()
        if codec_tag in _ENCRYPTED_CODEC_TAGS:
            return True
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in _ENCRYPTION_TAG_KEYS and _truthy(child):
                return True
            if key == "side_data_type" and "encrypt" in str(child).lower():
                return True
            if _contains_encryption(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_encryption(child) for child in value)
    return False


def _duration_ms(value: Any) -> int | None:
    seconds = _optional_float(value)
    if seconds is None or seconds < 0:
        return None
    return round(seconds * 1_000)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_non_negative_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "encrypted",
        "cenc",
        "cbcs",
    }


def _list_length(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _safe_detail(payload: bytes) -> str | None:
    text = payload.decode("utf-8", errors="replace").strip()
    return text[-4_096:] if text else None


def _camel_case(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)
