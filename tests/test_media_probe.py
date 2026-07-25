from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from backend.media_probe import (
    BoundedProcessRunner,
    MediaProbe,
    MediaProbeError,
    MediaProbeErrorCode,
    MediaProbeEvidenceError,
    MediaProbePolicy,
    ProcessLimits,
    ProcessResult,
    canonical_local_media_file,
    validated_media_probe_payload,
)


class FixtureRunner:
    def __init__(
        self,
        probe_payload: dict[str, Any],
        *,
        probe_returncode: int = 0,
        decode_returncode: int = 0,
    ) -> None:
        self.probe_payload = probe_payload
        self.probe_returncode = probe_returncode
        self.decode_returncode = decode_returncode
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        command: tuple[str, ...] | list[str],
        *,
        limits: ProcessLimits,
    ) -> ProcessResult:
        del limits
        argv = tuple(command)
        self.commands.append(argv)
        if "-version" in argv:
            executable = Path(argv[0]).name
            return ProcessResult(
                returncode=0,
                stdout=(
                    f"{executable} version 8.0-test\n"
                    "configuration: --enable-gpl --enable-libass\n"
                ).encode(),
                stderr=b"",
                elapsed_ms=1,
            )
        if "-show_streams" in argv:
            return ProcessResult(
                returncode=self.probe_returncode,
                stdout=json.dumps(self.probe_payload).encode(),
                stderr=b"probe fixture failed" if self.probe_returncode else b"",
                elapsed_ms=2,
            )
        return ProcessResult(
            returncode=self.decode_returncode,
            stdout=b"",
            stderr=b"decode fixture failed" if self.decode_returncode else b"",
            elapsed_ms=3,
        )


def media_payload() -> dict[str, Any]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_long_name": "H.264",
                "profile": "High",
                "codec_type": "video",
                "codec_tag_string": "avc1",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p10le",
                "avg_frame_rate": "30000/1001",
                "duration": "12.345",
                "bit_rate": "4000000",
                "color_space": "bt2020nc",
                "color_transfer": "smpte2084",
                "color_primaries": "bt2020",
                "color_range": "tv",
                "disposition": {"default": 1, "attached_pic": 0},
                "tags": {"language": "und", "rotate": "90"},
                "side_data_list": [
                    {"side_data_type": "Mastering display metadata"}
                ],
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
                "channel_layout": "stereo",
                "duration": "12.345",
                "disposition": {"default": 1},
                "tags": {"language": "zho", "title": "Main mix"},
            },
            {
                "index": 2,
                "codec_name": "mov_text",
                "codec_type": "subtitle",
                "disposition": {"default": 0, "forced": 0},
                "tags": {"language": "eng", "title": "Existing captions"},
            },
        ],
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "format_long_name": "QuickTime / MOV",
            "duration": "12.345",
            "bit_rate": "4200000",
        },
        "programs": [],
        "chapters": [{"id": 0}],
    }


def test_extensionless_media_is_accepted_by_content(tmp_path: Path) -> None:
    source = tmp_path / "recording-without-extension"
    source.write_bytes(b"fixture")
    runner = FixtureRunner(media_payload())

    result = MediaProbe(runner=runner).probe(source)

    assert result.source_path == str(source.resolve())
    assert result.audio_stream_indexes == (1,)
    assert result.video_stream_indexes == (0,)
    assert result.subtitle_stream_indexes == (2,)
    assert result.existing_subtitle_codecs == ("mov_text",)
    assert result.has_hdr_video is True
    assert result.has_rotation_metadata is True
    assert result.decode_smoke_test_passed is True
    assert any("-show_streams" in command for command in runner.commands)
    decode_command = runner.commands[-1]
    assert "-frames:a" in decode_command
    assert "-frames:v" in decode_command
    assert decode_command[-3:] == ("-f", "null", "-")


def test_misleading_extension_is_not_an_admission_rule(tmp_path: Path) -> None:
    source = tmp_path / "meeting.not-a-media-extension"
    source.write_bytes(b"fixture")
    payload = media_payload()
    payload["streams"] = [payload["streams"][1]]
    runner = FixtureRunner(payload)

    result = MediaProbe(runner=runner).probe(source)

    assert result.audio_stream_indexes == (1,)
    assert result.video_stream_indexes == ()
    assert "video-stream-missing" in result.warnings


def test_probe_result_validates_against_contract(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"fixture")
    result = MediaProbe(runner=FixtureRunner(media_payload())).probe(source)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "contracts" / "media-probe.schema.json"
        ).read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(schema).validate(result.to_dict())
    assert len(result.probe_fingerprint_sha256) == 64
    assert result.ffprobe.version_line == "ffprobe version 8.0-test"
    assert result.ffmpeg is not None
    assert result.ffmpeg.version_line == "ffmpeg version 8.0-test"


def test_probe_fingerprint_tampering_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"fixture")
    result = MediaProbe(runner=FixtureRunner(media_payload())).probe(source)

    with pytest.raises(MediaProbeEvidenceError):
        validated_media_probe_payload(
            replace(result, probe_fingerprint_sha256="0" * 64)
        )


def test_decode_smoke_test_maps_every_admitted_stream_by_exact_index(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cover-first.media"
    source.write_bytes(b"fixture")
    payload = media_payload()
    payload["streams"] = [
        {
            "index": 0,
            "codec_name": "mjpeg",
            "codec_type": "video",
            "disposition": {"attached_pic": 1},
        },
        {
            "index": 2,
            "codec_name": "aac",
            "codec_type": "audio",
            "sample_rate": "48000",
            "channels": 2,
            "disposition": {"default": 1},
        },
        {
            "index": 4,
            "codec_name": "h264",
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "color_transfer": "bt709",
            "disposition": {"attached_pic": 0},
        },
    ]
    runner = FixtureRunner(payload)

    result = MediaProbe(runner=runner).probe(source)

    assert result.audio_stream_indexes == (2,)
    assert result.video_stream_indexes == (4,)
    decode_command = runner.commands[-1]
    mapped_streams = [
        decode_command[index + 1]
        for index, value in enumerate(decode_command[:-1])
        if value == "-map"
    ]
    assert mapped_streams == ["0:2", "0:4"]
    assert "0:0" not in mapped_streams


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda payload: payload.update({"streams": []}),
            MediaProbeErrorCode.NO_MEDIA_STREAM,
        ),
        (
            lambda payload: payload["streams"][0].update({"codec_name": "unknown"}),
            MediaProbeErrorCode.UNSUPPORTED_CODEC,
        ),
        (
            lambda payload: payload["streams"][0].update(
                {"codec_tag_string": "encv"}
            ),
            MediaProbeErrorCode.ENCRYPTED_MEDIA,
        ),
        (
            lambda payload: payload["streams"][0].update(
                {"tags": {"encryption_scheme": "cenc"}}
            ),
            MediaProbeErrorCode.ENCRYPTED_MEDIA,
        ),
    ],
)
def test_fail_closed_content_rejections(
    tmp_path: Path,
    mutation: Any,
    expected_code: MediaProbeErrorCode,
) -> None:
    source = tmp_path / "untrusted.bin"
    source.write_bytes(b"fixture")
    payload = media_payload()
    mutation(payload)

    with pytest.raises(MediaProbeError) as raised:
        MediaProbe(runner=FixtureRunner(payload)).probe(source)

    assert raised.value.code is expected_code


def test_decode_failure_rejects_identifiable_but_unsupported_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "future-codec.media"
    source.write_bytes(b"fixture")

    with pytest.raises(MediaProbeError) as raised:
        MediaProbe(
            runner=FixtureRunner(media_payload(), decode_returncode=1)
        ).probe(source)

    assert raised.value.code is MediaProbeErrorCode.DECODE_FAILED
    assert raised.value.detail == "decode fixture failed"


def test_probe_failure_does_not_fall_back_to_extension(tmp_path: Path) -> None:
    source = tmp_path / "looks-valid.mp4"
    source.write_bytes(b"not media")

    with pytest.raises(MediaProbeError) as raised:
        MediaProbe(
            runner=FixtureRunner(media_payload(), probe_returncode=1)
        ).probe(source)

    assert raised.value.code is MediaProbeErrorCode.PROBE_FAILED


def test_metadata_timestamp_change_with_stable_bytes_is_accepted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stable.wav"
    source.write_bytes(b"fixture")

    class TimestampTouchRunner(FixtureRunner):
        def run(
            self,
            command: tuple[str, ...] | list[str],
            *,
            limits: ProcessLimits,
        ) -> ProcessResult:
            result = super().run(command, limits=limits)
            if "-show_streams" in command:
                stat = source.stat()
                os.utime(
                    source,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
                )
            return result

    result = MediaProbe(runner=TimestampTouchRunner(media_payload())).probe(
        source
    )

    assert result.source_sha256
    assert result.source_size_bytes == len(b"fixture")


def test_content_change_during_probe_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "mutated.wav"
    source.write_bytes(b"fixture")

    class ContentMutationRunner(FixtureRunner):
        def run(
            self,
            command: tuple[str, ...] | list[str],
            *,
            limits: ProcessLimits,
        ) -> ProcessResult:
            result = super().run(command, limits=limits)
            if "-show_streams" in command:
                source.write_bytes(b"changed")
            return result

    with pytest.raises(MediaProbeError) as raised:
        MediaProbe(runner=ContentMutationRunner(media_payload())).probe(source)

    assert raised.value.code is MediaProbeErrorCode.SOURCE_CHANGED


def test_local_path_boundary_rejects_url_relative_and_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(MediaProbeError) as url_error:
        canonical_local_media_file("https://example.test/media.mp4")
    assert url_error.value.code is MediaProbeErrorCode.INVALID_PATH

    with pytest.raises(MediaProbeError) as relative_error:
        canonical_local_media_file("media.mp4")
    assert relative_error.value.code is MediaProbeErrorCode.INVALID_PATH

    with pytest.raises(MediaProbeError) as directory_error:
        canonical_local_media_file(tmp_path)
    assert directory_error.value.code is MediaProbeErrorCode.NOT_REGULAR_FILE


def test_bounded_runner_enforces_output_limit(tmp_path: Path) -> None:
    script = tmp_path / "large_output.py"
    script.write_text("import sys\nsys.stdout.write('x' * 200000)\n", encoding="utf-8")

    with pytest.raises(MediaProbeError) as raised:
        BoundedProcessRunner().run(
            (sys.executable, str(script)),
            limits=ProcessLimits(
                timeout_seconds=5,
                max_stdout_bytes=1_024,
                max_stderr_bytes=1_024,
            ),
        )

    assert raised.value.code is MediaProbeErrorCode.OUTPUT_LIMIT


def test_bounded_runner_enforces_timeout(tmp_path: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")

    with pytest.raises(MediaProbeError) as raised:
        BoundedProcessRunner().run(
            (sys.executable, str(script)),
            limits=ProcessLimits(
                timeout_seconds=0.05,
                max_stdout_bytes=1_024,
                max_stderr_bytes=1_024,
                poll_interval_seconds=0.005,
            ),
        )

    assert raised.value.code is MediaProbeErrorCode.TOOL_TIMEOUT
