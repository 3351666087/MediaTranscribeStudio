from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator

from backend.media_probe import (
    MediaProbeResult,
    MediaStream,
    ProcessResult,
    ToolEvidence,
)
from backend.subtitle_delivery import (
    BoundedDeliveryRunner,
    BurnInVideoStrategy,
    DeliveryProcessLimits,
    KaraokeTimingEvidence,
    KaraokeTimingSource,
    SubtitleDeliveryError,
    SubtitleDeliveryErrorCode,
    SubtitleDeliveryExecutor,
    SubtitleDeliveryPolicy,
)
from backend.subtitles import build_subtitle_output_plan


SRT_TEXT = """1
00:00:00,000 --> 00:00:02,000
Hello world
"""
WEBVTT_TEXT = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello world
"""
ASS_TEXT = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello world
"""
KARAOKE_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\k20}Hello {\\kf30}world
"""


def _tool(name: str) -> ToolEvidence:
    return ToolEvidence(
        executable=name,
        version_line=f"{name} version test",
        configuration="--enable-libass --enable-gpl",
        fingerprint_sha256=hashlib.sha256(name.encode()).hexdigest(),
    )


def _stream(
    index: int,
    stream_type: str,
    codec: str,
    *,
    attached_picture: bool = False,
) -> MediaStream:
    return MediaStream(
        index=index,
        type=stream_type,
        codec_name=codec,
        codec_long_name=codec,
        codec_tag=None,
        profile=None,
        duration_ms=10_000,
        bit_rate=None,
        language="und",
        title=None,
        default=index == 0,
        forced=False,
        attached_picture=attached_picture,
        width=1920 if stream_type == "video" else None,
        height=1080 if stream_type == "video" else None,
        pixel_format="yuv420p" if stream_type == "video" else None,
        frame_rate="30/1" if stream_type == "video" else None,
        sample_rate=48_000 if stream_type == "audio" else None,
        channels=2 if stream_type == "audio" else None,
        channel_layout="stereo" if stream_type == "audio" else None,
    )


def _probe_result(
    path: Path,
    *,
    audio: tuple[str, ...] = ("aac",),
    video: tuple[str, ...] = ("h264",),
    subtitles: tuple[str, ...] = (),
    duration_ms: int | None = 10_000,
    chapters: int = 1,
    hdr: bool = False,
    rotation: bool = False,
) -> MediaProbeResult:
    streams: list[MediaStream] = []
    for codec in video:
        streams.append(_stream(len(streams), "video", codec))
    for codec in audio:
        streams.append(_stream(len(streams), "audio", codec))
    for codec in subtitles:
        streams.append(_stream(len(streams), "subtitle", codec))
    fingerprint = hashlib.sha256(
        (
            str(path.resolve())
            + repr(audio)
            + repr(video)
            + repr(subtitles)
            + repr(duration_ms)
        ).encode()
    ).hexdigest()
    return MediaProbeResult(
        schema_version="1.0.0",
        source_path=str(path.resolve(strict=True)),
        source_size_bytes=path.stat().st_size,
        format_names=("mov", "mp4"),
        format_long_name="fixture",
        duration_ms=duration_ms,
        bit_rate=1_000_000,
        programs=0,
        chapters=chapters,
        streams=tuple(streams),
        audio_stream_indexes=tuple(
            stream.index for stream in streams if stream.type == "audio"
        ),
        video_stream_indexes=tuple(
            stream.index
            for stream in streams
            if stream.type == "video" and not stream.attached_picture
        ),
        subtitle_stream_indexes=tuple(
            stream.index for stream in streams if stream.type == "subtitle"
        ),
        existing_subtitle_codecs=tuple(sorted(set(subtitles))),
        has_hdr_video=hdr,
        has_rotation_metadata=rotation,
        decode_smoke_tested=True,
        decode_smoke_test_passed=True,
        warnings=(),
        ffprobe=_tool("ffprobe"),
        ffmpeg=_tool("ffmpeg"),
        probe_fingerprint_sha256=fingerprint,
    )


class ProbeFixture:
    def __init__(
        self,
        source: Path,
        *,
        source_audio: tuple[str, ...] = ("aac",),
        source_video: tuple[str, ...] = ("h264",),
        source_subtitles: tuple[str, ...] = (),
        output_audio: tuple[str, ...] | None = None,
        output_video: tuple[str, ...] | None = None,
        output_subtitles: tuple[str, ...] = (),
        source_duration_ms: int | None = 10_000,
        output_duration_ms: int | None = 10_000,
        source_chapters: int = 1,
        output_chapters: int = 1,
        source_hdr: bool = False,
        output_hdr: bool | None = None,
        source_rotation: bool = False,
        output_rotation: bool | None = None,
        output_sequence: list[dict[str, Any]] | None = None,
    ) -> None:
        self.source = source.resolve()
        self.source_options = {
            "audio": source_audio,
            "video": source_video,
            "subtitles": source_subtitles,
            "duration_ms": source_duration_ms,
            "chapters": source_chapters,
            "hdr": source_hdr,
            "rotation": source_rotation,
        }
        self.output_options = {
            "audio": output_audio if output_audio is not None else source_audio,
            "video": output_video if output_video is not None else source_video,
            "subtitles": output_subtitles,
            "duration_ms": output_duration_ms,
            "chapters": output_chapters,
            "hdr": source_hdr if output_hdr is None else output_hdr,
            "rotation": (
                source_rotation if output_rotation is None else output_rotation
            ),
        }
        self.output_sequence = list(output_sequence or [])
        self.calls: list[Path] = []

    def probe(self, source: str | Path) -> MediaProbeResult:
        path = Path(source).resolve(strict=True)
        self.calls.append(path)
        if os.path.samefile(path, self.source):
            return _probe_result(path, **self.source_options)
        options = dict(self.output_options)
        if self.output_sequence:
            options.update(self.output_sequence.pop(0))
        return _probe_result(path, **options)


class RunnerFixture:
    def __init__(
        self,
        *,
        returncode: int = 0,
        create_output: bool = True,
        output_bytes: bytes = b"derived-media-fixture",
        encoders: tuple[str, ...] = (
            "mov_text",
            "webvtt",
            "ass",
            "subrip",
            "libx264",
            "libx265",
            "libvpx-vp9",
            "libsvtav1",
            "prores_ks",
            "ffv1",
        ),
        filters: tuple[str, ...] = ("subtitles",),
        muxers: tuple[str, ...] = ("mp4", "mov", "matroska", "webm"),
        capability_returncode: int = 0,
        on_delivery: Callable[[tuple[str, ...]], None] | None = None,
    ) -> None:
        self.returncode = returncode
        self.create_output = create_output
        self.output_bytes = output_bytes
        self.encoders = encoders
        self.filters = filters
        self.muxers = muxers
        self.capability_returncode = capability_returncode
        self.on_delivery = on_delivery
        self.commands: list[tuple[str, ...]] = []
        self.limits: list[DeliveryProcessLimits] = []

    def run(
        self,
        command: tuple[str, ...] | list[str],
        *,
        limits: DeliveryProcessLimits,
    ) -> ProcessResult:
        argv = tuple(command)
        self.commands.append(argv)
        self.limits.append(limits)
        if "-encoders" in argv:
            return self._capability_result(self.encoders, "V.....")
        if "-filters" in argv:
            return self._capability_result(self.filters, "TSC")
        if "-muxers" in argv:
            return self._capability_result(self.muxers, "E")
        if self.on_delivery is not None:
            self.on_delivery(argv)
        if self.create_output:
            Path(argv[-1]).write_bytes(self.output_bytes)
        return ProcessResult(
            returncode=self.returncode,
            stdout=b"",
            stderr=b"fixture ffmpeg failure" if self.returncode else b"",
            elapsed_ms=25,
        )

    def _capability_result(
        self,
        names: tuple[str, ...],
        flags: str,
    ) -> ProcessResult:
        payload = "\n".join(f" {flags} {name} fixture" for name in names).encode()
        return ProcessResult(
            returncode=self.capability_returncode,
            stdout=payload,
            stderr=b"capability failed" if self.capability_returncode else b"",
            elapsed_ms=2,
        )


def _write_subtitle(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _soft_plan(
    source: Path,
    output: Path,
    subtitle: Path,
    *,
    subtitle_format: str = "srt",
    subtitle_codec: str | None = None,
):
    return build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_path=subtitle,
        subtitle_format=subtitle_format,
        subtitle_codec=subtitle_codec,
        mode="soft-mux",
    )


def _burn_plan(
    source: Path,
    output: Path,
    subtitle: Path,
    *,
    subtitle_format: str = "ass",
):
    return build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_path=subtitle,
        subtitle_format=subtitle_format,
        mode="burn-in",
    )


def _schema() -> dict[str, Any]:
    return json.loads(
        (
            Path(__file__).parents[1]
            / "contracts"
            / "subtitle-delivery.schema.json"
        ).read_text(encoding="utf-8")
    )


def test_sidecar_delivery_is_atomic_source_immutable_and_schema_valid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable-source")
    output = tmp_path / "captions.srt"
    runner = RunnerFixture()
    probe = ProbeFixture(source)
    plan = build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_format="srt",
        mode="sidecar",
    )

    receipt = SubtitleDeliveryExecutor(runner=runner, probe=probe).deliver(
        plan,
        sidecar_payload=SRT_TEXT,
    )

    assert output.read_text(encoding="utf-8") == SRT_TEXT
    assert source.read_bytes() == b"immutable-source"
    assert runner.commands == []
    assert probe.calls == [source.resolve()]
    assert receipt.source_before == receipt.source_after
    assert receipt.qa.post_publish_probe_verified is False
    assert not list(tmp_path.glob(".*.mts-*"))
    Draft202012Validator(_schema()).validate(receipt.to_dict())


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "captions.srt"
    source.write_bytes(b"source")
    output.write_bytes(b"keep-me")
    plan = build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_format="srt",
        mode="sidecar",
    )

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(probe=ProbeFixture(source)).deliver(
            plan,
            sidecar_payload=SRT_TEXT,
        )

    assert raised.value.code is SubtitleDeliveryErrorCode.OUTPUT_EXISTS
    assert output.read_bytes() == b"keep-me"


def test_hardlink_alias_to_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "alias.srt"
    source.write_bytes(b"source")
    os.link(source, output)
    plan = build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_format="srt",
        mode="sidecar",
    )

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(probe=ProbeFixture(source)).deliver(
            plan,
            sidecar_payload=SRT_TEXT,
        )

    assert raised.value.code is SubtitleDeliveryErrorCode.PATH_ALIAS
    assert source.read_bytes() == b"source"


@pytest.mark.parametrize(
    ("subtitle_format", "suffix", "payload"),
    [
        ("srt", ".srt", "not a timestamp"),
        ("webvtt", ".vtt", "not webvtt"),
        ("ass", ".ass", "[Script Info]\nmissing events"),
    ],
)
def test_invalid_sidecar_syntax_fails_without_artifact(
    tmp_path: Path,
    subtitle_format: str,
    suffix: str,
    payload: str,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / f"captions{suffix}"
    plan = build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_format=subtitle_format,
        mode="sidecar",
    )

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(probe=ProbeFixture(source)).deliver(
            plan,
            sidecar_payload=payload,
        )

    assert raised.value.code is SubtitleDeliveryErrorCode.SUBTITLE_INVALID
    assert not output.exists()


def test_karaoke_tags_require_matching_word_timing_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "captions.ass"
    plan = build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_format="ass",
        mode="sidecar",
    )
    executor = SubtitleDeliveryExecutor(probe=ProbeFixture(source))

    with pytest.raises(SubtitleDeliveryError) as missing:
        executor.deliver(plan, sidecar_payload=KARAOKE_ASS)
    assert (
        missing.value.code
        is SubtitleDeliveryErrorCode.KARAOKE_EVIDENCE_REQUIRED
    )

    payload = KARAOKE_ASS.encode()
    evidence = KaraokeTimingEvidence(
        source=KaraokeTimingSource.FORCED_ALIGNER,
        subtitle_sha256=hashlib.sha256(payload).hexdigest(),
        word_count=2,
    )
    receipt = executor.deliver(
        plan,
        sidecar_payload=payload,
        karaoke_timing_evidence=evidence,
    )

    assert receipt.karaoke_timing_detected is True
    assert receipt.karaoke_timing_verified is True
    assert receipt.to_dict()["karaokeTiming"]["synthesized"] is False


def test_mismatched_karaoke_evidence_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "captions.ass"
    plan = build_subtitle_output_plan(
        source_path=source,
        output_path=output,
        subtitle_format="ass",
        mode="sidecar",
    )
    evidence = KaraokeTimingEvidence(
        source="human-authored",
        subtitle_sha256="0" * 64,
        word_count=2,
    )

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(probe=ProbeFixture(source)).deliver(
            plan,
            sidecar_payload=KARAOKE_ASS,
            karaoke_timing_evidence=evidence,
        )

    assert (
        raised.value.code
        is SubtitleDeliveryErrorCode.KARAOKE_EVIDENCE_REQUIRED
    )
    assert not output.exists()


def test_soft_mux_mp4_selects_only_the_new_mov_text_stream(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"
    source.write_bytes(b"immutable-source")
    runner = RunnerFixture()
    probe = ProbeFixture(
        source,
        source_subtitles=("mov_text",),
        output_subtitles=("mov_text", "mov_text"),
        source_rotation=True,
        output_rotation=True,
    )

    receipt = SubtitleDeliveryExecutor(runner=runner, probe=probe).deliver(
        _soft_plan(source, output, subtitle),
        subtitle_language="zh-Hans",
        subtitle_title="Chinese captions",
        make_subtitle_default=True,
    )

    delivery_command = runner.commands[-1]
    assert delivery_command[0] == "ffmpeg"
    assert "-c:s:1" in delivery_command
    assert delivery_command[delivery_command.index("-c:s:1") + 1] == "mov_text"
    assert "language=zh-Hans" in delivery_command
    assert "title=Chinese captions" in delivery_command
    assert "-n" in delivery_command
    assert "-y" not in delivery_command
    assert str(output) not in delivery_command
    assert Path(delivery_command[-1]).parent == output.parent
    assert output.is_file()
    assert source.read_bytes() == b"immutable-source"
    assert len(probe.calls) == 3
    assert probe.calls[1] != output.resolve()
    assert probe.calls[2] == output.resolve()
    assert receipt.selected_subtitle_codec == "mov_text"
    assert receipt.qa.subtitle_stream_verified is True
    Draft202012Validator(_schema()).validate(receipt.to_dict())


@pytest.mark.parametrize(
    ("subtitle_format", "text", "expected_codec"),
    [
        ("srt", SRT_TEXT, "subrip"),
        ("webvtt", WEBVTT_TEXT, "webvtt"),
        ("ass", ASS_TEXT, "ass"),
    ],
)
def test_matroska_selects_format_preserving_soft_subtitle_codecs(
    tmp_path: Path,
    subtitle_format: str,
    text: str,
    expected_codec: str,
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(
        tmp_path / f"captions.{subtitle_format}",
        text,
    )
    output = tmp_path / f"delivered-{subtitle_format}.mkv"
    runner = RunnerFixture()
    probe = ProbeFixture(source, output_subtitles=(expected_codec,))

    receipt = SubtitleDeliveryExecutor(runner=runner, probe=probe).deliver(
        _soft_plan(
            source,
            output,
            subtitle,
            subtitle_format=subtitle_format,
        )
    )

    assert receipt.selected_subtitle_codec == expected_codec
    assert expected_codec in runner.commands[-1]
    assert output.exists()


def test_webm_converts_text_subtitle_to_webvtt_only_when_available(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.webm"
    runner = RunnerFixture()
    probe = ProbeFixture(
        source,
        source_audio=("opus",),
        source_video=("vp9",),
        output_audio=("opus",),
        output_video=("vp9",),
        output_subtitles=("webvtt",),
    )

    receipt = SubtitleDeliveryExecutor(runner=runner, probe=probe).deliver(
        _soft_plan(source, output, subtitle)
    )

    assert receipt.selected_subtitle_codec == "webvtt"


def test_unsupported_soft_mux_container_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.avi"

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=ProbeFixture(source),
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is SubtitleDeliveryErrorCode.CONTAINER_UNSUPPORTED
    assert not output.exists()


def test_explicit_unsafe_subtitle_codec_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.ass", ASS_TEXT)
    output = tmp_path / "delivered.mp4"

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=ProbeFixture(source),
        ).deliver(
            _soft_plan(
                source,
                output,
                subtitle,
                subtitle_format="ass",
                subtitle_codec="ass",
            )
        )

    assert raised.value.code is SubtitleDeliveryErrorCode.CODEC_UNSUPPORTED


def test_missing_local_encoder_fails_before_ffmpeg_delivery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"
    runner = RunnerFixture(encoders=("subrip",))

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=runner,
            probe=ProbeFixture(source),
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is SubtitleDeliveryErrorCode.CODEC_UNSUPPORTED
    assert len(runner.commands) == 3
    assert not output.exists()


def test_capability_listing_failure_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(capability_returncode=1),
            probe=ProbeFixture(source),
        ).deliver(_soft_plan(source, output, subtitle))

    assert (
        raised.value.code
        is SubtitleDeliveryErrorCode.CAPABILITY_PROBE_FAILED
    )
    assert not output.exists()


@pytest.mark.parametrize(
    ("returncode", "create_output", "expected"),
    [
        (1, True, SubtitleDeliveryErrorCode.PROCESS_FAILED),
        (0, False, SubtitleDeliveryErrorCode.OUTPUT_MISSING),
    ],
)
def test_ffmpeg_failure_or_missing_output_cleans_temporary_artifacts(
    tmp_path: Path,
    returncode: int,
    create_output: bool,
    expected: SubtitleDeliveryErrorCode,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(
                returncode=returncode,
                create_output=create_output,
            ),
            probe=ProbeFixture(source, output_subtitles=("mov_text",)),
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is expected
    assert not output.exists()
    assert not list(tmp_path.glob(".*.mts-*"))


def test_empty_ffmpeg_output_is_rejected_and_cleaned(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(output_bytes=b""),
            probe=ProbeFixture(source, output_subtitles=("mov_text",)),
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is SubtitleDeliveryErrorCode.OUTPUT_EMPTY
    assert not output.exists()
    assert not list(tmp_path.glob(".*.mts-*"))


@pytest.mark.parametrize(
    ("probe_changes", "expected"),
    [
        (
            {"output_duration_ms": 20_000},
            SubtitleDeliveryErrorCode.QA_FAILED,
        ),
        (
            {"output_audio": ()},
            SubtitleDeliveryErrorCode.QA_FAILED,
        ),
        (
            {"output_subtitles": ()},
            SubtitleDeliveryErrorCode.QA_FAILED,
        ),
        (
            {"output_chapters": 0},
            SubtitleDeliveryErrorCode.QA_FAILED,
        ),
    ],
)
def test_media_qa_failures_never_publish(
    tmp_path: Path,
    probe_changes: dict[str, Any],
    expected: SubtitleDeliveryErrorCode,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"
    options = {"output_subtitles": ("mov_text",), **probe_changes}

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=ProbeFixture(source, **options),
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is expected
    assert not output.exists()
    assert not list(tmp_path.glob(".*.mts-*"))


def test_post_publication_probe_failure_rolls_back_our_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"
    probe = ProbeFixture(
        source,
        output_subtitles=("mov_text",),
        output_sequence=[
            {},
            {"duration_ms": 30_000},
        ],
    )

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=probe,
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is SubtitleDeliveryErrorCode.QA_FAILED
    assert not output.exists()


def test_source_mutation_is_detected_and_published_output_is_rolled_back(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"

    def mutate_source(_: tuple[str, ...]) -> None:
        source.write_bytes(b"source-was-mutated")

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(on_delivery=mutate_source),
            probe=ProbeFixture(source, output_subtitles=("mov_text",)),
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is SubtitleDeliveryErrorCode.SOURCE_CHANGED
    assert not output.exists()


def test_concurrent_destination_creation_is_preserved_not_overwritten(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"

    def create_competing_output(_: tuple[str, ...]) -> None:
        output.write_bytes(b"other-process")

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(on_delivery=create_competing_output),
            probe=ProbeFixture(source, output_subtitles=("mov_text",)),
        ).deliver(_soft_plan(source, output, subtitle))

    assert raised.value.code is SubtitleDeliveryErrorCode.OUTPUT_EXISTS
    assert output.read_bytes() == b"other-process"
    assert not list(tmp_path.glob(".*.mts-*"))


def test_burn_in_requires_explicit_strategy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.ass", ASS_TEXT)
    output = tmp_path / "delivered.mp4"

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=ProbeFixture(source),
        ).deliver(_burn_plan(source, output, subtitle))

    assert (
        raised.value.code
        is SubtitleDeliveryErrorCode.VIDEO_STRATEGY_REQUIRED
    )
    assert not output.exists()


def test_burn_in_command_preserves_audio_metadata_and_all_video_streams(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source clip.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions,zh.ass", ASS_TEXT)
    output = tmp_path / "delivered.mp4"
    runner = RunnerFixture()
    probe = ProbeFixture(
        source,
        source_audio=("aac", "ac3"),
        source_video=("h264", "mjpeg"),
        output_audio=("aac", "ac3"),
        output_video=("h264", "mjpeg"),
    )

    receipt = SubtitleDeliveryExecutor(runner=runner, probe=probe).deliver(
        _burn_plan(source, output, subtitle),
        burn_in_strategy=BurnInVideoStrategy.H264_HIGH_QUALITY,
    )

    command = runner.commands[-1]
    assert command.count("-map") == 4
    assert "0:0" in command and "0:1" in command
    assert "0:2" in command and "0:3" in command
    assert "-filter:v:0" in command
    filter_value = command[command.index("-filter:v:0") + 1]
    assert "subtitles=filename=" in filter_value
    assert r"\," in filter_value
    assert command[command.index("-c:v:0") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "copy"
    assert command[command.index("-map_metadata") + 1] == "0"
    assert command[command.index("-map_chapters") + 1] == "0"
    assert "-n" in command and "-y" not in command
    assert receipt.selected_video_strategy is BurnInVideoStrategy.H264_HIGH_QUALITY
    assert receipt.qa.subtitle_stream_verified is False
    Draft202012Validator(_schema()).validate(receipt.to_dict())


def test_incompatible_burn_in_strategy_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.ass", ASS_TEXT)
    output = tmp_path / "delivered.webm"

    with pytest.raises(SubtitleDeliveryError) as raised:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=ProbeFixture(
                source,
                source_audio=("opus",),
                source_video=("vp9",),
            ),
        ).deliver(
            _burn_plan(source, output, subtitle),
            burn_in_strategy="h264-high-quality",
        )

    assert (
        raised.value.code
        is SubtitleDeliveryErrorCode.VIDEO_STRATEGY_UNSUPPORTED
    )


def test_burn_in_rejects_audio_only_and_hdr_sources(tmp_path: Path) -> None:
    subtitle = _write_subtitle(tmp_path / "captions.ass", ASS_TEXT)

    audio_source = tmp_path / "audio.m4a"
    audio_source.write_bytes(b"audio")
    with pytest.raises(SubtitleDeliveryError) as audio_error:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=ProbeFixture(
                audio_source,
                source_video=(),
                output_video=(),
            ),
        ).deliver(
            _burn_plan(audio_source, tmp_path / "audio.mp4", subtitle),
            burn_in_strategy="h264-high-quality",
        )
    assert (
        audio_error.value.code
        is SubtitleDeliveryErrorCode.VIDEO_STRATEGY_UNSUPPORTED
    )

    hdr_source = tmp_path / "hdr.mp4"
    hdr_source.write_bytes(b"hdr")
    with pytest.raises(SubtitleDeliveryError) as hdr_error:
        SubtitleDeliveryExecutor(
            runner=RunnerFixture(),
            probe=ProbeFixture(hdr_source, source_hdr=True),
        ).deliver(
            _burn_plan(hdr_source, tmp_path / "hdr-output.mp4", subtitle),
            burn_in_strategy="h265-high-quality",
        )
    assert (
        hdr_error.value.code
        is SubtitleDeliveryErrorCode.HDR_BURN_IN_UNSUPPORTED
    )


def test_burn_in_missing_filter_or_encoder_fails_before_delivery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.ass", ASS_TEXT)

    for runner in (
        RunnerFixture(encoders=("mov_text",)),
        RunnerFixture(filters=()),
    ):
        output = tmp_path / f"delivered-{len(runner.commands)}.mp4"
        with pytest.raises(SubtitleDeliveryError) as raised:
            SubtitleDeliveryExecutor(
                runner=runner,
                probe=ProbeFixture(source),
            ).deliver(
                _burn_plan(source, output, subtitle),
                burn_in_strategy="h264-high-quality",
            )
        assert raised.value.code is SubtitleDeliveryErrorCode.CODEC_UNSUPPORTED
        assert not output.exists()


def test_receipt_schema_rejects_shell_or_source_integrity_regressions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    subtitle = _write_subtitle(tmp_path / "captions.srt", SRT_TEXT)
    output = tmp_path / "delivered.mp4"
    receipt = SubtitleDeliveryExecutor(
        runner=RunnerFixture(),
        probe=ProbeFixture(source, output_subtitles=("mov_text",)),
    ).deliver(_soft_plan(source, output, subtitle))
    validator = Draft202012Validator(_schema())
    payload = receipt.to_dict()
    validator.validate(payload)

    shell_payload = json.loads(json.dumps(payload))
    shell_payload["command"]["shell"] = True
    assert list(validator.iter_errors(shell_payload))

    integrity_payload = json.loads(json.dumps(payload))
    integrity_payload["sourceIntegrity"]["unchanged"] = False
    assert list(validator.iter_errors(integrity_payload))


def test_bounded_runner_invokes_subprocess_with_shell_false() -> None:
    class FinishedProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

    with patch(
        "backend.subtitle_delivery.subprocess.Popen",
        return_value=FinishedProcess(),
    ) as popen:
        result = BoundedDeliveryRunner().run(
            ("ffmpeg", "-version"),
            limits=DeliveryProcessLimits(
                timeout_seconds=1,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
            ),
        )

    assert result.returncode == 0
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.kwargs["stdin"] is not None


def test_bounded_runner_enforces_timeout_without_real_ffmpeg() -> None:
    class HangingProcess:
        returncode: int | None = None
        stopped = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.stopped = True
            self.returncode = -15

        def wait(self, timeout: float) -> int:
            del timeout
            return int(self.returncode or 0)

    process = HangingProcess()
    with (
        patch(
            "backend.subtitle_delivery.subprocess.Popen",
            return_value=process,
        ),
        patch(
            "backend.subtitle_delivery.time.monotonic",
            side_effect=(0.0, 1.0),
        ),
        pytest.raises(SubtitleDeliveryError) as raised,
    ):
        BoundedDeliveryRunner().run(
            ("ffmpeg", "-version"),
            limits=DeliveryProcessLimits(
                timeout_seconds=0.05,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
            ),
        )

    assert raised.value.code is SubtitleDeliveryErrorCode.PROCESS_TIMEOUT
    assert process.stopped is True


def test_bounded_runner_enforces_diagnostic_output_limit_without_ffmpeg() -> None:
    class OverflowProcess:
        returncode: int | None = None
        stopped = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args
            kwargs["stdout"].write(b"x" * 2048)
            kwargs["stdout"].flush()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.stopped = True
            self.returncode = -15

        def wait(self, timeout: float) -> int:
            del timeout
            return int(self.returncode or 0)

    created: list[OverflowProcess] = []

    def create_process(*args: Any, **kwargs: Any) -> OverflowProcess:
        process = OverflowProcess(*args, **kwargs)
        created.append(process)
        return process

    with (
        patch(
            "backend.subtitle_delivery.subprocess.Popen",
            side_effect=create_process,
        ),
        pytest.raises(SubtitleDeliveryError) as raised,
    ):
        BoundedDeliveryRunner().run(
            ("ffmpeg", "-version"),
            limits=DeliveryProcessLimits(
                timeout_seconds=1,
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
                poll_interval_seconds=0.005,
            ),
        )

    assert raised.value.code is SubtitleDeliveryErrorCode.PROCESS_OUTPUT_LIMIT
    assert created[0].stopped is True


def test_delivery_process_limits_are_hard_bounded() -> None:
    with pytest.raises(ValueError):
        DeliveryProcessLimits(timeout_seconds=0)
    with pytest.raises(ValueError):
        DeliveryProcessLimits(timeout_seconds=86_401)
    with pytest.raises(ValueError):
        DeliveryProcessLimits(max_stdout_bytes=100)

    policy = SubtitleDeliveryPolicy()
    assert policy.process_limits.timeout_seconds == 14_400
    assert policy.process_limits.max_stderr_bytes == 8 * 1024 * 1024
