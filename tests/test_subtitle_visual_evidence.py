from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock, patch

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from backend.subtitle_visual_evidence import (
    BoundedEvidenceRunner,
    ComponentDescriptor,
    ContrastObservation,
    CueFrameObservation,
    EvidenceProcessLimits,
    EvidenceProcessResult,
    FontEvidenceObservation,
    FrameObservation,
    PillowFrameAnalyzer,
    SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION,
    SubtitleVisualEvidenceCollector,
    SubtitleVisualEvidenceError,
    SubtitleVisualEvidenceErrorCode,
    SubtitleVisualEvidencePolicy,
    VerifiedFontClaim,
    canonical_json,
    default_subtitle_visual_evidence_sampling,
    deterministic_sha256,
    verify_evidence_artifact_hash,
)
from backend.subtitle_visual_qa import (
    default_subtitle_visual_qa_policy,
    evaluate_subtitle_visual_qa,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "contracts" / "subtitle-render-evidence.schema.json"
QA_SCHEMA_PATH = ROOT / "contracts" / "subtitle-visual-qa.schema.json"
SHA_A = "a" * 64


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    qa_schema = json.loads(QA_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    registry = Registry().with_resource(
        qa_schema["$id"],
        Resource.from_contents(qa_schema),
    )
    return Draft202012Validator(schema, registry=registry)


def _files(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source.mov"
    rendered = tmp_path / "rendered.mov"
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    source.write_bytes(b"immutable-source-media")
    rendered.write_bytes(b"immutable-rendered-media")
    ffmpeg.write_bytes(b"fake-ffmpeg-binary")
    ffprobe.write_bytes(b"fake-ffprobe-binary")
    return {
        "source": source,
        "rendered": rendered,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
    }


def _request(
    files: dict[str, Path],
    *,
    karaoke_mode: str = "none",
    word_timing: dict[str, Any] | None = None,
    fractions: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "subtitle-render-evidence-request",
        "schemaVersion": "1.0.0",
        "collectionId": "visual-evidence-fixture",
        "sourceMediaPath": str(files["source"].resolve()),
        "renderedMediaPath": str(files["rendered"].resolve()),
        "renderArtifact": {
            "renderer": "ffmpeg-libass",
            "rendererVersion": "ffmpeg fixture / libass fixture",
            "renderConfigurationSha256": SHA_A,
        },
        "policy": default_subtitle_visual_qa_policy(),
        "sampling": {
            "strategy": "cue-interior-contrast-candidates-v1",
            "candidateFractions": fractions or [0.5],
        },
        "speakers": [
            {"speakerId": "speaker-a", "color": "#00A8E8"},
        ],
        "cues": [
            {
                "cueId": "cue-1",
                "startMs": 1000,
                "endMs": 4000,
                "text": "Hello world",
                "speakerId": "speaker-a",
                "styleId": "youtube-clean",
                "renderedLines": ["Hello world"],
                "karaokeMode": karaoke_mode,
                "wordTimingEvidence": word_timing,
                "bounds": {
                    "x": 300,
                    "y": 790,
                    "width": 1320,
                    "height": 180,
                },
                "requestedFontFamilies": [
                    "Noto Sans",
                    "Arial",
                ],
            }
        ],
    }


class FakeRunner:
    def __init__(
        self,
        *,
        on_run: Callable[[tuple[str, ...]], None] | None = None,
        oversized: bool = False,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.cwds: list[Path] = []
        self.on_run = on_run
        self.oversized = oversized

    def run(
        self,
        command: list[str] | tuple[str, ...],
        *,
        limits: EvidenceProcessLimits,
        cwd: Path,
    ) -> EvidenceProcessResult:
        del limits
        argv = tuple(command)
        self.commands.append(argv)
        self.cwds.append(Path(cwd))
        if self.on_run is not None:
            self.on_run(argv)
        if self.oversized:
            return EvidenceProcessResult(
                returncode=0,
                stdout=b"x" * 2_000,
                stderr=b"",
            )
        if "-version" in argv:
            name = Path(argv[0]).stem
            return EvidenceProcessResult(
                returncode=0,
                stdout=f"{name} version fixture\n".encode(),
                stderr=b"",
            )
        if "-show_frames" in argv:
            interval = argv[argv.index("-read_intervals") + 1]
            seconds = interval.split("%", 1)[0]
            payload = {
                "frames": [
                    {
                        "best_effort_timestamp_time": seconds,
                        "width": 1920,
                        "height": 1080,
                    }
                ]
            }
            return EvidenceProcessResult(
                returncode=0,
                stdout=json.dumps(payload).encode(),
                stderr=b"",
            )
        if "-show_entries" in argv:
            payload = {
                "streams": [{"width": 1920, "height": 1080}],
                "format": {"duration": "10.000"},
            }
            return EvidenceProcessResult(
                returncode=0,
                stdout=json.dumps(payload).encode(),
                stderr=b"",
            )
        if "-frames:v" in argv:
            output = Path(argv[-1])
            input_path = Path(argv[argv.index("-i") + 1])
            timestamp = argv[argv.index("-ss") + 1]
            output.write_bytes(
                b"PNG-FIXTURE\x00"
                + input_path.name.encode()
                + b"\x00"
                + timestamp.encode()
            )
            return EvidenceProcessResult(0, b"", b"")
        raise AssertionError(f"unexpected command: {argv!r}")


class FakeAnalyzer:
    descriptor = ComponentDescriptor(
        name="fake-render-analyzer",
        version="1.0.0",
        configuration_sha256=deterministic_sha256(
            {"fixture": "frame-analysis"}
        ),
    )

    def __init__(self, *, wrong_cue: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.wrong_cue = wrong_cue

    def analyze(
        self,
        *,
        rendered_frame_path: Path,
        source_frame_path: Path,
        frame_id: str,
        timestamp_ms: int,
        width_px: int,
        height_px: int,
        cues: list[dict[str, Any]],
        contrast_policy: dict[str, Any],
    ) -> FrameObservation:
        self.calls.append(
            {
                "rendered": rendered_frame_path,
                "source": source_frame_path,
                "frameId": frame_id,
                "timestampMs": timestamp_ms,
                "cues": [cue["cueId"] for cue in cues],
                "contrastPolicy": contrast_policy,
            }
        )
        instances = []
        for cue in cues:
            cue_id = "wrong-cue" if self.wrong_cue else cue["cueId"]
            instances.append(
                CueFrameObservation(
                    cue_id=cue_id,
                    bounds=cue["bounds"],
                    ink_bounds={
                        "x": 340,
                        "y": 820,
                        "width": 1240,
                        "height": 110,
                    },
                    clipped_pixel_count=0,
                    edge_touching_pixel_count=0,
                    overflow_detected=False,
                    contrast_samples=(
                        ContrastObservation(
                            background_class="dark",
                            foreground_rgb="#FFFFFF",
                            background_rgb="#000000",
                            foreground_pixel_count=96,
                            background_pixel_count=128,
                        ),
                        ContrastObservation(
                            background_class="light",
                            foreground_rgb="#000000",
                            background_rgb="#FFFFFF",
                            foreground_pixel_count=96,
                            background_pixel_count=128,
                        ),
                    ),
                )
            )
        return FrameObservation(
            width_px=width_px,
            height_px=height_px,
            instances=tuple(instances),
        )


class VerifiedFontProvider:
    descriptor = ComponentDescriptor(
        name="fixture-font-provider",
        version="1.0.0",
        configuration_sha256=deterministic_sha256(
            {"fixture": "real-artifact-paths"}
        ),
    )

    def __init__(
        self,
        *,
        evidence_path: Path,
        font_path: Path,
        missing_path: bool = False,
    ) -> None:
        self.evidence_path = evidence_path
        self.font_path = font_path
        self.missing_path = missing_path

    def collect(
        self,
        *,
        cue: dict[str, Any],
        frame_id: str,
        timestamp_ms: int,
        rendered_frame_path: Path,
        source_frame_path: Path,
    ) -> FontEvidenceObservation:
        del frame_id, timestamp_ms, rendered_frame_path, source_frame_path
        evidence_path = (
            self.evidence_path.with_name("missing-evidence.bin")
            if self.missing_path
            else self.evidence_path
        )
        expected = sum(
            1 for character in cue["text"] if not character.isspace()
        )
        claim = VerifiedFontClaim(
            status="verified-installed",
            verification_method="fontconfig-scan",
            evidence_artifact_path=evidence_path,
            font_artifact_path=self.font_path,
        )
        return FontEvidenceObservation(
            resolved_family="Noto Sans",
            resolution_verified=True,
            glyph_coverage_verified=True,
            verification_method="font-cmap-and-shaping",
            evidence_artifact_path=evidence_path,
            covered_renderable_code_points=expected,
            missing_code_points=(),
            tofu_glyph_count=0,
            installation=claim,
            embedding=None,
        )


def _collector(
    files: dict[str, Path],
    *,
    runner: FakeRunner | None = None,
    analyzer: FakeAnalyzer | None = None,
    font_provider: Any = None,
    policy: SubtitleVisualEvidencePolicy | None = None,
) -> SubtitleVisualEvidenceCollector:
    return SubtitleVisualEvidenceCollector(
        ffmpeg_path=files["ffmpeg"],
        ffprobe_path=files["ffprobe"],
        runner=runner or FakeRunner(),
        analyzer=analyzer or FakeAnalyzer(),
        font_evidence_provider=font_provider,
        policy=policy,
    )


def test_schema_is_draft_2020_12_and_versioned(
    validator: Draft202012Validator,
) -> None:
    assert validator.schema["$schema"].endswith("draft/2020-12/schema")
    assert SUBTITLE_RENDER_EVIDENCE_SCHEMA_VERSION == "1.0.0"
    assert validator.schema["$id"].endswith("/1.0.0")


def test_request_and_result_are_schema_valid_and_qa_aligned(
    tmp_path: Path,
    validator: Draft202012Validator,
) -> None:
    files = _files(tmp_path)
    request = _request(files)
    assert not list(validator.iter_errors(request))

    result = _collector(files).collect(request)
    payload = result.to_dict()
    assert not list(validator.iter_errors(payload))
    assert payload["qaRequest"]["kind"] == "subtitle-visual-qa-request"
    assert payload["qaRequest"]["frames"][0]["instances"][0][
        "cueId"
    ] == "cue-1"
    qa_result = evaluate_subtitle_visual_qa(result.qa_request).to_dict()
    assert qa_result["metrics"]["framesEvaluated"] == 1
    assert qa_result["fontTruth"]["evidenceInstances"] == 0


def test_result_is_canonical_and_every_hash_binding_verifies(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    payload = _collector(files).collect(_request(files)).to_dict()

    assert verify_evidence_artifact_hash(payload)
    assert payload["qaRequestSha256"] == deterministic_sha256(
        payload["qaRequest"]
    )
    selection = dict(payload["selection"])
    selection_hash = selection.pop("selectionArtifactSha256")
    assert selection_hash == deterministic_sha256(selection)
    assert canonical_json(payload) == canonical_json(
        {key: payload[key] for key in reversed(tuple(payload))}
    )


def test_same_inputs_produce_identical_evidence(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    request = _request(files, fractions=[0.25, 0.5, 0.75])
    first = _collector(files).collect(request).to_dict()
    second = _collector(files).collect(request).to_dict()
    assert first == second
    assert [item["fraction"] for item in first["selection"]["candidates"]] == [
        0.25,
        0.5,
        0.75,
    ]
    assert first["qaRequest"]["sampling"]["expectedFrameIds"] == [
        item["frameId"] for item in first["selection"]["frames"]
    ]


def test_explicit_tools_argv_only_and_same_directory_cleanup(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    runner = FakeRunner()
    source_before = files["source"].read_bytes()
    source_stat = files["source"].stat()

    _collector(files, runner=runner).collect(_request(files))

    assert runner.commands
    assert all(isinstance(command, tuple) for command in runner.commands)
    assert {
        command[0] for command in runner.commands
    } == {str(files["ffmpeg"].resolve()), str(files["ffprobe"].resolve())}
    assert all(cwd.parent == files["rendered"].parent for cwd in runner.cwds)
    assert all(not cwd.exists() for cwd in runner.cwds)
    assert files["source"].read_bytes() == source_before
    assert files["source"].stat().st_mtime_ns == source_stat.st_mtime_ns
    assert not list(
        files["rendered"].parent.glob(
            ".mts-subtitle-visual-evidence-*"
        )
    )


def test_ffmpeg_outputs_are_temporary_and_never_alias_media(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    runner = FakeRunner()
    _collector(files, runner=runner).collect(_request(files))
    extraction_commands = [
        command for command in runner.commands if "-frames:v" in command
    ]
    assert len(extraction_commands) == 2
    for command in extraction_commands:
        output = Path(command[-1])
        input_path = Path(command[command.index("-i") + 1])
        assert output != input_path
        assert output not in {files["source"], files["rendered"]}
        assert "-nostdin" in command
        assert "-an" in command
        assert "-sn" in command
        assert "-dn" in command


def test_injected_runner_output_is_bounded(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    policy = SubtitleVisualEvidencePolicy(
        process_limits=EvidenceProcessLimits(
            timeout_seconds=1,
            max_stdout_bytes=1_024,
            max_stderr_bytes=1_024,
        )
    )
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(
            files,
            runner=FakeRunner(oversized=True),
            policy=policy,
        ).collect(_request(files))
    assert (
        captured.value.code
        is SubtitleVisualEvidenceErrorCode.PROCESS_OUTPUT_LIMIT
    )


def test_source_mutation_during_collection_fails_closed(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    mutated = False

    def mutate_on_extraction(argv: tuple[str, ...]) -> None:
        nonlocal mutated
        if "-frames:v" in argv and not mutated:
            files["source"].write_bytes(b"mutated-source")
            mutated = True

    runner = FakeRunner(on_run=mutate_on_extraction)
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(files, runner=runner).collect(_request(files))
    assert captured.value.code is SubtitleVisualEvidenceErrorCode.SOURCE_CHANGED
    assert not list(
        files["rendered"].parent.glob(
            ".mts-subtitle-visual-evidence-*"
        )
    )


def test_frame_analyzer_must_cover_every_active_cue(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(
            files,
            analyzer=FakeAnalyzer(wrong_cue=True),
        ).collect(_request(files))
    assert (
        captured.value.code
        is SubtitleVisualEvidenceErrorCode.ANALYSIS_INCOMPLETE
    )


def test_speaker_style_cue_bindings_are_hash_bound(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    payload = _collector(files).collect(_request(files)).to_dict()
    binding = payload["bindings"][0]
    assert binding["cueId"] == "cue-1"
    assert binding["speakerId"] == "speaker-a"
    assert binding["styleId"] == "youtube-clean"
    assert len(binding["bindingArtifactSha256"]) == 64
    instance = payload["qaRequest"]["frames"][0]["instances"][0]
    assert instance["bounds"] == _request(files)["cues"][0]["bounds"]
    assert instance["inkBounds"]["width"] == 1240
    assert {
        item["backgroundClass"] for item in instance["contrastSamples"]
    } == {"dark", "light"}
    assert all(
        item["sampledFromRenderedFrame"]
        for item in instance["contrastSamples"]
    )


def test_no_font_provider_makes_no_installation_or_embedding_claim(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    payload = _collector(files).collect(_request(files)).to_dict()
    evidence = payload["qaRequest"]["frames"][0]["instances"][0][
        "fontEvidence"
    ]
    assert evidence is None
    assert "verified-installed" not in json.dumps(payload)
    assert "verified-embedded" not in json.dumps(payload)


def test_positive_font_claim_hashes_real_artifacts(
    tmp_path: Path,
    validator: Draft202012Validator,
) -> None:
    files = _files(tmp_path)
    evidence_path = tmp_path / "font-evidence.json"
    font_path = tmp_path / "font.ttf"
    evidence_path.write_bytes(b'{"resolved":"Noto Sans"}')
    font_path.write_bytes(b"fixture-font-bytes")
    provider = VerifiedFontProvider(
        evidence_path=evidence_path,
        font_path=font_path,
    )

    payload = _collector(
        files, font_provider=provider
    ).collect(_request(files)).to_dict()
    assert not list(validator.iter_errors(payload))
    evidence = payload["qaRequest"]["frames"][0]["instances"][0][
        "fontEvidence"
    ]
    assert evidence["evidenceArtifactSha256"] == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    assert evidence["installation"] == {
        "status": "verified-installed",
        "verificationMethod": "fontconfig-scan",
        "evidenceArtifactSha256": hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest(),
        "fontArtifactSha256": hashlib.sha256(
            font_path.read_bytes()
        ).hexdigest(),
    }
    assert evidence["embedding"] == {
        "status": "not-asserted",
        "verificationMethod": "not-provided",
        "evidenceArtifactSha256": None,
        "fontArtifactSha256": None,
    }


def test_positive_font_claim_without_real_artifact_fails_closed(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    evidence_path = tmp_path / "font-evidence.json"
    font_path = tmp_path / "font.ttf"
    evidence_path.write_bytes(b"evidence")
    font_path.write_bytes(b"font")
    provider = VerifiedFontProvider(
        evidence_path=evidence_path,
        font_path=font_path,
        missing_path=True,
    )
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(
            files, font_provider=provider
        ).collect(_request(files))
    assert (
        captured.value.code
        is SubtitleVisualEvidenceErrorCode.INPUT_MISSING
    )


def test_word_timing_is_passed_through_without_synthesis(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    text = "Hello world"
    timing = {
        "source": "forced-aligner",
        "verified": True,
        "evidenceArtifactSha256": "b" * 64,
        "cueTextSha256": hashlib.sha256(text.encode()).hexdigest(),
        "words": [
            {"text": "Hello", "startMs": 1000, "endMs": 2200},
            {"text": "world", "startMs": 2200, "endMs": 3900},
        ],
    }
    request = _request(
        files,
        karaoke_mode="word-progress",
        word_timing=timing,
    )
    payload = _collector(files).collect(request).to_dict()
    assert payload["qaRequest"]["cues"][0]["wordTimingEvidence"] == timing

    no_timing = _collector(files).collect(
        _request(files, karaoke_mode="word-progress")
    ).to_dict()
    assert (
        no_timing["qaRequest"]["cues"][0]["wordTimingEvidence"] is None
    )


@pytest.mark.parametrize(
    "source",
    ["segment-interpolation", "synthetic-even-split"],
)
def test_synthetic_word_timing_sources_are_rejected(
    tmp_path: Path,
    source: str,
) -> None:
    files = _files(tmp_path)
    text = "Hello world"
    timing = {
        "source": source,
        "verified": True,
        "evidenceArtifactSha256": "b" * 64,
        "cueTextSha256": hashlib.sha256(text.encode()).hexdigest(),
        "words": [
            {"text": "Hello", "startMs": 1000, "endMs": 2200},
            {"text": "world", "startMs": 2200, "endMs": 3900},
        ],
    }
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(files).collect(
            _request(
                files,
                karaoke_mode="word-progress",
                word_timing=timing,
            )
        )
    assert captured.value.code is SubtitleVisualEvidenceErrorCode.INVALID_REQUEST


def test_missing_pillow_fails_closed_before_reading_images(
    tmp_path: Path,
) -> None:
    def unavailable() -> Any:
        raise SubtitleVisualEvidenceError(
            SubtitleVisualEvidenceErrorCode.ANALYZER_UNAVAILABLE,
            "Pillow unavailable",
        )

    analyzer = PillowFrameAnalyzer(image_module_loader=unavailable)
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        analyzer.analyze(
            rendered_frame_path=tmp_path / "rendered.png",
            source_frame_path=tmp_path / "source.png",
            frame_id="frame-1",
            timestamp_ms=1000,
            width_px=100,
            height_px=100,
            cues=[],
            contrast_policy={
                "darkMaximumLuminance": 0.35,
                "lightMinimumLuminance": 0.65,
            },
        )
    assert (
        captured.value.code
        is SubtitleVisualEvidenceErrorCode.ANALYZER_UNAVAILABLE
    )


def test_pillow_analyzer_collects_real_visible_ink_and_dark_light_pixels(
    tmp_path: Path,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    source = image_module.new("RGB", (100, 100), "#000000")
    for x_coord in range(50, 100):
        for y_coord in range(100):
            source.putpixel((x_coord, y_coord), (255, 255, 255))
    rendered = source.copy()
    for x_coord in range(20, 40):
        for y_coord in range(40, 60):
            rendered.putpixel((x_coord, y_coord), (255, 255, 255))
    for x_coord in range(60, 80):
        for y_coord in range(40, 60):
            rendered.putpixel((x_coord, y_coord), (0, 0, 0))
    source_path = tmp_path / "source.png"
    rendered_path = tmp_path / "rendered.png"
    source.save(source_path)
    rendered.save(rendered_path)

    observation = PillowFrameAnalyzer().analyze(
        rendered_frame_path=rendered_path,
        source_frame_path=source_path,
        frame_id="frame-real-pixels",
        timestamp_ms=1000,
        width_px=100,
        height_px=100,
        cues=[
            {
                "cueId": "cue-real",
                "bounds": {
                    "x": 10,
                    "y": 30,
                    "width": 80,
                    "height": 40,
                },
            }
        ],
        contrast_policy={
            "darkMaximumLuminance": 0.35,
            "lightMinimumLuminance": 0.65,
        },
    )
    instance = observation.instances[0]
    assert instance.ink_bounds == {
        "x": 20,
        "y": 40,
        "width": 60,
        "height": 20,
    }
    assert instance.clipped_pixel_count == 0
    assert not instance.overflow_detected
    assert {
        sample.background_class for sample in instance.contrast_samples
    } == {"dark", "light"}
    assert all(
        sample.foreground_pixel_count == 400
        for sample in instance.contrast_samples
    )


def test_bounded_runner_invokes_subprocess_with_shell_false_and_argv(
    tmp_path: Path,
) -> None:
    process = Mock()
    process.poll.return_value = 0
    process.returncode = 0
    command = ("tool.exe", "argument with spaces", "--flag")
    with patch("backend.subtitle_visual_evidence.subprocess.Popen") as popen:
        popen.return_value = process
        result = BoundedEvidenceRunner().run(
            command,
            limits=EvidenceProcessLimits(
                timeout_seconds=1,
                max_stdout_bytes=1_024,
                max_stderr_bytes=1_024,
            ),
            cwd=tmp_path,
        )
    assert result.returncode == 0
    assert popen.call_args.args[0] == command
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.kwargs["stdin"] is not None
    assert popen.call_args.kwargs["cwd"] == str(tmp_path.resolve())


@pytest.mark.parametrize(
    "field,value",
    [
        ("sourceMediaPath", "https://example.invalid/source.mov"),
        ("renderedMediaPath", "relative.mov"),
    ],
)
def test_nonlocal_or_relative_media_paths_are_rejected(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    files = _files(tmp_path)
    request = _request(files)
    request[field] = value
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(files).collect(request)
    assert captured.value.code is SubtitleVisualEvidenceErrorCode.INVALID_PATH


def test_source_and_rendered_alias_is_rejected(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    request = _request(files)
    request["renderedMediaPath"] = request["sourceMediaPath"]
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(files).collect(request)
    assert captured.value.code is SubtitleVisualEvidenceErrorCode.PATH_ALIAS


def test_sampling_fractions_must_be_unique_and_ascending(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    request = _request(files)
    request["sampling"]["candidateFractions"] = [0.75, 0.25, 0.25]
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        _collector(files).collect(request)
    assert captured.value.code is SubtitleVisualEvidenceErrorCode.INVALID_REQUEST


def test_default_sampling_is_detached() -> None:
    first = default_subtitle_visual_evidence_sampling()
    first["candidateFractions"].append(0.9)
    second = default_subtitle_visual_evidence_sampling()
    assert second == {
        "strategy": "cue-interior-contrast-candidates-v1",
        "candidateFractions": [0.25, 0.5, 0.75],
    }


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(SubtitleVisualEvidenceError) as captured:
        canonical_json({"bad": float("nan")})
    assert captured.value.code is SubtitleVisualEvidenceErrorCode.INVALID_REQUEST


def test_collection_does_not_require_real_ffmpeg(
    tmp_path: Path,
) -> None:
    files = _files(tmp_path)
    for tool in (files["ffmpeg"], files["ffprobe"]):
        assert not os.access(tool, os.X_OK) or os.name == "nt"
    payload = _collector(files).collect(_request(files)).to_dict()
    assert payload["tools"]["ffmpeg"]["path"] == str(
        files["ffmpeg"].resolve()
    )
    assert payload["tools"]["ffprobe"]["path"] == str(
        files["ffprobe"].resolve()
    )
