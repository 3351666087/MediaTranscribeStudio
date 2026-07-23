from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from backend.subtitle_visual_qa import (
    SUBTITLE_VISUAL_QA_SCHEMA_VERSION,
    SubtitleVisualQAInputError,
    contrast_ratio,
    default_subtitle_visual_qa_policy,
    delta_e_2000,
    deterministic_sha256,
    evaluate_subtitle_visual_qa,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "subtitle-visual-qa.schema.json"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _font_evidence(
    text: str,
    *,
    installation_status: str = "not-asserted",
    embedding_status: str = "not-asserted",
) -> dict[str, Any]:
    expected = sum(1 for character in text if not character.isspace())

    def claim(status: str) -> dict[str, Any]:
        if status == "not-asserted":
            return {
                "status": status,
                "verificationMethod": "not-provided",
                "evidenceArtifactSha256": None,
                "fontArtifactSha256": None,
            }
        return {
            "status": status,
            "verificationMethod": "container-font-inspection",
            "evidenceArtifactSha256": SHA_C,
            "fontArtifactSha256": SHA_D,
        }

    return {
        "requestedFamilies": ["Noto Sans CJK SC", "Noto Sans"],
        "resolvedFamily": "Noto Sans CJK SC",
        "resolutionVerified": True,
        "glyphCoverageVerified": True,
        "verificationMethod": "libass-render-report",
        "evidenceArtifactSha256": SHA_B,
        "expectedRenderableCodePoints": expected,
        "coveredRenderableCodePoints": expected,
        "missingCodePoints": [],
        "tofuGlyphCount": 0,
        "installation": claim(installation_status),
        "embedding": claim(embedding_status),
    }


def _samples(prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "sampleId": f"{prefix}-dark",
            "backgroundClass": "dark",
            "foregroundRgb": "#FFFFFF",
            "backgroundRgb": "#000000",
            "foregroundPixelCount": 96,
            "backgroundPixelCount": 128,
            "sampledFromRenderedFrame": True,
            "sampleArtifactSha256": SHA_C,
        },
        {
            "sampleId": f"{prefix}-light",
            "backgroundClass": "light",
            "foregroundRgb": "#000000",
            "backgroundRgb": "#FFFFFF",
            "foregroundPixelCount": 96,
            "backgroundPixelCount": 128,
            "sampledFromRenderedFrame": True,
            "sampleArtifactSha256": SHA_D,
        },
    ]


def _word_evidence(
    text: str,
    words: list[dict[str, Any]],
    *,
    source: str = "forced-aligner",
    verified: bool = True,
) -> dict[str, Any]:
    return {
        "source": source,
        "verified": verified,
        "evidenceArtifactSha256": SHA_D,
        "cueTextSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "words": words,
    }


def _request() -> dict[str, Any]:
    first_text = "你好世界"
    second_text = "Ready now"
    return {
        "kind": "subtitle-visual-qa-request",
        "schemaVersion": "1.0.0",
        "analysisId": "visual-qa-pass",
        "renderArtifact": {
            "artifactSha256": SHA_A,
            "renderer": "ffmpeg-libass",
            "rendererVersion": "ffmpeg 8.0 / libass 0.17",
            "renderConfigurationSha256": SHA_B,
        },
        "policy": default_subtitle_visual_qa_policy(),
        "sampling": {
            "strategy": "cue-midpoint-plus-scene-extremes-v1",
            "selectionArtifactSha256": SHA_C,
            "expectedFrameIds": ["frame-1", "frame-2"],
        },
        "speakers": [
            {"speakerId": "speaker-a", "color": "#00A8E8"},
            {"speakerId": "speaker-b", "color": "#FF2D95"},
        ],
        "cues": [
            {
                "cueId": "cue-1",
                "startMs": 1000,
                "endMs": 4000,
                "text": first_text,
                "speakerId": "speaker-a",
                "styleId": "speaker-color",
                "renderedLines": ["你好", "世界"],
                "karaokeMode": "word-progress",
                "wordTimingEvidence": _word_evidence(
                    first_text,
                    [
                        {"text": "你好", "startMs": 1000, "endMs": 2300},
                        {"text": "世界", "startMs": 2300, "endMs": 3900},
                    ],
                ),
            },
            {
                "cueId": "cue-2",
                "startMs": 5000,
                "endMs": 8000,
                "text": second_text,
                "speakerId": "speaker-b",
                "styleId": "speaker-color",
                "renderedLines": ["Ready", "now"],
                "karaokeMode": "none",
                "wordTimingEvidence": None,
            },
        ],
        "frames": [
            {
                "frameId": "frame-1",
                "timestampMs": 2500,
                "widthPx": 1920,
                "heightPx": 1080,
                "imageSha256": SHA_A,
                "instances": [
                    {
                        "cueId": "cue-1",
                        "bounds": {
                            "x": 300,
                            "y": 790,
                            "width": 1320,
                            "height": 180,
                        },
                        "inkBounds": {
                            "x": 340,
                            "y": 820,
                            "width": 1240,
                            "height": 110,
                        },
                        "clippedPixelCount": 0,
                        "edgeTouchingPixelCount": 0,
                        "overflowDetected": False,
                        "fontEvidence": _font_evidence(first_text),
                        "contrastSamples": _samples("cue-1"),
                    }
                ],
            },
            {
                "frameId": "frame-2",
                "timestampMs": 6500,
                "widthPx": 1920,
                "heightPx": 1080,
                "imageSha256": SHA_B,
                "instances": [
                    {
                        "cueId": "cue-2",
                        "bounds": {
                            "x": 300,
                            "y": 790,
                            "width": 1320,
                            "height": 180,
                        },
                        "inkBounds": {
                            "x": 340,
                            "y": 820,
                            "width": 1240,
                            "height": 110,
                        },
                        "clippedPixelCount": 0,
                        "edgeTouchingPixelCount": 0,
                        "overflowDetected": False,
                        "fontEvidence": _font_evidence(second_text),
                        "contrastSamples": _samples("cue-2"),
                    }
                ],
            },
        ],
    }


def _codes(result: Any, gate: str) -> set[str]:
    return {
        item["code"]
        for item in result.to_dict()["gates"][gate]["violations"]
    }


def test_schema_is_draft_2020_12_and_versioned(
    validator: Draft202012Validator,
) -> None:
    assert validator.schema["$schema"].endswith("draft/2020-12/schema")
    assert SUBTITLE_VISUAL_QA_SCHEMA_VERSION == "1.0.0"
    assert validator.schema["$id"].endswith("/1.0.0")


def test_passing_request_and_result_validate_against_one_contract(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    assert not list(validator.iter_errors(request))

    result = evaluate_subtitle_visual_qa(request)
    payload = result.to_dict()

    assert result.passed
    assert payload["passed"]
    assert payload["failClosed"] is True
    assert payload["failureCodes"] == []
    assert payload["metrics"]["framesExpected"] == 2
    assert payload["metrics"]["framesEvaluated"] == 2
    assert payload["metrics"]["verifiedWordTimingCues"] == 1
    assert payload["metrics"]["backgroundClassesCovered"] == ["dark", "light"]
    assert not list(validator.iter_errors(payload))


def test_deterministic_output_ignores_mapping_insertion_order() -> None:
    request = _request()
    reordered = {key: request[key] for key in reversed(tuple(request))}
    reordered["policy"] = {
        key: request["policy"][key]
        for key in reversed(tuple(request["policy"]))
    }

    first = evaluate_subtitle_visual_qa(request)
    second = evaluate_subtitle_visual_qa(reordered)

    assert first.input_sha256 == second.input_sha256
    assert first.canonical_json() == second.canonical_json()
    assert deterministic_sha256(request) == deterministic_sha256(reordered)


def test_default_policy_is_detached_and_explicit() -> None:
    first = default_subtitle_visual_qa_policy()
    second = default_subtitle_visual_qa_policy()
    first["safeArea"]["leftRatio"] = 0.2
    first["contrast"]["requiredBackgroundClasses"].remove("light")

    assert second["safeArea"]["leftRatio"] == 0.05
    assert second["contrast"]["requiredBackgroundClasses"] == [
        "dark",
        "light",
    ]
    assert second["karaoke"]["requireVerifiedWordTiming"] is True


def test_missing_expected_frame_and_unrepresented_cue_fail_closed(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["frames"].pop()
    assert not list(validator.iter_errors(request))
    result = evaluate_subtitle_visual_qa(request)

    assert not result.passed
    assert {
        "expected-frame-missing",
        "cue-not-represented",
    } <= _codes(result, "samplingEvidence")


def test_unexpected_frame_and_timestamp_outside_cue_fail_closed() -> None:
    request = _request()
    request["sampling"]["expectedFrameIds"] = ["frame-1", "unknown-frame"]
    request["frames"][0]["timestampMs"] = 4500
    result = evaluate_subtitle_visual_qa(request)

    assert "unexpected-frame" in _codes(result, "samplingEvidence")
    assert "expected-frame-missing" in _codes(result, "samplingEvidence")
    assert "frame-outside-cue-time" in _codes(result, "samplingEvidence")


def test_safe_area_uses_visible_ink_and_fails_when_it_leaves_margin() -> None:
    request = _request()
    request["frames"][0]["instances"][0]["inkBounds"]["x"] = 20
    result = evaluate_subtitle_visual_qa(request)

    assert not result.to_dict()["gates"]["safeArea"]["passed"]
    assert _codes(result, "safeArea") == {"outside-safe-area"}


def test_visible_ink_touching_frame_edge_is_a_clipping_failure() -> None:
    request = _request()
    request["policy"]["safeArea"] = {
        "leftRatio": 0,
        "rightRatio": 0,
        "topRatio": 0,
        "bottomRatio": 0,
    }
    instance = request["frames"][0]["instances"][0]
    instance["bounds"]["x"] = 0
    instance["inkBounds"]["x"] = 0
    result = evaluate_subtitle_visual_qa(request)

    assert "ink-touches-frame-edge" in _codes(result, "clipping")


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda item: item["bounds"].update({"x": 1800, "width": 200}),
            "bounds-outside-frame",
        ),
        (
            lambda item: item["inkBounds"].update({"y": 1040, "height": 80}),
            "ink-outside-frame",
        ),
        (
            lambda item: item["inkBounds"].update({"x": 250}),
            "ink-outside-declared-bounds",
        ),
        (
            lambda item: item.update({"clippedPixelCount": 3}),
            "clipped-pixels-detected",
        ),
        (
            lambda item: item.update({"edgeTouchingPixelCount": 1}),
            "frame-edge-touch-detected",
        ),
        (
            lambda item: item.update({"overflowDetected": True}),
            "renderer-overflow-detected",
        ),
    ],
)
def test_clipping_and_overflow_signals_are_hard_failures(
    mutation: Any,
    expected_code: str,
) -> None:
    request = _request()
    mutation(request["frames"][0]["instances"][0])
    result = evaluate_subtitle_visual_qa(request)

    assert expected_code in _codes(result, "clipping")


def test_low_contrast_is_calculated_from_pixels_not_a_claimed_score() -> None:
    request = _request()
    sample = request["frames"][0]["instances"][0]["contrastSamples"][0]
    sample["foregroundRgb"] = "#333333"
    sample["backgroundRgb"] = "#222222"
    result = evaluate_subtitle_visual_qa(request)

    assert "contrast-below-threshold" in _codes(result, "contrast")
    assert result.to_dict()["metrics"]["minimumContrastRatio"] < 4.5


def test_missing_dark_or_light_style_evidence_fails_closed() -> None:
    request = _request()
    for frame in request["frames"]:
        frame["instances"][0]["contrastSamples"] = [
            frame["instances"][0]["contrastSamples"][0]
        ]
    result = evaluate_subtitle_visual_qa(request)

    assert "required-background-class-missing" in _codes(result, "contrast")


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        (
            "sampledFromRenderedFrame",
            False,
            "unrendered-pixel-sample",
        ),
        (
            "foregroundPixelCount",
            1,
            "foreground-sample-too-small",
        ),
        (
            "backgroundPixelCount",
            1,
            "background-sample-too-small",
        ),
    ],
)
def test_contrast_sample_provenance_and_size_are_hard_gates(
    field: str,
    value: Any,
    expected_code: str,
) -> None:
    request = _request()
    request["frames"][0]["instances"][0]["contrastSamples"][0][field] = value
    result = evaluate_subtitle_visual_qa(request)

    assert expected_code in _codes(result, "contrast")


def test_background_class_must_match_sampled_luminance() -> None:
    request = _request()
    sample = request["frames"][0]["instances"][0]["contrastSamples"][0]
    sample["backgroundClass"] = "dark"
    sample["backgroundRgb"] = "#FFFFFF"
    sample["foregroundRgb"] = "#000000"
    result = evaluate_subtitle_visual_qa(request)

    assert "background-class-contradicted" in _codes(result, "contrast")


def test_empty_contrast_samples_cannot_pass_by_omission(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["frames"][0]["instances"][0]["contrastSamples"] = []
    assert not list(validator.iter_errors(request))
    result = evaluate_subtitle_visual_qa(request)

    assert "contrast-evidence-missing" in _codes(result, "contrast")


def test_duplicate_contrast_sample_identity_is_rejected() -> None:
    request = _request()
    samples = request["frames"][0]["instances"][0]["contrastSamples"]
    samples[1]["sampleId"] = samples[0]["sampleId"]

    with pytest.raises(SubtitleVisualQAInputError, match="duplicate sampleId"):
        evaluate_subtitle_visual_qa(request)


def test_missing_font_evidence_cannot_pass_by_omission(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["frames"][0]["instances"][0]["fontEvidence"] = None
    assert not list(validator.iter_errors(request))
    result = evaluate_subtitle_visual_qa(request)

    assert "font-evidence-missing" in _codes(result, "fontGlyph")
    assert result.to_dict()["fontTruth"]["evidenceInstances"] == 1


def test_unverified_font_and_missing_glyph_evidence_fail_together() -> None:
    request = _request()
    evidence = request["frames"][0]["instances"][0]["fontEvidence"]
    evidence["resolutionVerified"] = False
    evidence["glyphCoverageVerified"] = False
    evidence["verificationMethod"] = "not-provided"
    evidence["evidenceArtifactSha256"] = None
    evidence["coveredRenderableCodePoints"] = 3
    evidence["missingCodePoints"] = ["U+4E16"]
    evidence["tofuGlyphCount"] = 1
    result = evaluate_subtitle_visual_qa(request)

    assert {
        "font-resolution-unverified",
        "font-verification-artifact-missing",
        "glyph-coverage-unverified",
        "glyph-coverage-incomplete",
        "missing-glyphs-detected",
        "tofu-glyphs-detected",
    } <= _codes(result, "fontGlyph")


def test_glyph_expected_count_is_bound_to_exact_cue_text() -> None:
    request = _request()
    request["frames"][0]["instances"][0]["fontEvidence"][
        "expectedRenderableCodePoints"
    ] = 99
    result = evaluate_subtitle_visual_qa(request)

    assert "expected-codepoint-count-mismatch" in _codes(result, "fontGlyph")


def test_no_installation_or_embedding_claim_is_made_without_evidence() -> None:
    result = evaluate_subtitle_visual_qa(_request()).to_dict()

    assert result["fontTruth"]["installation"] == {
        "verified": 0,
        "notAsserted": 2,
    }
    assert result["fontTruth"]["embedding"] == {
        "verified": 0,
        "notAsserted": 2,
    }
    assert "installed" not in json.dumps(result["fontTruth"])
    assert "embedded" not in json.dumps(result["fontTruth"])


def test_verified_installation_and_embedding_claims_require_bound_evidence(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["frames"][0]["instances"][0]["fontEvidence"] = _font_evidence(
        "你好世界",
        installation_status="verified-installed",
        embedding_status="verified-embedded",
    )
    assert not list(validator.iter_errors(request))

    result = evaluate_subtitle_visual_qa(request)
    truth = result.to_dict()["fontTruth"]
    assert result.passed
    assert truth["installation"]["verified"] == 1
    assert truth["embedding"]["verified"] == 1


def test_positive_font_claim_without_hash_is_rejected_by_schema_and_python(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    claim = request["frames"][0]["instances"][0]["fontEvidence"][
        "installation"
    ]
    claim["status"] = "verified-installed"
    claim["verificationMethod"] = "directwrite-enumeration"

    assert list(validator.iter_errors(request))
    with pytest.raises(
        SubtitleVisualQAInputError,
        match="positive claim requires",
    ):
        evaluate_subtitle_visual_qa(request)


def test_line_count_length_reconstruction_and_reading_speed_are_hard_gates() -> None:
    request = _request()
    cue = request["cues"][0]
    cue["endMs"] = 1100
    cue["renderedLines"] = ["你", "好", "WRONG"]
    request["policy"]["layout"]["maxCharactersPerLine"] = 2
    result = evaluate_subtitle_visual_qa(request)

    assert {
        "line-count-exceeded",
        "line-length-exceeded",
        "rendered-text-mismatch",
        "reading-speed-exceeded",
    } <= _codes(result, "lineReading")


def test_similar_and_identical_speaker_colors_fail_distinguishability() -> None:
    request = _request()
    request["speakers"][1]["color"] = "#00A9E8"
    result = evaluate_subtitle_visual_qa(request)
    assert "speaker-colors-not-distinct" in _codes(result, "speakerColor")

    request["speakers"][1]["color"] = "#00A8E8"
    result = evaluate_subtitle_visual_qa(request)
    assert "speaker-colors-not-distinct" in _codes(result, "speakerColor")
    assert result.to_dict()["metrics"]["minimumSpeakerDeltaE2000"] == 0.0


def test_visual_overlap_is_computed_from_ink_rectangles() -> None:
    request = _request()
    request["cues"][1]["startMs"] = 2000
    request["cues"][1]["endMs"] = 4000
    request["frames"][0]["instances"].append(
        copy.deepcopy(request["frames"][1]["instances"][0])
    )
    request["frames"][0]["instances"][1]["inkBounds"] = {
        "x": 1000,
        "y": 850,
        "width": 600,
        "height": 100,
    }
    request["frames"].pop()
    request["sampling"]["expectedFrameIds"] = ["frame-1"]
    result = evaluate_subtitle_visual_qa(request)

    assert "subtitle-ink-overlap" in _codes(result, "visualOverlap")
    assert result.to_dict()["metrics"]["maximumOverlapAreaPx"] > 0


def test_spatially_separated_simultaneous_subtitles_do_not_fail_overlap() -> None:
    request = _request()
    request["cues"][1]["startMs"] = 2000
    request["cues"][1]["endMs"] = 4000
    second = copy.deepcopy(request["frames"][1]["instances"][0])
    second["bounds"] = {"x": 300, "y": 580, "width": 1320, "height": 160}
    second["inkBounds"] = {"x": 340, "y": 610, "width": 1240, "height": 90}
    request["frames"][0]["instances"].append(second)
    request["frames"].pop()
    request["sampling"]["expectedFrameIds"] = ["frame-1"]
    result = evaluate_subtitle_visual_qa(request)

    assert result.to_dict()["gates"]["visualOverlap"]["passed"]
    assert result.to_dict()["metrics"]["maximumOverlapAreaPx"] == 0


def test_word_progress_without_word_timing_never_passes(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["cues"][0]["wordTimingEvidence"] = None
    assert not list(validator.iter_errors(request))
    result = evaluate_subtitle_visual_qa(request)

    assert "word-timing-evidence-missing" in _codes(
        result, "karaokeAuthenticity"
    )


@pytest.mark.parametrize(
    "source",
    ["segment-interpolation", "synthetic-even-split"],
)
def test_synthetic_segment_timing_is_rejected_as_fake_karaoke(
    source: str,
) -> None:
    request = _request()
    request["cues"][0]["wordTimingEvidence"]["source"] = source
    result = evaluate_subtitle_visual_qa(request)

    assert "synthetic-word-timing-source" in _codes(
        result, "karaokeAuthenticity"
    )


def test_word_timing_must_be_verified_hash_bound_and_reconstruct_text() -> None:
    request = _request()
    evidence = request["cues"][0]["wordTimingEvidence"]
    evidence["verified"] = False
    evidence["cueTextSha256"] = SHA_A
    evidence["words"][1]["text"] = "错误"
    result = evaluate_subtitle_visual_qa(request)

    assert {
        "word-timing-unverified",
        "word-timing-text-hash-mismatch",
        "word-text-reconstruction-mismatch",
    } <= _codes(result, "karaokeAuthenticity")


def test_word_timings_must_be_positive_ordered_and_inside_the_cue() -> None:
    request = _request()
    words = request["cues"][0]["wordTimingEvidence"]["words"]
    words[0]["endMs"] = words[0]["startMs"]
    words[1]["startMs"] = 900
    words[1]["endMs"] = 4500
    result = evaluate_subtitle_visual_qa(request)

    assert {
        "word-duration-invalid",
        "word-outside-cue-time",
        "word-timing-overlap",
    } <= _codes(result, "karaokeAuthenticity")


def test_karaoke_policy_cannot_disable_true_word_timing_gate(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["policy"]["karaoke"]["requireVerifiedWordTiming"] = False

    assert list(validator.iter_errors(request))
    with pytest.raises(SubtitleVisualQAInputError, match="must be true"):
        evaluate_subtitle_visual_qa(request)


def test_duplicate_cue_instance_in_one_frame_is_rejected() -> None:
    request = _request()
    request["frames"][0]["instances"].append(
        copy.deepcopy(request["frames"][0]["instances"][0])
    )
    result = evaluate_subtitle_visual_qa(request)

    assert "duplicate-cue-instance" in _codes(result, "samplingEvidence")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("kind",), "wrong-kind"),
        (("schemaVersion",), "2.0.0"),
        (("policy", "safeArea", "leftRatio"), float("nan")),
        (("policy", "contrast", "requiredBackgroundClasses"), []),
        (("policy", "contrast", "darkMaximumLuminance"), 0.8),
        (("policy", "contrast", "lightMinimumLuminance"), 0.2),
        (("cues", 0, "speakerId"), "missing-speaker"),
        (("frames", 0, "instances", 0, "cueId"), "missing-cue"),
    ],
)
def test_malformed_or_ambiguous_input_is_rejected(
    path: tuple[Any, ...],
    value: Any,
) -> None:
    request = _request()
    cursor: Any = request
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value

    with pytest.raises(SubtitleVisualQAInputError):
        evaluate_subtitle_visual_qa(request)


def test_unknown_properties_are_rejected_by_schema_and_python(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["networkUrl"] = "https://example.invalid"

    assert list(validator.iter_errors(request))
    with pytest.raises(SubtitleVisualQAInputError, match="unsupported keys"):
        evaluate_subtitle_visual_qa(request)


def test_color_math_has_expected_extremes() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#123456", "#123456") == pytest.approx(1.0)
    assert delta_e_2000("#123456", "#123456") == pytest.approx(0.0)
    assert delta_e_2000("#000000", "#FFFFFF") > 90


def test_failure_codes_and_violations_are_stably_sorted(
    validator: Draft202012Validator,
) -> None:
    request = _request()
    request["frames"][0]["instances"][0]["overflowDetected"] = True
    request["frames"][0]["instances"][0]["clippedPixelCount"] = 2
    request["cues"][0]["wordTimingEvidence"] = None

    result = evaluate_subtitle_visual_qa(request)
    payload = result.to_dict()
    clipping_codes = [
        item["code"] for item in payload["gates"]["clipping"]["violations"]
    ]

    assert clipping_codes == sorted(clipping_codes)
    assert payload["failureCodes"] == sorted(payload["failureCodes"])
    assert not list(validator.iter_errors(payload))
