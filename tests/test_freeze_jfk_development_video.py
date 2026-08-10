from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.persistence import canonical_json_sha256
from tools.freeze_jfk_development_video import (
    CASE_ID,
    HISTORICAL_WINDOW_SHA256,
    REFERENCE_CUES,
    SOURCE_SHA256,
    WINDOW_END_MS,
    WINDOW_SHA256,
    WINDOW_START_MS,
    JfkVideoFreezeError,
    _assert_truth_redacted,
    _ffmpeg_arguments,
    blind_body,
    reference_body,
)


def _probe(*, source: bool = False) -> dict[str, object]:
    if source:
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "vp9",
                    "width": 320,
                    "height": 240,
                },
                {
                    "codec_type": "audio",
                    "codec_name": "opus",
                    "sample_rate": "48000",
                    "channels": 2,
                },
            ],
            "format": {"duration": "930.103000"},
        }
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 320,
                "height": 240,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "2997/100",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
        "format": {"duration": "13.013013"},
    }


def _without_canonical(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
    return body


def test_reference_pins_public_domain_source_window_and_official_cues(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.webm"
    window = tmp_path / "window.mp4"
    blind = tmp_path / "blind.json"

    value = reference_body(
        source=source,
        window=window,
        blind_manifest=blind,
        source_probe=_probe(source=True),
        window_probe=_probe(),
        ffmpeg_version=("ffmpeg version fixture",),
    )

    _without_canonical(value)
    assert value["evaluation"] == {
        "split": "development",
        "tuningEligible": True,
        "heldOut": False,
    }
    assert value["source"]["derivative"]["sha256"] == SOURCE_SHA256
    assert value["source"]["page"]["revisionId"] == 1_141_760_595
    assert value["license"]["shortName"] == "Public domain"
    assert value["license"]["attributionRequired"] is False
    assert value["window"]["sourceStartMs"] == WINDOW_START_MS
    assert value["window"]["sourceEndMs"] == WINDOW_END_MS
    assert value["window"]["sha256"] == WINDOW_SHA256
    assert value["window"]["artifactContinuity"] == {
        "historicalSha256": HISTORICAL_WINDOW_SHA256,
        "historicalByteIdentityMatched": False,
        "samePinnedSourceAndWindow": True,
        "note": (
            "The prior artifact used a different encoder build or recipe. "
            "This Windows freeze records a new immutable byte identity."
        ),
    }
    truth = value["referenceTruth"]
    assert truth["expectedSpeakerCount"] == 1
    assert truth["timedText"]["revisionId"] == 917_372_138
    assert truth["timedText"]["cues"] == list(REFERENCE_CUES)


def test_blind_manifest_is_runner_compatible_and_truth_redacted(
    tmp_path: Path,
) -> None:
    window = tmp_path / "window.mp4"
    manifest = tmp_path / "blind.json"

    value = blind_body(
        window=window,
        manifest_path=manifest,
        window_probe=_probe(),
    )

    _without_canonical(value)
    assert value["artifactType"] == "truth-redacted-local-product-matrix"
    assert value["selection"]["splits"] == ["development"]
    assert value["selection"]["heldOutExplicitlyUnlocked"] is False
    assert value["counts"] == {"cases": 1, "missingCases": 0}
    case = value["cases"][0]
    assert case["id"] == CASE_ID
    assert case["path"] == "window.mp4"
    assert case["sha256"] == WINDOW_SHA256
    assert case["durationSeconds"] == 13.013013
    encoded = json.dumps(value, ensure_ascii=False)
    assert all(str(cue["text"]) not in encoded for cue in REFERENCE_CUES)
    assert "917372138" not in encoded


def test_blind_redaction_rejects_nested_reference_truth() -> None:
    with pytest.raises(JfkVideoFreezeError, match="referenceTranscript"):
        _assert_truth_redacted(
            {"cases": [{"referenceTranscript": "secret answer"}]}
        )


def test_ffmpeg_recipe_is_bounded_and_reproducible() -> None:
    arguments = _ffmpeg_arguments("{source}", "{output}")

    assert arguments[arguments.index("-ss") + 1] == "832.500"
    assert arguments[arguments.index("-t") + 1] == "13.000"
    assert arguments[arguments.index("-c:v") + 1] == "libx264"
    assert arguments[arguments.index("-c:a") + 1] == "aac"
    assert arguments[arguments.index("-threads:v") + 1] == "1"
    assert arguments[-1] == "{output}"
