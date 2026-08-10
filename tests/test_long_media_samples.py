from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import long_media_samples
from tools.long_media_samples import (
    AudioFrameFeature,
    LongMediaSampleError,
    ReferenceTurn,
    build_long_media_matrix,
    mark_activity_and_changes,
    parse_rttm,
    select_reference_speaker_window,
    select_stratified_windows,
)


def _frames(duration_seconds: int) -> tuple[AudioFrameFeature, ...]:
    raw = [
        AudioFrameFeature(
            start_ms=index * 1000,
            end_ms=(index + 1) * 1000,
            rms_db=-50.0 if index % 17 == 0 else -18.0,
            spectral_centroid_hz=400.0 + (index % 11) * 170.0,
            zero_crossing_rate=0.02 + (index % 7) * 0.01,
        )
        for index in range(duration_seconds)
    ]
    return mark_activity_and_changes(raw)[0]


def test_stratified_selection_is_deterministic_and_spans_timeline() -> None:
    frames = _frames(600)

    first = select_stratified_windows(
        frames,
        duration_ms=600_000,
        window_ms=60_000,
        random_seed=42,
    )
    second = select_stratified_windows(
        frames,
        duration_ms=600_000,
        window_ms=60_000,
        random_seed=42,
    )

    assert first == second
    assert [item["reason"] for item in first[:3]] == [
        "stratum-start",
        "stratum-middle",
        "stratum-end",
    ]
    assert first[0]["startMs"] < 60_000
    assert 240_000 <= first[1]["startMs"] <= 300_000
    assert first[2]["endMs"] == 600_000
    assert all(item["selectionUsesModelScores"] is False for item in first)
    assert len(first) == 8


def test_builder_rejects_dataless_source_before_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(
        long_media_samples,
        "_path_is_dataless",
        lambda path: path == source,
    )

    with pytest.raises(LongMediaSampleError, match="dataless cloud placeholder"):
        build_long_media_matrix(
            (source,),
            output_root=tmp_path / "output",
        )


def _source_reference(path: Path, source: Path, *, duration_seconds: float) -> Path:
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    value = {
        "schemaVersion": "1.0.0",
        "artifactType": "long-media-source-reference",
        "source": {
            "dataset": "wikimedia-commons",
            "revision": "commons-page-revision-123+etag-fixture",
            "provider": "Wikimedia Commons",
            "sourceUrl": "https://upload.wikimedia.org/fixture.ogg",
            "descriptionUrl": "https://commons.wikimedia.org/wiki/File:Fixture.ogg",
            "sha256": source_sha256,
            "bytes": source.stat().st_size,
            "durationSeconds": duration_seconds,
            "languageTags": ["en-US"],
            "region": "US",
            "recordingType": "real-recording",
            "speechNature": "natural-multi-speaker",
            "immutableEvidence": {
                "pageRevisionId": 123,
                "etag": "fixture",
            },
            "license": {
                "id": "cc-by-2.5",
                "shortName": "CC BY 2.5",
                "url": "https://creativecommons.org/licenses/by/2.5/",
                "attributionRequired": True,
            },
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fake_media_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    duration_ms: int,
) -> None:
    def fake_probe(path: Path, *, ffprobe: str) -> dict[str, object]:
        if path == source.resolve():
            return {
                "durationMs": duration_ms,
                "format": {
                    "format_name": "ogg",
                    "duration": f"{duration_ms / 1000:.6f}",
                    "size": str(source.stat().st_size),
                },
                "streams": [
                    {
                        "codec_type": "audio",
                        "codec_name": "vorbis",
                        "sample_rate": "48000",
                        "channels": 2,
                    }
                ],
            }
        return {
            "durationMs": 60_000,
            "format": {"format_name": "wav", "duration": "60.000000"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "sample_rate": "16000",
                    "channels": 1,
                }
            ],
        }

    monkeypatch.setattr(long_media_samples, "probe_media", fake_probe)
    monkeypatch.setattr(
        long_media_samples,
        "analyze_audio",
        lambda path, *, ffmpeg: (_frames(duration_ms // 1000), -30.0),
    )
    monkeypatch.setattr(
        long_media_samples,
        "_extract_audio_window",
        lambda source, output, **kwargs: (
            output.parent.mkdir(parents=True, exist_ok=True),
            output.write_bytes(output.name.encode("utf-8")),
        ),
    )


def test_builder_freezes_held_out_source_evidence_and_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.ogg"
    source.write_bytes(b"pinned public recording")
    reference = _source_reference(
        tmp_path / "source-reference.json",
        source,
        duration_seconds=600.0,
    )
    _fake_media_pipeline(monkeypatch, source, duration_ms=600_000)

    value = build_long_media_matrix(
        (source,),
        output_root=tmp_path / "output",
        evaluation_split="held-out",
        source_reference_paths=(reference,),
    )

    frozen_source = value["sources"][0]
    assert frozen_source["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert frozen_source["dataset"] == "wikimedia-commons"
    assert frozen_source["revision"] == "commons-page-revision-123+etag-fixture"
    assert frozen_source["license"] == "cc-by-2.5"
    assert frozen_source["languageTags"] == ["en-US"]
    assert frozen_source["recordingType"] == "real-recording"
    assert frozen_source["licenseEvidence"]["sourceReferenceSha256"] == hashlib.sha256(
        reference.read_bytes()
    ).hexdigest()
    assert value["evaluationSplit"] == "held-out"
    assert {row["evaluationSplit"] for row in value["cases"]} == {"held-out"}
    assert {row["region"] for row in value["cases"]} == {"US"}


def test_builder_rejects_short_source_and_missing_held_out_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.ogg"
    source.write_bytes(b"short recording")
    _fake_media_pipeline(monkeypatch, source, duration_ms=299_999)

    with pytest.raises(LongMediaSampleError, match="at least 300 seconds"):
        build_long_media_matrix(
            (source,),
            output_root=tmp_path / "short-output",
        )

    _fake_media_pipeline(monkeypatch, source, duration_ms=600_000)
    with pytest.raises(LongMediaSampleError, match="source reference"):
        build_long_media_matrix(
            (source,),
            output_root=tmp_path / "held-output",
            evaluation_split="held-out",
        )


def test_activity_threshold_retains_loud_frames_and_change_scores() -> None:
    frames, threshold = mark_activity_and_changes(
        (
            AudioFrameFeature(0, 1000, -60.0, 100.0, 0.01),
            AudioFrameFeature(1000, 2000, -20.0, 1800.0, 0.20),
            AudioFrameFeature(2000, 3000, -18.0, 1850.0, 0.19),
        )
    )

    assert -60.0 < threshold < -18.0
    assert [frame.active for frame in frames] == [False, True, True]
    assert frames[1].acoustic_change_score > frames[2].acoustic_change_score


def test_reference_window_uses_exact_truth_speaker_count() -> None:
    turns = tuple(
        ReferenceTurn(
            recording_id="meeting",
            speaker_id=f"speaker-{index}",
            start_ms=index * 1000,
            end_ms=70_000 + index * 1000,
        )
        for index in range(1, 6)
    ) + (
        ReferenceTurn("meeting", "speaker-6", 100_000, 150_000),
    )

    selected = select_reference_speaker_window(
        turns,
        media_duration_ms=180_000,
        window_ms=60_000,
        target_speaker_count=5,
        random_seed=7,
    )

    assert selected["selectionTruth"] == "reference-rttm"
    assert selected["selectionUsesModelScores"] is False
    assert selected["speakerSet"] == [
        "speaker-1",
        "speaker-2",
        "speaker-3",
        "speaker-4",
        "speaker-5",
    ]


def test_reference_window_fails_when_exact_count_is_missing() -> None:
    turns = (
        ReferenceTurn("meeting", "speaker-1", 0, 40_000),
        ReferenceTurn("meeting", "speaker-2", 0, 40_000),
    )

    with pytest.raises(LongMediaSampleError, match="exactly 5 speakers"):
        select_reference_speaker_window(
            turns,
            media_duration_ms=60_000,
            window_ms=60_000,
            target_speaker_count=5,
            random_seed=7,
        )


def test_rttm_requires_recording_id_when_multiple_are_present(
    tmp_path: Path,
) -> None:
    rttm = tmp_path / "multi.rttm"
    rttm.write_text(
        "SPEAKER one 1 0.0 1.0 <NA> <NA> a <NA> <NA>\n"
        "SPEAKER two 1 0.0 1.0 <NA> <NA> b <NA> <NA>\n",
        encoding="utf-8",
    )

    with pytest.raises(LongMediaSampleError, match="multiple recordings"):
        parse_rttm(rttm)
    assert parse_rttm(rttm, recording_id="two") == (
        ReferenceTurn("two", "b", 0, 1000),
    )
