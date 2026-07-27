from __future__ import annotations

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
