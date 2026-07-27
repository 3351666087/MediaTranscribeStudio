from __future__ import annotations

import pytest

from tools.pyannote_runtime import _sample_bounds


def test_sample_bounds_clamp_sub_millisecond_end_rounding_overrun() -> None:
    assert _sample_bounds(
        start_ms=0,
        end_ms=59_960,
        sample_rate=16_000,
        frame_count=959_355,
    ) == (0, 959_355)


def test_sample_bounds_reject_material_end_overrun() -> None:
    with pytest.raises(
        ValueError,
        match="inference interval exceeds audio",
    ):
        _sample_bounds(
            start_ms=0,
            end_ms=59_960,
            sample_rate=16_000,
            frame_count=959_343,
        )


def test_sample_bounds_reject_empty_interval_at_audio_end() -> None:
    with pytest.raises(
        ValueError,
        match="inference interval exceeds audio",
    ):
        _sample_bounds(
            start_ms=60_000,
            end_ms=60_001,
            sample_rate=16_000,
            frame_count=960_000,
        )
