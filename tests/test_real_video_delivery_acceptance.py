from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from tools.run_real_video_delivery_acceptance import (
    _frame_metrics,
    _representative_indexes,
)


def _frame(path: Path, *, overlay: bool) -> None:
    image = Image.new("RGB", (320, 180), "#303030")
    if overlay:
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 120, 240, 160), fill="white")
    image.save(path)


def test_representative_indexes_are_bounded_unique_and_include_edges() -> None:
    assert _representative_indexes(1) == (0,)
    assert _representative_indexes(2) == (0, 1)
    assert _representative_indexes(17) == (0, 4, 8, 12, 16)


def test_frame_metrics_requires_visible_safe_overlay(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    expected = tmp_path / "expected.png"
    actual = tmp_path / "actual.png"
    _frame(source, overlay=False)
    _frame(expected, overlay=True)
    _frame(actual, overlay=True)

    result = _frame_metrics(source, expected, actual)

    assert result["passed"] is True
    assert result["expectedOverlayVisible"] is True
    assert result["insideConservativeSafeArea"] is True
    assert result["burnInVisible"] is True


def test_frame_metrics_rejects_missing_overlay(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    expected = tmp_path / "expected.png"
    _frame(source, overlay=False)
    _frame(expected, overlay=False)

    assert _frame_metrics(source, expected, None) == {
        "passed": False,
        "reason": "no-overlay-pixels",
    }
