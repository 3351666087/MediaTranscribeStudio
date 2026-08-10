from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.benchmark_w2vbert2_verification import (
    W2VBert2BenchmarkError,
    _checkpoint_without_projection,
    _inference_config,
    _verify_declared_files,
)


def _row(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_candidate_files_are_size_and_digest_bound(tmp_path: Path) -> None:
    payload = b"pinned checkpoint fixture"
    target = tmp_path / "nested" / "model.pt"
    target.parent.mkdir()
    target.write_bytes(payload)

    assert _verify_declared_files(tmp_path, [_row("nested/model.pt", payload)]) == [
        _row("nested/model.pt", payload)
    ]
    target.write_bytes(payload + b"changed")
    with pytest.raises(W2VBert2BenchmarkError, match="no longer matches"):
        _verify_declared_files(tmp_path, [_row("nested/model.pt", payload)])


def test_candidate_files_reject_traversal(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-model.pt"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(W2VBert2BenchmarkError, match="metadata is invalid"):
            _verify_declared_files(tmp_path, [_row("../outside-model.pt", b"outside")])
    finally:
        outside.unlink(missing_ok=True)


def test_inference_config_disables_training_only_mask_parameter() -> None:
    raw = {
        "model_type": "wav2vec2-bert",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "apply_spec_augment": True,
        "mask_feature_prob": 0.1,
        "mask_time_prob": 0.05,
    }
    result = _inference_config(raw)

    assert result["apply_spec_augment"] is False
    assert result["mask_feature_prob"] == 0.0
    assert result["mask_time_prob"] == 0.0
    assert raw["mask_time_prob"] == 0.05


def test_checkpoint_allows_only_the_known_training_projection() -> None:
    tensor = object()
    assert _checkpoint_without_projection(
        {"frontend.encoder.weight": tensor, "projection.weight": tensor}
    ) == {"frontend.encoder.weight": tensor}
    with pytest.raises(W2VBert2BenchmarkError, match="classification keys"):
        _checkpoint_without_projection({"frontend.encoder.weight": tensor})
    with pytest.raises(W2VBert2BenchmarkError, match="classification keys"):
        _checkpoint_without_projection(
            {
                "frontend.encoder.weight": tensor,
                "projection.weight": tensor,
                "projection.bias": tensor,
            }
        )
