from __future__ import annotations

import argparse

import numpy as np
import pytest

from backend.persistence import canonical_json_sha256
from tools.benchmark_redimnet2_batch_scan import (
    ReDimNet2BenchmarkError,
    _attach_canonical_digest,
    _batch_sizes,
    _compare_trial_score_sets,
    _compare_vector_sets,
    _release_model_holder,
    _validated_resource_snapshot,
    _vector_set_sha256,
)
from tools.benchmark_redimnet2_verification import (
    _embedding_set_sha256 as _verification_embedding_set_sha256,
)


class _CpuDevice:
    type = "cpu"


class _UnusedCuda:
    def synchronize(self, device: object) -> None:  # pragma: no cover
        raise AssertionError("CPU release must not call CUDA")


class _FakeTorch:
    cuda = _UnusedCuda()


class _FakeModel:
    pass


def _trial_row(trial_id: str, cosine: float) -> dict[str, object]:
    return {
        "trialId": trial_id,
        "enrollmentClipId": "left",
        "testClipId": "right",
        "sameSpeaker": True,
        "cosine": cosine,
    }


def test_batch_size_parser_requires_unique_positive_values() -> None:
    assert _batch_sizes("4, 8,12,16") == (4, 8, 12, 16)
    for invalid in ("", "4,,8", "4,4", "0,4", "four,8"):
        with pytest.raises(argparse.ArgumentTypeError):
            _batch_sizes(invalid)


def test_vector_and_score_comparisons_apply_fixed_tolerance() -> None:
    reference_vectors = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.5, -0.5], dtype=np.float32),
    }
    close_vectors = {
        "a": np.asarray([1.0 + 1e-6, 0.0], dtype=np.float32),
        "b": np.asarray([0.5, -0.5], dtype=np.float32),
    }
    far_vectors = {
        **close_vectors,
        "b": np.asarray([0.5, -0.49], dtype=np.float32),
    }
    close = _compare_vector_sets(
        reference_vectors,
        close_vectors,
        numpy_module=np,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    far = _compare_vector_sets(
        reference_vectors,
        far_vectors,
        numpy_module=np,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    assert close["consistent"] is True
    assert close["clipCount"] == 2
    assert close["valueCount"] == 4
    assert far["consistent"] is False

    reference_scores = [_trial_row("trial-1", 0.9)]
    close_scores = [_trial_row("trial-1", 0.900001)]
    far_scores = [_trial_row("trial-1", 0.8)]
    assert _compare_trial_score_sets(
        reference_scores,
        close_scores,
        numpy_module=np,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )["consistent"] is True
    assert _compare_trial_score_sets(
        reference_scores,
        far_scores,
        numpy_module=np,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )["consistent"] is False


def test_vector_digest_is_order_independent_and_value_bound() -> None:
    first = {
        "b": np.asarray([2.0], dtype=np.float32),
        "a": np.asarray([1.0], dtype=np.float32),
    }
    reordered = {"a": first["a"], "b": first["b"]}
    changed = {"a": first["a"], "b": np.asarray([2.1], dtype=np.float32)}
    assert _vector_set_sha256(first, numpy_module=np) == _vector_set_sha256(
        reordered, numpy_module=np
    )
    assert _vector_set_sha256(first, numpy_module=np) != _vector_set_sha256(
        changed, numpy_module=np
    )
    assert _vector_set_sha256(
        first, numpy_module=np
    ) == _verification_embedding_set_sha256(first, numpy_module=np)


def test_release_drops_the_only_model_reference() -> None:
    holder = {"model": _FakeModel()}
    collection_calls: list[bool] = []
    evidence = _release_model_holder(
        holder,
        torch_module=_FakeTorch(),
        target_device=_CpuDevice(),
        collect_garbage=lambda: collection_calls.append(True) or 0,
    )
    assert holder == {}
    assert evidence["modelObjectCollected"] is True
    assert evidence["cudaCacheEmptied"] is False
    assert evidence["cublasWorkspacesCleared"] is False
    assert evidence["cufftPlanCacheCleared"] is False
    assert collection_calls == [True, True]


def test_resource_snapshot_and_canonical_digest_are_strict() -> None:
    snapshot = {
        "processRssBytes": 100,
        "cudaAllocatedBytes": 10,
        "cudaReservedBytes": 20,
        "cudaPeakAllocatedBytes": 30,
        "cudaPeakReservedBytes": 40,
    }
    assert _validated_resource_snapshot(lambda: snapshot) == {
        key: snapshot[key] for key in sorted(snapshot)
    }
    with pytest.raises(ReDimNet2BenchmarkError, match="invalid value"):
        _validated_resource_snapshot(
            lambda: {**snapshot, "cudaAllocatedBytes": True}
        )

    report = _attach_canonical_digest(
        {"schemaVersion": "1.0.0", "passed": True}
    )
    body = dict(report)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
