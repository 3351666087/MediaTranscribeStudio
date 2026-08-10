from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_palabra_redimnet2_verification as benchmark


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_source_snapshot_rejects_local_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.email", "fixture@example.com")
    _git(source, "config", "user.name", "Fixture")
    _git(source, "config", "core.autocrlf", "false")
    required = source / "model.py"
    required.write_text("frozen = True\n", encoding="utf-8")
    _git(source, "add", "model.py")
    _git(source, "commit", "-m", "fixture")
    commit = _git(source, "rev-parse", "HEAD")
    _git(source, "tag", "-a", "v1", "-m", "fixture tag")
    tag_object = _git(source, "rev-parse", "v1")

    evidence = benchmark.verify_source_snapshot(
        source,
        expected_commit=commit,
        expected_tag="v1",
        expected_tag_object=tag_object,
        required_files=("model.py",),
    )
    assert evidence["requiredFilesMatchCommit"] is True

    required.write_text("frozen = False\n", encoding="utf-8")
    with pytest.raises(
        benchmark.PalabraReDimNet2BenchmarkError,
        match="differs from commit",
    ):
        benchmark.verify_source_snapshot(
            source,
            expected_commit=commit,
            expected_tag="v1",
            expected_tag_object=tag_object,
            required_files=("model.py",),
        )


def test_candidate_lock_pins_official_asset_and_blocks_promotion() -> None:
    lock = json.loads(benchmark.DEFAULT_CANDIDATE_LOCK.read_text(encoding="utf-8"))

    assert lock["model"]["commit"] == benchmark.EXPECTED_COMMIT
    assert lock["model"]["assetBytes"] == benchmark.EXPECTED_CHECKPOINT_BYTES
    assert lock["model"]["assetSha256"] == benchmark.EXPECTED_CHECKPOINT_SHA256
    assert lock["license"]["declaredSpdx"] == "MIT"
    assert lock["license"]["licenseTextFilePresentAtPinnedTag"] is False
    assert lock["license"]["productionPromotionAllowed"] is False


def test_batch_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        benchmark.run_benchmark(
            checkpoint_path=tmp_path / "checkpoint.pt",
            source_path=tmp_path / "source",
            trial_manifest_path=tmp_path / "trials.json",
            candidate_lock_path=benchmark.DEFAULT_CANDIDATE_LOCK,
            device="cpu",
            batch_size=0,
        )


def test_cuda_measurement_primes_context_before_reset() -> None:
    calls: list[tuple[str, object | None]] = []

    class FakeCuda:
        def get_device_properties(self, target: object) -> None:
            calls.append(("properties", target))

        def empty_cache(self) -> None:
            calls.append(("empty", None))

        def reset_peak_memory_stats(self, target: object) -> None:
            calls.append(("reset", target))

    target = object()
    benchmark._initialize_cuda_measurement(
        SimpleNamespace(cuda=FakeCuda()),
        target,
    )

    assert calls == [
        ("properties", target),
        ("empty", None),
        ("reset", target),
    ]
