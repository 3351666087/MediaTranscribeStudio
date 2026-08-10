from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.persistence import canonical_json_sha256
from tools import evaluate_campplus_trials


class _FakeAdapter:
    created: list["_FakeAdapter"] = []

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        embedding_batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self._model_instance: Any = object()
        self.contexts: list[Any] = []
        self.created.append(self)

    def _embed_slices(
        self, clips: list[Any], context: Any
    ) -> list[tuple[float, ...]]:
        context.raise_if_cancelled()
        self.contexts.append(context)
        return [(float(index), 1.0) for index, _ in enumerate(clips)]

    def release_resources(self) -> None:
        self._model_instance = None


def test_wrapper_uses_production_adapter_and_rebinds_canonical(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _FakeAdapter.created.clear()

    def shared_runner(**kwargs: Any) -> dict[str, Any]:
        verifier = kwargs["verifier_factory"](
            model_path=kwargs["model_path"], device=kwargs["device"]
        )
        assert verifier._embeddings(["a", "b"]) == [
            (0.0, 1.0),
            (1.0, 1.0),
        ]
        verifier.release_resources()
        assert verifier._pipeline_instance is None
        return {
            "schemaVersion": "1.0.0",
            "benchmark": "frozen-speaker-verification-trials",
            "model": {"modelKey": "camplus"},
            "partition": {"evaluationSplit": "development"},
            "execution": {"batchSize": kwargs["batch_size"]},
            "resources": {"snapshots": {}},
            "scores": {"rocAuc": 1.0},
            "canonicalSha256": "stale",
        }

    monkeypatch.setattr(
        evaluate_campplus_trials,
        "_run_shared_evaluation",
        shared_runner,
    )
    report = evaluate_campplus_trials.run_evaluation(
        model_path=tmp_path,
        trial_manifest_path=tmp_path / "trials.json",
        device="cpu",
        batch_size=7,
        adapter_factory=_FakeAdapter,
    )

    assert _FakeAdapter.created[0].embedding_batch_size == 7
    assert report["execution"]["adapterId"] == "CAM++"
    assert report["execution"]["productionRole"] == "primary-voiceprint"
    canonical = dict(report)
    declared = canonical.pop("canonicalSha256")
    assert declared == canonical_json_sha256(canonical)


def test_batch_size_must_be_positive(tmp_path: Path) -> None:
    try:
        evaluate_campplus_trials.run_evaluation(
            model_path=tmp_path,
            trial_manifest_path=tmp_path / "trials.json",
            device="cpu",
            batch_size=0,
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("zero batch size was accepted")
