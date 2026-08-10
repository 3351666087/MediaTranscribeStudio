"""Evaluate production CAM++ on the shared frozen verification trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256  # noqa: E402
from backend.production_runners import LocalFunAsrCamPlusAdapter  # noqa: E402
from tools.benchmark_eres2netv2_residency import (  # noqa: E402
    _reset_cuda_peaks,
    _resource_snapshot,
)
from tools.evaluate_eres2netv2_trials import (  # noqa: E402
    _write_report,
    run_evaluation as _run_shared_evaluation,
)


class _BenchmarkContext:
    def raise_if_cancelled(self) -> None:
        return None


class _CamPlusTrialVerifier:
    """Expose the production CAM++ embedding path through the shared runner."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        embedding_batch_size: int,
        adapter_factory: Callable[..., Any] = LocalFunAsrCamPlusAdapter,
    ) -> None:
        self._adapter = adapter_factory(
            model_path=model_path,
            device=device,
            embedding_batch_size=embedding_batch_size,
        )
        self._context = _BenchmarkContext()

    @property
    def _pipeline_instance(self) -> Any:
        return self._adapter._model_instance

    def _embeddings(self, clips: Sequence[Any]) -> list[tuple[float, ...]]:
        return self._adapter._embed_slices(clips, self._context)

    def release_resources(self) -> None:
        self._adapter.release_resources()


def run_evaluation(
    *,
    model_path: Path,
    trial_manifest_path: Path,
    device: str,
    batch_size: int,
    adapter_factory: Callable[..., Any] = LocalFunAsrCamPlusAdapter,
    resource_probe: Callable[[], Mapping[str, Any]] = _resource_snapshot,
    reset_resource_peaks: Callable[[], None] = _reset_cuda_peaks,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    def verifier_factory(*, model_path: Path, device: str) -> _CamPlusTrialVerifier:
        return _CamPlusTrialVerifier(
            model_path=model_path,
            device=device,
            embedding_batch_size=batch_size,
            adapter_factory=adapter_factory,
        )

    report = _run_shared_evaluation(
        model_path=model_path,
        trial_manifest_path=trial_manifest_path,
        device=device,
        batch_size=batch_size,
        verifier_factory=verifier_factory,
        resource_probe=resource_probe,
        reset_resource_peaks=reset_resource_peaks,
    )
    report.pop("canonicalSha256", None)
    report["execution"]["adapterId"] = "CAM++"
    report["execution"]["productionRole"] = "primary-voiceprint"
    report["canonicalSha256"] = canonical_json_sha256(report)
    return report


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=_positive_integer, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_evaluation(
        model_path=args.model_path,
        trial_manifest_path=args.trial_manifest,
        device=args.device,
        batch_size=args.batch_size,
    )
    _write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonicalSha256": report["canonicalSha256"],
                "modelKey": report["model"]["modelKey"],
                "evaluationSplit": report["partition"]["evaluationSplit"],
                "scores": report["scores"],
                "execution": report["execution"],
                "resources": {
                    key: value
                    for key, value in report["resources"].items()
                    if key != "snapshots"
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
