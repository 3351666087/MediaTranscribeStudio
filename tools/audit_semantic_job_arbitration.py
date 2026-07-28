#!/usr/bin/env python3
"""Run one real local job-level semantic arbitration audit."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.local_llm import LocalLLMConfig, OllamaLocalProvider
from backend.persistence import (
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
)
from backend.semantic_candidate_lattice import (
    build_semantic_candidate_lattice_from_document,
    validate_semantic_candidate_lattice,
)
from backend.semantic_composition import (
    SemanticJobArbitrationRunner,
    validate_semantic_job_arbitration,
)


class _RecordingProvider:
    def __init__(self, delegate: OllamaLocalProvider) -> None:
        self._delegate = delegate
        self.provider_id = delegate.provider_id
        self.provider_version = delegate.provider_version
        self.network_policy = delegate.network_policy
        self.config = delegate.config
        self.responses: list[dict[str, Any]] = []

    @property
    def generation_metrics(self) -> Mapping[str, int]:
        return self._delegate.generation_metrics

    def generate_json(self, **kwargs: Any) -> Mapping[str, Any]:
        result = dict(self._delegate.generate_json(**kwargs))
        self.responses.append(result)
        return result

    def release_resources(self) -> None:
        self._delegate.release_resources()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--lattice", type=Path)
    parser.add_argument("--carry-lattice", type=Path)
    parser.add_argument("--carry-arbitration", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser


def _summary(
    *,
    document: dict[str, Any],
    lattice: dict[str, Any],
    arbitration: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "artifactType": "semantic-job-arbitration-audit",
        "jobId": document["jobId"],
        "input": {
            "transcriptSha256": canonical_json_sha256(document),
            "sourceMediaSha256": lattice["binding"]["sourceMediaSha256"],
            "latticeSha256": lattice["latticeSha256"],
        },
        "model": arbitration["model"],
        "provider": arbitration["provider"],
        "promptVersion": arbitration["promptVersion"],
        "status": arbitration["status"],
        "metrics": {
            **arbitration["metrics"],
            "elapsedSeconds": round(elapsed_seconds, 6),
        },
        "requestedDomains": sorted(
            {
                request["domain"]
                for request in arbitration["candidateGenerationRequests"]
            }
        ),
        "selectedDomains": sorted(
            {selection["domain"] for selection in arbitration["selections"]}
        ),
        "claimsFinalQualityImprovement": False,
        "conclusion": (
            "candidate-generation-required"
            if arbitration["candidateGenerationRequests"]
            else "ready-for-deterministic-composition"
        ),
        "arbitrationArtifactSha256": canonical_json_sha256(arbitration),
    }


def main() -> int:
    args = _parser().parse_args()
    document = read_json_strict(args.transcript.resolve())
    lattice = (
        build_semantic_candidate_lattice_from_document(document)
        if args.lattice is None
        else validate_semantic_candidate_lattice(
            read_json_strict(args.lattice.resolve()),
            expected_source_media_sha256=document["source"]["sha256"],
            expected_transcript_sha256=canonical_json_sha256(document),
        )
    )
    if (args.carry_lattice is None) != (args.carry_arbitration is None):
        raise ValueError(
            "carry lattice and arbitration must be supplied together"
        )
    carried_lattice = (
        validate_semantic_candidate_lattice(
            read_json_strict(args.carry_lattice.resolve()),
            expected_source_media_sha256=document["source"]["sha256"],
            expected_transcript_sha256=canonical_json_sha256(document),
        )
        if args.carry_lattice is not None
        else None
    )
    carried_arbitration = (
        validate_semantic_job_arbitration(
            read_json_strict(args.carry_arbitration.resolve()),
            expected_job_id=document["jobId"],
            expected_lattice=carried_lattice,
        )
        if args.carry_arbitration is not None
        and carried_lattice is not None
        else None
    )
    provider = _RecordingProvider(
        OllamaLocalProvider(
            LocalLLMConfig(
                model=args.model,
                endpoint=args.endpoint,
                timeout_seconds=args.timeout_seconds,
                context_tokens=args.context_tokens,
                output_tokens=args.output_tokens,
                keep_alive="5m",
                release_on_close=True,
            )
        )
    )
    runner = SemanticJobArbitrationRunner(
        provider=provider,
        model=args.model,
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        batch_size=args.batch_size,
    )
    started = time.monotonic()
    output = args.output_directory.resolve()
    try:
        try:
            arbitration = runner.run(
                document,
                candidate_lattice=lattice,
                carried_lattice=carried_lattice,
                carried_arbitration=carried_arbitration,
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            output.mkdir(parents=True, exist_ok=False)
            atomic_write_json_no_replace(
                output / "semantic-candidate-lattice.v1.json",
                lattice,
            )
            atomic_write_json_no_replace(
                output / "raw-model-responses.v1.json",
                {
                    "schemaVersion": "1.0.0",
                    "responses": provider.responses,
                },
            )
            atomic_write_json_no_replace(
                output / "audit-failure.v1.json",
                {
                    "schemaVersion": "1.0.0",
                    "artifactType": "semantic-job-arbitration-audit-failure",
                    "jobId": document["jobId"],
                    "model": args.model,
                    "latticeSha256": lattice["latticeSha256"],
                    "elapsedSeconds": round(elapsed, 6),
                    "completedProviderCalls": len(provider.responses),
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                    "claimsFinalQualityImprovement": False,
                },
            )
            raise
    finally:
        runner.release_resources()
    elapsed = time.monotonic() - started
    report = _summary(
        document=document,
        lattice=lattice,
        arbitration=arbitration,
        elapsed_seconds=elapsed,
    )
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json_no_replace(
        output / "semantic-candidate-lattice.v1.json",
        lattice,
    )
    atomic_write_json_no_replace(
        output / "semantic-job-arbitration.v1.json",
        arbitration,
    )
    atomic_write_json_no_replace(
        output / "audit-report.v1.json",
        report,
    )
    atomic_write_json_no_replace(
        output / "raw-model-responses.v1.json",
        {
            "schemaVersion": "1.0.0",
            "responses": provider.responses,
        },
    )
    print(
        f"{report['status']}: "
        f"{report['metrics']['selectedGroupCount']} selected, "
        f"{report['metrics']['candidateGenerationRequestCount']} requests, "
        f"{report['metrics']['elapsedSeconds']} seconds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
