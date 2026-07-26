"""Run the mandatory semantic stage against a frozen transcript.

The summary intentionally contains hashes and aggregate metrics only.  It is
safe to retain as audit evidence without copying transcript text into a
secondary report.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Sequence

# Keep direct ``python tools/run_real_semantic_acceptance.py`` invocation
# consistent with the other repository acceptance tools.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import LocalLLMConfig, OllamaLocalProvider, SemanticProcessingRunner
from backend.persistence import (
    atomic_write_json,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--speaker-top-k", type=int, default=3)
    parser.add_argument("--replace", action="store_true")
    return parser


def _max_rss_mb() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 3)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transcript = args.transcript.expanduser().resolve()
    output = args.output.expanduser().resolve()
    artifact_path = output / "semantic" / "semantic-suggestions.v1.json"
    summary_path = output / "semantic-acceptance.v1.json"
    if not args.replace and (artifact_path.exists() or summary_path.exists()):
        raise RuntimeError("refusing to overwrite existing semantic evidence")

    document = read_json_strict(transcript)
    source_canonical_sha = canonical_json_sha256(document)
    source_file_sha = sha256_file(transcript)
    provider = OllamaLocalProvider(
        LocalLLMConfig(
            model=args.model,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            context_tokens=args.context_tokens,
            output_tokens=args.output_tokens,
        )
    )
    started = time.monotonic()
    artifact = SemanticProcessingRunner(
        provider=provider,
        model=args.model,
        batch_size=args.batch_size,
        speaker_top_k=args.speaker_top_k,
    ).run(document)
    elapsed = time.monotonic() - started
    after = read_json_strict(transcript)
    after_canonical_sha = canonical_json_sha256(after)
    if after_canonical_sha != source_canonical_sha:
        raise RuntimeError("source transcript changed during semantic run")

    atomic_write_json(artifact_path, artifact)
    summary = {
        "schemaVersion": "1.0.0",
        "status": artifact["status"],
        "model": artifact["model"],
        "provider": artifact["provider"],
        "source": {
            "path": str(transcript),
            "fileSha256": source_file_sha,
            "canonicalSha256": source_canonical_sha,
            "segmentCount": len(document.get("segments", [])),
        },
        "artifact": {
            "path": str(artifact_path),
            "canonicalSha256": canonical_json_sha256(artifact),
            "fileSha256": sha256_file(artifact_path),
        },
        "elapsedSeconds": round(elapsed, 3),
        "maxRssMb": _max_rss_mb(),
        "metrics": artifact["metrics"],
        "sourceUnchanged": after_canonical_sha == source_canonical_sha,
        "transcriptTextPersistedInSummary": False,
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
