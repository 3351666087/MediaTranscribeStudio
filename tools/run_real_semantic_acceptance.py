"""Run the mandatory semantic stage against a frozen transcript.

The summary intentionally contains hashes and aggregate metrics only.  It is
safe to retain as audit evidence without copying transcript text into a
secondary report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import resource as _resource
except ImportError:  # pragma: no cover - exercised by the Windows runtime
    _resource = None

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
    parser.add_argument("--model", default="qwen3.5:27b-q4_K_M")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--speaker-top-k", type=int, default=3)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument(
        "--release-on-close",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--replace", action="store_true")
    return parser


def _max_rss_evidence() -> dict[str, Any]:
    if _resource is not None:
        try:
            raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or raw < 0
            ):
                raise ValueError("invalid ru_maxrss")
            # macOS reports ru_maxrss in bytes; Linux reports KiB.
            divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
            return {
                "available": True,
                "source": "resource.getrusage.ru_maxrss",
                "valueMb": round(float(raw) / divisor, 3),
            }
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            raw_peak_bytes = getattr(
                psutil.Process().memory_info(),
                "peak_wset",
                None,
            )
            if (
                isinstance(raw_peak_bytes, bool)
                or not isinstance(raw_peak_bytes, (int, float))
                or not math.isfinite(float(raw_peak_bytes))
                or raw_peak_bytes < 0
            ):
                raise ValueError("invalid peak working set")
            return {
                "available": True,
                "source": "psutil.Process.memory_info.peak_wset",
                "valueMb": round(
                    float(raw_peak_bytes) / (1024 * 1024),
                    3,
                ),
            }
        except (psutil.Error, AttributeError, OSError, TypeError, ValueError):
            pass
    return {
        "available": False,
        "failureCode": "PROCESS_PEAK_RSS_UNAVAILABLE",
    }


def _max_rss_mb() -> float | None:
    evidence = _max_rss_evidence()
    value = evidence.get("valueMb")
    return float(value) if isinstance(value, (int, float)) else None


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
            keep_alive=args.keep_alive,
            release_on_close=args.release_on_close,
        )
    )
    runner = SemanticProcessingRunner(
        provider=provider,
        model=args.model,
        batch_size=args.batch_size,
        speaker_top_k=args.speaker_top_k,
    )
    started = time.monotonic()
    try:
        artifact = runner.run(document)
    except BaseException as primary_error:
        try:
            runner.release_resources()
        except Exception as release_error:
            primary_error.add_note(
                "semantic stage resource release also failed: "
                f"{type(release_error).__name__}"
            )
        raise
    runner.release_resources()
    elapsed = time.monotonic() - started
    after = read_json_strict(transcript)
    after_canonical_sha = canonical_json_sha256(after)
    if after_canonical_sha != source_canonical_sha:
        raise RuntimeError("source transcript changed during semantic run")

    atomic_write_json(artifact_path, artifact)
    max_rss_evidence = _max_rss_evidence()
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
        "maxRssMb": max_rss_evidence.get("valueMb"),
        "maxRssEvidence": max_rss_evidence,
        "execution": {
            "timeoutSeconds": args.timeout_seconds,
            "contextTokens": args.context_tokens,
            "outputTokens": args.output_tokens,
            "batchSize": args.batch_size,
            "speakerTopK": args.speaker_top_k,
            "keepAlive": args.keep_alive,
            "releaseOnClose": args.release_on_close,
        },
        "metrics": artifact["metrics"],
        "sourceUnchanged": after_canonical_sha == source_canonical_sha,
        "transcriptTextPersistedInSummary": False,
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
