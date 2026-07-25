"""Run and verify local business suggestions from an unapproved transcript.

This tool deliberately does not bypass review gates. It binds the transcript
and review queue by hash, runs the loopback-only Ollama business pipeline, and
records translation, polish, and summary outputs as suggestions that still
require human approval.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.business_contracts import validate_business_output_contract  # noqa: E402
from backend.business_processing import (  # noqa: E402
    BusinessProcessingConfig,
    BusinessProcessingRunner,
)
from backend.local_llm import LocalLLMConfig, OllamaLocalProvider  # noqa: E402
from backend.persistence import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.review import validate_review_state  # noqa: E402


def _sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number


def _assert_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _safe_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    allowed_root = (PROJECT_ROOT / ".runtime_cache" / "outputs").resolve()
    try:
        output.relative_to(allowed_root)
    except ValueError as exc:
        raise RuntimeError(
            "output must be a child of the repository .runtime_cache/outputs"
        ) from exc
    if output == allowed_root:
        raise RuntimeError("output must be a specific acceptance subdirectory")
    return output


def validate_business_artifacts(
    document: Mapping[str, Any],
    artifacts: Sequence[Path],
    *,
    config: BusinessProcessingConfig,
) -> dict[str, Any]:
    """Validate public contracts, approval policy, and immutable provenance."""

    expected_variants = {
        *(f"translation:{target}" for target in config.translation_targets),
        *(("polish",) if config.polish else ()),
        *(("summary",) if config.summary else ()),
    }
    manifest_paths = [path for path in artifacts if path.name == "business-manifest.v1.json"]
    if len(manifest_paths) != 1:
        raise RuntimeError("business processing must produce exactly one manifest")

    source_document_hash = canonical_json_sha256(document)
    artifact_records: list[dict[str, Any]] = []
    observed_variants: set[str] = set()
    variant_metrics: dict[str, Any] = {}
    for path in artifacts:
        if not path.is_file():
            raise RuntimeError(f"business artifact is missing: {path}")
        value = read_json_strict(path)
        if value.get("applicationPolicy") != "suggestion-only":
            raise RuntimeError(f"business artifact is not suggestion-only: {path.name}")
        if value.get("requiresHumanApproval") is not True:
            raise RuntimeError(f"business artifact lacks human approval gate: {path.name}")

        record = {
            "path": str(path.resolve()),
            "sizeBytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "canonicalSha256": canonical_json_sha256(value),
        }
        if path == manifest_paths[0]:
            if value.get("sourceDocumentHash") != source_document_hash:
                raise RuntimeError("business manifest source hash does not match transcript")
            if value.get("rawTranscriptImmutable") is not True:
                raise RuntimeError("business manifest does not preserve the raw transcript")
            completeness = value.get("completeness")
            if (
                not isinstance(completeness, Mapping)
                or completeness.get("allRequestedTasksCompleted") is not True
            ):
                raise RuntimeError("business manifest is incomplete")
            record["kind"] = "manifest"
        else:
            variant = value.get("variant")
            if not isinstance(variant, str) or variant not in expected_variants:
                raise RuntimeError(f"unexpected business variant in {path.name}")
            if variant in observed_variants:
                raise RuntimeError(f"duplicate business variant: {variant}")
            validate_business_output_contract(value, variant=variant)
            observed_variants.add(variant)
            record["kind"] = "business-suggestion"
            record["variant"] = variant
            if variant.startswith("translation:"):
                segments = value.get("segments")
                variant_metrics[variant] = {
                    "status": value.get("status"),
                    "segmentCount": len(segments) if isinstance(segments, list) else 0,
                    "targetLanguage": value.get("targetLanguage"),
                }
            elif variant == "polish":
                variant_metrics[variant] = {
                    "status": value.get("status"),
                    "segmentCount": len(value.get("segments", [])),
                    "diffCount": len(value.get("diff", [])),
                }
            else:
                evidence_ids = {
                    segment_id
                    for field in ("keyPoints", "topics", "actionItems")
                    for item in value.get(field, [])
                    if isinstance(item, Mapping)
                    for segment_id in item.get("evidenceSegmentIds", [])
                    if isinstance(segment_id, str)
                }
                variant_metrics[variant] = {
                    "status": value.get("status"),
                    "evidenceSegmentCount": len(evidence_ids),
                    "executiveSummaryNonempty": bool(
                        str(value.get("executiveSummary") or "").strip()
                    ),
                }
        artifact_records.append(record)

    if observed_variants != expected_variants:
        missing = sorted(expected_variants - observed_variants)
        extra = sorted(observed_variants - expected_variants)
        raise RuntimeError(f"business variants mismatch: missing={missing}, extra={extra}")
    return {
        "sourceDocumentCanonicalSha256": source_document_hash,
        "artifacts": artifact_records,
        "variants": variant_metrics,
    }


def run_unapproved_business_acceptance(
    *,
    transcript_path: Path,
    review_queue_path: Path,
    output_directory: Path,
    expected_transcript_sha256: str,
    expected_review_queue_sha256: str,
    expected_open_review_items: int,
    config: BusinessProcessingConfig,
    provider: OllamaLocalProvider,
    replace: bool,
    resume: bool,
) -> Path:
    transcript_path = transcript_path.expanduser().resolve(strict=True)
    review_queue_path = review_queue_path.expanduser().resolve(strict=True)
    output_directory = _safe_output(output_directory)
    if output_directory.exists():
        if replace:
            shutil.rmtree(output_directory)
        elif not resume:
            raise RuntimeError(
                "output exists; pass --resume to reuse checkpoints or --replace "
                "to regenerate it"
            )

    transcript_sha256 = _assert_hash(
        transcript_path, expected_transcript_sha256, "transcript"
    )
    review_queue_sha256 = _assert_hash(
        review_queue_path, expected_review_queue_sha256, "review queue"
    )
    document = read_json_strict(transcript_path)
    review_queue = read_json_strict(review_queue_path)
    job_id = str(document.get("jobId") or "")
    document, review_queue = validate_review_state(
        document,
        review_queue,
        expected_job_id=job_id,
        high_margin_threshold=0.35,
    )
    open_review_items = sum(
        1
        for item in review_queue.get("items", [])
        if isinstance(item, Mapping) and item.get("status") == "open"
    )
    if open_review_items != expected_open_review_items:
        raise RuntimeError(
            "open review count mismatch: "
            f"expected {expected_open_review_items}, observed {open_review_items}"
        )

    canonical_before = canonical_json_sha256(document)
    artifacts = BusinessProcessingRunner(provider=provider).run(
        document,
        output_directory=output_directory,
        config=config,
    )
    if canonical_json_sha256(document) != canonical_before:
        raise RuntimeError("business processing mutated the transcript document")
    if sha256_file(transcript_path) != transcript_sha256:
        raise RuntimeError("persisted transcript changed during business processing")
    verified = validate_business_artifacts(document, artifacts, config=config)

    acceptance = {
        "schemaVersion": "1.0.0",
        "artifactType": "unapproved-business-acceptance",
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
        "releaseApproved": False,
        "source": {
            "jobId": job_id,
            "transcriptPath": str(transcript_path),
            "transcriptSha256": transcript_sha256,
            "transcriptCanonicalSha256": verified[
                "sourceDocumentCanonicalSha256"
            ],
            "reviewQueuePath": str(review_queue_path),
            "reviewQueueSha256": review_queue_sha256,
            "openReviewItems": open_review_items,
        },
        "config": config.as_dict(),
        "artifacts": verified["artifacts"],
        "variants": verified["variants"],
        "qualityBoundary": {
            "technicalBusinessProcessingPassed": True,
            "linguisticQualityPassed": False,
            "speakerQualityPassed": False,
            "releaseApproved": False,
            "reason": "UNAPPROVED_SOURCE_AND_OPEN_REVIEW_ITEMS",
        },
    }
    acceptance_path = output_directory / "business" / "unapproved-business-acceptance.v1.json"
    atomic_write_json(acceptance_path, acceptance)
    return acceptance_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-transcript-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-review-queue-sha256", type=_sha256, required=True)
    parser.add_argument(
        "--expected-open-review-items", type=_nonnegative_int, required=True
    )
    parser.add_argument("--translation-target", action="append", default=[])
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--output-locale", default="zh-CN")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.replace and args.resume:
        raise RuntimeError("--replace and --resume are mutually exclusive")
    config = BusinessProcessingConfig(
        translation_targets=tuple(args.translation_target),
        polish=args.polish,
        summary=args.summary,
        model=args.model,
        output_locale=args.output_locale,
    )
    if not config.enabled:
        raise RuntimeError("at least one translation, polish, or summary task is required")
    provider = OllamaLocalProvider(
        LocalLLMConfig(
            model=args.model,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
            context_tokens=args.context_tokens,
            output_tokens=args.output_tokens,
        )
    )
    acceptance_path = run_unapproved_business_acceptance(
        transcript_path=args.transcript,
        review_queue_path=args.review_queue,
        output_directory=args.output,
        expected_transcript_sha256=args.expected_transcript_sha256,
        expected_review_queue_sha256=args.expected_review_queue_sha256,
        expected_open_review_items=args.expected_open_review_items,
        config=config,
        provider=provider,
        replace=args.replace,
        resume=args.resume,
    )
    acceptance = read_json_strict(acceptance_path)
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(acceptance_path.parent.parent),
                "acceptanceSha256": sha256_file(acceptance_path),
                "openReviewItems": acceptance["source"]["openReviewItems"],
                "releaseApproved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
