"""Render and verify an acceptance-only Java report from a real transcript.

This tool never grants human or linguistic approval. It binds immutable input
hashes, runs the production Python-to-Java adapter, verifies the Java manifest
and PDFBox inspection, and records unresolved review items in a separate
acceptance artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.adapters import AdapterContext, JavaPdfRendererAdapter  # noqa: E402
from backend.models import SpeakerCountPolicy, StartJobRequest  # noqa: E402
from backend.persistence import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.review import validate_review_state  # noqa: E402
from reporting.java_pdf_client import JavaPdfClient  # noqa: E402
from reporting.report_document_assembler import ReportDocumentAssembler  # noqa: E402


EXPECTED_HARD_GATES = 13
EXPECTED_FACETS = 14


def _sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return normalized


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _probe_duration_ms(ffprobe: Path, source: Path) -> int:
    result = subprocess.run(  # noqa: S603 - fixed argv local tool boundary
        (
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise RuntimeError(f"FFprobe failed: {detail}")
    payload = json.loads(result.stdout)
    try:
        duration_ms = round(float(payload["format"]["duration"]) * 1_000)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("FFprobe did not return a finite media duration") from exc
    if duration_ms < 1:
        raise RuntimeError("source media duration must be positive")
    return duration_ms


def _assert_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _manifest_artifacts(output: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise RuntimeError("Java manifest has no artifacts")
    verified: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Java manifest artifact {index} is malformed")
        relative = raw.get("relativePath")
        expected_hash = raw.get("sha256")
        expected_bytes = raw.get("bytes")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"Java manifest artifact {index} has no relativePath")
        path = (output / relative).resolve()
        try:
            path.relative_to(output)
        except ValueError as exc:
            raise RuntimeError("Java manifest artifact escapes output root") from exc
        if not path.is_file():
            raise RuntimeError(f"Java manifest artifact is missing: {relative}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Java manifest artifact changed: {relative}")
        verified.append(
            {
                "artifactId": raw.get("artifactId"),
                "type": raw.get("type"),
                "path": str(path),
                "sizeBytes": expected_bytes,
                "sha256": observed_hash,
            }
        )
    return verified


def _require_true(value: Mapping[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if value.get(key) is not True:
            raise RuntimeError(f"PDF inspection failed: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-source-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-transcript-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-review-queue-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-open-review-items", type=_positive_int, required=True)
    parser.add_argument("--minimum-score", type=float, default=85.0)
    parser.add_argument("--preferred-font", default="LXGW WenKai")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve(strict=True)
    transcript_path = args.transcript.expanduser().resolve(strict=True)
    review_path = args.review_queue.expanduser().resolve(strict=True)
    jar = args.jar.expanduser().resolve(strict=True)
    java = args.java.expanduser().resolve(strict=True)
    ffprobe = args.ffprobe.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    allowed_output_root = (PROJECT_ROOT / ".runtime_cache" / "outputs").resolve()
    try:
        output.relative_to(allowed_output_root)
    except ValueError as exc:
        raise RuntimeError(
            "output must be a child of the repository .runtime_cache/outputs"
        ) from exc
    if output == allowed_output_root:
        raise RuntimeError("output must be a specific acceptance subdirectory")
    if output.exists():
        if not args.replace:
            raise RuntimeError("output exists; pass --replace to regenerate it")
        shutil.rmtree(output)

    source_hash = _assert_hash(source, args.expected_source_sha256, "source")
    transcript_hash = _assert_hash(
        transcript_path, args.expected_transcript_sha256, "transcript"
    )
    review_hash = _assert_hash(
        review_path, args.expected_review_queue_sha256, "review queue"
    )
    transcript = read_json_strict(transcript_path)
    review_queue = read_json_strict(review_path)
    input_job_id = str(transcript.get("jobId") or "")
    transcript, review_queue = validate_review_state(
        transcript,
        review_queue,
        expected_job_id=input_job_id,
        high_margin_threshold=0.35,
    )
    open_review_items = sum(
        1
        for item in review_queue.get("items", [])
        if isinstance(item, Mapping) and item.get("status") == "open"
    )
    if open_review_items != args.expected_open_review_items:
        raise RuntimeError(
            "open review count mismatch: "
            f"expected {args.expected_open_review_items}, observed {open_review_items}"
        )
    transcript_duration_ms = transcript["source"]["durationMs"]
    media_duration_ms = _probe_duration_ms(ffprobe, source)
    if abs(media_duration_ms - transcript_duration_ms) > 100:
        raise RuntimeError(
            "source and transcript durations differ by more than 100 ms: "
            f"{media_duration_ms} vs {transcript_duration_ms}"
        )

    java_client = JavaPdfClient.from_jar(
        jar,
        java_executable=java,
        allowed_output_root=output.parent,
        minimum_score=args.minimum_score,
        preferred_font=args.preferred_font,
        capture_dpi=144,
        timeout_seconds=300,
    )
    adapter = JavaPdfRendererAdapter(
        assembler=ReportDocumentAssembler(pipeline_version="real-report-acceptance-v1"),
        java_client=java_client,
    )
    request = StartJobRequest(
        job_id=args.job_id,
        source_path=source,
        output_directory=output,
        speaker_policy=SpeakerCountPolicy.from_payload(
            {"speakerCountMode": "auto"}
        ),
        render_pdf=True,
        language=str(transcript.get("language") or "und"),
    )
    result = adapter.render(
        transcript,
        request,
        AdapterContext(
            job_id=args.job_id,
            output_directory=output,
            cancellation=threading.Event(),
        ),
    )
    result.validate()

    report_document_path = output / "input" / "report-document.json"
    xhtml_path = output / "render" / "report.xhtml"
    manifest_path = output / "artifacts" / "manifest.json"
    quality_path = output / "artifacts" / "quality-report.json"
    inspection_path = output / "artifacts" / "pdf-inspection.json"
    repair_path = output / "artifacts" / "repair-queue.json"
    report_document = read_json_strict(report_document_path)
    manifest = read_json_strict(manifest_path)
    quality = read_json_strict(quality_path)
    inspection = read_json_strict(inspection_path)
    repair = read_json_strict(repair_path)
    xhtml = xhtml_path.read_text(encoding="utf-8")

    report_review_count = sum(
        1
        for segment in report_document["segments"]
        if segment.get("reviewStatus") == "review-required"
    )
    unavailable_confidence_count = sum(
        1
        for segment in report_document["segments"]
        if segment.get("evidence", {}).get("asr", {}).get("confidenceAvailable")
        is False
    )
    if report_review_count < 1 or unavailable_confidence_count < 1:
        raise RuntimeError("acceptance input did not preserve review/confidence blockers")
    if "VERIFIED TRANSCRIPT" in xhtml:
        raise RuntimeError("unapproved XHTML falsely claims a verified transcript")
    if "未审核 · 需要复核" not in xhtml and "UNAPPROVED · REVIEW REQUIRED" not in xhtml:
        raise RuntimeError("unapproved XHTML lacks a visible review-required label")
    unavailable_label_count = xhtml.count("ASR 置信度不可用") + xhtml.count(
        "ASR confidence unavailable"
    )
    if unavailable_label_count != unavailable_confidence_count:
        raise RuntimeError("XHTML confidence availability labels do not match evidence")
    if any(
        "{" in str(segment["evidence"]["asr"]["provider"])
        for segment in report_document["segments"]
    ):
        raise RuntimeError("structured ASR provider leaked into report as dictionary text")

    hard_gates = quality.get("hardGates")
    facets = quality.get("facets")
    if (
        quality.get("status") != "passed"
        or quality.get("hardGatesPassed") is not True
        or not isinstance(hard_gates, list)
        or len(hard_gates) != EXPECTED_HARD_GATES
        or not isinstance(facets, list)
        or len(facets) != EXPECTED_FACETS
    ):
        raise RuntimeError("Java quality report did not pass the complete gate set")
    repair_items = repair.get("repairs")
    if not isinstance(repair_items, list) or repair_items:
        raise RuntimeError("Java repair queue is not empty")
    _require_true(
        inspection,
        (
            "allFontsEmbedded",
            "searchableText",
            "transcriptTextIntegrity",
            "segmentCountIntegrity",
            "speakerSetIntegrity",
            "timestampIntegrity",
            "noReplacementCharacters",
        ),
    )
    if not inspection.get("pages") or not all(
        page.get("a4") is True for page in inspection["pages"]
    ):
        raise RuntimeError("PDF inspection found a non-A4 or missing page")
    artifacts = _manifest_artifacts(output, manifest)

    acceptance = {
        "schemaVersion": "1.0.0",
        "artifactType": "unapproved-java-report-acceptance",
        "applicationPolicy": "suggestion-only",
        "requiresHumanApproval": True,
        "releaseApproved": False,
        "source": {
            "mediaPath": str(source),
            "mediaSha256": source_hash,
            "mediaDurationMs": media_duration_ms,
            "transcriptPath": str(transcript_path),
            "transcriptSha256": transcript_hash,
            "transcriptCanonicalSha256": canonical_json_sha256(transcript),
            "reviewQueuePath": str(review_path),
            "reviewQueueSha256": review_hash,
            "openReviewItems": open_review_items,
        },
        "renderer": {
            "provider": "java-openhtmltopdf-pdfbox",
            "rendererVersion": result.renderer_version,
            "javaManifestSha256": sha256_file(manifest_path),
        },
        "truthfulPresentation": {
            "visibleReviewRequiredLabel": True,
            "forbiddenVerifiedTranscriptLabelAbsent": True,
            "reportReviewRequiredSegments": report_review_count,
            "asrConfidenceUnavailableSegments": unavailable_confidence_count,
            "asrConfidenceUnavailableLabels": unavailable_label_count,
            "structuredProviderIdsPreserved": True,
        },
        "quality": {
            "status": quality["status"],
            "score": quality["score"],
            "minimumScore": quality["minimumScore"],
            "hardGatesPassed": True,
            "hardGateCount": len(hard_gates),
            "facetCount": len(facets),
            "repairCount": 0,
        },
        "inspection": {
            "path": str(inspection_path),
            "sha256": sha256_file(inspection_path),
            "validator": inspection["validator"],
            "validatorVersion": inspection["validatorVersion"],
            "pageCount": inspection["pageCount"],
            "allPagesA4": True,
            "allFontsEmbedded": True,
            "searchableText": True,
            "transcriptTextIntegrity": True,
            "segmentCountIntegrity": True,
            "speakerSetIntegrity": True,
            "timestampIntegrity": True,
            "noReplacementCharacters": True,
        },
        "artifacts": artifacts,
        "qualityBoundary": {
            "technicalReportPassed": True,
            "linguisticQualityPassed": False,
            "speakerQualityPassed": False,
            "reason": "OPEN_REVIEW_ITEMS_AND_NO_REFERENCE_TRUTH",
        },
    }
    acceptance_path = output / "artifacts" / "unapproved-report-acceptance.v1.json"
    atomic_write_json(acceptance_path, acceptance)
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(output),
                "acceptanceSha256": sha256_file(acceptance_path),
                "qualityScore": quality["score"],
                "pageCount": inspection["pageCount"],
                "openReviewItems": open_review_items,
                "releaseApproved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
