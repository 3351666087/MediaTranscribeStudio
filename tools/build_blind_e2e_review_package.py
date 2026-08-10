"""Build a reproducible blind review package from completed product runs.

The reviewer packet contains opaque case and option identifiers, frozen media,
whitelisted transcript evidence, and optional subtitle/PDF output. Original
paths, run labels, model identities, and the ordering seed are kept in a
separate identity vault. Reference answers and automatic scores are never
accepted, read, or copied by this tool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.errors import WorkerError  # noqa: E402
from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)


SCHEMA_VERSION = "1.0.0"
ORDERING_ALGORITHM = "sha256-seeded-per-case-order-v1"
PDF_SCAN_SCHEMA_VERSION = "1.0.0"
PDF_SCAN_ARTIFACT_TYPE = "blind-review-pdf-identity-scan"
PDF_SCAN_VALIDATOR = "PDFBox"
PDF_SCAN_VALIDATOR_VERSION = "2.0.30"
DEFAULT_PDF_SCANNER_JAR = PROJECT_ROOT / "pdf-renderer" / "target" / "pdf-renderer.jar"
BLIND_PDF_SANITIZATION_PROFILE = "report-title-and-source-file-v1"
BLIND_PDF_TITLE = "Blind Review Transcript"
_RESERVED_BLIND_IDENTITIES = frozenset({"production", "challenger"})
_PDF_SCAN_TIMEOUT_SECONDS = 180
_MAX_PDF_SCAN_OUTPUT_BYTES = 1024 * 1024
_SUBTITLE_SUFFIXES = frozenset({".srt", ".vtt", ".webvtt", ".ass"})
_PDF_SUFFIXES = frozenset({".pdf"})
_MEDIA_SUFFIXES = frozenset(
    {
        ".aac",
        ".flac",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)
_DISPOSITIONS = frozenset(
    {"no-transcribable-speech", "transcribable-speech"}
)


class BlindE2EReviewPackageError(ValueError):
    """Raised when completed run evidence cannot be frozen safely."""


@dataclass(frozen=True)
class CandidateRun:
    candidate_id: str
    run_root: Path


@dataclass(frozen=True)
class _RunLayout:
    candidate_id: str
    run_root: Path
    outputs_root: Path
    results_root: Path | None


@dataclass(frozen=True)
class _PdfScanner:
    java_executable: str
    jar_path: Path
    jar_bytes: int
    jar_sha256: str


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = read_json_strict(path)
    except (OSError, UnicodeError, ValueError, WorkerError) as exc:
        raise BlindE2EReviewPackageError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise BlindE2EReviewPackageError(f"{label} must be an object: {path}")
    return value


def _resolve_layout(candidate: CandidateRun) -> _RunLayout:
    candidate_id = candidate.candidate_id.strip()
    if not candidate_id or len(candidate_id) > 200 or any(
        ord(character) < 32 for character in candidate_id
    ):
        raise BlindE2EReviewPackageError("candidate IDs must be non-empty text")
    root = candidate.run_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise BlindE2EReviewPackageError(
            f"candidate run root is not a directory: {root}"
        )
    nested_outputs = root / "outputs"
    if nested_outputs.is_dir():
        outputs_root = nested_outputs.resolve(strict=True)
        results = root / "results"
        results_root = results.resolve(strict=True) if results.is_dir() else None
        run_root = root
    else:
        outputs_root = root
        run_root = root.parent if root.name.casefold() == "outputs" else root
        results = run_root / "results"
        results_root = results.resolve(strict=True) if results.is_dir() else None
    return _RunLayout(
        candidate_id=candidate_id,
        run_root=run_root,
        outputs_root=outputs_root,
        results_root=results_root,
    )


def _available_case_ids(layout: _RunLayout) -> set[str]:
    return {
        item.name
        for item in layout.outputs_root.iterdir()
        if item.is_dir() and (item / "checkpoint.v2.json").is_file()
    }


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BlindE2EReviewPackageError(f"{label} must be an integer >= {minimum}")
    return value


def _nonempty_text(
    value: Any,
    *,
    label: str,
    preserve: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlindE2EReviewPackageError(f"{label} must be non-empty text")
    return value if preserve else value.strip()


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise BlindE2EReviewPackageError(f"{label} must be a boolean")
    return value


def _segment_views(
    value: Any,
    *,
    text_field: str,
    label: str,
    allow_empty: bool,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise BlindE2EReviewPackageError(f"{label} must be a segment array")
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        field = f"{label}[{index}]"
        if not isinstance(raw, Mapping):
            raise BlindE2EReviewPackageError(f"{field} must be an object")
        start_ms = _positive_int(
            raw.get("startMs"),
            label=f"{field}.startMs",
            allow_zero=True,
        )
        end_ms = _positive_int(raw.get("endMs"), label=f"{field}.endMs")
        if end_ms <= start_ms:
            raise BlindE2EReviewPackageError(f"{field} has an invalid time range")
        text = _nonempty_text(
            raw.get(text_field),
            label=f"{field}.{text_field}",
            preserve=True,
        )
        item = {
            "segmentId": _nonempty_text(raw.get("id"), label=f"{field}.id"),
            "startMs": start_ms,
            "endMs": end_ms,
            "speakerId": _nonempty_text(
                raw.get("speakerId"),
                label=f"{field}.speakerId",
            ),
            "language": _nonempty_text(
                raw.get("language"),
                label=f"{field}.language",
            ),
            "text": text,
            "overlapping": _boolean(
                raw.get("overlapping", False),
                label=f"{field}.overlapping",
            ),
        }
        output.append(item)
    return output


def _timeline_view(
    value: Any,
    *,
    final_segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw_turns = value.get("turns")
        if not isinstance(raw_turns, list):
            raise BlindE2EReviewPackageError("final timeline turns must be an array")
        turns: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_turns):
            field = f"final.timeline.turns[{index}]"
            if not isinstance(raw, Mapping):
                raise BlindE2EReviewPackageError(f"{field} must be an object")
            start_ms = _positive_int(
                raw.get("startMs"),
                label=f"{field}.startMs",
                allow_zero=True,
            )
            end_ms = _positive_int(raw.get("endMs"), label=f"{field}.endMs")
            if end_ms <= start_ms:
                raise BlindE2EReviewPackageError(f"{field} has an invalid time range")
            turns.append(
                {
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "speakerId": _nonempty_text(
                        raw.get("speakerId"),
                        label=f"{field}.speakerId",
                    ),
                    "overlap": _boolean(
                        raw.get("overlap", False),
                        label=f"{field}.overlap",
                    ),
                }
            )
        raw_speaker_ids = value.get("speakerIds")
        if not isinstance(raw_speaker_ids, list):
            raise BlindE2EReviewPackageError("final timeline speakerIds are invalid")
        speaker_ids = [
            _nonempty_text(item, label="final.timeline.speakerIds[]")
            for item in raw_speaker_ids
        ]
        if len(set(speaker_ids)) != len(speaker_ids):
            raise BlindE2EReviewPackageError(
                "final timeline speakerIds must be unique"
            )
        speaker_count = _positive_int(
            value.get("speakerCount"),
            label="final.timeline.speakerCount",
            allow_zero=True,
        )
        if speaker_count != len(speaker_ids):
            raise BlindE2EReviewPackageError(
                "final timeline speakerCount does not match speakerIds"
            )
        known_speakers = set(speaker_ids)
        if any(turn["speakerId"] not in known_speakers for turn in turns):
            raise BlindE2EReviewPackageError(
                "final timeline turn refers to an unknown speaker"
            )
        return {
            "speakerCount": speaker_count,
            "speakerIds": speaker_ids,
            "turns": turns,
        }
    speaker_ids = sorted({str(item["speakerId"]) for item in final_segments})
    if not speaker_ids:
        return {"speakerCount": 0, "speakerIds": [], "turns": []}
    return {
        "speakerCount": len(speaker_ids),
        "speakerIds": speaker_ids,
        "turns": [
            {
                "startMs": int(item["startMs"]),
                "endMs": int(item["endMs"]),
                "speakerId": str(item["speakerId"]),
                "overlap": bool(item["overlapping"]),
            }
            for item in final_segments
        ],
    }


def _file_evidence(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise BlindE2EReviewPackageError(f"artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _json_snapshot(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _file_evidence(path)
    value = _json_object(path, label=label)
    after = _file_evidence(path)
    if before != after:
        raise BlindE2EReviewPackageError(
            f"{label} changed while being scanned: {path}"
        )
    return value, {
        **after,
        "canonicalSha256": canonical_json_sha256(value),
    }


def _artifact_files(
    case_root: Path,
    checkpoint: Mapping[str, Any],
) -> tuple[list[Path], list[Path]]:
    declared_paths: set[str] = set()

    def add_path(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            declared_paths.add(value.strip())

    artifact_paths = checkpoint.get("artifactPaths")
    if isinstance(artifact_paths, list):
        for value in artifact_paths:
            add_path(value)
    transcript_exports = checkpoint.get("transcriptExports")
    if isinstance(transcript_exports, list):
        for receipt in transcript_exports:
            if isinstance(receipt, Mapping):
                add_path(receipt.get("path"))
    publication = checkpoint.get("outputPublication")
    if isinstance(publication, Mapping):
        customer_artifacts = publication.get("customerArtifacts")
        if isinstance(customer_artifacts, list):
            for receipt in customer_artifacts:
                if isinstance(receipt, Mapping):
                    add_path(receipt.get("path"))

    subtitles: list[Path] = []
    pdfs: list[Path] = []
    for value in sorted(declared_paths):
        path = Path(value).expanduser()
        suffix = path.suffix.casefold()
        if suffix not in _SUBTITLE_SUFFIXES | _PDF_SUFFIXES:
            continue
        if not path.is_absolute():
            path = case_root / path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(case_root)
        except (OSError, ValueError) as exc:
            raise BlindE2EReviewPackageError(
                f"declared reviewer artifact is missing or outside its case: {path}"
            ) from exc
        if not resolved.is_file():
            raise BlindE2EReviewPackageError(
                f"declared reviewer artifact is not a file: {resolved}"
            )
        if suffix in _SUBTITLE_SUFFIXES:
            subtitles.append(resolved)
        else:
            pdfs.append(resolved)
    return sorted(set(subtitles)), sorted(set(pdfs))


def _model_identity_values(value: Any) -> list[str]:
    identities: set[str] = set()

    def normalized_key(value: Any) -> str:
        return "".join(
            character
            for character in str(value).casefold()
            if character.isalnum()
        )

    def visit(item: Any, *, model_context: bool = False) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                key_name = normalized_key(key)
                direct_identity = key_name in {
                    "model",
                    "modeldigest",
                    "modelid",
                    "modelname",
                    "modelpath",
                }
                contextual_identity = model_context and key_name in {
                    "digest",
                    "id",
                    "name",
                    "path",
                }
                if direct_identity or contextual_identity:
                    if isinstance(nested, str) and nested.strip():
                        identities.add(nested.strip())
                visit(
                    nested,
                    model_context=(
                        model_context
                        or key_name
                        in {"modelmanifest", "modelregistry", "models"}
                    ),
                )
        elif isinstance(item, list):
            for nested in item:
                visit(nested, model_context=model_context)

    visit(value)
    return sorted(identities)


def _scan_case(layout: _RunLayout, case_id: str) -> dict[str, Any]:
    case_root = (layout.outputs_root / case_id).resolve(strict=True)
    checkpoint_path = case_root / "checkpoint.v2.json"
    transcript_path = case_root / "transcript-document.v2.json"
    final_path = case_root / "final-adjudicated-transcript.v1.json"
    checkpoint, checkpoint_evidence = _json_snapshot(
        checkpoint_path,
        label="checkpoint",
    )
    if checkpoint.get("status") != "completed":
        raise BlindE2EReviewPackageError(
            f"candidate {layout.candidate_id}/{case_id} is not completed"
        )
    final, final_evidence = _json_snapshot(
        final_path,
        label="final transcript",
    )
    source_path = _nonempty_text(
        checkpoint.get("sourcePath"),
        label=f"candidate {layout.candidate_id}/{case_id} sourcePath",
    )
    source_audio = Path(source_path).expanduser().resolve(strict=True)
    if not source_audio.is_file():
        raise BlindE2EReviewPackageError(f"source audio is missing: {source_audio}")
    disposition = _nonempty_text(
        final.get("disposition", "transcribable-speech"),
        label="final.disposition",
    )
    if disposition not in _DISPOSITIONS:
        raise BlindE2EReviewPackageError(
            f"final.disposition is unsupported: {disposition}"
        )
    allow_empty = disposition == "no-transcribable-speech"
    if transcript_path.is_file():
        transcript, transcript_evidence = _json_snapshot(
            transcript_path,
            label="transcript document",
        )
        raw_segments = _segment_views(
            transcript.get("segments"),
            text_field="rawText",
            label="transcript.segments",
            allow_empty=allow_empty,
        )
    elif allow_empty:
        transcript = None
        transcript_evidence = None
        raw_segments = []
    else:
        raise BlindE2EReviewPackageError(
            f"transcript document is missing: {transcript_path}"
        )
    final_segments = _segment_views(
        final.get("segments"),
        text_field="finalText",
        label="final.segments",
        allow_empty=allow_empty,
    )
    timeline = _timeline_view(final.get("timeline"), final_segments=final_segments)
    subtitles, pdfs = _artifact_files(case_root, checkpoint)
    standard_pdf = (case_root / "render" / "report.pdf").resolve()
    pdf_render_inputs = None
    if standard_pdf in pdfs:
        report_document_path = case_root / "input" / "report-document.json"
        render_manifest_path = case_root / "artifacts" / "manifest.json"
        if not report_document_path.is_file() or not render_manifest_path.is_file():
            raise BlindE2EReviewPackageError(
                "standard Java PDF is missing its report document or render manifest"
            )
        pdf_render_inputs = {
            "reportDocument": _file_evidence(report_document_path),
            "renderManifest": _file_evidence(render_manifest_path),
        }
    result_path = (
        layout.results_root / f"{case_id}-result.json"
        if layout.results_root is not None
        else None
    )
    if result_path is not None and result_path.is_file():
        result_evidence = _file_evidence(result_path)
    else:
        result_evidence = None
    original_artifacts = {
        "checkpoint": checkpoint_evidence,
        "transcriptDocument": transcript_evidence,
        "finalTranscript": final_evidence,
        "subtitles": [_file_evidence(path) for path in subtitles],
        "pdfs": [_file_evidence(path) for path in pdfs],
        "pdfRenderInputs": pdf_render_inputs,
        "result": result_evidence,
    }
    return {
        "caseRoot": str(case_root),
        "sourceAudio": _file_evidence(source_audio),
        "disposition": disposition,
        "rawAsrSegments": raw_segments,
        "finalSegments": final_segments,
        "speakerTimeline": timeline,
        "subtitles": subtitles,
        "pdfs": pdfs,
        "originalArtifacts": original_artifacts,
        "detectedModelIdentities": sorted(
            set(
                _model_identity_values(checkpoint)
                + _model_identity_values(transcript)
                + _model_identity_values(final)
            )
        ),
    }


def _ordering_key(seed: str, *parts: str) -> str:
    payload = "\0".join((ORDERING_ALGORITHM, seed, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_suffix(path: Path, *, default: str) -> str:
    suffix = path.suffix.casefold()
    if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
        return default
    return suffix


def _media_suffix(path: Path) -> str:
    suffix = path.suffix.casefold()
    return suffix if suffix in _MEDIA_SUFFIXES else ".media"


def _copy_frozen(source: Path, target: Path, *, expected_sha256: str) -> None:
    if target.exists():
        raise BlindE2EReviewPackageError(f"refusing to overwrite alias: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    actual = sha256_file(target)
    if actual != expected_sha256:
        raise BlindE2EReviewPackageError(
            f"artifact changed while the review package was frozen: {source}"
        )


def _transcript_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the content-bearing portion that blind PDF rerendering preserves."""

    segments = document.get("segments")
    if not isinstance(segments, list):
        raise BlindE2EReviewPackageError(
            "blind PDF report document segments must be an array"
        )
    projection_segments: list[dict[str, Any]] = []
    for index, value in enumerate(segments):
        if not isinstance(value, Mapping):
            raise BlindE2EReviewPackageError(
                f"blind PDF report document segments[{index}] must be an object"
            )
        projection_segments.append(
            {
                key: value.get(key)
                for key in (
                    "id",
                    "startMs",
                    "endMs",
                    "speakerId",
                    "rawText",
                    "normalizedText",
                    "displayText",
                )
            }
        )
    return {
        "documentId": document.get("documentId"),
        "segments": projection_segments,
    }


def _blind_safe_report_document(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Copy a ReportDocument while changing only two identity-bearing labels."""

    source = document.get("source")
    if not isinstance(source, Mapping):
        raise BlindE2EReviewPackageError(
            "blind PDF report document source must be an object"
        )
    original_projection = _transcript_projection(document)
    sanitized = copy.deepcopy(dict(document))
    sanitized_source = sanitized.get("source")
    if not isinstance(sanitized_source, dict):
        raise BlindE2EReviewPackageError(
            "blind PDF report document source could not be copied"
        )
    original_name = str(source.get("fileName") or "").strip()
    if not original_name:
        raise BlindE2EReviewPackageError(
            "blind PDF report document source.fileName is empty"
        )
    sanitized_source["fileName"] = "source-media" + _media_suffix(Path(original_name))
    sanitized["source"] = sanitized_source
    sanitized["title"] = BLIND_PDF_TITLE
    if _transcript_projection(sanitized) != original_projection:
        raise BlindE2EReviewPackageError(
            "blind PDF sanitization changed transcript content"
        )
    return sanitized, canonical_json_sha256(original_projection)


def _manifest_entry_sha(
    manifest: Mapping[str, Any],
    *,
    relative_path: str,
    artifact_type: str,
) -> str:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise BlindE2EReviewPackageError(
            "standard Java PDF render manifest artifacts must be an array"
        )
    normalized_target = relative_path.replace("\\", "/")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("type") == artifact_type
        and str(item.get("relativePath") or "").replace("\\", "/")
        == normalized_target
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise BlindE2EReviewPackageError(
            f"standard Java PDF render manifest is missing {artifact_type}"
        )
    return str(matches[0]["sha256"])


def _rerender_blind_safe_pdf(
    *,
    scanned: Mapping[str, Any],
    source_pdf: Path,
    original_pdf: Mapping[str, Any],
    alias_target: Path,
    scanner: _PdfScanner,
    review_case_id: str,
    review_candidate_id: str,
) -> tuple[str, dict[str, Any]] | None:
    """Rerender the standard product PDF with identity-bearing labels removed."""

    case_root = Path(str(scanned["caseRoot"])).expanduser().resolve(strict=True)
    expected_pdf = (case_root / "render" / "report.pdf").resolve()
    if source_pdf.resolve() != expected_pdf:
        return None
    raw_inputs = scanned.get("originalArtifacts", {}).get("pdfRenderInputs")
    if not isinstance(raw_inputs, Mapping):
        return None
    report_evidence = raw_inputs.get("reportDocument")
    manifest_evidence = raw_inputs.get("renderManifest")
    if not isinstance(report_evidence, Mapping) or not isinstance(
        manifest_evidence, Mapping
    ):
        raise BlindE2EReviewPackageError(
            "standard Java PDF render inputs are incomplete"
        )
    report_path = Path(str(report_evidence.get("path"))).expanduser().resolve(
        strict=True
    )
    manifest_path = Path(str(manifest_evidence.get("path"))).expanduser().resolve(
        strict=True
    )
    if report_path != (case_root / "input" / "report-document.json").resolve():
        raise BlindE2EReviewPackageError(
            "standard Java PDF report document is rebound outside its case"
        )
    if manifest_path != (case_root / "artifacts" / "manifest.json").resolve():
        raise BlindE2EReviewPackageError(
            "standard Java PDF render manifest is rebound outside its case"
        )
    if _file_evidence(report_path)["sha256"] != report_evidence.get("sha256"):
        raise BlindE2EReviewPackageError(
            "standard Java PDF report document changed before blind rerender"
        )
    if _file_evidence(manifest_path)["sha256"] != manifest_evidence.get("sha256"):
        raise BlindE2EReviewPackageError(
            "standard Java PDF render manifest changed before blind rerender"
        )
    manifest = _json_object(manifest_path, label="standard Java PDF render manifest")
    if _manifest_entry_sha(
        manifest,
        relative_path="input/report-document.json",
        artifact_type="report-document",
    ) != report_evidence.get("sha256"):
        raise BlindE2EReviewPackageError(
            "standard Java PDF render manifest does not bind its report document"
        )
    if _manifest_entry_sha(
        manifest,
        relative_path="render/report.pdf",
        artifact_type="pdf",
    ) != original_pdf.get("sha256"):
        raise BlindE2EReviewPackageError(
            "standard Java PDF render manifest does not bind its PDF"
        )
    report_document = _json_object(report_path, label="standard Java PDF report document")
    sanitized, projection_sha = _blind_safe_report_document(report_document)
    alias_target.parent.mkdir(parents=True, exist_ok=True)
    _verify_pdf_scanner_unchanged(scanner)
    try:
        from reporting.java_pdf_client import JavaPdfClient

        with tempfile.TemporaryDirectory(
            dir=str(alias_target.parent),
            prefix=".blind-pdf-render-",
        ) as temporary:
            temporary_root = Path(temporary).resolve()
            client = JavaPdfClient.from_jar(
                scanner.jar_path,
                java_executable=scanner.java_executable,
                allowed_output_root=temporary_root,
                renderer_cwd=PROJECT_ROOT,
                timeout_seconds=float(_PDF_SCAN_TIMEOUT_SECONDS),
            )
            outcome = client.render(
                sanitized,
                job_id=f"blind-{review_case_id}-{review_candidate_id}-pdf",
                output_directory=temporary_root / "output",
            )
            rendered_pdf = outcome.artifact_paths["pdfPath"]
            rendered_sha = sha256_file(rendered_pdf)
            _copy_frozen(
                rendered_pdf,
                alias_target,
                expected_sha256=rendered_sha,
            )
    except Exception as exc:
        if isinstance(exc, BlindE2EReviewPackageError):
            raise
        raise BlindE2EReviewPackageError(
            "standard Java PDF could not be rerendered for blind review"
        ) from exc
    if _file_evidence(source_pdf)["sha256"] != original_pdf.get("sha256"):
        raise BlindE2EReviewPackageError(
            "source PDF changed during blind rerender"
        )
    if _file_evidence(report_path)["sha256"] != report_evidence.get("sha256"):
        raise BlindE2EReviewPackageError(
            "source report document changed during blind rerender"
        )
    _verify_pdf_scanner_unchanged(scanner)
    result = outcome.result
    quality = result.get("quality")
    return rendered_sha, {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "blind-review-pdf-rerender",
        "sanitizationProfile": BLIND_PDF_SANITIZATION_PROFILE,
        "changedFields": ["$.title", "$.source.fileName"],
        "originalPdfSha256": str(original_pdf["sha256"]),
        "originalReportDocumentSha256": str(report_evidence["sha256"]),
        "blindReportDocumentSha256": outcome.report_document_sha256,
        "preservedTranscriptCanonicalSha256": projection_sha,
        "rendererJarSha256": scanner.jar_sha256,
        "rendererVersion": str(result.get("rendererVersion") or ""),
        "qualityScore": float(quality.get("score")) if isinstance(quality, Mapping) else None,
    }


def _resolve_pdf_scanner(
    *,
    java_executable: str | Path | None,
    pdf_scanner_jar: Path | None,
) -> _PdfScanner:
    requested_java = "java" if java_executable is None else os.fspath(java_executable)
    resolved_java = shutil.which(requested_java)
    if resolved_java is None:
        raise BlindE2EReviewPackageError(
            "declared reviewer PDFs require an available Java executable"
        )
    jar = (
        DEFAULT_PDF_SCANNER_JAR
        if pdf_scanner_jar is None
        else pdf_scanner_jar
    ).expanduser()
    try:
        resolved_jar = jar.resolve(strict=True)
    except OSError as exc:
        raise BlindE2EReviewPackageError(
            "declared reviewer PDFs require the PDFBox scanner JAR"
        ) from exc
    if not resolved_jar.is_file():
        raise BlindE2EReviewPackageError(
            "the configured PDFBox scanner JAR is not a regular file"
        )
    return _PdfScanner(
        java_executable=str(Path(resolved_java).resolve()),
        jar_path=resolved_jar,
        jar_bytes=resolved_jar.stat().st_size,
        jar_sha256=sha256_file(resolved_jar),
    )


def _verify_pdf_scanner_unchanged(scanner: _PdfScanner) -> None:
    try:
        unchanged = (
            scanner.jar_path.is_file()
            and scanner.jar_path.stat().st_size == scanner.jar_bytes
            and sha256_file(scanner.jar_path) == scanner.jar_sha256
        )
    except OSError as exc:
        raise BlindE2EReviewPackageError(
            "the PDFBox scanner JAR disappeared during blind packaging"
        ) from exc
    if not unchanged:
        raise BlindE2EReviewPackageError(
            "the PDFBox scanner JAR changed during blind packaging"
        )


def _expected_pdf_sensitive_count(values: Sequence[str]) -> int:
    normalized = {
        unicodedata.normalize("NFKC", value.strip()).lower()
        for value in values
        if len(unicodedata.normalize("NFKC", value.strip())) >= 4
    }
    return len(normalized)


def _pdf_scan_integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan returned an invalid result contract"
        )
    return value


def _scan_pdf_identity(
    *,
    pdf_path: Path,
    expected_pdf_sha256: str,
    sensitive_values: Sequence[str],
    scanner: _PdfScanner,
) -> dict[str, Any]:
    if not sensitive_values or len(sensitive_values) > 1024 or any(
        len(value) > 8192 for value in sensitive_values
    ):
        raise BlindE2EReviewPackageError(
            "blind PDF scan sensitive values exceed the scanner contract"
        )
    _verify_pdf_scanner_unchanged(scanner)
    request = {
        "schemaVersion": PDF_SCAN_SCHEMA_VERSION,
        "pdfPath": str(pdf_path.resolve(strict=True)),
        "expectedPdfSha256": expected_pdf_sha256,
        "sensitiveValues": list(sensitive_values),
    }
    request_bytes = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    try:
        completed = subprocess.run(
            [
                scanner.java_executable,
                "-jar",
                str(scanner.jar_path),
                "--blind-review-scan",
            ],
            input=request_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=PROJECT_ROOT,
            check=False,
            timeout=_PDF_SCAN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan could not complete"
        ) from exc
    _verify_pdf_scanner_unchanged(scanner)
    if completed.returncode != 0:
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan rejected a reviewer PDF"
        )
    if not completed.stdout or len(completed.stdout) > _MAX_PDF_SCAN_OUTPUT_BYTES:
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan returned an invalid result contract"
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan returned an invalid result contract"
        ) from exc
    expected_fields = {
        "schemaVersion",
        "artifactType",
        "validator",
        "validatorVersion",
        "status",
        "pdfSha256",
        "sensitiveValueCount",
        "pageCount",
        "pageTextCount",
        "documentInfoEntryCount",
        "xmpPacketCount",
        "attachmentCount",
        "annotationCount",
        "formFieldCount",
        "scannedCosObjectCount",
        "decodedStreamBytes",
        "allRequiredSurfacesScanned",
        "findings",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan returned an invalid result contract"
        )
    exact = {
        "schemaVersion": PDF_SCAN_SCHEMA_VERSION,
        "artifactType": PDF_SCAN_ARTIFACT_TYPE,
        "validator": PDF_SCAN_VALIDATOR,
        "validatorVersion": PDF_SCAN_VALIDATOR_VERSION,
        "status": "passed",
        "pdfSha256": expected_pdf_sha256,
        "allRequiredSurfacesScanned": True,
        "findings": [],
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan did not prove that the reviewer PDF is clean"
        )
    integer_fields = (
        "sensitiveValueCount",
        "pageCount",
        "pageTextCount",
        "documentInfoEntryCount",
        "xmpPacketCount",
        "attachmentCount",
        "annotationCount",
        "formFieldCount",
        "scannedCosObjectCount",
        "decodedStreamBytes",
    )
    counters = {field: _pdf_scan_integer(payload, field) for field in integer_fields}
    if counters["sensitiveValueCount"] != _expected_pdf_sensitive_count(
        sensitive_values
    ):
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan returned an invalid sensitive-value count"
        )
    if counters["pageTextCount"] != counters["pageCount"]:
        raise BlindE2EReviewPackageError(
            "PDFBox blind scan did not inspect every page text surface"
        )
    if sha256_file(pdf_path) != expected_pdf_sha256:
        raise BlindE2EReviewPackageError(
            "reviewer PDF changed during the PDFBox blind scan"
        )
    return {
        "schemaVersion": PDF_SCAN_SCHEMA_VERSION,
        "artifactType": PDF_SCAN_ARTIFACT_TYPE,
        "validator": PDF_SCAN_VALIDATOR,
        "validatorVersion": PDF_SCAN_VALIDATOR_VERSION,
        "status": "passed",
        "pdfSha256": expected_pdf_sha256,
        "scannerJarSha256": scanner.jar_sha256,
        "allRequiredSurfacesScanned": True,
        **counters,
    }


def _contains_sensitive_token(path: Path, tokens: Sequence[bytes]) -> bool:
    # Artifacts may be binary (notably PDFs), so scan bytes directly.  bytes.lower()
    # gives the intended ASCII-insensitive matching for model IDs and paths while
    # preserving arbitrary binary payloads.
    active = [token.lower() for token in tokens if len(token) >= 4]
    if not active:
        return False
    overlap = max(len(token) for token in active) - 1
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = (tail + chunk).lower()
            if any(token in value for token in active):
                return True
            tail = value[-overlap:] if overlap else b""
    return False


def _sensitive_patterns(values: Sequence[str]) -> tuple[bytes, ...]:
    patterns: set[bytes] = set()
    for raw in values:
        value = raw.strip()
        if len(value) < 4:
            continue
        text_variants = {
            value,
            value.casefold(),
            value.replace("\\", "/"),
            value.replace("/", "\\"),
        }
        for text in tuple(text_variants):
            text_variants.add(text.casefold())
        for text in text_variants:
            for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
                patterns.add(text.encode(encoding))
    return tuple(sorted(patterns, key=lambda item: (len(item), item)))


def _sensitive_values(
    *,
    layouts: Sequence[_RunLayout],
    case_ids: Sequence[str],
    scans: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, ...]:
    values = {
        *_RESERVED_BLIND_IDENTITIES,
        *(layout.candidate_id for layout in layouts),
        *(str(layout.run_root) for layout in layouts),
        *(str(layout.outputs_root) for layout in layouts),
        *(str(layout.results_root) for layout in layouts if layout.results_root),
        *case_ids,
    }

    def visit(item: Any, *, key: str | None = None) -> None:
        if isinstance(item, Mapping):
            for nested_key, nested in item.items():
                visit(nested, key=str(nested_key))
        elif isinstance(item, list):
            for nested in item:
                visit(nested, key=key)
        elif key in {"path", "caseRoot"} and isinstance(item, str):
            values.add(item)

    for scanned in scans.values():
        values.update(str(item) for item in scanned["detectedModelIdentities"])
        visit(scanned)
    expanded: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        expanded.update(
            {
                value,
                value.replace("\\", "/"),
                value.replace("/", "\\"),
            }
        )
    return tuple(sorted(expanded))


def _hash_bound_artifacts(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _hash_bound_artifacts(nested)
            for key, nested in value.items()
            if key != "path"
        }
    if isinstance(value, list):
        return [_hash_bound_artifacts(item) for item in value]
    return value


def _file_evidence_rows(value: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if (
            isinstance(value.get("path"), str)
            and isinstance(value.get("sha256"), str)
            and isinstance(value.get("bytes"), int)
        ):
            rows.append(value)
        else:
            for nested in value.values():
                rows.extend(_file_evidence_rows(nested))
    elif isinstance(value, list):
        for nested in value:
            rows.extend(_file_evidence_rows(nested))
    return rows


def _verify_scans_unchanged(
    scans: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    checked: set[tuple[str, str, int]] = set()
    for scanned in scans.values():
        for evidence in _file_evidence_rows(scanned):
            path = Path(str(evidence["path"])).expanduser()
            expected_sha256 = str(evidence["sha256"])
            expected_bytes = int(evidence["bytes"])
            key = (str(path), expected_sha256, expected_bytes)
            if key in checked:
                continue
            checked.add(key)
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise BlindE2EReviewPackageError(
                    f"source evidence disappeared while freezing: {path}"
                ) from exc
            if (
                not resolved.is_file()
                or resolved.stat().st_size != expected_bytes
                or sha256_file(resolved) != expected_sha256
            ):
                raise BlindE2EReviewPackageError(
                    f"source evidence changed while freezing: {resolved}"
                )


def _original_alias_evidence(
    scanned: Mapping[str, Any],
    *,
    kind: str,
    source_path: Path,
) -> Mapping[str, Any]:
    evidence_key = "subtitles" if kind == "subtitle" else "pdfs"
    for evidence in scanned["originalArtifacts"][evidence_key]:
        if Path(str(evidence["path"])) == source_path:
            return evidence
    raise BlindE2EReviewPackageError(
        f"declared {kind} has no frozen source evidence: {source_path}"
    )


def _blind_view(
    *,
    package_id: str,
    review_case_id: str,
    review_candidate_id: str,
    scanned: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": "blind-e2e-candidate-evidence",
        "packageId": package_id,
        "reviewCaseId": review_case_id,
        "reviewCandidateId": review_candidate_id,
        "disposition": scanned["disposition"],
        "speakerTimeline": scanned["speakerTimeline"],
        "rawAsrSegments": scanned["rawAsrSegments"],
        "finalSegments": scanned["finalSegments"],
    }
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def _parse_candidate(value: str) -> CandidateRun:
    candidate_id, separator, raw_path = value.partition("=")
    if not separator or not candidate_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("candidate must use LABEL=RUN_ROOT")
    return CandidateRun(candidate_id.strip(), Path(raw_path.strip()))


def build_review_package(
    *,
    candidates: Sequence[CandidateRun],
    output_root: Path,
    seed: str,
    case_ids: Sequence[str] = (),
    java_executable: str | Path | None = None,
    pdf_scanner_jar: Path | None = None,
) -> dict[str, Any]:
    normalized_seed = seed.strip()
    if not normalized_seed or len(normalized_seed) > 512:
        raise BlindE2EReviewPackageError("ordering seed must be non-empty text")
    if len(candidates) < 2:
        raise BlindE2EReviewPackageError(
            "blind review requires at least two candidates"
        )
    layouts = sorted(
        (_resolve_layout(candidate) for candidate in candidates),
        key=lambda item: item.candidate_id,
    )
    candidate_ids = [layout.candidate_id for layout in layouts]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise BlindE2EReviewPackageError("candidate IDs must be unique")

    available = {layout.candidate_id: _available_case_ids(layout) for layout in layouts}
    requested = tuple(
        sorted({item.strip() for item in case_ids if item.strip()})
    )
    if requested:
        selected_case_ids = list(requested)
    else:
        first = available[layouts[0].candidate_id]
        if any(values != first for values in available.values()):
            raise BlindE2EReviewPackageError(
                "candidate runs expose different case sets; select cases explicitly"
            )
        selected_case_ids = sorted(first)
    if not selected_case_ids:
        raise BlindE2EReviewPackageError("blind review contains no cases")
    for layout in layouts:
        missing = sorted(set(selected_case_ids) - available[layout.candidate_id])
        if missing:
            raise BlindE2EReviewPackageError(
                f"candidate {layout.candidate_id} is missing cases: "
                + ", ".join(missing)
            )

    scans: dict[tuple[str, str], dict[str, Any]] = {}
    for layout in layouts:
        for case_id in selected_case_ids:
            scans[(layout.candidate_id, case_id)] = _scan_case(layout, case_id)
    for case_id in selected_case_ids:
        source_hashes = {
            str(scans[(layout.candidate_id, case_id)]["sourceAudio"]["sha256"])
            for layout in layouts
        }
        if len(source_hashes) != 1:
            raise BlindE2EReviewPackageError(
                f"candidate runs are rebound to different source audio for {case_id}"
            )
    _verify_scans_unchanged(scans)
    sensitive_values = _sensitive_values(
        layouts=layouts,
        case_ids=selected_case_ids,
        scans=scans,
    )
    sensitive_tokens = _sensitive_patterns(sensitive_values)
    declared_pdf_count = sum(len(scanned["pdfs"]) for scanned in scans.values())
    pdf_scanner = (
        _resolve_pdf_scanner(
            java_executable=java_executable,
            pdf_scanner_jar=pdf_scanner_jar,
        )
        if declared_pdf_count
        else None
    )

    identity = {
        "schemaVersion": SCHEMA_VERSION,
        "orderingAlgorithm": ORDERING_ALGORITHM,
        "orderingSeed": normalized_seed,
        "candidates": [
            {
                "candidateId": layout.candidate_id,
                "cases": [
                    {
                        "caseId": case_id,
                        "sourceSha256": scans[(layout.candidate_id, case_id)][
                            "sourceAudio"
                        ]["sha256"],
                        "artifactEvidence": _hash_bound_artifacts(
                            scans[(layout.candidate_id, case_id)][
                                "originalArtifacts"
                            ]
                        ),
                    }
                    for case_id in selected_case_ids
                ],
            }
            for layout in layouts
        ],
    }
    package_id = "blind-e2e-" + canonical_json_sha256(identity)
    ordered_cases = sorted(
        selected_case_ids,
        key=lambda item: (_ordering_key(normalized_seed, "case", item), item),
    )

    target = output_root.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise BlindE2EReviewPackageError(f"refusing to overwrite output: {target}")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    review_root = stage / "reviewer-packet"
    identity_root = stage / "identity-vault"
    review_root.mkdir()
    identity_root.mkdir()
    blind_cases: list[dict[str, Any]] = []
    unblind_cases: list[dict[str, Any]] = []
    try:
        for case_index, case_id in enumerate(ordered_cases, start=1):
            review_case_id = f"case-{case_index:03d}"
            case_root = review_root / "cases" / review_case_id
            first_scan = scans[(layouts[0].candidate_id, case_id)]
            source = Path(str(first_scan["sourceAudio"]["path"]))
            audio_relative = (
                Path("cases")
                / review_case_id
                / ("source-audio" + _media_suffix(source))
            )
            audio_target = review_root / audio_relative
            _copy_frozen(
                source,
                audio_target,
                expected_sha256=str(first_scan["sourceAudio"]["sha256"]),
            )
            ordered_candidates = sorted(
                layouts,
                key=lambda layout: (
                    _ordering_key(
                        normalized_seed,
                        "candidate",
                        case_id,
                        layout.candidate_id,
                    ),
                    layout.candidate_id,
                ),
            )
            blind_candidate_rows: list[dict[str, Any]] = []
            unblind_candidate_rows: list[dict[str, Any]] = []
            for candidate_index, layout in enumerate(ordered_candidates, start=1):
                review_candidate_id = f"option-{candidate_index:02d}"
                scanned = scans[(layout.candidate_id, case_id)]
                candidate_relative = (
                    Path("cases") / review_case_id / review_candidate_id
                )
                evidence_relative = candidate_relative / "candidate-evidence.v1.json"
                evidence_target = review_root / evidence_relative
                evidence = _blind_view(
                    package_id=package_id,
                    review_case_id=review_case_id,
                    review_candidate_id=review_candidate_id,
                    scanned=scanned,
                )
                atomic_write_json_no_replace(evidence_target, evidence)
                artifacts: list[dict[str, Any]] = [
                    {
                        "kind": "candidate-evidence",
                        "path": evidence_relative.as_posix(),
                        "bytes": evidence_target.stat().st_size,
                        "sha256": sha256_file(evidence_target),
                        "canonicalSha256": evidence["canonicalSha256"],
                    }
                ]
                original_aliases: list[dict[str, Any]] = []
                for kind, paths in (
                    ("subtitle", scanned["subtitles"]),
                    ("pdf", scanned["pdfs"]),
                ):
                    for artifact_index, source_path in enumerate(paths, start=1):
                        source_path = Path(source_path)
                        suffix = _safe_suffix(
                            source_path,
                            default=".bin",
                        )
                        alias_relative = candidate_relative / (
                            f"{kind}-{artifact_index:02d}{suffix}"
                        )
                        alias_target = review_root / alias_relative
                        original = _original_alias_evidence(
                            scanned,
                            kind=kind,
                            source_path=source_path,
                        )
                        pdf_rerender = None
                        if kind == "pdf":
                            if pdf_scanner is None:
                                raise BlindE2EReviewPackageError(
                                    "declared reviewer PDF has no PDFBox scanner"
                                )
                            pdf_rerender = _rerender_blind_safe_pdf(
                                scanned=scanned,
                                source_pdf=source_path,
                                original_pdf=original,
                                alias_target=alias_target,
                                scanner=pdf_scanner,
                                review_case_id=review_case_id,
                                review_candidate_id=review_candidate_id,
                            )
                        if pdf_rerender is None:
                            _copy_frozen(
                                source_path,
                                alias_target,
                                expected_sha256=str(original["sha256"]),
                            )
                            alias_sha256 = sha256_file(alias_target)
                        else:
                            alias_sha256, pdf_rerender = pdf_rerender
                        pdf_identity_scan = None
                        if kind == "pdf":
                            pdf_identity_scan = _scan_pdf_identity(
                                pdf_path=alias_target,
                                expected_pdf_sha256=alias_sha256,
                                sensitive_values=sensitive_values,
                                scanner=pdf_scanner,
                            )
                        if _contains_sensitive_token(alias_target, sensitive_tokens):
                            raise BlindE2EReviewPackageError(
                                f"{kind} artifact contains a blinded identity: "
                                f"{source_path}"
                            )
                        alias = {
                            "kind": kind,
                            "path": alias_relative.as_posix(),
                            "bytes": alias_target.stat().st_size,
                            "sha256": alias_sha256,
                        }
                        if pdf_identity_scan is not None:
                            alias["blindIdentityScan"] = pdf_identity_scan
                        if pdf_rerender is not None:
                            alias["blindPdfRerender"] = pdf_rerender
                        artifacts.append(alias)
                        original_aliases.append(
                            {
                                "reviewArtifact": alias,
                                "originalArtifact": original,
                            }
                        )
                if _contains_sensitive_token(evidence_target, sensitive_tokens):
                    raise BlindE2EReviewPackageError(
                        "normalized candidate evidence contains a blinded identity"
                    )
                blind_candidate_rows.append(
                    {
                        "reviewCandidateId": review_candidate_id,
                        "artifacts": artifacts,
                    }
                )
                unblind_candidate_rows.append(
                    {
                        "reviewCandidateId": review_candidate_id,
                        "candidateId": layout.candidate_id,
                        "runRoot": str(layout.run_root),
                        "caseRoot": scanned["caseRoot"],
                        "sourceAudio": scanned["sourceAudio"],
                        "detectedModelIdentities": scanned[
                            "detectedModelIdentities"
                        ],
                        "originalArtifacts": scanned["originalArtifacts"],
                        "reviewAliases": original_aliases,
                    }
                )
            blind_cases.append(
                {
                    "reviewCaseId": review_case_id,
                    "sourceAudio": {
                        "path": audio_relative.as_posix(),
                        "bytes": audio_target.stat().st_size,
                        "sha256": sha256_file(audio_target),
                    },
                    "candidates": blind_candidate_rows,
                }
            )
            unblind_cases.append(
                {
                    "reviewCaseId": review_case_id,
                    "caseId": case_id,
                    "sourceAudio": first_scan["sourceAudio"],
                    "candidates": unblind_candidate_rows,
                }
            )

        review_form = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-e2e-human-review-form",
            "packageId": package_id,
            "automaticScoring": False,
            "cases": [
                {
                    "reviewCaseId": case["reviewCaseId"],
                    "candidateOrder": [
                        item["reviewCandidateId"] for item in case["candidates"]
                    ],
                    "preferredCandidateId": None,
                    "tie": None,
                    "speakerTimelineReview": None,
                    "rawAsrReview": None,
                    "finalTranscriptReview": None,
                    "subtitlePdfReview": None,
                    "notes": [],
                }
                for case in blind_cases
            ],
        }
        review_form_path = review_root / "human-review-form.v1.json"
        atomic_write_json_no_replace(review_form_path, review_form)

        unblind_body = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-e2e-unblind-mapping",
            "packageId": package_id,
            "referenceInputsAccepted": False,
            "ordering": {
                "algorithm": ORDERING_ALGORITHM,
                "seed": normalized_seed,
            },
            "inputIdentityCanonicalSha256": canonical_json_sha256(identity),
            "cases": unblind_cases,
        }
        unblind = {
            **unblind_body,
            "canonicalSha256": canonical_json_sha256(unblind_body),
        }
        unblind_path = identity_root / "unblind-mapping.v1.json"
        atomic_write_json_no_replace(unblind_path, unblind)

        blind_body = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-e2e-review-package",
            "packageId": package_id,
            "candidateOrdering": {
                "algorithm": ORDERING_ALGORITHM,
                "seedSha256": hashlib.sha256(
                    normalized_seed.encode("utf-8")
                ).hexdigest(),
                "perCase": True,
            },
            "blindnessPolicy": {
                "candidateIdentityPersisted": False,
                "originalPathPersisted": False,
                "referenceAnswerPersisted": False,
                "automaticScorePersisted": False,
                "referenceInputsAccepted": False,
            },
            "pdfIdentityScanning": {
                "requiredForDeclaredPdf": True,
                "declaredPdfCount": declared_pdf_count,
                "validator": PDF_SCAN_VALIDATOR,
                "validatorVersion": PDF_SCAN_VALIDATOR_VERSION,
                "scannerJarSha256": (
                    pdf_scanner.jar_sha256 if pdf_scanner is not None else None
                ),
            },
            "identityCommitment": {
                "fileSha256": sha256_file(unblind_path),
                "canonicalSha256": unblind["canonicalSha256"],
            },
            "reviewForm": {
                "path": review_form_path.relative_to(review_root).as_posix(),
                "sha256": sha256_file(review_form_path),
            },
            "counts": {
                "cases": len(blind_cases),
                "candidatesPerCase": len(layouts),
            },
            "cases": blind_cases,
        }
        blind = {
            **blind_body,
            "canonicalSha256": canonical_json_sha256(blind_body),
        }
        blind_path = review_root / "blind-review-manifest.v1.json"
        serialized_blind = json.dumps(blind, ensure_ascii=False, sort_keys=True)
        serialized_blind_folded = serialized_blind.casefold()
        if any(
            len(value) >= 4 and value.casefold() in serialized_blind_folded
            for value in sensitive_values
        ):
            raise BlindE2EReviewPackageError(
                "blind manifest contains an original candidate or case identity"
            )
        for path in review_root.rglob("*"):
            if path.is_file() and _contains_sensitive_token(
                path,
                sensitive_tokens,
            ):
                raise BlindE2EReviewPackageError(
                    f"reviewer packet contains a blinded identity: {path}"
                )
        atomic_write_json_no_replace(blind_path, blind)

        package_manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactType": "blind-e2e-package-files",
            "packageId": package_id,
            "reviewerPacket": {
                "relativePath": (
                    "reviewer-packet/blind-review-manifest.v1.json"
                ),
                "fileSha256": sha256_file(blind_path),
                "canonicalSha256": blind["canonicalSha256"],
            },
            "identityVault": {
                "relativePath": "identity-vault/unblind-mapping.v1.json",
                "fileSha256": sha256_file(unblind_path),
                "canonicalSha256": unblind["canonicalSha256"],
            },
            "referenceInputsAccepted": False,
        }
        atomic_write_json_no_replace(
            stage / "package-manifest.v1.json",
            package_manifest,
        )
        _verify_scans_unchanged(scans)
        if pdf_scanner is not None:
            _verify_pdf_scanner_unchanged(pdf_scanner)
        if target.exists():
            raise BlindE2EReviewPackageError(f"refusing to overwrite output: {target}")
        os.rename(stage, target)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return {
        "packageId": package_id,
        "packageRoot": str(target.resolve()),
        "reviewManifest": str(
            (
                target
                / "reviewer-packet"
                / "blind-review-manifest.v1.json"
            ).resolve()
        ),
        "identityMapping": str(
            (
                target
                / "identity-vault"
                / "unblind-mapping.v1.json"
            ).resolve()
        ),
        "caseCount": len(blind_cases),
        "candidateCount": len(layouts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        required=True,
        help="candidate run as LABEL=RUN_ROOT; repeat for every candidate",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--java-executable",
        help="Java executable used only when a declared reviewer PDF is present",
    )
    parser.add_argument(
        "--pdf-scanner-jar",
        type=Path,
        help=(
            "PDFBox scanner JAR; defaults to "
            "pdf-renderer/target/pdf-renderer.jar"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_review_package(
        candidates=args.candidate,
        output_root=args.output_root,
        seed=args.seed,
        case_ids=args.case,
        java_executable=args.java_executable,
        pdf_scanner_jar=args.pdf_scanner_jar,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
