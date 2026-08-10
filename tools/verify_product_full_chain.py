"""Verify a completed production run and every customer-delivery binding.

This verifier is intentionally independent from the worker that produced the
artifacts.  It accepts only the persisted smoke result and one case output
directory, recomputes file hashes, and fails closed on incomplete review,
semantic composition, subtitle publication, transcript exports, or PDF QA.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    sha256_file,
)


SCHEMA_VERSION = "1.0.0"
ARTIFACT_TYPE = "product-full-chain-verification"
REQUIRED_FORMATS = frozenset({"pdf", "txt", "json", "srt", "webvtt", "ass"})
REQUIRED_SUBTITLE_FORMATS = frozenset({"srt", "webvtt", "ass"})
REQUIRED_TRANSCRIPT_FORMATS = frozenset({"txt", "json"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<tail>.*)$")


class ProductFullChainVerificationError(ValueError):
    """Raised when persisted product evidence does not prove completion."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ProductFullChainVerificationError(
            f"{label} contains a non-finite JSON number: {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProductFullChainVerificationError(
                    f"{label} contains duplicate field: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except ProductFullChainVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductFullChainVerificationError(
            f"cannot read {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ProductFullChainVerificationError(f"{label} must be an object")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductFullChainVerificationError(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str, non_empty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (non_empty and not value):
        requirement = "a non-empty array" if non_empty else "an array"
        raise ProductFullChainVerificationError(f"{field} must be {requirement}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProductFullChainVerificationError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _host_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    match = _DRIVE_PATH_RE.fullmatch(raw)
    if os.name != "nt" and match is not None:
        tail = PurePosixPath(match.group("tail").replace("\\", "/"))
        return Path("/mnt") / match.group("drive").lower() / Path(*tail.parts)
    return Path(raw)


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ProductFullChainVerificationError(
            f"{label} must not be a symlink: {path}"
        )
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ProductFullChainVerificationError(
            f"{label} is missing: {path}"
        ) from exc
    if not resolved.is_file():
        raise ProductFullChainVerificationError(
            f"{label} must be a regular non-symlink file: {resolved}"
        )
    return resolved


def _output_artifact_path(
    raw: Any,
    *,
    output_root: Path,
    label: str,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ProductFullChainVerificationError(f"{label} must be a path")
    resolved = _regular_file(_host_path(raw.strip()), label=label)
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ProductFullChainVerificationError(
            f"{label} escapes the case output directory: {resolved}"
        ) from exc
    return resolved


def _relative_artifact_path(
    raw: Any,
    *,
    output_root: Path,
    label: str,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ProductFullChainVerificationError(
            f"{label} must be a relative path"
        )
    relative = Path(raw.strip().replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ProductFullChainVerificationError(
            f"{label} must stay below the case output directory"
        )
    return _regular_file(output_root / relative, label=label)


def _verify_file(
    path: Path,
    *,
    label: str,
    expected_sha256: Any | None = None,
    expected_size: Any | None = None,
    evidence: dict[str, dict[str, Any]],
) -> None:
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None:
        expected_digest = _sha256(expected_sha256, field=f"{label}.sha256")
        if actual_sha256 != expected_digest:
            raise ProductFullChainVerificationError(
                f"{label} SHA-256 does not match: {path}"
            )
    if expected_size is not None:
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ProductFullChainVerificationError(
                f"{label}.size must be a non-negative integer"
            )
        if actual_size != expected_size:
            raise ProductFullChainVerificationError(
                f"{label} size does not match: {path}"
            )
    evidence[str(path)] = {
        "path": str(path),
        "sizeBytes": actual_size,
        "sha256": actual_sha256,
    }


def _verify_result(
    result: Mapping[str, Any],
    *,
    expected_job_id: str,
    require_review_resume: bool,
) -> None:
    if (
        result.get("status") != "observed"
        or result.get("terminal_type") != "job.completed"
        or result.get("job_id") != expected_job_id
        or result.get("exit_code") != 0
        or result.get("shutdown_acknowledged") is not True
        or result.get("forced_cleanup_pids") != []
        or result.get("error") is not None
    ):
        raise ProductFullChainVerificationError(
            "smoke result does not prove a clean job.completed terminal state"
        )
    terminal = _mapping(result.get("terminal_event"), field="terminal_event")
    payload = _mapping(terminal.get("payload"), field="terminal_event.payload")
    if (
        terminal.get("jobId") != expected_job_id
        or terminal.get("type") != "job.completed"
        or payload.get("status") != "completed"
    ):
        raise ProductFullChainVerificationError(
            "terminal event does not match the completed job"
        )
    if require_review_resume and payload.get("operation") != "resume":
        raise ProductFullChainVerificationError(
            "completed job does not prove the supplied manual review was resumed"
        )


def _verify_review(
    checkpoint: Mapping[str, Any],
    *,
    output_root: Path,
    require_manual_review: bool,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, int]:
    queue_path = _output_artifact_path(
        checkpoint.get("reviewQueuePath"),
        output_root=output_root,
        label="reviewQueuePath",
    )
    queue = _load_json(queue_path, label="review queue")
    items = _sequence(queue.get("items"), field="review.items")
    decisions = _sequence(queue.get("decisions"), field="review.decisions")
    if checkpoint.get("reviewOpenCount") != 0 or queue.get("openCount") != 0:
        raise ProductFullChainVerificationError("review queue is still open")
    allowed_status = {"accepted", "rejected"}
    if any(
        not isinstance(item, Mapping) or item.get("status") not in allowed_status
        for item in items
    ):
        raise ProductFullChainVerificationError(
            "every review item must have a terminal manual decision"
        )
    item_ids = {
        str(item.get("id"))
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    review_decisions: list[Mapping[str, Any]] = []
    pre_review_decisions: list[Mapping[str, Any]] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ProductFullChainVerificationError(
                f"review.decisions[{index}] must be an object"
            )
        command = decision.get("command")
        item_id = decision.get("item_id", decision.get("itemId"))
        if command == "speaker.merge" or decision.get("type") == "speaker.merge":
            pre_review_decisions.append(decision)
        elif command == "review.submit" or item_id in item_ids or len(decisions) == len(items):
            review_decisions.append(decision)
        else:
            raise ProductFullChainVerificationError(
                "review decision has neither a review item nor a merge command"
            )
    if len(review_decisions) != len(items):
        raise ProductFullChainVerificationError(
            "review decision count does not match item count"
        )
    reviewed_item_ids = {
        str(decision.get("item_id", decision.get("itemId")))
        for decision in review_decisions
    }
    if reviewed_item_ids != item_ids:
        raise ProductFullChainVerificationError(
            "review decisions do not cover exactly every queue item"
        )
    if require_manual_review:
        for index, decision in enumerate(review_decisions + pre_review_decisions):
            audit = _mapping(
                decision.get("audit"),
                field=f"review.decisions[{index}].audit",
            )
            if audit.get("source") not in {"human", "codex-agent"}:
                raise ProductFullChainVerificationError(
                    "review decisions must come from a manual authority"
                )
    _verify_file(queue_path, label="review queue", evidence=evidence)
    return {
        "itemCount": len(items),
        "acceptedCount": sum(item.get("status") == "accepted" for item in items),
        "rejectedCount": sum(item.get("status") == "rejected" for item in items),
        "decisionCount": len(review_decisions),
        "preReviewDecisionCount": len(pre_review_decisions),
    }


def _verify_semantic(
    checkpoint: Mapping[str, Any],
    *,
    output_root: Path,
    expected_job_id: str,
    expected_model: str | None,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    semantic = _mapping(checkpoint.get("semantic"), field="checkpoint.semantic")
    provenance = _mapping(
        semantic.get("provenance"), field="checkpoint.semantic.provenance"
    )
    if (
        semantic.get("required") is not True
        or semantic.get("status") != "completed"
        or semantic.get("mode") != "candidate-composition"
        or semantic.get("autoApply") is not True
        or semantic.get("error") is not None
    ):
        raise ProductFullChainVerificationError(
            "semantic candidate composition is not complete and auto-applied"
        )
    model = provenance.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ProductFullChainVerificationError(
            "semantic provenance has no model identity"
        )
    if expected_model is not None and model != expected_model:
        raise ProductFullChainVerificationError(
            "semantic model does not match the expected candidate"
        )
    artifact_paths = _sequence(
        semantic.get("artifactPaths"), field="checkpoint.semantic.artifactPaths"
    )
    resolved: list[Path] = []
    for index, raw in enumerate(artifact_paths):
        path = _output_artifact_path(
            raw,
            output_root=output_root,
            label=f"semantic.artifactPaths[{index}]",
        )
        _verify_file(path, label=f"semantic artifact {index}", evidence=evidence)
        resolved.append(path)
    composition_path = _output_artifact_path(
        semantic.get("artifactPath"),
        output_root=output_root,
        label="semantic.artifactPath",
    )
    if composition_path not in resolved:
        raise ProductFullChainVerificationError(
            "semantic composition is missing from artifactPaths"
        )
    composition = _load_json(composition_path, label="semantic composition")
    if (
        composition.get("artifactType") != "semantic-composition"
        or composition.get("jobId") != expected_job_id
        or composition.get("status") != "composition-complete"
    ):
        raise ProductFullChainVerificationError(
            "semantic composition artifact is not terminal or is rebound"
        )
    return {
        "model": model,
        "promptVersion": provenance.get("promptVersion"),
        "roundCount": provenance.get("roundCount"),
        "artifactCount": len(resolved),
        "compositionSha256": sha256_file(composition_path),
    }


def _verify_transcript_exports(
    checkpoint: Mapping[str, Any],
    *,
    output_root: Path,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = _sequence(
        checkpoint.get("transcriptExports"), field="checkpoint.transcriptExports"
    )
    formats: set[str] = set()
    for index, row in enumerate(rows):
        item = _mapping(row, field=f"transcriptExports[{index}]")
        output_format = item.get("format")
        if not isinstance(output_format, str) or output_format in formats:
            raise ProductFullChainVerificationError(
                "transcript export formats must be unique text values"
            )
        path = _output_artifact_path(
            item.get("path"),
            output_root=output_root,
            label=f"transcriptExports[{index}].path",
        )
        _verify_file(
            path,
            label=f"transcript export {output_format}",
            expected_sha256=item.get("sha256"),
            expected_size=item.get("size"),
            evidence=evidence,
        )
        formats.add(output_format)
    if not REQUIRED_TRANSCRIPT_FORMATS <= formats:
        raise ProductFullChainVerificationError(
            "TXT and JSON transcript exports are both required"
        )
    return {"formats": sorted(formats), "count": len(rows)}


def _verify_publication(
    checkpoint: Mapping[str, Any],
    *,
    output_root: Path,
    expected_recipe_sha256: str | None,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    publication = _mapping(
        checkpoint.get("outputPublication"),
        field="checkpoint.outputPublication",
    )
    if publication.get("status") != "published" or publication.get("error") is not None:
        raise ProductFullChainVerificationError(
            "output publication is not in a clean published state"
        )
    manifest_path = _output_artifact_path(
        publication.get("manifestPath"),
        output_root=output_root,
        label="outputPublication.manifestPath",
    )
    _verify_file(
        manifest_path,
        label="output publication manifest",
        expected_sha256=publication.get("manifestFileSha256"),
        evidence=evidence,
    )
    manifest = _load_json(manifest_path, label="output publication manifest")
    declared_manifest_sha = _sha256(
        manifest.get("manifestSha256"), field="manifest.manifestSha256"
    )
    body = dict(manifest)
    body.pop("manifestSha256", None)
    if canonical_json_sha256(body) != declared_manifest_sha:
        raise ProductFullChainVerificationError(
            "output publication manifest self hash does not verify"
        )
    if publication.get("manifestSha256") != declared_manifest_sha:
        raise ProductFullChainVerificationError(
            "checkpoint publication manifest hash is stale"
        )
    recipe_sha = _sha256(manifest.get("recipeSha256"), field="manifest.recipeSha256")
    if expected_recipe_sha256 is not None and recipe_sha != expected_recipe_sha256:
        raise ProductFullChainVerificationError(
            "output publication uses an unexpected recipe"
        )
    source = _mapping(manifest.get("source"), field="manifest.source")
    if source.get("unchanged") is not True:
        raise ProductFullChainVerificationError(
            "publication does not prove immutable source media"
        )
    source_path = _regular_file(_host_path(str(source.get("path"))), label="source media")
    _verify_file(
        source_path,
        label="source media",
        expected_sha256=source.get("sha256"),
        expected_size=source.get("sizeBytes"),
        evidence=evidence,
    )
    rows = _sequence(manifest.get("customerArtifacts"), field="customerArtifacts")
    formats: set[str] = set()
    for index, row in enumerate(rows):
        item = _mapping(row, field=f"customerArtifacts[{index}]")
        subtitle_format = item.get("subtitleFormat")
        if not isinstance(subtitle_format, str) or subtitle_format in formats:
            raise ProductFullChainVerificationError(
                "published subtitle formats must be unique text values"
            )
        if (
            item.get("artifactType") != "subtitle-sidecar"
            or item.get("deliveryMode") != "sidecar"
        ):
            raise ProductFullChainVerificationError(
                "full-chain verifier requires sidecar subtitle receipts"
            )
        path = _output_artifact_path(
            item.get("path"),
            output_root=output_root,
            label=f"customerArtifacts[{index}].path",
        )
        _verify_file(
            path,
            label=f"subtitle {subtitle_format}",
            expected_sha256=item.get("sha256"),
            expected_size=item.get("sizeBytes"),
            evidence=evidence,
        )
        source_integrity = _mapping(
            item.get("sourceIntegrity"),
            field=f"customerArtifacts[{index}].sourceIntegrity",
        )
        if (
            source_integrity.get("unchanged") is not True
            or source_integrity.get("sourceSha256") != source.get("sha256")
        ):
            raise ProductFullChainVerificationError(
                "subtitle receipt is not bound to the immutable source"
            )
        formats.add(subtitle_format)
    if not REQUIRED_SUBTITLE_FORMATS <= formats:
        raise ProductFullChainVerificationError(
            "SRT, WebVTT, and ASS sidecars are all required"
        )
    transaction = _mapping(manifest.get("transaction"), field="manifest.transaction")
    required_transaction_flags = (
        "allConflictsCheckedBeforeWrites",
        "allMediaQaPassedBeforePublication",
        "rollbackSupported",
        "privatePathsExcluded",
        "sourceMediaImmutable",
    )
    if any(transaction.get(field) is not True for field in required_transaction_flags):
        raise ProductFullChainVerificationError(
            "publication transaction guarantees are incomplete"
        )
    return {
        "manifestSha256": declared_manifest_sha,
        "manifestFileSha256": sha256_file(manifest_path),
        "recipeSha256": recipe_sha,
        "subtitleFormats": sorted(formats),
        "sourceSha256": source.get("sha256"),
    }


def _verify_pdf(
    checkpoint: Mapping[str, Any],
    *,
    output_root: Path,
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if checkpoint.get("qualityStatus") != "passed":
        raise ProductFullChainVerificationError("checkpoint PDF quality did not pass")
    manifest_path = _output_artifact_path(
        checkpoint.get("renderManifestPath"),
        output_root=output_root,
        label="renderManifestPath",
    )
    manifest = _load_json(manifest_path, label="PDF render manifest")
    artifacts = _sequence(manifest.get("artifacts"), field="PDF manifest artifacts")
    pdf_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(artifacts):
        item = _mapping(row, field=f"PDF manifest artifacts[{index}]")
        if item.get("verified") is not True:
            raise ProductFullChainVerificationError(
                "PDF render manifest contains an unverified artifact"
            )
        path = _relative_artifact_path(
            item.get("relativePath"),
            output_root=output_root,
            label=f"PDF manifest artifacts[{index}].relativePath",
        )
        _verify_file(
            path,
            label=f"PDF artifact {index}",
            expected_sha256=item.get("sha256"),
            expected_size=item.get("bytes"),
            evidence=evidence,
        )
        if item.get("type") == "pdf":
            pdf_rows.append(item)
            if not path.read_bytes().startswith(b"%PDF-"):
                raise ProductFullChainVerificationError(
                    "declared PDF artifact has no PDF header"
                )
    if len(pdf_rows) != 1:
        raise ProductFullChainVerificationError(
            "PDF render manifest must contain exactly one PDF artifact"
        )
    _verify_file(manifest_path, label="PDF render manifest", evidence=evidence)

    quality_path = _output_artifact_path(
        checkpoint.get("qualityReportPath"),
        output_root=output_root,
        label="qualityReportPath",
    )
    quality = _load_json(quality_path, label="PDF quality report")
    minimum_score = quality.get("minimumScore")
    score = quality.get("score")
    if (
        quality.get("status") != "passed"
        or quality.get("hardGatesPassed") is not True
        or not isinstance(minimum_score, (int, float))
        or isinstance(minimum_score, bool)
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
        or score < minimum_score
        or quality.get("repairQueue") != []
        or quality.get("regressions") != []
    ):
        raise ProductFullChainVerificationError(
            "PDF quality report is not a clean terminal pass"
        )
    hard_gates = _sequence(quality.get("hardGates"), field="quality.hardGates")
    if any(
        not isinstance(gate, Mapping) or gate.get("status") != "passed"
        for gate in hard_gates
    ):
        raise ProductFullChainVerificationError("a PDF hard gate did not pass")
    for index, row in enumerate(
        _sequence(quality.get("evidence"), field="quality.evidence")
    ):
        item = _mapping(row, field=f"quality.evidence[{index}]")
        if item.get("verified") is not True:
            raise ProductFullChainVerificationError(
                "PDF quality evidence contains an unverified artifact"
            )
        path = _relative_artifact_path(
            item.get("relativePath"),
            output_root=output_root,
            label=f"quality.evidence[{index}].relativePath",
        )
        _verify_file(
            path,
            label=f"PDF quality evidence {index}",
            expected_sha256=item.get("sha256"),
            evidence=evidence,
        )
    _verify_file(quality_path, label="PDF quality report", evidence=evidence)
    return {
        "status": "passed",
        "score": score,
        "minimumScore": minimum_score,
        "hardGateCount": len(hard_gates),
        "pdfSha256": pdf_rows[0].get("sha256"),
    }


def verify_product_full_chain(
    *,
    case_output: Path,
    result_json: Path,
    expected_job_id: str | None = None,
    expected_model: str | None = None,
    expected_recipe_sha256: str | None = None,
    require_review_resume: bool = False,
) -> dict[str, Any]:
    if case_output.is_symlink():
        raise ProductFullChainVerificationError(
            "case output must be a regular non-symlink directory"
        )
    output_root = case_output.expanduser().resolve(strict=True)
    if not output_root.is_dir():
        raise ProductFullChainVerificationError(
            "case output must be a regular non-symlink directory"
        )
    result_path = _regular_file(result_json, label="smoke result")
    checkpoint_path = _regular_file(output_root / "checkpoint.v2.json", label="checkpoint")
    result = _load_json(result_path, label="smoke result")
    checkpoint = _load_json(checkpoint_path, label="checkpoint")
    job_id = expected_job_id or str(checkpoint.get("jobId") or "").strip()
    if not job_id or checkpoint.get("jobId") != job_id:
        raise ProductFullChainVerificationError("checkpoint jobId is invalid")
    _verify_result(
        result,
        expected_job_id=job_id,
        require_review_resume=require_review_resume,
    )
    if (
        checkpoint.get("status") != "completed"
        or checkpoint.get("stage") != "completed"
        or checkpoint.get("error") is not None
    ):
        raise ProductFullChainVerificationError(
            "checkpoint is not in a clean completed state"
        )
    recipe_sha = _mapping(
        checkpoint.get("outputCustomization"), field="outputCustomization"
    ).get("sha256")
    recipe_sha = _sha256(recipe_sha, field="outputCustomization.sha256")
    if expected_recipe_sha256 is not None:
        expected_recipe_sha256 = _sha256(
            expected_recipe_sha256, field="expected_recipe_sha256"
        )
        if recipe_sha != expected_recipe_sha256:
            raise ProductFullChainVerificationError(
                "checkpoint output recipe does not match the expected recipe"
            )
    recipe = _mapping(
        _mapping(
            checkpoint.get("outputCustomization"), field="outputCustomization"
        ).get("recipe"),
        field="outputCustomization.recipe",
    )
    delivery = _mapping(recipe.get("delivery"), field="outputCustomization.recipe.delivery")
    declared_formats = set(
        str(item).casefold()
        for item in _sequence(delivery.get("formats"), field="delivery.formats")
        if isinstance(item, str)
    )
    if not REQUIRED_FORMATS <= declared_formats:
        raise ProductFullChainVerificationError(
            "output recipe does not request the complete product format set"
        )

    evidence: dict[str, dict[str, Any]] = {}
    _verify_file(result_path, label="smoke result", evidence=evidence)
    _verify_file(checkpoint_path, label="checkpoint", evidence=evidence)
    review = _verify_review(
        checkpoint,
        output_root=output_root,
        require_manual_review=require_review_resume,
        evidence=evidence,
    )
    semantic = _verify_semantic(
        checkpoint,
        output_root=output_root,
        expected_job_id=job_id,
        expected_model=expected_model,
        evidence=evidence,
    )
    transcript_exports = _verify_transcript_exports(
        checkpoint, output_root=output_root, evidence=evidence
    )
    publication = _verify_publication(
        checkpoint,
        output_root=output_root,
        expected_recipe_sha256=expected_recipe_sha256 or recipe_sha,
        evidence=evidence,
    )
    pdf = _verify_pdf(checkpoint, output_root=output_root, evidence=evidence)

    final_path = _regular_file(
        output_root / "final-adjudicated-transcript.v1.json",
        label="final adjudicated transcript",
    )
    final_document = _load_json(final_path, label="final adjudicated transcript")
    final_review = _mapping(final_document.get("review"), field="final.review")
    final_semantic = _mapping(final_document.get("semantic"), field="final.semantic")
    if (
        final_document.get("jobId") != job_id
        or final_document.get("status") != "adjudication-complete"
        or final_review.get("openCount") != 0
        or final_semantic.get("status") != "composition-complete"
        or final_semantic.get("model") != semantic["model"]
    ):
        raise ProductFullChainVerificationError(
            "final adjudicated transcript is incomplete or rebound"
        )
    _verify_file(final_path, label="final adjudicated transcript", evidence=evidence)

    report_body = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactType": ARTIFACT_TYPE,
        "verifiedAt": datetime.now(UTC).isoformat(),
        "status": "passed",
        "jobId": job_id,
        "caseOutput": str(output_root),
        "resultJson": str(result_path),
        "terminalType": "job.completed",
        "reviewResumed": require_review_resume,
        "outputRecipeSha256": recipe_sha,
        "requestedFormats": sorted(declared_formats),
        "review": review,
        "semantic": semantic,
        "transcriptExports": transcript_exports,
        "publication": publication,
        "pdf": pdf,
        "finalAdjudicatedTranscriptSha256": sha256_file(final_path),
        "verifiedFileCount": len(evidence),
        "verifiedFilesCanonicalSha256": canonical_json_sha256(
            sorted(evidence.values(), key=lambda item: str(item["path"]).casefold())
        ),
        "verifiedFiles": sorted(
            evidence.values(), key=lambda item: str(item["path"]).casefold()
        ),
    }
    return {
        **report_body,
        "canonicalSha256": canonical_json_sha256(report_body),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--expected-job-id")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-recipe-sha256")
    parser.add_argument(
        "--require-review-resume",
        action="store_true",
        help="require a job.completed operation=resume and manual queue authority",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_product_full_chain(
            case_output=args.case_output,
            result_json=args.result_json,
            expected_job_id=args.expected_job_id,
            expected_model=args.expected_model,
            expected_recipe_sha256=args.expected_recipe_sha256,
            require_review_resume=args.require_review_resume,
        )
        if args.output is not None:
            atomic_write_json_no_replace(args.output.expanduser().absolute(), report)
    except (OSError, ProductFullChainVerificationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
