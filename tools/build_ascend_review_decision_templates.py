"""Build non-executable human-decision templates from visible ASCEND queues.

The builder accepts only the truth-redacted product manifest and production
output directories.  It has no scorer-vault, identity-vault, reference, or
automatic-score input.  Templates deliberately use null decisions so they
cannot be passed to the production review harness before a Codex/human reviewer
has inspected the visible audio, transcript, and speaker timeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from tools.build_ascend_heldout_product_manifest import (  # noqa: E402
    ARTIFACT_TYPE as HELD_OUT_MANIFEST_TYPE,
    DEFAULT_STAGE_ROOT,
    MANIFEST_NAME as HELD_OUT_MANIFEST_NAME,
)
from tools.freeze_ascend_code_switch_samples import (  # noqa: E402
    _rename_directory_no_replace,
    assert_truth_redacted,
)


DEFAULT_MANIFEST = DEFAULT_STAGE_ROOT / HELD_OUT_MANIFEST_NAME
DEFAULT_OUTPUTS_ROOT = (
    Path("D:/mts-eval/product-runs/semantic-heldout-ascend6-20260809/")
    / "winner-smoke"
    / "outputs"
    if os.name == "nt"
    else Path(
        "/mnt/d/mts-eval/product-runs/semantic-heldout-ascend6-20260809/"
        "winner-smoke/outputs"
    )
)
DEFAULT_TEMPLATE_ROOT = (
    Path("D:/mts-eval/semantic-heldout/ascend6-qwen35-27b-vs-9b-20260809/")
    / "review-decision-templates-r1"
    if os.name == "nt"
    else Path(
        "/mnt/d/mts-eval/semantic-heldout/"
        "ascend6-qwen35-27b-vs-9b-20260809/review-decision-templates-r1"
    )
)
DEFAULT_EXPECTED_MODEL = "qwen3.5:27b-q4_K_M"
TEMPLATE_ARTIFACT_TYPE = "ascend-held-out-manual-review-decision-template"
INDEX_NAME = "MANIFEST.v1.json"


class AscendReviewTemplateError(ValueError):
    """Raised when visible held-out review evidence is incomplete."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise AscendReviewTemplateError(
            f"{label} contains non-finite JSON number: {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AscendReviewTemplateError(
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
    except AscendReviewTemplateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AscendReviewTemplateError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AscendReviewTemplateError(f"{label} must contain an object")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AscendReviewTemplateError(f"{field} must be an object")
    return value


def _sha_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "fileSha256": sha256_file(path),
    }


def _canonical_document(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "canonicalSha256": canonical_json_sha256(value)}


def _validate_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise AscendReviewTemplateError(
            "held-out product manifest must be a regular non-symlink file"
        )
    value = _load_json(resolved, label="held-out product manifest")
    if value.get("artifactType") != HELD_OUT_MANIFEST_TYPE:
        raise AscendReviewTemplateError(
            "input is not an ASCEND held-out product manifest"
        )
    selection = _mapping(value.get("selection"), field="selection")
    truth_policy = _mapping(
        value.get("truthPersistencePolicy"),
        field="truthPersistencePolicy",
    )
    if (
        selection.get("evaluationSplit") != "held-out"
        or selection.get("sourceSplit") != "test"
        or selection.get("scorerVaultRead") is not False
        or truth_policy.get("scorerTruthRead") is not False
        or truth_policy.get("scorerVaultDependency") is not False
        or truth_policy.get("developmentReferenceCopied") is not False
    ):
        raise AscendReviewTemplateError(
            "held-out product manifest does not preserve the blind boundary"
        )
    declared = value.get("canonicalSha256")
    body = dict(value)
    body.pop("canonicalSha256", None)
    if not isinstance(declared, str) or declared != canonical_json_sha256(body):
        raise AscendReviewTemplateError(
            "held-out product manifest canonicalSha256 does not match"
        )
    try:
        assert_truth_redacted(value)
    except ValueError as exc:
        raise AscendReviewTemplateError(str(exc)) from exc
    return resolved, value


def _visible_text(item: Mapping[str, Any]) -> dict[str, str] | None:
    raw = item.get("text")
    if not isinstance(raw, Mapping):
        return None
    result = {
        field: str(raw[field])
        for field in ("rawText", "normalizedText", "displayText")
        if isinstance(raw.get(field), str)
    }
    return result or None


def _template_for_case(
    *,
    case: Mapping[str, Any],
    outputs_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_model: str,
) -> dict[str, Any]:
    case_id = str(case.get("id") or "").strip()
    if not case_id:
        raise AscendReviewTemplateError("held-out case id is missing")
    if (
        case.get("evaluationSplit") != "held-out"
        or case.get("tuningEligible") is not False
        or case.get("truthAccess") != "isolated-scorer-vault-only"
    ):
        raise AscendReviewTemplateError(f"{case_id} is not held-out only")
    case_output = (outputs_root / case_id).resolve(strict=True)
    try:
        case_output.relative_to(outputs_root)
    except ValueError as exc:
        raise AscendReviewTemplateError(
            f"{case_id} output escapes the configured root"
        ) from exc
    if not case_output.is_dir() or case_output.is_symlink():
        raise AscendReviewTemplateError(
            f"{case_id} output must be a regular directory"
        )
    queue_path = (case_output / "review" / "review-queue.json").resolve(
        strict=True
    )
    checkpoint_path = (case_output / "checkpoint.v2.json").resolve(strict=True)
    transcript_path = (
        case_output / "semantic" / "input-transcript.v2.json"
    ).resolve(strict=True)
    queue = _load_json(queue_path, label=f"{case_id} review queue")
    checkpoint = _load_json(checkpoint_path, label=f"{case_id} checkpoint")
    transcript = _load_json(
        transcript_path,
        label=f"{case_id} visible semantic input transcript",
    )
    expected_job_id = f"sample-{case_id}"
    if queue.get("schemaVersion") != "2.0.0" or queue.get("jobId") != expected_job_id:
        raise AscendReviewTemplateError(f"{case_id} review queue binding changed")
    items = queue.get("items")
    decisions = queue.get("decisions")
    if not isinstance(items, list) or not items:
        raise AscendReviewTemplateError(f"{case_id} review queue has no items")
    if not isinstance(decisions, list) or decisions:
        raise AscendReviewTemplateError(
            f"{case_id} initial queue already contains decisions"
        )
    open_items = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("status") == "open"
    ]
    if queue.get("openCount") != len(open_items) or len(open_items) != len(items):
        raise AscendReviewTemplateError(
            f"{case_id} queue is not a complete initial open queue"
        )
    semantic = _mapping(checkpoint.get("semantic"), field=f"{case_id}.semantic")
    provenance = _mapping(
        semantic.get("provenance"),
        field=f"{case_id}.semantic.provenance",
    )
    if (
        checkpoint.get("status") != "review_required"
        or checkpoint.get("stage") != "review_required"
        or checkpoint.get("reviewOpenCount") != len(open_items)
        or semantic.get("status") != "completed"
        or semantic.get("mode") != "candidate-composition"
        or semantic.get("autoApply") is not True
        or provenance.get("model") != expected_model
    ):
        raise AscendReviewTemplateError(
            f"{case_id} did not reach a 27B semantic review boundary"
        )
    if transcript.get("jobId") != expected_job_id:
        raise AscendReviewTemplateError(
            f"{case_id} visible transcript job binding changed"
        )
    transcript_source = _mapping(
        transcript.get("source"),
        field=f"{case_id}.transcript.source",
    )
    media = _mapping(case.get("media"), field=f"{case_id}.media")
    if transcript_source.get("sha256") != media.get("sha256"):
        raise AscendReviewTemplateError(
            f"{case_id} transcript is not bound to staged media"
        )
    speakers = transcript.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise AscendReviewTemplateError(f"{case_id} transcript has no speakers")
    speaker_ids = [
        str(speaker.get("id"))
        for speaker in speakers
        if isinstance(speaker, Mapping) and isinstance(speaker.get("id"), str)
    ]
    if not speaker_ids or len(set(speaker_ids)) != len(speaker_ids):
        raise AscendReviewTemplateError(
            f"{case_id} transcript speaker IDs are invalid"
        )
    target_speaker = speaker_ids[0]
    public_expected_count = case.get("expectedSpeakerCount")
    suggested_merges: list[dict[str, Any]] = []
    if public_expected_count == 1:
        for index, source_speaker in enumerate(speaker_ids[1:], start=1):
            suggested_merges.append(
                {
                    "type": "speaker.merge",
                    "sourceSpeakerId": source_speaker,
                    "targetSpeakerId": target_speaker,
                    "decisionId": f"{case_id}-merge-{index:04d}",
                    "reason": None,
                    "evidence": [
                        "audio:full",
                        "manifest:expected-speaker-count=1",
                        f"review-queue-file-sha256:{sha256_file(queue_path)}",
                    ],
                    "confidence": None,
                    "audit": {
                        "actor": "codex-semantic-adjudicator",
                        "source": "codex-agent",
                    },
                }
            )

    decision_rows: list[dict[str, Any]] = []
    for index, item in enumerate(open_items, start=1):
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            raise AscendReviewTemplateError(
                f"{case_id} queue contains an item without id"
            )
        scope = str(item.get("scope") or "")
        decision_rows.append(
            {
                "itemId": item_id,
                "scope": scope,
                "reasonCode": item.get("reasonCode"),
                "timeRange": (
                    dict(item["timeRange"])
                    if isinstance(item.get("timeRange"), Mapping)
                    else None
                ),
                "currentSpeakerId": item.get("speakerId"),
                "visibleText": _visible_text(item),
                "action": None,
                "targetSpeakerId": (
                    target_speaker
                    if scope == "segment" and public_expected_count == 1
                    else None
                ),
                "decisionId": f"{case_id}-review-{index:04d}",
                "reason": None,
                "evidence": [
                    (
                        "audio:full"
                        if not isinstance(item.get("timeRange"), Mapping)
                        else (
                            "audio:"
                            f"{item['timeRange'].get('startMs')}-"
                            f"{item['timeRange'].get('endMs')}ms"
                        )
                    ),
                    f"review-queue-file-sha256:{sha256_file(queue_path)}",
                    "visible-model-output-only",
                ],
                "confidence": None,
                "audit": {
                    "actor": "codex-semantic-adjudicator",
                    "source": "codex-agent",
                },
            }
        )

    return _canonical_document(
        {
            "schemaVersion": "1.0.0",
            "artifactType": TEMPLATE_ARTIFACT_TYPE,
            "caseId": case_id,
            "jobId": expected_job_id,
            "automaticScoring": False,
            "executable": False,
            "createdAt": datetime.now(UTC).isoformat(),
            "blindBoundary": {
                "publicTruthRedactedManifestOnly": True,
                "visibleAudioReviewed": False,
                "visibleTranscriptReviewed": False,
                "visibleSpeakerTimelineReviewed": False,
                "scorerTruthRead": False,
                "identityVaultRead": False,
                "referenceTranscriptRead": False,
            },
            "binding": {
                "heldOutManifest": {
                    **_sha_binding(manifest_path),
                    "canonicalSha256": manifest["canonicalSha256"],
                },
                "reviewQueue": _sha_binding(queue_path),
                "semanticInputTranscript": _sha_binding(transcript_path),
                "checkpoint": _sha_binding(checkpoint_path),
                "sourceMediaSha256": media.get("sha256"),
                "semanticModel": expected_model,
            },
            "publicCaseEvidence": {
                "expectedSpeakerCount": public_expected_count,
                "scenarios": list(case.get("scenarios") or []),
                "languageTags": list(case.get("languageTags") or []),
                "durationSeconds": media.get("durationSeconds"),
            },
            "visibleOutputEvidence": {
                "language": transcript.get("language"),
                "speakerIds": speaker_ids,
                "segmentCount": len(transcript.get("segments") or []),
                "reviewOpenCount": len(open_items),
                "semanticCompositionCompleted": True,
                "semanticAutoApplied": True,
            },
            "suggestedPreReviewCommands": suggested_merges,
            "decisions": decision_rows,
        }
    )


def build_templates(
    *,
    manifest_path: Path,
    outputs_root: Path,
    template_root: Path,
    expected_model: str,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    manifest_path, manifest = _validate_manifest(manifest_path)
    outputs_root = outputs_root.expanduser().resolve(strict=True)
    target = template_root.expanduser().absolute().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if not outputs_root.is_dir() or outputs_root.is_symlink():
        raise AscendReviewTemplateError(
            "production outputs root must be a regular directory"
        )
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not rows:
        raise AscendReviewTemplateError("held-out manifest has no cases")
    available = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    selected = list(case_ids) or list(available)
    if len(set(selected)) != len(selected):
        raise AscendReviewTemplateError("case selection contains duplicates")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise AscendReviewTemplateError(
            "unknown held-out case IDs: " + ", ".join(unknown)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".ascend-review-templates-", dir=target.parent)
    )
    try:
        index_rows: list[dict[str, Any]] = []
        for case_id in selected:
            template = _template_for_case(
                case=available[case_id],
                outputs_root=outputs_root,
                manifest_path=manifest_path,
                manifest=manifest,
                expected_model=expected_model,
            )
            filename = f"{case_id}.review-decisions.template.v1.json"
            path = temporary / filename
            atomic_write_json(path, template)
            index_rows.append(
                {
                    "caseId": case_id,
                    "path": filename,
                    "fileSha256": sha256_file(path),
                    "canonicalSha256": template["canonicalSha256"],
                    "openCount": len(template["decisions"]),
                }
            )
        index = _canonical_document(
            {
                "schemaVersion": "1.0.0",
                "artifactType": "ascend-held-out-review-decision-template-set",
                "createdAt": datetime.now(UTC).isoformat(),
                "automaticScoring": False,
                "executable": False,
                "blindBoundary": {
                    "scorerTruthRead": False,
                    "identityVaultRead": False,
                    "referenceTranscriptRead": False,
                },
                "heldOutManifest": {
                    **_sha_binding(manifest_path),
                    "canonicalSha256": manifest["canonicalSha256"],
                },
                "semanticModel": expected_model,
                "caseCount": len(index_rows),
                "cases": index_rows,
                "publication": {
                    "policy": "atomic-directory-no-replace",
                    "manifestWrittenLast": True,
                },
            }
        )
        index_path = temporary / INDEX_NAME
        atomic_write_json(index_path, index)
        _rename_directory_no_replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    final_index = target / INDEX_NAME
    return {
        "templateRoot": str(target),
        "index": str(final_index),
        "indexFileSha256": sha256_file(final_index),
        "indexCanonicalSha256": index["canonicalSha256"],
        "caseCount": len(index_rows),
        "scorerTruthRead": False,
        "identityVaultRead": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--expected-model", default=DEFAULT_EXPECTED_MODEL)
    parser.add_argument("--case", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_templates(
        manifest_path=args.manifest,
        outputs_root=args.outputs_root,
        template_root=args.template_root,
        expected_model=args.expected_model,
        case_ids=args.case,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
