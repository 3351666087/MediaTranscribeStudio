"""Materialize Codex's public-evidence adjudication as runner decisions.

The input is the non-executable template set produced by
``build_ascend_review_decision_templates``.  This tool deliberately consumes
only visible queue/template fields and a small, explicit case policy.  It has
no scorer-vault, identity-vault, reference transcript, or automatic-score
input.  The output is the strict ``production-review-decisions`` contract
accepted by ``run_sample_library``.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import (  # noqa: E402
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)


ARTIFACT_TYPE = "production-review-decisions"
INDEX_ARTIFACT_TYPE = "ascend-held-out-codex-review-decision-set"
DEFAULT_TEMPLATE_ROOT = Path(
    "/mnt/d/mts-eval/semantic-heldout/ascend6-qwen35-27b-vs-9b-20260809"
    "/review-decision-templates-r1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/d/mts-eval/semantic-heldout/ascend6-qwen35-27b-vs-9b-20260809"
    "/review-decisions-r1"
)
ACTOR = "codex-semantic-adjudicator"


class CodexDecisionBuildError(ValueError):
    """Raised when a visible template cannot be made executable safely."""


# These notes are the actual public-evidence baseline.  In particular, they
# state where the adjudicator refuses to infer words or language labels.
CASE_POLICIES: dict[str, dict[str, Any]] = {
    "ascend-held-test-01283": {
        "mergeSources": ["speaker-2"],
        "reason": (
            "Public manifest declares one speaker and the visible Pyannote "
            "regular/exclusive timeline is one continuous SPEAKER_00. The "
            "boundary is an intra-utterance code switch; keep both ASR text "
            "fragments exactly and do not invent the missing object."
        ),
        "confidence": 0.98,
    },
    "ascend-held-test-00368": {
        # The visible post-composition transcript exposes only speaker-1/2;
        # earlier internal cardinality windows are not executable evidence.
        "mergeSources": ["speaker-2"],
        "reason": (
            "Public manifest declares one speaker and the visible Pyannote "
            "regular/exclusive timeline is one continuous SPEAKER_00. The "
            "apparent turns are code-switch/acoustic over-splits; preserve "
            "the English text and do not fabricate a language-label repair."
        ),
        "confidence": 0.98,
    },
    "ascend-held-test-00228": {
        "mergeSources": ["speaker-2"],
        "reason": (
            "Public manifest declares one speaker and the visible Pyannote "
            "regular/exclusive timeline is one continuous SPEAKER_00. Keep "
            "the code-switch text literally, including 'U G'; visible "
            "evidence does not justify expanding it to a named entity."
        ),
        "confidence": 0.98,
    },
    "ascend-held-test-00386": {
        "mergeSources": ["speaker-2", "speaker-3"],
        "reason": (
            "Public manifest declares one speaker and the visible Pyannote "
            "regular/exclusive timeline is one continuous SPEAKER_00 across "
            "all five segments. Merge the false speaker partitions and "
            "preserve every ASR fragment, including code-switch wording."
        ),
        "confidence": 0.98,
    },
    "ascend-held-test-00955": {
        "mergeSources": ["speaker-2"],
        "reason": (
            "Public manifest declares one speaker and the visible Pyannote "
            "regular/exclusive timeline is one continuous SPEAKER_00. The "
            "second segment continues the same utterance; merge the acoustic "
            "over-split and keep the mixed-language ASR text unchanged."
        ),
        "confidence": 0.98,
    },
    "ascend-held-test-00919": {
        "mergeSources": ["speaker-2", "speaker-3", "speaker-4"],
        "reason": (
            "Public manifest declares one speaker and both visible Pyannote "
            "regular/exclusive timelines contain one continuous SPEAKER_00 "
            "from 31 to 3322 ms. The four short cardinality windows are one "
            "continuous education utterance, not four speakers. Merge the "
            "false partitions and preserve every visible ASR fragment "
            "literally, including 'P A'; visible evidence does not justify "
            "expanding or rewriting it."
        ),
        "confidence": 0.98,
    },
}


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CodexDecisionBuildError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CodexDecisionBuildError(f"{label} must be an object")
    return value


def _manual_fields(
    *,
    decision_id: str,
    reason: str,
    evidence: Sequence[str],
    confidence: float,
) -> dict[str, Any]:
    return {
        "decisionId": decision_id,
        "reason": reason,
        "evidence": list(dict.fromkeys(evidence)),
        "confidence": confidence,
        "audit": {"actor": ACTOR, "source": "codex-agent"},
    }


def _build_case(template: Mapping[str, Any]) -> dict[str, Any]:
    case_id = template.get("caseId")
    if not isinstance(case_id, str) or case_id not in CASE_POLICIES:
        raise CodexDecisionBuildError(f"no Codex policy for template case: {case_id}")
    if template.get("executable") is not False:
        raise CodexDecisionBuildError(f"template {case_id} is already executable")
    policy = CASE_POLICIES[case_id]
    job_id = template.get("jobId")
    if not isinstance(job_id, str) or not job_id.strip():
        raise CodexDecisionBuildError(f"{case_id} has no jobId")
    visible = template.get("visibleOutputEvidence")
    if not isinstance(visible, Mapping) or visible.get("semanticCompositionCompleted") is not True:
        raise CodexDecisionBuildError(f"{case_id} is not a completed semantic review boundary")
    queue_hash = template.get("binding", {}).get("reviewQueue", {}).get("fileSha256")
    if not isinstance(queue_hash, str) or len(queue_hash) != 64:
        raise CodexDecisionBuildError(f"{case_id} has no queue hash binding")

    merge_sources = list(policy["mergeSources"])
    suggested = template.get("suggestedPreReviewCommands")
    if not isinstance(suggested, list):
        raise CodexDecisionBuildError(f"{case_id} suggested merge commands are missing")
    pre_review: list[dict[str, Any]] = []
    for command in suggested:
        if not isinstance(command, Mapping):
            raise CodexDecisionBuildError(f"{case_id} contains an invalid merge command")
        source = command.get("sourceSpeakerId")
        target = command.get("targetSpeakerId")
        if source not in merge_sources or target != "speaker-1":
            continue
        pre_review.append(
            {
                "type": "speaker.merge",
                "sourceSpeakerId": source,
                "targetSpeakerId": target,
                **_manual_fields(
                    decision_id=str(command["decisionId"]),
                    reason=str(policy["reason"]),
                    evidence=[
                        "audio:full",
                        "pyannote:full-regular-exclusive-single-speaker",
                        "manifest:expected-speaker-count=1",
                        f"review-queue-file-sha256:{queue_hash}",
                    ],
                    confidence=float(policy["confidence"]),
                ),
            }
        )
    if {row["sourceSpeakerId"] for row in pre_review} != set(merge_sources):
        raise CodexDecisionBuildError(
            f"{case_id} merge policy does not match visible template speakers"
        )

    raw_decisions = template.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise CodexDecisionBuildError(f"{case_id} has no review decisions")
    decisions: list[dict[str, Any]] = []
    for row in raw_decisions:
        if not isinstance(row, Mapping):
            raise CodexDecisionBuildError(f"{case_id} contains an invalid queue row")
        item_id = row.get("itemId")
        decision_id = row.get("decisionId")
        if not isinstance(item_id, str) or not isinstance(decision_id, str):
            raise CodexDecisionBuildError(f"{case_id} queue row lacks stable IDs")
        scope = row.get("scope")
        raw_time = row.get("timeRange")
        evidence = [
            (
                "audio:full"
                if not isinstance(raw_time, Mapping)
                else f"audio:{raw_time.get('startMs')}-{raw_time.get('endMs')}ms"
            ),
            "pyannote:full-regular-exclusive-single-speaker",
            "text:visible-context-continuity",
            f"review-queue-file-sha256:{queue_hash}",
        ]
        reason = str(policy["reason"])
        if scope == "job":
            reason = (
                "The visible full-media timeline supports one continuous speaker; "
                "the automatic count/range is an over-split and is resolved by "
                "the explicit merge commands. "
                + str(policy["reason"])
            )
        item: dict[str, Any] = {
            "itemId": item_id,
            "action": "accept",
            **_manual_fields(
                decision_id=decision_id,
                reason=reason,
                evidence=evidence,
                confidence=float(policy["confidence"]),
            ),
        }
        if scope == "segment":
            target = row.get("targetSpeakerId")
            if not isinstance(target, str) or not target.strip():
                raise CodexDecisionBuildError(
                    f"{case_id} segment decision has no target speaker"
                )
            item["targetSpeakerId"] = target
        decisions.append(item)
    return {
        "schemaVersion": "1.0.0",
        "artifactType": ARTIFACT_TYPE,
        "jobId": job_id,
        "automaticScoring": False,
        "preReviewCommands": pre_review,
        "decisions": decisions,
    }


def build_decision_set(
    *,
    template_root: Path,
    output_root: Path,
    case_ids: Sequence[str] = (),
) -> dict[str, Any]:
    template_root = template_root.expanduser().resolve(strict=True)
    index = _load(template_root / "MANIFEST.v1.json", label="template index")
    if index.get("executable") is not False or index.get("automaticScoring") is not False:
        raise CodexDecisionBuildError("template index is not a non-executable blind set")
    rows = index.get("cases")
    if not isinstance(rows, list) or not rows:
        raise CodexDecisionBuildError("template index has no cases")
    selected = list(case_ids) or [str(row.get("caseId")) for row in rows]
    if len(set(selected)) != len(selected):
        raise CodexDecisionBuildError("duplicate case selection")
    available = {str(row.get("caseId")): row for row in rows if isinstance(row, Mapping)}
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise CodexDecisionBuildError("unknown template cases: " + ", ".join(unknown))
    target = output_root.expanduser().absolute().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ascend-codex-decisions-", dir=target.parent))
    try:
        out_rows: list[dict[str, Any]] = []
        for case_id in selected:
            template_path = (template_root / str(available[case_id]["path"])).resolve(strict=True)
            template = _load(template_path, label=f"{case_id} template")
            decision_set = _build_case(template)
            filename = f"{case_id}.review-decisions.json"
            path = temporary / filename
            atomic_write_json(path, decision_set)
            out_rows.append(
                {
                    "caseId": case_id,
                    "jobId": decision_set["jobId"],
                    "path": filename,
                    "fileSha256": sha256_file(path),
                    "decisionCount": len(decision_set["decisions"]),
                    "preReviewCommandCount": len(decision_set["preReviewCommands"]),
                    "auditSource": "codex-agent",
                }
            )
        body = {
            "schemaVersion": "1.0.0",
            "artifactType": INDEX_ARTIFACT_TYPE,
            "createdAt": datetime.now(UTC).isoformat(),
            "automaticScoring": False,
            "executable": True,
            "templateIndexSha256": sha256_file(template_root / "MANIFEST.v1.json"),
            "caseCount": len(out_rows),
            "cases": out_rows,
            "publication": {"policy": "atomic-directory-no-replace"},
        }
        atomic_write_json(
            temporary / "MANIFEST.v1.json",
            {**body, "canonicalSha256": canonical_json_sha256(body)},
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    manifest_path = target / "MANIFEST.v1.json"
    return {
        "outputRoot": str(target),
        "manifest": str(manifest_path),
        "manifestFileSha256": sha256_file(manifest_path),
        "caseCount": len(out_rows),
        "automaticScoring": False,
        "auditSource": "codex-agent",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_decision_set(
            template_root=args.template_root,
            output_root=args.output_root,
            case_ids=args.case,
        )
    except (CodexDecisionBuildError, FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
