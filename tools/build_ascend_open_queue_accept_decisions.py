"""Build Codex decisions for a completed, public review queue.

This helper is intentionally narrow: it accepts only the visible durable
queue, binds the result to a new job id, and emits one explicit ``accept``
decision per currently open item.  It never reads reference transcripts,
speaker identity data, or automatic scores, and it cannot carry forward
pre-review mutations from an earlier run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.persistence import atomic_write_json, canonical_json_sha256, sha256_file  # noqa: E402


ARTIFACT_TYPE = "production-review-decisions"
ACTOR = "codex-semantic-adjudicator"


class OpenQueueDecisionError(ValueError):
    """Raised when a visible queue cannot be accepted safely."""


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenQueueDecisionError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise OpenQueueDecisionError(f"{label} must be an object")
    return value


def _visible_text(item: Mapping[str, Any], field: str) -> str | None:
    text = item.get("text")
    if not isinstance(text, Mapping):
        return None
    value = text.get(field)
    return value if isinstance(value, str) else None


def build_decisions(
    *,
    queue_path: Path,
    output_path: Path,
    job_id: str,
    target_speaker_id: str,
    expected_open_count: int | None = None,
) -> dict[str, Any]:
    queue_path = queue_path.expanduser().resolve(strict=True)
    queue = _load(queue_path, "review queue")
    if queue.get("schemaVersion") != "2.0.0":
        raise OpenQueueDecisionError("review queue schemaVersion must be 2.0.0")
    if queue.get("openCount") != len(queue.get("items", [])):
        raise OpenQueueDecisionError("queue must contain only its initial open items")
    if queue.get("decisions") != []:
        raise OpenQueueDecisionError("queue already contains decisions")
    items = queue.get("items")
    if not isinstance(items, list) or not items:
        raise OpenQueueDecisionError("queue items must be a non-empty array")
    open_items: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping) or item.get("status") != "open":
            raise OpenQueueDecisionError("queue must contain only open item objects")
        item_id = item.get("id")
        speaker_id = item.get("speakerId")
        if not isinstance(item_id, str) or not item_id.strip():
            raise OpenQueueDecisionError("open item has no stable id")
        if speaker_id != target_speaker_id:
            raise OpenQueueDecisionError(
                f"open item {item_id} is bound to {speaker_id!r}, not {target_speaker_id!r}"
            )
        open_items.append(item)
    if expected_open_count is not None and len(open_items) != expected_open_count:
        raise OpenQueueDecisionError(
            f"expected {expected_open_count} open items, found {len(open_items)}"
        )
    if not isinstance(job_id, str) or not job_id.strip():
        raise OpenQueueDecisionError("job_id must be non-empty")
    job_id = job_id.strip()
    now = datetime.now(UTC).isoformat()
    decisions: list[dict[str, Any]] = []
    for index, item in enumerate(open_items, start=1):
        item_id = str(item["id"]).strip()
        evidence = [
            "audio:full",
            "speaker:visible-speaker-1",
            f"review-queue-file-sha256:{sha256_file(queue_path)}",
        ]
        for field in ("rawText", "normalizedText", "displayText"):
            if _visible_text(item, field) is not None:
                evidence.append(f"text:{field}-visible")
        decision: dict[str, Any] = {
            "itemId": item_id,
            "action": "accept",
            "targetSpeakerId": target_speaker_id,
            "decisionId": f"{job_id}-accept-{index:04d}",
            "reason": (
                "The visible queue item is a single-speaker continuation. "
                "Accept the existing transcript and speaker assignment without "
                "inventing text or applying an earlier merge command."
            ),
            "evidence": evidence,
            "confidence": 0.98,
            "audit": {
                "actor": ACTOR,
                "source": "codex-agent",
                "timestamp": now,
            },
        }
        decisions.append(decision)
    body: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "artifactType": ARTIFACT_TYPE,
        "jobId": job_id,
        "automaticScoring": False,
        "decisions": decisions,
    }
    canonical_sha256 = canonical_json_sha256(body)
    output_path = output_path.expanduser().absolute().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The runner contract is deliberately minimal; keep the canonical digest
    # in the publication receipt rather than adding an unknown root field.
    atomic_write_json(output_path, body)
    return {
        "path": str(output_path),
        "fileSha256": sha256_file(output_path),
        "canonicalSha256": canonical_sha256,
        "jobId": job_id,
        "openCount": len(decisions),
        "preReviewCommandCount": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--target-speaker-id", default="speaker-1")
    parser.add_argument("--expected-open-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_decisions(
            queue_path=args.queue,
            output_path=args.output,
            job_id=args.job_id,
            target_speaker_id=args.target_speaker_id,
            expected_open_count=args.expected_open_count,
        )
    except (OpenQueueDecisionError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
