"""Apply a development-frozen speaker threshold to a disjoint held-out set."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402


SUPPORTED_BENCHMARKS = frozenset(
    {
        "frozen-speaker-verification-trials",
        "redimnet2-frozen-speaker-verification",
        "w2vbert2-frozen-speaker-verification",
    }
)


class FixedSpeakerThresholdError(RuntimeError):
    """Raised when development evidence cannot govern held-out scoring."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixedSpeakerThresholdError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixedSpeakerThresholdError(f"{field} must be non-empty text")
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedSpeakerThresholdError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FixedSpeakerThresholdError(f"{field} must be finite")
    return result


def _read_canonical(path: Path, field: str) -> tuple[dict[str, Any], Path]:
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedSpeakerThresholdError(f"{field} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise FixedSpeakerThresholdError(f"{field} must be an object")
    declared = value.get("canonicalSha256")
    body = dict(value)
    body.pop("canonicalSha256", None)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or canonical_json_sha256(body) != declared
    ):
        raise FixedSpeakerThresholdError(
            f"{field} canonical SHA-256 does not match"
        )
    return value, resolved


def _manifest_from_report(
    report: Mapping[str, Any], label: str
) -> dict[str, Any]:
    evidence = _object(report.get("trialManifest"), f"{label}.trialManifest")
    manifest_path = Path(
        _text(evidence.get("path"), f"{label}.trialManifest.path")
    ).resolve(strict=True)
    if evidence.get("fileSha256") != sha256_file(manifest_path):
        raise FixedSpeakerThresholdError(
            f"{label} linked trial manifest file SHA-256 differs"
        )
    manifest, _ = _read_canonical(
        manifest_path, f"{label} linked trial manifest"
    )
    if evidence.get("canonicalSha256") != manifest["canonicalSha256"]:
        raise FixedSpeakerThresholdError(
            f"{label} linked trial manifest identity differs"
        )
    raw_clips = manifest.get("clips")
    raw_trials = manifest.get("trials")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise FixedSpeakerThresholdError(f"{label} manifest has no clips")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise FixedSpeakerThresholdError(f"{label} manifest has no trials")

    clips: dict[str, dict[str, str]] = {}
    splits: set[str] = set()
    for index, value in enumerate(raw_clips):
        clip = _object(value, f"{label}.clips[{index}]")
        clip_id = _text(clip.get("clipId"), f"{label}.clips[{index}].clipId")
        speaker_id = _text(
            clip.get("speakerId"), f"{label}.clips[{index}].speakerId"
        )
        recording_id = _text(
            clip.get("originalRecordingId"),
            f"{label}.clips[{index}].originalRecordingId",
        )
        split = _text(
            clip.get("evaluationSplit"),
            f"{label}.clips[{index}].evaluationSplit",
        )
        if clip_id in clips:
            raise FixedSpeakerThresholdError(f"{label} has duplicate clips")
        clips[clip_id] = {
            "speakerId": speaker_id,
            "recordingId": recording_id,
        }
        splits.add(split)
    if len(splits) != 1:
        raise FixedSpeakerThresholdError(f"{label} manifest mixes splits")

    identities: list[tuple[str, str, str, bool]] = []
    for index, value in enumerate(raw_trials):
        trial = _object(value, f"{label}.trials[{index}]")
        trial_id = _text(
            trial.get("trialId"), f"{label}.trials[{index}].trialId"
        )
        left_id = _text(
            trial.get("enrollmentClipId"),
            f"{label}.trials[{index}].enrollmentClipId",
        )
        right_id = _text(
            trial.get("testClipId"),
            f"{label}.trials[{index}].testClipId",
        )
        same_speaker = trial.get("sameSpeaker")
        if not isinstance(same_speaker, bool):
            raise FixedSpeakerThresholdError(f"{label} trial truth is invalid")
        try:
            left = clips[left_id]
            right = clips[right_id]
        except KeyError as exc:
            raise FixedSpeakerThresholdError(
                f"{label} trial references an unknown clip"
            ) from exc
        if left["recordingId"] == right["recordingId"]:
            raise FixedSpeakerThresholdError(
                f"{label} trial is not cross-recording"
            )
        if (left["speakerId"] == right["speakerId"]) is not same_speaker:
            raise FixedSpeakerThresholdError(
                f"{label} trial truth differs from speaker identity"
            )
        identities.append((trial_id, left_id, right_id, same_speaker))
    if len(set(identities)) != len(identities):
        raise FixedSpeakerThresholdError(f"{label} has duplicate trials")
    return {
        "path": manifest_path,
        "canonicalSha256": manifest["canonicalSha256"],
        "evaluationSplit": next(iter(splits)),
        "speakerIds": frozenset(item["speakerId"] for item in clips.values()),
        "identities": tuple(identities),
        "source": _object(manifest.get("source"), f"{label}.source"),
    }


def _normalized_report(path: Path, label: str) -> dict[str, Any]:
    report, resolved = _read_canonical(path, f"{label} report")
    benchmark = report.get("benchmark")
    if report.get("schemaVersion") != "1.0.0" or benchmark not in SUPPORTED_BENCHMARKS:
        raise FixedSpeakerThresholdError(f"{label} report type is unsupported")
    manifest = _manifest_from_report(report, label)
    model = _object(report.get("model"), f"{label}.model")
    if benchmark == "frozen-speaker-verification-trials":
        threshold = _finite(
            _object(report.get("scores"), f"{label}.scores").get(
                "equalErrorThreshold"
            ),
            f"{label}.scores.equalErrorThreshold",
        )
        raw_scores = report.get("trials")
        score_field = "cosineScore"
        descriptive = _object(report.get("scores"), f"{label}.scores")
        descriptive_eer = descriptive.get("equalErrorRate")
        descriptive_auc = descriptive.get("rocAuc")
    else:
        metrics = _object(report.get("metrics"), f"{label}.metrics")
        threshold = _finite(
            metrics.get("eerThreshold"), f"{label}.metrics.eerThreshold"
        )
        raw_scores = report.get("trialScores")
        score_field = "cosine"
        descriptive_eer = metrics.get("eer")
        descriptive_auc = metrics.get("auc")
    if threshold < -1.0 or threshold > 1.0:
        raise FixedSpeakerThresholdError(f"{label} threshold is not cosine")
    if not isinstance(raw_scores, list) or not raw_scores:
        raise FixedSpeakerThresholdError(f"{label} report has no trial scores")
    scored: list[tuple[tuple[str, str, str, bool], float]] = []
    for index, value in enumerate(raw_scores):
        trial = _object(value, f"{label}.trialScores[{index}]")
        identity = (
            _text(trial.get("trialId"), f"{label}.trialId"),
            _text(trial.get("enrollmentClipId"), f"{label}.enrollmentClipId"),
            _text(trial.get("testClipId"), f"{label}.testClipId"),
            trial.get("sameSpeaker"),
        )
        if not isinstance(identity[3], bool):
            raise FixedSpeakerThresholdError(f"{label} score truth is invalid")
        score = _finite(trial.get(score_field), f"{label}.{score_field}")
        if score < -1.0 or score > 1.0:
            raise FixedSpeakerThresholdError(f"{label} score is not cosine")
        scored.append((identity, score))
    if tuple(item[0] for item in scored) != manifest["identities"]:
        raise FixedSpeakerThresholdError(
            f"{label} scored trial identities differ from the manifest"
        )
    return {
        "path": resolved,
        "fileSha256": sha256_file(resolved),
        "canonicalSha256": report["canonicalSha256"],
        "benchmark": benchmark,
        "model": model,
        "modelIdentitySha256": canonical_json_sha256(model),
        "manifest": manifest,
        "threshold": threshold,
        "scores": tuple(scored),
        "descriptiveEer": _finite(descriptive_eer, f"{label}.descriptiveEer"),
        "descriptiveAuc": _finite(descriptive_auc, f"{label}.descriptiveAuc"),
    }


def _metrics_at_threshold(
    scores: Sequence[tuple[tuple[str, str, str, bool], float]],
    threshold: float,
) -> dict[str, float | int]:
    genuine = [score for identity, score in scores if identity[3]]
    impostor = [score for identity, score in scores if not identity[3]]
    if not genuine or not impostor:
        raise FixedSpeakerThresholdError("both trial classes are required")
    false_rejects = sum(score < threshold for score in genuine)
    false_accepts = sum(score >= threshold for score in impostor)
    false_reject_rate = false_rejects / len(genuine)
    false_accept_rate = false_accepts / len(impostor)
    return {
        "threshold": threshold,
        "genuineTrials": len(genuine),
        "impostorTrials": len(impostor),
        "falseRejects": false_rejects,
        "falseAccepts": false_accepts,
        "falseRejectRate": false_reject_rate,
        "falseAcceptRate": false_accept_rate,
        "balancedAccuracy": 1.0 - (false_reject_rate + false_accept_rate) / 2.0,
    }


def evaluate_fixed_threshold(
    *, development_report_path: Path, held_out_report_path: Path
) -> dict[str, Any]:
    development = _normalized_report(development_report_path, "development")
    held_out = _normalized_report(held_out_report_path, "heldOut")
    if development["manifest"]["evaluationSplit"] != "development":
        raise FixedSpeakerThresholdError(
            "threshold source must be the development split"
        )
    if held_out["manifest"]["evaluationSplit"] != "held-out":
        raise FixedSpeakerThresholdError("target must be the held-out split")
    if development["modelIdentitySha256"] != held_out["modelIdentitySha256"]:
        raise FixedSpeakerThresholdError(
            "development and held-out reports use different models"
        )
    overlap = sorted(
        development["manifest"]["speakerIds"]
        & held_out["manifest"]["speakerIds"]
    )
    if overlap:
        raise FixedSpeakerThresholdError(
            "development and held-out speakers are not disjoint"
        )
    threshold = development["threshold"]
    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "evaluation": "development-frozen-speaker-threshold",
        "model": development["model"],
        "modelIdentitySha256": development["modelIdentitySha256"],
        "thresholdPolicy": {
            "sourceSplit": "development",
            "sourceMetric": "equal-error operating point",
            "threshold": threshold,
            "heldOutThresholdFittingPerformed": False,
            "speakerDisjointSplitsVerified": True,
            "crossRecordingTrialsVerified": True,
        },
        "development": {
            "reportPath": str(development["path"]),
            "reportFileSha256": development["fileSha256"],
            "reportCanonicalSha256": development["canonicalSha256"],
            "trialManifestCanonicalSha256": development["manifest"][
                "canonicalSha256"
            ],
            "speakerCount": len(development["manifest"]["speakerIds"]),
            "metricsAtFrozenThreshold": _metrics_at_threshold(
                development["scores"], threshold
            ),
        },
        "heldOut": {
            "reportPath": str(held_out["path"]),
            "reportFileSha256": held_out["fileSha256"],
            "reportCanonicalSha256": held_out["canonicalSha256"],
            "trialManifestCanonicalSha256": held_out["manifest"][
                "canonicalSha256"
            ],
            "speakerCount": len(held_out["manifest"]["speakerIds"]),
            "metricsAtDevelopmentFrozenThreshold": _metrics_at_threshold(
                held_out["scores"], threshold
            ),
            "descriptiveOnly": {
                "eerFittedOnHeldOut": held_out["descriptiveEer"],
                "auc": held_out["descriptiveAuc"],
                "promotionDecisionUsesHeldOutEerThreshold": False,
            },
        },
    }
    report["canonicalSha256"] = canonical_json_sha256(report)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--held-out-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_fixed_threshold(
        development_report_path=args.development_report,
        held_out_report_path=args.held_out_report,
    )
    _write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonicalSha256": report["canonicalSha256"],
                "thresholdPolicy": report["thresholdPolicy"],
                "heldOut": report["heldOut"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
