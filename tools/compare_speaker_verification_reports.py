"""Compare two canonical speaker-verification reports with hard gates."""

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


class SpeakerVerificationComparisonError(RuntimeError):
    """Raised when reports are not comparable or fail integrity checks."""


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpeakerVerificationComparisonError(f"{field} must be an object")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeakerVerificationComparisonError(f"{field} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise SpeakerVerificationComparisonError(f"{field} must be finite")
    return output


def _load_report(path: Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpeakerVerificationComparisonError(
            f"{label} report is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SpeakerVerificationComparisonError(
            f"{label} report must be an object"
        )
    declared = value.get("canonicalSha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise SpeakerVerificationComparisonError(
            f"{label} report has no canonical SHA-256"
        )
    canonical = dict(value)
    canonical.pop("canonicalSha256", None)
    if canonical_json_sha256(canonical) != declared:
        raise SpeakerVerificationComparisonError(
            f"{label} report canonical SHA-256 does not match"
        )
    if value.get("schemaVersion") != "1.0.0" or value.get("benchmark") not in {
        "frozen-speaker-verification-trials",
        "redimnet2-frozen-speaker-verification",
    }:
        raise SpeakerVerificationComparisonError(
            f"{label} report type is unsupported"
        )
    return value, resolved


def _linked_trial_manifest(
    report: Mapping[str, Any], label: str
) -> dict[str, Any]:
    evidence = _object(report.get("trialManifest"), f"{label}.trialManifest")
    raw_path = evidence.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SpeakerVerificationComparisonError(
            f"{label} report has no linked trial manifest path"
        )
    path = Path(raw_path).resolve(strict=True)
    if evidence.get("fileSha256") != sha256_file(path):
        raise SpeakerVerificationComparisonError(
            f"{label} linked trial manifest file SHA-256 differs"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpeakerVerificationComparisonError(
            f"{label} linked trial manifest is invalid"
        ) from exc
    if not isinstance(manifest, dict):
        raise SpeakerVerificationComparisonError(
            f"{label} linked trial manifest must be an object"
        )
    declared = manifest.get("canonicalSha256")
    canonical = dict(manifest)
    canonical.pop("canonicalSha256", None)
    if (
        not isinstance(declared, str)
        or canonical_json_sha256(canonical) != declared
        or evidence.get("canonicalSha256") != declared
    ):
        raise SpeakerVerificationComparisonError(
            f"{label} linked trial manifest canonical SHA-256 differs"
        )
    source = _object(manifest.get("source"), f"{label}.manifest.source")
    clips = manifest.get("clips")
    trials = manifest.get("trials")
    counts = _object(manifest.get("counts"), f"{label}.manifest.counts")
    if not isinstance(clips, list) or not clips or not isinstance(trials, list) or not trials:
        raise SpeakerVerificationComparisonError(
            f"{label} linked trial manifest has no cases"
        )
    splits = {
        clip.get("evaluationSplit")
        for clip in clips
        if isinstance(clip, Mapping)
    }
    if len(splits) != 1:
        raise SpeakerVerificationComparisonError(
            f"{label} linked trial manifest mixes evaluation splits"
        )
    return {
        "path": path,
        "canonicalSha256": declared,
        "source": source,
        "clips": clips,
        "trials": trials,
        "counts": counts,
        "evaluationSplit": next(iter(splits)),
    }


def _trial_identity(
    trials: Any, *, score_field: str | None, label: str
) -> tuple[Any, ...]:
    if not isinstance(trials, list) or not trials:
        raise SpeakerVerificationComparisonError(
            f"{label} report contains no trials"
        )
    identity: list[tuple[Any, ...]] = []
    for index, value in enumerate(trials):
        trial = _object(value, f"{label}.trials[{index}]")
        if score_field is not None:
            score = _finite(
                trial.get(score_field),
                f"{label}.trials[{index}].{score_field}",
            )
            if score < -1.0 or score > 1.0:
                raise SpeakerVerificationComparisonError(
                    f"{label} trial score is outside cosine bounds"
                )
        identity.append(
            (
                trial.get("trialId"),
                trial.get("enrollmentClipId"),
                trial.get("testClipId"),
                trial.get("sameSpeaker"),
            )
        )
    if len(set(identity)) != len(identity):
        raise SpeakerVerificationComparisonError(
            f"{label} report contains duplicate trials"
        )
    return tuple(identity)


def _normalize_report(
    report: Mapping[str, Any], path: Path, label: str
) -> dict[str, Any]:
    manifest = _linked_trial_manifest(report, label)
    manifest_trials = _trial_identity(
        manifest["trials"], score_field=None, label=f"{label}.manifest"
    )
    benchmark = report["benchmark"]
    model = _object(report.get("model"), f"{label}.model")
    if benchmark == "frozen-speaker-verification-trials":
        scored_trials = _trial_identity(
            report.get("trials"),
            score_field="cosineScore",
            label=label,
        )
        scores = _object(report.get("scores"), f"{label}.scores")
        normalized_scores = {
            "rocAuc": _finite(scores.get("rocAuc"), f"{label}.rocAuc"),
            "equalErrorRate": _finite(
                scores.get("equalErrorRate"), f"{label}.equalErrorRate"
            ),
            "minimumDetectionCostP01": _finite(
                scores.get("minimumDetectionCostP01"),
                f"{label}.minimumDetectionCostP01",
            ),
            "balancedAccuracyAtEerThreshold": _finite(
                scores.get("balancedAccuracyAtEerThreshold"),
                f"{label}.balancedAccuracyAtEerThreshold",
            ),
        }
        resources = _object(report.get("resources"), f"{label}.resources")
        peak_reserved = _finite(
            resources.get("peakCudaReservedMb"),
            f"{label}.peakCudaReservedMb",
        )
        retained_allocated: float | None = _finite(
            resources.get("retainedCudaAllocatedMb"),
            f"{label}.retainedCudaAllocatedMb",
        )
        partition = _object(report.get("partition"), f"{label}.partition")
        source_evidence = _object(report.get("source"), f"{label}.source")
        expected = {
            "evaluationSplit": manifest["evaluationSplit"],
            "sourceRecordingId": manifest["source"].get("recordingId"),
            "speakerCount": manifest["counts"].get("speakers"),
            "clipCount": manifest["counts"].get("clips"),
            "trialCount": manifest["counts"].get("totalTrials"),
        }
        if any(partition.get(key) != value for key, value in expected.items()):
            raise SpeakerVerificationComparisonError(
                f"{label} report partition differs from its linked manifest"
            )
        if (
            source_evidence.get("audioSha256")
            != manifest["source"].get("audioSha256")
            or source_evidence.get("annotationSha256")
            != manifest["source"].get("annotationSha256")
        ):
            raise SpeakerVerificationComparisonError(
                f"{label} report source differs from its linked manifest"
            )
        model_key = model.get("modelKey")
    else:
        scored_trials = _trial_identity(
            report.get("trialScores"),
            score_field="cosine",
            label=label,
        )
        scores = _object(report.get("metrics"), f"{label}.metrics")
        normalized_scores = {
            "rocAuc": _finite(scores.get("auc"), f"{label}.auc"),
            "equalErrorRate": _finite(scores.get("eer"), f"{label}.eer"),
            "minimumDetectionCostP01": _finite(
                scores.get("minDcfPTarget0.01"),
                f"{label}.minDcfPTarget0.01",
            ),
            "balancedAccuracyAtEerThreshold": _finite(
                scores.get("balancedAccuracyAtEer"),
                f"{label}.balancedAccuracyAtEer",
            ),
        }
        execution = _object(report.get("execution"), f"{label}.execution")
        peak_reserved = _finite(
            execution.get("peakReservedVramBytes"),
            f"{label}.peakReservedVramBytes",
        ) / (1024.0 * 1024.0)
        retained_allocated = None
        trial_evidence = _object(
            report.get("trialManifest"), f"{label}.trialManifest"
        )
        if (
            trial_evidence.get("evaluationSplit") != manifest["evaluationSplit"]
            or trial_evidence.get("sourceAudioSha256")
            != manifest["source"].get("audioSha256")
            or trial_evidence.get("sourceAnnotationSha256")
            != manifest["source"].get("annotationSha256")
            or execution.get("speakerCount") != manifest["counts"].get("speakers")
            or execution.get("clipCount") != manifest["counts"].get("clips")
            or execution.get("trialCount")
            != manifest["counts"].get("totalTrials")
        ):
            raise SpeakerVerificationComparisonError(
                f"{label} report evidence differs from its linked manifest"
            )
        model_key = model.get("repoId")
    if scored_trials != manifest_trials:
        raise SpeakerVerificationComparisonError(
            f"{label} scored trial identities differ from its linked manifest"
        )
    if not isinstance(model_key, str) or not model_key:
        raise SpeakerVerificationComparisonError(f"{label} model key is missing")
    manifest_sha = model.get("manifestFileSha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise SpeakerVerificationComparisonError(
            f"{label} model manifest SHA-256 is missing"
        )
    return {
        "path": path,
        "fileSha256": sha256_file(path),
        "canonicalSha256": report["canonicalSha256"],
        "modelKey": model_key,
        "modelManifestSha256": manifest_sha,
        "trialManifestCanonicalSha256": manifest["canonicalSha256"],
        "evaluationSplit": manifest["evaluationSplit"],
        "sourceRecordingId": manifest["source"].get("recordingId"),
        "speakerCount": manifest["counts"].get("speakers"),
        "clipCount": manifest["counts"].get("clips"),
        "trialCount": manifest["counts"].get("totalTrials"),
        "audioSha256": manifest["source"].get("audioSha256"),
        "annotationSha256": manifest["source"].get("annotationSha256"),
        "trials": scored_trials,
        "scores": normalized_scores,
        "peakCudaReservedMb": peak_reserved,
        "retainedCudaAllocatedMb": retained_allocated,
    }


def _gate(
    *,
    metric: str,
    baseline: float,
    challenger: float,
    direction: str,
) -> dict[str, Any]:
    if direction == "higher-or-equal":
        passed = challenger >= baseline
    elif direction == "lower-or-equal":
        passed = challenger <= baseline
    else:  # pragma: no cover - private fixed call sites.
        raise ValueError("unsupported comparison direction")
    return {
        "metric": metric,
        "direction": direction,
        "baseline": baseline,
        "challenger": challenger,
        "delta": challenger - baseline,
        "passed": passed,
    }


def compare_reports(
    *,
    baseline_report_path: Path,
    challenger_report_path: Path,
    vram_limit_mb: float,
    retained_vram_limit_mb: float = 64.0,
) -> dict[str, Any]:
    if (
        not math.isfinite(vram_limit_mb)
        or vram_limit_mb <= 0.0
        or not math.isfinite(retained_vram_limit_mb)
        or retained_vram_limit_mb < 0.0
    ):
        raise ValueError("VRAM limits are invalid")
    baseline_raw, baseline_path = _load_report(
        baseline_report_path, "baseline"
    )
    challenger_raw, challenger_path = _load_report(
        challenger_report_path, "challenger"
    )
    baseline = _normalize_report(baseline_raw, baseline_path, "baseline")
    challenger = _normalize_report(
        challenger_raw, challenger_path, "challenger"
    )
    if baseline["modelManifestSha256"] == challenger["modelManifestSha256"]:
        raise SpeakerVerificationComparisonError(
            "baseline and challenger resolve to the same model"
        )
    baseline_trials = baseline["trials"]
    challenger_trials = challenger["trials"]
    if baseline_trials != challenger_trials:
        raise SpeakerVerificationComparisonError(
            "baseline and challenger trial identities differ"
        )
    evidence_fields = (
        "trialManifestCanonicalSha256",
        "evaluationSplit",
        "sourceRecordingId",
        "speakerCount",
        "clipCount",
        "trialCount",
        "audioSha256",
        "annotationSha256",
    )
    for field in evidence_fields:
        if baseline[field] != challenger[field]:
            raise SpeakerVerificationComparisonError(
                f"reports differ at {field}"
            )
    split = baseline["evaluationSplit"]
    if split == "held-out":
        raise SpeakerVerificationComparisonError(
            "raw held-out EER reports cannot establish promotion; compare "
            "development-frozen threshold evidence instead"
        )

    baseline_scores = baseline["scores"]
    challenger_scores = challenger["scores"]
    quality_gates = [
        _gate(
            metric="rocAuc",
            baseline=_finite(baseline_scores.get("rocAuc"), "baseline.rocAuc"),
            challenger=_finite(
                challenger_scores.get("rocAuc"), "challenger.rocAuc"
            ),
            direction="higher-or-equal",
        ),
        _gate(
            metric="equalErrorRate",
            baseline=_finite(
                baseline_scores.get("equalErrorRate"),
                "baseline.equalErrorRate",
            ),
            challenger=_finite(
                challenger_scores.get("equalErrorRate"),
                "challenger.equalErrorRate",
            ),
            direction="lower-or-equal",
        ),
        _gate(
            metric="minimumDetectionCostP01",
            baseline=_finite(
                baseline_scores.get("minimumDetectionCostP01"),
                "baseline.minimumDetectionCostP01",
            ),
            challenger=_finite(
                challenger_scores.get("minimumDetectionCostP01"),
                "challenger.minimumDetectionCostP01",
            ),
            direction="lower-or-equal",
        ),
        _gate(
            metric="balancedAccuracyAtEerThreshold",
            baseline=_finite(
                baseline_scores.get("balancedAccuracyAtEerThreshold"),
                "baseline.balancedAccuracyAtEerThreshold",
            ),
            challenger=_finite(
                challenger_scores.get("balancedAccuracyAtEerThreshold"),
                "challenger.balancedAccuracyAtEerThreshold",
            ),
            direction="higher-or-equal",
        ),
    ]
    peak_reserved = challenger["peakCudaReservedMb"]
    retained_allocated = challenger["retainedCudaAllocatedMb"]
    resource_gates = [
        {
            "metric": "peakCudaReservedMb",
            "limit": vram_limit_mb,
            "challenger": peak_reserved,
            "evidenceAvailable": True,
            "passed": peak_reserved <= vram_limit_mb,
        },
        {
            "metric": "retainedCudaAllocatedMb",
            "limit": retained_vram_limit_mb,
            "challenger": retained_allocated,
            "evidenceAvailable": retained_allocated is not None,
            "passed": (
                retained_allocated is not None
                and retained_allocated <= retained_vram_limit_mb
            ),
        },
    ]
    quality_passed = all(item["passed"] for item in quality_gates)
    resources_passed = all(item["passed"] for item in resource_gates)
    peak_resource_passed = resource_gates[0]["passed"]
    if not quality_passed:
        decision = "reject-development-or-regression-regression"
    elif not peak_resource_passed:
        decision = "reject-resource-gate"
    elif retained_allocated is None:
        decision = "advance-to-runtime-release-validation"
    elif not resources_passed:
        decision = "reject-resource-release-gate"
    else:
        decision = "advance-to-held-out"
    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "comparison": "speaker-verification-hard-gate",
        "baselineReport": {
            "path": str(baseline["path"]),
            "fileSha256": baseline["fileSha256"],
            "canonicalSha256": baseline["canonicalSha256"],
            "modelKey": baseline["modelKey"],
            "modelManifestSha256": baseline["modelManifestSha256"],
        },
        "challengerReport": {
            "path": str(challenger["path"]),
            "fileSha256": challenger["fileSha256"],
            "canonicalSha256": challenger["canonicalSha256"],
            "modelKey": challenger["modelKey"],
            "modelManifestSha256": challenger["modelManifestSha256"],
        },
        "evidence": {
            "trialManifestCanonicalSha256": baseline[
                "trialManifestCanonicalSha256"
            ],
            "evaluationSplit": split,
            "sourceRecordingId": baseline["sourceRecordingId"],
            "recordingCount": 1,
            "speakerCount": baseline["speakerCount"],
            "clipCount": baseline["clipCount"],
            "trialCount": len(baseline_trials),
            "trialIdentitiesMatch": True,
        },
        "gates": {
            "quality": quality_gates,
            "resources": resource_gates,
            "qualityPassed": quality_passed,
            "resourcesPassed": resources_passed,
            "candidateAdvancementAllowed": (
                quality_passed and peak_resource_passed
            ),
            "heldOutRequired": True,
            "heldOutSatisfied": False,
        },
        "decision": {
            "status": decision,
            "productionPromotionAllowed": False,
        },
        "limitations": [
            "The evidence contains one source recording and is not recording-independent replication.",
            "Trials share clips and must not be treated as 512 independent recordings.",
            "Speaker verification quality does not by itself prove diarization DER/JER quality.",
        ],
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
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--challenger-report", type=Path, required=True)
    parser.add_argument("--vram-limit-mb", type=float, required=True)
    parser.add_argument("--retained-vram-limit-mb", type=float, default=64.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_reports(
        baseline_report_path=args.baseline_report,
        challenger_report_path=args.challenger_report,
        vram_limit_mb=args.vram_limit_mb,
        retained_vram_limit_mb=args.retained_vram_limit_mb,
    )
    _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
