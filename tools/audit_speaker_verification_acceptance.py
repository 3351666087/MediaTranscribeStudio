"""Audit frozen speaker candidates and score the development-selected winner.

The winner and its operating threshold are selected from development evidence
only.  Held-out labels are consumed only after the calibrator is frozen and
are emitted solely as aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.persistence import canonical_json_sha256, sha256_file  # noqa: E402
from tools.evaluate_fixed_speaker_threshold import (  # noqa: E402
    FixedSpeakerThresholdError,
    _normalized_report,
)


class SpeakerVerificationAcceptanceError(RuntimeError):
    """Raised when frozen evidence cannot support final aggregate metrics."""


_CALIBRATOR_KEYS = frozenset({"calibrator", "calibrationModel", "scoreCalibrator"})
_PREPROCESSING_KEYS = frozenset(
    {"preprocessing", "preprocessingProfile", "normalizationProfile"}
)
_EMBEDDING_SHA_KEYS = frozenset(
    {"embeddingSha256", "embeddingSetSha256", "vectorSetSha256"}
)
_IDENTITY_KEYS = frozenset(
    {
        "speakerId",
        "clipId",
        "trialId",
        "enrollmentClipId",
        "testClipId",
    }
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<tail>.*)$")


def _host_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a D:/... evidence path on both Windows and WSL."""

    raw = os.fspath(value)
    match = _WINDOWS_ABSOLUTE_RE.fullmatch(raw)
    if os.name != "nt" and match is not None:
        tail = PurePosixPath(match.group("tail").replace("\\", "/"))
        return Path("/mnt") / match.group("drive").lower() / Path(*tail.parts)
    return Path(raw)


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpeakerVerificationAcceptanceError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpeakerVerificationAcceptanceError(
            f"{field} must be non-empty text"
        )
    return value.strip()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeakerVerificationAcceptanceError(f"{field} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise SpeakerVerificationAcceptanceError(f"{field} must be finite")
    return output


def _read_canonical(path: Path, field: str) -> tuple[dict[str, Any], Path]:
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpeakerVerificationAcceptanceError(
            f"{field} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SpeakerVerificationAcceptanceError(f"{field} must be an object")
    declared = value.get("canonicalSha256")
    canonical = dict(value)
    canonical.pop("canonicalSha256", None)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or canonical_json_sha256(canonical) != declared
    ):
        raise SpeakerVerificationAcceptanceError(
            f"{field} canonical SHA-256 does not match"
        )
    return value, resolved


def _nested_value(document: Any, names: frozenset[str]) -> Any | None:
    if isinstance(document, Mapping):
        for key, value in document.items():
            if str(key) in names and value not in (None, "", [], {}):
                return value
            found = _nested_value(value, names)
            if found is not None:
                return found
    elif isinstance(document, (list, tuple)):
        for value in document:
            found = _nested_value(value, names)
            if found is not None:
                return found
    return None


def _model_revision(model: Mapping[str, Any]) -> str | None:
    value = model.get("revision") or model.get("releaseTag")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nested_mapping_value(document: Any, path: Sequence[str]) -> Any | None:
    current = document
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _config_file_for_model(
    model: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path] | None:
    """Load the pinned model config and bind it to the local model manifest."""

    raw_root = model.get("path")
    if not isinstance(raw_root, str) or not raw_root.strip():
        return None
    root = _host_path(raw_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        return None
    raw_manifest = model.get("manifestPath")
    manifest_path = (
        _host_path(raw_manifest)
        if isinstance(raw_manifest, str) and raw_manifest.strip()
        else root / ".mts-model-manifest.json"
    )
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        return None
    declared_manifest_sha = model.get("manifestFileSha256")
    if (
        isinstance(declared_manifest_sha, str)
        and sha256_file(manifest_path) != declared_manifest_sha
    ):
        return None
    declared_files = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    for relative in ("configuration.json", "config.yaml", "config.yml"):
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
        if not candidate.is_file() or candidate.is_symlink():
            continue
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        declared = declared_files.get(relative)
        if isinstance(declared, Mapping):
            if (
                declared.get("sha256") != sha256_file(candidate)
                or declared.get("size") != candidate.stat().st_size
            ):
                return None
        try:
            if candidate.suffix.casefold() == ".json":
                config = json.loads(candidate.read_text(encoding="utf-8"))
            else:
                import yaml

                config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, ImportError):
            return None
        if isinstance(config, Mapping):
            return candidate, dict(config), manifest_path
    return None


def _preprocessing_evidence(
    *,
    candidate_id: str,
    model: Mapping[str, Any],
    source: Mapping[str, Any],
    benchmark: str,
) -> dict[str, Any]:
    """Extract an explicit input contract without inventing embedding data."""

    config_result = None
    try:
        config_result = _config_file_for_model(model)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        config_result = None
    source_audio = source.get("audio")
    source_audio = source_audio if isinstance(source_audio, Mapping) else {}
    sample_rate = source_audio.get("sampleRateHz")
    channels = source_audio.get("channels")
    profile: dict[str, Any] = {
        "profileId": "evaluation-input-mono-16khz-f32-window-v1",
        "sampleRateHz": sample_rate,
        "channels": channels,
        "sampleDtype": "float32",
        "downmix": "mean-if-multichannel",
        "windowing": "round(startMs*sampleRate/1000):round(endMs*sampleRate/1000)",
        "sourceManifestAudio": {
            key: source_audio.get(key)
            for key in ("codec", "sampleRateHz", "channels", "durationMs")
            if key in source_audio
        },
    }
    if config_result is None:
        return {
            "status": "missing",
            "reason": "pinned model configuration could not be loaded and bound",
            "profile": profile,
        }
    config_path, config, manifest_path = config_result
    configured_sample_rate = (
        _nested_mapping_value(config, ("model", "model_config", "sample_rate"))
        or _nested_mapping_value(config, ("frontend_conf", "fs"))
        or _nested_mapping_value(config, ("dataset_args", "resample_rate"))
    )
    frontend = (
        _nested_mapping_value(config, ("dataset_args", "frontend"))
        or _nested_mapping_value(config, ("frontend",))
    )
    feature_config = (
        _nested_mapping_value(config, ("dataset_args", "tfmel_args"))
        or _nested_mapping_value(config, ("model", "model_config"))
    )
    profile["modelConfig"] = {
        "configuredSampleRateHz": configured_sample_rate,
        "frontend": frontend,
        "featureConfig": feature_config,
    }
    implementation_name = (
        "evaluate_campplus_trials.py"
        if candidate_id == "camplus"
        else "benchmark_redimnet2_verification.py"
        if benchmark == "redimnet2-frozen-speaker-verification"
        else "evaluate_eres2netv2_trials.py"
    )
    implementation_path = PROJECT_ROOT / "tools" / implementation_name
    implementation_sha = (
        sha256_file(implementation_path)
        if implementation_path.is_file()
        else None
    )
    valid = (
        isinstance(configured_sample_rate, (int, float))
        and not isinstance(configured_sample_rate, bool)
        and int(configured_sample_rate) == 16_000
        and sample_rate == 16_000
        and channels == 1
        and implementation_sha is not None
    )
    profile["evidence"] = {
        "modelManifestPath": str(manifest_path),
        "modelManifestFileSha256": sha256_file(manifest_path),
        "modelConfigPath": str(config_path),
        "modelConfigFileSha256": sha256_file(config_path),
        "implementationPath": str(implementation_path),
        "implementationSha256": implementation_sha,
    }
    return {
        "status": "present" if valid else "missing",
        "reason": None
        if valid
        else "model/source audio contract does not agree on mono 16 kHz",
        "profile": profile,
    }


def _metric_value(report: Mapping[str, Any], *names: str) -> float:
    scores = report.get("scores")
    metrics = report.get("metrics")
    containers = [value for value in (scores, metrics) if isinstance(value, Mapping)]
    for container in containers:
        for name in names:
            if name in container:
                return _finite(container[name], f"metrics.{name}")
    raise SpeakerVerificationAcceptanceError(
        f"speaker report is missing metrics {names}"
    )


def _validate_fixed_report(
    *,
    fixed_path: Path,
    development: Mapping[str, Any],
    held_out: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    fixed, resolved = _read_canonical(fixed_path, "fixed-threshold report")
    if (
        fixed.get("schemaVersion") != "1.0.0"
        or fixed.get("evaluation") != "development-frozen-speaker-threshold"
    ):
        raise SpeakerVerificationAcceptanceError(
            "fixed-threshold report type is unsupported"
        )
    if fixed.get("modelIdentitySha256") != development["modelIdentitySha256"]:
        raise SpeakerVerificationAcceptanceError(
            "fixed-threshold model identity differs from development"
        )
    if development["modelIdentitySha256"] != held_out["modelIdentitySha256"]:
        raise SpeakerVerificationAcceptanceError(
            "development and held-out model identities differ"
        )
    policy = _object(fixed.get("thresholdPolicy"), "thresholdPolicy")
    if (
        policy.get("sourceSplit") != "development"
        or policy.get("heldOutThresholdFittingPerformed") is not False
        or policy.get("speakerDisjointSplitsVerified") is not True
        or policy.get("crossRecordingTrialsVerified") is not True
    ):
        raise SpeakerVerificationAcceptanceError(
            "fixed-threshold policy does not preserve held-out isolation"
        )
    threshold = _finite(policy.get("threshold"), "thresholdPolicy.threshold")
    if not math.isclose(
        threshold,
        float(development["threshold"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SpeakerVerificationAcceptanceError(
            "fixed threshold differs from development EER threshold"
        )
    for label, normalized in (
        ("development", development),
        ("heldOut", held_out),
    ):
        evidence = _object(fixed.get(label), label)
        if (
            evidence.get("reportFileSha256") != normalized["fileSha256"]
            or evidence.get("reportCanonicalSha256")
            != normalized["canonicalSha256"]
            or evidence.get("trialManifestCanonicalSha256")
            != normalized["manifest"]["canonicalSha256"]
        ):
            raise SpeakerVerificationAcceptanceError(
                f"fixed-threshold {label} evidence differs from its report"
            )
    return fixed, resolved


def _embedding_freeze_evidence(
    *,
    candidate_id: str,
    reports: tuple[Path, Path] | None,
    development: Mapping[str, Any],
    held_out: Mapping[str, Any],
) -> dict[str, Any] | None:
    if reports is None:
        return None
    output: dict[str, Any] = {
        "hashAlgorithm": "clip-id-sorted-float32-le-v1",
        "modelIdentitySha256": development["modelIdentitySha256"],
        "splits": {},
    }
    for label, expected_split, path, normalized in (
        ("development", "development", reports[0], development),
        ("heldOut", "held-out", reports[1], held_out),
    ):
        document, resolved = _read_canonical(
            path, f"{candidate_id} {label} embedding-freeze report"
        )
        model = _object(document.get("model"), f"{candidate_id}.{label}.model")
        if canonical_json_sha256(model) != development["modelIdentitySha256"]:
            raise SpeakerVerificationAcceptanceError(
                f"{candidate_id} {label} embedding model identity differs"
            )
        trial_manifest = _object(
            document.get("trialManifest"),
            f"{candidate_id}.{label}.trialManifest",
        )
        if trial_manifest.get("canonicalSha256") != normalized["manifest"][
            "canonicalSha256"
        ]:
            raise SpeakerVerificationAcceptanceError(
                f"{candidate_id} {label} embedding trial manifest differs"
            )
        partition = document.get("partition")
        report_split = (
            partition.get("evaluationSplit")
            if isinstance(partition, Mapping)
            else trial_manifest.get("evaluationSplit")
        )
        if report_split != expected_split:
            raise SpeakerVerificationAcceptanceError(
                f"{candidate_id} {label} embedding split differs"
            )
        execution = _object(
            document.get("execution"), f"{candidate_id}.{label}.execution"
        )
        algorithm = execution.get("embeddingSetHashAlgorithm")
        digest = execution.get("embeddingSetSha256")
        if algorithm != output["hashAlgorithm"]:
            raise SpeakerVerificationAcceptanceError(
                f"{candidate_id} {label} embedding hash algorithm differs"
            )
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise SpeakerVerificationAcceptanceError(
                f"{candidate_id} {label} embedding SHA-256 is invalid"
            )
        dimensions = execution.get("embeddingDimensions")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise SpeakerVerificationAcceptanceError(
                f"{candidate_id} {label} embedding dimensions are invalid"
            )
        output["splits"][label] = {
            "evaluationSplit": expected_split,
            "reportPath": str(resolved),
            "reportFileSha256": sha256_file(resolved),
            "reportCanonicalSha256": document["canonicalSha256"],
            "trialManifestCanonicalSha256": trial_manifest["canonicalSha256"],
            "embeddingDimensions": dimensions,
            "embeddingSetSha256": digest,
            "device": execution.get("device"),
        }
    if (
        output["splits"]["development"]["embeddingSetSha256"]
        == output["splits"]["heldOut"]["embeddingSetSha256"]
    ):
        raise SpeakerVerificationAcceptanceError(
            f"{candidate_id} embedding sets unexpectedly match across splits"
        )
    output["canonicalSha256"] = canonical_json_sha256(output)
    return output


def _score_set_sha256(
    scores: Sequence[tuple[tuple[str, str, str, bool], float]],
) -> str:
    return canonical_json_sha256(
        [
            {
                "ordinal": index,
                "sameSpeaker": bool(identity[3]),
                "score": float(score),
            }
            for index, (identity, score) in enumerate(scores)
        ]
    )


def _operating_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    genuine = [float(row["score"]) for row in rows if bool(row["label"])]
    impostor = [float(row["score"]) for row in rows if not bool(row["label"])]
    if not genuine or not impostor:
        raise SpeakerVerificationAcceptanceError(
            "operating metrics require both trial classes"
        )
    auc_numerator = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in genuine
        for negative in impostor
    )
    unique = sorted(set((*genuine, *impostor)))
    thresholds = [unique[0] - 1e-12]
    thresholds.extend(
        (left + right) / 2.0 for left, right in zip(unique, unique[1:])
    )
    thresholds.append(unique[-1] + 1e-12)
    curve: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        far = sum(score >= threshold for score in impostor) / len(impostor)
        frr = sum(score < threshold for score in genuine) / len(genuine)
        curve.append((threshold, far, frr))
    eer_row = min(
        curve,
        key=lambda item: (
            abs(item[1] - item[2]),
            item[1] + item[2],
            item[0],
        ),
    )
    min_dcf_row = min(
        curve,
        key=lambda item: (0.99 * item[1] + 0.01 * item[2], item[0]),
    )
    return {
        "trialCount": len(rows),
        "genuineTrialCount": len(genuine),
        "impostorTrialCount": len(impostor),
        "eer": (eer_row[1] + eer_row[2]) / 2.0,
        "eerThreshold": eer_row[0],
        "farAtEerThreshold": eer_row[1],
        "frrAtEerThreshold": eer_row[2],
        "auc": auc_numerator / (len(genuine) * len(impostor)),
        "minDcfPTarget0.01": 0.99 * min_dcf_row[1] + 0.01 * min_dcf_row[2],
        "minDcfThreshold": min_dcf_row[0],
    }


def _metrics_at_threshold(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, float | int | None]:
    genuine = [float(row["score"]) for row in rows if bool(row["label"])]
    impostor = [float(row["score"]) for row in rows if not bool(row["label"])]
    false_rejects = sum(score < threshold for score in genuine)
    false_accepts = sum(score >= threshold for score in impostor)
    frr = false_rejects / len(genuine) if genuine else None
    far = false_accepts / len(impostor) if impostor else None
    balanced = (
        1.0 - (far + frr) / 2.0
        if far is not None and frr is not None
        else None
    )
    return {
        "threshold": threshold,
        "genuineTrialCount": len(genuine),
        "impostorTrialCount": len(impostor),
        "falseRejectCount": false_rejects,
        "falseAcceptCount": false_accepts,
        "frr": frr,
        "far": far,
        "balancedAccuracy": balanced,
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _platt_objective(
    values: Sequence[float],
    labels: Sequence[float],
    *,
    coefficient: float,
    intercept: float,
    regularization: float,
) -> float:
    epsilon = 1e-15
    loss = 0.5 * regularization * coefficient * coefficient
    for value, label in zip(values, labels):
        probability = min(
            1.0 - epsilon,
            max(epsilon, _sigmoid(coefficient * value + intercept)),
        )
        loss -= label * math.log(probability) + (1.0 - label) * math.log(
            1.0 - probability
        )
    return loss


def _fit_platt_calibrator(
    scores: Sequence[tuple[tuple[str, str, str, bool], float]],
) -> dict[str, Any]:
    if len(scores) < 2:
        raise SpeakerVerificationAcceptanceError(
            "calibrator requires at least two development trials"
        )
    raw = [float(score) for _identity, score in scores]
    labels = [1.0 if identity[3] else 0.0 for identity, _score in scores]
    if not 0.0 < sum(labels) < len(labels):
        raise SpeakerVerificationAcceptanceError(
            "calibrator requires both development trial classes"
        )
    mean = statistics.fmean(raw)
    scale = statistics.pstdev(raw)
    if scale <= 1e-12:
        raise SpeakerVerificationAcceptanceError(
            "development scores have no calibration variance"
        )
    values = [(score - mean) / scale for score in raw]
    prior = sum(labels) / len(labels)
    coefficient = 0.0
    intercept = math.log(prior / (1.0 - prior))
    regularization = 1e-6
    maximum_iterations = 200
    # Double precision plus the fixed L2 term leaves a small stationary
    # gradient floor; 1e-8 is tighter than the reported metric precision while
    # still allowing deterministic convergence across the Windows runtimes.
    tolerance = 1e-8
    converged = False
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        gradient_coefficient = regularization * coefficient
        gradient_intercept = 0.0
        hessian_cc = regularization
        hessian_ci = 0.0
        hessian_ii = 0.0
        for value, label in zip(values, labels):
            probability = _sigmoid(coefficient * value + intercept)
            residual = probability - label
            curvature = probability * (1.0 - probability)
            gradient_coefficient += residual * value
            gradient_intercept += residual
            hessian_cc += curvature * value * value
            hessian_ci += curvature * value
            hessian_ii += curvature
        determinant = hessian_cc * hessian_ii - hessian_ci * hessian_ci
        if determinant <= 1e-18:
            raise SpeakerVerificationAcceptanceError(
                "development calibrator Hessian is singular"
            )
        delta_coefficient = (
            gradient_coefficient * hessian_ii
            - gradient_intercept * hessian_ci
        ) / determinant
        delta_intercept = (
            gradient_intercept * hessian_cc
            - gradient_coefficient * hessian_ci
        ) / determinant
        if max(abs(delta_coefficient), abs(delta_intercept)) <= tolerance:
            converged = True
            break
        objective = _platt_objective(
            values,
            labels,
            coefficient=coefficient,
            intercept=intercept,
            regularization=regularization,
        )
        step = 1.0
        accepted = False
        while step >= 2.0**-30:
            candidate_coefficient = coefficient - step * delta_coefficient
            candidate_intercept = intercept - step * delta_intercept
            candidate_objective = _platt_objective(
                values,
                labels,
                coefficient=candidate_coefficient,
                intercept=candidate_intercept,
                regularization=regularization,
            )
            if candidate_objective <= objective:
                coefficient = candidate_coefficient
                intercept = candidate_intercept
                accepted = True
                break
            step /= 2.0
        if not accepted:
            raise SpeakerVerificationAcceptanceError(
                "development calibrator line search failed"
            )
    if not converged:
        raise SpeakerVerificationAcceptanceError(
            "development calibrator did not converge"
        )
    calibrator: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "kind": "platt-logistic-standardized-score-v1",
        "fitPolicy": {
            "sourceSplit": "development",
            "heldOutFittingPerformed": False,
            "regularizationL2": regularization,
            "maximumIterations": maximum_iterations,
            "tolerance": tolerance,
        },
        "trainingEvidence": {
            "trialCount": len(scores),
            "genuineTrialCount": int(sum(labels)),
            "impostorTrialCount": len(labels) - int(sum(labels)),
            "scoreSetCanonicalSha256": _score_set_sha256(scores),
        },
        "standardization": {"mean": mean, "scale": scale},
        "parameters": {
            "coefficient": coefficient,
            "intercept": intercept,
        },
        "optimization": {"converged": True, "iterations": iterations},
    }
    calibrator["canonicalSha256"] = canonical_json_sha256(calibrator)
    return calibrator


def _probabilities(
    rows: Sequence[Mapping[str, Any]], calibrator: Mapping[str, Any]
) -> list[float]:
    standardization = _object(
        calibrator.get("standardization"), "calibrator.standardization"
    )
    parameters = _object(calibrator.get("parameters"), "calibrator.parameters")
    mean = _finite(standardization.get("mean"), "calibrator.mean")
    scale = _finite(standardization.get("scale"), "calibrator.scale")
    coefficient = _finite(parameters.get("coefficient"), "calibrator.coefficient")
    intercept = _finite(parameters.get("intercept"), "calibrator.intercept")
    return [
        _sigmoid(coefficient * ((float(row["score"]) - mean) / scale) + intercept)
        for row in rows
    ]


def _calibration_metrics(
    rows: Sequence[Mapping[str, Any]], calibrator: Mapping[str, Any]
) -> dict[str, Any]:
    probabilities = _probabilities(rows, calibrator)
    labels = [1.0 if bool(row["label"]) else 0.0 for row in rows]
    if not probabilities:
        raise SpeakerVerificationAcceptanceError("calibration bucket is empty")
    epsilon = 1e-15
    brier = statistics.fmean(
        (probability - label) ** 2
        for probability, label in zip(probabilities, labels)
    )
    log_loss = statistics.fmean(
        -label * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        - (1.0 - label)
        * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for probability, label in zip(probabilities, labels)
    )
    bin_count = 10
    bins: list[dict[str, Any]] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        indexes = [
            row_index
            for row_index, probability in enumerate(probabilities)
            if probability >= lower
            and (probability < upper or index == bin_count - 1)
        ]
        if not indexes:
            bins.append(
                {
                    "index": index,
                    "lowerInclusive": lower,
                    "upperInclusive": upper if index == bin_count - 1 else None,
                    "upperExclusive": None if index == bin_count - 1 else upper,
                    "count": 0,
                    "meanProbability": None,
                    "empiricalPositiveRate": None,
                    "absoluteGap": None,
                }
            )
            continue
        mean_probability = statistics.fmean(probabilities[item] for item in indexes)
        empirical_rate = statistics.fmean(labels[item] for item in indexes)
        gap = abs(mean_probability - empirical_rate)
        weighted_gap += len(indexes) * gap
        maximum_gap = max(maximum_gap, gap)
        bins.append(
            {
                "index": index,
                "lowerInclusive": lower,
                "upperInclusive": upper if index == bin_count - 1 else None,
                "upperExclusive": None if index == bin_count - 1 else upper,
                "count": len(indexes),
                "meanProbability": mean_probability,
                "empiricalPositiveRate": empirical_rate,
                "absoluteGap": gap,
            }
        )
    return {
        "trialCount": len(rows),
        "brierScore": brier,
        "logLoss": log_loss,
        "expectedCalibrationError10": weighted_gap / len(rows),
        "maximumCalibrationError10": maximum_gap,
        "bins": bins,
    }


def _manifest_genres(normalized: Mapping[str, Any]) -> dict[str, str]:
    manifest, _resolved = _read_canonical(
        Path(normalized["manifest"]["path"]), "linked trial manifest"
    )
    genres: dict[str, str] = {}
    raw_clips = manifest.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise SpeakerVerificationAcceptanceError("linked manifest has no clips")
    for index, raw in enumerate(raw_clips):
        clip = _object(raw, f"clips[{index}]")
        clip_id = _text(clip.get("clipId"), f"clips[{index}].clipId")
        genre = _text(clip.get("genre"), f"clips[{index}].genre")
        if clip_id in genres:
            raise SpeakerVerificationAcceptanceError(
                "linked manifest contains duplicate clips"
            )
        genres[clip_id] = genre
    return genres


def _aggregate_rows(normalized: Mapping[str, Any]) -> list[dict[str, Any]]:
    genres = _manifest_genres(normalized)
    rows: list[dict[str, Any]] = []
    for identity, score in normalized["scores"]:
        _trial_id, left_id, right_id, label = identity
        try:
            left_genre = genres[left_id]
            right_genre = genres[right_id]
        except KeyError as exc:
            raise SpeakerVerificationAcceptanceError(
                "trial score references a clip without genre truth"
            ) from exc
        pair = "+".join(sorted((left_genre, right_genre)))
        rows.append(
            {
                "score": float(score),
                "label": bool(label),
                "genreRelation": (
                    "same-genre" if left_genre == right_genre else "cross-genre"
                ),
                "genrePair": pair,
            }
        )
    return rows


def _bucket_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    threshold: float,
    calibrator: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    output: list[dict[str, Any]] = []
    for bucket_id in sorted(grouped):
        bucket = grouped[bucket_id]
        genuine_count = sum(bool(row["label"]) for row in bucket)
        impostor_count = len(bucket) - genuine_count
        output.append(
            {
                "bucket": bucket_id,
                "trialCount": len(bucket),
                "genuineTrialCount": genuine_count,
                "impostorTrialCount": impostor_count,
                "operating": (
                    _operating_metrics(bucket)
                    if genuine_count and impostor_count
                    else None
                ),
                "atDevelopmentFrozenThreshold": _metrics_at_threshold(
                    bucket, threshold
                ),
                "calibration": _calibration_metrics(bucket, calibrator),
            }
        )
    return output


def _assert_aggregate_only(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(str(key) for key in value if str(key) in _IDENTITY_KEYS)
        if forbidden:
            raise SpeakerVerificationAcceptanceError(
                f"aggregate report contains identity fields: {forbidden}"
            )
        for nested in value.values():
            _assert_aggregate_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_aggregate_only(nested)


def _candidate_inventory(
    *,
    candidate_id: str,
    development_path: Path,
    held_out_path: Path,
    fixed_path: Path,
    embedding_reports: tuple[Path, Path] | None = None,
) -> dict[str, Any]:
    try:
        development = _normalized_report(development_path, "development")
        held_out = _normalized_report(held_out_path, "heldOut")
    except FixedSpeakerThresholdError as exc:
        raise SpeakerVerificationAcceptanceError(str(exc)) from exc
    if development["manifest"]["evaluationSplit"] != "development":
        raise SpeakerVerificationAcceptanceError(
            f"{candidate_id} threshold source is not development"
        )
    if held_out["manifest"]["evaluationSplit"] != "held-out":
        raise SpeakerVerificationAcceptanceError(
            f"{candidate_id} target is not held-out"
        )
    if development["manifest"]["speakerIds"] & held_out["manifest"]["speakerIds"]:
        raise SpeakerVerificationAcceptanceError(
            f"{candidate_id} development and held-out speakers overlap"
        )
    fixed, fixed_resolved = _validate_fixed_report(
        fixed_path=fixed_path,
        development=development,
        held_out=held_out,
    )
    development_document, development_resolved = _read_canonical(
        development_path, f"{candidate_id} development report"
    )
    held_out_document, held_out_resolved = _read_canonical(
        held_out_path, f"{candidate_id} held-out report"
    )
    model = _object(development_document.get("model"), f"{candidate_id}.model")
    revision = _model_revision(model)
    development_calibrator = _fit_platt_calibrator(development["scores"])
    preprocessing_evidence = _preprocessing_evidence(
        candidate_id=candidate_id,
        model=model,
        source=development["manifest"]["source"],
        benchmark=str(development_document.get("benchmark")),
    )
    calibrator = _nested_value(
        (development_document, fixed), _CALIBRATOR_KEYS
    )
    preprocessing = _nested_value(
        (development_document, held_out_document), _PREPROCESSING_KEYS
    )
    embedding_sha = _nested_value(
        (development_document, held_out_document), _EMBEDDING_SHA_KEYS
    )
    embedding_freeze = _embedding_freeze_evidence(
        candidate_id=candidate_id,
        reports=embedding_reports,
        development=development,
        held_out=held_out,
    )
    effective_embedding_sha = (
        {
            label: row["embeddingSetSha256"]
            for label, row in embedding_freeze["splits"].items()
        }
        if embedding_freeze is not None
        else embedding_sha
    )
    threshold = float(development["threshold"])
    development_rows = _aggregate_rows(development)
    held_out_rows = _aggregate_rows(held_out)
    development_operating = _operating_metrics(development_rows)
    held_out_operating = _operating_metrics(held_out_rows)
    for label, normalized, document, computed in (
        (
            "development",
            development,
            development_document,
            development_operating,
        ),
        ("heldOut", held_out, held_out_document, held_out_operating),
    ):
        expected_min_dcf = _metric_value(
            document,
            "minimumDetectionCostP01",
            "minDcfPTarget0.01",
        )
        comparisons = (
            ("eer", computed["eer"], normalized["descriptiveEer"]),
            ("auc", computed["auc"], normalized["descriptiveAuc"]),
            ("minDCF", computed["minDcfPTarget0.01"], expected_min_dcf),
        )
        for metric, actual, expected in comparisons:
            if not math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise SpeakerVerificationAcceptanceError(
                    f"{candidate_id} {label} {metric} does not reproduce "
                    "the frozen report"
                )
    held_out_fixed = _metrics_at_threshold(held_out_rows, threshold)
    fixed_evidence = _object(fixed.get("heldOut"), "fixed.heldOut")
    frozen_metrics = _object(
        fixed_evidence.get("metricsAtDevelopmentFrozenThreshold"),
        "fixed held-out metrics",
    )
    for output_key, fixed_key in (
        ("far", "falseAcceptRate"),
        ("frr", "falseRejectRate"),
    ):
        if not math.isclose(
            float(held_out_fixed[output_key]),
            _finite(frozen_metrics.get(fixed_key), f"fixed.{fixed_key}"),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise SpeakerVerificationAcceptanceError(
                f"{candidate_id} fixed-threshold metrics differ from "
                "frozen evidence"
            )
    held_out_final_metrics = {
        "scoreSetCanonicalSha256": _score_set_sha256(held_out["scores"]),
        "operatingDescriptiveOnly": held_out_operating,
        "atDevelopmentFrozenThreshold": held_out_fixed,
        "calibration": _calibration_metrics(
            held_out_rows, development_calibrator
        ),
        "buckets": {
            "genreRelation": _bucket_metrics(
                held_out_rows,
                field="genreRelation",
                threshold=threshold,
                calibrator=development_calibrator,
            ),
            "genrePair": _bucket_metrics(
                held_out_rows,
                field="genrePair",
                threshold=threshold,
                calibrator=development_calibrator,
            ),
        },
    }
    return {
        "candidateId": candidate_id,
        "development": development,
        "heldOut": held_out,
        "fixed": fixed,
        "developmentDocument": development_document,
        "heldOutDocument": held_out_document,
        "public": {
            "candidateId": candidate_id,
            "model": dict(model),
            "evidence": {
                "developmentReport": {
                    "path": str(development_resolved),
                    "fileSha256": sha256_file(development_resolved),
                    "canonicalSha256": development_document["canonicalSha256"],
                    "trialManifestCanonicalSha256": development["manifest"][
                        "canonicalSha256"
                    ],
                },
                "heldOutReport": {
                    "path": str(held_out_resolved),
                    "fileSha256": sha256_file(held_out_resolved),
                    "canonicalSha256": held_out_document["canonicalSha256"],
                    "trialManifestCanonicalSha256": held_out["manifest"][
                        "canonicalSha256"
                    ],
                },
                "fixedThresholdReport": {
                    "path": str(fixed_resolved),
                    "fileSha256": sha256_file(fixed_resolved),
                    "canonicalSha256": fixed["canonicalSha256"],
                },
                "embeddingFreeze": embedding_freeze,
            },
            "developmentSelectionMetrics": {
                "eer": development["descriptiveEer"],
                "auc": development["descriptiveAuc"],
            },
            "freezeAuditBeforeThisRun": {
                "threshold": {
                    "status": "present",
                    "value": development["threshold"],
                    "sourceSplit": "development",
                },
                "calibrator": {
                    "status": "present" if calibrator is not None else "missing",
                },
                "modelRevision": {
                    "status": "present" if revision is not None else "missing",
                    "value": revision,
                },
                "preprocessing": {
                    "status": preprocessing_evidence["status"],
                    "value": (
                        preprocessing
                        if preprocessing is not None
                        else preprocessing_evidence
                    ),
                },
                "embeddingSetSha256": {
                    "status": "present" if embedding_sha is not None else "missing",
                    "value": embedding_sha,
                },
            },
            "developmentOnlyCalibrator": development_calibrator,
            "heldOutFinalMetrics": held_out_final_metrics,
            "freezeAuditAfterThisRun": {
                "threshold": {
                    "status": "present",
                    "value": development["threshold"],
                    "sourceSplit": "development",
                },
                "calibrator": {
                    "status": "generated-development-only",
                    "canonicalSha256": development_calibrator[
                        "canonicalSha256"
                    ],
                },
                "modelRevision": {
                    "status": "present" if revision is not None else "missing",
                    "value": revision,
                },
                "preprocessing": {
                    "status": preprocessing_evidence["status"],
                    "profile": preprocessing_evidence.get("profile"),
                },
                "embeddingSetSha256": {
                    "status": (
                        "present" if effective_embedding_sha is not None else "missing"
                    ),
                    "value": effective_embedding_sha,
                },
            },
        },
        "developmentCalibrator": development_calibrator,
        "preprocessingEvidence": preprocessing_evidence,
        "developmentRows": development_rows,
        "heldOutRows": held_out_rows,
        "developmentOperating": development_operating,
        "heldOutOperating": held_out_operating,
        "heldOutFixed": held_out_fixed,
        "heldOutFinalMetrics": held_out_final_metrics,
        "embeddingFreeze": embedding_freeze,
    }


def build_acceptance_report(
    *,
    candidates: Sequence[tuple[str, Path, Path, Path]],
    embedding_reports: Mapping[str, tuple[Path, Path]] | None = None,
    expected_winner: str | None = None,
) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")
    candidate_ids = [candidate[0] for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    embedding_by_candidate = dict(embedding_reports or {})
    unknown_embedding_candidates = sorted(
        set(embedding_by_candidate) - set(candidate_ids)
    )
    if unknown_embedding_candidates:
        raise ValueError(
            "embedding reports reference unknown candidates: "
            f"{unknown_embedding_candidates}"
        )
    inventory = [
        _candidate_inventory(
            candidate_id=candidate_id,
            development_path=development,
            held_out_path=held_out,
            fixed_path=fixed,
            embedding_reports=embedding_by_candidate.get(candidate_id),
        )
        for candidate_id, development, held_out, fixed in candidates
    ]
    development_manifests = {
        row["development"]["manifest"]["canonicalSha256"] for row in inventory
    }
    held_out_manifests = {
        row["heldOut"]["manifest"]["canonicalSha256"] for row in inventory
    }
    if len(development_manifests) != 1 or len(held_out_manifests) != 1:
        raise SpeakerVerificationAcceptanceError(
            "candidates do not share identical frozen trial manifests"
        )
    embedding_freeze_complete = all(
        row["embeddingFreeze"] is not None for row in inventory
    )
    winner = min(
        inventory,
        key=lambda row: (
            float(row["development"]["descriptiveEer"]),
            -float(row["development"]["descriptiveAuc"]),
            str(row["candidateId"]),
        ),
    )
    if expected_winner is not None and winner["candidateId"] != expected_winner:
        raise SpeakerVerificationAcceptanceError(
            f"development winner is {winner['candidateId']}, not {expected_winner}"
        )
    threshold = float(winner["development"]["threshold"])
    calibrator = winner["developmentCalibrator"]
    development_rows = _aggregate_rows(winner["development"])
    held_out_rows = _aggregate_rows(winner["heldOut"])
    development_operating = _operating_metrics(development_rows)
    held_out_operating = _operating_metrics(held_out_rows)
    for label, normalized, computed in (
        ("development", winner["development"], development_operating),
        ("heldOut", winner["heldOut"], held_out_operating),
    ):
        if not math.isclose(
            float(computed["eer"]),
            float(normalized["descriptiveEer"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(computed["auc"]),
            float(normalized["descriptiveAuc"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise SpeakerVerificationAcceptanceError(
                f"{label} aggregate metrics do not reproduce the frozen report"
            )
    held_out_fixed = _metrics_at_threshold(held_out_rows, threshold)
    fixed_evidence = _object(winner["fixed"].get("heldOut"), "fixed.heldOut")
    frozen_metrics = _object(
        fixed_evidence.get("metricsAtDevelopmentFrozenThreshold"),
        "fixed held-out metrics",
    )
    for output_key, fixed_key in (("far", "falseAcceptRate"), ("frr", "falseRejectRate")):
        if not math.isclose(
            float(held_out_fixed[output_key]),
            _finite(frozen_metrics.get(fixed_key), f"fixed.{fixed_key}"),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise SpeakerVerificationAcceptanceError(
                "recomputed fixed-threshold metrics differ from frozen evidence"
            )
    report: dict[str, Any] = {
        "schemaVersion": "1.0.0",
        "artifactType": "speaker-verification-final-acceptance",
        "selectionPolicy": {
            "winnerSourceSplit": "development",
            "primaryMetric": "lowest-eer",
            "tieBreakers": ["highest-auc", "candidate-id"],
            "heldOutUsedForWinnerSelection": False,
            "selectedCandidateId": winner["candidateId"],
        },
        "heldOutUsagePolicy": {
            "speakerDisjointFromDevelopment": True,
            "thresholdFittedOnHeldOut": False,
            "calibratorFittedOnHeldOut": False,
            "descriptiveEerThresholdUsedForPromotion": False,
            "identityDisclosure": "aggregate-only",
        },
        "trialManifests": {
            "developmentCanonicalSha256": next(iter(development_manifests)),
            "heldOutCanonicalSha256": next(iter(held_out_manifests)),
        },
        "embeddingFreezePolicy": {
            "algorithm": "clip-id-sorted-float32-le-v1",
            "candidateCount": len(inventory),
            "developmentAndHeldOutRequired": True,
            "allCandidatesComplete": embedding_freeze_complete,
        },
        "candidateInventory": [row["public"] for row in inventory],
        "winnerEvidence": {
            "candidateId": winner["candidateId"],
            "model": dict(winner["development"]["model"]),
            "threshold": {
                "value": threshold,
                "sourceSplit": "development",
                "sourceMetric": "equal-error operating point",
                "canonicalEvidenceSha256": winner["fixed"]["canonicalSha256"],
            },
            "calibrator": calibrator,
            "development": {
                "scoreSetCanonicalSha256": _score_set_sha256(
                    winner["development"]["scores"]
                ),
                "operating": development_operating,
                "atDevelopmentFrozenThreshold": _metrics_at_threshold(
                    development_rows, threshold
                ),
                "calibration": _calibration_metrics(
                    development_rows, calibrator
                ),
            },
            "heldOut": {
                "scoreSetCanonicalSha256": _score_set_sha256(
                    winner["heldOut"]["scores"]
                ),
                "operatingDescriptiveOnly": held_out_operating,
                "atDevelopmentFrozenThreshold": held_out_fixed,
                "calibration": _calibration_metrics(held_out_rows, calibrator),
                "buckets": {
                    "genreRelation": _bucket_metrics(
                        held_out_rows,
                        field="genreRelation",
                        threshold=threshold,
                        calibrator=calibrator,
                    ),
                    "genrePair": _bucket_metrics(
                        held_out_rows,
                        field="genrePair",
                        threshold=threshold,
                        calibrator=calibrator,
                    ),
                },
            },
            "remainingFreezeGaps": (
                []
                if embedding_freeze_complete
                else [
                    "clip embedding set SHA-256 is absent for one or more candidates"
                ]
            ),
        },
    }
    _assert_aggregate_only(report)
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
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=4,
        metavar=("ID", "DEVELOPMENT", "HELD_OUT", "FIXED_THRESHOLD"),
        required=True,
        help="Repeat for every candidate in the frozen development comparison.",
    )
    parser.add_argument(
        "--embedding-report",
        action="append",
        nargs=3,
        metavar=("ID", "DEVELOPMENT", "HELD_OUT"),
        help="Bind development/held-out reports containing embedding-set SHA-256.",
    )
    parser.add_argument("--expected-winner")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embedding_reports: dict[str, tuple[Path, Path]] = {}
    for candidate_id, development, held_out in args.embedding_report or []:
        if candidate_id in embedding_reports:
            raise ValueError(f"duplicate embedding report id: {candidate_id}")
        embedding_reports[candidate_id] = (Path(development), Path(held_out))
    report = build_acceptance_report(
        candidates=[
            (candidate_id, Path(development), Path(held_out), Path(fixed))
            for candidate_id, development, held_out, fixed in args.candidate
        ],
        embedding_reports=embedding_reports,
        expected_winner=args.expected_winner,
    )
    _write_report(args.output, report)
    summary = report["winnerEvidence"]["heldOut"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "fileSha256": sha256_file(args.output.resolve()),
                "canonicalSha256": report["canonicalSha256"],
                "selectedCandidateId": report["selectionPolicy"][
                    "selectedCandidateId"
                ],
                "heldOutOperating": summary["operatingDescriptiveOnly"],
                "heldOutAtDevelopmentFrozenThreshold": summary[
                    "atDevelopmentFrozenThreshold"
                ],
                "heldOutCalibration": {
                    key: value
                    for key, value in summary["calibration"].items()
                    if key != "bins"
                },
                "remainingFreezeGaps": report["winnerEvidence"][
                    "remainingFreezeGaps"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
