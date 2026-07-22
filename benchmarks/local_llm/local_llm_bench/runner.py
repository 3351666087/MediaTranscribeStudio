from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import HARNESS_VERSION
from .contracts import load_output_schema, parse_and_validate
from .data import DatasetBundle, Example, make_runtime_input
from .metrics import aggregate_results, failure_modes, recommendation
from .ollama_client import OllamaClient, OllamaResponse
from .text_utils import (
    changed_source_spans,
    cjk_retention,
    normalized_distance,
    protected_tokens_preserved,
    span_overlap_score,
)

UNSAFE_TEXT_ERROR_CODES = {
    "risky_text_modified",
    "unchanged_text_modified",
    "review_or_refused_text_modified",
    "protected_token_changed",
    "output_too_short",
    "output_too_long",
    "edit_ratio_exceeded",
    "cjk_retention_too_low",
    "unglossed_term_correction",
}


def run(
    *,
    root: Path,
    dataset: DatasetBundle,
    model: str,
    ollama_host: str,
    timeout_seconds: float,
    seed: int,
    max_retries: int,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    client = OllamaClient(ollama_host, timeout_seconds)
    output_schema = load_output_schema(root)
    system_prompt = (root / "prompts" / "conservative_zh_cleanup_system.txt").read_text(
        encoding="utf-8"
    )
    model_info = _model_info(client, model)
    started_at = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    records: list[dict[str, Any]] = []

    total = len(dataset.examples)
    for index, example in enumerate(dataset.examples, start=1):
        print(
            f"[{index:03d}/{total:03d}] split={example.split:<7} "
            f"kind={example.kind:<16} chars={len(example.source_text):03d}",
            flush=True,
        )
        record = _run_example(
            client=client,
            model=model,
            system_prompt=system_prompt,
            output_schema=output_schema,
            example=example,
            seed=seed,
            max_retries=max_retries,
        )
        records.append(record)

    wall_ms = (time.perf_counter() - wall_started) * 1000
    ended_at = datetime.now(timezone.utc)
    metrics = aggregate_results(records)
    report = {
        "schemaVersion": "1.1",
        "harnessVersion": HARNESS_VERSION,
        "generatedAt": ended_at.isoformat(),
        "privacy": {
            "externalDataViaEnvironmentOnly": True,
            "sourceTextPersisted": False,
            "targetTextPersisted": False,
            "contextTextPersisted": False,
            "modelResponseTextPersisted": False,
            "speakerNamesPersisted": False,
            "speakerTruthPersisted": False,
            "sampleResultsAreAnonymized": True,
            "sourceLeakCheckPassed": False,
            "normalizedLeakCheckPassed": False,
            "sourceFingerprintsPersisted": True,
            "sourceFingerprintsAreLinkableMetadata": True,
        },
        "environment": _environment_info(),
        "sourceEvidence": {
            "fileSha256": dataset.source_hashes,
            "combinedFingerprintSha256": dataset.combined_fingerprint,
        },
        "dataset": {
            "partitionStrategy": "continuous_time_first_70_percent_dev_last_30_percent_heldout",
            "samplingStrategy": (
                "deterministic_even_temporal_stratified_safety_no_random_seed"
            ),
            "durationMs": dataset.duration_ms,
            "splitBoundaryMs": dataset.split_boundary_ms,
            "counts": dataset.counts,
            "semanticEligibility": (
                "one_to_one_turn_alignment_same_speaker_excluding_manual_splits_and_speaker_overrides"
            ),
            "safetyChallenges": [
                "manual_split_source_turn_requires_review",
                "whole_turn_speaker_override_requires_review",
            ],
        },
        "configuration": {
            "model": model,
            "ollamaHost": _redact_host(ollama_host),
            "thinkingEnabled": False,
            "thinkingOutputAccepted": False,
            "seed": seed,
            "seedScope": "model_generation_only",
            "temperature": 0,
            "topP": 0.1,
            "numCtx": 4096,
            "numPredict": 512,
            "maxRetries": max_retries,
            "glossaryItems": 0,
            "speakerMutationAllowed": False,
            "turnSplitAllowed": False,
            "unglossedTermCorrectionAllowed": False,
        },
        "model": model_info,
        "run": {
            "startedAt": started_at.isoformat(),
            "endedAt": ended_at.isoformat(),
            "wallTimeMs": round(wall_ms, 3),
            "sampleCount": len(records),
            "completedCount": sum(not record.get("runtimeError") for record in records),
            "runtimeFailureCount": sum(bool(record.get("runtimeError")) for record in records),
            "retryCount": sum(bool(record.get("retried")) for record in records),
        },
        "metrics": metrics,
        "failureModes": failure_modes(records),
        "recommendation": recommendation(metrics),
        "sampleResults": records,
    }
    _assert_report_has_no_private_data(
        report,
        dataset.source_texts_for_privacy_check,
        dataset.speaker_truth_for_privacy_check,
    )
    report["privacy"]["sourceLeakCheckPassed"] = True
    report["privacy"]["normalizedLeakCheckPassed"] = True
    _write_json_safe(root, output_json, report)
    _write_text_safe(root, output_markdown, render_markdown(report))
    return report


def _run_example(
    *,
    client: OllamaClient,
    model: str,
    system_prompt: str,
    output_schema: dict[str, Any],
    example: Example,
    seed: int,
    max_retries: int,
) -> dict[str, Any]:
    input_payload = make_runtime_input(example)
    responses: list[OllamaResponse] = []
    validation = None
    runtime_error = None
    first_pass_json_valid = False
    first_pass_contract_valid = False
    raw_hash = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat(
                model=model,
                system_prompt=system_prompt,
                input_payload=input_payload,
                output_schema=output_schema,
                seed=seed,
                retry_note=attempt > 0,
            )
        except RuntimeError as exc:
            runtime_error = str(exc)
            break
        responses.append(response)
        raw_hash = hashlib.sha256(response.content.encode("utf-8")).hexdigest()
        validation = parse_and_validate(
            response.content,
            sample_id=example.sample_id,
            source_text=example.source_text,
            input_risk_flags=example.input_risk_flags,
        )
        if attempt == 0:
            first_pass_json_valid = validation.json_valid
            first_pass_contract_valid = validation.valid
        if validation.valid:
            break

    if validation is None:
        return _failed_record(example, responses, runtime_error or "no_response")

    parsed = validation.parsed or {}
    parsed_output_text = parsed.get("normalizedText")
    has_output_text = isinstance(parsed_output_text, str)
    output_text = parsed_output_text if has_output_text else example.source_text
    decision = parsed.get("decision") if isinstance(parsed.get("decision"), str) else None

    semantic = example.kind == "semantic_cleanup"
    baseline_target_distance = (
        normalized_distance(example.source_text, example.target_text) if semantic else None
    )
    output_target_distance = (
        normalized_distance(output_text, example.target_text)
        if semantic and validation.schema_valid
        else None
    )
    improved = bool(
        semantic
        and output_target_distance is not None
        and baseline_target_distance is not None
        and output_target_distance + 1e-12 < baseline_target_distance
    )
    tied = bool(
        semantic
        and output_target_distance is not None
        and baseline_target_distance is not None
        and abs(output_target_distance - baseline_target_distance) <= 1e-12
    )
    worsened = bool(
        semantic
        and output_target_distance is not None
        and baseline_target_distance is not None
        and output_target_distance > baseline_target_distance + 1e-12
    )
    span_precision = span_recall = span_f1 = None
    if semantic and validation.schema_valid:
        span_precision, span_recall, span_f1 = span_overlap_score(
            changed_source_spans(example.source_text, output_text),
            changed_source_spans(example.source_text, example.target_text),
        )

    edit_ratio = (
        normalized_distance(example.source_text, output_text)
        if has_output_text
        else None
    )
    retention = (
        cjk_retention(example.source_text, output_text)
        if has_output_text
        else None
    )
    tokens_preserved = (
        protected_tokens_preserved(example.source_text, output_text)
        if has_output_text
        else None
    )
    auto_apply_candidate = bool(
        semantic
        and validation.valid
        and decision in {"unchanged", "normalized"}
        and not parsed.get("needsHumanReview")
        and edit_ratio is not None
        and edit_ratio <= 0.18
        and 0.75 <= len(output_text) / max(len(example.source_text), 1) <= 1.15
        and retention is not None
        and retention >= 0.85
        and tokens_preserved
    )
    errors = list(validation.errors)
    unsafe_text_modification = bool(
        has_output_text
        and output_text != example.source_text
        and UNSAFE_TEXT_ERROR_CODES.intersection(errors)
    )

    return {
        "sampleId": example.sample_id,
        "split": example.split,
        "kind": example.kind,
        "inputChars": len(example.source_text),
        "targetChars": len(example.target_text),
        "outputChars": len(output_text) if has_output_text else None,
        "responseSha256": raw_hash,
        "attemptCount": len(responses),
        "retried": len(responses) > 1,
        "firstPassJsonValid": first_pass_json_valid,
        "firstPassContractValid": first_pass_contract_valid,
        "jsonValid": validation.json_valid,
        "schemaValid": validation.schema_valid,
        "safetyValid": validation.safety_valid,
        "contractValid": validation.valid,
        "contractRejected": not validation.valid,
        "safetyViolation": validation.schema_valid and not validation.safety_valid,
        "decision": decision,
        "needsHumanReview": parsed.get("needsHumanReview")
        if type(parsed.get("needsHumanReview")) is bool
        else None,
        "errors": errors,
        "runtimeError": runtime_error,
        "forbiddenCapabilityAttempt": validation.forbidden_capability_attempt,
        "unsafeTextModification": unsafe_text_modification,
        "protectedTokensPreserved": tokens_preserved,
        "cjkRetention": _round_optional(retention),
        "inputOutputEditRatio": _round_optional(edit_ratio),
        "baselineTargetDistance": _round_optional(baseline_target_distance),
        "outputTargetDistance": _round_optional(output_target_distance),
        "exactTargetMatch": bool(semantic and output_text == example.target_text),
        "improvedAgainstTarget": improved,
        "tiedAgainstTarget": tied,
        "worsenedAgainstTarget": worsened,
        "editSpanPrecision": _round_optional(span_precision),
        "editSpanRecall": _round_optional(span_recall),
        "editSpanF1": _round_optional(span_f1),
        "autoApplyCandidate": auto_apply_candidate,
        "latencyMs": round(sum(response.wall_ms for response in responses), 3),
        "promptEvalCount": sum(response.prompt_eval_count for response in responses),
        "evalCount": sum(response.eval_count for response in responses),
        "promptEvalDurationMs": round(
            sum(response.prompt_eval_duration_ns for response in responses) / 1_000_000,
            3,
        ),
        "evalDurationMs": round(
            sum(response.eval_duration_ns for response in responses) / 1_000_000,
            3,
        ),
        "totalDurationMs": round(
            sum(response.total_duration_ns for response in responses) / 1_000_000,
            3,
        ),
        "loadDurationMs": round(
            sum(response.load_duration_ns for response in responses) / 1_000_000,
            3,
        ),
    }


def _failed_record(
    example: Example,
    responses: list[OllamaResponse],
    runtime_error: str,
) -> dict[str, Any]:
    return {
        "sampleId": example.sample_id,
        "split": example.split,
        "kind": example.kind,
        "inputChars": len(example.source_text),
        "targetChars": len(example.target_text),
        "outputChars": None,
        "responseSha256": None,
        "attemptCount": len(responses),
        "retried": len(responses) > 1,
        "firstPassJsonValid": False,
        "firstPassContractValid": False,
        "jsonValid": False,
        "schemaValid": False,
        "safetyValid": False,
        "contractValid": False,
        "contractRejected": True,
        "safetyViolation": False,
        "decision": None,
        "needsHumanReview": None,
        "errors": [],
        "runtimeError": runtime_error,
        "forbiddenCapabilityAttempt": False,
        "unsafeTextModification": False,
        "protectedTokensPreserved": None,
        "cjkRetention": None,
        "inputOutputEditRatio": None,
        "baselineTargetDistance": None,
        "outputTargetDistance": None,
        "exactTargetMatch": False,
        "improvedAgainstTarget": False,
        "tiedAgainstTarget": False,
        "worsenedAgainstTarget": False,
        "editSpanPrecision": None,
        "editSpanRecall": None,
        "editSpanF1": None,
        "autoApplyCandidate": False,
        "latencyMs": round(sum(response.wall_ms for response in responses), 3)
        if responses
        else None,
        "promptEvalCount": sum(response.prompt_eval_count for response in responses),
        "evalCount": sum(response.eval_count for response in responses),
        "promptEvalDurationMs": round(
            sum(response.prompt_eval_duration_ns for response in responses) / 1_000_000,
            3,
        ),
        "evalDurationMs": round(
            sum(response.eval_duration_ns for response in responses) / 1_000_000,
            3,
        ),
        "totalDurationMs": round(
            sum(response.total_duration_ns for response in responses) / 1_000_000,
            3,
        ),
        "loadDurationMs": round(
            sum(response.load_duration_ns for response in responses) / 1_000_000,
            3,
        ),
    }


def _model_info(client: OllamaClient, model: str) -> dict[str, Any]:
    tags = client.list_models()
    matched = None
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            matched = item
            break
    if matched is None:
        raise RuntimeError("ollama_model_not_installed")
    shown = client.show_model(model)
    details = matched.get("details") if isinstance(matched.get("details"), dict) else {}
    capabilities = shown.get("capabilities")
    if not isinstance(capabilities, list):
        capabilities = matched.get("capabilities")
    return {
        "name": model,
        "digestSha256": matched.get("digest"),
        "sizeBytes": matched.get("size"),
        "format": details.get("format"),
        "family": details.get("family"),
        "parameterSize": details.get("parameter_size"),
        "quantizationLevel": details.get("quantization_level"),
        "contextLength": details.get("context_length"),
        "capabilities": capabilities if isinstance(capabilities, list) else [],
        "templateSha256": _hash_optional(shown.get("template")),
        "modelInfoSha256": hashlib.sha256(
            json.dumps(shown.get("model_info", {}), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _environment_info() -> dict[str, Any]:
    return {
        "pythonExecutable": str(Path(sys.executable).resolve()),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "ollamaVersion": _safe_command(["ollama", "--version"]),
        "gpu": _safe_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def _safe_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return " ".join(completed.stdout.strip().split())[:500]


def _assert_report_has_no_private_data(
    report: dict[str, Any],
    private_texts: tuple[str, ...],
    private_speaker_truth: tuple[str, ...],
) -> None:
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    normalized_serialized = _privacy_normalize(serialized)
    for text in private_texts:
        stripped = text.strip()
        normalized = _privacy_normalize(stripped)
        if len(stripped) >= 4 and stripped in serialized:
            raise RuntimeError("privacy_source_text_leak_detected")
        if len(normalized) >= 8 and normalized in normalized_serialized:
            raise RuntimeError("privacy_normalized_source_text_leak_detected")
        if len(normalized) >= 24:
            for start in range(0, len(normalized) - 15, 8):
                if normalized[start : start + 16] in normalized_serialized:
                    raise RuntimeError("privacy_source_text_fragment_leak_detected")
    for speaker_value in private_speaker_truth:
        stripped = speaker_value.strip()
        normalized = _privacy_normalize(stripped)
        if len(normalized) < 2 or normalized.isdecimal():
            continue
        if stripped in serialized or normalized in normalized_serialized:
            raise RuntimeError("privacy_speaker_truth_leak_detected")


def _privacy_normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _write_json_safe(root: Path, path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text_safe(root, path, text)


def _write_text_safe(root: Path, path: Path, text: str) -> None:
    root = root.resolve()
    destination = path.resolve()
    if root != destination and root not in destination.parents:
        raise RuntimeError("output_path_outside_benchmark_root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["metrics"]["overall"]
    dev = report["metrics"]["dev"]
    heldout = report["metrics"]["heldout"]
    overlap = report["metrics"]["overlapEscalation"]
    run_info = report["run"]
    dataset = report["dataset"]["counts"]
    model = report["model"]
    lines = [
        f"# Local LLM benchmark: `{model['name']}`",
        "",
        f"- Generated: `{report['generatedAt']}`",
        f"- Harness: `{report['harnessVersion']}`",
        f"- Model digest: `{model.get('digestSha256')}`",
        f"- Quantization: `{model.get('quantizationLevel')}`",
        f"- Samples: **{run_info['sampleCount']}** "
        f"(dev {dataset['selectedDev']}, held-out {dataset['selectedHeldout']}, "
        f"safety {dataset['selectedSafety']})",
        f"- Wall time: **{run_info['wallTimeMs'] / 1000:.3f} s**",
        f"- Recommendation: **{report['recommendation']['tier']}**",
        "",
        "## Core results",
        "",
        "| Metric | Overall | Dev | Held-out |",
        "|---|---:|---:|---:|",
        _row("Final contract validity", overall, dev, heldout, "finalContractValidityRate"),
        _row("Contract rejection", overall, dev, heldout, "contractRejectionRate"),
        _row("Safety violation", overall, dev, heldout, "safetyViolationRate"),
        _row("Unsafe text modification", overall, dev, heldout, "unsafeTextModificationRate"),
        _row("Protected token preservation", overall, dev, heldout, "protectedTokenPreservationRate"),
        _row(
            "Mean CJK retention (all parsed outputs)",
            overall,
            dev,
            heldout,
            "meanCjkRetentionAllParsedOutputs",
        ),
        _row(
            "Mean CJK retention (contract-valid only)",
            overall,
            dev,
            heldout,
            "meanCjkRetentionContractValidOnly",
        ),
        "",
        f"- Overlap escalation precision / recall / F1: "
        f"`{overlap['precision']:.3f}` / `{overlap['recall']:.3f}` / `{overlap['f1']:.3f}`",
        "- Speaker immutability: `not_applicable_by_locked_contract`; "
        "the model has no writable speaker field, so this is not a speaker-accuracy score.",
        f"- Mean / P50 / P95 latency: `{overall['latencyMs']['mean']}` / "
        f"`{overall['latencyMs']['p50']}` / `{overall['latencyMs']['p95']}` ms",
        f"- Generation speed: `{overall['generationTokensPerSecond']}` tokens/s",
        f"- Auto-apply candidates: `{overall['autoApplyCandidateCount']}`; "
        f"measured regression rate: `{overall['autoApplyRegressionRate']:.3f}`",
        "",
        "## Text-direction diagnostics",
        "",
        "These target-distance metrics include schema-valid outputs only and are not "
        "deployment eligibility metrics.",
        "",
        "| Metric | Overall | Dev | Held-out |",
        "|---|---:|---:|---:|",
        _row("Improved against manual target", overall, dev, heldout, "improvedAgainstTargetRate"),
        _row("Tied against manual target", overall, dev, heldout, "tiedAgainstTargetRate"),
        _row("Worsened against manual target", overall, dev, heldout, "worsenedAgainstTargetRate"),
        _row("Exact target match", overall, dev, heldout, "exactTargetMatchRate"),
        "",
        "## Failure modes",
        "",
        "| Code | Count | Rate |",
        "|---|---:|---:|",
    ]
    if report["failureModes"]:
        for item in report["failureModes"]:
            lines.append(f"| `{item['code']}` | {item['count']} | {item['rate']:.3f} |")
    else:
        lines.append("| none | 0 | 0.000 |")
    lines.extend(
        [
            "",
            "## Safety conclusion",
            "",
            f"- Deployment tier: **{report['recommendation']['tier']}**",
            "- Speaker changes and turn splits were absent from the writable output contract.",
            "- Therefore speaker immutability is not measurable as an accuracy metric here.",
            "- `splitOperationF1` is intentionally not applicable because split generation is forbidden.",
            "- The useful safety metric is overlap escalation to human review.",
            "- The report contains no source, target, context, model output text, or speaker truth.",
            "- Source SHA-256 values are retained for reproducibility and are linkable dataset metadata.",
            "",
            "## Reproduction evidence",
            "",
            f"- Combined source fingerprint: `{report['sourceEvidence']['combinedFingerprintSha256']}`",
            f"- Python: `{report['environment']['pythonVersion']}` at "
            f"`{report['environment']['pythonExecutable']}`",
            f"- Ollama: `{report['environment']['ollamaVersion']}`",
            f"- GPU: `{report['environment']['gpu']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _row(
    label: str,
    overall: dict[str, Any],
    dev: dict[str, Any],
    heldout: dict[str, Any],
    key: str,
) -> str:
    return (
        f"| {label} | {_format_metric(overall.get(key))} | "
        f"{_format_metric(dev.get(key))} | {_format_metric(heldout.get(key))} |"
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _hash_optional(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redact_host(host: str) -> str:
    if host.startswith("http://127.0.0.1") or host.startswith("http://localhost"):
        return "http://localhost:11434"
    return "custom_local_endpoint"
