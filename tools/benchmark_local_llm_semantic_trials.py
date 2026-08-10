"""Run frozen semantic trials through the production Ollama semantic path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.errors import WorkerError  # noqa: E402
from backend.local_llm import (  # noqa: E402
    LocalLLMConfig,
    LocalLLMError,
    OllamaLocalProvider,
    estimate_input_tokens,
    parse_strict_json_object,
)
from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    sha256_file,
)
from backend.semantic_processing import (  # noqa: E402
    SEMANTIC_PROMPT_VERSION,
    SemanticProcessingRunner,
    validate_semantic_suggestions_artifact,
)
from backend.semantic_candidate_lattice import (  # noqa: E402
    build_semantic_candidate_lattice_from_document,
    validate_semantic_candidate_lattice,
)
from backend.semantic_composition import (  # noqa: E402
    SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION,
    SemanticCompositionError,
    SemanticJobArbitrationRunner,
    build_semantic_composition,
    validate_semantic_job_arbitration,
)
from tools.build_local_llm_semantic_trials import (  # noqa: E402
    CASE_CATEGORIES,
    CORPORA,
    DEFAULT_OUTPUT as DEFAULT_MANIFEST,
    SPLITS,
    load_manifest,
    protected_inventory,
)


DEFAULT_OUTPUT = Path(
    "D:/mts-eval/semantic-llm-v1/reports/local-llm-semantic-benchmark.v1.json"
)
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9._:/-]+$")
_MODEL_BLOB = re.compile(r"(?im)^FROM .*sha256-([0-9a-f]{64})\s*$")
SEMANTIC_PATHS = (
    "production-mandatory",
    "legacy-suggestions",
)
_DEFAULT_BATCH_SIZE = {
    "production-mandatory": 32,
    "legacy-suggestions": 3,
}
_DEPLOYMENT_BATCH_SIZE = 8
_EVALUATION_STAGES = ("screening", "promotion")
_OLLAMA_RESPONSE_LIMIT = 16 * 1024 * 1024
_SOURCE_FILES = {
    "benchmarkRunner": PROJECT_ROOT
    / "tools"
    / "benchmark_local_llm_semantic_trials.py",
    "trialManifestBuilder": PROJECT_ROOT
    / "tools"
    / "build_local_llm_semantic_trials.py",
    "localLlmProvider": PROJECT_ROOT / "backend" / "local_llm.py",
    "candidateLattice": PROJECT_ROOT
    / "backend"
    / "semantic_candidate_lattice.py",
    "mandatoryArbitration": PROJECT_ROOT
    / "backend"
    / "semantic_composition.py",
    "legacySuggestions": PROJECT_ROOT
    / "backend"
    / "semantic_processing.py",
}
_PROVIDER_COUNTERS = (
    "completedCalls",
    "totalDurationNanoseconds",
    "loadDurationNanoseconds",
    "promptEvalTokens",
    "promptEvalDurationNanoseconds",
    "outputTokens",
    "outputEvalDurationNanoseconds",
)
_BOOLEAN_RESULTS = (
    "executionSucceeded",
    "schemaContractPassed",
    "exactTextSelected",
    "lexicalPreservationPassed",
    "speakerAccuracyPassed",
    "speakerBoundaryControlPassed",
    "safeModificationPassed",
)


class SemanticBenchmarkError(ValueError):
    """Raised when semantic benchmark inputs or outputs are inconsistent."""


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Any,
        file_pointer: Any,
        status_code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        del request, file_pointer, status_code, message, headers, new_url
        return None


class _InstrumentedProvider:
    """Count attempted generations while preserving the provider contract."""

    def __init__(self, provider: OllamaLocalProvider) -> None:
        self.provider = provider
        self.attempted_calls = 0
        self.call_evidence: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    @property
    def generation_metrics(self) -> dict[str, int]:
        return self.provider.generation_metrics

    def generate_json(self, **kwargs: Any) -> Mapping[str, Any]:
        self.attempted_calls += 1
        system_prompt = str(kwargs.get("system_prompt") or "")
        user_prompt = str(kwargs.get("user_prompt") or "")
        response_schema = kwargs.get("response_schema")
        schema_sha = canonical_json_sha256(response_schema)
        evidence = {
            "callIndex": self.attempted_calls,
            "systemPromptSha256": _sha256_text(system_prompt),
            "userPromptSha256": _sha256_text(user_prompt),
            "responseSchemaCanonicalSha256": schema_sha,
            "estimatedInputTokens": estimate_input_tokens(
                system_prompt,
                user_prompt,
                json.dumps(
                    response_schema,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
            "requestEnvelopeCanonicalSha256": canonical_json_sha256(
                {
                    "model": kwargs.get("model"),
                    "temperature": kwargs.get("temperature"),
                    "systemPromptSha256": _sha256_text(system_prompt),
                    "userPromptSha256": _sha256_text(user_prompt),
                    "responseSchemaCanonicalSha256": schema_sha,
                }
            ),
            "outcome": "pending",
            "responseCanonicalSha256": None,
            "errorType": None,
        }
        self.call_evidence.append(evidence)
        try:
            response = self.provider.generate_json(**kwargs)
        except Exception as exc:
            evidence["outcome"] = "raised"
            evidence["errorType"] = type(exc).__name__
            raise
        evidence["outcome"] = "returned"
        evidence["responseCanonicalSha256"] = canonical_json_sha256(response)
        return response

    def release_resources(self) -> None:
        self.provider.release_resources()


class _StaticPlanningProvider:
    """Return current choices while recording the exact provider input size."""

    provider_id = "static-semantic-planning"
    provider_version = "1"
    network_policy = "loopback-only"

    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def generate_json(self, **kwargs: Any) -> Mapping[str, Any]:
        response_schema = kwargs.get("response_schema")
        schema_text = json.dumps(
            response_schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        estimate = estimate_input_tokens(
            str(kwargs["system_prompt"]),
            str(kwargs["user_prompt"]),
            schema_text,
        )
        prompt = parse_strict_json_object(str(kwargs["user_prompt"]))
        target_count = prompt.get("targetGroupCount")
        if (
            isinstance(target_count, bool)
            or not isinstance(target_count, int)
            or target_count < 1
        ):
            raise SemanticBenchmarkError(
                "mandatory planning prompt has no target group count"
            )
        self.calls.append(
            {
                "estimatedInputTokens": estimate,
                "targetGroupCount": target_count,
            }
        )
        return {"choiceIndexes": [0] * target_count}

    def release_resources(self) -> None:
        return None


def _sha256_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ollama_request(
    endpoint: str,
    path: str,
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    url = endpoint.rstrip("/") + path
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(
            dict(payload),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    opener = urllib.request.build_opener(_RejectRedirectHandler())
    try:
        with opener.open(request, timeout=30.0) as response:
            body = response.read(_OLLAMA_RESPONSE_LIMIT + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise SemanticBenchmarkError(
            f"Ollama evidence request failed for {path}"
        ) from exc
    if len(body) > _OLLAMA_RESPONSE_LIMIT:
        raise SemanticBenchmarkError(
            f"Ollama evidence response exceeded the limit for {path}"
        )
    try:
        return parse_strict_json_object(body.decode("utf-8", errors="strict"))
    except (UnicodeError, LocalLLMError) as exc:
        raise SemanticBenchmarkError(
            f"Ollama evidence response is invalid for {path}"
        ) from exc


def _normalized_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if not normalized.startswith("sha256:"):
        normalized = f"sha256:{normalized}"
    if re.fullmatch(r"sha256:[0-9a-f]{64}", normalized) is None:
        return None
    return normalized


def _model_runtime_provenance(config: LocalLLMConfig) -> dict[str, Any]:
    version = _ollama_request(config.endpoint, "/api/version").get("version")
    if not isinstance(version, str) or not version.strip():
        raise SemanticBenchmarkError("Ollama server version is unavailable")
    inventory = _ollama_request(config.endpoint, "/api/tags")
    raw_models = inventory.get("models")
    if not isinstance(raw_models, list):
        raise SemanticBenchmarkError("Ollama model inventory is invalid")
    matches = [
        item
        for item in raw_models
        if isinstance(item, Mapping)
        and config.model in (item.get("name"), item.get("model"))
    ]
    if len(matches) != 1:
        raise SemanticBenchmarkError(
            "configured Ollama model does not have one exact inventory entry"
        )
    model_entry = matches[0]
    actual_digest = _normalized_digest(model_entry.get("digest"))
    if actual_digest is None or actual_digest != config.expected_model_digest:
        raise SemanticBenchmarkError(
            "configured Ollama model manifest digest does not match"
        )
    shown = _ollama_request(
        config.endpoint,
        "/api/show",
        payload={"model": config.model, "verbose": False},
    )
    modelfile = shown.get("modelfile")
    blob_match = _MODEL_BLOB.search(modelfile) if isinstance(modelfile, str) else None
    capabilities = shown.get("capabilities")
    details = shown.get("details")
    return {
        "serverVersion": version.strip(),
        "model": config.model,
        "expectedManifestDigest": config.expected_model_digest,
        "actualManifestDigest": actual_digest,
        "manifestDigestVerified": True,
        "manifestSizeBytes": model_entry.get("size"),
        "modifiedAt": model_entry.get("modified_at"),
        "details": dict(details) if isinstance(details, Mapping) else {},
        "capabilities": (
            list(capabilities)
            if isinstance(capabilities, list)
            and all(isinstance(item, str) for item in capabilities)
            else []
        ),
        "modelBlobSha256": blob_match.group(1) if blob_match else None,
        "licenseSha256": _sha256_text(shown.get("license")),
        "modelfileSha256": _sha256_text(modelfile),
        "templateSha256": _sha256_text(shown.get("template")),
        "parametersSha256": _sha256_text(shown.get("parameters")),
    }


def _host_resource_state() -> dict[str, Any]:
    try:
        import psutil  # type: ignore[import-not-found]

        memory = psutil.virtual_memory()
        ollama_rss = 0
        ollama_process_count = 0
        observed_process_names: set[str] = set()
        for process in psutil.process_iter(["name", "memory_info"]):
            try:
                name = str(process.info.get("name") or "").casefold()
                if name not in {
                    "ollama",
                    "ollama.exe",
                    "llama-server",
                    "llama-server.exe",
                }:
                    continue
                memory_info = process.info.get("memory_info")
                ollama_rss += int(getattr(memory_info, "rss", 0))
                ollama_process_count += 1
                observed_process_names.add(name)
            except (psutil.Error, OSError, ValueError):
                continue
        return {
            "available": True,
            "totalPhysicalMemoryBytes": int(memory.total),
            "availablePhysicalMemoryBytes": int(memory.available),
            "usedPhysicalMemoryBytes": int(memory.used),
            "ollamaProcessCount": ollama_process_count,
            "ollamaAggregateRssBytes": ollama_rss,
            "observedProcessNames": sorted(observed_process_names),
            "processScope": "all-local-ollama-and-llama-server-processes",
        }
    except (ImportError, OSError, ValueError):
        return {"available": False, "failureCode": "HOST_MEMORY_UNAVAILABLE"}


def _resource_snapshot(
    endpoint: str,
    model: str,
    *,
    phase: str,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "phase": phase,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "host": _host_resource_state(),
    }
    try:
        value = _ollama_request(endpoint, "/api/ps")
        raw_models = value.get("models")
        if not isinstance(raw_models, list):
            raise SemanticBenchmarkError("Ollama residency response is invalid")
        selected = [
            item
            for item in raw_models
            if isinstance(item, Mapping)
            and model in (item.get("name"), item.get("model"))
        ]
        snapshot.update(
            {
                "ollamaApiAvailable": True,
                "selectedModelLoaded": bool(selected),
                "selectedModel": [
                    {
                        "name": item.get("name"),
                        "digest": _normalized_digest(item.get("digest")),
                        "sizeBytes": item.get("size"),
                        "sizeVramBytes": item.get("size_vram"),
                    }
                    for item in selected
                ],
            }
        )
    except SemanticBenchmarkError:
        snapshot.update(
            {
                "ollamaApiAvailable": False,
                "selectedModelLoaded": None,
                "selectedModel": [],
                "failureCode": "OLLAMA_RESIDENCY_UNAVAILABLE",
            }
        )
    return snapshot


def _wait_for_release_snapshot(endpoint: str, model: str) -> dict[str, Any]:
    snapshot = _resource_snapshot(
        endpoint,
        model,
        phase="after-release",
    )
    for _ in range(20):
        if (
            snapshot.get("ollamaApiAvailable") is True
            and snapshot.get("selectedModelLoaded") is False
        ):
            return snapshot
        time.sleep(0.25)
        snapshot = _resource_snapshot(
            endpoint,
            model,
            phase="after-release",
        )
    return snapshot


def _resource_summary(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    size_values: list[int] = []
    vram_values: list[int] = []
    rss_values: list[int] = []
    available_memory_values: list[int] = []
    for snapshot in snapshots:
        selected = snapshot.get("selectedModel")
        if isinstance(selected, list):
            for item in selected:
                if not isinstance(item, Mapping):
                    continue
                for field, target in (
                    ("sizeBytes", size_values),
                    ("sizeVramBytes", vram_values),
                ):
                    value = item.get(field)
                    if not isinstance(value, bool) and isinstance(value, int):
                        target.append(value)
        host = snapshot.get("host")
        if isinstance(host, Mapping):
            rss = host.get("ollamaAggregateRssBytes")
            available = host.get("availablePhysicalMemoryBytes")
            if not isinstance(rss, bool) and isinstance(rss, int):
                rss_values.append(rss)
            if not isinstance(available, bool) and isinstance(available, int):
                available_memory_values.append(available)
    after_release = snapshots[-1] if snapshots else {}
    return {
        "measurement": "ollama-api-ps-stage-snapshots-v1",
        "snapshotCount": len(snapshots),
        "peakSelectedModelBytes": max(size_values, default=None),
        "peakSelectedModelVramBytes": max(vram_values, default=None),
        "peakOllamaAggregateRssBytes": max(rss_values, default=None),
        "minimumAvailablePhysicalMemoryBytes": min(
            available_memory_values,
            default=None,
        ),
        "modelAbsentAfterRelease": bool(
            after_release.get("ollamaApiAvailable") is True
            and after_release.get("selectedModelLoaded") is False
        ),
        "snapshots": [dict(item) for item in snapshots],
    }


def _source_provenance(semantic_path: str) -> dict[str, Any]:
    prompt_version = (
        SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION
        if semantic_path == "production-mandatory"
        else SEMANTIC_PROMPT_VERSION
    )
    return {
        "semanticPath": semantic_path,
        "promptVersion": prompt_version,
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "sourceFileSha256": {
            key: sha256_file(path.resolve(strict=True))
            for key, path in sorted(_SOURCE_FILES.items())
        },
    }


def _provider_metrics(provider: Any) -> dict[str, int]:
    raw = provider.generation_metrics
    metrics: dict[str, int] = {}
    for key in _PROVIDER_COUNTERS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SemanticBenchmarkError(
                f"provider metric {key} is invalid"
            )
        metrics[key] = value
    return metrics


def _provider_call_evidence_complete(provider: _InstrumentedProvider) -> bool:
    if len(provider.call_evidence) != provider.attempted_calls:
        return False
    digest_fields = (
        "systemPromptSha256",
        "userPromptSha256",
        "responseSchemaCanonicalSha256",
        "requestEnvelopeCanonicalSha256",
    )
    for item in provider.call_evidence:
        if any(_normalized_digest(item.get(field)) is None for field in digest_fields):
            return False
        outcome = item.get("outcome")
        response_digest = item.get("responseCanonicalSha256")
        if outcome == "returned":
            if _normalized_digest(response_digest) is None:
                return False
        elif outcome == "raised":
            if response_digest is not None or not isinstance(
                item.get("errorType"),
                str,
            ):
                return False
        else:
            return False
    return True


def _counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    delta = {key: int(after[key]) - int(before[key]) for key in _PROVIDER_COUNTERS}
    if any(value < 0 for value in delta.values()):
        raise SemanticBenchmarkError("provider counters moved backwards")
    return delta


def _target_segment(case: Mapping[str, Any]) -> Mapping[str, Any]:
    document = case.get("document")
    segments = document.get("segments") if isinstance(document, Mapping) else None
    target_id = case.get("targetSegmentId")
    if not isinstance(segments, list):
        raise SemanticBenchmarkError("semantic case document has no segments")
    matches = [
        item
        for item in segments
        if isinstance(item, Mapping) and item.get("id") == target_id
    ]
    if len(matches) != 1:
        raise SemanticBenchmarkError("semantic case target is ambiguous")
    return matches[0]


def _failure_codes(artifact: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    for field in ("failures", "rejections"):
        rows = artifact.get(field)
        if not isinstance(rows, list):
            codes.append(f"INVALID_{field.upper()}_CONTRACT")
            continue
        for row in rows:
            code = row.get("code") if isinstance(row, Mapping) else None
            codes.append(
                str(code) if isinstance(code, str) and code else "UNKNOWN_FAILURE"
            )
    return sorted(codes)


def _clean_contract(artifact: Mapping[str, Any]) -> bool:
    metrics = artifact.get("metrics")
    return bool(
        artifact.get("status") == "completed"
        and isinstance(metrics, Mapping)
        and metrics.get("failureCount") == 0
        and metrics.get("rejectionCount") == 0
        and metrics.get("unresolvedSegmentCount") == 0
        and metrics.get("autoAppliedCount") == 0
        and artifact.get("failures") == []
        and artifact.get("rejections") == []
    )


def _failed_result(
    case: Mapping[str, Any],
    *,
    failure_code: str,
    semantic_path: str = "legacy-suggestions",
    execution_succeeded: bool = False,
    decision_sha256: str | None = None,
    selection_outcome: str = "execution-or-contract-invalid",
) -> dict[str, Any]:
    return {
        "caseId": case["caseId"],
        "caseSha256": case["caseSha256"],
        "category": case["category"],
        "split": case["split"],
        "corpus": case["corpus"],
        "semanticPath": semantic_path,
        **{key: False for key in _BOOLEAN_RESULTS},
        "executionSucceeded": execution_succeeded,
        "targetSuggestionPresent": False,
        "nonTargetSuggestionsAbsent": False,
        "targetDecisionPresent": False,
        "nonTargetSelectionsPreserved": False,
        "unexpectedCandidateSelectionsAbsent": False,
        "timelineSelectionMatched": False,
        "authorizedTextModification": False,
        "authorizedSpeakerModification": False,
        "expectedTextChangeMatched": False,
        "expectedSpeakerChangeMatched": False,
        "failureCodes": [failure_code],
        "providerCalls": 0,
        "decisionSha256": decision_sha256,
        "selectionOutcome": selection_outcome,
    }


def _legacy_decision_sha256(artifact: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "status": artifact.get("status"),
            "suggestions": artifact.get("suggestions"),
            "failures": artifact.get("failures"),
            "rejections": artifact.get("rejections"),
        }
    )


def score_case(
    case: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    document = case.get("document")
    if not isinstance(document, Mapping):
        raise SemanticBenchmarkError("semantic case document is invalid")
    target = _target_segment(case)
    target_id = str(case["targetSegmentId"])
    expected = case.get("expected")
    if not isinstance(expected, Mapping):
        raise SemanticBenchmarkError("semantic case expectation is invalid")
    try:
        validated = validate_semantic_suggestions_artifact(
            artifact,
            expected_job_id=str(document["jobId"]),
            expected_transcript_sha256=canonical_json_sha256(document),
        )
        artifact_valid = True
    except (WorkerError, ValueError) as exc:
        code = exc.code if isinstance(exc, WorkerError) else type(exc).__name__
        return _failed_result(case, failure_code=str(code))

    raw_suggestions = validated.get("suggestions")
    suggestions = raw_suggestions if isinstance(raw_suggestions, list) else []
    target_suggestions = [
        item
        for item in suggestions
        if isinstance(item, Mapping) and item.get("segmentId") == target_id
    ]
    non_target = [
        item
        for item in suggestions
        if not isinstance(item, Mapping) or item.get("segmentId") != target_id
    ]
    unique_target = len(target_suggestions) <= 1
    suggestion = target_suggestions[0] if len(target_suggestions) == 1 else None
    current_text = str(target["normalizedText"])
    current_speaker = str(target["speakerId"])
    selected_text = current_text
    selected_speaker = current_speaker
    changes: list[str] = []
    if isinstance(suggestion, Mapping):
        raw_changes = suggestion.get("changes")
        changes = list(raw_changes) if isinstance(raw_changes, list) else []
        proposal = suggestion.get("proposal")
        if isinstance(proposal, Mapping):
            if "text" in changes and isinstance(
                proposal.get("normalizedText"), str
            ):
                selected_text = str(proposal["normalizedText"])
            if "speaker" in changes and isinstance(
                proposal.get("targetSpeakerId"), str
            ):
                selected_speaker = str(proposal["targetSpeakerId"])

    asr = target["evidence"]["asr"]
    eligible_texts = {
        str(item["text"])
        for item in asr["nBest"]
        if item.get("lexicalRepairEligible") is True
    }
    allowed_speakers = {
        str(item["speakerId"])
        for item in target["speakerScores"]
        if isinstance(item, Mapping)
    }
    text_changed = selected_text != current_text
    speaker_changed = selected_speaker != current_speaker
    authorized_text = not text_changed or selected_text in eligible_texts
    authorized_speaker = (
        not speaker_changed or selected_speaker in allowed_speakers
    )
    contract_passed = artifact_valid and _clean_contract(validated)
    exact_text = contract_passed and selected_text == expected["normalizedText"]
    lexical_preserved = bool(
        contract_passed
        and protected_inventory(selected_text) == expected["protectedInventory"]
    )
    speaker_accurate = bool(
        contract_passed and selected_speaker == expected["speakerId"]
    )
    expected_text_change = bool(expected["textChangeRequired"])
    expected_speaker_change = bool(expected["speakerChangeRequired"])
    text_change_matched = contract_passed and text_changed == expected_text_change
    speaker_change_matched = (
        contract_passed and speaker_changed == expected_speaker_change
    )
    boundary_control = bool(
        contract_passed
        and (
            not bool(expected["boundaryMustRemain"])
            or (
                selected_speaker == current_speaker
                and selected_speaker == expected["speakerId"]
            )
        )
    )
    safe_modification = bool(
        contract_passed
        and unique_target
        and not non_target
        and authorized_text
        and authorized_speaker
        and exact_text
        and lexical_preserved
        and speaker_accurate
        and text_change_matched
        and speaker_change_matched
        and boundary_control
    )
    if contract_passed and safe_modification:
        selection_outcome = "expected-state-selected"
    elif contract_passed and not text_changed and not speaker_changed:
        selection_outcome = "current-state-selected"
    elif contract_passed:
        selection_outcome = "reachable-wrong-selection"
    else:
        selection_outcome = "execution-or-contract-invalid"
    metrics = validated.get("metrics")
    provider_calls = (
        int(metrics.get("providerCalls", 0))
        if isinstance(metrics, Mapping)
        else 0
    )
    return {
        "caseId": case["caseId"],
        "caseSha256": case["caseSha256"],
        "category": case["category"],
        "split": case["split"],
        "corpus": case["corpus"],
        "semanticPath": "legacy-suggestions",
        "executionSucceeded": artifact_valid,
        "schemaContractPassed": contract_passed,
        "exactTextSelected": exact_text,
        "lexicalPreservationPassed": lexical_preserved,
        "speakerAccuracyPassed": speaker_accurate,
        "speakerBoundaryControlPassed": boundary_control,
        "safeModificationPassed": safe_modification,
        "targetSuggestionPresent": suggestion is not None,
        "nonTargetSuggestionsAbsent": not non_target,
        "targetDecisionPresent": suggestion is not None,
        "nonTargetSelectionsPreserved": not non_target,
        "unexpectedCandidateSelectionsAbsent": not non_target,
        "timelineSelectionMatched": True,
        "authorizedTextModification": authorized_text,
        "authorizedSpeakerModification": authorized_speaker,
        "expectedTextChangeMatched": text_change_matched,
        "expectedSpeakerChangeMatched": speaker_change_matched,
        "failureCodes": _failure_codes(validated),
        "providerCalls": provider_calls,
        "decisionSha256": _legacy_decision_sha256(validated),
        "selectionOutcome": selection_outcome,
    }


def score_mandatory_case(
    case: Mapping[str, Any],
    lattice: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    document = case.get("document")
    expected = case.get("expected")
    if not isinstance(document, Mapping) or not isinstance(expected, Mapping):
        raise SemanticBenchmarkError("semantic case contract is invalid")
    validated_lattice = validate_semantic_candidate_lattice(
        lattice,
        expected_transcript_sha256=canonical_json_sha256(document),
    )
    try:
        validated = validate_semantic_job_arbitration(
            artifact,
            expected_job_id=str(document["jobId"]),
            expected_lattice=validated_lattice,
        )
    except (SemanticCompositionError, WorkerError, ValueError) as exc:
        code = exc.code if isinstance(exc, WorkerError) else type(exc).__name__
        return _failed_result(
            case,
            failure_code=str(code),
            semantic_path="production-mandatory",
        )

    decision_sha = str(validated["decisionSha256"])
    requests = validated.get("candidateGenerationRequests")
    if validated.get("status") != "ready-to-compose" or requests != []:
        return _failed_result(
            case,
            failure_code="MANDATORY_CANDIDATE_GENERATION_REQUIRED",
            semantic_path="production-mandatory",
            execution_succeeded=True,
            decision_sha256=decision_sha,
            selection_outcome="candidate-generation-required",
        )
    try:
        composition = build_semantic_composition(
            document,
            validated_lattice,
            validated,
        )
    except (SemanticCompositionError, WorkerError, ValueError):
        return _failed_result(
            case,
            failure_code="MANDATORY_COMPOSITION_INVALID",
            semantic_path="production-mandatory",
            execution_succeeded=True,
            decision_sha256=decision_sha,
            selection_outcome="composition-invalid",
        )

    target = _target_segment(case)
    target_id = str(case["targetSegmentId"])
    raw_final_segments = composition.get("segments")
    final_segments = (
        raw_final_segments if isinstance(raw_final_segments, list) else []
    )
    final_by_id = {
        str(item.get("id")): item
        for item in final_segments
        if isinstance(item, Mapping)
    }
    final_target = final_by_id.get(target_id)
    if not isinstance(final_target, Mapping):
        return _failed_result(
            case,
            failure_code="MANDATORY_TARGET_MISSING",
            semantic_path="production-mandatory",
            execution_succeeded=True,
            decision_sha256=decision_sha,
            selection_outcome="composition-contract-invalid",
        )

    current_text = str(target["normalizedText"])
    current_speaker = str(target["speakerId"])
    selected_text = str(final_target.get("finalText") or "")
    selected_speaker = str(final_target.get("speakerId") or "")
    text_changed = selected_text != current_text
    speaker_changed = selected_speaker != current_speaker
    asr = target["evidence"]["asr"]
    eligible_texts = {
        str(item["text"])
        for item in asr["nBest"]
        if item.get("lexicalRepairEligible") is True
    }
    allowed_speakers = {
        str(item["speakerId"])
        for item in target["speakerScores"]
        if isinstance(item, Mapping)
    }
    authorized_text = not text_changed or selected_text in eligible_texts
    authorized_speaker = (
        not speaker_changed or selected_speaker in allowed_speakers
    )
    non_target_preserved = len(final_by_id) == len(document["segments"])
    for segment in document["segments"]:
        segment_id = str(segment["id"])
        if segment_id == target_id:
            continue
        final_segment = final_by_id.get(segment_id)
        non_target_preserved = bool(
            non_target_preserved
            and isinstance(final_segment, Mapping)
            and str(final_segment.get("finalText"))
            == str(segment["normalizedText"])
            and str(final_segment.get("speakerId"))
            == str(segment["speakerId"])
            and str(final_segment.get("language"))
            == str(segment.get("language") or document.get("language") or "und")
        )
    groups = {
        str(group["groupId"]): {
            **dict(group),
            "domain": str(domain["domain"]),
        }
        for domain in validated_lattice["domains"]
        for group in domain["groups"]
    }
    selected_by_group = {
        str(item["groupId"]): str(item["selectedCandidateId"])
        for item in validated["selections"]
    }
    permitted_changed_groups: set[str] = set()
    for group_id, group in groups.items():
        if (
            bool(expected["textChangeRequired"])
            and group["domain"] == "asr-text"
            and group["scopeId"] == f"segment:{target_id}"
        ):
            permitted_changed_groups.add(group_id)
        if bool(expected["speakerChangeRequired"]) and (
            (
                group["domain"] == "speaker-assignment"
                and group["scopeId"] == f"segment:{target_id}"
            )
            or group["domain"] == "speaker-cardinality-timeline"
        ):
            permitted_changed_groups.add(group_id)
    unexpected_selections_absent = all(
        group_id in permitted_changed_groups
        or selected_by_group.get(group_id) == group["currentCandidateId"]
        for group_id, group in groups.items()
    )
    timeline_groups = [
        (group_id, group)
        for group_id, group in groups.items()
        if group["domain"] == "speaker-cardinality-timeline"
    ]
    timeline_selection_matched = bool(
        len(timeline_groups) == 1
        and (
            selected_by_group.get(timeline_groups[0][0])
            != timeline_groups[0][1]["currentCandidateId"]
        )
        == bool(expected["speakerChangeRequired"])
    )
    all_current_selections = bool(
        len(selected_by_group) == len(groups)
        and all(
            selected_by_group.get(group_id) == group["currentCandidateId"]
            for group_id, group in groups.items()
        )
    )
    selected_target_domains = {
        str(item.get("domain"))
        for item in validated["selections"]
        if item.get("scopeId") == f"segment:{target_id}"
    }
    target_decision_present = {
        "speaker-assignment",
        "language-span",
        "asr-text",
    }.issubset(selected_target_domains)
    contract_passed = bool(
        composition.get("status") == "composition-complete"
        and target_decision_present
        and non_target_preserved
    )
    exact_text = bool(
        contract_passed and selected_text == expected["normalizedText"]
    )
    lexical_preserved = bool(
        contract_passed
        and protected_inventory(selected_text) == expected["protectedInventory"]
    )
    speaker_accurate = bool(
        contract_passed and selected_speaker == expected["speakerId"]
    )
    expected_text_change = bool(expected["textChangeRequired"])
    expected_speaker_change = bool(expected["speakerChangeRequired"])
    text_change_matched = bool(
        contract_passed and text_changed == expected_text_change
    )
    speaker_change_matched = bool(
        contract_passed and speaker_changed == expected_speaker_change
    )
    boundary_control = bool(
        contract_passed
        and (
            not bool(expected["boundaryMustRemain"])
            or (
                selected_speaker == current_speaker
                and selected_speaker == expected["speakerId"]
            )
        )
    )
    safe_modification = bool(
        contract_passed
        and authorized_text
        and authorized_speaker
        and unexpected_selections_absent
        and timeline_selection_matched
        and exact_text
        and lexical_preserved
        and speaker_accurate
        and text_change_matched
        and speaker_change_matched
        and boundary_control
    )
    expected_state_selected = bool(
        contract_passed
        and unexpected_selections_absent
        and timeline_selection_matched
        and exact_text
        and speaker_accurate
        and text_change_matched
        and speaker_change_matched
    )
    if not contract_passed:
        selection_outcome = "composition-contract-invalid"
    elif expected_state_selected:
        selection_outcome = "expected-state-selected"
    elif all_current_selections:
        selection_outcome = "current-state-selected"
    else:
        selection_outcome = "reachable-wrong-selection"
    return {
        "caseId": case["caseId"],
        "caseSha256": case["caseSha256"],
        "category": case["category"],
        "split": case["split"],
        "corpus": case["corpus"],
        "semanticPath": "production-mandatory",
        "executionSucceeded": True,
        "schemaContractPassed": contract_passed,
        "exactTextSelected": exact_text,
        "lexicalPreservationPassed": lexical_preserved,
        "speakerAccuracyPassed": speaker_accurate,
        "speakerBoundaryControlPassed": boundary_control,
        "safeModificationPassed": safe_modification,
        "targetSuggestionPresent": target_decision_present,
        "nonTargetSuggestionsAbsent": non_target_preserved,
        "targetDecisionPresent": target_decision_present,
        "nonTargetSelectionsPreserved": non_target_preserved,
        "unexpectedCandidateSelectionsAbsent": unexpected_selections_absent,
        "timelineSelectionMatched": timeline_selection_matched,
        "authorizedTextModification": authorized_text,
        "authorizedSpeakerModification": authorized_speaker,
        "expectedTextChangeMatched": text_change_matched,
        "expectedSpeakerChangeMatched": speaker_change_matched,
        "failureCodes": [],
        "providerCalls": 0,
        "decisionSha256": decision_sha,
        "selectionOutcome": selection_outcome,
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    applicable = list(rows)
    if field == "speakerBoundaryControlPassed":
        applicable = [
            row
            for row in rows
            if row.get("category") == "speaker-boundary-control"
        ]
    passed = sum(row.get(field) is True for row in applicable)
    return {
        "passed": passed,
        "applicable": len(applicable),
        "rate": passed / len(applicable) if applicable else None,
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failure_codes = Counter(
        str(code)
        for row in rows
        for code in row.get("failureCodes", [])
    )
    token_totals = {
        key: sum(
            int(row.get("tokenCounters", {}).get(key, 0))
            for row in rows
            if isinstance(row.get("tokenCounters"), Mapping)
        )
        for key in _PROVIDER_COUNTERS
    }
    return {
        "caseCount": len(rows),
        "wallTimeSeconds": sum(float(row.get("wallTimeSeconds", 0.0)) for row in rows),
        "providerCalls": sum(int(row.get("providerCalls", 0)) for row in rows),
        "tokenCounters": token_totals,
        "failureCodeCounts": dict(sorted(failure_codes.items())),
        "selectionOutcomeCounts": dict(
            sorted(
                Counter(
                    str(row.get("selectionOutcome") or "missing")
                    for row in rows
                ).items()
            )
        ),
        "rates": {
            field: _rate(rows, field)
            for field in _BOOLEAN_RESULTS
        },
    }


def _breakdowns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": aggregate_results(rows),
        "byCorpus": {
            corpus: aggregate_results(
                [row for row in rows if row.get("corpus") == corpus]
            )
            for corpus in CORPORA
        },
        "bySplit": {
            split: aggregate_results(
                [row for row in rows if row.get("split") == split]
            )
            for split in SPLITS
        },
        "byCategory": {
            category: aggregate_results(
                [row for row in rows if row.get("category") == category]
            )
            for category in CASE_CATEGORIES
        },
        "bySplitAndCategory": {
            split: {
                category: aggregate_results(
                    [
                        row
                        for row in rows
                        if row.get("split") == split
                        and row.get("category") == category
                    ]
                )
                for category in CASE_CATEGORIES
            }
            for split in SPLITS
        },
    }


def _selected_cases(
    manifest: Mapping[str, Any],
    *,
    split: str,
    case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = manifest["cases"]
    selected = [
        dict(row)
        for row in rows
        if split == "all" or row.get("split") == split
    ]
    if case_ids:
        requested = set(case_ids)
        available = {str(row["caseId"]) for row in selected}
        unknown = sorted(requested - available)
        if unknown:
            raise SemanticBenchmarkError(
                "unknown or split-excluded case IDs: " + ", ".join(unknown)
            )
        selected = [row for row in selected if row["caseId"] in requested]
    if not selected:
        raise SemanticBenchmarkError("semantic benchmark selection is empty")
    return selected


def _nearest_rank(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values, default=None),
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "maximum": max(values, default=None),
        "mean": sum(values) / len(values) if values else None,
    }


def _leaf_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {
            text
            for item in value.values()
            for text in _leaf_strings(item)
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return {text for item in value for text in _leaf_strings(item)}
    return set()


def _sensitive_trial_texts(cases: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    text_fields = {"text", "rawtext", "normalizedtext", "displaytext"}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in text_fields and isinstance(item, str):
                    if item:
                        result.add(item)
                else:
                    collect(item)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for item in value:
                collect(item)

    for case in cases:
        collect(case.get("document"))
        collect(case.get("expected"))
    return result


def _assert_report_redacted(
    report: Mapping[str, Any],
    *,
    cases: Sequence[Mapping[str, Any]],
) -> None:
    forbidden_keys = {
        "document",
        "expected",
        "promptbody",
        "requestbody",
        "responsebody",
        "rawresponse",
        "systemprompt",
        "transcripttext",
        "userprompt",
    }

    def inspect_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise SemanticBenchmarkError(
                        "semantic benchmark report contains a forbidden payload field"
                    )
                inspect_keys(item)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for item in value:
                inspect_keys(item)

    inspect_keys(report)
    report_strings = _leaf_strings(report)
    leaked = sorted(
        text
        for text in _sensitive_trial_texts(cases)
        if (
            text in report_strings
            or (
                len(text) >= 4
                and any(text in report_text for report_text in report_strings)
            )
        )
    )
    if leaked:
        raise SemanticBenchmarkError(
            "semantic benchmark report contains transcript or expected-answer text"
        )


def analyze_mandatory_batch_sizes(
    *,
    manifest_path: Path,
    batch_sizes: Sequence[int] = (3, 8, 16, 32),
    context_tokens: int = 8_192,
    output_tokens: int = 1_024,
) -> dict[str, Any]:
    if (
        isinstance(context_tokens, bool)
        or not isinstance(context_tokens, int)
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or context_tokens < 1_024
        or output_tokens < 128
        or output_tokens > context_tokens
    ):
        raise SemanticBenchmarkError("static batch token budget is invalid")
    normalized_batch_sizes = sorted(set(batch_sizes))
    if (
        not normalized_batch_sizes
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 32
            for value in normalized_batch_sizes
        )
    ):
        raise SemanticBenchmarkError(
            "static mandatory batch sizes must be unique values from 1 through 32"
        )
    resolved_manifest = manifest_path.resolve(strict=True)
    manifest = load_manifest(resolved_manifest)
    cases = [dict(item) for item in manifest["cases"]]
    input_budget = context_tokens - output_tokens
    rows: list[dict[str, Any]] = []
    for batch_size in normalized_batch_sizes:
        calls_per_case: list[int] = []
        estimates: list[int] = []
        overflow_case_ids: list[str] = []
        planning_failure_case_ids: list[str] = []
        for case in cases:
            provider = _StaticPlanningProvider()
            runner = SemanticJobArbitrationRunner(
                provider=provider,
                model="static-planning",
                context_tokens=context_tokens,
                output_tokens=output_tokens,
                batch_size=batch_size,
                max_batch_attempts=1,
            )
            try:
                lattice = build_semantic_candidate_lattice_from_document(
                    case["document"]
                )
                runner.run(case["document"], candidate_lattice=lattice)
            except (
                LocalLLMError,
                SemanticCompositionError,
                WorkerError,
                ValueError,
            ):
                planning_failure_case_ids.append(str(case["caseId"]))
            call_estimates = [
                int(item["estimatedInputTokens"])
                for item in provider.calls
            ]
            calls_per_case.append(len(call_estimates))
            estimates.extend(call_estimates)
            if any(value > input_budget for value in call_estimates):
                overflow_case_ids.append(str(case["caseId"]))
        overflow_call_count = sum(value > input_budget for value in estimates)
        rows.append(
            {
                "batchSize": batch_size,
                "caseCount": len(cases),
                "totalProviderCalls": sum(calls_per_case),
                "callsPerCase": _distribution(calls_per_case),
                "estimatedInputTokens": _distribution(estimates),
                "inputTokenBudget": input_budget,
                "overflowCallCount": overflow_call_count,
                "overflowCaseCount": len(overflow_case_ids),
                "overflowCaseIds": overflow_case_ids,
                "planningFailureCount": len(planning_failure_case_ids),
                "planningFailureCaseIds": planning_failure_case_ids,
            }
        )
    eligible = [
        row
        for row in rows
        if row["overflowCallCount"] == 0
        and row["planningFailureCount"] == 0
    ]
    selected = min(
        eligible,
        key=lambda row: (row["totalProviderCalls"], row["batchSize"]),
        default=None,
    )
    body = {
        "schemaVersion": "1.0.0",
        "artifactType": "mandatory-semantic-static-batch-preflight",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(resolved_manifest),
            "fileSha256": sha256_file(resolved_manifest),
            "canonicalSha256": manifest["canonicalSha256"],
            "schemaVersion": manifest["schemaVersion"],
            "caseCount": manifest["counts"]["cases"],
            "candidateGeneratorRevision": manifest["candidateGenerator"][
                "revision"
            ],
        },
        "semanticPath": "production-mandatory",
        "promptVersion": SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION,
        "answerUsePolicy": "current-candidate-only-no-expected-answer-access",
        "evaluationPolicy": {
            "stage": "structural-preflight",
            "structuralPlanningOnly": True,
            "qualityScoringPerformed": False,
            "expectedAnswersUsedForPlanning": False,
            "heldOutQualityObserved": False,
            "allSplitsTraversedForStructuralSizing": True,
            "productionPromotionEvidence": False,
        },
        "contextTokens": context_tokens,
        "outputTokens": output_tokens,
        "inputTokenBudget": input_budget,
        "deploymentBatchSize": _DEPLOYMENT_BATCH_SIZE,
        "batchSizes": rows,
        "selection": {
            "policy": "zero-overflow-minimum-total-calls-then-smallest-batch",
            "selectedBatchSize": (
                int(selected["batchSize"]) if selected is not None else None
            ),
            "passed": selected is not None,
            "configurationParity": bool(
                selected is not None
                and selected["batchSize"] == _DEPLOYMENT_BATCH_SIZE
            ),
            "promotionUsePolicy": (
                "screening-only; a finalist requires separate batch-8 "
                "development, held-out, and production-smoke evidence"
            ),
        },
        "sourceProvenance": _source_provenance("production-mandatory"),
    }
    return {**body, "canonicalSha256": canonical_json_sha256(body)}


def run_benchmark(
    *,
    manifest_path: Path,
    endpoint: str,
    model: str,
    model_digest: str,
    split: str = "development",
    case_ids: Sequence[str] = (),
    timeout_seconds: float = 600.0,
    context_tokens: int = 8_192,
    output_tokens: int = 1_024,
    batch_size: int | None = None,
    semantic_path: str = "production-mandatory",
    evaluation_stage: str = "screening",
    allow_held_out: bool = False,
    max_batch_attempts: int = 2,
    replicate_index: int = 1,
) -> dict[str, Any]:
    if split not in {*SPLITS, "all"}:
        raise SemanticBenchmarkError("benchmark split is invalid")
    if semantic_path not in SEMANTIC_PATHS:
        raise SemanticBenchmarkError("benchmark semantic path is invalid")
    if evaluation_stage not in _EVALUATION_STAGES:
        raise SemanticBenchmarkError("benchmark evaluation stage is invalid")
    if not isinstance(allow_held_out, bool):
        raise SemanticBenchmarkError("held-out authorization must be boolean")
    if split in {"held-out", "all"} and (
        evaluation_stage != "promotion" or not allow_held_out
    ):
        raise SemanticBenchmarkError(
            "held-out benchmark access requires promotion stage and explicit unlock"
        )
    if evaluation_stage == "promotion" and semantic_path != "production-mandatory":
        raise SemanticBenchmarkError(
            "promotion evaluation requires the production mandatory path"
        )
    if not isinstance(model, str) or _SAFE_MODEL_NAME.fullmatch(model) is None:
        raise SemanticBenchmarkError("benchmark model name is invalid")
    normalized_model_digest = _normalized_digest(model_digest)
    if normalized_model_digest is None:
        raise SemanticBenchmarkError(
            "benchmark model digest must be a pinned SHA-256 digest"
        )
    resolved_batch_size = (
        (
            _DEPLOYMENT_BATCH_SIZE
            if evaluation_stage == "promotion"
            else _DEFAULT_BATCH_SIZE[semantic_path]
        )
        if batch_size is None
        else batch_size
    )
    maximum_batch_size = 32 if semantic_path == "production-mandatory" else 6
    if (
        isinstance(resolved_batch_size, bool)
        or not isinstance(resolved_batch_size, int)
        or not 1 <= resolved_batch_size <= maximum_batch_size
    ):
        raise SemanticBenchmarkError(
            f"{semantic_path} batch size must be between 1 and "
            f"{maximum_batch_size}"
        )
    if (
        evaluation_stage == "promotion"
        and resolved_batch_size != _DEPLOYMENT_BATCH_SIZE
    ):
        raise SemanticBenchmarkError(
            "promotion evaluation must use deployment batch size 8"
        )
    if (
        isinstance(max_batch_attempts, bool)
        or not isinstance(max_batch_attempts, int)
        or not 1 <= max_batch_attempts <= 3
    ):
        raise SemanticBenchmarkError("max batch attempts must be between 1 and 3")
    if (
        isinstance(replicate_index, bool)
        or not isinstance(replicate_index, int)
        or replicate_index < 1
    ):
        raise SemanticBenchmarkError("replicate index must be positive")
    resolved_manifest = manifest_path.resolve(strict=True)
    manifest = load_manifest(resolved_manifest)
    cases = _selected_cases(
        manifest,
        split=split,
        case_ids=case_ids,
    )
    config = LocalLLMConfig(
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        temperature=0.0,
        top_p=0.1,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        keep_alive="10m",
        release_on_close=True,
        offline_only=True,
        expected_model_digest=normalized_model_digest,
    )
    model_runtime = _model_runtime_provenance(config)
    provider = _InstrumentedProvider(OllamaLocalProvider(config))
    runner: SemanticProcessingRunner | SemanticJobArbitrationRunner
    if semantic_path == "production-mandatory":
        runner = SemanticJobArbitrationRunner(
            provider=provider,
            model=config.model,
            batch_size=resolved_batch_size,
            max_batch_attempts=max_batch_attempts,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
        )
    else:
        runner = SemanticProcessingRunner(
            provider=provider,
            model=config.model,
            batch_size=resolved_batch_size,
            speaker_top_k=3,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
        )
    results: list[dict[str, Any]] = []
    resource_snapshots = [
        _resource_snapshot(
            config.endpoint,
            config.model,
            phase="before-run",
        )
    ]
    benchmark_started = time.perf_counter()
    release_succeeded = False
    release_failure_code: str | None = None
    try:
        for case in cases:
            before = _provider_metrics(provider)
            attempted_before = provider.attempted_calls
            evidence_before = len(provider.call_evidence)
            started = time.perf_counter()
            try:
                if semantic_path == "production-mandatory":
                    lattice = build_semantic_candidate_lattice_from_document(
                        case["document"]
                    )
                    artifact = runner.run(
                        case["document"],
                        candidate_lattice=lattice,
                    )
                    result = score_mandatory_case(case, lattice, artifact)
                else:
                    artifact = runner.run(case["document"])
                    result = score_case(case, artifact)
            except (
                LocalLLMError,
                SemanticCompositionError,
                WorkerError,
                ValueError,
            ) as exc:
                code = exc.code if isinstance(exc, WorkerError) else type(exc).__name__
                result = _failed_result(
                    case,
                    failure_code=str(code),
                    semantic_path=semantic_path,
                )
            elapsed = time.perf_counter() - started
            after = _provider_metrics(provider)
            result["wallTimeSeconds"] = elapsed
            result["tokenCounters"] = _counter_delta(before, after)
            result["providerCalls"] = provider.attempted_calls - attempted_before
            result["providerCallEvidence"] = [
                dict(item) for item in provider.call_evidence[evidence_before:]
            ]
            results.append(result)
            resource_snapshots.append(
                _resource_snapshot(
                    config.endpoint,
                    config.model,
                    phase=f"after-case:{case['caseId']}",
                )
            )
    finally:
        resource_snapshots.append(
            _resource_snapshot(
                config.endpoint,
                config.model,
                phase="before-release",
            )
        )
        try:
            runner.release_resources()
            release_succeeded = True
        except (LocalLLMError, WorkerError, ValueError) as exc:
            release_failure_code = (
                exc.code if isinstance(exc, WorkerError) else type(exc).__name__
            )
        resource_snapshots.append(
            _wait_for_release_snapshot(config.endpoint, config.model)
        )
    total_wall_seconds = time.perf_counter() - benchmark_started
    resources = _resource_summary(resource_snapshots)
    breakdowns = _breakdowns(results)
    all_contracts = all(row["schemaContractPassed"] is True for row in results)
    all_safe = all(row["safeModificationPassed"] is True for row in results)
    all_executed = all(
        row["executionSucceeded"] is True for row in results
    )
    all_provider_calls_hashed = _provider_call_evidence_complete(provider)
    release_verified = bool(
        release_succeeded and resources["modelAbsentAfterRelease"] is True
    )
    path_validation_passed = bool(
        all_executed
        and all_contracts
        and all_safe
        and all_provider_calls_hashed
        and release_verified
    )
    configuration_parity = bool(
        semantic_path == "production-mandatory"
        and resolved_batch_size == _DEPLOYMENT_BATCH_SIZE
    )
    screening_gate_passed = bool(
        semantic_path == "production-mandatory"
        and evaluation_stage == "screening"
        and path_validation_passed
    )
    promotion_gate_passed = bool(
        semantic_path == "production-mandatory"
        and evaluation_stage == "promotion"
        and configuration_parity
        and path_validation_passed
    )
    stage_gate_passed = screening_gate_passed or promotion_gate_passed
    decision_set_sha256 = canonical_json_sha256(
        [
            {
                "caseId": row["caseId"],
                "decisionSha256": row.get("decisionSha256"),
            }
            for row in results
        ]
    )
    prompt_version = (
        SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION
        if semantic_path == "production-mandatory"
        else SEMANTIC_PROMPT_VERSION
    )
    runner_class = (
        "backend.semantic_composition.SemanticJobArbitrationRunner"
        if semantic_path == "production-mandatory"
        else "backend.semantic_processing.SemanticProcessingRunner"
    )
    report_body = {
        "schemaVersion": "2.0.0",
        "benchmark": "production-local-llm-semantic-trials",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(resolved_manifest),
            "fileSha256": sha256_file(resolved_manifest),
            "canonicalSha256": manifest["canonicalSha256"],
            "caseCount": manifest["counts"]["cases"],
            "schemaVersion": manifest["schemaVersion"],
            "candidateGeneratorRevision": manifest["candidateGenerator"][
                "revision"
            ],
        },
        "productionPath": {
            "semanticPath": semantic_path,
            "providerClass": "backend.local_llm.OllamaLocalProvider",
            "runnerClass": runner_class,
            "promptVersion": prompt_version,
            "endpoint": config.endpoint,
            "model": config.model,
            "expectedModelDigest": config.expected_model_digest,
            "providerId": provider.provider_id,
            "providerVersion": provider.provider_version,
            "networkPolicy": provider.network_policy,
            "timeoutSeconds": config.timeout_seconds,
            "contextTokens": config.context_tokens,
            "outputTokens": config.output_tokens,
            "batchSize": resolved_batch_size,
            "deploymentBatchSize": _DEPLOYMENT_BATCH_SIZE,
            "configurationParity": configuration_parity,
            "evaluationStage": evaluation_stage,
            "maxBatchAttempts": (
                max_batch_attempts
                if semantic_path == "production-mandatory"
                else None
            ),
            "temperature": config.temperature,
            "topP": config.top_p,
        },
        "modelRuntime": model_runtime,
        "sourceProvenance": _source_provenance(semantic_path),
        "determinism": {
            "replicateIndex": replicate_index,
            "temperature": config.temperature,
            "topP": config.top_p,
            "seed": None,
            "seedControlConfigured": False,
            "bitwiseRepeatabilityClaimed": False,
            "decisionSetSha256": decision_set_sha256,
            "comparisonRequirement": (
                "match decisionSetSha256 across independently executed replicates"
            ),
        },
        "selection": {
            "split": split,
            "evaluationStage": evaluation_stage,
            "heldOutExplicitlyUnlocked": allow_held_out,
            "caseCount": len(cases),
            "explicitCaseSelection": bool(case_ids),
            "caseIds": [str(case["caseId"]) for case in cases],
        },
        "execution": {
            "wallTimeSeconds": total_wall_seconds,
            "resourceReleaseSucceeded": release_succeeded,
            "resourceReleaseFailureCode": release_failure_code,
            "resourceReleaseVerifiedAbsent": resources[
                "modelAbsentAfterRelease"
            ],
            "attemptedProviderCalls": provider.attempted_calls,
            "providerCounters": _provider_metrics(provider),
            "providerCallEvidenceCanonicalSha256": canonical_json_sha256(
                provider.call_evidence
            ),
            "payloadPersistencePolicy": {
                "systemPromptBodiesPersisted": False,
                "userPromptBodiesPersisted": False,
                "responseSchemaBodiesPersisted": False,
                "responseBodiesPersisted": False,
                "transcriptTextPersisted": False,
                "callDigestsPersisted": True,
            },
            "resources": resources,
        },
        "validation": {
            "allCasesExecuted": all_executed,
            "allSchemaContractsPassed": all_contracts,
            "allSafeModificationsPassed": all_safe,
            "allProviderCallsHashed": all_provider_calls_hashed,
            "resourceReleasePassed": release_verified,
            "pathValidationPassed": path_validation_passed,
            "configurationParity": configuration_parity,
            "screeningGatePassed": screening_gate_passed,
            "promotionConfigurationGatePassed": promotion_gate_passed,
            "productionMandatoryPathEvaluated": (
                semantic_path == "production-mandatory"
            ),
            "productionMandatoryGatePassed": promotion_gate_passed,
            "legacyDiagnosticOnly": semantic_path == "legacy-suggestions",
            "standaloneProductionPromotionApproved": False,
            "promotionEvidenceRequirements": [
                "batch-8-development",
                "batch-8-held-out",
                "batch-8-production-smoke",
            ],
            "passed": stage_gate_passed,
        },
        "breakdowns": breakdowns,
        "cases": results,
    }
    _assert_report_redacted(report_body, cases=cases)
    return {
        **report_body,
        "canonicalSha256": canonical_json_sha256(report_body),
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:27b-q4_K_M")
    parser.add_argument("--model-digest")
    parser.add_argument(
        "--semantic-path",
        choices=SEMANTIC_PATHS,
        default="production-mandatory",
    )
    parser.add_argument(
        "--evaluation-stage",
        choices=_EVALUATION_STAGES,
        default="screening",
    )
    parser.add_argument(
        "--split",
        choices=("all", *SPLITS),
        default="development",
    )
    parser.add_argument(
        "--unlock-held-out",
        action="store_true",
        help="explicitly authorize held-out quality evaluation in promotion stage",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=600.0,
    )
    parser.add_argument(
        "--context-tokens",
        type=_positive_integer,
        default=8_192,
    )
    parser.add_argument(
        "--output-tokens",
        type=_positive_integer,
        default=1_024,
    )
    parser.add_argument("--batch-size", type=_positive_integer)
    parser.add_argument(
        "--max-batch-attempts",
        type=_positive_integer,
        default=2,
    )
    parser.add_argument(
        "--replicate-index",
        type=_positive_integer,
        default=1,
    )
    parser.add_argument(
        "--static-batch-preflight",
        action="store_true",
        help="analyze mandatory prompt sizes without loading a model",
    )
    parser.add_argument(
        "--preflight-batch-size",
        action="append",
        type=_positive_integer,
        default=[],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.static_batch_preflight:
        report = analyze_mandatory_batch_sizes(
            manifest_path=args.manifest,
            batch_sizes=(
                args.preflight_batch_size
                if args.preflight_batch_size
                else (3, 8, 16, 32)
            ),
            context_tokens=args.context_tokens,
            output_tokens=args.output_tokens,
        )
        atomic_write_json_no_replace(args.output.resolve(), report)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "canonicalSha256": report["canonicalSha256"],
                    "selection": report["selection"],
                    "batchSizes": report["batchSizes"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report["selection"]["passed"] else 2
    if args.model_digest is None:
        parser.error(
            "--model-digest is required unless --static-batch-preflight is used"
        )
    report = run_benchmark(
        manifest_path=args.manifest,
        endpoint=args.endpoint,
        model=args.model,
        model_digest=args.model_digest,
        split=args.split,
        case_ids=args.case_id,
        timeout_seconds=args.timeout_seconds,
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        batch_size=args.batch_size,
        semantic_path=args.semantic_path,
        evaluation_stage=args.evaluation_stage,
        allow_held_out=args.unlock_held_out,
        max_batch_attempts=args.max_batch_attempts,
        replicate_index=args.replicate_index,
    )
    atomic_write_json_no_replace(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "canonicalSha256": report["canonicalSha256"],
                "validation": report["validation"],
                "overall": report["breakdowns"]["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
