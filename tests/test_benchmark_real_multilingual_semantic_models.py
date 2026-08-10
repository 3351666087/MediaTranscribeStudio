from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from backend.persistence import canonical_json_sha256, sha256_file
from backend.semantic_candidate_lattice import (
    build_semantic_candidate_lattice_from_document,
)
import tools.benchmark_real_multilingual_semantic_models as benchmark_module
from tools.benchmark_real_multilingual_semantic_models import (
    RealMultilingualSemanticBenchmarkError,
    SemanticModelSpec,
    _default_provider_factory,
    benchmark_real_multilingual_semantic_models,
    compare_composition_to_baseline,
    load_frozen_semantic_cases,
    load_model_specs_from_manifest,
    parse_model_specs,
    publish_model_reports_no_replace,
)
from tools.freeze_real_multilingual_semantic_development import (
    _semantic_calibration_target,
)


def _segment(
    segment_id: str,
    *,
    start_ms: int,
    speaker_id: str,
    text: str,
) -> dict[str, Any]:
    return {
        "id": segment_id,
        "startMs": start_ms,
        "endMs": start_ms + 1_000,
        "speakerId": speaker_id,
        "rawText": text,
        "normalizedText": text,
        "displayText": text,
        "confidence": 0.9,
        "speakerScores": [{"speakerId": speaker_id, "score": 0.9}],
        "speakerMargin": 0.5,
        "overlapping": False,
        "humanLocked": False,
        "revisions": [],
        "language": "zh",
        "evidence": {"asr": {"provider": "fixture-asr"}},
    }


def _document() -> dict[str, Any]:
    return {
        "schemaVersion": "2.0.0",
        "documentId": "document-frozen-semantic-fixture",
        "jobId": "job-frozen-semantic-fixture",
        "generatedAt": "2026-08-09T00:00:00Z",
        "language": "zh",
        "source": {
            "fileName": "frozen-fixture.wav",
            "sha256": "a" * 64,
            "durationMs": 2_000,
        },
        "speakerPolicy": {
            "mode": "manual",
            "resolvedCount": 2,
            "speakerIds": ["speaker-1", "speaker-2"],
        },
        "speakers": [{"id": "speaker-1"}, {"id": "speaker-2"}],
        "segments": [
            _segment(
                "segment-1",
                start_ms=0,
                speaker_id="speaker-1",
                text="SECRET_TRANSCRIPT_ALPHA",
            ),
            _segment(
                "segment-2",
                start_ms=1_000,
                speaker_id="speaker-2",
                text="SECRET_TRANSCRIPT_BETA",
            ),
        ],
        "provenance": {"offline": True, "models": []},
    }


def _baseline(document: dict[str, Any], lattice: dict[str, Any]) -> dict[str, Any]:
    segments = [
        {
            "id": segment["id"],
            "startMs": segment["startMs"],
            "endMs": segment["endMs"],
            "speakerId": segment["speakerId"],
            "language": segment["language"],
            "finalText": segment["normalizedText"],
            "overlapping": False,
        }
        for segment in document["segments"]
    ]
    return {
        "schemaVersion": "1.0.0",
        "artifactType": "final-adjudicated-transcript",
        "status": "final",
        "disposition": "transcribable-speech",
        "adjudicationSource": "codex-manual",
        "input": {
            "transcriptDocumentSha256": canonical_json_sha256(document),
            "candidateLatticeSha256": lattice["latticeSha256"],
        },
        "segments": segments,
        "timeline": {
            "turns": [
                {
                    "startMs": segment["startMs"],
                    "endMs": segment["endMs"],
                    "speakerId": segment["speakerId"],
                    "overlap": False,
                }
                for segment in segments
            ]
        },
    }


def _manifest_value(*, case_count: int = 1) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for index in range(case_count):
        document = _document()
        if index:
            document["documentId"] = f"document-frozen-semantic-fixture-{index}"
            document["jobId"] = f"job-frozen-semantic-fixture-{index}"
            document["source"]["sha256"] = f"{index + 10:064x}"
        lattice = build_semantic_candidate_lattice_from_document(document)
        baseline = _baseline(document, lattice)
        cases.append(
            {
                "caseId": f"frozen-zh-case-{index:02d}",
                "language": "zh",
                "document": document,
                "lattice": lattice,
                "baseline": baseline,
                "semanticCalibrationTarget": _semantic_calibration_target(
                    lattice,
                    baseline,
                ),
            }
        )
    return {
        "schemaVersion": "1.0.0",
        "cases": cases,
    }


def _write_manifest(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "frozen-cases.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _refresh_calibration_sha256(target: dict[str, Any]) -> None:
    body = copy.deepcopy(target)
    body.pop("canonicalSha256", None)
    target["canonicalSha256"] = canonical_json_sha256(body)


class _SelectingProvider:
    provider_id = "fixture-semantic-provider"
    provider_version = "1"
    network_policy = "loopback-only"

    def __init__(
        self,
        model: str,
        events: list[str],
        lattice_hashes: list[tuple[str, str]],
        *,
        fail: bool = False,
        fail_once: bool = False,
        invalid_once: bool = False,
    ) -> None:
        self.model = model
        self.events = events
        self.lattice_hashes = lattice_hashes
        self.fail = fail
        self.fail_once = fail_once
        self.invalid_once = invalid_once
        self.attempted_calls = 0
        self.completed_calls = 0
        self._model_digest_verified = False

    @property
    def generation_metrics(self) -> dict[str, int]:
        return {
            "completedCalls": self.completed_calls,
            "totalDurationNanoseconds": self.completed_calls * 1_000_000_000,
            "loadDurationNanoseconds": (
                500_000_000 if self.completed_calls else 0
            ),
        }

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.attempted_calls += 1
        self.events.append(f"generate:{self.model}")
        prompt = json.loads(kwargs["user_prompt"])
        candidate_lattice = prompt["candidateLattice"]
        self.lattice_hashes.append(
            (self.model, canonical_json_sha256(candidate_lattice))
        )
        if self.fail or (self.fail_once and self.attempted_calls == 1):
            raise ValueError("fixture provider failure")
        groups = sorted(
            candidate_lattice["targetGroups"],
            key=lambda item: item["groupPosition"],
        )
        choices = {
            str(group["groupPosition"]): next(
                candidate["choiceIndex"]
                for candidate in group["candidates"]
                if candidate["current"] is True
            )
            for group in groups
        }
        if self.invalid_once and self.attempted_calls == 1:
            self.completed_calls += 1
            self._model_digest_verified = True
            return {
                "choiceByPosition": {
                    position: 999 for position in choices
                }
            }
        self.completed_calls += 1
        self._model_digest_verified = True
        return {"choiceByPosition": choices}

    def release_resources(self) -> None:
        self.events.append(f"release:{self.model}")


class _RequestingProvider(_SelectingProvider):
    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.attempted_calls += 1
        self.events.append(f"generate:{self.model}")
        prompt = json.loads(kwargs["user_prompt"])
        candidate_lattice = prompt["candidateLattice"]
        self.lattice_hashes.append(
            (self.model, canonical_json_sha256(candidate_lattice))
        )
        properties = kwargs["response_schema"]["properties"][
            "choiceByPosition"
        ]["properties"]
        choices = {
            position: (
                -1
                if -1 in schema["enum"]
                else next(value for value in schema["enum"] if value >= 0)
            )
            for position, schema in properties.items()
        }
        self.completed_calls += 1
        self._model_digest_verified = True
        return {"choiceByPosition": choices}


class _ScopedRequestProvider(_SelectingProvider):
    def __init__(
        self,
        model: str,
        events: list[str],
        lattice_hashes: list[tuple[str, str]],
        *,
        request_domain: str,
        request_scope_id: str,
        captured_prompts: list[str] | None = None,
    ) -> None:
        super().__init__(model, events, lattice_hashes)
        self.request_domain = request_domain
        self.request_scope_id = request_scope_id
        self.captured_prompts = captured_prompts

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.attempted_calls += 1
        self.events.append(f"generate:{self.model}")
        user_prompt = kwargs["user_prompt"]
        if self.captured_prompts is not None:
            self.captured_prompts.append(user_prompt)
        prompt = json.loads(user_prompt)
        candidate_lattice = prompt["candidateLattice"]
        self.lattice_hashes.append(
            (self.model, canonical_json_sha256(candidate_lattice))
        )
        choices: dict[str, int] = {}
        for group in candidate_lattice["targetGroups"]:
            candidates = group["candidates"]
            choices[str(group["groupPosition"])] = (
                -1
                if group["domain"] == self.request_domain
                and group["scopeId"] == self.request_scope_id
                else next(
                    candidate["choiceIndex"]
                    for candidate in candidates
                    if candidate["current"] is True
                )
            )
        self.completed_calls += 1
        self._model_digest_verified = True
        return {"choiceByPosition": choices}


class _EquivalentTimelineSelectingProvider(_SelectingProvider):
    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.attempted_calls += 1
        self.events.append(f"generate:{self.model}")
        prompt = json.loads(kwargs["user_prompt"])
        candidate_lattice = prompt["candidateLattice"]
        self.lattice_hashes.append(
            (self.model, canonical_json_sha256(candidate_lattice))
        )
        choices: dict[str, int] = {}
        for group in candidate_lattice["targetGroups"]:
            candidates = group["candidates"]
            if (
                group["domain"] == "speaker-cardinality-timeline"
                and len(candidates) > 1
            ):
                selected = next(
                    candidate
                    for candidate in candidates
                    if candidate["current"] is False
                )
            else:
                selected = next(
                    candidate
                    for candidate in candidates
                    if candidate["current"] is True
                )
            choices[str(group["groupPosition"])] = selected["choiceIndex"]
        self.completed_calls += 1
        self._model_digest_verified = True
        return {"choiceByPosition": choices}


def _fixture_resource_probe(
    spec: SemanticModelSpec,
    phase: str,
) -> dict[str, Any]:
    loaded = phase not in {
        "before-run",
        "before-provider-call:1",
        "after-release",
    }
    return {
        "phase": phase,
        "capturedAt": "2026-08-09T00:00:00Z",
        "ollamaApiAvailable": True,
        "selectedModelLoaded": loaded,
        "selectedModel": (
            [
                {
                    "model": spec.model,
                    "digest": spec.digest,
                    "sizeBytes": 1_000,
                    "sizeVramBytes": 600,
                }
            ]
            if loaded
            else []
        ),
        "process": {
            "available": True,
            "benchmarkProcessRssBytes": 100,
            "ollamaProcessCount": 1 if loaded else 0,
            "ollamaPids": [123] if loaded else [],
            "ollamaAggregateRssBytes": 800 if loaded else 0,
            "observedProcessNames": ["ollama.exe"] if loaded else [],
            "totalPhysicalMemoryBytes": 32_000,
            "availablePhysicalMemoryBytes": 20_000 if loaded else 25_000,
            "usedPhysicalMemoryBytes": 12_000 if loaded else 7_000,
            "gpuProcessMemory": {
                "available": True,
                "measurement": "fixture-nvidia-smi",
                "matchedProcessCount": 1 if loaded else 0,
                "ollamaAggregateVramBytes": 500 if loaded else 0,
                "processes": (
                    [{"pid": 123, "usedVramBytes": 500}] if loaded else []
                ),
            },
            "wholeGpuMemory": {
                "available": True,
                "measurement": "fixture-whole-gpu-memory",
                "aggregateUsedVramBytes": 1_500 if loaded else 1_000,
                "aggregateFreeVramBytes": 6_500 if loaded else 7_000,
                "includesOtherProcesses": True,
                "diagnosticOnly": True,
                "devices": [
                    {
                        "gpuIndex": 0,
                        "usedVramBytes": 1_500 if loaded else 1_000,
                        "freeVramBytes": 6_500 if loaded else 7_000,
                    }
                ],
            },
        },
    }


def test_load_cases_rejects_missing_manual_adjudication_source(
    tmp_path: Path,
) -> None:
    manifest = _manifest_value()
    manifest["cases"][0]["baseline"].pop("adjudicationSource")

    with pytest.raises(
        RealMultilingualSemanticBenchmarkError,
        match="must declare human or Codex",
    ):
        load_frozen_semantic_cases(_write_manifest(tmp_path, manifest))


@pytest.mark.parametrize(
    "binding_field",
    ("documentSha256", "latticeSha256", "baselineSha256"),
)
def test_load_cases_rejects_rebound_declared_hashes(
    tmp_path: Path,
    binding_field: str,
) -> None:
    manifest = _manifest_value()
    manifest["cases"][0][binding_field] = "b" * 64

    with pytest.raises(RealMultilingualSemanticBenchmarkError, match="hash does not match"):
        load_frozen_semantic_cases(_write_manifest(tmp_path, manifest))


def test_load_cases_rejects_internal_document_and_lattice_rebinding(
    tmp_path: Path,
) -> None:
    baseline_rebound = _manifest_value()
    baseline_rebound["cases"][0]["baseline"]["input"][
        "transcriptDocumentSha256"
    ] = "b" * 64
    with pytest.raises(RealMultilingualSemanticBenchmarkError, match="another document"):
        load_frozen_semantic_cases(_write_manifest(tmp_path, baseline_rebound))

    lattice_rebound = _manifest_value()
    lattice_rebound["cases"][0]["lattice"]["binding"]["transcriptSha256"] = (
        "b" * 64
    )
    with pytest.raises(RealMultilingualSemanticBenchmarkError, match="invalid or rebound"):
        load_frozen_semantic_cases(_write_manifest(tmp_path, lattice_rebound))


def test_load_cases_requires_hidden_calibration_target_and_keeps_it_out_of_bindings(
    tmp_path: Path,
) -> None:
    missing = _manifest_value()
    missing["cases"][0].pop("semanticCalibrationTarget")
    with pytest.raises(
        RealMultilingualSemanticBenchmarkError,
        match="semanticCalibrationTarget must be an object",
    ):
        load_frozen_semantic_cases(_write_manifest(tmp_path, missing))

    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    case = case_set.cases[0]
    assert case.semantic_calibration_target["groups"]
    serialized_case_binding = json.dumps(case.input_binding(), sort_keys=True)
    serialized_set_binding = json.dumps(case_set.binding(), sort_keys=True)
    for forbidden in (
        "semanticCalibrationTarget",
        "baselineProjectionSha256",
        "targetCandidateId",
        "acceptableCandidateIds",
    ):
        assert forbidden not in serialized_case_binding
        assert forbidden not in serialized_set_binding


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("canonical", "canonical SHA-256 does not match"),
        ("coverage", "group coverage does not match"),
        ("group-binding", "rebound to another lattice group"),
        ("projection", "baseline projection SHA-256 does not match"),
        ("acceptable", "not exact eligible matches"),
        ("target", "not the canonical reachable target"),
        ("alternative-count", "does not match the lattice"),
        ("counts", "counts do not match"),
        ("counts-type", "counts do not match"),
    ),
)
def test_load_cases_strictly_validates_hidden_calibration_target(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    manifest = _manifest_value()
    target = manifest["cases"][0]["semanticCalibrationTarget"]
    first_group = target["groups"][0]
    if mutation == "canonical":
        target["canonicalSha256"] = "f" * 64
    elif mutation == "coverage":
        target["groups"].pop()
        _refresh_calibration_sha256(target)
    elif mutation == "group-binding":
        first_group["groupId"] = "group-rebound"
        _refresh_calibration_sha256(target)
    elif mutation == "projection":
        first_group["baselineProjectionSha256"] = "f" * 64
        _refresh_calibration_sha256(target)
    elif mutation == "acceptable":
        first_group["acceptableCandidateIds"] = []
        _refresh_calibration_sha256(target)
    elif mutation == "target":
        first_group["targetCandidateId"] = "candidate-unreachable"
        _refresh_calibration_sha256(target)
    elif mutation == "alternative-count":
        first_group["eligibleAlternativeCount"] += 1
        _refresh_calibration_sha256(target)
    elif mutation == "counts":
        target["counts"]["preservation-only"] += 1
        _refresh_calibration_sha256(target)
    elif mutation == "counts-type":
        target["counts"]["select"] = False
        _refresh_calibration_sha256(target)

    with pytest.raises(RealMultilingualSemanticBenchmarkError, match=message):
        load_frozen_semantic_cases(_write_manifest(tmp_path, manifest))


def test_model_specs_preserve_config_identity_and_digest(tmp_path: Path) -> None:
    config_manifest = tmp_path / "config-set.manifest.json"
    config_manifest.write_text(
        json.dumps(
            {
                "configs": [
                    {
                        "modelId": "semantic-qwen-fixture",
                        "model": "qwen-fixture:27b-q4_K_M",
                        "digest": "sha256:" + "c" * 64,
                        "configPath": "D:/fixtures/qwen-27b.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_model_specs_from_manifest(config_manifest)
    assert specs == (
        SemanticModelSpec(
            model="qwen-fixture:27b-q4_K_M",
            digest="sha256:" + "c" * 64,
            model_id="semantic-qwen-fixture",
            config_path="D:/fixtures/qwen-27b.json",
        ),
    )
    assert parse_model_specs(["qwen-fixture:9b=" + "d" * 64])[0].digest == (
        "sha256:" + "d" * 64
    )


def test_default_provider_keeps_one_model_batch_warm_then_releases() -> None:
    provider = _default_provider_factory(
        SemanticModelSpec("semantic-fixture:27b", "e" * 64),
        endpoint="http://127.0.0.1:11434",
        timeout_seconds=10.0,
        context_tokens=32_768,
        output_tokens=4_096,
    )

    assert provider.config.keep_alive == "10m"
    assert provider.config.release_on_close is True
    assert provider.config.expected_model_digest == "sha256:" + "e" * 64


def test_comparison_reports_four_independent_exact_match_dimensions() -> None:
    manifest = _manifest_value()
    baseline = manifest["cases"][0]["baseline"]
    composition = {
        "segments": copy.deepcopy(baseline["segments"]),
        "timeline": copy.deepcopy(baseline["timeline"]),
    }

    exact = compare_composition_to_baseline(composition, baseline)
    assert exact["exactMatch"] is True
    assert all(
        exact[field]["exactMatch"] is True
        for field in ("speaker", "language", "text", "timeline")
    )

    mutations = {
        "speaker": lambda value: value["segments"][0].update(
            speakerId="speaker-2"
        ),
        "language": lambda value: value["segments"][0].update(language="en"),
        "text": lambda value: value["segments"][0].update(
            finalText="CHANGED_TRANSCRIPT"
        ),
        "timeline": lambda value: value["timeline"]["turns"][0].update(
            speakerId="speaker-2"
        ),
    }
    for field, mutate in mutations.items():
        changed = copy.deepcopy(composition)
        mutate(changed)
        comparison = compare_composition_to_baseline(changed, baseline)
        assert comparison[field]["exactMatch"] is False
        assert comparison["exactMatch"] is False
        serialized = json.dumps(comparison, sort_keys=True)
        assert "SECRET_TRANSCRIPT_ALPHA" not in serialized
        assert "CHANGED_TRANSCRIPT" not in serialized


def test_benchmark_reuses_frozen_lattice_and_releases_models_serially(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    original_lattice_hash = canonical_json_sha256(case_set.cases[0].lattice)
    events: list[str] = []
    lattice_hashes: list[tuple[str, str]] = []

    def provider_factory(spec: SemanticModelSpec) -> _SelectingProvider:
        events.append(f"factory:{spec.model}")
        return _SelectingProvider(spec.model, events, lattice_hashes)

    specs = (
        SemanticModelSpec("semantic-a:9b", "1" * 64),
        SemanticModelSpec("semantic-b:27b", "2" * 64),
    )
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        specs,
        provider_factory=provider_factory,
        now=lambda: "2026-08-09T01:00:00Z",
        max_batch_attempts=1,
    )

    assert events == [
        "factory:semantic-a:9b",
        "generate:semantic-a:9b",
        "release:semantic-a:9b",
        "factory:semantic-b:27b",
        "generate:semantic-b:27b",
        "release:semantic-b:27b",
    ]
    assert len(lattice_hashes) == 2
    assert lattice_hashes[0][0] == "semantic-a:9b"
    assert lattice_hashes[1][0] == "semantic-b:27b"
    assert lattice_hashes[0][1] == lattice_hashes[1][1]
    assert canonical_json_sha256(case_set.cases[0].lattice) == original_lattice_hash
    assert all(
        model["cases"][0]["input"]["latticeSha256"]
        == case_set.cases[0].lattice_sha256
        for model in report["models"]
    )
    assert all(
        model["cases"][0]["comparison"]["exactMatch"] is True
        for model in report["models"]
    )
    for model in report["models"]:
        calibration = model["cases"][0]["calibration"]
        assert calibration["groupCount"] == 8
        assert calibration["correctCount"] == 8
        assert calibration["microAccuracy"] == 1.0
        assert calibration["byExpectedAction"]["preservation-only"] == {
            "count": 8,
            "correctCount": 8,
            "accuracy": 1.0,
        }
        assert all(
            set(group) == {
                "groupBindingSha256",
                "expectedAction",
                "actualAction",
                "correct",
            }
            for group in calibration["groups"]
        )
        aggregate_calibration = model["aggregate"]["calibration"]
        assert aggregate_calibration["groupCount"] == 8
        assert aggregate_calibration["correctCount"] == 8
        assert aggregate_calibration["microAccuracy"] == 1.0
    assert all(
        model["execution"]["release"]["attempted"] is True
        and model["execution"]["release"]["succeeded"] is True
        and model["execution"]["release"]["postReleaseModelAbsent"] is False
        and model["execution"]["release"]["transitionVerified"] is False
        and model["execution"]["release"]["failureCode"] is None
        for model in report["models"]
    )
    assert report["execution"]["requestedModelCount"] == 2
    assert report["execution"]["executedModelCount"] == 2
    assert report["execution"]["skippedModelCount"] == 0
    assert report["validation"]["allModelsIndependent"] is False
    assert all(
        model["execution"]["isolation"]["evidenceSource"]
        == "custom-provider-resource-probe-unavailable"
        and model["execution"]["isolation"]["verified"] is False
        and model["execution"]["isolation"]["continuationAllowed"] is True
        for model in report["models"]
    )
    configuration = report["executionConfiguration"]
    assert configuration["providerMode"] == "custom-provider"
    assert configuration["endpoint"] is None
    assert configuration["endpointApplied"] is False
    assert configuration["timeoutSeconds"] is None
    assert configuration["timeoutAppliedByDefaultProvider"] is False
    assert configuration["topP"] is None
    assert configuration["keepAliveWithinModelBatch"] is None
    assert report["evaluationPolicy"]["productionConfigMutated"] is False
    assert report["evaluationPolicy"][
        "challengerMayReplaceProductionImmediatelyAfterBlindWin"
    ] is True
    assert report["evaluationPolicy"]["incumbentProtection"] is False
    assert report["validation"]["promotionAuthorized"] is False
    assert report["comparison"]["winner"] is None

    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert "SECRET_TRANSCRIPT_ALPHA" not in serialized
    assert "SECRET_TRANSCRIPT_BETA" not in serialized
    assert "choiceByPosition" not in serialized
    for forbidden in (
        "semanticCalibrationTarget",
        "baselineProjectionSha256",
        "targetCandidateId",
        "acceptableCandidateIds",
    ):
        assert forbidden not in serialized


def test_calibration_accepts_semantically_equivalent_timeline_selection(
    tmp_path: Path,
) -> None:
    manifest = _manifest_value()
    case = manifest["cases"][0]
    document = case["document"]
    turns = [
        {
            "startMs": segment["startMs"],
            "endMs": segment["endMs"],
            "speakerId": segment["speakerId"],
        }
        for segment in document["segments"]
    ]
    document["speakerTimeline"] = {
        "provider": {"id": "fixture-timeline", "version": "1"},
        "regular": {
            "sha256": "b" * 64,
            "turns": turns,
        },
    }
    lattice = build_semantic_candidate_lattice_from_document(document)
    baseline = _baseline(document, lattice)
    case["lattice"] = lattice
    case["baseline"] = baseline
    case["semanticCalibrationTarget"] = _semantic_calibration_target(
        lattice,
        baseline,
    )
    case_set = load_frozen_semantic_cases(_write_manifest(tmp_path, manifest))
    target = case_set.cases[0].semantic_calibration_target
    select_groups = [
        group
        for group in target["groups"]
        if group["expectedAction"] == "select"
    ]
    assert len(select_groups) == 1
    assert len(select_groups[0]["acceptableCandidateIds"]) == 2

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-equivalent:9b", "9" * 64)],
        provider_factory=lambda spec: _EquivalentTimelineSelectingProvider(
            spec.model,
            [],
            [],
        ),
        max_batch_attempts=1,
    )

    calibration = report["models"][0]["cases"][0]["calibration"]
    assert calibration["byExpectedAction"]["select"] == {
        "count": 1,
        "correctCount": 1,
        "accuracy": 1.0,
    }
    assert calibration["microAccuracy"] == 1.0
    assert report["models"][0]["aggregate"]["calibration"][
        "microAccuracy"
    ] == 1.0


def test_calibration_scores_default_challenger_without_leaking_hidden_target(
    tmp_path: Path,
) -> None:
    manifest = _manifest_value()
    case = manifest["cases"][0]
    baseline_secret = "CALIBRATION_BASELINE_SECRET_NEVER_SENT_TO_PROVIDER"
    case["baseline"]["segments"][0]["finalText"] = baseline_secret
    case["semanticCalibrationTarget"] = _semantic_calibration_target(
        case["lattice"],
        case["baseline"],
    )
    case_set = load_frozen_semantic_cases(_write_manifest(tmp_path, manifest))
    target = case_set.cases[0].semantic_calibration_target
    assert target["counts"]["request-default-challenger"] == 1
    captured_prompts: list[str] = []

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-request-target:9b", "a" * 64)],
        provider_factory=lambda spec: _ScopedRequestProvider(
            spec.model,
            [],
            [],
            request_domain="asr-text",
            request_scope_id="segment:segment-1",
            captured_prompts=captured_prompts,
        ),
        max_batch_attempts=1,
    )

    assert captured_prompts
    provider_serialized = "\n".join(captured_prompts)
    calibration = report["models"][0]["cases"][0]["calibration"]
    assert calibration["byExpectedAction"]["request-default-challenger"] == {
        "count": 1,
        "correctCount": 1,
        "accuracy": 1.0,
    }
    assert calibration["microAccuracy"] == 1.0
    report_serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for serialized in (provider_serialized, report_serialized):
        assert baseline_secret not in serialized
        for forbidden in (
            "semanticCalibrationTarget",
            "baselineProjectionSha256",
            "targetCandidateId",
            "acceptableCandidateIds",
        ):
            assert forbidden not in serialized


def test_calibration_rejects_non_default_challenger_request() -> None:
    manifest = _manifest_value()
    case = manifest["cases"][0]
    case["baseline"]["segments"][0]["finalText"] = "UNREACHABLE_BASELINE"
    target = _semantic_calibration_target(case["lattice"], case["baseline"])
    request_group = next(
        group
        for group in target["groups"]
        if group["expectedAction"] == "request-default-challenger"
    )
    scored = benchmark_module._score_semantic_calibration(
        {
            "selections": [],
            "candidateGenerationRequests": [
                {
                    "groupId": request_group["groupId"],
                    "requestKind": "local-asr-redecode",
                }
            ],
        },
        target,
    )
    row = scored["groups"][target["groups"].index(request_group)]
    assert row["actualAction"] == "request-other-challenger"
    assert row["correct"] is False


def test_benchmark_records_cold_warm_latency_ram_vram_digest_and_release(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value(case_count=2))
    )
    events: list[str] = []
    spec = SemanticModelSpec("semantic-resource:27b", "5" * 64)

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [spec],
        provider_factory=lambda selected: _SelectingProvider(
            selected.model,
            events,
            [],
        ),
        resource_probe=_fixture_resource_probe,
        max_batch_attempts=1,
    )

    model = report["models"][0]
    assert [
        case["execution"]["latencyClass"] for case in model["cases"]
    ] == ["cold-start-case", "warm-case"]
    calls = [
        call
        for case in model["cases"]
        for call in case["execution"]["providerCallEvidence"]
    ]
    assert [call["latencyClass"] for call in calls] == ["cold", "warm"]
    assert all(call["wallTimeSeconds"] >= 0.0 for call in calls)
    assert calls[0]["nativeMetrics"]["totalDurationNanoseconds"] == 1_000_000_000
    assert calls[0]["nativeMetrics"]["loadDurationNanoseconds"] == 500_000_000
    assert calls[1]["nativeMetrics"]["totalDurationNanoseconds"] == 1_000_000_000
    assert calls[1]["nativeMetrics"]["loadDurationNanoseconds"] == 0
    assert model["aggregate"]["providerCallLatency"][
        "coldStartVerified"
    ] is True
    assert model["aggregate"]["providerCallLatency"][
        "warmCallsObserved"
    ] is True
    assert model["aggregate"]["providerCallLatency"]["byLatencyClass"][
        "cold"
    ]["providerLoadDuration"]["p50Seconds"] == 0.5
    assert model["aggregate"]["caseLatency"]["byLatencyClass"][
        "warm-case"
    ]["count"] == 1

    resources = model["execution"]["resources"]
    assert resources["peakSelectedModelBytes"] == 1_000
    assert resources["peakSelectedModelVramBytes"] == 600
    assert resources["peakOllamaAggregateRssBytes"] == 800
    assert resources["peakBenchmarkProcessRssBytes"] == 100
    assert resources["peakOllamaProcessVramBytes"] == 500
    assert resources["minimumAvailablePhysicalMemoryBytes"] == 20_000
    assert resources["runtimeDigestMatchesExpected"] is True
    assert resources["modelAbsentAfterRelease"] is True
    assert model["provider"]["actualDigest"] == spec.digest
    assert model["provider"]["actualDigestSource"] == "custom-resource-probe"
    assert model["provider"]["digestVerified"] is False
    assert model["execution"]["release"]["verifiedModelAbsent"] is True
    assert model["execution"]["release"]["transitionVerified"] is True
    assert model["execution"]["release"][
        "releaseTransitionVerified"
    ] is True
    assert model["validation"]["resourceReleaseVerifiedAbsent"] is True
    assert report["evaluationPolicy"][
        "automaticLatencyAndResourceMetricsAreDiagnosticOnly"
    ] is True
    assert report["validation"]["promotionAuthorized"] is False


def test_invalid_response_retry_is_recorded_as_recovered_without_payload(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-recovery:27b", "6" * 64)],
        provider_factory=lambda spec: _SelectingProvider(
            spec.model,
            [],
            [],
            invalid_once=True,
        ),
        resource_probe=_fixture_resource_probe,
        max_batch_attempts=2,
    )

    model = report["models"][0]
    case = model["cases"][0]
    assert case["execution"]["status"] == "composition-complete"
    recovery = case["execution"]["failureRecovery"]
    assert recovery["recoveryAttempted"] is True
    assert recovery["validationRetryCallCount"] == 1
    assert recovery["providerErrorCallCount"] == 0
    assert recovery["recovered"] is True
    assert recovery["exhausted"] is False
    assert recovery["maxBatchAttempts"] == 2
    calls = case["execution"]["providerCallEvidence"]
    assert len(calls) == 2
    assert calls[0]["latencyClass"] == "cold"
    assert calls[0]["previousResponseRejected"] is False
    assert calls[1]["latencyClass"] == "warm"
    assert calls[1]["previousResponseRejected"] is True
    assert calls[1]["retryAttempt"] == 2
    assert calls[1]["validationFailureCode"] is not None
    assert model["aggregate"]["failureRecovery"] == {
        "caseCount": 1,
        "recoveryAttemptedCaseCount": 1,
        "recoveredCaseCount": 1,
        "exhaustedCaseCount": 0,
        "validationRetryCallCount": 1,
        "providerErrorCallCount": 0,
        "validationFailureCodeCounts": {
            calls[1]["validationFailureCode"]: 1
        },
    }
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert "SECRET_TRANSCRIPT_ALPHA" not in serialized
    assert '"choiceByPosition"' not in serialized


def test_runtime_digest_mismatch_is_visible_and_not_masked_by_provider_flag(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )

    def mismatched_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        snapshot = _fixture_resource_probe(spec, phase)
        for selected in snapshot["selectedModel"]:
            selected["digest"] = "sha256:" + "7" * 64
        return snapshot

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-digest:27b", "8" * 64)],
        provider_factory=lambda spec: _SelectingProvider(spec.model, [], []),
        resource_probe=mismatched_probe,
        max_batch_attempts=1,
    )

    provider = report["models"][0]["provider"]
    assert provider["expectedDigest"] == "sha256:" + "8" * 64
    assert provider["actualDigest"] == "sha256:" + "7" * 64
    assert provider["actualDigestSource"] == "custom-resource-probe"
    assert provider["runtimeDigestObserved"] is True
    assert provider["runtimeDigestMatchesExpected"] is False
    assert provider["digestVerified"] is False
    assert report["validation"]["passed"] is True
    assert report["validation"]["passedMeaning"] == (
        "artifact-safety-and-no-production-mutation-only"
    )
    assert report["validation"]["promotionAuthorized"] is False


def test_provider_failure_is_recorded_and_resources_are_released(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    events: list[str] = []

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-failure:9b", "3" * 64)],
        provider_factory=lambda spec: _SelectingProvider(
            spec.model,
            events,
            [],
            fail=True,
        ),
        max_batch_attempts=1,
    )

    model = report["models"][0]
    assert model["cases"][0]["execution"]["failureCodes"] == ["ValueError"]
    assert model["aggregate"]["failureCodeCounts"] == {"ValueError": 1}
    recovery = model["cases"][0]["execution"]["failureRecovery"]
    assert recovery["failureObserved"] is True
    assert recovery["retryAttempted"] is False
    assert recovery["recoveryAttempted"] is False
    assert recovery["exhausted"] is False
    assert recovery["terminalFailureWithoutRetry"] is True
    assert model["cases"][0]["execution"]["providerCallEvidence"][0][
        "latencyClass"
    ] == "failed-before-classification"
    assert model["aggregate"]["providerCallLatency"][
        "coldStartVerified"
    ] is False
    assert model["aggregate"]["failureRecovery"][
        "providerErrorCallCount"
    ] == 1
    assert model["execution"]["release"]["succeeded"] is True
    assert events == [
        "generate:semantic-failure:9b",
        "release:semantic-failure:9b",
    ]


def test_failed_first_call_is_not_cold_and_later_absent_call_is_reload(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value(case_count=2))
    )
    probe_phases: list[str] = []

    def reload_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        probe_phases.append(phase)
        if phase.startswith("before-provider-call:"):
            return _fixture_resource_probe(spec, "before-provider-call:1")
        return _fixture_resource_probe(spec, phase)

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-reload:9b", "9" * 64)],
        provider_factory=lambda spec: _SelectingProvider(
            spec.model,
            [],
            [],
            fail_once=True,
        ),
        resource_probe=reload_probe,
        max_batch_attempts=1,
    )

    model = report["models"][0]
    calls = [
        call
        for case in model["cases"]
        for call in case["execution"]["providerCallEvidence"]
    ]
    assert [call["latencyClass"] for call in calls] == [
        "failed-before-classification",
        "reload",
    ]
    assert [
        case["execution"]["latencyClass"] for case in model["cases"]
    ] == ["failed-provider-call-case", "reload-case"]
    latency = model["aggregate"]["providerCallLatency"]
    assert latency["coldStartVerified"] is False
    assert latency["reloadCallsObserved"] is True
    assert latency["failedCallsExcludedFromColdWarm"] == 1
    assert "before-provider-call:1" in probe_phases
    assert "before-provider-call:2" in probe_phases


def test_unknown_immediate_pre_call_probe_cannot_inherit_cold_from_pre_run(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )

    def unknown_call_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        snapshot = _fixture_resource_probe(spec, phase)
        if phase.startswith("before-provider-call:"):
            snapshot["ollamaApiAvailable"] = False
            snapshot["selectedModelLoaded"] = None
            snapshot["selectedModel"] = []
        return snapshot

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-unknown-call:9b", "8" * 64)],
        provider_factory=lambda spec: _SelectingProvider(spec.model, [], []),
        resource_probe=unknown_call_probe,
        max_batch_attempts=1,
    )

    model = report["models"][0]
    call = model["cases"][0]["execution"]["providerCallEvidence"][0]
    assert call["preCallSelectedModelLoaded"] is None
    assert call["latencyClass"] == "cold-unverified"
    assert model["aggregate"]["providerCallLatency"][
        "coldStartVerified"
    ] is False
    assert model["cases"][0]["execution"]["latencyClass"] == (
        "latency-unverified-case"
    )


@pytest.mark.parametrize(
    ("loaded", "expected_class"),
    ((False, "cold-unverified"), (True, "warm-unverified")),
)
def test_api_unavailable_pre_call_probe_never_proves_cold_or_warm(
    tmp_path: Path,
    loaded: bool,
    expected_class: str,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )

    def unavailable_call_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        snapshot = _fixture_resource_probe(spec, phase)
        if phase == "before-provider-call:1":
            snapshot["ollamaApiAvailable"] = False
            snapshot["selectedModelLoaded"] = loaded
        return snapshot

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-api-unavailable:9b", "3" * 64)],
        provider_factory=lambda spec: _SelectingProvider(spec.model, [], []),
        resource_probe=unavailable_call_probe,
        max_batch_attempts=1,
    )

    model = report["models"][0]
    call = model["cases"][0]["execution"]["providerCallEvidence"][0]
    assert call["preCallOllamaApiAvailable"] is False
    assert call["preCallResidencyVerified"] is False
    assert call["preCallSelectedModelLoaded"] is loaded
    assert call["latencyClass"] == expected_class
    assert model["aggregate"]["providerCallLatency"][
        "coldStartVerified"
    ] is False


def test_terminal_validation_failure_preserves_nested_failure_codes(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-invalid:9b", "a" * 64)],
        provider_factory=lambda spec: _SelectingProvider(
            spec.model,
            [],
            [],
            invalid_once=True,
        ),
        resource_probe=_fixture_resource_probe,
        max_batch_attempts=1,
    )

    execution = report["models"][0]["cases"][0]["execution"]
    assert {
        "SEMANTIC_JOB_PROVIDER_FAILED",
        "SEMANTIC_RESPONSE_VALIDATION",
        "STRICT_JSON_OR_SCHEMA_INVALID",
    } <= set(execution["failureCodes"])
    recovery = execution["failureRecovery"]
    assert recovery["failureObserved"] is True
    assert recovery["retryAttempted"] is False
    assert recovery["exhausted"] is False
    assert recovery["terminalFailureWithoutRetry"] is True


def test_periodic_sampler_captures_transient_resource_maximum(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    periodic_seen = threading.Event()
    phases: list[str] = []

    def transient_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        phases.append(phase)
        snapshot = _fixture_resource_probe(spec, phase)
        if phase.startswith("periodic:"):
            snapshot["process"]["ollamaAggregateRssBytes"] = 9_000
            snapshot["process"]["gpuProcessMemory"][
                "ollamaAggregateVramBytes"
            ] = 7_000
            periodic_seen.set()
        return snapshot

    def waiting_provider(spec: SemanticModelSpec) -> _SelectingProvider:
        provider = _SelectingProvider(spec.model, [], [])
        generate = provider.generate_json

        def generate_after_sample(**kwargs: Any) -> dict[str, Any]:
            assert periodic_seen.wait(timeout=2.0)
            return generate(**kwargs)

        provider.generate_json = generate_after_sample  # type: ignore[method-assign]
        return provider

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-transient:9b", "b" * 64)],
        provider_factory=waiting_provider,
        resource_probe=transient_probe,
        resource_sample_interval_seconds=0.001,
        max_batch_attempts=1,
    )

    resources = report["models"][0]["execution"]["resources"]
    assert resources["periodicSnapshotCount"] >= 1
    assert resources["periodicSamplerStopped"] is True
    assert resources["sampleIntervalSeconds"] == 0.001
    assert resources["peakOllamaAggregateRssBytes"] == 9_000
    assert resources["peakOllamaProcessVramBytes"] == 7_000
    assert resources["snapshots"][-1]["phase"] == "after-release"
    assert max(
        index for index, phase in enumerate(phases) if phase.startswith("periodic:")
    ) < phases.index("before-release")


def test_loaded_model_with_zero_matched_pid_invalidates_process_ram_vram(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )

    def zero_pid_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        snapshot = _fixture_resource_probe(spec, phase)
        if snapshot["selectedModelLoaded"] is True:
            snapshot["process"]["ollamaProcessCount"] = 0
            snapshot["process"]["ollamaPids"] = []
            snapshot["process"][
                "ollamaAggregatePeakWorkingSetBytes"
            ] = 900
            snapshot["process"]["gpuProcessMemory"]["matchedProcessCount"] = 0
        return snapshot

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-zero-pid:9b", "c" * 64)],
        provider_factory=lambda spec: _SelectingProvider(spec.model, [], []),
        resource_probe=zero_pid_probe,
        max_batch_attempts=1,
    )

    resources = report["models"][0]["execution"]["resources"]
    assert resources["loadedModelWithoutMatchedProcess"] is True
    assert resources["peakOllamaAggregateRssBytes"] is None
    assert resources["maximumObservedOllamaAggregateRssBytes"] is None
    assert resources["peakOllamaAggregateWorkingSetBytes"] is None
    assert resources["peakOllamaProcessVramBytes"] is None
    assert resources["maximumObservedOllamaProcessVramBytes"] is None
    assert resources["processRamEvidenceComplete"] is False
    assert resources["processVramEvidenceComplete"] is False


@pytest.mark.parametrize(
    ("release_mode", "expected_stop_reason"),
    (
        ("missing", "RESOURCE_RELEASE_METHOD_MISSING"),
        ("raises", "RuntimeError"),
        ("resident", "MODEL_STILL_RESIDENT_AFTER_RELEASE"),
        ("unknown", "MODEL_RESIDENCY_UNKNOWN_AFTER_RELEASE"),
    ),
)
def test_default_provider_isolation_failure_stops_before_next_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_mode: str,
    expected_stop_reason: str,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    factory_calls: list[str] = []

    def fake_default_factory(
        spec: SemanticModelSpec,
        **_: Any,
    ) -> _SelectingProvider:
        factory_calls.append(spec.model)
        provider = _SelectingProvider(spec.model, [], [])
        if release_mode == "missing":
            provider.release_resources = None  # type: ignore[method-assign]
        elif release_mode == "raises":
            def fail_release() -> None:
                raise RuntimeError("fixture release failure")

            provider.release_resources = fail_release  # type: ignore[method-assign]
        return provider

    def isolation_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        if phase == "after-release" and release_mode == "resident":
            return _fixture_resource_probe(spec, "before-release")
        if phase == "after-release" and release_mode == "unknown":
            snapshot = _fixture_resource_probe(spec, phase)
            snapshot["ollamaApiAvailable"] = False
            snapshot["selectedModelLoaded"] = None
            return snapshot
        return _fixture_resource_probe(spec, phase)

    monkeypatch.setattr(
        benchmark_module,
        "_default_provider_factory",
        fake_default_factory,
    )
    specs = [
        SemanticModelSpec("semantic-a:9b", "d" * 64),
        SemanticModelSpec("semantic-b:9b", "e" * 64),
    ]
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        specs,
        resource_probe=isolation_probe,
        max_batch_attempts=1,
    )

    assert factory_calls == ["semantic-a:9b"]
    assert report["execution"]["requestedModelCount"] == 2
    assert report["execution"]["executedModelCount"] == 1
    assert report["execution"]["skippedModelCount"] == 1
    assert report["execution"]["skippedModelOrder"] == ["semantic-b:9b"]
    assert report["execution"]["stopReason"] == expected_stop_reason
    assert report["validation"]["allModelsIndependent"] is False
    isolation = report["models"][0]["execution"]["isolation"]
    assert isolation["verified"] is False
    assert isolation["continuationAllowed"] is False


def test_initialization_failure_without_residency_does_not_claim_transition(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )

    def fail_factory(spec: SemanticModelSpec) -> _SelectingProvider:
        del spec
        raise RuntimeError("fixture initialization failure")

    def absent_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        return _fixture_resource_probe(spec, "after-release")

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-init-fail:9b", "f" * 64)],
        provider_factory=fail_factory,
        resource_probe=absent_probe,
        max_batch_attempts=1,
    )

    model = report["models"][0]
    release = model["execution"]["release"]
    assert release["attempted"] is False
    assert release["postReleaseModelAbsent"] is True
    assert release["transitionVerified"] is False
    assert release["releaseTransitionVerified"] is False
    assert model["validation"]["modelIsolationVerified"] is True
    calibration = model["aggregate"]["calibration"]
    assert calibration["groupCount"] == 8
    assert calibration["correctCount"] == 0
    assert calibration["microAccuracy"] == 0.0
    assert calibration["actualActionCounts"] == {"unavailable": 8}


def test_only_default_provider_and_probe_claim_ollama_digest_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )

    def fake_default_factory(
        spec: SemanticModelSpec,
        **_: Any,
    ) -> _SelectingProvider:
        return _SelectingProvider(spec.model, [], [])

    def fake_default_probe(
        endpoint: str,
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        assert endpoint == "http://127.0.0.1:11434"
        return _fixture_resource_probe(spec, phase)

    monkeypatch.setattr(
        benchmark_module,
        "_default_provider_factory",
        fake_default_factory,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_default_resource_probe",
        fake_default_probe,
    )
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-default:9b", "1" * 64)],
        max_batch_attempts=1,
    )

    provider = report["models"][0]["provider"]
    assert provider["actualDigestSource"] == "ollama-api-ps"
    assert provider["runtimeDigestMatchesExpected"] is True
    assert provider["digestVerified"] is True
    configuration = report["executionConfiguration"]
    assert configuration["providerMode"] == "default-ollama"
    assert configuration["endpointApplied"] is True
    assert configuration["topPAppliedByDefaultProvider"] is True
    assert configuration["keepAliveAppliedByDefaultProvider"] is True


def test_default_provider_self_report_cannot_override_custom_digest_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )

    def fake_default_factory(
        spec: SemanticModelSpec,
        **_: Any,
    ) -> _SelectingProvider:
        return _SelectingProvider(spec.model, [], [])

    def custom_absent_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        snapshot = _fixture_resource_probe(spec, "after-release")
        snapshot["phase"] = phase
        return snapshot

    monkeypatch.setattr(
        benchmark_module,
        "_default_provider_factory",
        fake_default_factory,
    )
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-custom-digest:9b", "4" * 64)],
        resource_probe=custom_absent_probe,
        max_batch_attempts=1,
    )

    provider = report["models"][0]["provider"]
    assert provider["runtimeDigestObserved"] is False
    assert provider["actualDigest"] is None
    assert provider["actualDigestSource"] is None
    assert provider["digestVerified"] is False
    assert provider["digestEvidenceAuthoritative"] is False


def test_custom_probe_residency_failure_stops_next_model(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    factory_calls: list[str] = []

    def resident_after_release_probe(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        if phase == "after-release":
            return _fixture_resource_probe(spec, "before-release")
        return _fixture_resource_probe(spec, phase)

    def factory(spec: SemanticModelSpec) -> _SelectingProvider:
        factory_calls.append(spec.model)
        return _SelectingProvider(spec.model, [], [])

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [
            SemanticModelSpec("semantic-custom-a:9b", "5" * 64),
            SemanticModelSpec("semantic-custom-b:9b", "6" * 64),
        ],
        provider_factory=factory,
        resource_probe=resident_after_release_probe,
        max_batch_attempts=1,
    )

    assert factory_calls == ["semantic-custom-a:9b"]
    assert report["execution"]["skippedModelCount"] == 1
    assert report["execution"]["stopReason"] == (
        "MODEL_STILL_RESIDENT_AFTER_RELEASE"
    )


def test_custom_provider_release_failure_stops_without_resource_probe(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    factory_calls: list[str] = []

    def factory(spec: SemanticModelSpec) -> _SelectingProvider:
        factory_calls.append(spec.model)
        provider = _SelectingProvider(spec.model, [], [])

        def fail_release() -> None:
            raise RuntimeError("fixture custom release failure")

        provider.release_resources = fail_release  # type: ignore[method-assign]
        return provider

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [
            SemanticModelSpec("semantic-release-a:9b", "9" * 64),
            SemanticModelSpec("semantic-release-b:9b", "a" * 64),
        ],
        provider_factory=factory,
        max_batch_attempts=1,
    )

    assert factory_calls == ["semantic-release-a:9b"]
    assert report["execution"]["skippedModelCount"] == 1
    assert report["execution"]["stopReason"] == "RuntimeError"
    assert report["models"][0]["execution"]["isolation"][
        "continuationAllowed"
    ] is False


def test_sampler_that_does_not_stop_blocks_any_following_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    factory_calls: list[str] = []

    class NeverStoppedSampler:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> tuple[list[dict[str, Any]], bool]:
            return [], False

    monkeypatch.setattr(benchmark_module, "_PeriodicResourceSampler", NeverStoppedSampler)

    def factory(spec: SemanticModelSpec) -> _SelectingProvider:
        factory_calls.append(spec.model)
        return _SelectingProvider(spec.model, [], [])

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [
            SemanticModelSpec("semantic-sampler-a:9b", "7" * 64),
            SemanticModelSpec("semantic-sampler-b:9b", "8" * 64),
        ],
        provider_factory=factory,
        resource_probe=_fixture_resource_probe,
        max_batch_attempts=1,
    )

    assert factory_calls == ["semantic-sampler-a:9b"]
    assert report["execution"]["skippedModelCount"] == 1
    assert report["execution"]["stopReason"] == "RESOURCE_SAMPLER_DID_NOT_STOP"


@pytest.mark.parametrize("interval", (0.0, -0.1, float("inf"), float("nan")))
def test_resource_sample_interval_must_be_finite_and_positive(
    tmp_path: Path,
    interval: float,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    with pytest.raises(
        RealMultilingualSemanticBenchmarkError,
        match="resource_sample_interval_seconds",
    ):
        benchmark_real_multilingual_semantic_models(
            case_set,
            [SemanticModelSpec("semantic-interval:9b", "2" * 64)],
            provider_factory=lambda spec: _SelectingProvider(
                spec.model,
                [],
                [],
            ),
            resource_sample_interval_seconds=interval,
            max_batch_attempts=1,
        )


def test_model_sidecars_are_immutable_no_replace(tmp_path: Path) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("semantic-a:9b", "4" * 64)],
        provider_factory=lambda spec: _SelectingProvider(spec.model, [], []),
        max_batch_attempts=1,
    )
    output = tmp_path / "model-results"

    published = publish_model_reports_no_replace(report, output)
    path = Path(published["semantic-a:9b"])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_model_reports_no_replace(report, output)
    assert path.read_bytes() == original


def _canonical_json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = copy.deepcopy(value)
    declared = body.pop("canonicalSha256")
    assert declared == canonical_json_sha256(body)
    return value


def _package_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
    }


def test_blind_review_package_is_reproducible_complete_and_identity_separated(
    tmp_path: Path,
) -> None:
    manifest = _manifest_value(case_count=2)
    for index, case in enumerate(manifest["cases"]):
        case["baseline"]["segments"][0]["finalText"] = (
            f"BASELINE_REFERENCE_ONLY_{index}"
        )
        case["semanticCalibrationTarget"] = _semantic_calibration_target(
            case["lattice"],
            case["baseline"],
        )
    case_set = load_frozen_semantic_cases(_write_manifest(tmp_path, manifest))
    specs = (
        SemanticModelSpec(
            "private-complete:9b",
            "1" * 64,
            model_id="private-complete-id",
            config_path="D:/private/complete.json",
        ),
        SemanticModelSpec(
            "private-request:14b",
            "2" * 64,
            model_id="private-request-id",
            config_path="D:/private/request.json",
        ),
        SemanticModelSpec(
            "private-failed:27b",
            "3" * 64,
            model_id="private-failed-id",
            config_path="D:/private/failed.json",
        ),
    )

    def factory(spec: SemanticModelSpec) -> _SelectingProvider:
        if spec.model == "private-request:14b":
            return _RequestingProvider(spec.model, [], [])
        return _SelectingProvider(
            spec.model,
            [],
            [],
            fail=spec.model == "private-failed:27b",
        )

    seed = "f94603a623e94b30899fdd6dc8d9894b"
    first_root = tmp_path / "blind-first"
    first_report = benchmark_real_multilingual_semantic_models(
        case_set,
        specs,
        provider_factory=factory,
        blind_review_output_root=first_root,
        blind_seed=seed,
        max_batch_attempts=1,
        now=lambda: "2026-08-09T02:00:00Z",
    )
    second_root = tmp_path / "blind-second"
    benchmark_real_multilingual_semantic_models(
        tuple(reversed(case_set.cases)),
        tuple(reversed(specs)),
        provider_factory=factory,
        blind_review_output_root=second_root,
        blind_seed=seed,
        max_batch_attempts=1,
        now=lambda: "2026-08-09T03:00:00Z",
    )

    assert _package_bytes(first_root) == _package_bytes(second_root)
    package_manifest = _canonical_json_file(first_root / "manifest.json")
    vault = _canonical_json_file(first_root / "identity-vault.json")
    assert vault["seedSha256"] == hashlib.sha256(seed.encode("utf-8")).hexdigest()
    assert seed not in json.dumps(vault, ensure_ascii=False)
    assert len(vault["cases"]) == 2

    evidence_paths = {
        row["relativePath"] for row in package_manifest["files"]
    }
    actual_child_paths = {
        path.relative_to(first_root).as_posix()
        for path in first_root.rglob("*.json")
        if path.name != "manifest.json"
    }
    assert evidence_paths == actual_child_paths
    for evidence in package_manifest["files"]:
        path = first_root / evidence["relativePath"]
        value = _canonical_json_file(path)
        assert evidence["canonicalSha256"] == value["canonicalSha256"]
        assert evidence["fileSha256"] == sha256_file(path)
        assert evidence["sizeBytes"] == path.stat().st_size
    receipt = first_report["blindReviewPublication"]
    assert receipt["manifestCanonicalSha256"] == package_manifest[
        "canonicalSha256"
    ]
    assert receipt["manifestFileSha256"] == sha256_file(
        first_root / "manifest.json"
    )

    vault_by_alias = {row["caseAlias"]: row for row in vault["cases"]}
    observed_statuses: set[str] = set()
    reviewer_serialized = ""
    for packet_path in sorted((first_root / "reviewer").glob("*.json")):
        packet = _canonical_json_file(packet_path)
        reviewer_serialized += json.dumps(packet, ensure_ascii=False, sort_keys=True)
        mapped = vault_by_alias[packet["caseAlias"]]
        packet_candidates = {
            row["candidateAlias"]: row for row in packet["candidates"]
        }
        vault_candidates = {
            row["candidateAlias"]: row for row in mapped["candidates"]
        }
        assert set(packet_candidates) == {
            "candidate-01",
            "candidate-02",
            "candidate-03",
        }
        assert set(packet_candidates) == set(vault_candidates)
        assert {
            row["model"] for row in vault_candidates.values()
        } == {spec.model for spec in specs}
        for alias, candidate in packet_candidates.items():
            result = candidate["result"]
            observed_statuses.add(result["status"])
            assert candidate["resultCanonicalSha256"] == canonical_json_sha256(
                result
            )
            assert candidate["resultCanonicalSha256"] == vault_candidates[alias][
                "resultCanonicalSha256"
            ]
            if result["status"] == "failed":
                assert result["failureCodes"] == [
                    "SEMANTIC_RESULT_UNAVAILABLE"
                ]
    assert observed_statuses == {
        "composition-complete",
        "candidate-generation-required",
        "failed",
    }
    assert not (first_root / "reviewer" / "identity-vault.json").exists()
    for spec in specs:
        for secret in (
            spec.model,
            spec.model_id,
            spec.digest,
            spec.config_path,
        ):
            assert secret not in reviewer_serialized
    assert seed not in reviewer_serialized
    assert hashlib.sha256(seed.encode("utf-8")).hexdigest() not in reviewer_serialized
    assert "BASELINE_REFERENCE_ONLY_0" not in reviewer_serialized
    assert "BASELINE_REFERENCE_ONLY_1" not in reviewer_serialized
    assert "SECRET_TRANSCRIPT_ALPHA" in reviewer_serialized

    package_serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(first_root.rglob("*.json"))
    )
    for forbidden in (
        "semanticCalibrationTarget",
        "baselineProjectionSha256",
        "targetCandidateId",
        "acceptableCandidateIds",
        "BASELINE_REFERENCE_ONLY_0",
        "BASELINE_REFERENCE_ONLY_1",
    ):
        assert forbidden not in package_serialized

    report_serialized = json.dumps(first_report, ensure_ascii=False, sort_keys=True)
    assert "SECRET_TRANSCRIPT_ALPHA" not in report_serialized
    assert "BASELINE_REFERENCE_ONLY_0" not in report_serialized
    assert first_report["evaluationPolicy"][
        "challengerMayReplaceProductionImmediatelyAfterBlindWin"
    ] is True
    assert first_report["evaluationPolicy"]["productionPointerPolicy"] == (
        "active-pointer-external-cas"
    )
    assert first_report["evaluationPolicy"]["promotionMarginRequired"] is False
    assert first_report["evaluationPolicy"]["promotionCooldownRequired"] is False
    assert first_report["evaluationPolicy"]["registryStatusGateRequired"] is False


def test_blind_review_arguments_are_paired_before_provider_start(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    factory_calls: list[str] = []

    def factory(spec: SemanticModelSpec) -> _SelectingProvider:
        factory_calls.append(spec.model)
        return _SelectingProvider(spec.model, [], [])

    with pytest.raises(
        RealMultilingualSemanticBenchmarkError,
        match="must be provided together",
    ):
        benchmark_real_multilingual_semantic_models(
            case_set,
            [SemanticModelSpec("pair-fixture:9b", "4" * 64)],
            provider_factory=factory,
            blind_review_output_root=tmp_path / "blind",
        )
    assert factory_calls == []


@pytest.mark.parametrize("target_kind", ("file", "empty-directory", "directory"))
def test_existing_blind_target_refuses_before_provider_or_probe(
    tmp_path: Path,
    target_kind: str,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    target = tmp_path / f"existing-{target_kind}"
    if target_kind == "file":
        target.write_text("owned", encoding="utf-8")
    else:
        target.mkdir()
        if target_kind == "directory":
            (target / "owned.txt").write_text("owned", encoding="utf-8")
    original = (
        target.read_bytes()
        if target.is_file()
        else (target / "owned.txt").read_bytes()
        if (target / "owned.txt").exists()
        else None
    )
    factory_calls: list[str] = []
    probe_calls: list[str] = []

    def factory(spec: SemanticModelSpec) -> _SelectingProvider:
        factory_calls.append(spec.model)
        return _SelectingProvider(spec.model, [], [])

    def probe(spec: SemanticModelSpec, phase: str) -> dict[str, Any]:
        probe_calls.append(phase)
        return _fixture_resource_probe(spec, phase)

    with pytest.raises(FileExistsError):
        benchmark_real_multilingual_semantic_models(
            case_set,
            [SemanticModelSpec("existing-fixture:9b", "5" * 64)],
            provider_factory=factory,
            resource_probe=probe,
            blind_review_output_root=target,
            blind_seed="existing-target-seed",
            max_batch_attempts=1,
        )
    assert factory_calls == []
    assert probe_calls == []
    if original is not None:
        assert (
            target.read_bytes()
            if target.is_file()
            else (target / "owned.txt").read_bytes()
        ) == original


def test_existing_broken_symlink_blind_target_refuses_before_provider(
    tmp_path: Path,
) -> None:
    target = tmp_path / "broken-target"
    try:
        target.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError:
        pytest.skip("the test filesystem does not permit symlinks")
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    factory_calls: list[str] = []

    with pytest.raises(FileExistsError):
        benchmark_real_multilingual_semantic_models(
            case_set,
            [SemanticModelSpec("symlink-fixture:9b", "6" * 64)],
            provider_factory=lambda spec: (
                factory_calls.append(spec.model)
                or _SelectingProvider(spec.model, [], [])
            ),
            blind_review_output_root=target,
            blind_seed="broken-symlink-seed",
            max_batch_attempts=1,
        )
    assert factory_calls == []
    assert target.is_symlink()


def test_wsl_publish_preflight_fails_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    target = tmp_path / "wsl-target"
    factory_calls: list[str] = []

    monkeypatch.setattr(benchmark_module, "_running_on_wsl_mount", lambda _: True)

    def unsupported_rename(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(
        benchmark_module,
        "_rename_directory_no_replace",
        unsupported_rename,
    )
    with pytest.raises(OSError, match="Invalid argument"):
        benchmark_real_multilingual_semantic_models(
            case_set,
            [SemanticModelSpec("wsl-preflight:9b", "9" * 64)],
            provider_factory=lambda spec: (
                factory_calls.append(spec.model)
                or _SelectingProvider(spec.model, [], [])
            ),
            blind_review_output_root=target,
            blind_seed="wsl-preflight-seed",
            max_batch_attempts=1,
        )
    assert factory_calls == []
    assert not target.exists()
    assert not list(tmp_path.glob(".wsl-target.publish-*"))


@pytest.mark.parametrize(
    "failed_name",
    ("case-001.json", "identity-vault.json", "manifest.json"),
)
def test_blind_staging_failure_leaves_no_target_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_name: str,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    target = tmp_path / f"failed-{failed_name.replace('.', '-')}"
    real_writer = benchmark_module.atomic_write_json_no_replace

    def failing_writer(path: Path, value: Any) -> None:
        if path.name == failed_name:
            raise OSError("injected blind package write failure")
        real_writer(path, value)

    monkeypatch.setattr(
        benchmark_module,
        "atomic_write_json_no_replace",
        failing_writer,
    )
    with pytest.raises(OSError, match="injected blind package write failure"):
        benchmark_real_multilingual_semantic_models(
            case_set,
            [SemanticModelSpec("write-failure-fixture:9b", "7" * 64)],
            provider_factory=lambda spec: _SelectingProvider(spec.model, [], []),
            blind_review_output_root=target,
            blind_seed="write-failure-seed",
            max_batch_attempts=1,
        )
    assert not target.exists()
    assert list(tmp_path.glob(f".{target.name}.staging-*")) == []


def test_blind_publish_race_preserves_concurrent_target_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    target = tmp_path / "race-target"
    real_rename = benchmark_module._rename_directory_no_replace

    def racing_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_text("concurrent", encoding="utf-8")
        real_rename(source, destination)

    monkeypatch.setattr(
        benchmark_module,
        "_rename_directory_no_replace",
        racing_rename,
    )
    with pytest.raises(FileExistsError):
        benchmark_real_multilingual_semantic_models(
            case_set,
            [SemanticModelSpec("race-fixture:9b", "8" * 64)],
            provider_factory=lambda spec: _SelectingProvider(spec.model, [], []),
            blind_review_output_root=target,
            blind_seed="race-seed",
            max_batch_attempts=1,
        )
    assert (target / "owner.txt").read_text(encoding="utf-8") == "concurrent"
    assert list(tmp_path.glob(f".{target.name}.staging-*")) == []


def test_blind_review_initialization_failure_fills_every_case(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value(case_count=2))
    )
    target = tmp_path / "initialization-failure"

    def fail_factory(spec: SemanticModelSpec) -> _SelectingProvider:
        del spec
        raise RuntimeError("private initialization fingerprint")

    report = benchmark_real_multilingual_semantic_models(
        case_set,
        [SemanticModelSpec("initialization-failure:9b", "a" * 64)],
        provider_factory=fail_factory,
        blind_review_output_root=target,
        blind_seed="initialization-failure-seed",
        max_batch_attempts=1,
    )

    assert report["models"][0]["execution"]["status"] == "failed-to-initialize"
    packets = [
        _canonical_json_file(path)
        for path in sorted((target / "reviewer").glob("*.json"))
    ]
    assert len(packets) == 2
    assert all(
        packet["candidates"][0]["result"]
        == {
            "status": "failed",
            "failureCodes": ["SEMANTIC_RESULT_UNAVAILABLE"],
        }
        for packet in packets
    )
    assert "private initialization fingerprint" not in json.dumps(
        packets,
        ensure_ascii=False,
    )


def test_blind_review_marks_models_skipped_after_isolation_failure(
    tmp_path: Path,
) -> None:
    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    specs = (
        SemanticModelSpec("executed-before-stop:9b", "b" * 64),
        SemanticModelSpec("skipped-after-stop:14b", "c" * 64),
    )
    factory_calls: list[str] = []

    def factory(spec: SemanticModelSpec) -> _SelectingProvider:
        factory_calls.append(spec.model)
        return _SelectingProvider(spec.model, [], [])

    def resident_after_release(
        spec: SemanticModelSpec,
        phase: str,
    ) -> dict[str, Any]:
        if phase == "after-release":
            return _fixture_resource_probe(spec, "before-release")
        return _fixture_resource_probe(spec, phase)

    target = tmp_path / "skipped-model"
    report = benchmark_real_multilingual_semantic_models(
        case_set,
        specs,
        provider_factory=factory,
        resource_probe=resident_after_release,
        blind_review_output_root=target,
        blind_seed="skipped-model-seed",
        max_batch_attempts=1,
    )

    assert factory_calls == ["executed-before-stop:9b"]
    assert report["execution"]["skippedModelOrder"] == [
        "skipped-after-stop:14b"
    ]
    packet = _canonical_json_file(target / "reviewer" / "case-001.json")
    vault = _canonical_json_file(target / "identity-vault.json")
    vault_candidates = {
        row["candidateAlias"]: row
        for row in vault["cases"][0]["candidates"]
    }
    results_by_model = {
        vault_candidates[candidate["candidateAlias"]]["model"]: candidate["result"]
        for candidate in packet["candidates"]
    }
    assert results_by_model["executed-before-stop:9b"]["status"] == (
        "composition-complete"
    )
    assert results_by_model["skipped-after-stop:14b"] == {
        "status": "not-executed",
        "failureCodes": ["MODEL_NOT_EXECUTED"],
    }


def test_wddm_process_vram_na_keeps_whole_gpu_diagnostic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: tuple[str, ...], **_: Any) -> subprocess.CompletedProcess[str]:
        output = (
            "123, N/A\n"
            if any("query-compute-apps" in value for value in command)
            else "0, 314159, 271828\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)
    process_vram = benchmark_module._nvidia_process_memory(frozenset({123}))
    whole_gpu = benchmark_module._nvidia_whole_gpu_memory()
    assert process_vram["available"] is False
    assert process_vram["ollamaAggregateVramBytes"] is None
    assert whole_gpu["available"] is True
    assert whole_gpu["aggregateUsedVramBytes"] == 314159 * 1024 * 1024
    assert whole_gpu["aggregateFreeVramBytes"] == 271828 * 1024 * 1024
    assert whole_gpu["includesOtherProcesses"] is True
    assert whole_gpu["diagnosticOnly"] is True

    case_set = load_frozen_semantic_cases(
        _write_manifest(tmp_path, _manifest_value())
    )
    spec = SemanticModelSpec("wddm-fixture:9b", "9" * 64)

    def run_with_whole_gpu(value: int, root: Path) -> dict[str, Any]:
        def probe(selected: SemanticModelSpec, phase: str) -> dict[str, Any]:
            snapshot = _fixture_resource_probe(selected, phase)
            snapshot["process"]["gpuProcessMemory"] = copy.deepcopy(process_vram)
            snapshot["process"]["wholeGpuMemory"] = {
                "available": True,
                "measurement": "nvidia-smi-whole-gpu-memory-v1",
                "aggregateUsedVramBytes": value,
                "aggregateFreeVramBytes": value + 1,
                "includesOtherProcesses": True,
                "diagnosticOnly": True,
                "devices": [],
            }
            return snapshot

        return benchmark_real_multilingual_semantic_models(
            case_set,
            [spec],
            provider_factory=lambda selected: _SelectingProvider(
                selected.model,
                [],
                [],
            ),
            resource_probe=probe,
            blind_review_output_root=root,
            blind_seed="wddm-blind-seed",
            max_batch_attempts=1,
        )

    first_root = tmp_path / "wddm-first"
    second_root = tmp_path / "wddm-second"
    first = run_with_whole_gpu(123_456_789, first_root)
    second = run_with_whole_gpu(987_654_321, second_root)
    resources = first["models"][0]["execution"]["resources"]
    assert resources["peakOllamaProcessVramBytes"] is None
    assert resources["processVramEvidenceComplete"] is False
    assert resources["peakWholeGpuUsedVramBytes"] == 123_456_789
    assert resources["minimumWholeGpuFreeVramBytes"] == 123_456_790
    assert resources["wholeGpuVramEvidenceAvailable"] is True
    assert resources["wholeGpuMemoryIncludesOtherProcesses"] is True
    assert resources["wholeGpuMemoryDiagnosticOnly"] is True
    assert first["evaluationPolicy"][
        "automaticLatencyAndResourceMetricsAreDiagnosticOnly"
    ] is True
    assert _package_bytes(first_root) == _package_bytes(second_root)
    assert first["blindReviewPublication"]["manifestCanonicalSha256"] == second[
        "blindReviewPublication"
    ]["manifestCanonicalSha256"]
