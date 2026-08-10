"""Run the visible semantic-regression fixtures through one local Ollama model.

This runner intentionally does not load the frozen benchmark manifest, reference
transcripts, scorer output, or identity vault.  It only consumes the small
fixture descriptors committed under ``benchmarks/product_reviews`` and records
the complete prompt/response exchange for a focused model check.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.local_llm import LocalLLMConfig, OllamaLocalProvider  # noqa: E402
from backend.persistence import (  # noqa: E402
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
    sha256_file,
)
from backend.semantic_candidate_lattice import (  # noqa: E402
    build_semantic_candidate_lattice,
)
from backend.semantic_composition import (  # noqa: E402
    SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION,
    SemanticJobArbitrationRunner,
)


MODEL = "qwen3.5:27b-q4_K_M"
MODEL_DIGEST = "sha256:7653528ba5cba4dd8e19da24aaddc7f4d0b5ecd93571c0825dfd4137958ec06e"
BATCH_SIZE = 8
FIXTURE_NAMES = (
    "fleurs_ar_eg_validation_090.semantic-v13-major-regression.v1.json",
    "fleurs_es_419_validation_009.semantic-v14-major-regression.v1.json",
    "fleurs_id_id_validation_003.semantic-v14-major-regression.v1.json",
    "fleurs_ja_jp_validation_090.semantic-v14-major-regression.v1.json",
    "minds_fr_fr_089.semantic-v13-major-regression.v1.json",
    "minds_zh_cn_258.semantic-v13-major-regression.v1.json",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json_no_replace(path, dict(value))


def _candidate(
    payload: Mapping[str, Any],
    *,
    current: bool,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "payload": dict(payload),
        "producers": [dict(producer)],
        "selectionEligible": True,
        "eligibilityReason": "eligible",
        "isCurrent": current,
    }


def _build_document(fixture: Mapping[str, Any]) -> dict[str, Any]:
    visible = fixture["visibleInput"]
    binding = fixture["inputBinding"]
    configured_speaker_ids = visible.get("speakerIds")
    if isinstance(configured_speaker_ids, list) and configured_speaker_ids:
        speaker_ids = [str(item) for item in configured_speaker_ids]
    else:
        speaker_ids = list(
            dict.fromkeys(
                str(item["speakerId"]) for item in visible["segments"]
            )
        )
    if not speaker_ids:
        raise ValueError(f"fixture has no visible speakers: {fixture['fixtureId']}")
    segments: list[dict[str, Any]] = []
    for item in visible["segments"]:
        speaker_id = str(item["speakerId"])
        text = str(item["text"])
        segments.append(
            {
                "id": str(item["segmentId"]),
                "startMs": int(item["startMs"]),
                "endMs": int(item["endMs"]),
                "speakerId": speaker_id,
                "rawText": text,
                "normalizedText": text,
                "displayText": text,
                "confidence": 0.8,
                "speakerScores": [
                    {
                        "speakerId": candidate_id,
                        "score": 0.9 if candidate_id == speaker_id else 0.6,
                    }
                    for candidate_id in speaker_ids
                ],
                "speakerMargin": 0.3,
                "overlapping": False,
                "humanLocked": False,
                "revisions": [],
                "language": str(item["language"]),
                "evidence": {"asr": {"provider": "frozen-visible-fixture"}},
            }
        )
    return {
        "schemaVersion": "2.0.0",
        "documentId": f"doc-{fixture['fixtureId']}",
        "jobId": f"job-{fixture['fixtureId']}",
        "generatedAt": "2026-08-10T00:00:00Z",
        "language": str(visible["language"]),
        "source": {
            "fileName": "frozen-visible-regression.wav",
            "sha256": str(binding["sourceMediaSha256"]),
            "durationMs": int(binding["sourceDurationMs"]),
        },
        "speakerPolicy": {
            "mode": "auto",
            "resolvedCount": len(speaker_ids),
            "speakerIds": speaker_ids,
            "requireExactSet": True,
            "unknownSpeakerAllowed": False,
            "speakerChangeRequiresEvidence": True,
        },
        "speakers": [{"id": speaker_id} for speaker_id in speaker_ids],
        "segments": segments,
        "provenance": {
            "offline": True,
            "models": [],
            "frozenTranscriptDocumentSha256": str(
                binding["transcriptDocumentSha256"]
            ),
            "frozenCandidateLatticeSha256": str(
                binding["candidateLatticeSha256"]
            ),
        },
    }


def _build_lattice(document: Mapping[str, Any]) -> dict[str, Any]:
    speaker_ids = [
        str(item) for item in document["speakerPolicy"]["speakerIds"]
    ]
    segments = list(document["segments"])
    duration_ms = int(document["source"]["durationMs"])
    producer = {
        "producerType": "model",
        "systemId": "frozen-visible-fixture-builder",
        "revision": "1.0.0",
        "artifactSha256": "c" * 64,
        "modelManifestSha256": "d" * 64,
        "identityStatus": "manifest-bound",
    }
    current_turns = [
        {
            "startMs": int(segment["startMs"]),
            "endMs": int(segment["endMs"]),
            "speakerId": str(segment["speakerId"]),
            "overlap": False,
        }
        for segment in segments
    ]
    consolidated_turns = [
        {**turn, "speakerId": speaker_ids[0]}
        for turn in current_turns
    ]
    groups: dict[str, list[dict[str, Any]]] = {
        "speech-disposition": [
            {
                "scopeId": "media",
                "candidates": [
                    _candidate(
                        {
                            "classification": "transcribable-speech",
                            "startMs": 0,
                            "endMs": duration_ms,
                        },
                        current=True,
                        producer=producer,
                    )
                ],
            }
        ],
        "speaker-cardinality-timeline": [
            {
                "scopeId": "media",
                "candidates": [
                    _candidate(
                        {
                            "speakerCount": len(speaker_ids),
                            "speakerIds": speaker_ids,
                            "timelineKind": "current-transcript",
                            "startMs": 0,
                            "endMs": duration_ms,
                            "turns": current_turns,
                        },
                        current=True,
                        producer=producer,
                    ),
                    _candidate(
                        {
                            "speakerCount": 1,
                            "speakerIds": [speaker_ids[0]],
                            "timelineKind": "overlap-preserving",
                            "startMs": 0,
                            "endMs": duration_ms,
                            "turns": consolidated_turns,
                        },
                        current=False,
                        producer=producer,
                    ),
                    _candidate(
                        {
                            "speakerCount": 1,
                            "speakerIds": [speaker_ids[0]],
                            "timelineKind": "single-speaker",
                            "startMs": 0,
                            "endMs": duration_ms,
                            "turns": consolidated_turns,
                        },
                        current=False,
                        producer=producer,
                    ),
                ],
            }
        ],
        "speaker-assignment": [],
        "language-span": [],
        "asr-text": [],
    }
    base_language = str(document["language"]).split("-", 1)[0]
    for segment in segments:
        segment_id = str(segment["id"])
        scope_id = f"segment:{segment_id}"
        current_speaker = str(segment["speakerId"])
        segment_start = int(segment["startMs"])
        segment_end = int(segment["endMs"])
        segment_language = str(segment["language"])
        groups["speaker-assignment"].append(
            {
                "scopeId": scope_id,
                "candidates": [
                    _candidate(
                        {
                            "segmentId": segment_id,
                            "startMs": segment_start,
                            "endMs": segment_end,
                            "speakerId": speaker_id,
                            "score": 0.9 if speaker_id == current_speaker else 0.6,
                        },
                        current=speaker_id == current_speaker,
                        producer=producer,
                    )
                    for speaker_id in speaker_ids
                ],
            }
        )
        groups["language-span"].append(
            {
                "scopeId": scope_id,
                "candidates": [
                    _candidate(
                        {
                            "segmentId": segment_id,
                            "startMs": segment_start,
                            "endMs": segment_end,
                            "language": language,
                            "confidence": None,
                        },
                        current=language == segment_language,
                        producer=producer,
                    )
                    for language in (segment_language, base_language)
                ],
            }
        )
        candidate_set_sha256 = _sha256_text(segment_id)
        groups["asr-text"].append(
            {
                "scopeId": scope_id,
                "candidates": [
                    _candidate(
                        {
                            "segmentId": segment_id,
                            "startMs": segment_start,
                            "endMs": segment_end,
                            "text": str(segment["normalizedText"]),
                            "language": segment_language,
                            "sourceCandidateId": (
                                f"asr-{candidate_set_sha256[:24]}"
                            ),
                            "candidateSetSha256": candidate_set_sha256,
                        },
                        current=True,
                        producer=producer,
                    )
                ],
            }
        )
    return build_semantic_candidate_lattice(
        source_media_sha256=str(document["source"]["sha256"]),
        transcript_sha256=canonical_json_sha256(document),
        transcript_schema_version=str(document["schemaVersion"]),
        source_duration_ms=duration_ms,
        candidate_groups=groups,
    )


class _RecordingProvider:
    """Capture complete local exchanges while preserving the provider contract."""

    def __init__(self, delegate: OllamaLocalProvider) -> None:
        self.delegate = delegate
        self.config = delegate.config
        self.provider_id = delegate.provider_id
        self.provider_version = delegate.provider_version
        self.network_policy = delegate.network_policy
        self.calls: list[dict[str, Any]] = []

    @property
    def generation_metrics(self) -> dict[str, int]:
        return self.delegate.generation_metrics

    def generate_json(self, **kwargs: Any) -> Mapping[str, Any]:
        started = time.perf_counter_ns()
        metrics_before = self.generation_metrics
        response: Mapping[str, Any] | None = None
        error: dict[str, Any] | None = None
        try:
            response = self.delegate.generate_json(**kwargs)
            return response
        except Exception as exc:  # noqa: BLE001 - persist the exact failure
            diagnostics = getattr(exc, "diagnostics", None)
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "diagnostics": dict(diagnostics or {}),
            }
            raise
        finally:
            metrics_after = self.generation_metrics
            system_prompt = str(kwargs.get("system_prompt") or "")
            user_prompt = str(kwargs.get("user_prompt") or "")
            schema = kwargs.get("response_schema")
            response_value = dict(response) if response is not None else None
            self.calls.append(
                {
                    "callIndex": len(self.calls) + 1,
                    "model": str(kwargs.get("model") or ""),
                    "temperature": kwargs.get("temperature"),
                    "systemPrompt": system_prompt,
                    "userPrompt": user_prompt,
                    "responseSchema": schema,
                    "response": response_value,
                    "systemPromptSha256": _sha256_text(system_prompt),
                    "userPromptSha256": _sha256_text(user_prompt),
                    "responseSchemaSha256": canonical_json_sha256(schema)
                    if isinstance(schema, Mapping)
                    else None,
                    "responseCanonicalSha256": canonical_json_sha256(
                        response_value
                    )
                    if response_value is not None
                    else None,
                    "error": error,
                    "wallTimeNanoseconds": time.perf_counter_ns() - started,
                    "providerMetricsBefore": metrics_before,
                    "providerMetricsAfter": metrics_after,
                }
            )

    def release_resources(self) -> None:
        self.delegate.release_resources()


def _load_fixture(path: Path) -> dict[str, Any]:
    forbidden = {"reference", "identity-vault", "scorer", "vault"}
    if any(token in str(path).casefold() for token in forbidden):
        raise ValueError("regression runner refuses hidden/reference paths")
    fixture = read_json_strict(path)
    if not isinstance(fixture, dict):
        raise ValueError("fixture root must be an object")
    expected = fixture.get("canonicalSha256")
    body = copy.deepcopy(fixture)
    body.pop("canonicalSha256", None)
    if expected != canonical_json_sha256(body):
        raise ValueError(f"fixture canonical hash mismatch: {path}")
    return fixture


def _expected_model_digest(endpoint: str, model: str) -> str:
    import urllib.request

    with urllib.request.urlopen(endpoint.rstrip("/") + "/api/tags", timeout=30) as response:
        inventory = json.loads(response.read().decode("utf-8"))
    for item in inventory.get("models", []):
        if isinstance(item, Mapping) and model in {
            item.get("name"),
            item.get("model"),
        }:
            digest = str(item.get("digest") or "").casefold()
            if not digest.startswith("sha256:"):
                digest = f"sha256:{digest}"
            return digest
    raise ValueError(f"model is absent from Ollama inventory: {model}")


def run(
    *,
    output: Path,
    endpoint: str,
    timeout_seconds: float,
    model: str = MODEL,
    model_digest: str = MODEL_DIGEST,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    model = model.strip()
    if not model:
        raise ValueError("model must be non-empty")
    model_digest = model_digest.strip().casefold()
    if not model_digest.startswith("sha256:"):
        model_digest = f"sha256:{model_digest}"
    if (
        len(model_digest) != 71
        or any(character not in "0123456789abcdef" for character in model_digest[7:])
    ):
        raise ValueError("model digest must be SHA-256")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 32
    ):
        raise ValueError("batch size must be between 1 and 32")
    fixture_paths = [
        PROJECT_ROOT
        / "benchmarks"
        / "product_reviews"
        / "development-20260809"
        / name
        for name in FIXTURE_NAMES
    ]
    fixtures = [_load_fixture(path) for path in fixture_paths]
    actual_digest = _expected_model_digest(endpoint, model)
    if actual_digest != model_digest:
        raise ValueError(
            f"pinned model digest mismatch: expected {model_digest}, got {actual_digest}"
        )
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "run-input-fixtures.json",
        {
            "schemaVersion": "1.0.0",
            "model": model,
            "modelDigest": actual_digest,
            "fixtureNames": list(FIXTURE_NAMES),
            "fixtureFileSha256": {
                path.name: sha256_file(path) for path in fixture_paths
            },
            "hiddenReferenceAccess": False,
        },
    )
    config = LocalLLMConfig(
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        temperature=0.0,
        top_p=0.1,
        context_tokens=32_768,
        output_tokens=4_096,
        keep_alive="10m",
        release_on_close=True,
        offline_only=True,
        expected_model_digest=actual_digest,
    )
    provider = _RecordingProvider(OllamaLocalProvider(config))
    runner = SemanticJobArbitrationRunner(
        provider=provider,
        model=model,
        batch_size=batch_size,
        max_batch_attempts=2,
        context_tokens=config.context_tokens,
        output_tokens=config.output_tokens,
    )
    case_rows: list[dict[str, Any]] = []
    started = time.perf_counter_ns()
    try:
        for fixture_path, fixture in zip(fixture_paths, fixtures):
            case_dir = output / fixture["fixtureId"]
            case_dir.mkdir()
            document = _build_document(fixture)
            lattice = _build_lattice(document)
            _write_json(case_dir / "visible-fixture.json", fixture)
            _write_json(case_dir / "generated-document.json", document)
            _write_json(case_dir / "generated-lattice.json", lattice)
            case_started = time.perf_counter_ns()
            call_start_index = len(provider.calls)
            error: dict[str, Any] | None = None
            arbitration: dict[str, Any] | None = None
            try:
                arbitration = runner.run(document, candidate_lattice=lattice)
            except Exception as exc:  # noqa: BLE001 - persist exact case failure
                diagnostics = getattr(exc, "diagnostics", None)
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "diagnostics": dict(diagnostics or {}),
                }
            case_calls = provider.calls[call_start_index:]
            for index, call in enumerate(case_calls, start=1):
                _write_json(case_dir / f"llm-call-{index:02d}.json", call)
            if arbitration is not None:
                _write_json(case_dir / "semantic-job-arbitration.json", arbitration)
            expected = {
                (
                    str(item["domain"]),
                    str(item["scopeId"]),
                    str(item["requestKind"]),
                )
                for item in fixture["blindReviewFinding"]["expectedRequests"]
            }
            actual = {
                (
                    str(item["domain"]),
                    str(item["scopeId"]),
                    str(item["requestKind"]),
                )
                for item in (arbitration or {}).get(
                    "candidateGenerationRequests", []
                )
            }
            row = {
                "fixtureId": fixture["fixtureId"],
                "fixtureFile": str(fixture_path),
                "sourceMediaSha256": fixture["inputBinding"]["sourceMediaSha256"],
                "generatedDocumentCanonicalSha256": canonical_json_sha256(document),
                "generatedLatticeCanonicalSha256": canonical_json_sha256(lattice),
                "status": (arbitration or {}).get("status"),
                "arbitrationCanonicalSha256": canonical_json_sha256(arbitration)
                if arbitration is not None
                else None,
                "expectedRequestSet": [list(item) for item in sorted(expected)],
                "actualRequestSet": [list(item) for item in sorted(actual)],
                "requestSetExactMatch": actual == expected,
                "providerCallCount": len(case_calls),
                "wallTimeNanoseconds": time.perf_counter_ns() - case_started,
                "error": error,
            }
            _write_json(case_dir / "case-result.json", row)
            case_rows.append(row)
    finally:
        runner.release_resources()
    report = {
        "schemaVersion": "1.0.0",
        "artifactType": "semantic-regression-fixture-real-run",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "modelDigest": actual_digest,
        "endpoint": endpoint,
        "promptVersion": SEMANTIC_JOB_ARBITRATION_PROMPT_VERSION,
        "rubricVersion": "multilingual-fidelity-v2",
        "contextTokens": config.context_tokens,
        "outputTokens": config.output_tokens,
        "batchSize": batch_size,
        "hiddenReferenceAccess": False,
        "wallTimeNanoseconds": time.perf_counter_ns() - started,
        "providerMetrics": provider.generation_metrics,
        "caseResults": case_rows,
        "allRequestSetsExact": all(
            row["requestSetExactMatch"] for row in case_rows
        ),
        "allCasesCompleted": all(row["error"] is None for row in case_rows),
        "callEvidenceCanonicalSha256": canonical_json_sha256(provider.calls),
    }
    _write_json(output / "run-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new empty output directory (prefer D:/mts-eval)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-digest", default=MODEL_DIGEST)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    report = run(
        output=args.output,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        model=args.model,
        model_digest=args.model_digest,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (
        report["allCasesCompleted"] and report["allRequestSetsExact"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
