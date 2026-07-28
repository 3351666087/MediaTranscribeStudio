#!/usr/bin/env python3
"""Generate real VAD, LID, and local ASR challengers for one semantic job."""

from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.adapters import AdapterContext
from backend.persistence import (
    atomic_write_json_no_replace,
    canonical_json_sha256,
    read_json_strict,
)
from backend.production_runners import LocalQwen3AsrAdapter
from backend.semantic_candidate_generation import (
    SemanticCandidateGenerationRegistry,
    build_asr_text_challenger_result,
    build_open_set_lid_challenger_result,
    build_voice_activity_challenger_result,
)
from backend.semantic_candidate_lattice import validate_semantic_candidate_lattice
from backend.semantic_composition import validate_semantic_job_arbitration
from backend.speaker_pipeline import PreparedAudio, SpeechWindow
from backend.voice_activity import validate_voice_activity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--lattice", required=True, type=Path)
    parser.add_argument("--arbitration", required=True, type=Path)
    parser.add_argument("--voice-activity", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-inference-batch-size", type=int, default=2)
    parser.add_argument("--max-generated-tokens", type=int, default=128)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segment_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("transcript must contain segments")
    segments = {
        str(segment["id"]): dict(segment)
        for segment in raw_segments
        if isinstance(segment, dict)
    }
    if len(segments) != len(raw_segments):
        raise ValueError("transcript segments must be unique objects")
    return segments


def main() -> int:
    args = _parser().parse_args()
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    document = read_json_strict(args.transcript.resolve())
    transcript_sha = canonical_json_sha256(document)
    lattice = validate_semantic_candidate_lattice(
        read_json_strict(args.lattice.resolve()),
        expected_source_media_sha256=document["source"]["sha256"],
        expected_transcript_sha256=transcript_sha,
    )
    arbitration = validate_semantic_job_arbitration(
        read_json_strict(args.arbitration.resolve()),
        expected_job_id=document["jobId"],
        expected_lattice=lattice,
    )
    voice_activity = validate_voice_activity(
        read_json_strict(args.voice_activity.resolve())
    )
    audio_path = args.audio.resolve()
    if (
        _sha256_file(audio_path) != document["source"]["sha256"]
        or voice_activity["sourceSha256"] != document["source"]["sha256"]
        or voice_activity["jobId"] != document["jobId"]
        or voice_activity["mediaDurationMs"] != document["source"]["durationMs"]
    ):
        raise ValueError("audio or voice-activity evidence is rebound to another job")

    segments = _segment_index(document)
    requested_scopes = {
        str(request["scopeId"])
        for request in arbitration["candidateGenerationRequests"]
        if request["domain"] in {"language-span", "asr-text"}
    }
    requested_segment_ids = {
        scope.removeprefix("segment:")
        for scope in requested_scopes
        if scope.startswith("segment:")
    }
    if requested_segment_ids != set(segments):
        raise ValueError(
            "real semantic challenger requires language and ASR requests for "
            "every transcript segment"
        )

    ordered_segments = sorted(
        segments.values(),
        key=lambda item: (
            int(item["startMs"]),
            int(item["endMs"]),
            str(item["id"]),
        ),
    )
    windows = tuple(
        SpeechWindow(
            window_id=str(segment["id"]),
            start_ms=int(segment["startMs"]),
            end_ms=int(segment["endMs"]),
        )
        for segment in ordered_segments
    )
    prepared = PreparedAudio(
        duration_ms=int(document["source"]["durationMs"]),
        source_fingerprint=str(document["source"]["sha256"]),
        normalization_profile="mono-16khz-f32-v1",
        windows=windows,
        stage_durations_ms={
            "decode": 0.0,
            "normalize": 0.0,
            "vad": 0.0,
            "boundary": 0.0,
        },
        audio_path=str(audio_path),
    )
    context = AdapterContext(
        job_id=str(document["jobId"]),
        output_directory=output.parent,
        cancellation=threading.Event(),
    )
    adapter = LocalQwen3AsrAdapter(
        model_path=args.qwen_model.resolve(),
        forced_aligner_path=None,
        device_map=args.device,
        torch_dtype=args.dtype,
        max_inference_batch_size=args.max_inference_batch_size,
        retry_empty_results=False,
    )
    started = time.monotonic()
    try:
        hypotheses = adapter.transcribe_batch(
            prepared,
            windows,
            context,
            requested_language="auto",
            max_generated_tokens=args.max_generated_tokens,
        )
    finally:
        adapter.release_resources()
    elapsed = time.monotonic() - started
    if len(hypotheses) != len(windows):
        raise ValueError("local ASR re-decode omitted one or more segment results")
    result_by_segment = {
        window.window_id: hypothesis
        for window, hypothesis in zip(windows, hypotheses, strict=True)
    }

    def speech_handler(
        _request: dict[str, Any],
        _document: dict[str, Any],
        _lattice: dict[str, Any],
    ) -> dict[str, Any]:
        return build_voice_activity_challenger_result(
            voice_activity=voice_activity,
            artifact_sha256=canonical_json_sha256(voice_activity),
        )

    def lid_handler(
        request: dict[str, Any],
        _document: dict[str, Any],
        _lattice: dict[str, Any],
    ) -> dict[str, Any]:
        segment_id = str(request["scopeId"]).removeprefix("segment:")
        segment = segments[segment_id]
        evidence = result_by_segment[segment_id].evidence
        return build_open_set_lid_challenger_result(
            segment_id=segment_id,
            start_ms=int(segment["startMs"]),
            end_ms=int(segment["endMs"]),
            language=str(evidence["language"]),
            confidence=None,
            system_id=str(evidence["modelId"]),
            revision=str(evidence["modelRevision"]),
            artifact_sha256=str(evidence["candidateSetSha256"]),
            model_manifest_sha256=str(evidence["modelManifestSha256"]),
            evidence_sha256=str(evidence["candidateSetSha256"]),
        )

    def asr_handler(
        request: dict[str, Any],
        _document: dict[str, Any],
        _lattice: dict[str, Any],
    ) -> dict[str, Any]:
        segment_id = str(request["scopeId"]).removeprefix("segment:")
        segment = segments[segment_id]
        return build_asr_text_challenger_result(
            segment_id=segment_id,
            start_ms=int(segment["startMs"]),
            end_ms=int(segment["endMs"]),
            candidate_set=result_by_segment[segment_id].evidence,
        )

    generation = SemanticCandidateGenerationRegistry(
        {
            "speech-disposition-challenger": speech_handler,
            "open-set-lid": lid_handler,
            "provider-native-nbest": asr_handler,
        }
    ).fulfill(
        document,
        lattice,
        arbitration,
    )
    evidence_artifact = {
        "schemaVersion": "1.0.0",
        "artifactType": "semantic-local-asr-redecode-evidence",
        "jobId": document["jobId"],
        "sourceMediaSha256": document["source"]["sha256"],
        "inputLatticeSha256": lattice["latticeSha256"],
        "arbitrationDecisionSha256": arbitration["decisionSha256"],
        "model": adapter.evidence_cache_identity()["asrModel"],
        "device": args.device,
        "dtype": args.dtype,
        "maxGeneratedTokens": args.max_generated_tokens,
        "elapsedSeconds": round(elapsed, 6),
        "segments": [
            {
                "segmentId": window.window_id,
                "startMs": window.start_ms,
                "endMs": window.end_ms,
                "text": hypothesis.text,
                "language": hypothesis.evidence["language"],
                "candidateSetSha256": hypothesis.evidence[
                    "candidateSetSha256"
                ],
            }
            for window, hypothesis in zip(windows, hypotheses, strict=True)
        ],
        "claimsFinalQualityImprovement": False,
    }
    audit = {
        "schemaVersion": "1.0.0",
        "artifactType": "semantic-evidence-challenger-audit",
        "jobId": document["jobId"],
        "inputLatticeSha256": lattice["latticeSha256"],
        "outputLatticeSha256": generation["outputLattice"]["latticeSha256"],
        "generationArtifactSha256": canonical_json_sha256(generation),
        "redecodeEvidenceSha256": canonical_json_sha256(evidence_artifact),
        "metrics": {
            **generation["metrics"],
            "asrSegmentCount": len(hypotheses),
            "asrElapsedSeconds": round(elapsed, 6),
        },
        "claimsFinalQualityImprovement": False,
        "conclusion": "candidate-generation-completed",
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json_no_replace(
        output / "semantic-local-asr-redecode-evidence.v1.json",
        evidence_artifact,
    )
    atomic_write_json_no_replace(
        output / "semantic-candidate-generation.v1.json",
        generation,
    )
    atomic_write_json_no_replace(
        output / "semantic-candidate-lattice.extended.v1.json",
        generation["outputLattice"],
    )
    atomic_write_json_no_replace(output / "audit-report.v1.json", audit)
    print(
        f"{generation['status']}: "
        f"{generation['metrics']['fulfilledRequestCount']} requests fulfilled, "
        f"{generation['metrics']['outputAvailableGroupCount']} groups available, "
        f"{elapsed:.6f} ASR seconds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
