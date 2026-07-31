"""Production-local challenger registry for mandatory semantic composition."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adapters import AdapterContext
from .errors import WorkerError
from .persistence import canonical_json_sha256, read_json_strict
from .production_runners import LocalPyannoteAuditAdapter, LocalQwen3AsrAdapter
from .semantic_candidate_generation import (
    SemanticCandidateGenerationRegistry,
    build_asr_text_challenger_result,
    build_open_set_lid_challenger_result,
    build_timeline_challenger_result,
    build_voice_activity_challenger_result,
)
from .speaker_pipeline import PreparedAudio, SpeechWindow
from .voice_activity import validate_voice_activity


class ProductionSemanticCandidateRegistry(
    SemanticCandidateGenerationRegistry
):
    """Generate real VAD, diarization, LID, speaker, and ASR candidates."""

    def __init__(
        self,
        *,
        asr_adapter: LocalQwen3AsrAdapter,
        pyannote_adapter: LocalPyannoteAuditAdapter,
        context: AdapterContext,
        max_generated_tokens: int = 128,
    ) -> None:
        self.asr_adapter = asr_adapter
        self.pyannote_adapter = pyannote_adapter
        self.context = context
        if (
            isinstance(max_generated_tokens, bool)
            or not isinstance(max_generated_tokens, int)
            or max_generated_tokens < 32
            or max_generated_tokens > 512
        ):
            raise ValueError(
                "semantic ASR max_generated_tokens must be between 32 and 512"
            )
        self.max_generated_tokens = max_generated_tokens
        self._lock = threading.RLock()
        self._asr_by_segment: dict[str, Any] | None = None
        self._timeline_results: list[dict[str, Any]] | None = None
        super().__init__(
            {
                "speech-disposition-challenger": self._speech,
                "timeline-challenger": self._timeline,
                "speaker-assignment-challenger": self._speaker_assignment,
                "open-set-lid": self._language,
                "provider-native-nbest": self._asr,
            }
        )

    def release_resources(self) -> None:
        first_error: Exception | None = None
        for adapter in (self.asr_adapter, self.pyannote_adapter):
            release = getattr(adapter, "release_resources", None)
            if not callable(release):
                continue
            try:
                release()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    @staticmethod
    def _segments(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        raw_segments = document.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise WorkerError(
                "SEMANTIC_CHALLENGER_INPUT_INVALID",
                "semantic challengers require transcript segments",
            )
        segments = {
            str(segment.get("id") or ""): dict(segment)
            for segment in raw_segments
            if isinstance(segment, Mapping)
        }
        if (
            len(segments) != len(raw_segments)
            or "" in segments
        ):
            raise WorkerError(
                "SEMANTIC_CHALLENGER_INPUT_INVALID",
                "semantic challenger segment identities are invalid",
            )
        return segments

    @staticmethod
    def _prepared_audio(
        document: Mapping[str, Any],
        segments: Mapping[str, Mapping[str, Any]],
    ) -> PreparedAudio:
        audio_paths: set[str] = set()
        profiles: set[str] = set()
        for segment in segments.values():
            evidence = segment.get("evidence")
            preparation = (
                evidence.get("preparation")
                if isinstance(evidence, Mapping)
                else None
            )
            if not isinstance(preparation, Mapping):
                raise WorkerError(
                    "SEMANTIC_CHALLENGER_INPUT_INVALID",
                    "semantic challengers require preparation evidence",
                )
            audio_path = preparation.get("audioPath")
            canonical_audio_path = preparation.get("canonicalAudioPath")
            if not isinstance(canonical_audio_path, str):
                canonical_audio_path = audio_path
            profile = preparation.get("normalizationProfile")
            if not isinstance(audio_path, str) or not Path(audio_path).is_file():
                raise WorkerError(
                    "PREPARED_AUDIO_MISSING",
                    "semantic challengers require persisted normalized audio",
                )
            if (
                not isinstance(canonical_audio_path, str)
                or not Path(canonical_audio_path).is_file()
            ):
                raise WorkerError(
                    "PREPARED_AUDIO_MISSING",
                    "semantic challengers require the canonical normalized timeline",
                )
            if not isinstance(profile, str) or not profile:
                raise WorkerError(
                    "SEMANTIC_CHALLENGER_INPUT_INVALID",
                    "semantic challengers require a normalization profile",
                )
            audio_paths.add(canonical_audio_path)
            profiles.add(profile)
        if len(audio_paths) != 1 or len(profiles) != 1:
            raise WorkerError(
                "SEMANTIC_CHALLENGER_INPUT_INVALID",
                "semantic challenger segments must share one normalized timeline",
            )
        source = document.get("source")
        if not isinstance(source, Mapping):
            raise WorkerError(
                "SEMANTIC_CHALLENGER_INPUT_INVALID",
                "semantic challengers require transcript source evidence",
            )
        ordered = sorted(
            segments.values(),
            key=lambda item: (
                int(item["startMs"]),
                int(item["endMs"]),
                str(item["id"]),
            ),
        )
        return PreparedAudio(
            duration_ms=int(source["durationMs"]),
            source_fingerprint=str(source["sha256"]),
            normalization_profile=next(iter(profiles)),
            windows=tuple(
                SpeechWindow(
                    window_id=str(segment["id"]),
                    start_ms=int(segment["startMs"]),
                    end_ms=int(segment["endMs"]),
                )
                for segment in ordered
            ),
            stage_durations_ms={
                "decode": 0.0,
                "normalize": 0.0,
                "vad": 0.0,
                "boundary": 0.0,
            },
            audio_path=next(iter(audio_paths)),
        )

    def _ensure_asr(
        self,
        document: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if self._asr_by_segment is not None:
                return self._asr_by_segment
            segments = self._segments(document)
            prepared = self._prepared_audio(document, segments)
            hypotheses = self.asr_adapter.transcribe_batch(
                prepared,
                prepared.windows,
                self.context,
                requested_language="auto",
                max_generated_tokens=self.max_generated_tokens,
            )
            if len(hypotheses) != len(prepared.windows):
                raise WorkerError(
                    "SEMANTIC_ASR_REDECODE_INCOMPLETE",
                    "semantic ASR re-decode omitted transcript segments",
                )
            self._asr_by_segment = {
                window.window_id: hypothesis
                for window, hypothesis in zip(
                    prepared.windows,
                    hypotheses,
                    strict=True,
                )
            }
            return self._asr_by_segment

    def _ensure_timelines(
        self,
        document: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self._timeline_results is not None:
                return self._timeline_results
            segments = self._segments(document)
            prepared = self._prepared_audio(document, segments)
            raw = self.pyannote_adapter.timeline_challenger(
                str(prepared.audio_path),
                duration_ms=prepared.duration_ms,
                context=self.context,
            )
            result_sets: list[tuple[str, list[dict[str, Any]]]] = [
                ("overlap-preserving", raw["speakerTurns"])
            ]
            exclusive = raw.get("exclusiveSpeakerTurns")
            if isinstance(exclusive, list) and exclusive:
                result_sets.append(("single-speaker", exclusive))
            outputs: list[dict[str, Any]] = []
            for timeline_kind, turns in result_sets:
                result = build_timeline_challenger_result(
                    turns=turns,
                    source_duration_ms=prepared.duration_ms,
                    system_id=str(raw["modelId"]),
                    revision=str(raw["modelRevision"]),
                    artifact_sha256=canonical_json_sha256(
                        {
                            "timelineKind": timeline_kind,
                            "turns": turns,
                        }
                    ),
                    model_manifest_sha256=str(
                        raw["modelManifestSha256"]
                    ),
                    local_speaker_field="localSpeaker",
                )
                payload = dict(result["candidates"][0]["payload"])
                payload["timelineKind"] = timeline_kind
                outputs.append(
                    {
                        "producer": result["producer"],
                        "payload": payload,
                    }
                )
            self._timeline_results = outputs
            return self._timeline_results

    def _speech(
        self,
        _request: Mapping[str, Any],
        document: Mapping[str, Any],
        _lattice: Mapping[str, Any],
    ) -> dict[str, Any]:
        path = self.context.output_directory / "voice-activity.v1.json"
        voice = validate_voice_activity(read_json_strict(path))
        if (
            voice["jobId"] != document.get("jobId")
            or voice["sourceSha256"]
            != document.get("source", {}).get("sha256")
        ):
            raise WorkerError(
                "SEMANTIC_CHALLENGER_BINDING_INVALID",
                "voice activity is rebound to another semantic job",
            )
        return build_voice_activity_challenger_result(
            voice_activity=voice,
            artifact_sha256=canonical_json_sha256(voice),
        )

    def _timeline(
        self,
        _request: Mapping[str, Any],
        document: Mapping[str, Any],
        _lattice: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "candidateResults": self._ensure_timelines(document),
        }

    def _speaker_assignment(
        self,
        request: Mapping[str, Any],
        document: Mapping[str, Any],
        _lattice: Mapping[str, Any],
    ) -> dict[str, Any]:
        segment_id = str(request["scopeId"]).removeprefix("segment:")
        segment = self._segments(document)[segment_id]
        timeline_result = self._ensure_timelines(document)[0]
        timeline = timeline_result["payload"]
        start_ms = int(segment["startMs"])
        end_ms = int(segment["endMs"])
        overlap_by_speaker = {
            speaker_id: sum(
                max(
                    0,
                    min(end_ms, int(turn["endMs"]))
                    - max(start_ms, int(turn["startMs"])),
                )
                for turn in timeline["turns"]
                if turn["speakerId"] == speaker_id
            )
            for speaker_id in timeline["speakerIds"]
        }
        speaker_id = min(
            overlap_by_speaker,
            key=lambda item: (-overlap_by_speaker[item], item),
        )
        score = overlap_by_speaker[speaker_id] / (end_ms - start_ms)
        return {
            "producer": timeline_result["producer"],
            "candidates": [
                {
                    "payload": {
                        "segmentId": segment_id,
                        "startMs": start_ms,
                        "endMs": end_ms,
                        "speakerId": speaker_id,
                        "score": score,
                    }
                }
            ],
        }

    def _language(
        self,
        request: Mapping[str, Any],
        document: Mapping[str, Any],
        _lattice: Mapping[str, Any],
    ) -> dict[str, Any]:
        segment_id = str(request["scopeId"]).removeprefix("segment:")
        segment = self._segments(document)[segment_id]
        evidence = self._ensure_asr(document)[segment_id].evidence
        raw_language = evidence.get("language")
        if not isinstance(raw_language, str) or not raw_language:
            hypotheses = evidence.get("nBest")
            first = (
                hypotheses[0]
                if isinstance(hypotheses, list) and hypotheses
                else None
            )
            raw_language = (
                first.get("language")
                if isinstance(first, Mapping)
                else None
            )
        if not isinstance(raw_language, str) or not raw_language:
            raise WorkerError(
                "SEMANTIC_LID_EVIDENCE_INVALID",
                "ASR re-decode did not bind a detected language",
                details={"segmentId": segment_id},
            )
        return build_open_set_lid_challenger_result(
            segment_id=segment_id,
            start_ms=int(segment["startMs"]),
            end_ms=int(segment["endMs"]),
            language=raw_language,
            confidence=None,
            system_id=str(evidence["modelId"]),
            revision=str(evidence["modelRevision"]),
            artifact_sha256=str(evidence["candidateSetSha256"]),
            model_manifest_sha256=str(evidence["modelManifestSha256"]),
            evidence_sha256=str(evidence["candidateSetSha256"]),
        )

    def _asr(
        self,
        request: Mapping[str, Any],
        document: Mapping[str, Any],
        _lattice: Mapping[str, Any],
    ) -> dict[str, Any]:
        segment_id = str(request["scopeId"]).removeprefix("segment:")
        segment = self._segments(document)[segment_id]
        hypothesis = self._ensure_asr(document)[segment_id]
        return build_asr_text_challenger_result(
            segment_id=segment_id,
            start_ms=int(segment["startMs"]),
            end_ms=int(segment["endMs"]),
            candidate_set=hypothesis.evidence,
        )


__all__ = ["ProductionSemanticCandidateRegistry"]
