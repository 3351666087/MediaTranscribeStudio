from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from backend import production_runners
from backend.adapters import AdapterContext
from backend.asr_evidence import build_asr_candidate_set
from backend.models import SpeakerScore, TranscriptSegment
from backend.pipeline_metrics import PipelineMetricsCollector
from backend.production_runners import LocalMossFormer2SeparationAdapter
from backend.speaker_pipeline import (
    AsrHypothesis,
    EmbeddingRecord,
    OverlapDecision,
    OverlapRecoveryInterval,
    PreparedAudio,
    SeparatedSpeechChannel,
    SpeakerPipeline,
    SpeakerPipelineConfig,
    SpeechWindow,
    _ClusterResult,
)


class _UnusedPreparation:
    adapter_id = "preparation"
    version = "1.0.0"


class _UnusedOverlap:
    adapter_id = "overlap"
    version = "1.0.0"


class _Embedding:
    adapter_id = "campp"
    version = "1.0.0"

    def embed_batch(self, prepared, windows, context):
        output = []
        for window in windows:
            vector = (
                (0.0, 1.0)
                if "ch2" in window.window_id
                else (1.0, 0.0)
            )
            output.append(
                EmbeddingRecord(
                    window_id=window.window_id,
                    vector=vector,
                    confidence=1.0,
                )
            )
        return output


class _Separation:
    adapter_id = "MossFormer2_SS_16K"
    version = "1.0.0"

    def separate_batch(self, prepared, intervals, context):
        interval = intervals[0]
        duration = interval.context_end_ms - interval.context_start_ms
        return [
            SeparatedSpeechChannel(
                candidate_id=(
                    f"overlap-recovery.{interval.interval_id}.ch{channel}"
                ),
                interval_id=interval.interval_id,
                channel_index=channel,
                audio_path=f"/tmp/channel-{channel}.wav",
                audio_sha256=str(channel) * 64,
                duration_ms=duration,
            )
            for channel in (1, 2)
        ]


class _Asr:
    adapter_id = "qwen"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.token_budgets: list[int | None] = []

    def transcribe_batch(
        self,
        prepared,
        windows,
        context,
        *,
        requested_language,
        max_generated_tokens=None,
    ):
        window = windows[0]
        self.calls.append(window.window_id)
        self.token_budgets.append(max_generated_tokens)
        tokens = [
            {"text": "次", "startMs": 800, "endMs": 900},
            {"text": "要", "startMs": 900, "endMs": 1000},
            {"text": "内容", "startMs": 1000, "endMs": 1200},
        ]
        candidate_set = build_asr_candidate_set(
            model_id="fixture-asr",
            model_revision="fixture",
            model_manifest_sha256="a" * 64,
            model_identity_status="injected-fixture",
            source_audio_sha256=prepared.source_fingerprint,
            normalization_profile=prepared.normalization_profile,
            source_window_id=window.window_id,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            hypotheses=[
                {
                    "text": "次要内容",
                    "language": "zh",
                    "tokens": tokens,
                    "acousticScore": None,
                    "acousticScoreStatus": "provider-unavailable",
                    "decodeScore": None,
                    "decodeScoreStatus": "provider-unavailable",
                }
            ],
        )
        return [
            AsrHypothesis(
                window_id=window.window_id,
                text="次要内容",
                confidence=0.0,
                evidence={
                    "language": "zh",
                    "timestamps": tokens,
                    **candidate_set,
                },
            )
        ]


def _pipeline() -> tuple[SpeakerPipeline, _Asr]:
    asr = _Asr()
    pipeline = SpeakerPipeline(
        preparation_adapter=_UnusedPreparation(),
        asr_adapter=asr,
        embedding_adapter=_Embedding(),
        overlap_adapter=_UnusedOverlap(),
        separation_adapter=_Separation(),
        config=SpeakerPipelineConfig(
            overlap_recovery_mode="guarded",
            overlap_recovery_margin_threshold=0.18,
        ),
    )
    return pipeline, asr


def _prepared() -> PreparedAudio:
    return PreparedAudio(
        duration_ms=4000,
        source_fingerprint="f" * 64,
        normalization_profile="mono-16khz-f32-v1",
        windows=(
            SpeechWindow("source-1", 0, 2000),
            SpeechWindow("source-2", 2000, 4000),
        ),
        stage_durations_ms={
            "decode": 0.0,
            "normalize": 0.0,
            "vad": 0.0,
            "boundary": 0.0,
        },
        audio_path="/tmp/source.wav",
    )


def _clusters() -> _ClusterResult:
    return _ClusterResult(
        count=2,
        confidence=1.0,
        candidate_min=2,
        candidate_max=2,
        assignments=(0, 1),
        scores=((1.0, 0.0), (0.0, 1.0)),
    )


def _primary_segment() -> TranscriptSegment:
    full_timeline = {
        "scope": "full-normalized-timeline",
        "startMs": 0,
        "endMs": 4000,
        "turnCount": 2,
        "localSpeakerCount": 2,
        "localSpeakers": ["A", "B"],
        "speakerTurns": [
            {"startMs": 0, "endMs": 2000, "localSpeaker": "A"},
            {"startMs": 2000, "endMs": 4000, "localSpeaker": "B"},
        ],
        "speakerTurnsSha256": "a" * 64,
    }
    return TranscriptSegment(
        segment_id="source-1",
        start_ms=0,
        end_ms=4000,
        speaker_id="speaker-1",
        raw_text="主要内容",
        normalized_text="主要内容",
        display_text="主要内容",
        confidence=1.0,
        speaker_scores=(
            SpeakerScore("speaker-1", 1.0),
            SpeakerScore("speaker-2", 0.0),
        ),
        speaker_margin=1.0,
        evidence={
            "overlap": {"fullTimelineInference": full_timeline}
        },
    )


def test_guarded_overlap_recovery_publishes_only_aligned_secondary_channel(
    tmp_path: Path,
) -> None:
    pipeline, asr = _pipeline()
    overlap = (
        OverlapDecision(
            window_id="source-1",
            overlapping=True,
            evidence={
                "detectorStatus": "EVALUATED",
                "overlapDetectorRun": True,
                "reviewStatus": "REVIEW_REQUIRED",
                "overlapIntervals": [
                    {
                        "startMs": 1000,
                        "endMs": 2000,
                        "localSpeakers": ["A", "B"],
                    }
                ],
            },
        ),
        OverlapDecision(
            window_id="source-2",
            overlapping=False,
            evidence={
                "detectorStatus": "EVALUATED",
                "overlapDetectorRun": True,
                "reviewStatus": "NOT_REQUIRED",
                "overlapIntervals": [],
            },
        ),
    )
    metrics = PipelineMetricsCollector(job_id="overlap-test", duration_ms=4000)
    context = AdapterContext(
        job_id="overlap-test",
        output_directory=tmp_path,
        cancellation=threading.Event(),
    )
    segments = pipeline._recover_overlap_speech(
        prepared=_prepared(),
        overlap=overlap,
        embeddings=(
            EmbeddingRecord("source-1", (1.0, 0.0)),
            EmbeddingRecord("source-2", (0.0, 1.0)),
        ),
        clusters=_clusters(),
        segments=(_primary_segment(),),
        requested_language="auto",
        context=context,
        metrics=metrics,
    )

    assert len(segments) == 2
    recovered = segments[1]
    assert asr.calls == [recovered.segment_id + ".source"]
    assert asr.token_budgets == [34]
    assert recovered.speaker_id == "speaker-2"
    assert recovered.raw_text == "次要内容"
    assert (recovered.start_ms, recovered.end_ms) == (1000, 2000)
    assert recovered.overlapping is True
    assert recovered.evidence["overlapRecovery"][
        "contextPublishedAsSpeech"
    ] is False
    assert recovered.evidence["overlapRecovery"]["asrMaxNewTokens"] == 34
    assert recovered.evidence["overlap"]["fullTimelineInference"] == (
        _primary_segment().evidence["overlap"]["fullTimelineInference"]
    )
    policy = metrics.as_dict()["policy"]
    assert policy["overlapRecoveryAsrCandidateCount"] == 1
    assert policy["overlapRecoveryAsrMinTokenBudgetApplied"] == 34
    assert policy["overlapRecoveryAsrMaxTokenBudgetApplied"] == 34
    timestamps = recovered.evidence["asr"]["timestamps"]
    assert timestamps == [
        {"text": "次", "startMs": 1200, "endMs": 1300},
        {"text": "要", "startMs": 1300, "endMs": 1400},
        {"text": "内容", "startMs": 1400, "endMs": 1600},
    ]
    candidates = pipeline._select_candidates(
        segments,
        _prepared().windows,
        _clusters(),
    )
    assert any(
        item.segment_id == recovered.segment_id
        and "OVERLAP" in item.reasons
        for item in candidates
    )


def test_mossformer_adapter_writes_two_hash_bound_full_length_channels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.wav"
    samples = np.linspace(-0.25, 0.25, 32_000, dtype=np.float32)
    sf.write(source, samples, 16_000, subtype="PCM_16")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    prepared = PreparedAudio(
        duration_ms=2000,
        source_fingerprint=fingerprint,
        normalization_profile="mono-16khz-f32-v1",
        windows=(SpeechWindow("source", 0, 2000),),
        stage_durations_ms={
            "decode": 0.0,
            "normalize": 0.0,
            "vad": 0.0,
            "boundary": 0.0,
        },
        audio_path=str(source),
    )
    model_path = tmp_path / "model"
    model_path.mkdir()

    class Separator:
        def __call__(self, mixture):
            import torch

            assert torch.is_inference_mode_enabled()
            return np.stack((mixture, -mixture), axis=0)

    adapter = LocalMossFormer2SeparationAdapter(
        model_path=model_path,
        separator_factory=lambda **_: Separator(),
    )
    pcm_buffer_id = production_runners._pcm_buffer_key(
        fingerprint,
        source,
    )
    production_runners._SHARED_PCM_STORE.put(
        pcm_buffer_id,
        samples,
        16_000,
    )
    monkeypatch.setattr(
        production_runners,
        "_load_audio",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("separator must reuse the shared PCM buffer")
        ),
    )
    context = AdapterContext(
        job_id="separator-test",
        output_directory=tmp_path / "output",
        cancellation=threading.Event(),
    )
    channels = adapter.separate_batch(
        prepared,
        (
            OverlapRecoveryInterval(
                interval_id="overlap-1",
                detected_start_ms=600,
                detected_end_ms=1400,
                context_start_ms=0,
                context_end_ms=2000,
                local_speakers=("A", "B"),
            ),
        ),
        context,
    )

    assert len(channels) == 2
    for index, channel in enumerate(channels, start=1):
        path = Path(channel.audio_path)
        assert channel.channel_index == index
        assert channel.duration_ms == 2000
        assert channel.evidence["pcmBufferId"] == pcm_buffer_id
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            channel.audio_sha256
        )
        output, sample_rate = sf.read(path, always_2d=False)
        assert sample_rate == 16_000
        assert output.shape == samples.shape
    production_runners._SHARED_PCM_STORE.clear()
