from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from llm_processor import LLMAnalysis, LLMProcessor
from speaker_semantic_arbiter import SpeakerSemanticArbiter
import transcriber as transcriber_module
from transcriber import Transcriber, TranscriptionSegment


def _build_llm_processor() -> LLMProcessor:
    cfg: Dict[str, Any] = {
        "translation": {
            "enabled": False,
            "target_language": "zh",
            "max_concurrent": 1,
            "skip_same_language": True,
        },
        "llm": {
            "enabled": True,
            "api_key": "test-key",
            "model": "qwen3.5-plus",
            "max_concurrent": 1,
            "provider_max_concurrent": 1,
            "timeout": 60,
            "max_retries": 0,
            "retry_delay": 0.1,
            "max_tokens": 1024,
            "hierarchical_summary": {
                "enabled": True,
                "chunk_target_chars": 8000,
                "single_pass_chars": 6000,
                "chunk_overlap_chars": 400,
                "reduce_target_chars": 10000,
                "max_chunks": 32,
            },
            "request_limits": {
                "segment_text_chars": 4000,
                "payload_warn_chars": 12000,
                "payload_hard_chars": 18000,
            },
            "speaker_arbitration": {"enabled": False},
            "optimize_language": {"enabled": False},
        },
    }
    return LLMProcessor(cfg)


def verify_llm_chunking() -> Dict[str, Any]:
    processor = _build_llm_processor()
    text = "\n".join(
        f"[S{i % 3}] " + ("这是一个较长的测试句子，用于验证LLM摘要切块策略。" * 18)
        for i in range(36)
    )
    chunks = processor._split_text_for_analysis(text)
    if len(chunks) <= 1:
        raise AssertionError(f"expected multiple chunks, got {len(chunks)}")
    if max(len(chunk) for chunk in chunks) > 6500:
        raise AssertionError(f"chunk too large: {max(len(chunk) for chunk in chunks)}")
    return {
        "chunk_count": len(chunks),
        "chunk_sizes": [len(chunk) for chunk in chunks],
    }


def verify_llm_reduce_grouping() -> Dict[str, Any]:
    processor = _build_llm_processor()
    analyses: List[LLMAnalysis] = []
    for idx in range(12):
        analyses.append(
            LLMAnalysis(
                summary=(f"chunk-{idx} summary " * 70).strip(),
                key_points=[f"point-{idx}-{n} " * 8 for n in range(6)],
                action_items=[f"action-{idx}-{n} " * 6 for n in range(4)],
                topics=[f"topic-{idx}-{n} " * 6 for n in range(4)],
            )
        )
    groups = processor._split_chunk_analyses_for_reduce(analyses)
    if len(groups) <= 1:
        raise AssertionError("expected grouped reduce inputs")
    if any(processor._estimate_reduce_input_chars(group) > 11000 for group in groups):
        raise AssertionError("reduce group exceeded expected budget")
    return {
        "group_count": len(groups),
        "group_sizes": [len(group) for group in groups],
        "group_chars": [processor._estimate_reduce_input_chars(group) for group in groups],
    }


def verify_llm_segment_splitting() -> Dict[str, Any]:
    processor = _build_llm_processor()
    calls: List[Dict[str, Any]] = []

    async def fake_call_api(messages, enable_thinking=None, request_label=""):
        calls.append(
            {
                "label": request_label,
                "chars": processor._messages_char_count(messages),
            }
        )
        return "ok"

    processor._call_api = fake_call_api  # type: ignore[method-assign]
    long_text = "这是一个超长分段测试。" * 900
    translated = asyncio.run(processor._translate_text(long_text, "zh", "en"))
    optimized = asyncio.run(
        processor._optimize_text(
            long_text,
            "zh",
            strict_keep_text=True,
            infer_missing_words=True,
        )
    )

    translate_calls = [item for item in calls if str(item["label"]).startswith("translate_segment")]
    optimize_calls = [item for item in calls if str(item["label"]) == "optimize_segment"]
    if len(translate_calls) <= 1:
        raise AssertionError(f"expected translation split, got {len(translate_calls)} call(s)")
    if len(optimize_calls) <= 1:
        raise AssertionError(f"expected optimization split, got {len(optimize_calls)} call(s)")
    if max(item["chars"] for item in calls) > processor.request_payload_hard_chars:
        raise AssertionError("segment request exceeded hard payload limit")
    return {
        "translate_calls": len(translate_calls),
        "optimize_calls": len(optimize_calls),
        "max_payload_chars": max(item["chars"] for item in calls),
        "translated_preview": translated[:20],
        "optimized_preview": optimized[:20],
    }


def verify_speaker_arbitration_windowing() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "llm": {
            "speaker_arbitration": {
                "enabled": True,
                "window_chars": 9000,
                "max_segments_per_window": 36,
                "window_overlap_segments": 4,
            }
        }
    }
    arbiter = SpeakerSemanticArbiter(
        cfg,
        llm_call=lambda *args, **kwargs: None,  # type: ignore[arg-type]
    )
    segments = [
        TranscriptionSegment(
            start=float(i * 2),
            end=float((i * 2) + 1),
            text=("这是一个用于speaker arbitration窗口分块的长句子。" * 10),
            speaker="A" if i % 2 == 0 else "B",
        )
        for i in range(60)
    ]
    windows = arbiter._build_windows(segments)
    if len(windows) <= 1:
        raise AssertionError("expected multiple speaker arbitration windows")
    rendered_chars = [
        sum(len(arbiter._window_line(item)) for item in window)
        for window in windows
    ]
    if any(len(window) > arbiter.max_segments_per_window for window in windows):
        raise AssertionError("speaker arbitration window exceeded segment cap")
    if any(chars > arbiter.window_chars + 400 for chars in rendered_chars):
        raise AssertionError("speaker arbitration window exceeded char budget unexpectedly")
    return {
        "window_count": len(windows),
        "window_sizes": [len(window) for window in windows],
        "window_chars": rendered_chars,
    }


def _build_segments() -> List[TranscriptionSegment]:
    return [
        TranscriptionSegment(start=0.0, end=2.0, text="alpha one", speaker="A"),
        TranscriptionSegment(start=2.0, end=4.0, text="beta two", speaker="B"),
        TranscriptionSegment(start=4.0, end=6.0, text="gamma three", speaker="A"),
    ]


def verify_pyannote_hybrid_probe_not_skipped() -> Dict[str, Any]:
    cfg = Config()
    transcriber = Transcriber(cfg)
    transcriber.has_mps = True

    audio_np = np.zeros(int(16000 * 900), dtype=np.float32)
    segments = _build_segments()
    probe_calls: Dict[str, Any] = {"count": 0}

    def fake_msdd_cfg() -> Dict[str, Any]:
        return {
            "enabled": True,
            "final_strategy": "hybrid",
            "num_speakers": 0,
            "min_speakers": 1,
            "max_speakers": 4,
            "pyannote_fallback": {
                "enabled": True,
                "device": "mps",
                "require_gpu": True,
            },
        }

    original_probe = transcriber_module.should_probe_pyannote_hybrid
    try:
        transcriber_module.should_probe_pyannote_hybrid = lambda *args, **kwargs: True
        transcriber.use_nemo_msdd_pipeline = lambda: True  # type: ignore[method-assign]
        transcriber._startup_preload_nemo_models_once = lambda *args, **kwargs: None  # type: ignore[method-assign]
        transcriber._nemo_msdd_cfg = fake_msdd_cfg  # type: ignore[method-assign]
        transcriber._nemo_sortformer_cfg = lambda *args, **kwargs: {"enabled": True}  # type: ignore[method-assign]
        transcriber._nemo_final_diarization_strategy = lambda *args, **kwargs: "hybrid"  # type: ignore[method-assign]
        transcriber._nemo_msdd_model_is_available = lambda *args, **kwargs: True  # type: ignore[method-assign]
        transcriber._diarize_audio_nemo_sortformer = lambda *args, **kwargs: [  # type: ignore[method-assign]
            {"start": 0.0, "end": 3.0, "speaker": "0"}
        ]
        transcriber._diarize_audio = lambda *args, **kwargs: [  # type: ignore[method-assign]
            {"start": 0.0, "end": 3.0, "speaker": "0"}
        ]
        transcriber._diarize_audio_pyannote_fallback = (  # type: ignore[method-assign]
            lambda *args, **kwargs: probe_calls.__setitem__("count", probe_calls["count"] + 1) or []
        )
        transcriber._posterior_fusion_decoder.decode = lambda *args, **kwargs: type(  # type: ignore[method-assign]
            "DecodeResult",
            (),
            {"segments": [], "overlap_seed_regions": [], "overlap_tracks": [], "route": ""},
        )()
        transcriber.assign_speakers(
            audio_np,
            16000,
            segments,
            file_name="verify-long-audio.mkv",
            map_speakers=True,
        )
    finally:
        transcriber_module.should_probe_pyannote_hybrid = original_probe

    if probe_calls["count"] != 1:
        raise AssertionError(f"expected pyannote probe to run once, got {probe_calls['count']}")
    return {"probe_calls": probe_calls["count"]}


def verify_pyannote_gpu_only_request() -> Dict[str, Any]:
    cfg = Config()
    transcriber = Transcriber(cfg)
    transcriber.has_mps = True
    recorder: Dict[str, Any] = {}
    original_sync_mps = transcriber_module._sync_mps
    original_empty_cache = transcriber_module.smart_empty_cache

    class FakeTurn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class FakeOutput:
        def itertracks(self, yield_label: bool = False):
            yield FakeTurn(0.0, 1.5), "track", "0"
            yield FakeTurn(1.5, 3.0), "track", "1"

    class FakePipeline:
        def to(self, device: Any):
            recorder["pipeline_device"] = str(device)
            return self

        def __call__(self, inputs: Dict[str, Any], **kwargs: Any):
            recorder["waveform_shape"] = tuple(inputs["waveform"].shape)
            recorder["infer_kwargs"] = dict(kwargs)
            return FakeOutput()

    def fake_move_waveform(waveform, device):
        recorder["waveform_device"] = device
        return waveform

    transcriber_module._sync_mps = lambda: None
    transcriber_module.smart_empty_cache = lambda *args, **kwargs: None
    transcriber._load_pyannote_pipeline_with_retry = lambda *args, **kwargs: FakePipeline()  # type: ignore[method-assign]
    transcriber._move_torch_waveform = fake_move_waveform  # type: ignore[method-assign]
    transcriber._refine_pyannote_diar_segments = lambda **kwargs: kwargs["diar_segments"]  # type: ignore[method-assign]
    transcriber._log_pyannote_pipeline_device_summary = lambda **kwargs: None  # type: ignore[method-assign]

    cfg_block = {
        "num_speakers": 0,
        "min_speakers": 1,
        "max_speakers": 4,
        "pyannote_fallback": {
            "enabled": True,
            "device": "mps",
            "require_gpu": True,
            "model_name": "pyannote/speaker-diarization-community-1",
        },
    }
    try:
        diar_segments = transcriber._diarize_audio_pyannote_fallback(
            audio_np=np.zeros(int(16000 * 4), dtype=np.float32),
            sample_rate=16000,
            segments=_build_segments(),
            cfg=cfg_block,
            file_name="verify-gpu-only.mkv",
        )
        if not diar_segments:
            raise AssertionError("expected fake pyannote diarization output")
        if "mps" not in str(recorder.get("pipeline_device", "")):
            raise AssertionError(f"pipeline device mismatch: {recorder.get('pipeline_device')}")
        if recorder.get("waveform_device") != "mps":
            raise AssertionError(f"waveform device mismatch: {recorder.get('waveform_device')}")

        transcriber_no_gpu = Transcriber(cfg)
        transcriber_no_gpu.has_mps = False
        skipped = transcriber_no_gpu._diarize_audio_pyannote_fallback(
            audio_np=np.zeros(int(16000 * 4), dtype=np.float32),
            sample_rate=16000,
            segments=_build_segments(),
            cfg=cfg_block,
            file_name="verify-gpu-missing.mkv",
        )
        if skipped:
            raise AssertionError("expected pyannote to skip when GPU is required but unavailable")
        if "gpu-required" not in str(transcriber_no_gpu._pyannote_diar_runtime_error).lower():
            raise AssertionError(transcriber_no_gpu._pyannote_diar_runtime_error)
    finally:
        transcriber_module._sync_mps = original_sync_mps
        transcriber_module.smart_empty_cache = original_empty_cache

    return {
        "pipeline_device": recorder.get("pipeline_device"),
        "waveform_device": recorder.get("waveform_device"),
        "segments": len(diar_segments),
        "gpu_missing_error": transcriber_no_gpu._pyannote_diar_runtime_error,
    }


def main() -> None:
    payload = {
        "llm_chunking": verify_llm_chunking(),
        "llm_reduce_grouping": verify_llm_reduce_grouping(),
        "llm_segment_splitting": verify_llm_segment_splitting(),
        "speaker_arbitration_windowing": verify_speaker_arbitration_windowing(),
        "pyannote_probe": verify_pyannote_hybrid_probe_not_skipped(),
        "pyannote_gpu_only": verify_pyannote_gpu_only_request(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
