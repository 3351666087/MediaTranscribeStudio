"""
transcriber.py - ASR engine wrapper with runtime fallback strategies.

Typical behavior on a GPU system:
  1. Try FunASR on CUDA (ASR + VAD, optionally punc/spk when VRAM is enough)
  2. Try faster-whisper on CUDA (auto downgrade model size if VRAM is limited)
  3. Fall back to FunASR on CPU
  4. Fall back to faster-whisper on CPU

Key goals:
  - Prefer successful transcription over strict engine affinity
  - Prefer native accelerators on each platform (CUDA / MPS / MLX)
  - Recover from accelerator/OOM conditions when possible
  - Keep cleanup and fallback behavior predictable
"""

import gc
import hashlib
import importlib
import inspect
import json
import collections
import logging
import math
import os
import re
import select
import signal
import socket
import subprocess
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import warnings
import wave
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple

try:
    import sitecustomize  # noqa: F401
except Exception:
    sitecustomize = None  # type: ignore[assignment]

import numpy as np
import torch

from config import DEFAULT_NGC_API_KEY
from diar_fusion import PosteriorFusionDecoder, select_hybrid_candidate, should_probe_pyannote_hybrid
from diar_fusion.utils import count_diar_speakers
from output_layout import resolve_runtime_artifact_path
from runtime_paths import (
    APP_ROOT,
    INTERNAL_ROOT,
    detect_download_route,
    find_tool_executable,
    resolve_app_writable_path,
    runtime_cache_root,
)
from utils import (
    get_torch_dtype,
    mlx_whisper_is_available,
    mlx_whisper_status,
    mps_is_available,
    mps_unavailable_reason,
    smart_empty_cache,
)

logger = logging.getLogger(__name__)

_SIGKILL_COMPAT_APPLIED = False
_HF_HUB_AUTH_TOKEN_COMPAT_APPLIED = False
_LIGHTNING_TORCH_LOAD_COMPAT_APPLIED = False
_NEMO_FRAME_VAD_STATE_DICT_COMPAT_APPLIED = False
_NEMO_MSDD_MPS_VIEW_COMPAT_APPLIED = False
_TORCH_LOAD_WEIGHTS_ONLY_COMPAT_APPLIED = False
_TORCH_TENSOR_FORMAT_COMPAT_APPLIED = False
_TORCH_TENSOR_FORMAT_ND_FALLBACK_LOGGED = False
_NEMO_SPEAKER_UTILS_COMPAT_APPLIED = False
_NEMO_SPEAKER_UTILS_COMPAT_FALLBACK_LOGGED = False
_SPEECHBRAIN_FETCH_COMPAT_APPLIED = False
_PYANNOTE_SPK_DIAR_PLDA_COMPAT_APPLIED = False
_PYANNOTE_SPK_DIAR_PLDA_COMPAT_USED_LOGGED = False
_PYANNOTE_MODEL_VERSION_COMPAT_APPLIED = False
_PYTORCH_LIGHTNING_MODEL_SUMMARY_COMPAT_APPLIED = False
_WINDOWS_SYMLINK_COPY_COMPAT_APPLIED = False
_WINDOWS_SYMLINK_COPY_COMPAT_USED_LOGGED = False
_INVALID_NEMO_ARTIFACT_WARNED: set[str] = set()


def _inspect_nemo_artifact(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists():
            return False, "missing"
        if not path.is_file():
            return False, "not a file"
        size = int(path.stat().st_size)
    except Exception as e:
        return False, str(e or "stat failed")

    if size <= 0:
        return False, "empty file"

    try:
        if tarfile.is_tarfile(path) or zipfile.is_zipfile(path):
            return True, ""
    except Exception:
        pass

    preview = b""
    try:
        with path.open("rb") as fh:
            preview = fh.read(160)
    except Exception:
        pass

    preview_text = preview.lstrip()[:80].decode("utf-8", errors="ignore").strip()
    if preview.lstrip().startswith((b"{", b"[")):
        return False, f"looks like JSON payload instead of a NeMo archive ({size} bytes)"
    if size < 4096:
        return False, f"file too small to be a NeMo archive ({size} bytes)"
    if preview_text:
        return False, f"unrecognized NeMo archive format ({size} bytes, head={preview_text[:48]!r})"
    return False, f"unrecognized NeMo archive format ({size} bytes)"


def _warn_invalid_nemo_artifact(path: Path, reason: str) -> None:
    key = str(path)
    if key in _INVALID_NEMO_ARTIFACT_WARNED:
        return
    _INVALID_NEMO_ARTIFACT_WARNED.add(key)
    logger.warning("Ignoring invalid NeMo artifact %s: %s", path, reason)


# GPU memory utilities

def _force_cuda_cleanup():
    """Force CUDA memory cleanup."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def _log_gpu_mem(tag: str = ""):
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - alloc
    logger.info(
        f"  GPU[{tag}]: alloc={alloc:.2f}GB, "
        f"reserved={reserved:.2f}GB, "
        f"free={free:.2f}GB / {total:.1f}GB"
    )


def _get_free_gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    return (total - allocated) / (1024**3)


def _get_total_gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _sync_mps():
    backend = getattr(torch, "mps", None)
    if backend is None:
        return
    try:
        backend.synchronize()
    except Exception:
        pass


def _resample_audio_linear_np(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Cheap linear resample used by fallback diarization when torchaudio is unavailable."""
    x = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    src = int(source_sr)
    dst = int(target_sr)
    if src <= 0 or dst <= 0 or src == dst:
        return x

    out_len = max(1, int(round(float(x.size) * float(dst) / float(src))))
    if out_len == x.size:
        return x
    src_idx = np.arange(x.size, dtype=np.float32)
    dst_idx = np.linspace(0.0, float(max(0, x.size - 1)), out_len, dtype=np.float32)
    return np.interp(dst_idx, src_idx, x).astype(np.float32, copy=False)


# Core data structures

class TranscriptionSegment:
    """"""

    __slots__ = [
        "start",
        "end",
        "text",
        "speaker",
        "language",
        "confidence",
        "words",
        "semantic_highlights",
        "arbitration_confidence",
        "speaker_role",
    ]

    def __init__(
        self,
        start=0.0,
        end=0.0,
        text="",
        speaker="UNKNOWN",
        language="",
        confidence=0.0,
        words=None,
        semantic_highlights=None,
        arbitration_confidence=0.0,
        speaker_role="",
        **kwargs,
    ):
        self.start = float(start)
        self.end = float(end)
        self.text = str(text)
        self.speaker = str(speaker)
        self.language = str(language) if language else ""
        self.confidence = float(confidence) if confidence else 0.0
        self.words = self._normalize_words(words)
        self.semantic_highlights = self._normalize_highlights(
            semantic_highlights if semantic_highlights is not None else kwargs.get("semantic_highlights")
        )
        try:
            self.arbitration_confidence = float(
                arbitration_confidence
                if arbitration_confidence is not None
                else kwargs.get("arbitration_confidence", 0.0)
            )
        except Exception:
            self.arbitration_confidence = 0.0
        self.speaker_role = str(
            speaker_role if speaker_role is not None else kwargs.get("speaker_role", "")
        ).strip()

    @staticmethod
    def _normalize_words(words: Any) -> List[Dict[str, Any]]:
        if not isinstance(words, (list, tuple)):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in words:
            if isinstance(item, dict):
                text = str(item.get("text", item.get("word", "")) or "")
                start = item.get("start")
                end = item.get("end")
                probability = item.get("probability", item.get("prob"))
            else:
                text = str(
                    getattr(item, "text", getattr(item, "word", "")) or ""
                )
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
                probability = getattr(item, "probability", getattr(item, "prob", None))

            if not text.strip():
                continue

            entry: Dict[str, Any] = {"text": text}
            try:
                if start is not None:
                    entry["start"] = float(start)
            except Exception:
                pass
            try:
                if end is not None:
                    entry["end"] = float(end)
            except Exception:
                pass
            try:
                if probability is not None:
                    entry["probability"] = float(probability)
            except Exception:
                pass
            speaker = item.get("speaker") if isinstance(item, dict) else getattr(item, "speaker", None)
            if speaker is not None:
                speaker_text = str(speaker or "").strip()
                if speaker_text:
                    entry["speaker"] = speaker_text
            speaker_confidence = (
                item.get("speaker_confidence")
                if isinstance(item, dict)
                else getattr(item, "speaker_confidence", None)
            )
            try:
                if speaker_confidence is not None:
                    entry["speaker_confidence"] = float(speaker_confidence)
            except Exception:
                pass
            overlap_speakers = (
                item.get("overlap_speakers")
                if isinstance(item, dict)
                else getattr(item, "overlap_speakers", None)
            )
            if isinstance(overlap_speakers, (list, tuple)):
                normalized_overlap = [
                    str(value or "").strip()
                    for value in overlap_speakers
                    if str(value or "").strip()
                ]
                if normalized_overlap:
                    entry["overlap_speakers"] = normalized_overlap
            speaker_candidates = (
                item.get("speaker_candidates")
                if isinstance(item, dict)
                else getattr(item, "speaker_candidates", None)
            )
            if isinstance(speaker_candidates, (list, tuple)):
                normalized_candidates: List[Dict[str, Any]] = []
                for candidate in speaker_candidates:
                    if not isinstance(candidate, dict):
                        continue
                    candidate_speaker = str(candidate.get("speaker", "") or "").strip()
                    if not candidate_speaker:
                        continue
                    payload: Dict[str, Any] = {"speaker": candidate_speaker}
                    try:
                        payload["confidence"] = float(candidate.get("confidence", 0.0) or 0.0)
                    except Exception:
                        payload["confidence"] = 0.0
                    sources = candidate.get("sources")
                    if isinstance(sources, (list, tuple)):
                        payload["sources"] = [
                            str(source or "").strip()
                            for source in sources
                            if str(source or "").strip()
                        ]
                    normalized_candidates.append(payload)
                if normalized_candidates:
                    entry["speaker_candidates"] = normalized_candidates
            normalized.append(entry)

        return normalized

    @staticmethod
    def _normalize_highlights(values: Any) -> List[str]:
        if not isinstance(values, (list, tuple)):
            return []
        normalized: List[str] = []
        seen: set[str] = set()
        for item in values:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    def to_dict(self) -> dict:
        payload = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker": self.speaker,
            "language": self.language,
            "confidence": round(self.confidence, 3),
        }
        if self.words:
            payload["words"] = [dict(item) for item in self.words]
        if self.semantic_highlights:
            payload["semantic_highlights"] = list(self.semantic_highlights)
        if self.arbitration_confidence > 0:
            payload["arbitration_confidence"] = round(self.arbitration_confidence, 3)
        if self.speaker_role:
            payload["speaker_role"] = self.speaker_role
        return payload


# Main transcriber implementation

class Transcriber:
    """ASR runtime wrapper with platform-aware accelerator selection and fallbacks."""

    def __init__(
        self,
        config,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.config = config
        self.asr_cfg = config["asr"]
        self.perf_cfg = config["performance"]
        self._progress_callback = progress_callback
        self._runtime_temp_dir: Optional[Path] = None

        self.engine_name = self.asr_cfg["engine"]
        self.is_macos = sys.platform == "darwin"
        self.has_cuda = bool(torch.cuda.is_available() and not self.is_macos)
        self.has_mps = bool(self.is_macos and mps_is_available())
        mlx_whisper_ok, mlx_whisper_detail = mlx_whisper_status()
        self.has_mlx_whisper = bool(self.is_macos and mlx_whisper_ok)
        self._mlx_install_detail = str(mlx_whisper_detail or "").strip()
        if self.is_macos and not self.has_mps:
            reason = mps_unavailable_reason()
            if reason:
                logger.info(f"PyTorch MPS unavailable on this host: {reason}")
        if self.is_macos:
            logger.info(
                "MLX Whisper detection: available=%s%s",
                self.has_mlx_whisper,
                f", detail={self._mlx_install_detail}" if self._mlx_install_detail else "",
            )
        if self.has_cuda:
            self.device_str = "cuda:0"
        elif self.has_mps:
            self.device_str = "mps"
        else:
            self.device_str = "cpu"
        self.dtype = get_torch_dtype(self.perf_cfg["dtype"])

        self.speaker_labels = config["output"].get(
            "speaker_labels", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

        self._total_gpu_gb = _get_total_gpu_gb()

        # Engine/model handles
        self._funasr_model = None
        self._funasr_device = None  # FunASR model device
        self._funasr_has_punc = False
        self._funasr_has_spk = False
        self._faster_whisper_model = None
        self._fw_model_size = ""
        self._mlx_whisper_module = None
        self._mlx_model_ref = ""
        self._mlx_runtime_probe_ok: Optional[bool] = None
        self._mlx_runtime_probe_error = ""
        self._mlx_worker_proc = None
        self._mlx_worker_model_ref = ""
        self._mlx_worker_lock = threading.Lock()
        self._mlx_worker_stderr_thread: Optional[threading.Thread] = None
        self._mlx_worker_stderr_tail: collections.deque[str] = collections.deque(
            maxlen=80
        )
        self._mlx_disabled_for_session = False
        self._mlx_disabled_reason = ""
        self._fw_cfg = None
        self._fw_device = None
        self._fw_startup_preload_thread: Optional[threading.Thread] = None
        self._fw_startup_preload_done = False
        self._fw_startup_preload_error = ""
        self._engine_ready = False
        self._current_engine = None

        # OOM recovery state
        self._oom_retries = 0
        self._max_oom_retries = 3
        self._model_cache_status: Dict[str, bool] = {}
        self._nemo_model_cache_status: Dict[str, bool] = {}
        self._nemo_model_cache_paths: Dict[str, str] = {}
        self._nemo_startup_preload_attempted = False
        self._nemo_startup_preload_summary = ""
        self._lang_probe_model = None
        self._lang_probe_model_size = ""
        self._lang_probe_init_failed = False
        self._lang_probe_init_error = ""
        self._nemo_msdd_disabled = False
        self._nemo_msdd_disabled_reason = ""
        self._nemo_msdd_runtime_error = ""
        self._nemo_msdd_model_unavailable = False
        self._nemo_msdd_model_unavailable_reason = ""
        self.last_speaker_languages: Dict[str, str] = {}
        self.last_diarization_route = ""
        self.last_detected_speaker_ids: List[str] = []
        self.last_detected_speaker_count = 0
        self.last_language_probe: Dict[str, Any] = {
            "language": "",
            "probability": 0.0,
            "source": "",
            "offset_sec": 0.0,
        }
        self._pyannote_diar_pipeline = None
        self._pyannote_diar_runtime_error = ""
        self._pyannote_diar_setup: Dict[str, Any] = {}
        self._pyannote_incompatible_models: set[str] = set()
        self._speaker_embedding_accel_status: Dict[str, bool] = {}
        self._nemo_sortformer_model = None
        self._nemo_sortformer_setup: Dict[str, Any] = {}
        self._nemo_sortformer_runtime_error = ""
        self._nemo_sortformer_incompatible = False
        self._nemo_sortformer_incompatible_reason = ""
        self._overlap_osd_pipeline = None
        self._overlap_osd_runtime_error = ""
        self._overlap_osd_setup: Dict[str, Any] = {}
        self._overlap_separator_model = None
        self._overlap_separator_setup: Dict[str, Any] = {}
        self._overlap_separator_runtime_error = ""
        self._pyannote_separation_blocked = False
        self._pyannote_separation_block_reason = ""
        self._posterior_fusion_decoder = PosteriorFusionDecoder(logger=logger)
        self._aggressive_cuda_cleanup = bool(
            self.perf_cfg.get("aggressive_cuda_cleanup", False)
        )
        self._force_cuda_cleanup_every = max(
            1,
            self._safe_int(self.perf_cfg.get("force_cuda_cleanup_every", 8), 8),
        )
        self._transcribe_calls_since_cleanup = 0
        self._hf_endpoint_probe_cache: Dict[Tuple[str, str], Tuple[Tuple[str, str], str]] = {}
        self._torchaudio_fallback_blocked = False
        self._torchaudio_fallback_reason = ""
        self._lang_probe_lock = threading.Lock()

        self._configure_runtime_warning_filters()
        self._ensure_diarization_dependency_compat()
        self._set_default_hf_endpoint()
        self._init_engine()
        self._start_faster_whisper_startup_preload()

    # Engine/model unload helpers

    def _init_engine(self):
        """Initialize ASR engine with fallback strategies."""
        errors = []

        engine_family = self._engine_family(self.engine_name)
        if engine_family == "funasr":
            strategies = self._build_funasr_strategies()
        elif engine_family == "faster_whisper":
            strategies = self._build_whisper_family_strategies()
        else:
            # auto
            strategies = self._build_auto_engine_strategies()

        for name, init_fn in strategies:
            if self._engine_ready:
                break

            logger.info(f"Trying engine: {name}...")
            _force_cuda_cleanup()
            _log_gpu_mem("before-init")

            ok, err = init_fn()
            if ok:
                self._engine_ready = True
                _log_gpu_mem("after-init")
                logger.info(f"Engine ready: {name}")
                break
            else:
                errors.append(f"{name}: {err}")
                logger.warning(f"?{name}: {err}")
                self._unload_all()
                _force_cuda_cleanup()

        if not self._engine_ready:
            detail = "\n".join(f"  - {e}" for e in errors)
            raise RuntimeError(
                f"No ASR engine initialized.\n{detail}\n\n"
                f"Fixes:\n"
                f"  1. Ensure model downloads can reach Hugging Face / NGC\n"
                f"  2. pip install funasr modelscope\n"
                f"  3. pip install faster-whisper\n"
            )

    def _language_probe_candidate_model_sizes(self) -> List[str]:
        fw_cfg = self.asr_cfg.get(
            "faster_whisper", self.asr_cfg.get("whisperx", {})
        ) or {}
        preferred: List[str] = []
        cfg_model_size = str(fw_cfg.get("model_size", "") or "").strip().lower()
        if cfg_model_size:
            preferred.append(cfg_model_size)
        preferred.extend(["small", "base", "tiny", "medium", "large-v3", "large-v2"])
        return self._dedupe_preserve_order(
            [item for item in preferred if str(item or "").strip()]
        )

    @staticmethod
    def _faster_whisper_warmup_audio(
        sample_rate: int = 16000,
        duration_ms: int = 500,
    ) -> np.ndarray:
        total_samples = max(
            sample_rate // 2,
            int(sample_rate * max(100, int(duration_ms)) / 1000),
        )
        return np.zeros((total_samples,), dtype=np.float32)

    def _warmup_faster_whisper_model(self, model: Any, *, label: str) -> None:
        warm_audio = self._faster_whisper_warmup_audio()
        segments_gen, _info = model.transcribe(
            warm_audio,
            beam_size=1,
            best_of=1,
            patience=1.0,
            temperature=0.0,
            vad_filter=False,
            word_timestamps=False,
            without_timestamps=True,
        )
        for _ in segments_gen:
            pass
        logger.debug("faster-whisper warmup ready: %s", label)

    def _should_preload_language_probe_in_background(self) -> bool:
        if self._faster_whisper_model is not None:
            return False
        if self._lang_probe_model is not None or self._lang_probe_init_failed:
            return False
        lang_cfg = self.config.get("language", {}) or {}
        if not bool(lang_cfg.get("auto_detect", True)):
            return False
        return (
            self._choose_cached_fw_model(self._language_probe_candidate_model_sizes())
            is not None
        )

    def _run_faster_whisper_startup_preload(self) -> None:
        ready_items: List[str] = []
        try:
            if self._faster_whisper_model is not None:
                current_label = self._fw_model_size or self._fw_device or "current"
                self._warmup_faster_whisper_model(
                    self._faster_whisper_model,
                    label=current_label,
                )
                ready_items.append(f"current:{current_label}")

            if self._should_preload_language_probe_in_background():
                probe_model = self._get_language_probe_model(allow_download=False)
                if probe_model is not None:
                    probe_label = self._lang_probe_model_size or "cached"
                    self._warmup_faster_whisper_model(
                        probe_model,
                        label=f"probe:{probe_label}",
                    )
                    ready_items.append(f"probe:{probe_label}")
        except Exception as e:
            self._fw_startup_preload_error = str(e)[:200]
            logger.debug("faster-whisper startup preload skipped: %s", e)
        finally:
            self._fw_startup_preload_done = True
            if ready_items:
                logger.info(
                    "faster-whisper startup preload ready: %s",
                    ", ".join(ready_items),
                )

    def _start_faster_whisper_startup_preload(self) -> None:
        if not self.is_macos:
            return

        wants_current = self._faster_whisper_model is not None
        wants_probe = self._should_preload_language_probe_in_background()
        if not wants_current and not wants_probe:
            return

        thread = self._fw_startup_preload_thread
        if thread is not None and thread.is_alive():
            return

        self._fw_startup_preload_done = False
        self._fw_startup_preload_error = ""
        self._fw_startup_preload_thread = threading.Thread(
            target=self._run_faster_whisper_startup_preload,
            name="mts-fw-preload",
            daemon=True,
        )
        self._fw_startup_preload_thread.start()

    def _wait_for_faster_whisper_startup_preload(self, reason: str) -> None:
        thread = self._fw_startup_preload_thread
        if thread is None or not thread.is_alive():
            return
        if threading.current_thread() is thread:
            return
        logger.debug("Waiting for faster-whisper startup preload (%s)...", reason)
        thread.join()

    # FunASR CUDA initialization (VRAM-aware feature loading)

    @staticmethod
    def _normalize_language_tag(language: str) -> str:
        lang = (language or "").strip().lower().replace("_", "-")
        if not lang or lang == "auto":
            return ""
        alias = {
            "cn": "zh",
            "zh-cn": "zh",
            "zh-sg": "zh",
            "zh-hans": "zh",
            "zh-hant": "zh",
            "zh-tw": "zh",
            "cmn": "zh",
            "cantonese": "yue",
        }
        lang = alias.get(lang, lang)
        if "-" in lang:
            lang = lang.split("-", 1)[0]
        return lang

    def _funasr_model_name(self) -> str:
        cfg = self.asr_cfg.get("funasr", {}) or {}
        return str(cfg.get("model", "") or "").lower()

    def _funasr_trust_remote_code(self) -> bool:
        cfg = self.asr_cfg.get("funasr", {}) or {}
        return self._safe_bool(cfg.get("trust_remote_code", True), True)

    def _create_funasr_model_compat(self, auto_model_cls: Any, kwargs: Dict[str, Any]) -> Any:
        init_kwargs = dict(kwargs or {})
        init_kwargs.setdefault("trust_remote_code", self._funasr_trust_remote_code())
        try:
            return auto_model_cls(**init_kwargs)
        except Exception as e:
            if self._is_unexpected_kwarg_error(e, "trust_remote_code"):
                retry_kwargs = dict(init_kwargs)
                retry_kwargs.pop("trust_remote_code", None)
                return auto_model_cls(**retry_kwargs)
            raise

    @staticmethod
    def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
        result: List[Any] = []
        seen: set[Any] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _engine_family(engine_name: Optional[str]) -> str:
        value = str(engine_name or "").strip().lower().replace("-", "_")
        if value in {"mlx_whisper", "faster_whisper", "whisperx"}:
            return "faster_whisper"
        if value == "funasr":
            return "funasr"
        return value or "auto"

    @staticmethod
    def _strategy_label(name: str) -> str:
        mapping = {
            "funasr_cuda": "FunASR CUDA",
            "funasr_mps": "FunASR MPS",
            "funasr_cpu": "FunASR CPU",
            "faster_whisper_cuda": "faster-whisper CUDA",
            "faster_whisper_cpu": "faster-whisper CPU",
            "mlx_whisper": "MLX Whisper",
        }
        return mapping.get(name, name)

    @staticmethod
    def _normalize_device_pref(device_pref: Any) -> str:
        value = str(device_pref or "auto").strip().lower().replace("-", "_")
        if not value:
            return "auto"
        if value.startswith("cuda"):
            return "cuda"
        aliases = {
            "gpu": "auto",
            "apple": "mps",
            "metal": "mps",
            "mps_0": "mps",
            "mlx_whisper": "mlx",
        }
        return aliases.get(value, value)

    def _torch_device_candidates(self, device_pref: Any, *, allow_mps: bool = True) -> List[str]:
        pref = self._normalize_device_pref(device_pref)
        candidates: List[str] = []
        if pref == "auto":
            if self.has_cuda:
                candidates.append("cuda")
            if allow_mps and self.has_mps:
                candidates.append("mps")
            candidates.append("cpu")
            return self._dedupe_preserve_order(candidates)
        if pref == "cuda":
            if self.has_cuda:
                candidates.append("cuda")
            candidates.append("cpu")
            return self._dedupe_preserve_order(candidates)
        if pref == "mps":
            if allow_mps and self.has_mps:
                candidates.append("mps")
            candidates.append("cpu")
            return self._dedupe_preserve_order(candidates)
        if pref == "cpu":
            return ["cpu"]
        return self._torch_device_candidates("auto", allow_mps=allow_mps)

    def _preferred_torch_device(self, device_pref: Any, *, allow_mps: bool = True) -> str:
        candidates = self._torch_device_candidates(device_pref, allow_mps=allow_mps)
        return candidates[0] if candidates else "cpu"

    @staticmethod
    def _python_subprocess_executable() -> str:
        candidates = [
            str(getattr(sys, "_base_executable", "") or "").strip(),
            str(sys.executable or "").strip(),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            name = Path(candidate).name.lower()
            if "python" in name:
                return candidate
        return candidates[-1] or "python3"

    def _build_funasr_strategies(self) -> List[Tuple[str, Callable[[], tuple]]]:
        cfg = self.asr_cfg.get("funasr", {}) or {}
        strategies: List[Tuple[str, Callable[[], tuple]]] = []
        for device in self._torch_device_candidates(cfg.get("device", "auto"), allow_mps=True):
            if device == "cuda":
                strategies.append(("funasr_cuda", self._try_init_funasr_cuda))
            elif device == "mps":
                strategies.append(("funasr_mps", self._try_init_funasr_mps))
            else:
                strategies.append(("funasr_cpu", self._try_init_funasr_cpu))
        return self._dedupe_preserve_order(strategies)

    def _build_whisper_family_strategies(self) -> List[Tuple[str, Callable[[], tuple]]]:
        cfg = self.asr_cfg.get("faster_whisper", self.asr_cfg.get("whisperx", {})) or {}
        pref = self._normalize_device_pref(cfg.get("device", "auto"))
        strategies: List[Tuple[str, Callable[[], tuple]]] = []
        explicit_mlx_engine = self._normalize_device_pref(self.engine_name) == "mlx"
        prefer_mlx = (
            explicit_mlx_engine
            or pref == "mlx"
            or self._safe_bool(cfg.get("prefer_mlx", False), False)
        )
        if self.is_macos and self.has_mlx_whisper and prefer_mlx:
            strategies.append(("mlx_whisper", self._try_init_mlx_whisper))
        for device in self._torch_device_candidates(pref, allow_mps=False):
            if device == "cuda":
                strategies.append(("faster_whisper_cuda", self._try_init_faster_whisper_cuda))
            else:
                strategies.append(("faster_whisper_cpu", self._try_init_faster_whisper_cpu))
        return self._dedupe_preserve_order(strategies)

    def _resolve_ffmpeg_binary(self) -> str:
        configured = str(self.config.get("audio.ffmpeg_path", "") or "").strip()
        resolved = find_tool_executable("ffmpeg", configured=configured)
        if resolved:
            return resolved
        try:
            import imageio_ffmpeg

            resolved = imageio_ffmpeg.get_ffmpeg_exe()
            if resolved:
                return str(resolved)
        except Exception:
            pass
        return ""

    def _mlx_subprocess_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        ffmpeg_bin = self._resolve_ffmpeg_binary()
        if ffmpeg_bin:
            env["FFMPEG_BINARY"] = ffmpeg_bin
            ffmpeg_dir = str(Path(ffmpeg_bin).parent)
            current_path = str(env.get("PATH", "") or "")
            if ffmpeg_dir and ffmpeg_dir not in current_path.split(os.pathsep):
                env["PATH"] = ffmpeg_dir + (os.pathsep + current_path if current_path else "")
        return env

    def _mlx_failure_reason(self) -> str:
        return (
            str(self._mlx_disabled_reason or "").strip()
            or str(self._mlx_runtime_probe_error or "").strip()
        )

    def _disable_mlx_for_session(self, reason: str) -> None:
        message = str(reason or "MLX disabled for this session").strip()
        self._mlx_disabled_for_session = True
        self._mlx_disabled_reason = message
        self._mlx_runtime_probe_ok = False
        self._mlx_runtime_probe_error = message
        self._stop_mlx_worker()

    def _resolve_effective_mlx_model_ref(
        self,
        model_ref: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> str:
        resolved = str(model_ref or "").strip()
        if not resolved:
            return resolved

        fw_cfg = cfg or self.asr_cfg.get("faster_whisper", {}) or {}
        if self._safe_bool(fw_cfg.get("allow_full_large_v3_mlx", False), False):
            return resolved

        lowered = resolved.lower()
        if lowered not in {
            "mlx-community/whisper-large-v3-mlx",
            "mlx-community/whisper-large-v3",
        }:
            return resolved

        safer = "mlx-community/whisper-large-v3-turbo"
        if safer.lower() != lowered:
            logger.warning(
                "MLX model %s is prone to severe stalls on some Apple runtimes; "
                "using %s instead. Set faster_whisper.allow_full_large_v3_mlx=true "
                "to keep the original MLX repo.",
                resolved,
                safer,
            )
        return safer

    def _mlx_worker_ready_timeout_sec(self) -> float:
        return 20.0

    def _mlx_transcribe_timeout_sec(
        self,
        audio_duration_sec: float,
        *,
        cfg: Optional[Dict[str, Any]] = None,
        word_timestamps: bool = False,
    ) -> float:
        fw_cfg = cfg or self.asr_cfg.get("faster_whisper", {}) or {}
        duration = max(0.0, float(audio_duration_sec or 0.0))
        base_sec = max(15.0, self._safe_float(fw_cfg.get("mlx_timeout_base_sec", 25.0), 25.0))
        min_sec = max(30.0, self._safe_float(fw_cfg.get("mlx_timeout_min_sec", 45.0), 45.0))
        timeout_rtf = max(1.2, self._safe_float(fw_cfg.get("mlx_timeout_rtf", 2.8), 2.8))
        max_sec = max(min_sec, self._safe_float(fw_cfg.get("mlx_timeout_max_sec", 900.0), 900.0))
        if word_timestamps:
            timeout_rtf += 0.8
        return min(max_sec, max(min_sec, base_sec + duration * timeout_rtf))

    @staticmethod
    def _mlx_worker_script() -> str:
        return (
            "import inspect\n"
            "import json\n"
            "import sys\n"
            "import traceback\n"
            "import numpy as np\n"
            "import mlx_whisper\n"
            "def _emit(payload):\n"
            "    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "transcribe_fn = getattr(mlx_whisper, 'transcribe')\n"
            "params = set(inspect.signature(transcribe_fn).parameters.keys())\n"
            "_emit({'event': 'ready'})\n"
            "for raw in sys.stdin:\n"
            "    line = str(raw or '').strip()\n"
            "    if not line:\n"
            "        continue\n"
            "    try:\n"
            "        request = json.loads(line)\n"
            "        if str(request.get('cmd', 'transcribe')) == 'quit':\n"
            "            _emit({'ok': True, 'event': 'bye'})\n"
            "            break\n"
            "        audio_format = str(request.get('audio_format', 'path') or 'path')\n"
            "        audio_path = str(request.get('audio_path', '') or '')\n"
            "        if audio_format == 'npy':\n"
            "            audio_input = np.load(audio_path, allow_pickle=False)\n"
            "        else:\n"
            "            audio_input = audio_path\n"
            "        kwargs = {}\n"
            "        model_ref = str(request.get('model_ref', '') or '')\n"
            "        if model_ref:\n"
            "            if 'path_or_hf_repo' in params:\n"
            "                kwargs['path_or_hf_repo'] = model_ref\n"
            "            elif 'path_or_repo' in params:\n"
            "                kwargs['path_or_repo'] = model_ref\n"
            "            elif 'model' in params:\n"
            "                kwargs['model'] = model_ref\n"
            "        if 'word_timestamps' in params:\n"
            "            kwargs['word_timestamps'] = bool(request.get('word_timestamps', False))\n"
            "        language = str(request.get('language', '') or '').strip()\n"
            "        if language and 'language' in params:\n"
            "            kwargs['language'] = language\n"
            "        try:\n"
            "            result = transcribe_fn(audio_input, **kwargs)\n"
            "        except TypeError:\n"
            "            kwargs.pop('language', None)\n"
            "            result = transcribe_fn(audio_input, **kwargs)\n"
            "        if hasattr(result, 'to_dict'):\n"
            "            result = result.to_dict()\n"
            "        _emit({'ok': True, 'result': result})\n"
            "    except Exception as exc:\n"
            "        _emit({'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:400]}', "
            "'traceback': traceback.format_exc(limit=6)[-1600:]})\n"
        )

    def _drain_mlx_worker_stderr(self, pipe: Any) -> None:
        try:
            while True:
                line = pipe.readline()
                if line == "":
                    break
                text = str(line or "").strip()
                if text:
                    self._mlx_worker_stderr_tail.append(text)
        except Exception:
            pass

    @staticmethod
    def _read_subprocess_line(stream: Any, timeout_sec: float) -> Optional[str]:
        if stream is None:
            return None
        timeout = max(0.05, float(timeout_sec or 0.0))
        end_time = time.monotonic() + timeout
        fd = stream.fileno()
        while True:
            remaining = end_time - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], min(0.20, remaining))
            if not ready:
                continue
            line = stream.readline()
            if line == "":
                return ""
            return line

    def _mlx_worker_stderr_summary(self) -> str:
        if not self._mlx_worker_stderr_tail:
            return ""
        return self._summarize_subprocess_error(
            " | ".join(list(self._mlx_worker_stderr_tail)[-8:])
        )

    def _stop_mlx_worker(self) -> None:
        proc = self._mlx_worker_proc
        self._mlx_worker_proc = None
        self._mlx_worker_model_ref = ""
        self._mlx_worker_stderr_thread = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except Exception:
                    pass

    def _ensure_mlx_worker(self, model_ref: str) -> Any:
        proc = self._mlx_worker_proc
        if (
            proc is not None
            and proc.poll() is None
            and str(self._mlx_worker_model_ref or "") == str(model_ref or "")
        ):
            return proc

        self._stop_mlx_worker()
        self._mlx_worker_stderr_tail = collections.deque(maxlen=80)
        proc = subprocess.Popen(
            [
                self._python_subprocess_executable(),
                "-u",
                "-c",
                self._mlx_worker_script(),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._mlx_subprocess_env(),
            bufsize=1,
        )
        self._mlx_worker_proc = proc
        self._mlx_worker_model_ref = str(model_ref or "")
        if proc.stderr is not None:
            self._mlx_worker_stderr_thread = threading.Thread(
                target=self._drain_mlx_worker_stderr,
                args=(proc.stderr,),
                name="mts-mlx-stderr",
                daemon=True,
            )
            self._mlx_worker_stderr_thread.start()

        line = self._read_subprocess_line(
            proc.stdout,
            timeout_sec=self._mlx_worker_ready_timeout_sec(),
        )
        if not line:
            detail = self._mlx_worker_stderr_summary() or f"exit={proc.poll()}"
            self._stop_mlx_worker()
            raise RuntimeError(f"mlx-whisper worker bootstrap failed: {detail}")
        try:
            payload = json.loads(line)
        except Exception:
            payload = {}
        if str(payload.get("event", "") or "") != "ready":
            detail = self._mlx_worker_stderr_summary() or str(line).strip()[:240]
            self._stop_mlx_worker()
            raise RuntimeError(f"mlx-whisper worker bootstrap failed: {detail}")
        return proc

    def _mlx_worker_request(
        self,
        *,
        model_ref: str,
        request: Dict[str, Any],
        timeout_sec: float,
    ) -> Any:
        with self._mlx_worker_lock:
            proc = self._ensure_mlx_worker(model_ref)
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("mlx-whisper worker pipe is unavailable")
            try:
                proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except Exception as exc:
                self._disable_mlx_for_session(
                    f"mlx-whisper worker request write failed: {str(exc)[:180]}"
                )
                raise RuntimeError(self._mlx_failure_reason()) from exc

            line = self._read_subprocess_line(proc.stdout, timeout_sec=timeout_sec)
            if line is None:
                detail = (
                    self._mlx_worker_stderr_summary()
                    or f"timed out after {timeout_sec:.1f}s"
                )
                self._disable_mlx_for_session(
                    f"mlx-whisper worker timed out: {detail}"
                )
                raise RuntimeError(self._mlx_failure_reason())
            if line == "":
                detail = self._mlx_worker_stderr_summary() or f"exit={proc.poll()}"
                self._disable_mlx_for_session(
                    f"mlx-whisper worker exited unexpectedly: {detail}"
                )
                raise RuntimeError(self._mlx_failure_reason())

            try:
                payload = json.loads(line)
            except Exception as exc:
                detail = str(line).strip()[:240] or self._mlx_worker_stderr_summary()
                self._disable_mlx_for_session(
                    f"mlx-whisper worker returned invalid JSON: {detail}"
                )
                raise RuntimeError(self._mlx_failure_reason()) from exc

            if not bool(payload.get("ok", False)):
                detail = (
                    str(payload.get("error") or "").strip()
                    or str(payload.get("traceback") or "").strip()
                    or self._mlx_worker_stderr_summary()
                    or "unknown worker error"
                )
                self._disable_mlx_for_session(f"mlx-whisper worker failed: {detail}")
                raise RuntimeError(self._mlx_failure_reason())
            return payload.get("result")

    @staticmethod
    def _summarize_subprocess_error(detail: str, *, returncode: int = 0) -> str:
        text = " ".join(str(detail or "").split()).strip()
        if "NSRangeException" in text:
            return "MLX runtime crashed during import (NSRangeException in Metal device init)"
        if "NSException" in text:
            return "MLX runtime crashed during import (NSException)"
        if text:
            return text[-240:]
        return f"exit={returncode}"

    def _probe_mlx_whisper_runtime(self) -> tuple:
        if self._mlx_disabled_for_session:
            return False, self._mlx_failure_reason()
        if self._mlx_runtime_probe_ok is not None:
            return self._mlx_runtime_probe_ok, self._mlx_runtime_probe_error

        runner = (
            "import inspect\n"
            "import mlx_whisper\n"
            "print(inspect.getfile(mlx_whisper))\n"
        )
        try:
            proc = subprocess.run(
                [
                    self._python_subprocess_executable(),
                    "-c",
                    runner,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._mlx_subprocess_env(),
            )
        except Exception as exc:
            self._mlx_runtime_probe_ok = False
            self._mlx_runtime_probe_error = str(exc)[:240]
            return False, self._mlx_runtime_probe_error

        if proc.returncode == 0:
            self._mlx_runtime_probe_ok = True
            self._mlx_runtime_probe_error = ""
            return True, ""

        detail = (proc.stderr or proc.stdout or "").strip()
        self._mlx_runtime_probe_ok = False
        self._mlx_runtime_probe_error = self._summarize_subprocess_error(
            detail,
            returncode=proc.returncode,
        )
        return False, self._mlx_runtime_probe_error

    def _resolve_mlx_whisper_model_ref(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        cfg = cfg or self.asr_cfg.get("faster_whisper", {}) or {}
        explicit_repo = str(cfg.get("mlx_model_repo", "") or "").strip()
        if explicit_repo:
            return explicit_repo

        model_ref = str(cfg.get("model_size", "small") or "small").strip()
        if "/" in model_ref or Path(model_ref).exists():
            return model_ref
        model_size = model_ref.lower()

        mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v2",
            "large-v2": "mlx-community/whisper-large-v2",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "turbo": "mlx-community/whisper-large-v3-turbo",
        }
        return mapping.get(model_size, "mlx-community/whisper-small-mlx")

    @staticmethod
    def _move_torch_waveform(waveform: torch.Tensor, runtime_device: str) -> torch.Tensor:
        if runtime_device == "cpu":
            return waveform
        move_kwargs = {"non_blocking": True} if runtime_device == "cuda" else {}
        return waveform.to(torch.device(runtime_device), **move_kwargs)

    def _resolve_hf_token(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        """
        Resolve HF auth token robustly:
        1) environment variables
        2) current ASR config hardcoded token
        """
        env_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or ""
        ).strip()
        if env_token:
            return env_token

        cfg = cfg or {}
        cfg_token = str(cfg.get("hf_token") or "").strip()
        if cfg_token:
            return cfg_token

        fw_cfg = self.asr_cfg.get("faster_whisper", {}) or {}
        wx_cfg = self.asr_cfg.get("whisperx", {}) or {}
        return str(
            fw_cfg.get("hf_token")
            or wx_cfg.get("hf_token")
            or ""
        ).strip()

    @staticmethod
    def _read_ngc_api_key_from_well_known_files() -> str:
        for candidate in (
            Path.home() / ".ngc" / "config",
            Path.home() / ".config" / "ngc" / "config",
        ):
            try:
                if not candidate.exists() or not candidate.is_file():
                    continue
                for line in candidate.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines():
                    text = str(line or "").strip()
                    if not text or text.startswith("#"):
                        continue
                    if "=" not in text:
                        continue
                    key, value = text.split("=", 1)
                    if key.strip().lower() not in {"apikey", "api_key"}:
                        continue
                    token = value.strip()
                    if token:
                        return token
            except Exception:
                continue
        return ""

    def _resolve_ngc_api_key(self, cfg: Optional[Dict[str, Any]] = None) -> str:
        env_key = (
            os.getenv("NGC_API_KEY")
            or os.getenv("NVIDIA_NGC_API_KEY")
            or os.getenv("NGC_CLI_API_KEY")
            or ""
        ).strip()
        if env_key:
            return env_key

        cfg = cfg or {}
        cfg_key = str(cfg.get("ngc_api_key") or "").strip()
        if cfg_key:
            return cfg_key

        nemo_cfg = self.asr_cfg.get("nemo_msdd", {}) or {}
        cfg_key = str(nemo_cfg.get("ngc_api_key") or "").strip()
        if cfg_key:
            return cfg_key
        file_key = self._read_ngc_api_key_from_well_known_files()
        if file_key:
            return file_key
        return DEFAULT_NGC_API_KEY

    @staticmethod
    def _is_nemo_auth_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return any(
            marker in text
            for marker in (
                "requires an ngc api key",
                "http error 401",
                "http error 403",
                "status code 401",
                "status code 403",
                "forbidden",
                "unauthorized",
            )
        )

    @staticmethod
    def _is_nemo_forbidden_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return "403" in text or "forbidden" in text

    def _nemo_auth_required_error(
        self,
        *,
        kind: str,
        model_name: str,
        filename: str,
    ) -> RuntimeError:
        return RuntimeError(
            "NeMo "
            f"{kind} model {model_name} requires an NGC API key or a local {filename} file. "
            "Set asr.nemo_msdd.ngc_api_key (or env NGC_API_KEY) or point the model path to a local .nemo file."
        )

    def _nemo_auth_forbidden_skip_error(
        self,
        *,
        kind: str,
        model_name: str,
        filename: str,
    ) -> RuntimeError:
        return RuntimeError(
            "NeMo "
            f"{kind} model {model_name} returned HTTP 403/forbidden for {filename}; "
            "skipping MSDD download for this run and falling back to the secondary diarization pipeline."
        )

    def _nemo_download_headers(
        self,
        *,
        cfg: Optional[Dict[str, Any]] = None,
        source: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        headers = {"User-Agent": "Mozilla/5.0"}
        ngc_key = str((source or {}).get("ngc_api_key") or "").strip()
        if not ngc_key:
            ngc_key = self._resolve_ngc_api_key(cfg)
        if ngc_key:
            headers["Authorization"] = f"Bearer {ngc_key}"
            headers["Content-Type"] = "application/json"
            headers["X-Api-Key"] = ngc_key
            headers["ngc-api-key"] = ngc_key
        return headers

    def _find_local_nemo_artifact(self, filename: str) -> Optional[Path]:
        name = str(filename or "").strip()
        if not name:
            return None

        direct_candidates = [
            self._nemo_model_cache_root() / "model_artifacts" / name,
            APP_ROOT / "output_files" / ".nemo_msdd" / "model_artifacts" / name,
            INTERNAL_ROOT / "model_caches" / "nemo" / "model_artifacts" / name,
        ]
        for candidate in direct_candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    valid, reason = _inspect_nemo_artifact(candidate)
                    if valid:
                        return candidate
                    _warn_invalid_nemo_artifact(candidate, reason)
            except Exception:
                continue

        nemo_roots: List[Path] = []
        for raw in (
            os.getenv("NEMO_CACHE_DIR"),
            os.getenv("NEMO_HOME"),
            str(INTERNAL_ROOT / "model_caches" / "nemo"),
            str(Path.home() / ".cache" / "torch" / "NeMo"),
        ):
            text = str(raw or "").strip()
            if not text:
                continue
            candidate_root = Path(text).expanduser()
            if candidate_root not in nemo_roots:
                nemo_roots.append(candidate_root)

        for nemo_cache_root in nemo_roots:
            direct = nemo_cache_root / name
            model_artifact_direct = nemo_cache_root / "model_artifacts" / name
            nested = nemo_cache_root / ".nemo_msdd" / "model_artifacts" / name
            try:
                if direct.exists() and direct.is_file():
                    valid, reason = _inspect_nemo_artifact(direct)
                    if valid:
                        return direct
                    _warn_invalid_nemo_artifact(direct, reason)
                if model_artifact_direct.exists() and model_artifact_direct.is_file():
                    valid, reason = _inspect_nemo_artifact(model_artifact_direct)
                    if valid:
                        return model_artifact_direct
                    _warn_invalid_nemo_artifact(model_artifact_direct, reason)
                if nested.exists() and nested.is_file():
                    valid, reason = _inspect_nemo_artifact(nested)
                    if valid:
                        return nested
                    _warn_invalid_nemo_artifact(nested, reason)
            except Exception:
                continue
            if nemo_cache_root.exists():
                try:
                    for candidate in nemo_cache_root.rglob(name):
                        if candidate.is_file():
                            valid, reason = _inspect_nemo_artifact(candidate)
                            if valid:
                                return candidate
                            _warn_invalid_nemo_artifact(candidate, reason)
                except Exception:
                    continue
        return None

    def _emit_progress(self, event: str, **payload):
        if not self._progress_callback:
            return
        try:
            self._progress_callback(event, **payload)
        except TypeError:
            # Compatibility path for callbacks that expect (event, payload_dict).
            self._progress_callback(event, payload)
        except Exception as e:
            logger.debug(f"Transcriber progress callback failed ({event}): {e}")

    def _maybe_force_cuda_cleanup(self, force: bool = False):
        if not self.has_cuda:
            return
        if force or self._aggressive_cuda_cleanup:
            self._transcribe_calls_since_cleanup = 0
            _force_cuda_cleanup()
            return
        self._transcribe_calls_since_cleanup += 1
        if self._transcribe_calls_since_cleanup >= self._force_cuda_cleanup_every:
            self._transcribe_calls_since_cleanup = 0
            _force_cuda_cleanup()

    def _fw_download_policy(
        self, cfg: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        fw_cfg = cfg or self.asr_cfg.get("faster_whisper", {}) or {}

        mirror_endpoint = str(
            fw_cfg.get("mirror_endpoint", "https://hf-mirror.com") or ""
        ).strip()
        official_endpoint = str(
            fw_cfg.get("official_endpoint", "https://huggingface.co") or ""
        ).strip() or "https://huggingface.co"

        try:
            mirror_retries = int(fw_cfg.get("mirror_retries", 3))
        except (TypeError, ValueError):
            mirror_retries = 3
        mirror_retries = max(0, mirror_retries)

        try:
            official_retries = int(fw_cfg.get("official_retries", 2))
        except (TypeError, ValueError):
            official_retries = 2
        official_retries = max(1, official_retries)

        try:
            retry_wait_sec = float(fw_cfg.get("download_retry_wait_sec", 1.5))
        except (TypeError, ValueError):
            retry_wait_sec = 1.5
        retry_wait_sec = max(0.0, retry_wait_sec)

        return {
            "mirror_endpoint": mirror_endpoint,
            "official_endpoint": official_endpoint,
            "mirror_retries": mirror_retries,
            "official_retries": official_retries,
            "retry_wait_sec": retry_wait_sec,
        }

    def _fw_auto_endpoint_probe_enabled(
        self, cfg: Optional[Dict[str, Any]] = None
    ) -> bool:
        fw_cfg = cfg or self.asr_cfg.get("faster_whisper", {}) or {}
        enabled = self._safe_bool(fw_cfg.get("auto_endpoint_probe", True), True)
        env_raw = str(os.getenv("HF_AUTO_ROUTE", "") or "").strip().lower()
        if env_raw:
            enabled = env_raw in {"1", "true", "yes", "on"}
        return bool(enabled)

    @staticmethod
    def _endpoint_host(endpoint: str) -> str:
        raw = str(endpoint or "").strip()
        if not raw:
            return ""
        if "://" not in raw:
            raw = f"https://{raw}"
        try:
            parsed = urllib.parse.urlparse(raw)
            host = str(parsed.netloc or "").rsplit("@", 1)[-1]
            if ":" in host:
                host = host.split(":", 1)[0]
            return host.strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _probe_host_latency(host: str, timeout_sec: float = 1.4) -> Optional[float]:
        target = str(host or "").strip()
        if not target:
            return None
        sock = None
        start = time.perf_counter()
        try:
            sock = socket.create_connection((target, 443), timeout=max(0.2, float(timeout_sec)))
            return max(0.001, float(time.perf_counter() - start))
        except Exception:
            return None
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def _probe_google_reachability(
        self, timeout_sec: float = 1.4
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        for host in ("www.google.com", "www.gstatic.com", "google.com"):
            latency = self._probe_host_latency(host, timeout_sec=timeout_sec)
            if latency is not None:
                return True, host, latency
        return False, None, None

    def _rank_hf_download_sources(
        self,
        *,
        mirror_endpoint: str,
        official_endpoint: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], str]:
        mirror_endpoint = str(mirror_endpoint or "").strip()
        official_endpoint = str(official_endpoint or "").strip()
        if not mirror_endpoint:
            return ["official"], "mirror endpoint unavailable"
        if mirror_endpoint == official_endpoint:
            return ["official"], "mirror endpoint equals official endpoint"

        startup_region = str(os.getenv("MTS_DOWNLOAD_REGION", "") or "").strip().lower()
        startup_note = str(os.getenv("MTS_DOWNLOAD_REGION_NOTE", "") or "").strip()
        if startup_region in {"cn", "global"}:
            order = ["mirror", "official"] if startup_region == "cn" else ["official", "mirror"]
            return order, startup_note or f"startup region={startup_region}"

        default_order = ["mirror", "official"]
        if not self._fw_auto_endpoint_probe_enabled(cfg):
            return default_order, "auto endpoint probe disabled"

        cache_key = (mirror_endpoint, official_endpoint)
        cached = self._hf_endpoint_probe_cache.get(cache_key)
        if cached:
            order, note = cached
            return list(order), str(note)

        fw_cfg = cfg or self.asr_cfg.get("faster_whisper", {}) or {}
        timeout_sec = max(
            0.20,
            min(
                1.50,
                self._safe_float(fw_cfg.get("endpoint_probe_timeout_sec", 0.55), 0.55),
            ),
        )

        route = detect_download_route(timeout_sec=timeout_sec)
        route_region = str(route.get("region", "") or "").strip().lower()
        if route_region == "global":
            order = ["official", "mirror"]
        elif route_region == "cn":
            order = ["mirror", "official"]
        else:
            order = list(default_order)
        note = str(route.get("note", "") or "").strip() or f"{order[0]} -> {order[1]}"
        self._hf_endpoint_probe_cache[cache_key] = ((order[0], order[1]), note)
        return order, note

    def _build_hf_download_attempts(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        policy = self._fw_download_policy(cfg=cfg)
        mirror_endpoint = str(policy.get("mirror_endpoint", "") or "").strip()
        official_endpoint = str(
            policy.get("official_endpoint", "https://huggingface.co") or ""
        ).strip() or "https://huggingface.co"
        mirror_retries = max(0, int(policy.get("mirror_retries", 3)))
        official_retries = max(1, int(policy.get("official_retries", 2)))
        retry_wait_sec = max(0.0, float(policy.get("retry_wait_sec", 1.5)))

        source_specs: Dict[str, Dict[str, Any]] = {
            "official": {
                "source": "official",
                "endpoint": official_endpoint,
                "retries": official_retries,
            }
        }
        if mirror_endpoint and mirror_retries > 0:
            source_specs["mirror"] = {
                "source": "mirror",
                "endpoint": mirror_endpoint,
                "retries": mirror_retries,
            }

        source_order, route_note = self._rank_hf_download_sources(
            mirror_endpoint=mirror_endpoint,
            official_endpoint=official_endpoint,
            cfg=cfg,
        )
        logger.info("HF endpoint priority: %s", route_note)

        attempts: List[Dict[str, Any]] = []
        seen_endpoints: set[str] = set()
        for source in source_order:
            spec = source_specs.get(source)
            if not spec:
                continue
            retries = max(0, int(spec.get("retries", 0)))
            endpoint = str(spec.get("endpoint", "") or "").strip()
            endpoint_key = endpoint.lower()
            if retries <= 0:
                continue
            if endpoint_key in seen_endpoints:
                continue
            seen_endpoints.add(endpoint_key)
            attempts.append(
                {
                    "source": str(spec.get("source", source) or source),
                    "endpoint": endpoint,
                    "retries": retries,
                }
            )

        if not attempts:
            attempts.append(
                {
                    "source": "official",
                    "endpoint": official_endpoint,
                    "retries": max(1, official_retries),
                }
            )
        return attempts, retry_wait_sec

    def _nemo_model_sources_cfg(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        root = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        sources = root.get("model_sources", {}) if isinstance(root, dict) else {}
        return sources if isinstance(sources, dict) else {}

    @staticmethod
    def _nemo_guess_hf_nemo_filename(repo_id: str) -> str:
        name = str(repo_id or "").strip().rsplit("/", 1)[-1].strip()
        if not name:
            return "model.nemo"
        if name.lower().endswith(".nemo"):
            return name
        return f"{name}.nemo"

    def _nemo_builtin_model_source(
        self,
        *,
        kind: str,
        model_name: str,
    ) -> Dict[str, Any]:
        key = str(model_name or "").strip().lower().replace("\\", "/")
        kind_key = str(kind or "").strip().lower()

        if kind_key == "vad":
            if key in {
                "",
                "vad_multilingual_marblenet",
                "nvidia/vad_multilingual_marblenet",
                "frame_vad_multilingual_marblenet_v2.0",
                "nvidia/frame_vad_multilingual_marblenet_v2.0",
            }:
                return {
                    "repo_id": "nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0",
                    "filename": "frame_vad_multilingual_marblenet_v2.0.nemo",
                }

        if kind_key == "speaker":
            if key in {
                "",
                "titanet_large",
                "speakerverification_en_titanet_large",
                "nvidia/speakerverification_en_titanet_large",
            }:
                return {
                    "repo_id": "nvidia/speakerverification_en_titanet_large",
                    "filename": "speakerverification_en_titanet_large.nemo",
                }

        if kind_key == "msdd":
            if key in {
                "",
                "diar_msdd_telephonic",
                "nvidia/diar_msdd_telephonic",
            }:
                return {
                    "url": "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/diar_msdd_telephonic/versions/1.0.1/files/diar_msdd_telephonic.nemo",
                    "filename": "diar_msdd_telephonic.nemo",
                    "requires_ngc_api_key": True,
                }

        if kind_key == "sortformer":
            default_repo = "nvidia/diar_streaming_sortformer_4spk-v2.1"
            repo_alias = {
                "": default_repo,
                "diar_streaming_sortformer_4spk-v2.1": "nvidia/diar_streaming_sortformer_4spk-v2.1",
                "nvidia/diar_streaming_sortformer_4spk-v2.1": "nvidia/diar_streaming_sortformer_4spk-v2.1",
                "diar_streaming_sortformer_4spk-v2": "nvidia/diar_streaming_sortformer_4spk-v2",
                "nvidia/diar_streaming_sortformer_4spk-v2": "nvidia/diar_streaming_sortformer_4spk-v2",
                "diar_sortformer_4spk-v1": "nvidia/diar_sortformer_4spk-v1",
                "nvidia/diar_sortformer_4spk-v1": "nvidia/diar_sortformer_4spk-v1",
            }
            repo_id = repo_alias.get(key, str(model_name or "").strip() or default_repo)
            if "/" not in repo_id:
                repo_id = f"nvidia/{repo_id}"
            file_map = {
                "nvidia/diar_streaming_sortformer_4spk-v2.1": "diar_streaming_sortformer_4spk-v2.1.nemo",
                "nvidia/diar_streaming_sortformer_4spk-v2": "diar_streaming_sortformer_4spk-v2.nemo",
                "nvidia/diar_sortformer_4spk-v1": "diar_sortformer_4spk-v1.nemo",
            }
            return {
                "repo_id": repo_id,
                "filename": file_map.get(
                    repo_id.lower(),
                    self._nemo_guess_hf_nemo_filename(repo_id),
                ),
            }

        return {}

    def _nemo_resolve_model_source(
        self,
        *,
        kind: str,
        model_name: str,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_name = str(model_name or "").strip()
        if raw_name:
            local_candidate = Path(raw_name).expanduser()
            if local_candidate.exists() and local_candidate.is_file():
                return {"local_path": str(local_candidate)}
            if raw_name.lower().startswith(("http://", "https://")):
                return {"url": raw_name}

        sources = self._nemo_model_sources_cfg(cfg=cfg)
        lookup_keys = [str(kind or "").strip(), raw_name, raw_name.lower()]
        for key in lookup_keys:
            if not key:
                continue
            entry = sources.get(key)
            if isinstance(entry, str):
                text = entry.strip()
                if not text:
                    continue
                if text.lower().startswith(("http://", "https://")):
                    return {"url": text}
                path_entry = Path(text).expanduser()
                if path_entry.exists() and path_entry.is_file():
                    return {"local_path": str(path_entry)}
                if "/" in text and not text.lower().endswith(".nemo"):
                    return {
                        "repo_id": text,
                        "filename": self._nemo_guess_hf_nemo_filename(text),
                    }
                continue

            if isinstance(entry, dict):
                source: Dict[str, Any] = {}
                for field in ("local_path", "repo_id", "filename", "url", "hf_token"):
                    value = str(entry.get(field, "") or "").strip()
                    if value:
                        source[field] = value
                if "requires_ngc_api_key" in entry:
                    source["requires_ngc_api_key"] = self._safe_bool(
                        entry.get("requires_ngc_api_key", False),
                        False,
                    )
                ngc_api_key = str(entry.get("ngc_api_key", "") or "").strip()
                if ngc_api_key:
                    source["ngc_api_key"] = ngc_api_key
                try:
                    source["retries"] = max(1, int(entry.get("retries", 1)))
                except Exception:
                    pass
                try:
                    source["retry_wait_sec"] = max(0.0, float(entry.get("retry_wait_sec", 0.0)))
                except Exception:
                    pass
                if source:
                    return source

        builtin = self._nemo_builtin_model_source(kind=kind, model_name=raw_name)
        if builtin:
            return builtin

        if raw_name and "/" in raw_name and not raw_name.lower().endswith(".nemo"):
            return {
                "repo_id": raw_name,
                "filename": self._nemo_guess_hf_nemo_filename(raw_name),
            }

        return {}

    def _download_hf_file_with_retry(
        self,
        *,
        repo_id: str,
        filename: str,
        cfg: Optional[Dict[str, Any]] = None,
        token: str = "",
        purpose: str = "hf-download",
    ) -> Path:
        from huggingface_hub import hf_hub_download

        attempts, retry_wait_sec = self._build_hf_download_attempts(cfg=cfg)
        total_attempts = sum(max(1, int(item.get("retries", 1))) for item in attempts)
        attempt_index = 0
        last_error: Optional[Exception] = None
        token = str(token or "").strip() or None

        for item in attempts:
            source = str(item.get("source", "official") or "official")
            endpoint = str(item.get("endpoint", "") or "").strip()
            retries = max(1, int(item.get("retries", 1)))
            for _ in range(retries):
                attempt_index += 1
                prev_endpoint = os.getenv("HF_ENDPOINT")
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint
                elif prev_endpoint is not None:
                    os.environ.pop("HF_ENDPOINT", None)

                try:
                    logger.info(
                        "  %s download attempt %d/%d via %s: %s",
                        purpose,
                        attempt_index,
                        total_attempts,
                        source,
                        endpoint or "<default>",
                    )
                    path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        token=token,
                        resume_download=True,
                        local_files_only=False,
                        endpoint=(endpoint or None),
                    )
                    return Path(path)
                except Exception as e:
                    last_error = e
                    logger.debug(
                        "%s download failed (%d/%d) via %s: %s",
                        purpose,
                        attempt_index,
                        total_attempts,
                        source,
                        e,
                    )
                    if retry_wait_sec > 0 and attempt_index < total_attempts:
                        time.sleep(retry_wait_sec)
                finally:
                    if prev_endpoint is None:
                        os.environ.pop("HF_ENDPOINT", None)
                    else:
                        os.environ["HF_ENDPOINT"] = prev_endpoint

        raise RuntimeError(
            f"{purpose} failed for {repo_id}/{filename} after "
            f"{attempt_index} attempts: {last_error}"
        )

    def _download_hf_snapshot_with_retry(
        self,
        *,
        repo_id: str,
        cfg: Optional[Dict[str, Any]] = None,
        token: str = "",
        purpose: str = "hf-snapshot",
    ) -> Path:
        self._ensure_diarization_dependency_compat()
        from huggingface_hub import snapshot_download

        attempts, retry_wait_sec = self._build_hf_download_attempts(cfg=cfg)
        total_attempts = sum(max(1, int(item.get("retries", 1))) for item in attempts)
        attempt_index = 0
        last_error: Optional[Exception] = None
        token = str(token or "").strip() or None

        for item in attempts:
            source = str(item.get("source", "official") or "official")
            endpoint = str(item.get("endpoint", "") or "").strip()
            retries = max(1, int(item.get("retries", 1)))
            for _ in range(retries):
                attempt_index += 1
                prev_endpoint = os.getenv("HF_ENDPOINT")
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint
                elif prev_endpoint is not None:
                    os.environ.pop("HF_ENDPOINT", None)

                try:
                    logger.info(
                        "  %s snapshot attempt %d/%d via %s: %s",
                        purpose,
                        attempt_index,
                        total_attempts,
                        source,
                        endpoint or "<default>",
                    )
                    path = snapshot_download(
                        repo_id=repo_id,
                        token=token,
                        resume_download=True,
                        local_files_only=False,
                        endpoint=(endpoint or None),
                    )
                    return Path(path)
                except Exception as e:
                    last_error = e
                    logger.debug(
                        "%s snapshot failed (%d/%d) via %s: %s",
                        purpose,
                        attempt_index,
                        total_attempts,
                        source,
                        e,
                    )
                    if retry_wait_sec > 0 and attempt_index < total_attempts:
                        time.sleep(retry_wait_sec)
                finally:
                    if prev_endpoint is None:
                        os.environ.pop("HF_ENDPOINT", None)
                    else:
                        os.environ["HF_ENDPOINT"] = prev_endpoint

        raise RuntimeError(
            f"{purpose} failed for {repo_id} after "
            f"{attempt_index} attempts: {last_error}"
        )

    def _should_stage_pyannote_snapshot(
        self,
        model_name: str,
        *,
        cfg: Optional[Dict[str, Any]] = None,
        purpose: str = "pyannote",
    ) -> bool:
        raw_name = str(model_name or "").strip()
        if not raw_name:
            return False
        if raw_name.lower().startswith(("http://", "https://")):
            return False
        local_candidate = Path(raw_name).expanduser()
        if local_candidate.exists():
            return False
        if "/" not in raw_name:
            return False

        default_value = "community-1" in raw_name.lower()
        if isinstance(cfg, dict) and "prefer_local_snapshot" in cfg:
            return self._safe_bool(
                cfg.get("prefer_local_snapshot", default_value),
                default_value,
            )
        return default_value

    def _prepare_pyannote_pipeline_source(
        self,
        model_name: str,
        *,
        cfg: Optional[Dict[str, Any]] = None,
        token: str = "",
        purpose: str = "pyannote",
    ) -> str:
        raw_name = str(model_name or "").strip()
        if not raw_name:
            return raw_name

        local_candidate = Path(raw_name).expanduser()
        if local_candidate.exists():
            return str(local_candidate)

        if not self._should_stage_pyannote_snapshot(
            raw_name,
            cfg=cfg,
            purpose=purpose,
        ):
            return raw_name

        snapshot_dir = self._download_hf_snapshot_with_retry(
            repo_id=raw_name,
            cfg=cfg,
            token=token,
            purpose=f"{purpose}-snapshot",
        )
        logger.info(
            "  %s using local pipeline snapshot: %s",
            purpose,
            snapshot_dir,
        )
        return str(snapshot_dir)

    @staticmethod
    def _rewrite_pyannote_model_placeholders(value: Any, model_root: Path) -> Any:
        if isinstance(value, str):
            text = str(value)
            if text == "$model":
                return str(model_root)
            if text.startswith("$model/"):
                relative = text.split("/", 1)[1].strip()
                resolved = (model_root / relative).resolve()
                if resolved.is_dir():
                    checkpoint = resolved / "pytorch_model.bin"
                    if checkpoint.exists() and checkpoint.is_file():
                        return str(checkpoint)
                return str(resolved)
            return value
        if isinstance(value, list):
            return [
                Transcriber._rewrite_pyannote_model_placeholders(item, model_root)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                Transcriber._rewrite_pyannote_model_placeholders(item, model_root)
                for item in value
            )
        if isinstance(value, dict):
            return {
                key: Transcriber._rewrite_pyannote_model_placeholders(item, model_root)
                for key, item in value.items()
            }
        return value

    def _apply_pyannote_runtime_config_compat(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(payload or {})
        pipeline_cfg = result.get("pipeline", {})
        if not isinstance(pipeline_cfg, dict):
            return result
        params = pipeline_cfg.get("params", {})
        if not isinstance(params, dict):
            return result

        clustering_name = str(params.get("clustering", "") or "").strip()
        if clustering_name == "VBxClustering":
            params["clustering"] = "AgglomerativeClustering"
            params.pop("plda", None)
            root_params = result.get("params", {})
            if isinstance(root_params, dict):
                clustering_params = root_params.get("clustering", {})
                if isinstance(clustering_params, dict):
                    clustering_params.pop("Fa", None)
                    clustering_params.pop("Fb", None)
                    root_params["clustering"] = clustering_params
                    result["params"] = root_params
            logger.info(
                "Applied pyannote local pipeline compatibility: "
                "VBxClustering -> AgglomerativeClustering"
            )
        pipeline_cfg["params"] = params
        result["pipeline"] = pipeline_cfg
        return result

    def _prepare_pyannote_local_checkpoint_path(self, checkpoint_path: str) -> str:
        raw_path = str(checkpoint_path or "").strip()
        if not raw_path:
            return raw_path

        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            return str(candidate)
        if not candidate.exists() or not candidate.is_dir():
            return raw_path

        config_path = candidate / "config.yaml"
        if not config_path.exists() or not config_path.is_file():
            return str(candidate)

        try:
            import yaml

            payload = yaml.safe_load(
                config_path.read_text(encoding="utf-8", errors="ignore")
            ) or {}
            rewritten = self._rewrite_pyannote_model_placeholders(
                payload,
                candidate.resolve(),
            )
            if isinstance(rewritten, dict):
                rewritten = self._apply_pyannote_runtime_config_compat(rewritten)
            cache_root = runtime_cache_root() / "pyannote_pipeline_configs"
            cache_root.mkdir(parents=True, exist_ok=True)
            digest = hashlib.md5(
                str(candidate.resolve()).encode("utf-8", errors="ignore")
            ).hexdigest()[:12]
            target_path = cache_root / f"{candidate.name}_{digest}.yaml"
            rendered = yaml.safe_dump(
                rewritten,
                sort_keys=False,
                allow_unicode=False,
            )
            existing = ""
            if target_path.exists():
                try:
                    existing = target_path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                except Exception:
                    existing = ""
            if rendered != existing:
                target_path.write_text(rendered, encoding="utf-8")
            logger.info(
                "Prepared pyannote local pipeline config: %s -> %s",
                candidate,
                target_path,
            )
            return str(target_path)
        except Exception as e:
            logger.debug(
                "Failed to rewrite pyannote local pipeline config for %s: %s",
                candidate,
                e,
            )
            return str(config_path)

    @staticmethod
    def _merge_nested_params(base: Any, override: Any) -> Any:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = dict(base)
            for key, value in override.items():
                merged[key] = Transcriber._merge_nested_params(merged.get(key), value)
            return merged
        if override is None:
            return base
        return override

    @staticmethod
    def _pyannote_parameter_default_value(parameter: Any) -> Any:
        try:
            from pyannote.pipeline.parameter import (
                Categorical,
                DiscreteUniform,
                Frozen,
                Integer,
                LogUniform,
                Uniform,
            )
        except Exception:
            return None

        if isinstance(parameter, Frozen):
            return parameter.value
        if isinstance(parameter, Categorical):
            return parameter.choices[0] if parameter.choices else None
        if isinstance(parameter, Integer):
            return int(parameter.low)
        if isinstance(parameter, DiscreteUniform):
            return float(parameter.low)
        if isinstance(parameter, Uniform):
            return float(parameter.low + parameter.high) / 2.0
        if isinstance(parameter, LogUniform):
            try:
                return float(math.sqrt(float(parameter.low) * float(parameter.high)))
            except Exception:
                return float(parameter.low)
        return None

    def _build_pyannote_complete_params(
        self,
        params_template: Any,
        *,
        preferred: Any = None,
    ) -> Any:
        if isinstance(params_template, dict):
            preferred_map = preferred if isinstance(preferred, dict) else {}
            result: Dict[str, Any] = {}
            for key, value in params_template.items():
                result[key] = self._build_pyannote_complete_params(
                    value,
                    preferred=preferred_map.get(key),
                )
            return result
        if preferred is not None:
            return preferred
        return self._pyannote_parameter_default_value(params_template)

    def _ensure_pyannote_pipeline_instantiated(self, pipeline: Any) -> Any:
        instantiated = getattr(pipeline, "instantiated", True)
        if instantiated:
            return pipeline
        if not hasattr(pipeline, "parameters") or not hasattr(pipeline, "instantiate"):
            return pipeline

        try:
            params_template = pipeline.parameters()
            current_params = pipeline.parameters(instantiated=True)
            completed = self._build_pyannote_complete_params(
                params_template,
                preferred=current_params,
            )
            pipeline.instantiate(completed)
            logger.info(
                "Applied pyannote compatibility shim: auto-instantiated missing pipeline parameters"
            )
        except Exception as e:
            logger.debug(
                "Failed to auto-instantiate pyannote pipeline parameters: %s",
                e,
            )
        return pipeline

    @staticmethod
    def _download_url_file_with_retry(
        *,
        url: str,
        target_path: Path,
        headers: Optional[Dict[str, str]] = None,
        retries: int = 3,
        retry_wait_sec: float = 1.5,
        timeout_sec: float = 60.0,
    ) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        last_error: Optional[Exception] = None
        retries = max(1, int(retries))
        wait_sec = max(0.0, float(retry_wait_sec))
        timeout_sec = max(3.0, float(timeout_sec))
        tmp_path = target_path.with_suffix(f"{target_path.suffix}.part")

        for attempt in range(1, retries + 1):
            try:
                request_headers = {"User-Agent": "Mozilla/5.0"}
                for key, value in (headers or {}).items():
                    if value:
                        request_headers[str(key)] = str(value)
                request = urllib.request.Request(str(url), headers=request_headers)
                with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                    with open(tmp_path, "wb") as f:
                        shutil.copyfileobj(response, f)
                tmp_path.replace(target_path)
                if target_path.exists() and target_path.stat().st_size > 0:
                    return target_path
                raise RuntimeError("downloaded file is empty")
            except Exception as e:
                last_error = e
                if attempt < retries and wait_sec > 0:
                    time.sleep(wait_sec)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

        raise RuntimeError(f"url download failed after {retries} attempts: {last_error}")

    def _prepare_nemo_model_path(
        self,
        *,
        kind: str,
        model_name: str,
        cfg: Optional[Dict[str, Any]] = None,
        purpose: str = "nemo-model",
    ) -> Optional[str]:
        source = self._nemo_resolve_model_source(
            kind=kind,
            model_name=model_name,
            cfg=cfg,
        )
        if not source:
            return None

        local_path = str(source.get("local_path", "") or "").strip()
        if local_path:
            path_obj = Path(local_path).expanduser()
            if path_obj.exists() and path_obj.is_file():
                valid, reason = _inspect_nemo_artifact(path_obj)
                if not valid:
                    raise RuntimeError(
                        f"Configured NeMo {kind} model path is not a valid archive: {path_obj} ({reason})"
                    )
                logger.info(
                    "  %s resolved local NeMo %s model: %s",
                    purpose,
                    kind,
                    path_obj,
                )
                return str(path_obj)
            return None

        repo_id = str(source.get("repo_id", "") or "").strip()
        if repo_id:
            filename = str(source.get("filename", "") or "").strip()
            if not filename:
                filename = self._nemo_guess_hf_nemo_filename(repo_id)
            token = str(source.get("hf_token", "") or "").strip()
            if not token:
                token = self._resolve_hf_token(cfg or {})
            model_file = self._download_hf_file_with_retry(
                repo_id=repo_id,
                filename=filename,
                cfg=cfg,
                token=token,
                purpose=purpose,
            )
            logger.info(
                "  %s cached HF model %s/%s -> %s",
                purpose,
                repo_id,
                filename,
                model_file,
            )
            return str(model_file)

        url = str(source.get("url", "") or "").strip()
        if url:
            cache_root = self._nemo_model_cache_root() / "model_artifacts"
            filename = str(source.get("filename", "") or "").strip()
            if not filename:
                parsed = urllib.parse.urlparse(url)
                candidate = Path(parsed.path).name
                if candidate:
                    filename = candidate
                else:
                    digest = hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest()
                    filename = f"{kind}_{digest}.nemo"
            cached_artifact = self._find_local_nemo_artifact(filename)
            if cached_artifact is not None:
                logger.info(
                    "  %s using cached NeMo %s artifact: %s",
                    purpose,
                    kind,
                    cached_artifact,
                )
                return str(cached_artifact)

            requires_ngc_api_key = self._safe_bool(
                source.get("requires_ngc_api_key", False),
                False,
            )
            if requires_ngc_api_key and not self._resolve_ngc_api_key(cfg):
                raise self._nemo_auth_required_error(
                    kind=kind,
                    model_name=model_name,
                    filename=filename,
                )
            target = cache_root / filename
            retries = max(
                1,
                self._safe_int(
                    source.get("retries", (cfg or {}).get("download_retries", 3)),
                    self._safe_int((cfg or {}).get("download_retries", 3), 3),
                ),
            )
            wait_sec = max(
                0.0,
                self._safe_float(
                    source.get(
                        "retry_wait_sec",
                        (cfg or {}).get("download_retry_wait_sec", 1.5),
                    ),
                    self._safe_float((cfg or {}).get("download_retry_wait_sec", 1.5), 1.5),
                ),
            )
            try:
                model_file = self._download_url_file_with_retry(
                    url=url,
                    target_path=target,
                    headers=self._nemo_download_headers(cfg=cfg, source=source),
                    retries=retries,
                    retry_wait_sec=wait_sec,
                )
            except Exception as e:
                if requires_ngc_api_key and self._is_nemo_auth_error(e):
                    if self._is_nemo_forbidden_error(e):
                        raise self._nemo_auth_forbidden_skip_error(
                            kind=kind,
                            model_name=model_name,
                            filename=filename,
                        ) from e
                    raise self._nemo_auth_required_error(
                        kind=kind,
                        model_name=model_name,
                        filename=filename,
                    ) from e
                raise
            logger.info(
                "  %s downloaded NeMo %s artifact: %s",
                purpose,
                kind,
                model_file,
            )
            valid, reason = _inspect_nemo_artifact(Path(model_file))
            if not valid:
                try:
                    Path(model_file).unlink()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Downloaded NeMo {kind} artifact is not a valid archive: {model_file} ({reason})"
                )
            return str(model_file)

        return None

    @staticmethod
    def _is_unexpected_kwarg_error(exc: Exception, kw_name: str) -> bool:
        msg = str(exc or "")
        return (
            "unexpected keyword argument" in msg
            and (f"'{kw_name}'" in msg or f"\"{kw_name}\"" in msg)
        )

    @staticmethod
    def _extract_unexpected_kwarg(exc: Exception) -> str:
        msg = str(exc or "")
        match = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", msg)
        if match:
            return str(match.group(1) or "")
        return ""

    @staticmethod
    def _is_nemo_frame_vad_state_dict_compat_error(exc: Exception) -> bool:
        msg = str(exc or "")
        msg_lower = msg.lower()
        return (
            "encdecframeclassificationmodel" in msg_lower
            and "missing key(s) in state_dict" in msg_lower
            and "loss.weight" in msg_lower
        )

    @staticmethod
    def _is_nemo_msdd_index_oob_error(exc: Exception) -> bool:
        msg = str(exc or "")
        msg_lower = msg.lower()
        if "out of bounds" in msg_lower and ("axis" in msg_lower or "dimension" in msg_lower):
            return True
        if "list index out of range" in msg_lower:
            return True
        if "too many indices for array" in msg_lower:
            return True
        return False

    @staticmethod
    def _is_windows_symlink_privilege_error(exc: Exception) -> bool:
        msg = str(exc or "")
        msg_lower = msg.lower()
        return (
            "winerror 1314" in msg_lower
            or "required privilege is not held by the client" in msg_lower
            or "客户端没有所需的特权" in msg
        )

    @staticmethod
    def _is_pyannote_known_compat_error(exc: Exception) -> bool:
        msg = str(exc or "")
        msg_lower = msg.lower()
        plda_incompat = (
            "unexpected keyword argument" in msg_lower
            and "'plda'" in msg_lower
            and "speakerdiarization" in msg_lower
        )
        torch_weights_only_incompat = (
            "weights only load failed" in msg_lower
            or (
                "unsupported global" in msg_lower
                and "torch.torch_version.torchversion" in msg_lower
            )
        )
        segmentation_template_incompat = (
            "repo id must use alphanumeric chars" in msg_lower
            and "$model/segmentation" in msg
        )
        symlink_privilege_incompat = Transcriber._is_windows_symlink_privilege_error(exc)
        return (
            plda_incompat
            or torch_weights_only_incompat
            or segmentation_template_incompat
            or symlink_privilege_incompat
        )

    @staticmethod
    def _is_pyannote_known_non_retryable_error(exc: Exception) -> bool:
        if Transcriber._is_pyannote_known_compat_error(exc):
            return True
        msg_lower = str(exc or "").lower()
        return any(
            marker in msg_lower
            for marker in (
                "401 client error",
                "403 client error",
                "404 client error",
                "entry not found",
                "repository not found",
                "revision not found",
                "not found for url",
                "unauthorized",
                "forbidden",
                "gated",
                "connecterror",
                "name or service not known",
                "temporary failure in name resolution",
                "failed to resolve",
                "nodename nor servname provided",
            )
        )

    @staticmethod
    def _is_sortformer_module_kwarg_incompat_error(exc: Exception) -> bool:
        msg = str(exc or "")
        return (
            "SortformerModules" in msg
            and "unexpected keyword argument" in msg
        )

    @staticmethod
    def _is_sortformer_audio_path_only_error(exc: Exception) -> bool:
        msg = str(exc or "")
        msg_lower = msg.lower()
        return (
            "only `str`" in msg_lower
            and "path to audio file" in msg_lower
            and "supported as input" in msg_lower
        )

    @staticmethod
    def _sortformer_modules_supports_kwarg(kw_name: str) -> bool:
        """
        Detect whether current NeMo SortformerModules runtime accepts a given kwarg.
        Used to proactively select a compatible checkpoint family.
        """
        key = str(kw_name or "").strip()
        if not key:
            return True
        try:
            Transcriber._ensure_signal_sigkill_compat()
            Transcriber._ensure_torchmetrics_get_num_classes_compat()
        except Exception:
            pass
        try:
            from nemo.collections.asr.modules import sortformer_modules  # type: ignore

            cls = getattr(sortformer_modules, "SortformerModules", None)
            if cls is None:
                return True
            sig = inspect.signature(cls.__init__)
            return key in sig.parameters
        except Exception:
            # Keep runtime permissive when introspection is unavailable.
            return True

    @staticmethod
    def _sortformer_diarize_supports_audio_input(model_cls: Any) -> bool:
        """
        NeMo sortformer diarize API differs across releases.
        This pipeline requires an in-memory audio entrypoint.
        """
        try:
            diarize_fn = getattr(model_cls, "diarize", None)
            if diarize_fn is None:
                return False
            sig = inspect.signature(diarize_fn)
            params = list(sig.parameters.values())
            names = {str(p.name or "") for p in params}
            if any(
                key in names
                for key in ("audio", "audio_signal", "audio_signals", "waveform", "waveforms")
            ):
                return True
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return True
            return False
        except Exception:
            return True

    @staticmethod
    def _sortformer_diarize_supports_kwarg(model_cls: Any, kw_name: str) -> bool:
        key = str(kw_name or "").strip()
        if not key:
            return False
        try:
            diarize_fn = getattr(model_cls, "diarize", None)
            if diarize_fn is None:
                return False
            sig = inspect.signature(diarize_fn)
            params = list(sig.parameters.values())
            names = {str(p.name or "") for p in params}
            if key in names:
                return True
            return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        except Exception:
            return True

    @staticmethod
    def _sortformer_preferred_speaker_kwarg(model_cls: Any) -> str:
        for key in ("num_speakers", "oracle_num_speakers", "target_num_speakers"):
            if Transcriber._sortformer_diarize_supports_kwarg(model_cls, key):
                return key
        return ""

    @staticmethod
    def _is_context_or_ooc_error_text(text: str) -> bool:
        msg = str(text or "").strip().lower()
        if not msg:
            return False
        markers = (
            "ooc",
            "out of context",
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "prompt is too long",
            "input is too long",
            "token limit",
        )
        return any(token in msg for token in markers)

    @staticmethod
    def _configure_runtime_warning_filters() -> None:
        # Known noisy warnings on Windows/macOS from torch distributed redirects.
        warnings.filterwarnings(
            "ignore",
            message=".*Redirects are currently not supported in Windows or MacOs.*",
        )
        # pyannote emits TF32 reproducibility notices through warnings; keep logs clean.
        warnings.filterwarnings(
            "ignore",
            message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
            module=r".*pyannote\.audio\.utils\.reproducibility.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"The pkg_resources package is slated for removal as early as 2025-11-30\..*",
            category=UserWarning,
        )
        try:
            from pyannote.audio.utils.reproducibility import ReproducibilityWarning  # type: ignore

            warnings.filterwarnings("ignore", category=ReproducibilityWarning)
        except Exception:
            pass
        # Reduce migration/info noise in pyannote/lightning/speechbrain dependency chain.
        logging.getLogger("pytorch_lightning.utilities.migration.utils").setLevel(logging.WARNING)
        logging.getLogger("speechbrain.utils.fetching").setLevel(logging.INFO)

    @staticmethod
    def _ensure_torchaudio_speechbrain_compat() -> None:
        """
        SpeechBrain still probes legacy torchaudio backend APIs that were removed
        in newer torchaudio versions (e.g. 2.10). Add lightweight shims so
        pyannote pipelines depending on speechbrain can import successfully.
        """
        try:
            import torchaudio  # type: ignore
        except Exception:
            return

        if not hasattr(torchaudio, "AudioMetaData"):
            metadata_cls = None
            # torchaudio moved this symbol across internal modules in recent versions.
            for module_name in (
                "torchaudio.backend.common",
                "torchaudio._backend.common",
                "torchaudio.io._compat",
            ):
                try:
                    module = __import__(module_name, fromlist=["AudioMetaData"])
                    candidate = getattr(module, "AudioMetaData", None)
                    if candidate is not None:
                        metadata_cls = candidate
                        break
                except Exception:
                    continue

            if metadata_cls is None:
                from typing import NamedTuple

                class _AudioMetaDataShim(NamedTuple):
                    sample_rate: int
                    num_frames: int
                    num_channels: int
                    bits_per_sample: int
                    encoding: str

                metadata_cls = _AudioMetaDataShim
            setattr(torchaudio, "AudioMetaData", metadata_cls)

        if not hasattr(torchaudio, "list_audio_backends"):
            def _list_audio_backends() -> List[str]:
                return ["ffmpeg"] if hasattr(torchaudio, "load") else []
            setattr(torchaudio, "list_audio_backends", _list_audio_backends)

        if not hasattr(torchaudio, "set_audio_backend"):
            def _set_audio_backend(_backend: Optional[str] = None) -> None:
                return None
            setattr(torchaudio, "set_audio_backend", _set_audio_backend)

        if not hasattr(torchaudio, "get_audio_backend"):
            def _get_audio_backend() -> Optional[str]:
                try:
                    backends = list(getattr(torchaudio, "list_audio_backends")() or [])
                except Exception:
                    backends = []
                return str(backends[0]) if backends else None
            setattr(torchaudio, "get_audio_backend", _get_audio_backend)

    @staticmethod
    def _ensure_windows_symlink_copy_compat() -> None:
        """
        Windows may deny symlink creation for non-admin sessions (WinError 1314).
        Patch os.symlink so runtime fetch paths can fall back to plain copy.
        """
        global _WINDOWS_SYMLINK_COPY_COMPAT_APPLIED
        if _WINDOWS_SYMLINK_COPY_COMPAT_APPLIED:
            return

        if os.name != "nt":
            _WINDOWS_SYMLINK_COPY_COMPAT_APPLIED = True
            return

        symlink_fn = getattr(os, "symlink", None)
        if not callable(symlink_fn):
            _WINDOWS_SYMLINK_COPY_COMPAT_APPLIED = True
            return
        if getattr(symlink_fn, "__mts_windows_copy_fallback__", False):
            _WINDOWS_SYMLINK_COPY_COMPAT_APPLIED = True
            return

        def _wrapped(src, dst, *args, __fn=symlink_fn, **kwargs):
            try:
                return __fn(src, dst, *args, **kwargs)
            except Exception as e:
                if not Transcriber._is_windows_symlink_privilege_error(e):
                    raise

                src_path = Path(src)
                dst_path = Path(dst)
                if not src_path.is_absolute():
                    src_path = (dst_path.parent / src_path).resolve()

                try:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                if dst_path.exists() or dst_path.is_symlink():
                    if dst_path.is_dir() and not dst_path.is_symlink():
                        shutil.rmtree(dst_path, ignore_errors=True)
                    else:
                        dst_path.unlink(missing_ok=True)

                if src_path.is_dir():
                    shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(src_path), str(dst_path))

                global _WINDOWS_SYMLINK_COPY_COMPAT_USED_LOGGED
                if not _WINDOWS_SYMLINK_COPY_COMPAT_USED_LOGGED:
                    logger.info(
                        "Applied Windows symlink fallback: copy on WinError 1314."
                    )
                    _WINDOWS_SYMLINK_COPY_COMPAT_USED_LOGGED = True
                return None

        setattr(_wrapped, "__mts_windows_copy_fallback__", True)
        os.symlink = _wrapped
        _WINDOWS_SYMLINK_COPY_COMPAT_APPLIED = True
        logger.info(
            "Applied Windows compatibility shim: os.symlink privilege fallback -> copy"
        )

    @staticmethod
    def _ensure_speechbrain_fetch_windows_compat() -> None:
        """
        On Windows without developer mode/admin privileges, SpeechBrain's default
        SYMLINK strategy can fail with WinError 1314 during pyannote model fetch.
        Fall back to COPY automatically for this specific case.
        """
        global _SPEECHBRAIN_FETCH_COMPAT_APPLIED
        if _SPEECHBRAIN_FETCH_COMPAT_APPLIED:
            return

        Transcriber._ensure_torchaudio_speechbrain_compat()
        Transcriber._ensure_windows_symlink_copy_compat()

        # Recover from previous failed partial imports (common on Windows when
        # torchaudio legacy backend symbols are missing during early import).
        if os.name == "nt":
            sb_root = sys.modules.get("speechbrain")
            if sb_root is not None and not hasattr(sb_root, "utils"):
                for module_name in list(sys.modules.keys()):
                    if module_name == "speechbrain" or module_name.startswith("speechbrain."):
                        sys.modules.pop(module_name, None)

        try:
            sb_fetching = __import__("speechbrain.utils.fetching", fromlist=["*"])  # type: ignore
        except Exception:
            return

        link_fn = getattr(sb_fetching, "link_with_strategy", None)
        fetch_fn = getattr(sb_fetching, "fetch", None)
        local_strategy_cls = getattr(sb_fetching, "LocalStrategy", None)
        if local_strategy_cls is None:
            return

        link_patched = bool(
            callable(link_fn)
            and getattr(link_fn, "__mts_windows_symlink_compat__", False)
        )
        fetch_patched = bool(
            callable(fetch_fn)
            and getattr(fetch_fn, "__mts_windows_symlink_fetch_default_copy__", False)
        )
        if link_patched and fetch_patched:
            _SPEECHBRAIN_FETCH_COMPAT_APPLIED = True
            return

        copy_strategy = getattr(local_strategy_cls, "COPY", None)
        symlink_strategy = getattr(local_strategy_cls, "SYMLINK", None)
        if copy_strategy is None:
            return

        patched_any = False

        if callable(link_fn) and not link_patched:
            def _wrapped(src, dst, local_strategy, __fn=link_fn):
                try:
                    return __fn(src, dst, local_strategy)
                except Exception as e:
                    if os.name != "nt":
                        raise
                    if not Transcriber._is_windows_symlink_privilege_error(e):
                        raise
                    is_symlink_request = True
                    if symlink_strategy is not None:
                        try:
                            is_symlink_request = bool(local_strategy == symlink_strategy)
                        except Exception:
                            is_symlink_request = True
                    if not is_symlink_request:
                        raise
                    logger.info(
                        "Applied SpeechBrain fetch fallback: SYMLINK -> COPY on Windows due to WinError 1314."
                    )
                    return __fn(src, dst, copy_strategy)

            setattr(_wrapped, "__mts_windows_symlink_compat__", True)
            sb_fetching.link_with_strategy = _wrapped
            patched_any = True

        if callable(fetch_fn) and not fetch_patched:
            def _wrapped_fetch(*args, __fn=fetch_fn, **kwargs):
                if os.name == "nt":
                    call_args = list(args)
                    local_strategy_index = 10
                    local_strategy_value = kwargs.get(
                        "local_strategy",
                        call_args[local_strategy_index]
                        if len(call_args) > local_strategy_index
                        else None,
                    )
                    force_copy = (
                        local_strategy_value is None
                        or (
                            symlink_strategy is not None
                            and local_strategy_value == symlink_strategy
                        )
                    )
                    if force_copy:
                        if "local_strategy" in kwargs or len(call_args) <= local_strategy_index:
                            kwargs["local_strategy"] = copy_strategy
                        else:
                            call_args[local_strategy_index] = copy_strategy
                    return __fn(*call_args, **kwargs)
                return __fn(*args, **kwargs)

            setattr(_wrapped_fetch, "__mts_windows_symlink_fetch_default_copy__", True)
            sb_fetching.fetch = _wrapped_fetch
            patched_any = True

        _SPEECHBRAIN_FETCH_COMPAT_APPLIED = True
        if patched_any:
            logger.info(
                "Applied SpeechBrain compatibility shim: enforce COPY strategy on Windows for local fetches"
            )

    @staticmethod
    def _ensure_torchmetrics_get_num_classes_compat() -> None:
        """
        NeMo (and some pyannote-adjacent stacks through Lightning) can import
        `get_num_classes` from torchmetrics.utilities.data. Newer torchmetrics
        versions removed it. Provide a small runtime shim when missing.
        """
        patched_names: List[str] = []

        try:
            import torchmetrics as tm  # type: ignore
            from torchmetrics.utilities import data as tm_data  # type: ignore
        except Exception:
            return

        if hasattr(tm_data, "get_num_classes"):
            has_num_classes = True
        else:
            has_num_classes = False

        if not has_num_classes:
            def _get_num_classes(
                preds: Any = None,
                target: Any = None,
                num_classes: Optional[int] = None,
            ) -> int:
                if num_classes is not None:
                    try:
                        n = int(num_classes)
                        if n > 0:
                            return n
                    except Exception:
                        pass

                max_idx = -1
                for item in (preds, target):
                    if item is None:
                        continue
                    try:
                        t = torch.as_tensor(item)
                    except Exception:
                        continue
                    if t.numel() <= 0:
                        continue
                    try:
                        candidate = int(torch.max(t).item())
                    except Exception:
                        continue
                    if candidate > max_idx:
                        max_idx = candidate

                return int(max_idx + 1) if max_idx >= 0 else 0

            setattr(tm_data, "get_num_classes", _get_num_classes)
            patched_names.append("utilities.data.get_num_classes")

        try:
            from torchmetrics.classification import FBetaScore, F1Score, JaccardIndex  # type: ignore
        except Exception:
            FBetaScore = None  # type: ignore[assignment]
            F1Score = None  # type: ignore[assignment]
            JaccardIndex = None  # type: ignore[assignment]
        try:
            from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure  # type: ignore
        except Exception:
            PeakSignalNoiseRatio = None  # type: ignore[assignment]
            StructuralSimilarityIndexMeasure = None  # type: ignore[assignment]
        try:
            from torchmetrics.utilities.compute import auc as tm_auc  # type: ignore
        except Exception:
            tm_auc = None  # type: ignore[assignment]

        if not hasattr(tm, "AUC") and tm_auc is not None:
            class _LegacyAUC(tm.Metric):  # type: ignore[misc]
                full_state_update = False

                def __init__(
                    self,
                    reorder: bool = False,
                    compute_on_step: bool = True,
                    dist_sync_on_step: bool = False,
                    process_group: Optional[Any] = None,
                    dist_sync_fn: Any = None,
                    **kwargs: Any,
                ) -> None:
                    super().__init__(
                        compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        dist_sync_fn=dist_sync_fn,
                        **kwargs,
                    )
                    self.reorder = bool(reorder)
                    self.add_state("x", default=[], dist_reduce_fx="cat")
                    self.add_state("y", default=[], dist_reduce_fx="cat")

                def update(self, x: Any, y: Any) -> None:
                    try:
                        x_tensor = torch.as_tensor(x).reshape(-1)
                        y_tensor = torch.as_tensor(y).reshape(-1)
                    except Exception:
                        return
                    if x_tensor.numel() <= 0 or y_tensor.numel() <= 0:
                        return
                    self.x.append(x_tensor)
                    self.y.append(y_tensor)

                def compute(self) -> torch.Tensor:
                    if not self.x or not self.y:
                        return torch.tensor(0.0)
                    try:
                        x_tensor = torch.cat([torch.as_tensor(item).reshape(-1) for item in self.x])
                        y_tensor = torch.cat([torch.as_tensor(item).reshape(-1) for item in self.y])
                    except Exception:
                        return torch.tensor(0.0)
                    return tm_auc(x_tensor, y_tensor, reorder=self.reorder)

            _LegacyAUC.__name__ = "AUC"
            _LegacyAUC.__qualname__ = "AUC"
            setattr(tm, "AUC", _LegacyAUC)
            patched_names.append("AUC")

        if not hasattr(tm, "F1") and F1Score is not None:
            class _LegacyF1(F1Score):  # type: ignore[misc]
                def __init__(
                    self,
                    num_classes: int,
                    threshold: float = 0.5,
                    average: str = "micro",
                    multilabel: bool = False,
                    compute_on_step: bool = True,
                    dist_sync_on_step: bool = False,
                    process_group: Optional[Any] = None,
                    **kwargs: Any,
                ) -> None:
                    task = "multilabel" if bool(multilabel) else "multiclass"
                    task_kwargs: Dict[str, Any] = {
                        "task": task,
                        "average": average,
                    }
                    if task == "multilabel":
                        task_kwargs["num_labels"] = max(1, int(num_classes))
                        task_kwargs["threshold"] = float(threshold)
                    else:
                        task_kwargs["num_classes"] = max(1, int(num_classes))
                    super().__init__(
                        compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        **task_kwargs,
                        **kwargs,
                    )

            _LegacyF1.__name__ = "F1"
            _LegacyF1.__qualname__ = "F1"
            setattr(tm, "F1", _LegacyF1)
            patched_names.append("F1")

        if not hasattr(tm, "FBeta") and FBetaScore is not None:
            class _LegacyFBeta(FBetaScore):  # type: ignore[misc]
                def __init__(
                    self,
                    num_classes: int,
                    beta: float = 1.0,
                    threshold: float = 0.5,
                    average: str = "micro",
                    multilabel: bool = False,
                    compute_on_step: bool = True,
                    dist_sync_on_step: bool = False,
                    process_group: Optional[Any] = None,
                    **kwargs: Any,
                ) -> None:
                    task = "multilabel" if bool(multilabel) else "multiclass"
                    task_kwargs: Dict[str, Any] = {
                        "task": task,
                        "average": average,
                        "beta": float(beta),
                    }
                    if task == "multilabel":
                        task_kwargs["num_labels"] = max(1, int(num_classes))
                        task_kwargs["threshold"] = float(threshold)
                    else:
                        task_kwargs["num_classes"] = max(1, int(num_classes))
                    super().__init__(
                        compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        **task_kwargs,
                        **kwargs,
                    )

            _LegacyFBeta.__name__ = "FBeta"
            _LegacyFBeta.__qualname__ = "FBeta"
            setattr(tm, "FBeta", _LegacyFBeta)
            patched_names.append("FBeta")

        if not hasattr(tm, "IoU") and JaccardIndex is not None:
            class _LegacyIoU(JaccardIndex):  # type: ignore[misc]
                def __init__(
                    self,
                    num_classes: int,
                    ignore_index: Optional[int] = None,
                    absent_score: float = 0.0,
                    threshold: float = 0.5,
                    reduction: str = "elementwise_mean",
                    compute_on_step: bool = True,
                    dist_sync_on_step: bool = False,
                    process_group: Optional[Any] = None,
                    **kwargs: Any,
                ) -> None:
                    del absent_score, threshold, reduction
                    super().__init__(
                        task="multiclass",
                        num_classes=max(1, int(num_classes)),
                        ignore_index=ignore_index,
                        compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        **kwargs,
                    )

            _LegacyIoU.__name__ = "IoU"
            _LegacyIoU.__qualname__ = "IoU"
            setattr(tm, "IoU", _LegacyIoU)
            patched_names.append("IoU")

        if not hasattr(tm, "PSNR") and PeakSignalNoiseRatio is not None:
            class _LegacyPSNR(PeakSignalNoiseRatio):  # type: ignore[misc]
                pass

            _LegacyPSNR.__name__ = "PSNR"
            _LegacyPSNR.__qualname__ = "PSNR"
            setattr(tm, "PSNR", _LegacyPSNR)
            patched_names.append("PSNR")

        if not hasattr(tm, "SSIM") and StructuralSimilarityIndexMeasure is not None:
            class _LegacySSIM(StructuralSimilarityIndexMeasure):  # type: ignore[misc]
                def __init__(
                    self,
                    kernel_size: Any = (11, 11),
                    sigma: Any = (1.5, 1.5),
                    reduction: str = "elementwise_mean",
                    data_range: Optional[float] = None,
                    k1: float = 0.01,
                    k2: float = 0.03,
                    compute_on_step: bool = True,
                    dist_sync_on_step: bool = False,
                    process_group: Optional[Any] = None,
                    **kwargs: Any,
                ) -> None:
                    super().__init__(
                        kernel_size=kernel_size,
                        sigma=sigma,
                        reduction=reduction,
                        data_range=data_range,
                        k1=k1,
                        k2=k2,
                        compute_on_step=compute_on_step,
                        dist_sync_on_step=dist_sync_on_step,
                        process_group=process_group,
                        **kwargs,
                    )

            _LegacySSIM.__name__ = "SSIM"
            _LegacySSIM.__qualname__ = "SSIM"
            setattr(tm, "SSIM", _LegacySSIM)
            patched_names.append("SSIM")

        if patched_names:
            logger.info(
                "Applied torchmetrics compatibility shim: %s",
                ", ".join(patched_names),
            )

    @staticmethod
    def _ensure_pytorch_lightning_model_summary_compat() -> None:
        """
        Older pyannote/audio releases can import
        `pytorch_lightning.utilities.model_summary`, while newer Lightning keeps
        the implementation under `lightning.pytorch.utilities.model_summary`.
        Register a module alias when the legacy path is missing.
        """
        global _PYTORCH_LIGHTNING_MODEL_SUMMARY_COMPAT_APPLIED
        if _PYTORCH_LIGHTNING_MODEL_SUMMARY_COMPAT_APPLIED:
            return

        legacy_name = "pytorch_lightning.utilities.model_summary"
        if legacy_name in sys.modules:
            _PYTORCH_LIGHTNING_MODEL_SUMMARY_COMPAT_APPLIED = True
            return

        target_module = None
        for candidate in (
            "lightning.pytorch.utilities.model_summary",
            "lightning.pytorch.utilities.model_summary.model_summary",
        ):
            try:
                target_module = importlib.import_module(candidate)
                break
            except Exception:
                continue

        if target_module is None:
            return

        sys.modules[legacy_name] = target_module
        try:
            import pytorch_lightning.utilities as plu  # type: ignore

            if not hasattr(plu, "model_summary"):
                setattr(plu, "model_summary", target_module)
        except Exception:
            pass

        _PYTORCH_LIGHTNING_MODEL_SUMMARY_COMPAT_APPLIED = True
        logger.info(
            "Applied pytorch_lightning compatibility shim: utilities.model_summary alias"
        )

    @staticmethod
    def _ensure_signal_sigkill_compat() -> None:
        """
        Some NeMo/pyannote dependency chains import UNIX-only signal constants.
        Add a minimal SIGKILL alias on Windows when absent.
        """
        global _SIGKILL_COMPAT_APPLIED
        if _SIGKILL_COMPAT_APPLIED:
            return
        _SIGKILL_COMPAT_APPLIED = True

        if hasattr(signal, "SIGKILL"):
            return

        fallback_name = ""
        fallback = getattr(signal, "SIGTERM", None)
        if fallback is not None:
            fallback_name = "SIGTERM"
        else:
            fallback = getattr(signal, "SIGINT", None)
            if fallback is not None:
                fallback_name = "SIGINT"
        if fallback is None:
            return
        try:
            setattr(signal, "SIGKILL", fallback)
            logger.info(
                "Applied signal compatibility shim: SIGKILL -> %s",
                fallback_name,
            )
        except Exception:
            return

    @staticmethod
    def _ensure_hf_hub_auth_token_kw_compat() -> None:
        """
        Bridge pyannote/neMo stacks that still pass use_auth_token to
        huggingface_hub versions that only accept token.
        """
        global _HF_HUB_AUTH_TOKEN_COMPAT_APPLIED
        if _HF_HUB_AUTH_TOKEN_COMPAT_APPLIED:
            return
        _HF_HUB_AUTH_TOKEN_COMPAT_APPLIED = True

        try:
            import huggingface_hub  # type: ignore
        except Exception:
            return

        patched = False

        def _patch_callable(owner: Any, attr_name: str) -> bool:
            nonlocal patched
            fn = getattr(owner, attr_name, None)
            if not callable(fn):
                return False
            if getattr(fn, "__mts_auth_token_kw_compat__", False):
                return False

            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None

            supports_use_auth_token = bool(sig and "use_auth_token" in sig.parameters)
            supports_token = bool(sig and "token" in sig.parameters)
            if supports_use_auth_token and not supports_token:
                return False

            def _wrapped(*args, __fn=fn, **kwargs):
                legacy_token = kwargs.pop("use_auth_token", None)
                if legacy_token is not None and "token" not in kwargs:
                    kwargs["token"] = legacy_token
                try:
                    return __fn(*args, **kwargs)
                except TypeError as e:
                    if legacy_token is None:
                        raise
                    if not Transcriber._is_unexpected_kwarg_error(e, "token"):
                        raise
                    retry_kwargs = dict(kwargs)
                    retry_kwargs.pop("token", None)
                    retry_kwargs["use_auth_token"] = legacy_token
                    return __fn(*args, **retry_kwargs)

            setattr(_wrapped, "__mts_auth_token_kw_compat__", True)
            setattr(owner, attr_name, _wrapped)
            patched = True
            return True

        modules_to_patch = [huggingface_hub]
        for submodule_name in (
            "file_download",
            "_snapshot_download",
            "utils._validators",
        ):
            try:
                mod = __import__(f"huggingface_hub.{submodule_name}", fromlist=["*"])
            except Exception:
                continue
            modules_to_patch.append(mod)

        for module in modules_to_patch:
            _patch_callable(module, "hf_hub_download")
            _patch_callable(module, "snapshot_download")

        for module in list(sys.modules.values()):
            if module is None:
                continue
            try:
                _patch_callable(module, "hf_hub_download")
                _patch_callable(module, "snapshot_download")
            except Exception:
                continue

        if patched:
            logger.info(
                "Applied huggingface_hub compatibility shim: use_auth_token -> token"
            )

    @staticmethod
    def _ensure_pyannote_speaker_diar_plda_compat(
        exc: Optional[Exception] = None,
        force: bool = False,
    ) -> None:
        """
        Some pyannote pipelines (e.g. speaker-diarization-community-1) pass
        `plda=...` into SpeakerDiarization.__init__, while older runtimes do not
        accept this kwarg. Ignore it at runtime when unsupported.
        """
        global _PYANNOTE_SPK_DIAR_PLDA_COMPAT_APPLIED
        if _PYANNOTE_SPK_DIAR_PLDA_COMPAT_APPLIED and not force:
            return

        classes: List[Any] = []
        seen: set[int] = set()

        def _add_class(candidate: Any) -> None:
            if not inspect.isclass(candidate):
                return
            if str(getattr(candidate, "__name__", "")) != "SpeakerDiarization":
                return
            key = id(candidate)
            if key in seen:
                return
            seen.add(key)
            classes.append(candidate)

        for module_name in (
            "pyannote.audio.pipelines.speaker_diarization",
            "pyannote.audio.pipelines",
        ):
            try:
                module = __import__(module_name, fromlist=["SpeakerDiarization"])
                _add_class(getattr(module, "SpeakerDiarization", None))
            except Exception:
                continue

        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith("pyannote.audio.pipelines"):
                continue
            try:
                _add_class(getattr(module, "SpeakerDiarization", None))
            except Exception:
                continue

        tb = getattr(exc, "__traceback__", None)
        while tb is not None:
            frame = tb.tb_frame
            _add_class(getattr(frame.f_locals.get("self", None), "__class__", None))
            tb = tb.tb_next

        if not classes:
            return

        patched_count = 0
        already_compatible = False

        for speaker_diar_cls in classes:
            init_fn = getattr(speaker_diar_cls, "__init__", None)
            if not callable(init_fn):
                continue
            if getattr(init_fn, "__mts_pyannote_plda_compat__", False):
                already_compatible = True
                continue

            try:
                sig = inspect.signature(init_fn)
                params = list(sig.parameters.values())
                if "plda" in sig.parameters:
                    already_compatible = True
                    continue
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                    already_compatible = True
                    continue
            except Exception:
                pass

            def _wrapped(self, *args, __fn=init_fn, **kwargs):
                applied = False
                call_kwargs = dict(kwargs)
                if "plda" in call_kwargs:
                    call_kwargs.pop("plda", None)
                    applied = True

                try:
                    result = __fn(self, *args, **call_kwargs)
                except TypeError as e:
                    if not Transcriber._is_unexpected_kwarg_error(e, "plda"):
                        raise
                    retry_kwargs = dict(call_kwargs)
                    retry_kwargs.pop("plda", None)
                    applied = True
                    result = __fn(self, *args, **retry_kwargs)

                if applied:
                    global _PYANNOTE_SPK_DIAR_PLDA_COMPAT_USED_LOGGED
                    if not _PYANNOTE_SPK_DIAR_PLDA_COMPAT_USED_LOGGED:
                        logger.info(
                            "Applied pyannote compatibility shim: "
                            "SpeakerDiarization.__init__ ignores unsupported 'plda' kwarg."
                        )
                        _PYANNOTE_SPK_DIAR_PLDA_COMPAT_USED_LOGGED = True
                return result

            setattr(_wrapped, "__mts_pyannote_plda_compat__", True)
            speaker_diar_cls.__init__ = _wrapped
            patched_count += 1

        if patched_count > 0 or already_compatible:
            _PYANNOTE_SPK_DIAR_PLDA_COMPAT_APPLIED = True
        if patched_count > 0:
            logger.info(
                "Applied pyannote compatibility shim: SpeakerDiarization.__init__ 'plda' kwarg fallback (%d class(es)).",
                patched_count,
            )

    @staticmethod
    def _ensure_pyannote_model_version_compat() -> None:
        global _PYANNOTE_MODEL_VERSION_COMPAT_APPLIED
        if _PYANNOTE_MODEL_VERSION_COMPAT_APPLIED:
            return

        try:
            import pytorch_lightning as pl
            from pyannote.audio import __version__ as pyannote_audio_version
            from pyannote.audio.core.model import Model
        except Exception:
            return

        on_load_checkpoint = getattr(Model, "on_load_checkpoint", None)
        if not callable(on_load_checkpoint):
            return
        if getattr(on_load_checkpoint, "__mts_pyannote_model_version_compat__", False):
            _PYANNOTE_MODEL_VERSION_COMPAT_APPLIED = True
            return

        def _wrapped(self, checkpoint, __fn=on_load_checkpoint):
            try:
                pa_section = checkpoint.setdefault("pyannote.audio", {})
                if isinstance(pa_section, dict):
                    versions = pa_section.setdefault("versions", {})
                    if isinstance(versions, dict):
                        versions.setdefault("torch", torch.__version__)
                        versions.setdefault("pyannote.audio", pyannote_audio_version)
                checkpoint.setdefault("pytorch-lightning_version", pl.__version__)
            except Exception:
                pass
            return __fn(self, checkpoint)

        setattr(_wrapped, "__mts_pyannote_model_version_compat__", True)
        Model.on_load_checkpoint = _wrapped
        _PYANNOTE_MODEL_VERSION_COMPAT_APPLIED = True
        logger.info(
            "Applied pyannote compatibility shim: Model checkpoint version fallback"
        )

    @staticmethod
    def _ensure_lightning_torch_load_weights_only_compat() -> None:
        """
        Torch 2.6+ defaults torch.load(weights_only=True). Older pyannote/lightning
        checkpoint paths rely on full object deserialization and pass weights_only=None.
        Patch lightning cloud loaders to default None -> False explicitly.
        """
        global _LIGHTNING_TORCH_LOAD_COMPAT_APPLIED
        if _LIGHTNING_TORCH_LOAD_COMPAT_APPLIED:
            return
        _LIGHTNING_TORCH_LOAD_COMPAT_APPLIED = True

        try:
            import lightning_fabric.utilities.cloud_io as lf_cloud_io  # type: ignore
        except Exception:
            return

        load_fn = getattr(lf_cloud_io, "_load", None)
        if not callable(load_fn):
            return
        if getattr(load_fn, "__mts_weights_only_compat__", False):
            return

        def _wrapped(*args, __fn=load_fn, **kwargs):
            call_args = list(args)
            if len(call_args) >= 3:
                if call_args[2] is None:
                    call_args[2] = False
            elif kwargs.get("weights_only", None) is None:
                kwargs["weights_only"] = False
            return __fn(*call_args, **kwargs)

        setattr(_wrapped, "__mts_weights_only_compat__", True)
        lf_cloud_io._load = _wrapped

        # pyannote binds this callable as `pl_load` at import time. Patch when already imported.
        try:
            import pyannote.audio.core.model as pyannote_model  # type: ignore

            if callable(getattr(pyannote_model, "pl_load", None)):
                pyannote_model.pl_load = _wrapped
        except Exception:
            pass

        logger.info(
            "Applied lightning_fabric compatibility shim: torch.load(weights_only=None) -> False"
        )

    @staticmethod
    def _ensure_torch_load_weights_only_compat() -> None:
        """
        Torch 2.6+ changed torch.load default to weights_only=True.
        Some pyannote dependency paths call torch.load directly (without lightning),
        and expect full checkpoint object loading behavior.
        """
        global _TORCH_LOAD_WEIGHTS_ONLY_COMPAT_APPLIED
        if _TORCH_LOAD_WEIGHTS_ONLY_COMPAT_APPLIED:
            return

        load_fn = getattr(torch, "load", None)
        if not callable(load_fn):
            return
        if getattr(load_fn, "__mts_torch_load_weights_only_compat__", False):
            _TORCH_LOAD_WEIGHTS_ONLY_COMPAT_APPLIED = True
            return

        def _wrapped(*args, __fn=load_fn, **kwargs):
            if kwargs.get("weights_only", None) is None:
                kwargs["weights_only"] = False
            return __fn(*args, **kwargs)

        setattr(_wrapped, "__mts_torch_load_weights_only_compat__", True)
        torch.load = _wrapped
        _TORCH_LOAD_WEIGHTS_ONLY_COMPAT_APPLIED = True
        logger.info(
            "Applied torch compatibility shim: torch.load(weights_only=None) -> False"
        )

    @staticmethod
    def _ensure_torch_tensor_format_compat() -> None:
        """
        Newer PyTorch versions are stricter with Tensor.__format__.
        Older NeMo diarization paths still do string formatting like
        '{0:0.4f}'.format(tensor_value), which can raise TypeError.
        """
        global _TORCH_TENSOR_FORMAT_COMPAT_APPLIED
        if _TORCH_TENSOR_FORMAT_COMPAT_APPLIED:
            return

        tensor_format = getattr(torch.Tensor, "__format__", None)
        if not callable(tensor_format):
            return
        if getattr(tensor_format, "__mts_tensor_format_compat__", False):
            _TORCH_TENSOR_FORMAT_COMPAT_APPLIED = True
            return

        def _wrapped(self, format_spec="", __fn=tensor_format):
            try:
                return __fn(self, format_spec)
            except TypeError as e:
                global _TORCH_TENSOR_FORMAT_ND_FALLBACK_LOGGED
                msg = str(e or "").lower()
                if "unsupported format string passed to tensor.__format__" not in msg:
                    raise
                try:
                    numel = int(self.numel())
                    if numel <= 0:
                        raise
                    if numel > 1 and not _TORCH_TENSOR_FORMAT_ND_FALLBACK_LOGGED:
                        logger.info(
                            "Applied torch Tensor.__format__ non-scalar fallback: "
                            "using first element for numeric format."
                        )
                        _TORCH_TENSOR_FORMAT_ND_FALLBACK_LOGGED = True
                    scalar = float(self.detach().reshape(-1)[0].cpu().item())
                    return format(scalar, str(format_spec or ""))
                except Exception:
                    pass
                raise

        setattr(_wrapped, "__mts_tensor_format_compat__", True)
        torch.Tensor.__format__ = _wrapped
        _TORCH_TENSOR_FORMAT_COMPAT_APPLIED = True
        logger.info(
            "Applied torch compatibility shim: Tensor.__format__ numeric fallback"
        )

    @staticmethod
    def _ensure_nemo_speaker_utils_compat() -> None:
        """
        NeMo speaker-utils changed tensor/list shapes across versions.
        Old code paths can index assuming [start, end, speaker] and fail with
        "index ... out of bounds" on single-speaker / shape-mismatch inputs.
        """
        global _NEMO_SPEAKER_UTILS_COMPAT_APPLIED
        if _NEMO_SPEAKER_UTILS_COMPAT_APPLIED:
            return

        try:
            from nemo.collections.asr.parts.utils import speaker_utils  # type: ignore
        except Exception:
            return

        generate_fn = getattr(speaker_utils, "generate_speaker_timestamps", None)
        if not callable(generate_fn):
            return
        if getattr(generate_fn, "__mts_generate_speaker_timestamps_compat__", False):
            _NEMO_SPEAKER_UTILS_COMPAT_APPLIED = True
            return

        def _to_float_scalar(value: Any, default: float) -> float:
            try:
                if torch.is_tensor(value):
                    if int(value.numel()) <= 0:
                        return float(default)
                    return float(value.detach().reshape(-1)[0].cpu().item())
                if isinstance(value, np.ndarray):
                    if value.size <= 0:
                        return float(default)
                    return float(value.reshape(-1)[0].item())
                if isinstance(value, (list, tuple)):
                    if not value:
                        return float(default)
                    return _to_float_scalar(value[0], default)
                return float(value)
            except Exception:
                return float(default)

        def _to_int_scalar(value: Any, default: int = 0) -> int:
            try:
                if torch.is_tensor(value):
                    if int(value.numel()) <= 0:
                        return int(default)
                    return int(value.detach().reshape(-1)[0].cpu().item())
                if isinstance(value, np.ndarray):
                    if value.size <= 0:
                        return int(default)
                    return int(value.reshape(-1)[0].item())
                if isinstance(value, (list, tuple)):
                    if not value:
                        return int(default)
                    return _to_int_scalar(value[0], default)
                return int(value)
            except Exception:
                return int(default)

        def _normalize_preds(msdd_preds: Any) -> torch.Tensor:
            try:
                preds = torch.as_tensor(msdd_preds).detach().to("cpu", dtype=torch.float32)
            except Exception:
                return torch.zeros((1, 1), dtype=torch.float32)
            if preds.ndim == 0:
                return preds.reshape(1, 1)
            if preds.ndim == 1:
                return preds.reshape(-1, 1)
            if preds.ndim == 2:
                return preds
            if preds.ndim >= 3:
                if preds.shape[0] == 1:
                    preds = preds[0]
                else:
                    preds = preds.reshape(-1, preds.shape[-1])
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            elif preds.ndim != 2:
                try:
                    preds = preds.reshape(preds.shape[0], -1)
                except Exception:
                    preds = torch.zeros((1, 1), dtype=torch.float32)
            if preds.numel() <= 0:
                return torch.zeros((1, 1), dtype=torch.float32)
            return preds

        def _normalize_cluster_item(item: Any, seg_idx: int, default_step: float) -> Tuple[float, float, int]:
            # Expected format: [start, end, speaker]. Fallback for newer/variant shapes.
            if isinstance(item, dict):
                st = _to_float_scalar(item.get("start", seg_idx * default_step), seg_idx * default_step)
                ed = _to_float_scalar(item.get("end", st + default_step), st + default_step)
                spk = _to_int_scalar(item.get("speaker", item.get("label", 0)), 0)
                return st, ed, spk
            if isinstance(item, (list, tuple, np.ndarray)):
                arr = np.asarray(item).reshape(-1)
                if arr.size >= 3:
                    st = _to_float_scalar(arr[0], seg_idx * default_step)
                    ed = _to_float_scalar(arr[1], st + default_step)
                    spk = _to_int_scalar(arr[2], 0)
                    return st, ed, spk
                if arr.size == 2:
                    st = _to_float_scalar(arr[0], seg_idx * default_step)
                    ed = _to_float_scalar(arr[1], st + default_step)
                    return st, ed, 0
                if arr.size == 1:
                    st = float(seg_idx) * default_step
                    return st, st + default_step, _to_int_scalar(arr[0], 0)
            if torch.is_tensor(item):
                arr = item.detach().reshape(-1).to("cpu")
                if int(arr.numel()) >= 3:
                    st = _to_float_scalar(arr[0], seg_idx * default_step)
                    ed = _to_float_scalar(arr[1], st + default_step)
                    spk = _to_int_scalar(arr[2], 0)
                    return st, ed, spk
                if int(arr.numel()) == 2:
                    st = _to_float_scalar(arr[0], seg_idx * default_step)
                    ed = _to_float_scalar(arr[1], st + default_step)
                    return st, ed, 0
                if int(arr.numel()) == 1:
                    st = float(seg_idx) * default_step
                    return st, st + default_step, _to_int_scalar(arr[0], 0)
            st = float(seg_idx) * default_step
            return st, st + default_step, _to_int_scalar(item, 0)

        def _robust_generate(
            clus_labels: List[Any],
            msdd_preds: Any,
            **params: Any,
        ) -> Tuple[List[str], List[str]]:
            preds = _normalize_preds(msdd_preds)
            frame_count = int(preds.shape[0]) if preds.ndim >= 1 else 1
            est_spks = int(max(1, preds.shape[-1])) if preds.ndim >= 2 else 1
            default_step = 0.02

            threshold = _to_float_scalar(params.get("threshold", 0.5), 0.5)
            overlap_limit = max(2, _to_int_scalar(params.get("overlap_infer_spk_limit", 5), 5))
            use_adaptive = bool(params.get("use_adaptive_thres", False))
            if use_adaptive and est_spks >= 2 and overlap_limit > 2:
                try:
                    threshold = float(
                        speaker_utils.get_adaptive_threshold(est_spks, threshold, overlap_limit)
                    )
                except Exception:
                    pass

            use_clus_as_main = bool(params.get("use_clus_as_main", False))
            infer_overlap = est_spks >= 2 and est_spks < overlap_limit
            max_overlap_spks = max(1, _to_int_scalar(params.get("max_overlap_spks", 2), 2))

            labels = list(clus_labels or [])
            if not labels:
                return [], []

            main_speaker_lines: List[str] = []
            overlap_speaker_list: List[List[int]] = [[] for _ in range(est_spks)]

            max_segments = len(labels)
            last_frame_idx = max(0, frame_count - 1)
            for seg_idx in range(max_segments):
                st, ed, clus_spk = _normalize_cluster_item(labels[seg_idx], seg_idx, default_step)
                if ed <= st:
                    ed = st + default_step

                row = preds[min(seg_idx, last_frame_idx)]
                if row.ndim != 1:
                    row = row.reshape(-1)
                row_np = row.cpu().numpy()
                if row_np.size <= 0:
                    row_np = np.zeros((est_spks,), dtype=np.float32)

                if use_clus_as_main:
                    main_spk_idx = int(clus_spk)
                else:
                    main_spk_idx = int(np.argmax(row_np))
                main_spk_idx = max(0, min(main_spk_idx, est_spks - 1))

                spk_for_seg = (row_np > float(threshold)).astype(np.int32)
                if infer_overlap and int(np.sum(spk_for_seg)) > 1:
                    idx_arr = np.argsort(row_np)[::-1]
                    for ovl_spk_idx in idx_arr[:max_overlap_spks].tolist():
                        ovl_spk_idx = int(ovl_spk_idx)
                        if ovl_spk_idx == main_spk_idx:
                            continue
                        if 0 <= ovl_spk_idx < len(overlap_speaker_list):
                            overlap_speaker_list[ovl_spk_idx].append(seg_idx)

                main_speaker_lines.append(f"{st:.3f} {ed:.3f} speaker_{main_spk_idx}")

            if not main_speaker_lines:
                return [], []

            cont_stamps = speaker_utils.get_contiguous_stamps(main_speaker_lines)
            maj_labels = speaker_utils.merge_stamps(cont_stamps)
            ovl_labels = speaker_utils.get_overlap_stamps(cont_stamps, overlap_speaker_list)
            return maj_labels, ovl_labels

        def _wrapped_generate(clus_labels, msdd_preds, __fn=generate_fn, **params):
            try:
                return __fn(clus_labels, msdd_preds, **params)
            except Exception as e:
                if not Transcriber._is_nemo_msdd_index_oob_error(e):
                    raise
                global _NEMO_SPEAKER_UTILS_COMPAT_FALLBACK_LOGGED
                if not _NEMO_SPEAKER_UTILS_COMPAT_FALLBACK_LOGGED:
                    logger.info(
                        "Applied NeMo speaker_utils compatibility fallback due to index error: %s",
                        e,
                    )
                    _NEMO_SPEAKER_UTILS_COMPAT_FALLBACK_LOGGED = True
                return _robust_generate(clus_labels, msdd_preds, **params)

        setattr(_wrapped_generate, "__mts_generate_speaker_timestamps_compat__", True)
        speaker_utils.generate_speaker_timestamps = _wrapped_generate
        _NEMO_SPEAKER_UTILS_COMPAT_APPLIED = True
        logger.info(
            "Applied NeMo compatibility shim: speaker_utils.generate_speaker_timestamps indexing fallback"
        )

    @staticmethod
    def _ensure_nemo_frame_vad_state_dict_compat() -> None:
        """
        Some NeMo Frame VAD checkpoints (notably multilingual MarbleNet variants)
        can miss `loss.weight` under newer runtimes when loaded with strict=True.
        Retry once with strict=False for this known compatibility case across
        restore/from_pretrained and direct load_state_dict paths.
        """
        global _NEMO_FRAME_VAD_STATE_DICT_COMPAT_APPLIED
        if _NEMO_FRAME_VAD_STATE_DICT_COMPAT_APPLIED:
            return

        try:
            from nemo.collections.asr.models import EncDecFrameClassificationModel
        except Exception:
            return

        restore_fn = getattr(EncDecFrameClassificationModel, "restore_from", None)
        from_pretrained_fn = getattr(EncDecFrameClassificationModel, "from_pretrained", None)
        load_state_dict_fn = getattr(EncDecFrameClassificationModel, "load_state_dict", None)
        patched_any = False
        restore_already_patched = callable(restore_fn) and bool(
            getattr(restore_fn, "__mts_state_dict_compat_restore__", False)
        )
        from_pretrained_already_patched = callable(from_pretrained_fn) and bool(
            getattr(from_pretrained_fn, "__mts_state_dict_compat_from_pretrained__", False)
        )
        load_state_dict_already_patched = callable(load_state_dict_fn) and bool(
            getattr(load_state_dict_fn, "__mts_state_dict_compat_load_state_dict__", False)
        )

        if (
            restore_already_patched
            and from_pretrained_already_patched
            and load_state_dict_already_patched
        ):
            _NEMO_FRAME_VAD_STATE_DICT_COMPAT_APPLIED = True
            return

        if callable(restore_fn) and not restore_already_patched:
            def _wrapped_restore(*args, __fn=restore_fn, **kwargs):
                try:
                    return __fn(*args, **kwargs)
                except Exception as e:
                    if not Transcriber._is_nemo_frame_vad_state_dict_compat_error(e):
                        raise
                    if kwargs.get("strict", None) is False:
                        raise
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["strict"] = False
                    try:
                        logger.info(
                            "Applied NeMo Frame VAD compatibility retry: strict=False "
                            "(state_dict missing loss.weight)."
                        )
                        return __fn(*args, **retry_kwargs)
                    except Exception as retry_e:
                        if Transcriber._is_unexpected_kwarg_error(retry_e, "strict"):
                            raise e
                        raise

            setattr(_wrapped_restore, "__mts_state_dict_compat_restore__", True)
            EncDecFrameClassificationModel.restore_from = _wrapped_restore
            patched_any = True

        if callable(from_pretrained_fn) and not from_pretrained_already_patched:
            def _wrapped_from_pretrained(*args, __fn=from_pretrained_fn, **kwargs):
                try:
                    return __fn(*args, **kwargs)
                except Exception as e:
                    if not Transcriber._is_nemo_frame_vad_state_dict_compat_error(e):
                        raise
                    if kwargs.get("strict", None) is False:
                        raise
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["strict"] = False
                    try:
                        logger.info(
                            "Applied NeMo Frame VAD from_pretrained compatibility retry: "
                            "strict=False (state_dict missing loss.weight)."
                        )
                        return __fn(*args, **retry_kwargs)
                    except Exception as retry_e:
                        if Transcriber._is_unexpected_kwarg_error(retry_e, "strict"):
                            raise e
                        raise

            setattr(
                _wrapped_from_pretrained,
                "__mts_state_dict_compat_from_pretrained__",
                True,
            )
            EncDecFrameClassificationModel.from_pretrained = _wrapped_from_pretrained
            patched_any = True

        if callable(load_state_dict_fn) and not load_state_dict_already_patched:
            def _wrapped_load_state_dict(self, *args, __fn=load_state_dict_fn, **kwargs):
                strict_flag: Optional[bool]
                if "strict" in kwargs:
                    strict_flag = bool(kwargs.get("strict"))
                elif len(args) >= 2:
                    try:
                        strict_flag = bool(args[1])
                    except Exception:
                        strict_flag = True
                else:
                    strict_flag = True
                try:
                    return __fn(self, *args, **kwargs)
                except Exception as e:
                    if not Transcriber._is_nemo_frame_vad_state_dict_compat_error(e):
                        raise
                    if strict_flag is False:
                        raise
                    retry_args = list(args)
                    retry_kwargs = dict(kwargs)
                    if "strict" in retry_kwargs:
                        retry_kwargs["strict"] = False
                    elif len(retry_args) >= 2:
                        retry_args[1] = False
                    else:
                        retry_kwargs["strict"] = False
                    logger.info(
                        "Applied NeMo Frame VAD load_state_dict compatibility retry: "
                        "strict=False (state_dict missing loss.weight)."
                    )
                    return __fn(self, *retry_args, **retry_kwargs)

            setattr(
                _wrapped_load_state_dict,
                "__mts_state_dict_compat_load_state_dict__",
                True,
            )
            EncDecFrameClassificationModel.load_state_dict = _wrapped_load_state_dict
            patched_any = True

        if patched_any:
            _NEMO_FRAME_VAD_STATE_DICT_COMPAT_APPLIED = True
            logger.info(
                "Applied NeMo compatibility shim: EncDecFrameClassificationModel strict fallback"
            )

    @staticmethod
    def _ensure_nemo_msdd_mps_view_compat() -> None:
        """
        NeMo's MSDD module uses Tensor.view() on conv outputs that can become
        non-contiguous on MPS. Reshape has equivalent intent here and avoids the
        runtime failure without broad torch monkeypatching.
        """
        global _NEMO_MSDD_MPS_VIEW_COMPAT_APPLIED
        if _NEMO_MSDD_MPS_VIEW_COMPAT_APPLIED:
            return

        try:
            from nemo.collections.asr.modules.msdd_diarizer import MSDD_module
        except Exception:
            return

        conv_scale_fn = getattr(MSDD_module, "conv_scale_weights", None)
        cosine_fn = getattr(MSDD_module, "cosine_similarity", None)
        conv_scale_already_patched = callable(conv_scale_fn) and bool(
            getattr(conv_scale_fn, "__mts_msdd_mps_view_compat__", False)
        )
        cosine_already_patched = callable(cosine_fn) and bool(
            getattr(cosine_fn, "__mts_msdd_mps_view_compat__", False)
        )
        if conv_scale_already_patched and cosine_already_patched:
            _NEMO_MSDD_MPS_VIEW_COMPAT_APPLIED = True
            return

        patched_any = False

        if callable(conv_scale_fn) and not conv_scale_already_patched:
            def _wrapped_conv_scale_weights(self, ms_avg_embs_perm, ms_emb_seq_single):
                ms_cnn_input_seq = torch.cat([ms_avg_embs_perm, ms_emb_seq_single], dim=2)
                ms_cnn_input_seq = ms_cnn_input_seq.unsqueeze(2).flatten(0, 1)

                conv_out = self.conv_forward(
                    ms_cnn_input_seq,
                    conv_module=self.conv[0],
                    bn_module=self.conv_bn[0],
                    first_layer=True,
                )
                for conv_idx in range(1, self.conv_repeat + 1):
                    conv_out = self.conv_forward(
                        conv_input=conv_out,
                        conv_module=self.conv[conv_idx],
                        bn_module=self.conv_bn[conv_idx],
                        first_layer=False,
                    )

                lin_input_seq = conv_out.reshape(
                    self.batch_size,
                    self.length,
                    self.cnn_output_ch * self.emb_dim,
                )
                hidden_seq = self.conv_to_linear(lin_input_seq)
                hidden_seq = self.dropout(torch.nn.functional.leaky_relu(hidden_seq))
                scale_weights = self.softmax(self.linear_to_weights(hidden_seq))
                scale_weights = scale_weights.unsqueeze(3).expand(-1, -1, -1, self.num_spks)
                return scale_weights

            setattr(_wrapped_conv_scale_weights, "__mts_msdd_mps_view_compat__", True)
            MSDD_module.conv_scale_weights = _wrapped_conv_scale_weights
            patched_any = True

        if callable(cosine_fn) and not cosine_already_patched:
            def _wrapped_cosine_similarity(self, scale_weights, ms_avg_embs, _ms_emb_seq):
                cos_dist_seq = self.cos_dist(_ms_emb_seq, ms_avg_embs)
                context_vectors = torch.mul(scale_weights, cos_dist_seq)
                context_vectors = context_vectors.reshape(self.batch_size, self.length, -1)
                context_emb = self.dist_to_emb(context_vectors)
                return context_emb

            setattr(_wrapped_cosine_similarity, "__mts_msdd_mps_view_compat__", True)
            MSDD_module.cosine_similarity = _wrapped_cosine_similarity
            patched_any = True

        if patched_any:
            _NEMO_MSDD_MPS_VIEW_COMPAT_APPLIED = True
            logger.info(
                "Applied NeMo MSDD compatibility shim: MPS-safe reshape for non-contiguous view paths"
            )

    def _ensure_diarization_dependency_compat(self) -> None:
        self._ensure_signal_sigkill_compat()
        self._ensure_hf_hub_auth_token_kw_compat()
        self._ensure_torchaudio_speechbrain_compat()
        self._ensure_windows_symlink_copy_compat()
        self._ensure_speechbrain_fetch_windows_compat()
        self._ensure_torchmetrics_get_num_classes_compat()
        self._ensure_pytorch_lightning_model_summary_compat()
        self._ensure_pyannote_speaker_diar_plda_compat()
        self._ensure_pyannote_model_version_compat()
        self._ensure_lightning_torch_load_weights_only_compat()
        self._ensure_torch_load_weights_only_compat()
        self._ensure_torch_tensor_format_compat()
        self._ensure_nemo_speaker_utils_compat()
        self._ensure_nemo_frame_vad_state_dict_compat()
        self._ensure_nemo_msdd_mps_view_compat()

    def _pyannote_pipeline_from_pretrained_compat(
        self,
        model_name: str,
        token: str = "",
        trust_remote_code: bool = True,
    ) -> Any:
        self._ensure_diarization_dependency_compat()
        model_name = self._prepare_pyannote_local_checkpoint_path(model_name)
        from pyannote.audio import Pipeline

        token = str(token or "").strip()
        trust_remote = bool(trust_remote_code)
        errors: List[Exception] = []

        supports_token_kw = True
        supports_use_auth_token_kw = False
        try:
            sig = inspect.signature(Pipeline.from_pretrained)
            params = set(sig.parameters.keys())
            supports_token_kw = "token" in params
            supports_use_auth_token_kw = "use_auth_token" in params
        except Exception:
            pass

        def _call_with_optional_trust(*args, **kwargs):
            kw = dict(kwargs)
            kw["trust_remote_code"] = trust_remote
            try:
                return Pipeline.from_pretrained(*args, **kw)
            except Exception as e:
                if self._is_unexpected_kwarg_error(e, "trust_remote_code"):
                    kw.pop("trust_remote_code", None)
                    return Pipeline.from_pretrained(*args, **kw)
                raise

        variants: List[Dict[str, Any]] = []
        if token:
            if supports_token_kw:
                variants.append({"token": token})
            if supports_use_auth_token_kw:
                variants.append({"use_auth_token": token})
        variants.append({})

        deduped_variants: List[Dict[str, Any]] = []
        seen_variant_keys: set[Tuple[Tuple[str, str], ...]] = set()
        for item in variants:
            normalized = tuple(sorted((str(k), str(v)) for k, v in item.items()))
            if normalized in seen_variant_keys:
                continue
            seen_variant_keys.add(normalized)
            deduped_variants.append(item)

        for call_kwargs in deduped_variants:
            retried_plda_compat = False
            retried_symlink_compat = False
            while True:
                try:
                    pipeline = _call_with_optional_trust(model_name, **call_kwargs)
                    return self._ensure_pyannote_pipeline_instantiated(pipeline)
                except Exception as e:
                    if (
                        not retried_plda_compat
                        and self._is_unexpected_kwarg_error(e, "plda")
                    ):
                        retried_plda_compat = True
                        self._ensure_pyannote_speaker_diar_plda_compat(exc=e, force=True)
                        logger.info(
                            "Detected pyannote 'plda' kwarg incompatibility. "
                            "Applied runtime shim and retrying once."
                        )
                        continue

                    if (
                        not retried_symlink_compat
                        and self._is_windows_symlink_privilege_error(e)
                    ):
                        retried_symlink_compat = True
                        self._ensure_windows_symlink_copy_compat()
                        self._ensure_speechbrain_fetch_windows_compat()
                        logger.info(
                            "Detected WinError 1314 during pyannote load. "
                            "Applied copy fallback and retrying once."
                        )
                        continue

                    errors.append(e)
                    if call_kwargs:
                        marker = ", ".join(f"{k}=..." for k in sorted(call_kwargs.keys()))
                        logger.debug(f"pyannote {marker} load failed: {e}")
                    break

        detail = " | ".join(str(err)[:220] for err in errors[-3:])
        raise RuntimeError(detail or f"Pipeline.from_pretrained failed: {model_name}")

    def _load_pyannote_pipeline_with_retry(
        self,
        model_name: str,
        *,
        cfg: Optional[Dict[str, Any]] = None,
        token: str = "",
        purpose: str = "pyannote",
    ) -> Any:
        attempts, retry_wait_sec = self._build_hf_download_attempts(cfg=cfg)
        total_attempts = sum(max(1, int(item.get("retries", 1))) for item in attempts)
        attempt_index = 0
        last_error: Optional[Exception] = None
        token = str(token or "").strip()
        trust_remote_code = True
        if isinstance(cfg, dict):
            trust_remote_code = self._safe_bool(
                cfg.get("trust_remote_code", True),
                True,
            )
        stop_retry_due_to_known_error = False
        prepared_source = str(model_name or "").strip()
        try:
            prepared_source = self._prepare_pyannote_pipeline_source(
                model_name=model_name,
                cfg=cfg,
                token=token,
                purpose=purpose,
            )
        except Exception as e:
            last_error = e
            logger.info(
                "%s local snapshot preparation failed for %s; falling back to direct load: %s",
                purpose,
                model_name,
                e,
            )

        if prepared_source and prepared_source != str(model_name or "").strip():
            try:
                return self._pyannote_pipeline_from_pretrained_compat(
                    model_name=prepared_source,
                    token=token,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as e:
                last_error = e
                logger.info(
                    "%s local snapshot load failed for %s; retrying direct repo load: %s",
                    purpose,
                    model_name,
                    e,
                )
                if self._is_pyannote_known_non_retryable_error(e):
                    logger.info(
                        "%s local snapshot hit a non-retryable error; "
                        "direct repo load is unlikely to help but will be attempted once per route.",
                        purpose,
                    )

        for item in attempts:
            source = str(item.get("source", "official") or "official")
            endpoint = str(item.get("endpoint", "") or "").strip()
            retries = max(1, int(item.get("retries", 1)))

            for _ in range(retries):
                attempt_index += 1
                prev_endpoint = os.getenv("HF_ENDPOINT")
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint
                elif prev_endpoint is not None:
                    os.environ.pop("HF_ENDPOINT", None)

                try:
                    logger.info(
                        "  %s load attempt %d/%d via %s: %s",
                        purpose,
                        attempt_index,
                        total_attempts,
                        source,
                        endpoint or "<default>",
                    )
                    return self._pyannote_pipeline_from_pretrained_compat(
                        model_name=model_name,
                        token=token,
                        trust_remote_code=trust_remote_code,
                    )
                except Exception as e:
                    last_error = e
                    logger.debug(
                        "%s load failed (%d/%d) via %s: %s",
                        purpose,
                        attempt_index,
                        total_attempts,
                        source,
                        e,
                    )
                    if self._is_pyannote_known_non_retryable_error(e):
                        stop_retry_due_to_known_error = True
                        logger.info(
                            "%s model hit a non-retryable error; "
                            "skip further retries for this candidate: %s",
                            purpose,
                            e,
                        )
                    if (
                        retry_wait_sec > 0
                        and attempt_index < total_attempts
                        and not stop_retry_due_to_known_error
                    ):
                        time.sleep(retry_wait_sec)
                finally:
                    if prev_endpoint is None:
                        os.environ.pop("HF_ENDPOINT", None)
                    else:
                        os.environ["HF_ENDPOINT"] = prev_endpoint
                if stop_retry_due_to_known_error:
                    break
            if stop_retry_due_to_known_error:
                break

        raise RuntimeError(
            f"{purpose} load failed for {model_name} after "
            f"{attempt_index} attempts: {last_error}"
        )

    def _set_default_hf_endpoint(self):
        """
        Set default HF endpoint using current dynamic route priority.
        """
        attempts, _retry_wait = self._build_hf_download_attempts()
        if not attempts:
            return
        preferred = str(attempts[0].get("endpoint", "") or "").strip()
        if not preferred:
            return

        current = os.getenv("HF_ENDPOINT", "").strip()
        if current == preferred:
            return
        os.environ["HF_ENDPOINT"] = preferred
        logger.info(f"HF endpoint default set to: {preferred}")

    def _nemo_msdd_cfg(self) -> Dict[str, Any]:
        return self.asr_cfg.get("nemo_msdd", {}) or {}

    def _mark_nemo_msdd_model_unavailable(self, reason: str):
        text = str(reason or "NeMo MSDD model unavailable")
        if (
            self._nemo_msdd_model_unavailable
            and self._nemo_msdd_model_unavailable_reason == text
        ):
            return
        self._nemo_msdd_model_unavailable = True
        self._nemo_msdd_model_unavailable_reason = text
        self._nemo_msdd_runtime_error = text
        logger.info("NeMo MSDD model unavailable for current run: %s", text)

    def _disable_nemo_msdd_for_session(self, reason: str):
        self._nemo_msdd_disabled = True
        self._nemo_msdd_disabled_reason = str(reason or "disabled")
        logger.warning(
            "NeMo MSDD disabled for current run: "
            f"{self._nemo_msdd_disabled_reason}"
        )

    def use_nemo_msdd_pipeline(self) -> bool:
        if self._nemo_msdd_disabled:
            return False
        return bool(self._nemo_msdd_cfg().get("enabled", False))

    def _nemo_msdd_model_is_available(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._nemo_msdd_model_unavailable:
            return False

        runtime_cfg = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        model_name = str(
            runtime_cfg.get("model_path", "diar_msdd_telephonic") or ""
        ).strip()
        if not model_name:
            return False

        source = self._nemo_resolve_model_source(
            kind="msdd",
            model_name=model_name,
            cfg=runtime_cfg,
        )
        if not source:
            return True

        local_path = str(source.get("local_path", "") or "").strip()
        if local_path:
            path_obj = Path(local_path).expanduser()
            if path_obj.exists() and path_obj.is_file():
                valid, reason = _inspect_nemo_artifact(path_obj)
                if valid:
                    return True
                self._mark_nemo_msdd_model_unavailable(
                    f"Configured NeMo MSDD model path is not a valid archive: {path_obj} ({reason})"
                )
                return False
            self._mark_nemo_msdd_model_unavailable(
                f"Configured NeMo MSDD model path does not exist: {path_obj}"
            )
            return False

        repo_id = str(source.get("repo_id", "") or "").strip()
        if repo_id:
            return True

        url = str(source.get("url", "") or "").strip()
        if not url:
            return True

        filename = str(source.get("filename", "") or "").strip()
        if not filename:
            parsed = urllib.parse.urlparse(url)
            filename = Path(parsed.path).name or "diar_msdd_telephonic.nemo"
        cached_artifact = self._find_local_nemo_artifact(filename)
        if cached_artifact is not None:
            return True

        requires_ngc_api_key = self._safe_bool(
            source.get("requires_ngc_api_key", False),
            False,
        )
        if requires_ngc_api_key and not self._resolve_ngc_api_key(runtime_cfg):
            self._mark_nemo_msdd_model_unavailable(
                str(
                    self._nemo_auth_required_error(
                        kind="msdd",
                        model_name=model_name,
                        filename=filename,
                    )
                )
            )
            return False
        return True

    @staticmethod
    def _safe_stem(file_name: str) -> str:
        raw = Path(file_name or "audio").stem
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._-")
        return safe or "audio"

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _safe_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    def _nemo_overlap_cfg(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        root = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        overlap_cfg = root.get("overlap_handling", {}) if isinstance(root, dict) else {}
        return overlap_cfg if isinstance(overlap_cfg, dict) else {}

    def _allow_cpu_retry_after_mps_failure(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        default_value = not self.is_macos
        section_cfg = cfg if isinstance(cfg, dict) else {}
        if "retry_cpu_on_mps_failure" in section_cfg:
            return self._safe_bool(
                section_cfg.get("retry_cpu_on_mps_failure", default_value),
                default_value,
            )
        root = self._nemo_msdd_cfg()
        if isinstance(root, dict) and "retry_cpu_on_mps_failure" in root:
            return self._safe_bool(
                root.get("retry_cpu_on_mps_failure", default_value),
                default_value,
            )
        return default_value

    def _nemo_final_diarization_strategy(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> str:
        root = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        raw = str((root or {}).get("final_strategy", "msdd_primary") or "").strip().lower()
        if raw in {"msdd", "msdd_primary", "msdd-first", "msdd_first"}:
            return "msdd_primary"
        if raw in {"msdd_only", "msdd-only"}:
            return "msdd_only"
        if raw in {"hybrid", "hybrid_gpu_first", "hybrid-gpu-first", "hybrid_first"}:
            return "hybrid"
        if raw in {"sortformer", "sortformer_primary", "sortformer-first", "sortformer_first"}:
            return "sortformer_primary"
        return "msdd_primary"

    def _nemo_sortformer_cfg(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        root = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        sf_cfg = root.get("sortformer", {}) if isinstance(root, dict) else {}
        return sf_cfg if isinstance(sf_cfg, dict) else {}

    @staticmethod
    def _parse_sortformer_segments(
        raw_output: Any,
        *,
        min_turn_sec: float = 0.05,
    ) -> List[Dict[str, Any]]:
        content = raw_output
        if isinstance(content, tuple):
            content = content[0] if content else []
        if not isinstance(content, (list, tuple)):
            return []

        if (
            len(content) == 1
            and isinstance(content[0], (list, tuple))
            and content[0]
            and isinstance(content[0][0], (list, tuple, dict))
        ):
            content = content[0]

        diar_segments: List[Dict[str, Any]] = []
        min_turn = max(0.01, float(min_turn_sec))

        for item in content:
            start: Optional[float] = None
            end: Optional[float] = None
            speaker = "0"

            if isinstance(item, dict):
                try:
                    start = float(item.get("start"))
                    end = float(item.get("end"))
                except Exception:
                    start = None
                    end = None
                speaker = str(
                    item.get("speaker", item.get("label", item.get("spk", "0")))
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    start = float(item[0])
                    end = float(item[1])
                except Exception:
                    start = None
                    end = None
                speaker = str(item[2])
            elif (
                hasattr(item, "start")
                and hasattr(item, "end")
                and hasattr(item, "speaker")
            ):
                try:
                    start = float(getattr(item, "start"))
                    end = float(getattr(item, "end"))
                except Exception:
                    start = None
                    end = None
                speaker = str(getattr(item, "speaker"))

            if start is None or end is None:
                continue
            if end - start < min_turn:
                continue
            diar_segments.append(
                {
                    "start": max(0.0, float(start)),
                    "end": max(float(start), float(end)),
                    "speaker": str(speaker if speaker else "0"),
                }
            )
        diar_segments.sort(key=lambda x: (float(x["start"]), float(x["end"]), str(x["speaker"])))
        return diar_segments

    def _run_sortformer_manifest_inference(
        self,
        *,
        model: Any,
        audio_np: np.ndarray,
        sample_rate: int,
        segments: List[TranscriptionSegment],
        batch_size: int,
        sf_cfg: Dict[str, Any],
        base_cfg: Optional[Dict[str, Any]] = None,
        file_name: str = "",
    ) -> List[Dict[str, Any]]:
        try:
            from omegaconf import OmegaConf
        except Exception as e:
            raise RuntimeError(f"sortformer-manifest-missing-omegaconf: {e}") from e

        audio_f32 = self._audio_to_numpy(audio_np)
        if audio_f32.size == 0:
            return []

        duration = len(audio_f32) / float(sample_rate)
        if duration <= 0:
            return []

        safe_stem = self._safe_stem(file_name)
        run_id = f"sortformer_{safe_stem}_{int(time.time() * 1000)}_{os.getpid()}"
        run_dir = self._nemo_run_root() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        keep_temp = self._safe_bool(
            self._nemo_msdd_cfg().get("keep_temp_files", False),
            False,
        )

        try:
            wav_path = run_dir / f"{safe_stem}.wav"
            self._write_wav_mono(wav_path, audio_f32, sample_rate)

            empty_rttm_path = run_dir / "empty.rttm"
            empty_rttm_path.write_text("", encoding="utf-8")

            default_window_sec = 90.0
            try:
                default_window_sec = self._safe_float(
                    getattr(getattr(model, "cfg", None), "test_ds", {}).get(
                        "session_len_sec",
                        90.0,
                    ),
                    90.0,
                )
            except Exception:
                default_window_sec = 90.0
            window_sec = max(
                15.0,
                self._safe_float(
                    sf_cfg.get("session_len_sec", default_window_sec),
                    default_window_sec,
                ),
            )

            manifest_items: List[Dict[str, Any]] = []
            root_cfg = base_cfg if isinstance(base_cfg, dict) else self._nemo_msdd_cfg()
            fixed_num = max(
                0,
                self._safe_int(
                    sf_cfg.get("num_speakers", root_cfg.get("num_speakers", 0)),
                    self._safe_int(root_cfg.get("num_speakers", 0), 0),
                ),
            )
            offset_sec = 0.0
            chunk_index = 0
            while offset_sec < duration - 1e-6:
                chunk_duration = min(window_sec, duration - offset_sec)
                start_ms = int(round(offset_sec * 1000.0))
                end_ms = int(round((offset_sec + chunk_duration) * 1000.0))
                manifest_items.append(
                    {
                        "audio_filepath": str(wav_path),
                        "offset": round(offset_sec, 3),
                        "duration": round(chunk_duration, 3),
                        "label": "infer",
                        "text": "-",
                        "rttm_filepath": str(empty_rttm_path),
                        "uem_filepath": "",
                        "uniq_id": f"{safe_stem}_{start_ms}_{end_ms}_{chunk_index}",
                    }
                )
                if fixed_num > 0:
                    manifest_items[-1]["num_speakers"] = int(fixed_num)
                offset_sec += chunk_duration
                chunk_index += 1

            manifest_path = run_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                for item in manifest_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            try:
                test_cfg = OmegaConf.create(
                    OmegaConf.to_container(
                        getattr(model.cfg, "test_ds"),
                        resolve=False,
                    )
                )
            except Exception:
                test_cfg = OmegaConf.create({})

            test_cfg.manifest_filepath = str(manifest_path)
            test_cfg.batch_size = int(max(1, batch_size))
            test_cfg.num_workers = 0 if sys.platform == "darwin" or os.name == "nt" else 1
            test_cfg.pin_memory = False
            test_cfg.shuffle = False
            test_cfg.seq_eval_mode = True
            test_cfg.validation_mode = True
            test_cfg.session_len_sec = -1
            if "use_lhotse" in test_cfg:
                test_cfg.use_lhotse = False

            model.setup_test_data(test_cfg)
            model.eval()
            with torch.inference_mode():
                model.test_batch()

            preds_list = list(getattr(model, "preds_total_list", []) or [])
            if not preds_list:
                return []

            try:
                dataset = model.test_dataloader().dataset
                diar_frame_length = self._safe_float(
                    getattr(dataset, "diar_frame_length", 0.08),
                    0.08,
                )
            except Exception:
                diar_frame_length = 0.08

            speech_regions = self._merge_time_regions(
                [
                    {"start": float(seg.start), "end": float(seg.end)}
                    for seg in segments or []
                    if float(getattr(seg, "end", 0.0)) > float(getattr(seg, "start", 0.0))
                ],
                max_gap_sec=0.12,
                min_duration_sec=0.05,
                max_end_sec=duration,
            )

            frame_segments: List[Dict[str, Any]] = []
            for manifest_item, pred_tensor in zip(manifest_items, preds_list):
                if not torch.is_tensor(pred_tensor):
                    pred_tensor = torch.as_tensor(pred_tensor)
                pred_tensor = pred_tensor.detach().float().cpu()
                if pred_tensor.ndim == 3 and pred_tensor.shape[0] == 1:
                    pred_tensor = pred_tensor[0]
                if pred_tensor.ndim != 2 or pred_tensor.shape[0] <= 0:
                    continue

                offset_base = self._safe_float(manifest_item.get("offset", 0.0), 0.0)
                speaker_ids = torch.argmax(pred_tensor, dim=-1).tolist()
                for frame_index, speaker_idx in enumerate(speaker_ids):
                    abs_start = offset_base + frame_index * diar_frame_length
                    abs_end = min(
                        duration,
                        offset_base + (frame_index + 1) * diar_frame_length,
                    )
                    if abs_end <= abs_start:
                        continue
                    if speech_regions and not any(
                        self._segment_overlap_seconds(
                            abs_start,
                            abs_end,
                            float(region.get("start", 0.0)),
                            float(region.get("end", 0.0)),
                        ) > 0.0
                        for region in speech_regions
                    ):
                        continue
                    frame_segments.append(
                        {
                            "start": abs_start,
                            "end": abs_end,
                            "speaker": str(int(speaker_idx)),
                        }
                    )

            if not frame_segments:
                return []

            min_turn_sec = max(
                0.01,
                self._safe_float(sf_cfg.get("min_turn_sec", 0.05), 0.05),
            )
            merge_gap_sec = max(
                0.0,
                self._safe_float(sf_cfg.get("merge_gap_sec", 0.08), 0.08),
            )

            merged_segments: List[Dict[str, Any]] = []
            for item in frame_segments:
                start = float(item.get("start", 0.0))
                end = max(start, float(item.get("end", start)))
                speaker = str(item.get("speaker", "0"))
                if (
                    merged_segments
                    and str(merged_segments[-1].get("speaker", "")) == speaker
                    and start <= float(merged_segments[-1].get("end", 0.0)) + merge_gap_sec
                ):
                    merged_segments[-1]["end"] = max(
                        float(merged_segments[-1]["end"]),
                        end,
                    )
                    continue
                merged_segments.append(
                    {"start": start, "end": end, "speaker": speaker}
                )

            diar_segments = [
                item
                for item in merged_segments
                if float(item.get("end", 0.0)) - float(item.get("start", 0.0)) >= min_turn_sec
            ]
            return diar_segments
        finally:
            if not keep_temp:
                shutil.rmtree(run_dir, ignore_errors=True)

    def _diarize_audio_nemo_sortformer(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        segments: List[TranscriptionSegment],
        cfg: Optional[Dict[str, Any]] = None,
        file_name: str = "",
    ) -> List[Dict[str, Any]]:
        base_cfg = dict(cfg or self._nemo_msdd_cfg())
        sf_cfg = self._nemo_sortformer_cfg(cfg)
        if not self._safe_bool(sf_cfg.get("enabled", True), True):
            return []
        if sample_rate <= 0 or audio_np is None or len(segments or []) < 2:
            return []
        if self._nemo_sortformer_incompatible:
            return []

        model_name = str(
            sf_cfg.get("model_name", "nvidia/diar_streaming_sortformer_4spk-v2.1")
            or "nvidia/diar_streaming_sortformer_4spk-v2.1"
        ).strip()
        if not model_name:
            model_name = "nvidia/diar_streaming_sortformer_4spk-v2.1"
        model_name_l = model_name.lower()
        runtime_supports_spkcache_len = self._sortformer_modules_supports_kwarg("spkcache_len")
        if (
            not runtime_supports_spkcache_len
            and "diar_streaming_sortformer_4spk-v2" in model_name_l
        ):
            compat_model_name = "nvidia/diar_sortformer_4spk-v1"
            logger.info(
                "NeMo sortformer runtime does not support 'spkcache_len'; "
                "switching model from %s to %s for compatibility.",
                model_name,
                compat_model_name,
            )
            model_name = compat_model_name

        device_pref = str(sf_cfg.get("device", "auto") or "auto").strip().lower()
        runtime_device = self._preferred_torch_device(device_pref, allow_mps=True)
        batch_size = max(1, self._safe_int(sf_cfg.get("batch_size", 1), 1))
        setup = {
            "model_name": model_name,
            "device": runtime_device,
            "batch_size": batch_size,
        }

        self._ensure_diarization_dependency_compat()
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except Exception as e:
            self._nemo_sortformer_runtime_error = f"sortformer-import-failed: {e}"
            return []

        try:
            if self._nemo_sortformer_model is None or self._nemo_sortformer_setup != setup:
                model = None
                token = str(sf_cfg.get("hf_token", "") or "").strip()
                if not token:
                    token = self._resolve_hf_token(self.asr_cfg.get("faster_whisper", {}) or {})
                load_map_location = "cpu" if runtime_device == "mps" else runtime_device

                local_model_path = self._prepare_nemo_model_path(
                    kind="sortformer",
                    model_name=model_name,
                    cfg=sf_cfg,
                    purpose="NeMo sortformer",
                )
                local_restore_error: Optional[Exception] = None
                if local_model_path:
                    try:
                        model = SortformerEncLabelModel.restore_from(
                            restore_path=str(local_model_path),
                            map_location=load_map_location,
                        )
                    except Exception as e:
                        local_restore_error = e
                        if self._is_sortformer_module_kwarg_incompat_error(e):
                            raise
                        logger.info(
                            "  NeMo sortformer local restore failed, "
                            "falling back to from_pretrained: %s",
                            e,
                        )
                if model is None:
                    attempts, retry_wait_sec = self._build_hf_download_attempts(cfg=sf_cfg)
                    total_attempts = sum(max(1, int(item.get("retries", 1))) for item in attempts)
                    attempt_index = 0
                    last_error: Optional[Exception] = None
                    for item in attempts:
                        source = str(item.get("source", "official") or "official")
                        endpoint = str(item.get("endpoint", "") or "").strip()
                        retries = max(1, int(item.get("retries", 1)))
                        for _ in range(retries):
                            attempt_index += 1
                            prev_endpoint = os.getenv("HF_ENDPOINT")
                            if endpoint:
                                os.environ["HF_ENDPOINT"] = endpoint
                            elif prev_endpoint is not None:
                                os.environ.pop("HF_ENDPOINT", None)
                            try:
                                logger.info(
                                    "  NeMo sortformer load attempt %d/%d via %s: %s",
                                    attempt_index,
                                    total_attempts,
                                    source,
                                    endpoint or "<default>",
                                )
                                variants: List[Dict[str, Any]] = [
                                    {
                                        "model_name": model_name,
                                        "token": (token or None),
                                        "map_location": load_map_location,
                                    },
                                    {
                                        "model_name": model_name,
                                        "map_location": load_map_location,
                                    },
                                    {
                                        "model_name": model_name,
                                        "token": (token or None),
                                    },
                                    {"model_name": model_name},
                                ]
                                variant_error: Optional[Exception] = None
                                model = None
                                for call_kwargs in variants:
                                    try:
                                        model = SortformerEncLabelModel.from_pretrained(**call_kwargs)
                                        variant_error = None
                                        break
                                    except Exception as inner_e:
                                        variant_error = inner_e
                                        msg = str(inner_e or "")
                                        if "unexpected keyword argument" in msg:
                                            continue
                                        raise
                                if model is None and variant_error is not None:
                                    raise variant_error
                                last_error = None
                                break
                            except Exception as e:
                                last_error = e
                                if retry_wait_sec > 0 and attempt_index < total_attempts:
                                    time.sleep(retry_wait_sec)
                            finally:
                                if prev_endpoint is None:
                                    os.environ.pop("HF_ENDPOINT", None)
                                else:
                                    os.environ["HF_ENDPOINT"] = prev_endpoint
                        if model is not None:
                            break
                    if model is None:
                        if local_restore_error is not None and last_error is None:
                            raise local_restore_error
                        if last_error is not None:
                            raise last_error

                if model is None:
                    raise RuntimeError("Sortformer model init returned None")
                if hasattr(model, "to"):
                    model = model.to(torch.device(runtime_device))
                if hasattr(model, "eval"):
                    model = model.eval()

                modules = getattr(model, "sortformer_modules", None)
                if modules is not None:
                    attr_map = {
                        "chunk_len": "chunk_len",
                        "chunk_right_context": "chunk_right_context",
                        "fifo_len": "fifo_len",
                        "spkcache_update_period": "spkcache_update_period",
                        "spkcache_len": "spkcache_len",
                    }
                    for cfg_key, attr_name in attr_map.items():
                        if not hasattr(modules, attr_name):
                            continue
                        raw_value = sf_cfg.get(cfg_key, None)
                        if raw_value is None:
                            continue
                        value = self._safe_int(raw_value, -1)
                        if value > 0:
                            setattr(modules, attr_name, int(value))
                    if hasattr(modules, "_check_streaming_parameters"):
                        try:
                            modules._check_streaming_parameters()
                        except Exception:
                            pass

                self._nemo_sortformer_model = model
                self._nemo_sortformer_setup = dict(setup)
                self._nemo_sortformer_runtime_error = ""
        except Exception as e:
            if self._is_sortformer_module_kwarg_incompat_error(e):
                bad_kw = self._extract_unexpected_kwarg(e)
                reason = (
                    f"SortformerModules does not support '{bad_kw}' in current NeMo runtime"
                    if bad_kw
                    else "SortformerModules config is incompatible with current NeMo runtime"
                )
                self._nemo_sortformer_incompatible = True
                self._nemo_sortformer_incompatible_reason = reason
                self._nemo_sortformer_runtime_error = f"sortformer-incompatible: {reason}"
                logger.info(
                    "NeMo sortformer diarization disabled for this session: %s. "
                    "Falling back to MSDD diarization.",
                    reason,
                )
                return []
            self._nemo_sortformer_runtime_error = f"sortformer-init-failed: {e}"
            logger.warning(f"NeMo sortformer diarization init failed: {e}")
            logger.debug(traceback.format_exc())
            if runtime_device == "mps" and not self._safe_bool(sf_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(sf_cfg):
                    logger.warning(
                        "NeMo sortformer MPS init failed and CPU retry is disabled; "
                        "continuing with the next diarization route. error=%s",
                        e,
                    )
                    return []
                logger.info("NeMo sortformer MPS init failed; retrying once on CPU.")
                retry_sf_cfg = dict(sf_cfg)
                retry_sf_cfg["device"] = "cpu"
                retry_sf_cfg["__cpu_retry_done"] = True
                retry_cfg = dict(base_cfg)
                retry_cfg["sortformer"] = retry_sf_cfg
                return self._diarize_audio_nemo_sortformer(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                )
            return []

        if self._nemo_sortformer_model is None:
            return []

        audio_f32 = self._audio_to_numpy(audio_np)
        if audio_f32.size == 0:
            return []

        try:
            if self._sortformer_diarize_supports_audio_input(type(self._nemo_sortformer_model)):
                fixed_num = max(
                    0,
                    self._safe_int(
                        sf_cfg.get("num_speakers", base_cfg.get("num_speakers", 0)),
                        self._safe_int(base_cfg.get("num_speakers", 0), 0),
                    ),
                )
                speaker_kwarg = ""
                if fixed_num > 0:
                    speaker_kwarg = self._sortformer_preferred_speaker_kwarg(
                        type(self._nemo_sortformer_model)
                    )
                diarize_kwargs: Dict[str, Any] = {
                    "audio": [audio_f32],
                    "batch_size": int(batch_size),
                    "sample_rate": int(sample_rate),
                }
                if fixed_num > 0 and speaker_kwarg:
                    diarize_kwargs[speaker_kwarg] = int(fixed_num)
                with torch.inference_mode():
                    try:
                        raw_output = self._nemo_sortformer_model.diarize(**diarize_kwargs)
                    except TypeError as e:
                        if speaker_kwarg and self._is_unexpected_kwarg_error(e, speaker_kwarg):
                            logger.info(
                                "NeMo sortformer runtime rejected speaker-count kwarg '%s'; retrying without it%s.",
                                speaker_kwarg,
                                f" ({file_name})" if file_name else "",
                            )
                            retry_kwargs = dict(diarize_kwargs)
                            retry_kwargs.pop(speaker_kwarg, None)
                            try:
                                raw_output = self._nemo_sortformer_model.diarize(**retry_kwargs)
                            except TypeError as inner_e:
                                retry_kwargs.pop("sample_rate", None)
                                try:
                                    raw_output = self._nemo_sortformer_model.diarize(**retry_kwargs)
                                except Exception as final_e:
                                    if not self._is_sortformer_audio_path_only_error(final_e):
                                        raise
                                    logger.info(
                                        "NeMo sortformer runtime rejected in-memory audio; "
                                        "switching to manifest-based inference%s.",
                                        f" ({file_name})" if file_name else "",
                                    )
                                    diar_segments = self._run_sortformer_manifest_inference(
                                        model=self._nemo_sortformer_model,
                                        audio_np=audio_f32,
                                        sample_rate=sample_rate,
                                        segments=segments,
                                        batch_size=batch_size,
                                        sf_cfg=sf_cfg,
                                        base_cfg=base_cfg,
                                        file_name=file_name,
                                    )
                                    raw_output = None
                        elif self._is_sortformer_audio_path_only_error(e):
                            logger.info(
                                "NeMo sortformer runtime rejected in-memory audio; "
                                "switching to manifest-based inference%s.",
                                f" ({file_name})" if file_name else "",
                            )
                            diar_segments = self._run_sortformer_manifest_inference(
                                model=self._nemo_sortformer_model,
                                audio_np=audio_f32,
                                sample_rate=sample_rate,
                                segments=segments,
                                batch_size=batch_size,
                                sf_cfg=sf_cfg,
                                base_cfg=base_cfg,
                                file_name=file_name,
                            )
                            raw_output = None
                        else:
                            diarize_kwargs.pop("sample_rate", None)
                            try:
                                raw_output = self._nemo_sortformer_model.diarize(**diarize_kwargs)
                            except Exception as inner_e:
                                if not self._is_sortformer_audio_path_only_error(inner_e):
                                    raise
                                logger.info(
                                    "NeMo sortformer runtime rejected in-memory audio; "
                                    "switching to manifest-based inference%s.",
                                    f" ({file_name})" if file_name else "",
                                )
                                diar_segments = self._run_sortformer_manifest_inference(
                                    model=self._nemo_sortformer_model,
                                    audio_np=audio_f32,
                                    sample_rate=sample_rate,
                                    segments=segments,
                                    batch_size=batch_size,
                                    sf_cfg=sf_cfg,
                                    base_cfg=base_cfg,
                                    file_name=file_name,
                                )
                                raw_output = None
                    except Exception as e:
                        if not self._is_sortformer_audio_path_only_error(e):
                            raise
                        logger.info(
                            "NeMo sortformer runtime rejected in-memory audio; "
                            "switching to manifest-based inference%s.",
                            f" ({file_name})" if file_name else "",
                        )
                        diar_segments = self._run_sortformer_manifest_inference(
                            model=self._nemo_sortformer_model,
                            audio_np=audio_f32,
                            sample_rate=sample_rate,
                            segments=segments,
                            batch_size=batch_size,
                            sf_cfg=sf_cfg,
                            base_cfg=base_cfg,
                            file_name=file_name,
                        )
                        raw_output = None

                if raw_output is not None:
                    candidate_output = raw_output
                    if isinstance(candidate_output, tuple):
                        candidate_output = candidate_output[0] if candidate_output else []
                    if (
                        isinstance(candidate_output, (list, tuple))
                        and len(candidate_output) == 1
                        and isinstance(candidate_output[0], (list, tuple, dict))
                    ):
                        candidate_output = candidate_output[0]

                    min_turn_sec = max(
                        0.01,
                        self._safe_float(sf_cfg.get("min_turn_sec", 0.05), 0.05),
                    )
                    diar_segments = self._parse_sortformer_segments(
                        candidate_output,
                        min_turn_sec=min_turn_sec,
                    )
            else:
                logger.info(
                    "NeMo sortformer runtime requires manifest-based inference; "
                    "using compatibility path%s.",
                    f" ({file_name})" if file_name else "",
                )
                diar_segments = self._run_sortformer_manifest_inference(
                    model=self._nemo_sortformer_model,
                    audio_np=audio_f32,
                    sample_rate=sample_rate,
                    segments=segments,
                    batch_size=batch_size,
                    sf_cfg=sf_cfg,
                    base_cfg=base_cfg,
                    file_name=file_name,
                )
        except Exception as e:
            self._nemo_sortformer_runtime_error = f"sortformer-run-failed: {e}"
            logger.warning(f"NeMo sortformer diarization failed: {e}")
            logger.debug(traceback.format_exc())
            if runtime_device == "mps" and not self._safe_bool(sf_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(sf_cfg):
                    logger.warning(
                        "NeMo sortformer MPS inference failed and CPU retry is disabled; "
                        "continuing with the next diarization route. error=%s",
                        e,
                    )
                    return []
                logger.info("NeMo sortformer MPS inference failed; retrying once on CPU.")
                retry_sf_cfg = dict(sf_cfg)
                retry_sf_cfg["device"] = "cpu"
                retry_sf_cfg["__cpu_retry_done"] = True
                retry_cfg = dict(base_cfg)
                retry_cfg["sortformer"] = retry_sf_cfg
                return self._diarize_audio_nemo_sortformer(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                )
            return []
        if not diar_segments:
            return []

        merge_gap_sec = max(0.0, self._safe_float(sf_cfg.get("merge_gap_sec", 0.08), 0.08))
        merged_segments: List[Dict[str, Any]] = []
        for item in diar_segments:
            start = float(item.get("start", 0.0))
            end = max(start, float(item.get("end", start)))
            speaker = str(item.get("speaker", "0"))
            if (
                merged_segments
                and str(merged_segments[-1].get("speaker", "")) == speaker
                and start <= float(merged_segments[-1].get("end", 0.0)) + merge_gap_sec
            ):
                merged_segments[-1]["end"] = max(float(merged_segments[-1]["end"]), end)
                continue
            merged_segments.append({"start": start, "end": end, "speaker": speaker})

        unique_speakers = sorted({str(seg["speaker"]) for seg in merged_segments})
        logger.info(
            "  NeMo sortformer diarization complete: %d speaker(s), %d turn(s)%s",
            len(unique_speakers),
            len(merged_segments),
            f" ({file_name})" if file_name else "",
        )
        self._nemo_sortformer_runtime_error = ""
        return merged_segments

    @staticmethod
    def _merge_time_regions(
        regions: List[Dict[str, Any]],
        max_gap_sec: float = 0.0,
        min_duration_sec: float = 0.0,
        max_end_sec: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        valid: List[Tuple[float, float]] = []
        for region in regions or []:
            try:
                start = float(region.get("start", 0.0))
                end = float(region.get("end", start))
            except Exception:
                continue
            if max_end_sec is not None:
                start = max(0.0, min(start, max_end_sec))
                end = max(0.0, min(end, max_end_sec))
            if end <= start:
                continue
            valid.append((start, end))

        if not valid:
            return []

        valid.sort(key=lambda item: (item[0], item[1]))
        merge_gap = max(0.0, float(max_gap_sec or 0.0))
        min_duration = max(0.0, float(min_duration_sec or 0.0))

        merged: List[Dict[str, float]] = []
        cur_start, cur_end = valid[0]
        for start, end in valid[1:]:
            if start <= cur_end + merge_gap:
                cur_end = max(cur_end, end)
            else:
                if cur_end - cur_start >= min_duration:
                    merged.append({"start": cur_start, "end": cur_end})
                cur_start, cur_end = start, end
        if cur_end - cur_start >= min_duration:
            merged.append({"start": cur_start, "end": cur_end})
        return merged

    @staticmethod
    def _segment_overlap_seconds(
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> float:
        return max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))

    @staticmethod
    def _audio_to_numpy(audio: Any) -> np.ndarray:
        if audio is None:
            return np.zeros(0, dtype=np.float32)
        if torch.is_tensor(audio):
            arr = audio.detach().cpu().numpy()
        elif isinstance(audio, np.ndarray):
            arr = audio
        else:
            arr = np.asarray(audio)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 0:
            return np.zeros(0, dtype=np.float32)
        if arr.ndim == 1:
            return np.ascontiguousarray(arr)
        if arr.ndim == 2:
            rows, cols = arr.shape
            if rows <= 8 and cols > rows * 8:
                return np.ascontiguousarray(arr.mean(axis=0))
            if cols <= 8 and rows > cols * 8:
                return np.ascontiguousarray(arr.mean(axis=1))
            return np.ascontiguousarray(arr.reshape(-1))

        time_axis = int(np.argmax(arr.shape))
        moved = np.moveaxis(arr, time_axis, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        return np.ascontiguousarray(flat.mean(axis=0))

    def _load_audio_file_mono(self, path: Path) -> np.ndarray:
        audio_path = Path(path)
        if not audio_path.exists() or not audio_path.is_file():
            return np.zeros(0, dtype=np.float32)
        try:
            import soundfile as sf

            data, _sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            return self._audio_to_numpy(data)
        except Exception:
            pass

        try:
            with wave.open(str(audio_path), "rb") as wf:
                channels = max(1, int(wf.getnchannels()))
                frames = wf.readframes(wf.getnframes())
                raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                if channels > 1:
                    raw = raw.reshape(-1, channels).mean(axis=1)
                return np.ascontiguousarray(raw, dtype=np.float32)
        except Exception:
            return np.zeros(0, dtype=np.float32)

    def _collect_audio_streams(self, output: Any) -> List[np.ndarray]:
        streams: List[np.ndarray] = []
        if output is None:
            return streams

        if isinstance(output, (str, os.PathLike, Path)):
            out_path = Path(str(output))
            if out_path.exists() and out_path.is_file():
                arr = self._load_audio_file_mono(out_path)
                if arr.size > 0:
                    return [arr]
                return []
            if out_path.exists() and out_path.is_dir():
                for ext in ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg", "*.aac"):
                    for item in sorted(out_path.glob(ext)):
                        arr = self._load_audio_file_mono(item)
                        if arr.size > 0:
                            streams.append(arr)
                return streams
            return streams

        if isinstance(output, dict):
            for key in ("output_wav", "wav", "audio", "separated_audio", "result", "results"):
                if key in output:
                    return self._collect_audio_streams(output[key])
            for value in output.values():
                streams.extend(self._collect_audio_streams(value))
            return streams

        if isinstance(output, (list, tuple)):
            if output and isinstance(output[0], (int, float, np.integer, np.floating)):
                arr = self._audio_to_numpy(np.asarray(output, dtype=np.float32))
                return [arr] if arr.size > 0 else []
            for item in output:
                streams.extend(self._collect_audio_streams(item))
            return streams

        if torch.is_tensor(output):
            arr = output.detach().cpu().numpy()
        elif isinstance(output, np.ndarray):
            arr = output
        else:
            arr = np.asarray(output)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 0:
            return streams
        if arr.ndim == 1:
            mono = self._audio_to_numpy(arr)
            return [mono] if mono.size > 0 else []
        if arr.ndim == 2:
            rows, cols = arr.shape
            if rows <= 8 and cols > rows * 8:
                return [
                    np.ascontiguousarray(arr[i], dtype=np.float32)
                    for i in range(rows)
                ]
            if cols <= 8 and rows > cols * 8:
                return [
                    np.ascontiguousarray(arr[:, i], dtype=np.float32)
                    for i in range(cols)
                ]
            mono = self._audio_to_numpy(arr)
            return [mono] if mono.size > 0 else []
        if arr.ndim == 3:
            if arr.shape[0] <= 8:
                return [
                    self._audio_to_numpy(arr[i])
                    for i in range(arr.shape[0])
                ]
            if arr.shape[1] <= 8:
                return [
                    self._audio_to_numpy(arr[:, i, :])
                    for i in range(arr.shape[1])
                ]
        mono = self._audio_to_numpy(arr)
        return [mono] if mono.size > 0 else []

    def _normalize_audio_streams(
        self,
        streams: List[np.ndarray],
        target_len: int = 0,
        max_streams: int = 0,
        min_stream_peak: float = 0.0,
    ) -> List[np.ndarray]:
        normalized: List[np.ndarray] = []
        for raw in streams or []:
            stream = self._audio_to_numpy(raw)
            if stream.size == 0:
                continue
            if target_len > 0:
                if stream.size < target_len:
                    stream = np.pad(stream, (0, target_len - stream.size), mode="constant")
                elif stream.size > target_len:
                    stream = stream[:target_len]
            peak = float(np.max(np.abs(stream))) if stream.size else 0.0
            if peak <= max(0.0, float(min_stream_peak or 0.0)):
                continue
            if peak > 1.0:
                stream = stream / peak
            normalized.append(np.ascontiguousarray(stream, dtype=np.float32))

        if not normalized:
            return []

        normalized.sort(key=lambda a: float(np.mean(np.abs(a))) if a.size else 0.0, reverse=True)
        if max_streams > 0:
            normalized = normalized[: max(1, int(max_streams))]
        return normalized

    def _derive_overlap_regions_from_diar_segments(
        self,
        diar_segments: List[Dict[str, Any]],
        min_duration_sec: float = 0.2,
        max_gap_sec: float = 0.08,
    ) -> List[Dict[str, float]]:
        if len(diar_segments or []) < 2:
            return []

        ordered = sorted(
            (
                {
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", item.get("start", 0.0))),
                    "speaker": str(item.get("speaker", "")),
                }
                for item in diar_segments
            ),
            key=lambda x: (x["start"], x["end"]),
        )

        overlaps: List[Dict[str, float]] = []
        for i, left in enumerate(ordered):
            left_start = float(left["start"])
            left_end = float(left["end"])
            left_spk = str(left["speaker"])
            if left_end <= left_start:
                continue
            for right in ordered[i + 1:]:
                right_start = float(right["start"])
                if right_start >= left_end:
                    break
                right_end = float(right["end"])
                if right_end <= right_start:
                    continue
                if left_spk and right.get("speaker") and left_spk == str(right["speaker"]):
                    continue
                overlap_start = max(left_start, right_start)
                overlap_end = min(left_end, right_end)
                if overlap_end - overlap_start <= 0:
                    continue
                overlaps.append({"start": overlap_start, "end": overlap_end})

        return self._merge_time_regions(
            overlaps,
            max_gap_sec=max_gap_sec,
            min_duration_sec=min_duration_sec,
        )

    def _detect_overlap_regions_osd(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        osd_cfg: Dict[str, Any],
    ) -> List[Dict[str, float]]:
        provider = str(osd_cfg.get("provider", "pyannote") or "pyannote").strip().lower()
        if provider not in {"pyannote", "pyannote.audio"}:
            return []

        model_name = str(
            osd_cfg.get("model_name", "pyannote/overlapped-speech-detection")
            or "pyannote/overlapped-speech-detection"
        ).strip()
        if not model_name:
            model_name = "pyannote/overlapped-speech-detection"

        device_pref = str(osd_cfg.get("device", "auto") or "auto").strip().lower()
        runtime_device = self._preferred_torch_device(device_pref, allow_mps=True)
        setup = {
            "provider": provider,
            "model_name": model_name,
            "device": runtime_device,
        }
        try:
            if (
                self._overlap_osd_pipeline is None
                or self._overlap_osd_setup != setup
            ):
                token = str(osd_cfg.get("hf_token", "") or "").strip()
                if not token:
                    token = self._resolve_hf_token(
                        self.asr_cfg.get("faster_whisper", {}) or {}
                    )
                pipeline = self._load_pyannote_pipeline_with_retry(
                    model_name=model_name,
                    cfg=osd_cfg,
                    token=token,
                    purpose="pyannote-osd",
                )

                if hasattr(pipeline, "to"):
                    pipeline = pipeline.to(torch.device(runtime_device))

                self._overlap_osd_pipeline = pipeline
                self._overlap_osd_setup = dict(setup)
                self._overlap_osd_runtime_error = ""
        except Exception as e:
            self._overlap_osd_runtime_error = f"osd-init-failed: {e}"
            logger.debug(f"OSD init failed: {e}")
            if self._preferred_torch_device(osd_cfg.get("device", "auto"), allow_mps=True) == "mps" and not self._safe_bool(osd_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(osd_cfg):
                    logger.warning(
                        "pyannote OSD MPS init failed and CPU retry is disabled; "
                        "skipping overlap OSD fallback. error=%s",
                        e,
                    )
                    return []
                logger.info("pyannote OSD MPS init failed; retrying once on CPU.")
                retry_cfg = dict(osd_cfg)
                retry_cfg["device"] = "cpu"
                retry_cfg["__cpu_retry_done"] = True
                return self._detect_overlap_regions_osd(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    osd_cfg=retry_cfg,
                )
            return []

        if self._overlap_osd_pipeline is None:
            return []

        audio_f32 = self._audio_to_numpy(audio_np)
        if audio_f32.size == 0:
            return []

        waveform = torch.from_numpy(np.ascontiguousarray(audio_f32, dtype=np.float32)).unsqueeze(0)
        runtime_device = str(self._overlap_osd_setup.get("device", "cpu") or "cpu")
        try:
            waveform = self._move_torch_waveform(waveform, runtime_device)
        except Exception as e:
            self._overlap_osd_runtime_error = f"osd-waveform-move-failed: {e}"
            if runtime_device == "mps" and not self._safe_bool(osd_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(osd_cfg):
                    logger.warning(
                        "pyannote OSD MPS waveform move failed and CPU retry is disabled; "
                        "skipping overlap OSD fallback. error=%s",
                        e,
                    )
                    return []
                logger.info("pyannote OSD MPS waveform move failed; retrying once on CPU.")
                retry_cfg = dict(osd_cfg)
                retry_cfg["device"] = "cpu"
                retry_cfg["__cpu_retry_done"] = True
                return self._detect_overlap_regions_osd(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    osd_cfg=retry_cfg,
                )
            return []
        try:
            with torch.inference_mode():
                output = self._overlap_osd_pipeline(
                    {"waveform": waveform, "sample_rate": int(sample_rate)}
                )
        except Exception as e:
            self._overlap_osd_runtime_error = f"osd-run-failed: {e}"
            logger.debug(f"OSD inference failed: {e}")
            if runtime_device == "mps" and not self._safe_bool(osd_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(osd_cfg):
                    logger.warning(
                        "pyannote OSD MPS inference failed and CPU retry is disabled; "
                        "skipping overlap OSD fallback. error=%s",
                        e,
                    )
                    return []
                logger.info("pyannote OSD MPS inference failed; retrying once on CPU.")
                retry_cfg = dict(osd_cfg)
                retry_cfg["device"] = "cpu"
                retry_cfg["__cpu_retry_done"] = True
                return self._detect_overlap_regions_osd(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    osd_cfg=retry_cfg,
                )
            return []

        timeline_obj: Any = output
        if hasattr(output, "get_timeline"):
            try:
                timeline_obj = output.get_timeline()
            except Exception:
                timeline_obj = output
        elif isinstance(output, dict):
            for key in ("overlap", "timeline", "segments", "output"):
                if key in output:
                    timeline_obj = output[key]
                    break

        if hasattr(timeline_obj, "support"):
            try:
                timeline_obj = timeline_obj.support()
            except Exception:
                pass

        regions: List[Dict[str, float]] = []
        iterable: Any = timeline_obj
        if hasattr(timeline_obj, "itersegments"):
            try:
                iterable = timeline_obj.itersegments()
            except Exception:
                iterable = timeline_obj
        try:
            for item in iterable or []:
                start = 0.0
                end = 0.0
                try:
                    if isinstance(item, dict):
                        start = float(item.get("start", 0.0))
                        end = float(item.get("end", start))
                    elif hasattr(item, "start") and hasattr(item, "end"):
                        start = float(item.start)
                        end = float(item.end)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        start = float(item[0])
                        end = float(item[1])
                    else:
                        continue
                except Exception:
                    continue
                if end > start:
                    regions.append({"start": max(0.0, start), "end": max(0.0, end)})
        except Exception as e:
            self._overlap_osd_runtime_error = f"osd-parse-failed: {e}"
            logger.debug(f"OSD output parsing failed: {e}")
            return []
        return regions

    def _detect_overlap_regions(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        diar_segments: List[Dict[str, Any]],
        cfg: Dict[str, Any],
        file_name: str = "",
        seed_regions: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, float]]:
        overlap_cfg = self._nemo_overlap_cfg(cfg)
        if not self._safe_bool(overlap_cfg.get("enabled", False), False):
            return []

        min_region_sec = max(
            0.05, self._safe_float(overlap_cfg.get("min_region_sec", 0.25), 0.25)
        )
        osd_cfg = overlap_cfg.get("osd", {}) or {}
        osd_enabled = self._safe_bool(osd_cfg.get("enabled", True), True)

        regions: List[Dict[str, float]] = []
        source_labels: List[str] = []
        if osd_enabled:
            regions = self._detect_overlap_regions_osd(
                audio_np=audio_np,
                sample_rate=sample_rate,
                osd_cfg=osd_cfg,
            )
            if regions:
                source_labels.append("osd")

        derived_regions = self._derive_overlap_regions_from_diar_segments(
            diar_segments=diar_segments,
            min_duration_sec=min_region_sec,
            max_gap_sec=self._safe_float(osd_cfg.get("merge_gap_sec", 0.08), 0.08),
        )
        if derived_regions:
            regions.extend(derived_regions)
            source_labels.append("diar")
        if seed_regions:
            regions.extend(
                {
                    "start": max(0.0, float(item.get("start", 0.0) or 0.0)),
                    "end": max(0.0, float(item.get("end", item.get("start", 0.0) or 0.0) or 0.0)),
                }
                for item in seed_regions
                if float(item.get("end", item.get("start", 0.0) or 0.0) or 0.0)
                > float(item.get("start", 0.0) or 0.0)
            )
            source_labels.append("hybrid")

        if not regions:
            return []

        audio_dur = len(self._audio_to_numpy(audio_np)) / float(max(1, sample_rate))
        pad_sec = max(0.0, self._safe_float(osd_cfg.get("pad_sec", 0.1), 0.1))
        padded = [
            {
                "start": max(0.0, float(item["start"]) - pad_sec),
                "end": min(audio_dur, float(item["end"]) + pad_sec),
            }
            for item in regions
        ]
        merged = self._merge_time_regions(
            padded,
            max_gap_sec=self._safe_float(osd_cfg.get("merge_gap_sec", 0.08), 0.08),
            min_duration_sec=min_region_sec,
            max_end_sec=audio_dur,
        )
        max_regions = max(1, self._safe_int(overlap_cfg.get("max_regions", 120), 120))
        if len(merged) > max_regions:
            merged = merged[:max_regions]

        if merged:
            logger.info(
                "  Overlap detection[%s] found %d region(s) for [%s].",
                "+".join(source_labels or ["diar"]),
                len(merged),
                file_name or "audio",
            )
        return merged

    def _load_overlap_separator_model(
        self,
        separation_cfg: Dict[str, Any],
    ) -> Any:
        provider_raw = str(
            separation_cfg.get("provider", "clearvoice") or "clearvoice"
        ).strip().lower()

        if provider_raw in {"pyannote", "pyannote.audio"}:
            provider_order = ["pyannote.audio"]
        elif provider_raw == "clearvoice":
            provider_order = ["clearvoice"]
        elif provider_raw in {"auto", ""}:
            prefer_pyannote = self._safe_bool(
                separation_cfg.get("prefer_pyannote", False), False
            )
            provider_order = (
                ["pyannote.audio", "clearvoice"]
                if prefer_pyannote
                else ["clearvoice", "pyannote.audio"]
            )
        else:
            self._overlap_separator_runtime_error = (
                f"unsupported separator provider: {provider_raw}"
            )
            return None

        if self._pyannote_separation_blocked:
            provider_order = [item for item in provider_order if item != "pyannote.audio"]
            if not provider_order:
                self._overlap_separator_runtime_error = (
                    self._pyannote_separation_block_reason
                    or "pyannote separation disabled due to runtime compatibility"
                )
                return None

        task = str(
            separation_cfg.get("task", "speech_separation")
            or "speech_separation"
        ).strip()
        clearvoice_model_name = str(
            separation_cfg.get("model_name", "MossFormer2_SS_16K") or ""
        ).strip() or "MossFormer2_SS_16K"
        pyannote_model_name = str(
            separation_cfg.get("pyannote_model_name", "") or ""
        ).strip()
        if not pyannote_model_name:
            pyannote_model_name = "pyannote/speech-separation-ami-1.0"
        pyannote_device_pref = str(
            separation_cfg.get("device", "auto") or "auto"
        ).strip().lower()
        pyannote_device_candidates = self._torch_device_candidates(
            pyannote_device_pref,
            allow_mps=True,
        )
        pyannote_runtime_device = pyannote_device_candidates[0] if pyannote_device_candidates else "cpu"

        if provider_raw in {"auto", ""} and self._overlap_separator_model is not None:
            cached_setup = self._overlap_separator_setup.get("separator", {})
            cached_provider = str(cached_setup.get("provider", "") or "").strip().lower()
            if (
                cached_provider == "clearvoice"
                and str(cached_setup.get("task", "") or "") == task
                and str(cached_setup.get("model_name", "") or "") == clearvoice_model_name
            ):
                return self._overlap_separator_model
            if (
                cached_provider in {"pyannote", "pyannote.audio"}
                and str(cached_setup.get("model_name", "") or "") == pyannote_model_name
                and str(cached_setup.get("device", "") or "") == pyannote_runtime_device
            ):
                return self._overlap_separator_model

        last_error = ""
        for provider in provider_order:
            if provider == "pyannote.audio":
                token = str(separation_cfg.get("hf_token", "") or "").strip()
                if not token:
                    token = self._resolve_hf_token(
                        self.asr_cfg.get("faster_whisper", {}) or {}
                    )
                for runtime_device in pyannote_device_candidates:
                    setup = {
                        "provider": "pyannote.audio",
                        "model_name": pyannote_model_name,
                        "device": runtime_device,
                    }
                    if (
                        self._overlap_separator_model is not None
                        and self._overlap_separator_setup.get("separator") == setup
                    ):
                        return self._overlap_separator_model

                    try:
                        pipeline = self._load_pyannote_pipeline_with_retry(
                            model_name=pyannote_model_name,
                            cfg=separation_cfg,
                            token=token,
                            purpose="pyannote-separation",
                        )
                        if hasattr(pipeline, "to"):
                            pipeline = pipeline.to(torch.device(runtime_device))
                        self._overlap_separator_model = pipeline
                        self._overlap_separator_setup["separator"] = setup
                        self._overlap_separator_runtime_error = ""
                        logger.info(
                            "  Overlap separator ready: provider=pyannote.audio, model=%s, device=%s",
                            pyannote_model_name,
                            runtime_device,
                        )
                        return pipeline
                    except Exception as e:
                        last_error = f"pyannote init failed: {e}"
                        if self._is_pyannote_known_non_retryable_error(e):
                            self._pyannote_separation_blocked = True
                            self._pyannote_separation_block_reason = last_error
                            logger.info(
                                "Overlap separator pyannote path is unavailable in current runtime; "
                                "disable pyannote separation for this session: %s",
                                e,
                            )
                            break
                        logger.warning(
                            "Overlap separator init failed (%s, device=%s): %s",
                            provider,
                            runtime_device,
                            e,
                        )
                continue

            if provider == "clearvoice":
                setup = {
                    "provider": "clearvoice",
                    "task": task,
                    "model_name": clearvoice_model_name,
                }
                if (
                    self._overlap_separator_model is not None
                    and self._overlap_separator_setup.get("separator") == setup
                ):
                    return self._overlap_separator_model

                try:
                    from clearvoice import ClearVoice
                except Exception as e:
                    last_error = f"clearvoice import failed: {e}"
                    logger.debug(last_error)
                    continue

                try:
                    model = ClearVoice(task=task, model_names=[clearvoice_model_name])
                    self._overlap_separator_model = model
                    self._overlap_separator_setup["separator"] = setup
                    self._overlap_separator_runtime_error = ""
                    logger.info(
                        "  Overlap separator ready: provider=clearvoice, model=%s",
                        clearvoice_model_name,
                    )
                    return model
                except Exception as e:
                    last_error = f"clearvoice init failed: {e}"
                    logger.warning(f"Overlap separator init failed ({provider}): {e}")
                    continue

        self._overlap_separator_runtime_error = (
            last_error or "overlap separator init failed"
        )
        self._overlap_separator_model = None
        return None

    def _run_overlap_separator(
        self,
        model: Any,
        audio_f32: np.ndarray,
        sample_rate: int,
        separation_cfg: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], List[str]]:
        setup = self._overlap_separator_setup.get("separator", {})
        provider = str(
            setup.get("provider", separation_cfg.get("provider", "clearvoice"))
            or "clearvoice"
        ).strip().lower()

        if provider in {"pyannote", "pyannote.audio"}:
            waveform = torch.from_numpy(
                np.ascontiguousarray(audio_f32, dtype=np.float32)
            ).unsqueeze(0)
            runtime_device = str(setup.get("device", "cpu") or "cpu")
            waveform = self._move_torch_waveform(waveform, runtime_device)
            fixed_num = max(
                0, self._safe_int(separation_cfg.get("num_speakers", 2), 2)
            )
            infer_kwargs: Dict[str, Any] = {}
            if fixed_num > 0:
                infer_kwargs["num_speakers"] = int(fixed_num)
            attempts = [
                (
                    "pipeline(dict, **kwargs)",
                    lambda: model(
                        {"waveform": waveform, "sample_rate": int(sample_rate)},
                        **infer_kwargs,
                    ),
                ),
                (
                    "pipeline(dict)",
                    lambda: model(
                        {"waveform": waveform, "sample_rate": int(sample_rate)}
                    ),
                ),
                ("pipeline(waveform, **kwargs)", lambda: model(waveform, **infer_kwargs)),
                ("pipeline(waveform)", lambda: model(waveform)),
            ]
        else:
            payload = np.expand_dims(
                np.ascontiguousarray(audio_f32, dtype=np.float32), axis=0
            )
            attempts = [
                ("model(payload, False)", lambda: model(payload, False)),
                ("model(payload)", lambda: model(payload)),
                ("model(audio, False)", lambda: model(audio_f32, False)),
                ("model(audio)", lambda: model(audio_f32)),
            ]
        errors: List[str] = []

        for label, call_fn in attempts:
            try:
                with torch.inference_mode():
                    output = call_fn()
                raw_streams = self._collect_audio_streams(output)
                streams = self._normalize_audio_streams(
                    raw_streams,
                    target_len=int(len(audio_f32)),
                    max_streams=max(1, self._safe_int(separation_cfg.get("max_streams", 2), 2)),
                    min_stream_peak=max(
                        0.0, self._safe_float(separation_cfg.get("min_stream_peak", 1e-4), 1e-4)
                    ),
                )
                if streams:
                    return streams, errors
                errors.append(f"{label}: empty-streams")
            except Exception as e:
                errors.append(f"{label}: {e}")

        return [], errors

    def _speakers_ranked_for_region(
        self,
        diar_segments: List[Dict[str, Any]],
        start: float,
        end: float,
        preferred_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if end <= start:
            return []
        scores: Dict[str, float] = {}
        for item in preferred_tracks or []:
            speaker = self._normalize_speaker_id(str(item.get("speaker", "0")))
            ov = self._segment_overlap_seconds(
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
                start,
                end,
            )
            if ov <= 0:
                continue
            confidence = max(0.1, float(item.get("confidence", 1.0) or 1.0))
            scores[speaker] = scores.get(speaker, 0.0) + ov * confidence
        for item in diar_segments or []:
            speaker = self._normalize_speaker_id(str(item.get("speaker", "0")))
            ov = self._segment_overlap_seconds(
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
                start,
                end,
            )
            if ov <= 0:
                continue
            scores[speaker] = scores.get(speaker, 0.0) + ov
        return [spk for spk, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]

    @staticmethod
    def _dominant_language_for_window(
        segments: List[TranscriptionSegment],
        start: float,
        end: float,
    ) -> str:
        counts: Dict[str, int] = {}
        for seg in segments or []:
            ov = max(0.0, min(float(seg.end), end) - max(float(seg.start), start))
            if ov <= 0:
                continue
            lang = str(getattr(seg, "language", "") or "").strip()
            if not lang:
                continue
            counts[lang] = counts.get(lang, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda pair: pair[1])[0]

    def _transcribe_overlap_regions(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        base_segments: List[TranscriptionSegment],
        diar_segments: List[Dict[str, Any]],
        overlap_regions: List[Dict[str, float]],
        cfg: Dict[str, Any],
        file_name: str = "",
        overlap_speaker_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TranscriptionSegment]:
        if not overlap_regions:
            return []

        overlap_cfg = self._nemo_overlap_cfg(cfg)
        separation_cfg = overlap_cfg.get("separation", {}) or {}
        separation_enabled = self._safe_bool(separation_cfg.get("enabled", True), True)
        allow_mixed_audio_fallback = self._safe_bool(
            separation_cfg.get("allow_mixed_audio_fallback", False),
            False,
        )

        audio_f32 = self._audio_to_numpy(audio_np)
        if audio_f32.size == 0 or sample_rate <= 0:
            return []
        total_dur = len(audio_f32) / float(sample_rate)

        window_pad = max(0.0, self._safe_float(overlap_cfg.get("window_pad_sec", 0.12), 0.12))
        max_region_sec = max(0.2, self._safe_float(overlap_cfg.get("max_region_sec", 20.0), 20.0))
        min_overlap_ratio = max(
            0.0, min(1.0, self._safe_float(overlap_cfg.get("min_segment_overlap_ratio", 0.15), 0.15))
        )

        separator_model = None
        if separation_enabled:
            separator_model = self._load_overlap_separator_model(separation_cfg)
            if separator_model is None:
                if allow_mixed_audio_fallback:
                    logger.info(
                        "  Overlap separation unavailable; using mixed-audio re-recognition for overlap windows."
                    )
                else:
                    logger.info(
                        "  Overlap separation unavailable; skipping destructive mixed-audio overlap fallback."
                    )

        recovered: List[TranscriptionSegment] = []
        fallback_windows = 0
        for idx, region in enumerate(overlap_regions, start=1):
            core_start = max(0.0, float(region.get("start", 0.0)))
            core_end = min(total_dur, float(region.get("end", core_start)))
            if core_end <= core_start:
                continue

            region_len = core_end - core_start
            if region_len > max_region_sec:
                center = (core_start + core_end) * 0.5
                core_start = max(0.0, center - max_region_sec * 0.5)
                core_end = min(total_dur, core_start + max_region_sec)

            win_start = max(0.0, core_start - window_pad)
            win_end = min(total_dur, core_end + window_pad)
            s0 = int(max(0, round(win_start * sample_rate)))
            s1 = int(min(len(audio_f32), round(win_end * sample_rate)))
            if s1 <= s0:
                continue
            window_audio = np.ascontiguousarray(audio_f32[s0:s1], dtype=np.float32)
            if window_audio.size == 0:
                continue

            speakers_ranked = self._speakers_ranked_for_region(
                diar_segments=diar_segments,
                start=core_start,
                end=core_end,
                preferred_tracks=overlap_speaker_hints,
            )
            if separation_enabled and separator_model is None and not allow_mixed_audio_fallback:
                continue

            streams = [window_audio]
            used_split_streams = False
            if separator_model is not None:
                split_streams, split_errors = self._run_overlap_separator(
                    model=separator_model,
                    audio_f32=window_audio,
                    sample_rate=sample_rate,
                    separation_cfg=separation_cfg,
                )
                if split_streams:
                    streams = split_streams
                    used_split_streams = len(split_streams) > 1
                else:
                    fallback_windows += 1
                    if split_errors:
                        logger.debug(
                            f"Overlap separation fallback [{file_name or 'audio'}][{idx}]: {split_errors[-1]}"
                        )
            language_hint = self._dominant_language_for_window(
                segments=base_segments,
                start=win_start,
                end=win_end,
            )

            for stream_idx, stream in enumerate(streams):
                if stream.size < int(max(1, sample_rate * 0.1)):
                    continue
                stream_tag = f"{file_name}_ov{idx}_s{stream_idx + 1}" if file_name else f"ov{idx}_s{stream_idx + 1}"
                try:
                    segs = self.transcribe(
                        stream,
                        sample_rate=sample_rate,
                        file_name=stream_tag,
                        map_speakers=False,
                        allow_speaker_first=False,
                        language_override=language_hint,
                        assign_speakers_enabled=False,
                        cleanup_after=False,
                    )
                except Exception as e:
                    logger.debug(f"Overlap re-transcribe failed [{stream_tag}]: {e}")
                    continue

                forced_speaker = (
                    speakers_ranked[stream_idx]
                    if used_split_streams and stream_idx < len(speakers_ranked)
                    else ""
                )
                for seg in segs:
                    text = str(seg.text or "").strip()
                    if not text:
                        continue
                    abs_start = float(seg.start) + win_start
                    abs_end = float(seg.end) + win_start
                    if abs_end <= abs_start:
                        abs_end = abs_start + 0.05

                    ov = self._segment_overlap_seconds(abs_start, abs_end, core_start, core_end)
                    seg_len = max(1e-6, abs_end - abs_start)
                    if ov <= 0 or (ov / seg_len) < min_overlap_ratio:
                        continue

                    if forced_speaker:
                        raw_speaker = forced_speaker
                    else:
                        raw_speaker = self._find_speaker_for_span(
                            diar_segments,
                            float(abs_start),
                            float(abs_end),
                            preferred_tracks=overlap_speaker_hints,
                        )

                    recovered.append(
                        TranscriptionSegment(
                            start=max(0.0, abs_start),
                            end=max(0.0, abs_end),
                            text=text,
                            speaker=self._normalize_speaker_id(raw_speaker),
                            language=str(seg.language or ""),
                            confidence=float(seg.confidence or 0.0),
                            words=[
                                {
                                    **word,
                                    **(
                                        {"start": float(word["start"]) + win_start}
                                        if isinstance(word, dict) and word.get("start") is not None
                                        else {}
                                    ),
                                    **(
                                        {"end": float(word["end"]) + win_start}
                                        if isinstance(word, dict) and word.get("end") is not None
                                        else {}
                                    ),
                                }
                                for word in list(getattr(seg, "words", []) or [])
                                if isinstance(word, dict)
                            ],
                        )
                    )

        self._maybe_force_cuda_cleanup()
        if not recovered:
            return []

        recovered.sort(key=lambda s: (float(s.start), float(s.end), str(s.speaker), str(s.text)))
        deduped: List[TranscriptionSegment] = []
        for seg in recovered:
            if not deduped:
                deduped.append(seg)
                continue
            prev = deduped[-1]
            same_text = str(prev.text).strip() == str(seg.text).strip()
            same_spk = str(prev.speaker) == str(seg.speaker)
            near_time = abs(float(prev.start) - float(seg.start)) <= 0.35 and abs(float(prev.end) - float(seg.end)) <= 0.45
            if same_text and same_spk and near_time:
                if float(seg.confidence or 0.0) > float(prev.confidence or 0.0):
                    deduped[-1] = seg
                continue
            deduped.append(seg)

        logger.info(
            f"  Overlap re-recognition kept {len(deduped)} segment(s)"
            f" from {len(overlap_regions)} region(s)."
        )
        if fallback_windows > 0:
            logger.info(
                f"  Overlap separation fell back to mixed-audio re-recognition on {fallback_windows} window(s)."
            )
        return deduped

    def _overlap_redecode_enabled(self, cfg: Optional[Dict[str, Any]] = None) -> bool:
        overlap_cfg = self._nemo_overlap_cfg(cfg)
        if not self._safe_bool(overlap_cfg.get("enabled", False), False):
            return False
        separation_cfg = overlap_cfg.get("separation", {}) or {}
        separation_enabled = self._safe_bool(separation_cfg.get("enabled", True), True)
        allow_mixed_audio_fallback = self._safe_bool(
            separation_cfg.get("allow_mixed_audio_fallback", False),
            False,
        )
        return bool(separation_enabled or allow_mixed_audio_fallback)

    def _merge_overlap_redecoded_segments(
        self,
        base_segments: List[TranscriptionSegment],
        overlap_segments: List[TranscriptionSegment],
        overlap_regions: List[Dict[str, float]],
        cfg: Dict[str, Any],
    ) -> List[TranscriptionSegment]:
        if not overlap_segments:
            return base_segments

        overlap_cfg = self._nemo_overlap_cfg(cfg)
        merge_cfg = overlap_cfg.get("merge", {}) or {}
        drop_original = self._safe_bool(merge_cfg.get("drop_original_overlap", True), True)
        drop_ratio = max(
            0.0,
            min(1.0, self._safe_float(merge_cfg.get("drop_if_overlap_ratio_ge", 0.2), 0.2)),
        )

        kept_base: List[TranscriptionSegment] = []
        dropped = 0
        for seg in base_segments:
            seg_start = float(seg.start)
            seg_end = float(seg.end)
            seg_len = max(1e-6, seg_end - seg_start)
            best_ratio = 0.0
            best_same_speaker_ratio = 0.0
            for region in overlap_regions:
                ov = self._segment_overlap_seconds(
                    seg_start,
                    seg_end,
                    float(region.get("start", 0.0)),
                    float(region.get("end", 0.0)),
                )
                if ov <= 0:
                    continue
                best_ratio = max(best_ratio, ov / seg_len)
            for overlap_seg in overlap_segments:
                ov = self._segment_overlap_seconds(
                    seg_start,
                    seg_end,
                    float(getattr(overlap_seg, "start", 0.0) or 0.0),
                    float(getattr(overlap_seg, "end", 0.0) or 0.0),
                )
                if ov <= 0.0:
                    continue
                if str(getattr(overlap_seg, "speaker", "") or "") == str(getattr(seg, "speaker", "") or ""):
                    best_same_speaker_ratio = max(best_same_speaker_ratio, ov / seg_len)
            if (
                drop_original
                and best_ratio >= max(0.80, drop_ratio)
                and best_same_speaker_ratio >= max(0.65, drop_ratio)
            ):
                dropped += 1
                continue
            kept_base.append(seg)

        merged = kept_base + overlap_segments
        merged.sort(key=lambda s: (float(s.start), float(s.end)))

        compact: List[TranscriptionSegment] = []
        for seg in merged:
            text = str(seg.text or "").strip()
            if not text:
                continue
            if float(seg.end) <= float(seg.start):
                seg.end = float(seg.start) + 0.05
            if not compact:
                compact.append(seg)
                continue
            prev = compact[-1]
            same_text = str(prev.text).strip() == text
            same_spk = str(prev.speaker) == str(seg.speaker)
            near_time = abs(float(prev.start) - float(seg.start)) <= 0.3 and abs(float(prev.end) - float(seg.end)) <= 0.4
            if same_text and same_spk and near_time:
                if float(seg.confidence or 0.0) > float(prev.confidence or 0.0):
                    compact[-1] = seg
                continue
            compact.append(seg)

        logger.info(
            f"  Overlap merge: base={len(base_segments)}, overlap_new={len(overlap_segments)}, "
            f"dropped_base_overlap={dropped}, final={len(compact)}."
        )
        return compact

    def set_runtime_temp_dir(self, temp_dir: Optional[Path]) -> None:
        if temp_dir is None:
            self._runtime_temp_dir = None
            return
        resolved = Path(temp_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        self._runtime_temp_dir = resolved

    def _nemo_model_cache_root(self) -> Path:
        cfg = self._nemo_msdd_cfg()
        raw_cache_dir = str(cfg.get("cache_dir", "output_files/.nemo_msdd") or "").strip()
        cache_dir = Path(raw_cache_dir or "output_files/.nemo_msdd")
        if not cache_dir.is_absolute():
            normalized = raw_cache_dir.replace("\\", "/")
            if normalized in {"", ".nemo_msdd", "output_files/.nemo_msdd", "_runtime_artifacts/nemo_cache"}:
                output_root = Path(self.config["paths"]["output_dir"])
                cache_dir = resolve_runtime_artifact_path(output_root, "nemo_cache")
            else:
                cache_dir = resolve_app_writable_path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _nemo_run_root(self) -> Path:
        if self._runtime_temp_dir is not None:
            run_root = self._runtime_temp_dir / "nemo_msdd"
            run_root.mkdir(parents=True, exist_ok=True)
            return run_root
        return self._nemo_model_cache_root()

    def _startup_preload_nemo_models_once(
        self,
        cfg: Dict[str, Any],
    ) -> None:
        if self._nemo_startup_preload_attempted:
            return
        self._nemo_startup_preload_attempted = True

        if not self._safe_bool(cfg.get("preload_models", True), True):
            self._nemo_startup_preload_summary = "disabled by config"
            logger.info("  NeMo startup preload skipped: disabled by config.")
            return

        msdd_model = str(cfg.get("model_path", "diar_msdd_telephonic") or "").strip()
        msdd_source = self._nemo_resolve_model_source(
            kind="msdd",
            model_name=msdd_model,
            cfg=cfg,
        )
        msdd_filename = str(msdd_source.get("filename", "") or "").strip()
        if not msdd_filename and msdd_model:
            if msdd_model.lower().endswith(".nemo"):
                msdd_filename = Path(msdd_model).name
            else:
                msdd_filename = "diar_msdd_telephonic.nemo"

        if msdd_filename:
            existing_msdd = self._find_local_nemo_artifact(msdd_filename)
            if existing_msdd is not None:
                logger.info("  NeMo MSDD local artifact present: %s", existing_msdd)
            else:
                logger.info("  NeMo MSDD local artifact missing: %s", msdd_filename)

        try:
            resolved = self._ensure_nemo_models_downloaded(cfg)
            self._nemo_startup_preload_summary = (
                ", ".join(f"{key}={value}" for key, value in sorted(resolved.items()))
                if resolved
                else "warmup completed"
            )
            if resolved:
                logger.info("  NeMo startup preload ready: %s", self._nemo_startup_preload_summary)
            else:
                logger.info("  NeMo startup preload completed with cached models.")
        except Exception as e:
            self._nemo_startup_preload_attempted = False
            self._nemo_startup_preload_summary = str(e or "startup preload failed")
            logger.info(
                "  NeMo startup preload incomplete; continuing with available diarization routes: %s",
                e,
            )

    def _ensure_nemo_models_downloaded(
        self,
        cfg: Dict[str, Any],
    ) -> Dict[str, str]:
        resolved_model_paths: Dict[str, str] = {}
        if not self._safe_bool(cfg.get("preload_models", True), True):
            return resolved_model_paths

        self._ensure_diarization_dependency_compat()
        try:
            from nemo.collections.asr.models import (
                EncDecClassificationModel,
                EncDecDiarLabelModel,
                EncDecFrameClassificationModel,
                EncDecSpeakerLabelModel,
            )
        except Exception as e:
            raise RuntimeError(f"NeMo model preload import failed: {e}") from e

        retries = max(1, self._safe_int(cfg.get("download_retries", 3), 3))
        wait_sec = max(
            0.0, self._safe_float(cfg.get("download_retry_wait_sec", 2.0), 2.0)
        )
        device_pref = str(cfg.get("download_device", "cpu") or "cpu").strip().lower()
        map_location = "cuda" if (self.has_cuda and device_pref in {"cuda", "gpu", "cuda:0"}) else "cpu"
        use_speaker_from_ckpt = self._safe_bool(
            cfg.get("use_speaker_model_from_ckpt", True), True
        )

        model_specs: List[Tuple[str, str, Any, str]] = []
        vad_model = str(cfg.get("vad_model", "vad_multilingual_marblenet") or "").strip()
        speaker_model = str(cfg.get("speaker_model", "titanet_large") or "").strip()
        msdd_model = str(cfg.get("model_path", "diar_msdd_telephonic") or "").strip()

        if vad_model:
            # VAD MarbleNet checkpoints are frame-classification models.
            # Using the generic classification class can trigger restore key mismatch.
            model_specs.append(("vad", vad_model, EncDecFrameClassificationModel, "vad_model"))
        if speaker_model and not use_speaker_from_ckpt:
            model_specs.append(("speaker", speaker_model, EncDecSpeakerLabelModel, "speaker_model"))
        if msdd_model and self._nemo_msdd_model_is_available(cfg):
            model_specs.append(("msdd", msdd_model, EncDecDiarLabelModel, "model_path"))

        pending_specs: List[Tuple[str, str, Any, str]] = []
        for kind, model_name, model_cls, cfg_key in model_specs:
            cache_key = f"{kind}:{model_name}:{map_location}".lower()
            if self._nemo_model_cache_status.get(cache_key, False):
                cached_path = str(self._nemo_model_cache_paths.get(cache_key, "") or "").strip()
                if cached_path:
                    resolved_model_paths[cfg_key] = cached_path
                continue
            pending_specs.append((kind, model_name, model_cls, cfg_key))

        if not pending_specs:
            return resolved_model_paths

        total_attempts = len(pending_specs)
        preload_t0 = time.time()
        self._emit_progress(
            "model_download",
            phase="start",
            family="nemo_msdd",
            kind="preload",
            repo="NeMo MSDD preload",
            source="nemo_preload",
            attempt=0,
            total_attempts=total_attempts,
            progress_percent=None,
        )

        for attempt_counter, (kind, model_name, model_cls, cfg_key) in enumerate(
            pending_specs, start=1
        ):
            cache_key = f"{kind}:{model_name}:{map_location}".lower()
            self._emit_progress(
                "model_download",
                phase="attempt",
                family="nemo_msdd",
                kind=kind,
                repo=model_name,
                source="nemo_preload",
                endpoint=f"map_location={map_location}",
                attempt=attempt_counter,
                total_attempts=total_attempts,
                progress_percent=None,
                phase_detail=f"{attempt_counter}/{total_attempts}",
            )
            logger.info(
                "  NeMo warmup %s model %s (%d/%d)",
                kind,
                model_name,
                attempt_counter,
                total_attempts,
            )

            loaded = None
            last_error: Optional[Exception] = None
            local_model_path: Optional[str] = None
            source_error: Optional[Exception] = None
            try:
                local_model_path = self._prepare_nemo_model_path(
                    kind=kind,
                    model_name=model_name,
                    cfg=cfg,
                    purpose=f"NeMo warmup {kind}",
                )
            except Exception as source_err:
                source_error = source_err
                logger.info(
                    "NeMo warmup source preparation failed for %s (%s): %s",
                    kind,
                    model_name,
                    source_err,
                )
                local_model_path = None

            if local_model_path:
                logger.info("  NeMo warmup %s using local model: %s", kind, local_model_path)
                restore_variants: List[Dict[str, Any]] = [
                    {
                        "restore_path": local_model_path,
                        "map_location": map_location,
                        "strict": False,
                    },
                    {
                        "restore_path": local_model_path,
                        "map_location": map_location,
                    },
                    {
                        "restore_path": local_model_path,
                        "strict": False,
                    },
                    {"restore_path": local_model_path},
                ]
                restore_error: Optional[Exception] = None
                for restore_kwargs in restore_variants:
                    try:
                        loaded = model_cls.restore_from(**restore_kwargs)
                        restore_error = None
                        break
                    except Exception as inner_restore_err:
                        restore_error = inner_restore_err
                        msg = str(inner_restore_err or "")
                        if "unexpected keyword argument" in msg:
                            continue
                        continue
                if loaded is not None:
                    resolved_model_paths[cfg_key] = str(local_model_path)
                    last_error = None
                elif restore_error is not None:
                    logger.info(
                        "NeMo warmup %s local restore failed for %s; falling back to from_pretrained: %s",
                        kind,
                        model_name,
                        restore_error,
                    )
                    last_error = restore_error

            if loaded is None:
                if source_error is not None and self._is_nemo_auth_error(source_error):
                    last_error = source_error
                else:
                    last_error = None
                if last_error is None:
                    for round_idx in range(1, retries + 1):
                        try:
                            variants = [
                                {"model_name": model_name, "map_location": map_location},
                                {"model_name": model_name},
                            ]
                            variant_error: Optional[Exception] = None
                            loaded = None
                            for call_kwargs in variants:
                                try:
                                    loaded = model_cls.from_pretrained(**call_kwargs)
                                    variant_error = None
                                    break
                                except Exception as inner_e:
                                    variant_error = inner_e
                                    if "unexpected keyword argument" in str(inner_e or ""):
                                        continue
                                    raise
                            if loaded is None and variant_error is not None:
                                raise variant_error
                            last_error = None
                            break
                        except Exception as e:
                            last_error = e
                            err_lower = str(e or "").lower()
                            if any(
                                marker in err_lower
                                for marker in (
                                    "not able to download url right now",
                                    "403",
                                    "forbidden",
                                )
                            ):
                                break
                            if wait_sec > 0 and round_idx < retries:
                                time.sleep(wait_sec)

            if loaded is None and last_error is not None:
                if kind == "msdd" and self._is_nemo_auth_error(last_error):
                    self._mark_nemo_msdd_model_unavailable(str(last_error))
                    logger.info(
                        "NeMo warmup skipping MSDD model %s for this run: %s",
                        model_name,
                        last_error,
                    )
                    continue
                self._emit_progress(
                    "model_download",
                    phase="failed",
                    family="nemo_msdd",
                    kind=kind,
                    repo=model_name,
                    source="nemo_preload",
                    endpoint=f"map_location={map_location}",
                    attempt=attempt_counter,
                    total_attempts=total_attempts,
                    progress_percent=round(
                        max(
                            0.0,
                            min(100.0, (attempt_counter / max(1, total_attempts)) * 100.0),
                        ),
                        2,
                    ),
                    error=str(last_error)[:200],
                    elapsed_sec=round(max(0.0, time.time() - preload_t0), 3),
                )
                raise RuntimeError(
                    f"NeMo warmup failed for {kind} model {model_name}: {last_error}"
                )

            if loaded is not None:
                try:
                    del loaded
                except Exception:
                    pass
                gc.collect()
                self._nemo_model_cache_status[cache_key] = True
                cached_path = str(resolved_model_paths.get(cfg_key, "") or "").strip()
                if cached_path:
                    self._nemo_model_cache_paths[cache_key] = cached_path
                self._emit_progress(
                    "model_download",
                    phase="done",
                    family="nemo_msdd",
                    kind=kind,
                    repo=model_name,
                    source="nemo_preload",
                    endpoint=f"map_location={map_location}",
                    attempt=attempt_counter,
                    total_attempts=total_attempts,
                    progress_percent=round(
                        max(
                            0.0,
                            min(100.0, (attempt_counter / max(1, total_attempts)) * 100.0),
                        ),
                        2,
                    ),
                    elapsed_sec=round(max(0.0, time.time() - preload_t0), 3),
                )

        # Emit one final completion event after all warmup models are ready.
        # The UI uses this to restore normal file-step progress display.
        self._emit_progress(
            "model_download",
            phase="done",
            family="nemo_msdd",
            kind="preload",
            repo="NeMo MSDD preload",
            source="nemo_preload",
            endpoint=f"map_location={map_location}",
            attempt=total_attempts,
            total_attempts=total_attempts,
            progress_percent=100.0,
            elapsed_sec=round(max(0.0, time.time() - preload_t0), 3),
        )
        return resolved_model_paths

    @staticmethod
    def _write_wav_mono(path: Path, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0 for diarization")

        audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32)
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.flatten()
        if audio_f32.size == 0:
            raise ValueError("empty audio for diarization")

        peak = float(np.max(np.abs(audio_f32))) if audio_f32.size else 0.0
        if peak > 1.0:
            audio_f32 = audio_f32 / peak

        pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())

        return audio_f32

    def _build_nemo_diarizer_cfg(
        self,
        manifest_path: Path,
        out_dir: Path,
        sample_rate: int = 16000,
        run_device: str = "cpu",
        cfg_override: Optional[Dict[str, Any]] = None,
        external_vad_manifest: Optional[Path] = None,
    ) -> Dict[str, Any]:
        cfg = cfg_override or self._nemo_msdd_cfg()
        num_speakers = max(0, self._safe_int(cfg.get("num_speakers", 0), 0))
        min_speakers = max(1, self._safe_int(cfg.get("min_speakers", 1), 1))
        max_speakers = max(min_speakers, self._safe_int(cfg.get("max_speakers", 8), 8))
        default_num_workers = 0 if os.name == "nt" else 1
        num_workers = max(
            0,
            self._safe_int(cfg.get("num_workers", default_num_workers), default_num_workers),
        )

        raw_threshold = cfg.get("sigmoid_threshold", 0.7)
        if isinstance(raw_threshold, (list, tuple)):
            sigmoid_threshold = [
                self._safe_float(item, 0.7)
                for item in raw_threshold
                if item is not None
            ]
        else:
            sigmoid_threshold = [self._safe_float(raw_threshold, 0.7)]
        if not sigmoid_threshold:
            sigmoid_threshold = [0.7]

        clustering_params: Dict[str, Any] = {
            "oracle_num_speakers": bool(num_speakers > 0),
            "min_num_speakers": int(min_speakers),
            "max_num_speakers": int(max_speakers),
            "enhanced_count_thres": 80,
            "max_rp_threshold": 0.25,
            "sparse_search_volume": 30,
            "maj_vote_spk_count": False,
            "chunk_cluster_count": 50,
            "embeddings_per_chunk": 10000,
        }
        if num_speakers > 0:
            clustering_params["num_speakers"] = int(num_speakers)

        msdd_params: Dict[str, Any] = {
            "use_speaker_model_from_ckpt": self._safe_bool(
                cfg.get("use_speaker_model_from_ckpt", True), True
            ),
            "infer_batch_size": max(1, self._safe_int(cfg.get("infer_batch_size", 25), 25)),
            "sigmoid_threshold": sigmoid_threshold,
            "seq_eval_mode": self._safe_bool(cfg.get("seq_eval_mode", False), False),
            "split_infer": self._safe_bool(cfg.get("split_infer", True), True),
            "diar_window_length": max(1, self._safe_int(cfg.get("diar_window_length", 50), 50)),
            "overlap_infer_spk_limit": max(
                1, self._safe_int(cfg.get("overlap_infer_spk_limit", 5), 5)
            ),
        }
        external_vad_path = str(external_vad_manifest) if external_vad_manifest is not None else ""
        vad_model_path = None if external_vad_path else str(cfg.get("vad_model", "vad_multilingual_marblenet"))

        return {
            "name": "ClusterDiarizer",
            "num_workers": int(num_workers),
            "sample_rate": max(1, int(sample_rate)),
            "batch_size": 64,
            "verbose": False,
            "device": str(run_device or "cpu"),
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(out_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "clustering": {
                    "parameters": clustering_params
                },
                "vad": {
                    "model_path": vad_model_path,
                    "external_vad_manifest": external_vad_path or None,
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": self._safe_bool(cfg.get("vad_smoothing", False), False),
                        "onset": self._safe_float(cfg.get("vad_onset", 0.8), 0.8),
                        "offset": self._safe_float(cfg.get("vad_offset", 0.6), 0.6),
                        "pad_onset": self._safe_float(cfg.get("vad_pad_onset", 0.05), 0.05),
                        "pad_offset": self._safe_float(cfg.get("vad_pad_offset", -0.05), -0.05),
                        "min_duration_on": self._safe_float(cfg.get("vad_min_duration_on", 0.1), 0.1),
                        "min_duration_off": self._safe_float(cfg.get("vad_min_duration_off", 0.2), 0.2),
                        "filter_speech_first": self._safe_bool(
                            cfg.get("vad_filter_speech_first", True), True
                        ),
                    },
                },
                "speaker_embeddings": {
                    "model_path": str(cfg.get("speaker_model", "titanet_large")),
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5],
                        "shift_length_in_sec": [0.75, 0.625, 0.5, 0.375, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": True,
                    },
                },
                "msdd_model": {
                    "model_path": str(cfg.get("model_path", "diar_msdd_telephonic")),
                    "parameters": msdd_params,
                },
            }
        }

    def _write_asr_segments_vad_manifest(
        self,
        *,
        manifest_path: Path,
        audio_path: Path,
        uniq_id: str,
        segments: List[TranscriptionSegment],
        total_duration: float,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if not segments:
            return None

        root_cfg = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        use_external_vad = self._safe_bool(root_cfg.get("prefer_asr_vad", True), True)
        if not use_external_vad:
            return None

        pad_sec = max(0.0, self._safe_float(root_cfg.get("asr_vad_pad_sec", 0.05), 0.05))
        merge_gap_sec = max(0.0, self._safe_float(root_cfg.get("asr_vad_merge_gap_sec", 0.15), 0.15))
        min_duration_sec = max(0.02, self._safe_float(root_cfg.get("asr_vad_min_duration_sec", 0.10), 0.10))
        max_duration = max(0.0, float(total_duration or 0.0))
        if max_duration <= 0.0:
            return None

        intervals: List[List[float]] = []
        for seg in segments:
            text = str(getattr(seg, "text", "") or "").strip()
            start = max(0.0, float(getattr(seg, "start", 0.0) or 0.0) - pad_sec)
            end = min(max_duration, max(start, float(getattr(seg, "end", start) or start) + pad_sec))
            if end - start < min_duration_sec:
                continue
            if not text and (end - start) < max(min_duration_sec, 0.25):
                continue
            intervals.append([start, end])

        if not intervals:
            return None

        intervals.sort(key=lambda item: (float(item[0]), float(item[1])))
        merged: List[List[float]] = []
        for start, end in intervals:
            if not merged:
                merged.append([start, end])
                continue
            if start <= float(merged[-1][1]) + merge_gap_sec:
                merged[-1][1] = max(float(merged[-1][1]), end)
                continue
            merged.append([start, end])

        valid = [
            [max(0.0, float(start)), min(max_duration, max(float(start), float(end)))]
            for start, end in merged
            if (float(end) - float(start)) >= min_duration_sec
        ]
        if not valid:
            return None

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for start, end in valid:
                item = {
                    "audio_filepath": str(audio_path),
                    "offset": round(float(start), 5),
                    "duration": round(max(0.0, float(end) - float(start)), 5),
                    "label": "UNK",
                    "uniq_id": str(uniq_id or audio_path.stem or "audio"),
                }
                if item["duration"] <= 0.0:
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(
            "  Built ASR-driven external VAD manifest for MSDD: %d region(s) -> %s",
            len(valid),
            manifest_path,
        )
        return manifest_path

    def _load_rttm_segments(self, rttm_path: Path) -> List[Dict[str, Any]]:
        diar_segments: List[Dict[str, Any]] = []
        with open(rttm_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                    continue
                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                except (TypeError, ValueError):
                    continue
                if duration <= 0:
                    continue
                diar_segments.append(
                    {
                        "start": max(0.0, start),
                        "end": max(0.0, start + duration),
                        "speaker": str(parts[7]),
                    }
                )
        diar_segments.sort(key=lambda x: (x["start"], x["end"]))
        return diar_segments

    def _resolve_rttm_path(self, out_dir: Path, wav_stem: str) -> Optional[Path]:
        pred_rttm_dir = out_dir / "pred_rttms"
        if pred_rttm_dir.exists():
            exact = pred_rttm_dir / f"{wav_stem}.rttm"
            if exact.exists():
                return exact
            for candidate in sorted(pred_rttm_dir.glob("*.rttm")):
                if wav_stem in candidate.stem:
                    return candidate
            for candidate in sorted(pred_rttm_dir.glob("*.rttm")):
                return candidate

        for candidate in sorted(out_dir.rglob("*.rttm")):
            if candidate.is_file():
                return candidate
        return None

    def _prefer_funasr_for_language(self, language: str) -> bool:
        """
        FunASR default model in this project is zh-centric (Mandarin-first).
        Use FunASR for Mandarin Chinese; use faster-whisper for most others.
        """
        lang = self._normalize_language_tag(language)
        if not lang:
            return False

        if lang == "zh":
            return True

        if lang == "yue":
            return True

        return False

    def _preferred_engine_for_language(self, language: str) -> str:
        if self._prefer_funasr_for_language(language):
            return "funasr"
        return "faster_whisper"

    def _preferred_engine_from_config(self) -> Optional[str]:
        lang_cfg = self.config.get("language", {}) or {}
        target_langs = list(lang_cfg.get("primary_languages", []) or [])
        fallback_lang = lang_cfg.get("fallback_language", "")
        if fallback_lang:
            target_langs.append(fallback_lang)
        if not target_langs:
            return None

        votes = {"funasr": 0, "faster_whisper": 0}
        for lang in target_langs:
            pref = self._preferred_engine_for_language(lang)
            votes[pref] += 1

        if votes["funasr"] > votes["faster_whisper"]:
            return "funasr"
        if votes["faster_whisper"] > votes["funasr"]:
            return "faster_whisper"

        if fallback_lang:
            return self._preferred_engine_for_language(fallback_lang)
        return None

    def _build_auto_engine_strategies(self):
        preferred = self._preferred_engine_from_config()
        logger.info(
            "ASR auto preference by target language: "
            f"{preferred or 'funasr-first(default)'}"
        )
        funasr_strategies = self._build_funasr_strategies()
        whisper_strategies = self._build_whisper_family_strategies()
        if (
            preferred is None
            and self.is_macos
            and self.has_mlx_whisper
            and not self.has_cuda
            and not self.has_mps
        ):
            logger.info(
                "ASR auto preference override: prefer MLX Whisper on macOS because "
                "PyTorch MPS is unavailable."
            )
            return self._dedupe_preserve_order(
                [("mlx_whisper", self._try_init_mlx_whisper)]
                + whisper_strategies
                + funasr_strategies
            )
        if preferred == "faster_whisper":
            return self._dedupe_preserve_order(whisper_strategies + funasr_strategies)
        return self._dedupe_preserve_order(funasr_strategies + whisper_strategies)

    def _try_init_funasr_cuda(self) -> tuple:
        """
        Initialize FunASR on CUDA with VRAM-aware optional components.

        Practical rule of thumb:
          - Base ASR + VAD requires about 2.5 GB free VRAM
          - Punctuation / speaker models are enabled only when extra VRAM is available
        """
        if not self.has_cuda:
            return False, "no CUDA"

        try:
            from funasr import AutoModel
        except ImportError:
            return False, "not installed (pip install funasr)"

        cfg = self.asr_cfg["funasr"]
        free_gb = _get_free_gpu_gb()

        # Base requirement for ASR + VAD on CUDA
        min_required = 2.5
        if free_gb < min_required:
            return False, (
                f"need >= {min_required}GB free, "
                f"only {free_gb:.1f}GB available"
            )

        kwargs = {
            "model": cfg["model"],
            "device": "cuda:0",
            "disable_update": cfg.get("disable_update", True),
        }

        # VAD is cheap enough to load when configured
        vad_model = cfg.get("vad_model", "")
        if vad_model:
            kwargs["vad_model"] = vad_model

        # Punctuation / speaker models are optional and VRAM-gated
        load_punc = False
        load_spk = False
        punc_model = cfg.get("punc_model", "")
        spk_model = cfg.get("spk_model", "")

        if free_gb >= 5.0 and punc_model:
            kwargs["punc_model"] = punc_model
            load_punc = True
        if free_gb >= 5.5 and spk_model:
            kwargs["spk_model"] = spk_model
            load_spk = True

        models = ["ASR", "VAD"]
        if load_punc:
            models.append("Punc")
        if load_spk:
            models.append("SPK")

        logger.info(
            f"  FunASR config: {'+'.join(models)} "
            f"(free={free_gb:.1f}GB)"
        )

        try:
            self._funasr_model = self._create_funasr_model_compat(
                AutoModel,
                kwargs,
            )
            self._funasr_device = "cuda:0"
            self._funasr_has_punc = load_punc
            self._funasr_has_spk = load_spk
            self.engine_name = "funasr"
            self._current_engine = "funasr_cuda"

            _log_gpu_mem("funasr-loaded")
            return True, ""

        except torch.cuda.OutOfMemoryError:
            return False, "CUDA OOM during model load"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False, "OOM during model load"
            return False, str(e)[:150]
        except Exception as e:
            return False, str(e)[:150]

    def _try_init_funasr_mps(self) -> tuple:
        if not self.has_mps:
            return False, "no MPS"

        try:
            from funasr import AutoModel
        except ImportError:
            return False, "not installed"

        cfg = self.asr_cfg["funasr"]
        try:
            logger.info("  FunASR Apple Metal (MPS): ASR+VAD (try Punc)...")
            base_kwargs = {
                "model": cfg["model"],
                "device": "mps",
                "disable_update": cfg.get("disable_update", True),
            }
            vad_model = cfg.get("vad_model", "")
            if vad_model:
                base_kwargs["vad_model"] = vad_model

            attempts: List[Tuple[Dict[str, Any], bool]] = []
            punc_model = str(cfg.get("punc_model", "") or "").strip()
            if punc_model:
                kwargs_with_punc = dict(base_kwargs)
                kwargs_with_punc["punc_model"] = punc_model
                attempts.append((kwargs_with_punc, True))
            attempts.append((base_kwargs, False))

            errors: List[str] = []
            for kwargs, has_punc in attempts:
                try:
                    self._funasr_model = self._create_funasr_model_compat(
                        AutoModel,
                        kwargs,
                    )
                    self._funasr_device = "mps"
                    self._funasr_has_punc = has_punc
                    self._funasr_has_spk = False
                    self.engine_name = "funasr"
                    self._current_engine = "funasr_mps"
                    logger.info(
                        "FunASR ready on Apple Metal (MPS)%s",
                        " + Punc" if has_punc else "",
                    )
                    return True, ""
                except Exception as e:
                    errors.append(str(e)[:150])
                    if has_punc:
                        logger.info(
                            "  FunASR MPS punc init failed, retrying ASR+VAD only: %s",
                            str(e)[:150],
                        )
                        continue
                    return False, str(e)[:150]

            return False, errors[-1] if errors else "unknown error"

        except Exception as e:
            return False, str(e)[:150]

    # FunASR CPU fallback initialization

    def _try_init_funasr_cpu(self) -> tuple:
        try:
            from funasr import AutoModel
        except ImportError:
            return False, "not installed"

        cfg = self.asr_cfg["funasr"]

        try:
            logger.info("  FunASR CPU: ASR+VAD (try Punc)...")
            base_kwargs = {
                "model": cfg["model"],
                "device": "cpu",
                "disable_update": cfg.get("disable_update", True),
            }
            vad_model = cfg.get("vad_model", "")
            if vad_model:
                base_kwargs["vad_model"] = vad_model

            attempts: List[Tuple[Dict[str, Any], bool]] = []
            punc_model = str(cfg.get("punc_model", "") or "").strip()
            if punc_model:
                kwargs_with_punc = dict(base_kwargs)
                kwargs_with_punc["punc_model"] = punc_model
                attempts.append((kwargs_with_punc, True))
            attempts.append((base_kwargs, False))

            errors: List[str] = []
            for kwargs, has_punc in attempts:
                try:
                    self._funasr_model = self._create_funasr_model_compat(
                        AutoModel,
                        kwargs,
                    )
                    self._funasr_device = "cpu"
                    self._funasr_has_punc = has_punc
                    self._funasr_has_spk = False
                    self.engine_name = "funasr"
                    self._current_engine = "funasr_cpu"
                    logger.info(
                        "FunASR ready on CPU%s",
                        " + Punc" if has_punc else "",
                    )
                    return True, ""
                except Exception as e:
                    errors.append(str(e)[:150])
                    if has_punc:
                        logger.info(
                            "  FunASR CPU punc init failed, retrying ASR+VAD only: %s",
                            str(e)[:150],
                        )
                        continue
                    return False, str(e)[:150]

            return False, errors[-1] if errors else "unknown error"

        except Exception as e:
            return False, str(e)[:150]

    def _try_init_mlx_whisper(self) -> tuple:
        if not self.is_macos:
            return False, "MLX Whisper is macOS-only"
        if not self.has_mlx_whisper:
            detail = self._mlx_install_detail or "not installed (pip install mlx-whisper)"
            return False, detail
        if self._mlx_disabled_for_session:
            return False, self._mlx_failure_reason()

        cfg = self.asr_cfg.get(
            "faster_whisper", self.asr_cfg.get("whisperx", {})
        )
        try:
            model_ref = self._resolve_effective_mlx_model_ref(
                self._resolve_mlx_whisper_model_ref(cfg),
                cfg,
            )
            probe_ok, probe_err = self._probe_mlx_whisper_runtime()
            if not probe_ok:
                self._disable_mlx_for_session(probe_err)
                return False, probe_err
            self._mlx_whisper_module = True
            self._mlx_model_ref = model_ref
            self._fw_cfg = cfg
            self._fw_device = "mlx"
            self.engine_name = "mlx_whisper"
            self._current_engine = "mlx_whisper"
            logger.info("mlx-whisper ready: %s", model_ref)
            self._start_faster_whisper_startup_preload()
            return True, ""
        except Exception as e:
            return False, str(e)[:150]

    # faster-whisper CUDA initialization

    def _try_init_faster_whisper_cuda(self) -> tuple:
        if not self.has_cuda:
            return False, "no CUDA"

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return False, "not installed (pip install faster-whisper)"

        cfg = self.asr_cfg.get(
            "faster_whisper", self.asr_cfg.get("whisperx", {})
        )
        compute_type = cfg.get("compute_type", "float16")
        model_size = cfg.get("model_size", "large-v3")
        try:
            configured_workers = int(cfg.get("num_workers", 1))
        except (TypeError, ValueError):
            configured_workers = 1
        configured_workers = max(1, configured_workers)
        tuned_workers = configured_workers
        if configured_workers <= 1:
            cpu_count = os.cpu_count() or 4
            tuned_workers = max(2, min(8, cpu_count // 2))
            if tuned_workers != configured_workers:
                logger.info(
                    "  Auto-tuned faster-whisper num_workers: %d -> %d",
                    configured_workers,
                    tuned_workers,
                )
        free_gb = _get_free_gpu_gb()

        # Estimated minimum free VRAM by model size (float16 baseline)
        requirements = {
            "large-v3": 4.5, "large-v2": 4.5, "large": 4.5,
            "medium": 3.0, "small": 1.8, "base": 1.0, "tiny": 0.6,
        }

        # int8 variants usually reduce VRAM usage noticeably
        if compute_type in ("int8", "int8_float16"):
            requirements = {k: v * 0.65 for k, v in requirements.items()}

        # Try configured size first, then downgrade until one fits
        order = [
            "large-v3", "large-v2", "large",
            "medium", "small", "base", "tiny",
        ]
        try:
            start_idx = order.index(model_size)
        except ValueError:
            start_idx = 0

        selected = None
        for candidate in order[start_idx:]:
            needed = requirements.get(candidate, 4.5)
            if free_gb >= needed:
                selected = candidate
                break

        if selected is None:
            return False, (
                f"not enough VRAM ({free_gb:.1f}GB free), "
                f"need >= {requirements.get('tiny', 0.6):.1f}GB"
            )

        if selected != model_size:
            logger.warning(
                f"  Auto-downgrade: {model_size} ?{selected} "
                f"({free_gb:.1f}GB free)"
            )

        try:
            self._ensure_fw_model_downloaded(selected, cfg=cfg)
            logger.info(
                f"  faster-whisper: {selected} / cuda / {compute_type}"
            )
            self._faster_whisper_model = WhisperModel(
                selected,
                device="cuda",
                device_index=0,
                compute_type=compute_type,
                num_workers=tuned_workers,
            )
            self._fw_cfg = cfg
            self._fw_device = "cuda"
            self._fw_model_size = selected
            self.engine_name = "faster_whisper"
            self._current_engine = "faster_whisper_cuda"
            self._start_faster_whisper_startup_preload()

            _log_gpu_mem("fw-loaded")
            return True, ""

        except Exception as e:
            return False, str(e)[:150]

    #  faster-whisper CPU 

    def _try_init_faster_whisper_cpu(self) -> tuple:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return False, "not installed"

        cfg = self.asr_cfg.get(
            "faster_whisper", self.asr_cfg.get("whisperx", {})
        )

        preferred_sizes: List[str] = []
        configured_size = str(cfg.get("model_size", "") or "").strip()
        for candidate in [
            configured_size,
            "small",
            "base",
            "tiny",
            "medium",
            "large-v3",
        ]:
            if candidate and candidate not in preferred_sizes:
                preferred_sizes.append(candidate)

        model_size = self._choose_cached_fw_model(preferred=preferred_sizes)
        if not model_size:
            download_errors: List[str] = []
            for candidate in preferred_sizes:
                try:
                    self._ensure_fw_model_downloaded(candidate, cfg=cfg)
                    model_size = candidate
                    break
                except Exception as e:
                    download_errors.append(f"{candidate}: {str(e)[:120]}")
            if not model_size:
                return False, (
                    "no cached model and download failed: "
                    + " | ".join(download_errors[:3])
                )

        compute_types: List[str] = []
        for candidate in [
            str(cfg.get("cpu_compute_type", "") or "").strip(),
            str(cfg.get("compute_type", "") or "").strip(),
            "int8",
            "float32",
            "default",
        ]:
            if candidate and candidate not in compute_types:
                compute_types.append(candidate)

        init_errors: List[str] = []
        try:
            configured_cpu_threads = int(cfg.get("cpu_threads", 0))
        except (TypeError, ValueError):
            configured_cpu_threads = 0
        configured_cpu_threads = max(0, configured_cpu_threads)
        cpu_threads = configured_cpu_threads
        if cpu_threads <= 0:
            cpu_count = os.cpu_count() or 4
            cpu_threads = max(4, min(8, cpu_count))
        elif cpu_threads < 4 and (os.cpu_count() or 4) >= 4:
            cpu_threads = 4
        if cpu_threads != configured_cpu_threads:
            logger.info(
                "  Auto-tuned faster-whisper cpu_threads: %d -> %d",
                configured_cpu_threads,
                cpu_threads,
            )
        for compute_type in compute_types:
            try:
                logger.info(f"  faster-whisper: {model_size} / cpu / {compute_type}")
                self._faster_whisper_model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                )
                self._fw_cfg = cfg
                self._fw_device = "cpu"
                self._fw_model_size = model_size
                self.engine_name = "faster_whisper"
                self._current_engine = "faster_whisper_cpu"
                logger.info(f"faster-whisper ready: {model_size} / CPU")
                self._start_faster_whisper_startup_preload()
                return True, ""
            except Exception as e:
                init_errors.append(f"{compute_type}: {str(e)[:120]}")
                self._unload_faster_whisper()

        return False, " | ".join(init_errors[:3])[:300]

    def _is_model_cached(self, model_size: str) -> bool:
        """ster-whisper"""
        if model_size in self._model_cache_status:
            return self._model_cache_status[model_size]

        cached = False
        try:
            from huggingface_hub import try_to_load_from_cache
            repo = f"Systran/faster-whisper-{model_size}"
            required_files = ("model.bin", "config.json")
            cache_hits = []
            for filename in required_files:
                result = try_to_load_from_cache(repo, filename)
                cache_hits.append(isinstance(result, str) and Path(result).exists())
            cached = all(cache_hits)
        except Exception:
            pass

        self._model_cache_status[model_size] = cached
        return cached

    def _ensure_fw_model_downloaded(
        self, model_size: str, cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Pre-download faster-whisper model.
        Endpoint priority is auto-ranked (Google connectivity + endpoint probe),
        with retries and mirror/official fallback.
        If all attempts fail, abort initialization.
        """
        if self._is_model_cached(model_size):
            return

        repo = f"Systran/faster-whisper-{model_size}"
        logger.info(
            f"  First run: downloading {repo} to cache (progress below)..."
        )

        from huggingface_hub import snapshot_download

        show_download_progress = bool(self.config.get("logging.show_progress", True))
        if show_download_progress:
            if os.getenv("HF_HUB_DISABLE_PROGRESS_BARS", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
            try:
                from huggingface_hub.utils import enable_progress_bars

                enable_progress_bars()
            except Exception:
                pass
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        token = self._resolve_hf_token(cfg)
        if not token:
            token = None

        attempts, retry_wait_sec = self._build_hf_download_attempts(cfg=cfg)

        total_attempts = sum(max(0, int(item["retries"])) for item in attempts)
        if total_attempts <= 0:
            raise RuntimeError("No download retries configured for faster-whisper.")
        download_t0 = time.time()

        self._emit_progress(
            "model_download",
            phase="start",
            family="faster_whisper",
            kind=model_size,
            repo=repo,
            attempt=0,
            total_attempts=total_attempts,
            progress_percent=None,
        )

        last_error: Optional[Exception] = None
        attempt_index = 0

        for item in attempts:
            source = str(item["source"])
            endpoint = str(item["endpoint"])
            retries = int(item["retries"])

            for _ in range(retries):
                attempt_index += 1
                logger.info(
                    f"  Download attempt {attempt_index}/{total_attempts} "
                    f"via {source}: {endpoint}"
                )
                self._emit_progress(
                    "model_download",
                    phase="attempt",
                    family="faster_whisper",
                    kind=model_size,
                    repo=repo,
                    source=source,
                    endpoint=endpoint,
                    attempt=attempt_index,
                    total_attempts=total_attempts,
                    progress_percent=None,
                    phase_detail=f"{attempt_index}/{total_attempts}",
                )

                prev_endpoint = os.getenv("HF_ENDPOINT")
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint
                elif prev_endpoint is not None:
                    os.environ.pop("HF_ENDPOINT", None)
                try:
                    snapshot_download(
                        repo_id=repo,
                        token=token,
                        resume_download=True,
                        local_files_only=False,
                    )
                    self._model_cache_status[model_size] = True
                    logger.info(f"  Model cached: {repo} (source={source})")
                    self._emit_progress(
                        "model_download",
                        phase="done",
                        family="faster_whisper",
                        kind=model_size,
                        repo=repo,
                        source=source,
                        endpoint=endpoint,
                        attempt=attempt_index,
                        total_attempts=total_attempts,
                        progress_percent=100.0,
                        elapsed_sec=round(max(0.0, time.time() - download_t0), 3),
                    )
                    return
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"  Download failed via {source} ({attempt_index}/{total_attempts}): {e}"
                    )
                    self._emit_progress(
                        "model_download",
                        phase="retry",
                        family="faster_whisper",
                        kind=model_size,
                        repo=repo,
                        source=source,
                        endpoint=endpoint,
                        attempt=attempt_index,
                        total_attempts=total_attempts,
                        progress_percent=None,
                        error=str(e)[:200],
                        phase_detail=f"{attempt_index}/{total_attempts}",
                    )
                    if retry_wait_sec > 0 and attempt_index < total_attempts:
                        time.sleep(retry_wait_sec)
                finally:
                    if prev_endpoint is None:
                        os.environ.pop("HF_ENDPOINT", None)
                    else:
                        os.environ["HF_ENDPOINT"] = prev_endpoint

        self._emit_progress(
            "model_download",
            phase="failed",
            family="faster_whisper",
            kind=model_size,
            repo=repo,
            attempt=attempt_index,
            total_attempts=total_attempts,
            progress_percent=round(
                max(0.0, min(100.0, (attempt_index / max(1, total_attempts)) * 100.0)),
                2,
            ),
            error=str(last_error)[:200] if last_error else "unknown",
            elapsed_sec=round(max(0.0, time.time() - download_t0), 3),
        )
        raise RuntimeError(
            f"Failed to download {repo} from mirror and official sources "
            f"after {attempt_index} attempts: {last_error}"
        )

    def _choose_cached_fw_model(
        self, preferred: List[str]
    ) -> Optional[str]:
        for size in preferred:
            if self._is_model_cached(size):
                return size
        return None

    def _language_fallback(self) -> str:
        lang_cfg = self.config.get("language", {}) or {}
        fallback = self._normalize_language_tag(
            str(lang_cfg.get("fallback_language", "zh") or "")
        )
        return fallback or "zh"

    def _build_language_probe_audio(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        *,
        start_sec: float = 0.0,
        probe_duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        lang_cfg = self.config.get("language", {}) or {}
        if probe_duration_sec is None:
            probe_duration_sec = lang_cfg.get("probe_duration_sec", 30)
        try:
            probe_duration_sec = float(probe_duration_sec)
        except (TypeError, ValueError):
            probe_duration_sec = 30.0
        probe_duration_sec = max(1.0, probe_duration_sec)

        audio_view = audio_np
        if isinstance(audio_view, np.ndarray) and audio_view.ndim > 1:
            audio_view = audio_view.reshape(-1)
        total_samples = int(len(audio_view))
        if total_samples <= 0:
            return np.zeros((0,), dtype=np.float32)

        max_samples = int(sample_rate * probe_duration_sec)
        if max_samples <= 0:
            return np.zeros((0,), dtype=np.float32)

        try:
            start_value = float(start_sec)
        except (TypeError, ValueError):
            start_value = 0.0
        start_idx = int(max(0.0, start_value) * sample_rate)
        if start_idx >= total_samples:
            start_idx = max(0, total_samples - max_samples)
        end_idx = min(total_samples, start_idx + max_samples)
        if end_idx <= start_idx:
            return np.zeros((0,), dtype=np.float32)

        probe_audio = np.ascontiguousarray(
            audio_view[start_idx:end_idx], dtype=np.float32
        )
        if probe_audio.ndim > 1:
            probe_audio = probe_audio.flatten()
        return probe_audio

    def _detect_language_with_fw_model(
        self, fw_model: Any, probe_audio: np.ndarray
    ) -> tuple[str, float]:
        fw_cfg = self.asr_cfg.get(
            "faster_whisper", self.asr_cfg.get("whisperx", {})
        ) or {}
        lang_cfg = self.config.get("language", {}) or {}
        probe_vad_filter = self._safe_bool(
            lang_cfg.get("probe_vad_filter", True), True
        )
        vad_parameters = fw_cfg.get("vad_parameters", {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        })
        transcribe_kwargs = {
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "vad_filter": probe_vad_filter,
            "vad_parameters": vad_parameters if probe_vad_filter else None,
            "word_timestamps": False,
            "without_timestamps": True,
        }
        try:
            segments_gen, info = fw_model.transcribe(
                probe_audio, **transcribe_kwargs
            )
            detected = self._normalize_language_tag(
                getattr(info, "language", "") if info else ""
            )
            prob = float(
                getattr(info, "language_probability", 0.0) if info else 0.0
            )
            try:
                del segments_gen
            except Exception:
                pass
            gc.collect()
            return detected, prob
        except Exception as e:
            logger.debug(f"Language probe failed: {e}")
            return "", 0.0

    def _get_language_probe_model(self, allow_download: bool = True):
        with self._lang_probe_lock:
            if self._lang_probe_model is not None:
                return self._lang_probe_model
            if self._lang_probe_init_failed:
                return None

            try:
                from faster_whisper import WhisperModel
            except ImportError:
                self._lang_probe_init_failed = True
                self._lang_probe_init_error = "faster-whisper not installed"
                return None

            fw_cfg = self.asr_cfg.get(
                "faster_whisper", self.asr_cfg.get("whisperx", {})
            ) or {}
            lang_cfg = self.config.get("language", {}) or {}
            model_size = self._choose_cached_fw_model(
                self._language_probe_candidate_model_sizes()
            )
            if not model_size:
                if not allow_download:
                    return None
                if not bool(lang_cfg.get("probe_allow_download", True)):
                    self._lang_probe_init_failed = True
                    self._lang_probe_init_error = (
                        "probe model download disabled by config"
                    )
                    return None

                probe_size = str(lang_cfg.get("probe_model_size", "tiny") or "").strip().lower()
                if not probe_size:
                    probe_size = "tiny"

                try:
                    self._ensure_fw_model_downloaded(probe_size, cfg=fw_cfg)
                    model_size = probe_size
                except Exception as e:
                    self._lang_probe_init_failed = True
                    self._lang_probe_init_error = (
                        f"probe model download failed: {str(e)[:200]}"
                    )
                    return None

            try:
                logger.info(
                    "Loading language probe model: "
                    f"faster-whisper {model_size} / cpu / int8"
                )
                self._lang_probe_model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self._lang_probe_model_size = model_size
                self._lang_probe_init_failed = False
                self._lang_probe_init_error = ""
                return self._lang_probe_model
            except Exception as e:
                logger.warning(f"Language probe model init failed: {e}")
                self._lang_probe_model = None
                self._lang_probe_model_size = ""
                self._lang_probe_init_failed = True
                self._lang_probe_init_error = (
                    f"probe model init failed: {str(e)[:200]}"
                )
                return None

    def detect_audio_language(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        file_name: str = "",
    ) -> str:
        self._wait_for_faster_whisper_startup_preload("language probe")
        fallback = self._language_fallback()
        lang_cfg = self.config.get("language", {}) or {}
        self.last_language_probe = {
            "language": fallback,
            "probability": 0.0,
            "source": "fallback",
            "offset_sec": 0.0,
        }
        if not bool(lang_cfg.get("auto_detect", True)):
            logger.info(
                f"Language auto-detect disabled, using fallback: {fallback}"
            )
            return fallback

        if audio_np is None or sample_rate <= 0:
            return fallback

        probe_duration_sec = self._safe_float(
            lang_cfg.get("probe_duration_sec", 30), 30.0
        )
        probe_duration_sec = max(1.0, probe_duration_sec)
        min_prob = max(
            0.0,
            min(
                1.0,
                self._safe_float(
                    lang_cfg.get("probe_min_probability", 0.55), 0.55
                ),
            ),
        )
        min_prob_outside_primary = max(
            min_prob,
            min(
                1.0,
                self._safe_float(
                    lang_cfg.get(
                        "probe_min_probability_outside_primary", 0.80
                    ),
                    0.80,
                ),
            ),
        )
        early_accept_prob = max(
            min_prob,
            min(
                1.0,
                self._safe_float(
                    lang_cfg.get("probe_early_accept_probability", 0.85),
                    0.85,
                ),
            ),
        )

        preferred_langs: List[str] = []
        for item in list(lang_cfg.get("primary_languages", []) or []):
            norm = self._normalize_language_tag(str(item))
            if norm and norm not in preferred_langs:
                preferred_langs.append(norm)
        if fallback and fallback not in preferred_langs:
            preferred_langs.append(fallback)
        preferred_lang_set = set(preferred_langs)

        if isinstance(audio_np, np.ndarray):
            audio_len_samples = int(audio_np.size)
        else:
            audio_len_samples = int(len(audio_np))
        audio_duration_sec = float(audio_len_samples) / float(sample_rate)
        max_start_sec = max(0.0, audio_duration_sec - probe_duration_sec)
        probe_offsets: List[float] = [0.0]
        extra_offsets = lang_cfg.get("probe_extra_offsets_sec", [60, 180])
        if isinstance(extra_offsets, (list, tuple)):
            for raw_value in extra_offsets:
                try:
                    offset = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if offset <= 0:
                    continue
                offset = min(max_start_sec, offset)
                if offset <= 0:
                    continue
                if any(abs(offset - existing) < 1.0 for existing in probe_offsets):
                    continue
                probe_offsets.append(offset)

        def _probe_once(probe_audio: np.ndarray) -> tuple[str, float, str]:
            if probe_audio.size == 0:
                return "", 0.0, ""
            if self._faster_whisper_model is not None:
                detected, prob = self._detect_language_with_fw_model(
                    self._faster_whisper_model, probe_audio
                )
                if detected:
                    return detected, prob, "current"
            probe_model = self._get_language_probe_model()
            if probe_model is not None:
                detected, prob = self._detect_language_with_fw_model(
                    probe_model, probe_audio
                )
                if detected:
                    return (
                        detected,
                        prob,
                        f"probe:{self._lang_probe_model_size or 'unknown'}",
                    )
            return "", 0.0, ""

        candidates: List[Dict[str, Any]] = []
        for index, offset_sec in enumerate(probe_offsets):
            probe_audio = self._build_language_probe_audio(
                audio_np,
                sample_rate,
                start_sec=offset_sec,
                probe_duration_sec=probe_duration_sec,
            )
            detected, prob, source = _probe_once(probe_audio)
            if detected:
                candidates.append({
                    "language": detected,
                    "probability": float(prob),
                    "source": source,
                    "offset_sec": float(offset_sec),
                })
                logger.info(
                    f"Pre-detect probe {index + 1}/{len(probe_offsets)} "
                    f"[{file_name or 'audio'} @ {offset_sec:.1f}s]: "
                    f"{detected} (p={prob:.2f}, source={source})"
                )
                if (
                    index == 0
                    and float(prob) >= early_accept_prob
                    and (
                        not preferred_lang_set
                        or detected in preferred_lang_set
                    )
                ):
                    self.last_language_probe = dict(candidates[-1])
                    logger.info(
                        f"Pre-detected language [{file_name or 'audio'}]: "
                        f"{detected} (p={prob:.2f})"
                    )
                    return detected

        if candidates:
            by_lang: Dict[str, Dict[str, float]] = {}
            for item in candidates:
                lang = str(item.get("language", "") or "")
                if not lang:
                    continue
                stats = by_lang.setdefault(lang, {
                    "score": 0.0,
                    "max_prob": 0.0,
                    "count": 0.0,
                })
                prob = float(item.get("probability", 0.0) or 0.0)
                stats["score"] += max(0.01, prob)
                stats["max_prob"] = max(stats["max_prob"], prob)
                stats["count"] += 1.0

            if by_lang:
                selected_lang = max(
                    by_lang.items(),
                    key=lambda pair: (
                        pair[1]["score"],
                        pair[1]["count"],
                        pair[1]["max_prob"],
                    ),
                )[0]
                selected_prob = float(by_lang[selected_lang]["max_prob"])
                selected_item = max(
                    (item for item in candidates if item["language"] == selected_lang),
                    key=lambda item: float(item.get("probability", 0.0) or 0.0),
                )

                if (
                    preferred_lang_set
                    and selected_lang not in preferred_lang_set
                    and selected_prob < min_prob_outside_primary
                ):
                    preferred_candidates = [
                        item
                        for item in candidates
                        if str(item.get("language", "") or "") in preferred_lang_set
                    ]
                    if preferred_candidates:
                        selected_item = max(
                            preferred_candidates,
                            key=lambda item: float(item.get("probability", 0.0) or 0.0),
                        )
                        selected_lang = str(selected_item.get("language", "") or "")
                        selected_prob = float(
                            selected_item.get("probability", 0.0) or 0.0
                        )

                if selected_lang and selected_prob >= min_prob:
                    self.last_language_probe = dict(selected_item)
                    logger.info(
                        f"Pre-detected language [{file_name or 'audio'}]: "
                        f"{selected_lang} (p={selected_prob:.2f})"
                    )
                    return selected_lang

        reason = self._lang_probe_init_error or "confidence below threshold"
        logger.info(
            "Language pre-detect low confidence or unavailable "
            f"({reason}), using fallback: {fallback}"
        )
        return fallback

    def ensure_engine_for_language(self, language: str) -> bool:
        """Switch ASR engine based on target language preference if needed."""
        lang = self._normalize_language_tag(language)
        target_engine = self._preferred_engine_for_language(lang)
        if self._engine_family(self.engine_name) == target_engine:
            return True

        previous_engine = self.engine_name
        logger.info(
            "Switching ASR engine for "
            f"language={lang or 'unknown'}: {previous_engine} -> {target_engine}"
        )

        self._unload_all()
        _force_cuda_cleanup()

        if target_engine == "faster_whisper":
            switch_plan = [
                (self._strategy_label(name), init_fn)
                for name, init_fn in self._build_whisper_family_strategies()
            ]
        else:
            switch_plan = [
                (self._strategy_label(name), init_fn)
                for name, init_fn in self._build_funasr_strategies()
            ]

        last_err = ""
        for label, init_fn in switch_plan:
            ok, err = init_fn()
            if ok:
                self._engine_ready = True
                return True
            last_err = err
            logger.info(f"  {label} unavailable: {err}")

        logger.warning(
            f"Failed to switch to {target_engine}: {last_err}. "
            f"Restoring previous engine ({previous_engine})."
        )
        self._restore_engine_after_switch(previous_engine)
        return False

    def _restore_engine_after_switch(self, previous_engine: str) -> bool:
        if self._engine_family(previous_engine) == "faster_whisper":
            restore_plan = [init_fn for _, init_fn in self._build_whisper_family_strategies()]
            restore_plan.extend(init_fn for _, init_fn in self._build_funasr_strategies())
        else:
            restore_plan = [init_fn for _, init_fn in self._build_funasr_strategies()]
            restore_plan.extend(init_fn for _, init_fn in self._build_whisper_family_strategies())

        restore_plan = self._dedupe_preserve_order(restore_plan)

        for init_fn in restore_plan:
            ok, _err = init_fn()
            if ok:
                self._engine_ready = True
                return True

        self._engine_ready = False
        self._current_engine = None
        logger.error("Failed to restore ASR engine after switch failure.")
        return False

    # Engine/model unload helpers

    def _unload_all(self):
        self._unload_funasr()
        self._unload_faster_whisper()
        self._nemo_sortformer_model = None
        self._nemo_sortformer_setup = {}

    def _unload_funasr(self):
        if self._funasr_model is not None:
            logger.info("Unloading FunASR...")
            try:
                for attr in [
                    "model", "vad_model", "punc_model", "spk_model"
                ]:
                    if hasattr(self._funasr_model, attr):
                        sub = getattr(self._funasr_model, attr)
                        if sub is not None:
                            del sub
            except Exception:
                pass
            del self._funasr_model
            self._funasr_model = None
            self._funasr_device = None
            _force_cuda_cleanup()

    def _unload_faster_whisper(self):
        self._wait_for_faster_whisper_startup_preload("unload faster-whisper")
        if self._faster_whisper_model is not None or self._mlx_whisper_module is not None:
            logger.info("Unloading faster-whisper...")
            self._stop_mlx_worker()
            try:
                if self._faster_whisper_model is not None and hasattr(self._faster_whisper_model, "model"):
                    del self._faster_whisper_model.model
            except Exception:
                pass
            if self._faster_whisper_model is not None:
                del self._faster_whisper_model
            self._faster_whisper_model = None
            self._fw_model_size = ""
            self._mlx_whisper_module = None
            self._mlx_model_ref = ""
            self._fw_cfg = None
            self._fw_device = None
            self._fw_startup_preload_thread = None
            self._fw_startup_preload_done = False
            self._fw_startup_preload_error = ""
            _force_cuda_cleanup()

    def _unload_language_probe(self):
        self._wait_for_faster_whisper_startup_preload("unload language probe")
        if self._lang_probe_model is not None:
            logger.info("Unloading language probe model...")
            try:
                if hasattr(self._lang_probe_model, "model"):
                    del self._lang_probe_model.model
            except Exception:
                pass
            del self._lang_probe_model
            self._lang_probe_model = None
            self._lang_probe_model_size = ""

    # Engine initialization and fallback ordering

    def transcribe(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        file_name: str = "",
        map_speakers: bool = True,
        allow_speaker_first: bool = True,
        language_override: str = "",
        assign_speakers_enabled: bool = True,
        cleanup_after: bool = True,
    ) -> List[TranscriptionSegment]:
        # Compatibility arg kept for old callers.
        _ = allow_speaker_first

        duration = len(audio_np) / sample_rate
        logger.info(
            f"Transcribing [{file_name}] "
            f"with {self._current_engine} ({duration:.1f}s)"
        )
        _log_gpu_mem("pre-transcribe")

        t0 = time.time()
        segments = []

        try:
            if self.engine_name == "funasr":
                segments = self._transcribe_funasr(
                    audio_np, sample_rate, file_name
                )
            elif self.engine_name == "faster_whisper":
                segments = self._transcribe_faster_whisper(
                    audio_np,
                    sample_rate,
                    file_name,
                    language_override=language_override,
                )
            elif self.engine_name == "mlx_whisper":
                segments = self._transcribe_mlx_whisper(
                    audio_np,
                    sample_rate,
                    file_name,
                    language_override=language_override,
                )
            else:
                raise RuntimeError(
                    f"Unknown engine: {self.engine_name}"
                )

            # Reset OOM retry counter after a successful transcription
            self._oom_retries = 0

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_str = str(e).lower()
            if "out of memory" in error_str:
                segments = self._handle_oom(
                    audio_np, sample_rate, file_name
                )
            elif "cuda" in error_str and "error" in error_str:
                segments = self._handle_oom(
                    audio_np, sample_rate, file_name
                )
            else:
                raise

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            logger.debug(traceback.format_exc())
            segments = self._try_fallback(
                audio_np, sample_rate, file_name
            )

        if language_override:
            for seg in segments:
                seg.language = language_override or seg.language

        segments = self._rebalance_segment_granularity(
            segments,
            file_name=file_name,
        )

        if assign_speakers_enabled:
            segments = self.assign_speakers(
                audio_np=audio_np,
                sample_rate=sample_rate,
                segments=segments,
                file_name=file_name,
                map_speakers=map_speakers,
            )

        elapsed = time.time() - t0
        rtf = elapsed / duration if duration > 0 else 0
        logger.info(
            f"Done: {len(segments)} segs, "
            f"{elapsed:.1f}s, RTF={rtf:.3f}"
        )

        if cleanup_after:
            self._maybe_force_cuda_cleanup()
        return segments

    # OOM recovery flow

    def _handle_oom(
        self, audio_np, sample_rate, file_name
    ) -> List[TranscriptionSegment]:
        """
        Recover from CUDA OOM / CUDA runtime failures.

        Strategy summary:
          - If current engine is FunASR CUDA, retry with minimal CUDA config first,
            then fall back to FunASR CPU.
          - If current engine is faster-whisper CUDA, retry with a smaller cached
            CUDA model, then fall back to CPU.
          - Final fallback is FunASR CPU.
        """
        self._oom_retries += 1
        if self._oom_retries > self._max_oom_retries:
            logger.error(
                f"OOM retry limit ({self._max_oom_retries}) reached"
            )
            return []

        logger.warning(
            f"OOM recovery attempt {self._oom_retries}/"
            f"{self._max_oom_retries}"
        )

        current = self._current_engine or ""

        # If the current engine is a CUDA variant, try CUDA-side recovery first
        if "cuda" in current:
            self._unload_all()
            _force_cuda_cleanup()
            _log_gpu_mem("post-unload")

            if "funasr" in current:
                # First retry FunASR CUDA without optional punc/spk models
                if self._funasr_has_punc or self._funasr_has_spk:
                    logger.info(
                        "  OOM recovery: retry FunASR CUDA "
                        "without punc/spk..."
                    )
                    ok = self._retry_funasr_cuda_minimal()
                    if ok:
                        try:
                            return self._transcribe_funasr(
                                audio_np, sample_rate, file_name
                            )
                        except Exception as e2:
                            logger.warning(f"  Minimal FunASR also OOM: {e2}")
                            self._unload_all()
                            _force_cuda_cleanup()

                # Then fall back to FunASR CPU
                logger.info(
                    "  OOM recovery: FunASR CPU (model already cached)..."
                )
                ok, _ = self._try_init_funasr_cpu()
                if ok:
                    try:
                        return self._transcribe_funasr(
                            audio_np, sample_rate, file_name
                        )
                    except Exception as e3:
                        logger.error(f"  FunASR CPU failed: {e3}")

            elif "faster_whisper" in current:
                # faster-whisper: cuda
                logger.info(
                    "  OOM recovery: try smaller fw model on CUDA..."
                )
                ok = self._retry_fw_cuda_smaller()
                if ok:
                    try:
                        return self._transcribe_faster_whisper(
                            audio_np, sample_rate, file_name
                        )
                    except Exception as e2:
                        logger.warning(f"  Smaller fw also failed: {e2}")
                        self._unload_all()
                        _force_cuda_cleanup()

        # Final CPU fallback
        self._unload_all()
        _force_cuda_cleanup()

        # Prefer FunASR CPU as the last-resort recovery path
        logger.info("  OOM recovery: FunASR CPU final fallback...")
        ok, _ = self._try_init_funasr_cpu()
        if ok:
            try:
                return self._transcribe_funasr(
                    audio_np, sample_rate, file_name
                )
            except Exception as e:
                logger.error(f"  FunASR CPU failed: {e}")

        logger.error("  All OOM recovery attempts failed")
        return []

    def _retry_funasr_cuda_minimal(self) -> bool:
        """Retry FunASR CUDA using only ASR + VAD (no punc/spk)."""
        try:
            from funasr import AutoModel
        except ImportError:
            return False

        cfg = self.asr_cfg["funasr"]
        _force_cuda_cleanup()
        free_gb = _get_free_gpu_gb()

        if free_gb < 2.0:
            logger.info(
                f"  Only {free_gb:.1f}GB free, "
                f"not enough for minimal FunASR CUDA"
            )
            return False

        try:
            kwargs = {
                "model": cfg["model"],
                "device": "cuda:0",
                "disable_update": True,
            }
            vad_model = cfg.get("vad_model", "")
            if vad_model:
                kwargs["vad_model"] = vad_model
            # Intentionally skip punc/spk in minimal retry mode

            self._funasr_model = self._create_funasr_model_compat(
                AutoModel,
                kwargs,
            )
            self._funasr_device = "cuda:0"
            self._funasr_has_punc = False
            self._funasr_has_spk = False
            self.engine_name = "funasr"
            self._current_engine = "funasr_cuda_minimal"

            _log_gpu_mem("funasr-minimal")
            return True

        except Exception as e:
            logger.warning(f"  Minimal FunASR CUDA failed: {e}")
            return False

    def _retry_fw_cuda_smaller(self) -> bool:
        """Retry faster-whisper on CUDA with a smaller cached model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return False

        cfg = self._fw_cfg or self.asr_cfg.get(
            "faster_whisper", self.asr_cfg.get("whisperx", {})
        )

        _force_cuda_cleanup()
        free_gb = _get_free_gpu_gb()

        # Prefer smaller cached models to avoid a fresh download during recovery
        candidates = ["small", "base", "tiny"]
        for model_size in candidates:
            # Minimal free-VRAM requirement per candidate
            needs = {"small": 1.8, "base": 1.0, "tiny": 0.6}
            if free_gb < needs.get(model_size, 1.0):
                continue

            # Skip candidates that are not already cached locally
            if not self._is_model_cached(model_size):
                continue

            try:
                logger.info(
                    f"  Retry fw: {model_size} / cuda / float16"
                )
                self._faster_whisper_model = WhisperModel(
                    model_size,
                    device="cuda",
                    device_index=0,
                    compute_type="float16",
                )
                self._fw_cfg = cfg
                self._fw_device = "cuda"
                self._fw_model_size = model_size
                self.engine_name = "faster_whisper"
                self._current_engine = f"faster_whisper_cuda_{model_size}"
                self._start_faster_whisper_startup_preload()
                _log_gpu_mem(f"fw-{model_size}")
                return True

            except Exception as e:
                logger.debug(f"  fw {model_size} failed: {e}")
                self._unload_faster_whisper()
                _force_cuda_cleanup()

        return False

    def _try_fallback(
        self, audio_np, sample_rate, file_name
    ) -> List[TranscriptionSegment]:
        """Fallback path when transcription fails."""
        logger.info("Trying fallback engine...")
        self._unload_all()
        _force_cuda_cleanup()

        fallback_plan = self._dedupe_preserve_order(
            self._build_funasr_strategies() + self._build_whisper_family_strategies()
        )
        for name, init_fn in fallback_plan:
            ok, _ = init_fn()
            if not ok:
                continue
            try:
                if self.engine_name == "funasr":
                    return self._transcribe_funasr(
                        audio_np, sample_rate, file_name
                    )
                if self.engine_name == "faster_whisper":
                    return self._transcribe_faster_whisper(
                        audio_np, sample_rate, file_name
                    )
                if self.engine_name == "mlx_whisper":
                    return self._transcribe_mlx_whisper(
                        audio_np, sample_rate, file_name
                    )
            except Exception as exc:
                logger.debug("Fallback engine %s failed: %s", name, exc)
                self._unload_all()

        return []

    # FunASR transcription path

    def _transcribe_funasr(
        self, audio_np: np.ndarray, sample_rate: int, file_name: str
    ) -> List[TranscriptionSegment]:
        if self._funasr_model is None:
            raise RuntimeError("FunASR not loaded")

        cfg = self.asr_cfg["funasr"]
        batch_size = cfg.get("batch_size", 4)

        # Reduce batch size under low VRAM to lower OOM risk
        if self._funasr_device == "cuda:0":
            free_gb = _get_free_gpu_gb()
            if free_gb < 2.0:
                batch_size = min(batch_size, 2)
                logger.info(
                    f"  Low VRAM ({free_gb:.1f}GB), "
                    f"reduced batch_size to {batch_size}"
                )

        audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32)

        hotword = cfg.get("hotword", "") or ""

        generate_kwargs = {
            "input": audio_f32,
            "batch_size_s": batch_size * 10,
            "hotword": hotword if hotword.strip() else "",
        }
        if self._funasr_has_punc:
            generate_kwargs["sentence_timestamp"] = True

        results = self._funasr_model.generate(
            **generate_kwargs,
        )

        return self._parse_funasr_results(results)

    @staticmethod
    def _normalize_funasr_sentence_text(item: Dict[str, Any]) -> str:
        for key in ("text", "sentence", "raw_text"):
            value = str(item.get(key, "") or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _split_funasr_text_fallback(
        text: str,
        *,
        total_duration_sec: float,
    ) -> List[str]:
        source = str(text or "").strip()
        if not source:
            return []

        compact_chars = [ch for ch in source if not ch.isspace()]
        compact_len = len(compact_chars)
        if compact_len <= 0:
            return [source]

        target_segments = max(1, int(math.ceil(max(0.0, total_duration_sec) / 14.0)))
        target_chars = max(24, min(160, int(math.ceil(compact_len / target_segments))))

        def _char_len(item: str) -> int:
            return len([ch for ch in str(item or "") if not ch.isspace()])

        def _regex_units(payload: str, pattern: str) -> List[str]:
            blocks = [part.strip() for part in re.split(r"[\r\n]+", payload) if part.strip()]
            units: List[str] = []
            for block in blocks:
                found = [item.strip() for item in re.findall(pattern, block) if item.strip()]
                if found:
                    units.extend(found)
                elif block:
                    units.append(block)
            return units

        def _hard_split(unit: str, hard_limit: int) -> List[str]:
            pieces: List[str] = []
            buf: List[str] = []
            count = 0
            for ch in str(unit or ""):
                buf.append(ch)
                if not ch.isspace():
                    count += 1
                if count >= hard_limit:
                    piece = "".join(buf).strip()
                    if piece:
                        pieces.append(piece)
                    buf = []
                    count = 0
            tail = "".join(buf).strip()
            if tail:
                pieces.append(tail)
            return pieces

        sentence_units = _regex_units(source, r".+?(?:[。！？!?；;]+|$)")
        if not sentence_units:
            sentence_units = [source]

        refined_units: List[str] = []
        clause_limit = max(18, int(target_chars * 0.7))
        for unit in sentence_units:
            unit_len = _char_len(unit)
            if unit_len <= max(target_chars + 20, int(target_chars * 1.35)):
                refined_units.append(unit)
                continue

            clause_units = _regex_units(unit, r".+?(?:[，,、：:]+|$)")
            if len(clause_units) > 1:
                refined_units.extend(clause_units)
                continue

            refined_units.extend(_hard_split(unit, max(20, clause_limit)))

        pieces: List[str] = []
        current_units: List[str] = []
        current_len = 0
        for unit in refined_units:
            unit_len = _char_len(unit)
            if (
                current_units
                and current_len + unit_len > target_chars
            ):
                chunk = "".join(current_units).strip()
                if chunk:
                    pieces.append(chunk)
                current_units = []
                current_len = 0
            current_units.append(unit)
            current_len += unit_len

        if current_units:
            chunk = "".join(current_units).strip()
            if chunk:
                pieces.append(chunk)

        return pieces or [source]

    @classmethod
    def _build_funasr_fallback_segments(
        cls,
        text: str,
        timestamps: Any,
        *,
        speaker: str = "0",
        confidence: float = 0.0,
    ) -> List[TranscriptionSegment]:
        source = str(text or "").strip()
        if not source:
            return []

        ts_pairs: List[Tuple[float, float]] = []
        for item in list(timestamps or []):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                start_ms = float(item[0])
                end_ms = float(item[1])
            except Exception:
                continue
            ts_pairs.append((start_ms, max(start_ms, end_ms)))

        if ts_pairs:
            start_sec = ts_pairs[0][0] / 1000.0
            end_sec = ts_pairs[-1][1] / 1000.0
        else:
            start_sec = 0.0
            end_sec = 0.0
        total_duration_sec = max(0.0, end_sec - start_sec)

        pieces = cls._split_funasr_text_fallback(
            source,
            total_duration_sec=total_duration_sec,
        )
        if len(pieces) <= 1:
            return [
                TranscriptionSegment(
                    start=start_sec,
                    end=end_sec,
                    text=source,
                    speaker=speaker,
                    confidence=confidence,
                )
            ]

        total_chars = max(
            1,
            sum(len([ch for ch in piece if not ch.isspace()]) for piece in pieces),
        )
        segments: List[TranscriptionSegment] = []
        used_chars = 0
        total_ts = len(ts_pairs)

        for index, piece in enumerate(pieces):
            piece_chars = max(1, len([ch for ch in piece if not ch.isspace()]))
            prev_ratio = used_chars / total_chars
            used_chars += piece_chars
            next_ratio = used_chars / total_chars

            if total_ts > 0:
                start_idx = min(total_ts - 1, int(math.floor(prev_ratio * total_ts)))
                end_idx = min(total_ts - 1, max(start_idx, int(math.ceil(next_ratio * total_ts)) - 1))
                seg_start = ts_pairs[start_idx][0] / 1000.0
                seg_end = ts_pairs[end_idx][1] / 1000.0
            else:
                seg_start = start_sec + total_duration_sec * prev_ratio
                seg_end = start_sec + total_duration_sec * next_ratio

            if index == len(pieces) - 1:
                seg_end = max(seg_end, end_sec)
            seg_end = max(seg_start, seg_end)
            segments.append(
                TranscriptionSegment(
                    start=seg_start,
                    end=seg_end,
                    text=piece.strip(),
                    speaker=speaker,
                    confidence=confidence,
                )
            )

        return segments

    def _parse_funasr_results(
        self, results: Any
    ) -> List[TranscriptionSegment]:
        segments = []
        if not results:
            return segments

        for result in results:
            if isinstance(result, dict):
                sentence_info = result.get("sentence_info", [])
                if sentence_info:
                    for sent in sentence_info:
                        text = self._normalize_funasr_sentence_text(sent)
                        if not text:
                            continue
                        conf = sent.get("confidence", 0)
                        segments.append(TranscriptionSegment(
                            start=sent.get("start", 0) / 1000.0,
                            end=sent.get("end", 0) / 1000.0,
                            text=text,
                            speaker=str(sent.get("spk", sent.get("speaker", "0"))),
                            confidence=(
                                float(conf)
                                if isinstance(conf, (int, float))
                                else 0.0
                            ),
                        ))
                    continue

                text = result.get("text", "").strip()
                if text:
                    ts = result.get("timestamp", [])
                    conf = result.get("confidence", 0)
                    fallback_segments = self._build_funasr_fallback_segments(
                        text,
                        ts,
                        speaker="0",
                        confidence=(
                            float(conf)
                            if isinstance(conf, (int, float))
                            else 0.0
                        ),
                    )
                    if fallback_segments:
                        segments.extend(fallback_segments)
                    else:
                        s = ts[0][0] / 1000.0 if ts else 0.0
                        e = ts[-1][1] / 1000.0 if ts else 0.0
                        segments.append(TranscriptionSegment(
                            start=s, end=e, text=text, speaker="0",
                        ))

            elif isinstance(result, str) and result.strip():
                segments.append(TranscriptionSegment(
                    text=result.strip(), speaker="0",
                ))

        return segments

    @classmethod
    def _split_overlong_segment(
        cls,
        seg: TranscriptionSegment,
        *,
        max_duration_sec: float = 28.0,
        max_chars: int = 140,
    ) -> List[TranscriptionSegment]:
        source = str(getattr(seg, "text", "") or "").strip()
        if not source:
            return [seg]

        start_sec = float(getattr(seg, "start", 0.0) or 0.0)
        end_sec = max(start_sec, float(getattr(seg, "end", start_sec) or start_sec))
        duration = max(0.0, end_sec - start_sec)
        compact_len = len([ch for ch in source if not ch.isspace()])
        if duration <= max_duration_sec and compact_len <= max_chars:
            return [seg]

        pieces = cls._split_funasr_text_fallback(
            source,
            total_duration_sec=duration,
        )
        if len(pieces) <= 1 and compact_len > max_chars:
            hard_limit = max(20, int(max_chars))
            pieces = []
            buf: List[str] = []
            compact_count = 0
            for ch in source:
                buf.append(ch)
                if not ch.isspace():
                    compact_count += 1
                if compact_count >= hard_limit:
                    piece = "".join(buf).strip()
                    if piece:
                        pieces.append(piece)
                    buf = []
                    compact_count = 0
            tail = "".join(buf).strip()
            if tail:
                pieces.append(tail)

        if len(pieces) <= 1:
            return [seg]

        ts_pairs: List[Tuple[float, float]] = []
        for word in list(getattr(seg, "words", []) or []):
            if not isinstance(word, dict):
                continue
            try:
                word_start = float(word.get("start"))
                word_end = float(word.get("end", word_start))
            except Exception:
                continue
            ts_pairs.append(
                (
                    max(start_sec, word_start) * 1000.0,
                    max(max(start_sec, word_start), word_end) * 1000.0,
                )
            )

        total_chars = max(
            1,
            sum(len([ch for ch in piece if not ch.isspace()]) for piece in pieces),
        )
        used_chars = 0
        total_ts = len(ts_pairs)
        split_segments: List[TranscriptionSegment] = []

        for index, piece in enumerate(pieces):
            piece_chars = max(1, len([ch for ch in piece if not ch.isspace()]))
            prev_ratio = used_chars / total_chars
            used_chars += piece_chars
            next_ratio = used_chars / total_chars

            if total_ts > 0:
                start_idx = min(total_ts - 1, int(math.floor(prev_ratio * total_ts)))
                end_idx = min(
                    total_ts - 1,
                    max(start_idx, int(math.ceil(next_ratio * total_ts)) - 1),
                )
                piece_start = ts_pairs[start_idx][0] / 1000.0
                piece_end = ts_pairs[end_idx][1] / 1000.0
            else:
                piece_start = start_sec + duration * prev_ratio
                piece_end = start_sec + duration * next_ratio

            if index == len(pieces) - 1:
                piece_end = max(piece_end, end_sec)
            piece_end = max(piece_start, piece_end)

            piece_words: List[Dict[str, Any]] = []
            for word in list(getattr(seg, "words", []) or []):
                if not isinstance(word, dict):
                    continue
                try:
                    word_start = float(word.get("start"))
                    word_end = float(word.get("end", word_start))
                except Exception:
                    continue
                word_mid = (word_start + word_end) * 0.5
                if word_mid < piece_start - 1e-3 or word_mid > piece_end + 1e-3:
                    continue
                piece_words.append(dict(word))

            split_segments.append(
                TranscriptionSegment(
                    start=piece_start,
                    end=piece_end,
                    text=piece.strip(),
                    speaker=str(getattr(seg, "speaker", "0") or "0"),
                    language=str(getattr(seg, "language", "") or ""),
                    confidence=float(getattr(seg, "confidence", 0.0) or 0.0),
                    words=piece_words or None,
                )
            )

        return split_segments or [seg]

    def _rebalance_segment_granularity(
        self,
        segments: List[TranscriptionSegment],
        *,
        file_name: str = "",
    ) -> List[TranscriptionSegment]:
        if not segments:
            return segments

        expanded: List[TranscriptionSegment] = []
        expanded_count = 0
        for seg in segments:
            parts = self._split_overlong_segment(seg)
            expanded.extend(parts)
            if len(parts) > 1:
                expanded_count += 1

        if expanded_count > 0:
            logger.info(
                "  Rebalanced overlong ASR segments: %d -> %d (%d expanded)%s",
                len(segments),
                len(expanded),
                expanded_count,
                f" ({file_name})" if file_name else "",
            )
        return expanded

    def _transcribe_mlx_whisper(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        file_name: str,
        language_override: str = "",
    ) -> List[TranscriptionSegment]:
        if self._mlx_whisper_module is None:
            raise RuntimeError("mlx-whisper not loaded")
        if self._mlx_disabled_for_session:
            raise RuntimeError(self._mlx_failure_reason())

        cfg = self._fw_cfg or self.asr_cfg.get(
            "faster_whisper", self.asr_cfg.get("whisperx", {})
        )
        language = (
            self._normalize_language_tag(language_override)
            or self._normalize_language_tag(cfg.get("language", ""))
        )
        model_ref = self._resolve_effective_mlx_model_ref(
            self._mlx_model_ref or self._resolve_mlx_whisper_model_ref(cfg),
            cfg,
        )
        word_timestamps = bool(
            cfg.get("word_timestamps", False)
            or self._subtitle_overlay_requests_word_timestamps()
        )
        audio_duration_sec = float(len(audio_np)) / float(max(1, sample_rate))
        timeout_sec = self._mlx_transcribe_timeout_sec(
            audio_duration_sec,
            cfg=cfg,
            word_timestamps=word_timestamps,
        )

        temp_root = self._runtime_temp_dir or Path(tempfile.gettempdir())
        temp_root.mkdir(parents=True, exist_ok=True)

        audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32)
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.flatten()
        if audio_f32.size == 0:
            raise ValueError("empty audio for mlx-whisper")
        peak = float(np.max(np.abs(audio_f32))) if audio_f32.size else 0.0
        if peak > 1.0:
            audio_f32 = audio_f32 / max(peak, 1e-6)

        audio_fd, audio_path_raw = tempfile.mkstemp(
            suffix=".npy" if sample_rate == 16000 else ".wav",
            prefix="mlx_whisper_",
            dir=str(temp_root),
        )
        audio_path = Path(audio_path_raw)
        try:
            logger.info(
                "  mlx-whisper: %s / Apple MLX (worker, timeout=%.1fs)",
                model_ref,
                timeout_sec,
            )
            if sample_rate == 16000:
                with os.fdopen(audio_fd, "wb") as audio_file:
                    np.save(audio_file, audio_f32, allow_pickle=False)
                audio_format = "npy"
            else:
                os.close(audio_fd)
                self._write_wav_mono(audio_path, audio_f32, sample_rate)
                audio_format = "path"

            result = self._mlx_worker_request(
                model_ref=model_ref,
                request={
                    "cmd": "transcribe",
                    "audio_path": str(audio_path),
                    "audio_format": audio_format,
                    "model_ref": str(model_ref),
                    "word_timestamps": bool(word_timestamps),
                    "language": str(language or ""),
                },
                timeout_sec=timeout_sec,
            )
        finally:
            try:
                os.unlink(str(audio_path))
            except OSError:
                pass

        return self._parse_mlx_whisper_results(
            result=result,
            audio_np=audio_np,
            sample_rate=sample_rate,
            language_hint=language,
        )

    def _parse_mlx_whisper_results(
        self,
        *,
        result: Any,
        audio_np: np.ndarray,
        sample_rate: int,
        language_hint: str = "",
    ) -> List[TranscriptionSegment]:
        segments: List[TranscriptionSegment] = []
        if not result:
            return segments

        if hasattr(result, "to_dict"):
            try:
                result = result.to_dict()
            except Exception:
                pass

        payload = result if isinstance(result, dict) else {}
        detected_lang = str(
            payload.get("language")
            or payload.get("detected_language")
            or language_hint
            or ""
        ).strip()
        raw_segments = payload.get("segments") or payload.get("chunks") or []
        total_duration = float(len(audio_np)) / float(max(1, sample_rate))

        for item in raw_segments or []:
            if hasattr(item, "to_dict"):
                try:
                    item = item.to_dict()
                except Exception:
                    pass
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            words_payload = item.get("words") or item.get("word_timestamps") or []
            words: List[Dict[str, Any]] = []
            for word in words_payload or []:
                if hasattr(word, "to_dict"):
                    try:
                        word = word.to_dict()
                    except Exception:
                        pass
                if not isinstance(word, dict):
                    continue
                word_text = str(
                    word.get("word")
                    or word.get("text")
                    or word.get("token")
                    or ""
                ).strip()
                if not word_text:
                    continue
                entry: Dict[str, Any] = {"text": word_text}
                if word.get("start", None) is not None:
                    entry["start"] = float(word.get("start", 0.0) or 0.0)
                if word.get("end", None) is not None:
                    entry["end"] = float(word.get("end", entry.get("start", 0.0)) or entry.get("start", 0.0))
                words.append(entry)

            segments.append(
                TranscriptionSegment(
                    start=max(0.0, start),
                    end=max(start, end),
                    text=text,
                    speaker="0",
                    language=detected_lang,
                    words=words or None,
                )
            )

        if segments:
            logger.info(
                "  mlx: lang=%s, segs=%d",
                detected_lang or "unknown",
                len(segments),
            )
            return segments

        text = str(payload.get("text", "") or "").strip()
        if text:
            segments.append(
                TranscriptionSegment(
                    start=0.0,
                    end=max(0.0, total_duration),
                    text=text,
                    speaker="0",
                    language=detected_lang,
                )
            )
        return segments

    def _subtitle_overlay_requests_word_timestamps(self) -> bool:
        try:
            if not bool(self.config.get("video_text_overlay.enabled", False)):
                return False
            effect = str(
                self.config.get("video_text_overlay.style_effect", "auto") or "auto"
            ).strip().lower()
            return effect in {"auto", "karaoke"}
        except Exception:
            return False

    # faster-whisper transcription path

    def _transcribe_faster_whisper(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        file_name: str,
        language_override: str = "",
    ) -> List[TranscriptionSegment]:
        self._wait_for_faster_whisper_startup_preload("transcribe")
        if self._faster_whisper_model is None:
            raise RuntimeError("faster-whisper not loaded")

        cfg = self._fw_cfg or {}
        audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32)
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.flatten()

        language = (
            self._normalize_language_tag(language_override)
            or cfg.get("language", None)
        )
        beam_size = cfg.get("beam_size", 5)
        vad_filter = cfg.get("vad_filter", True)
        vad_parameters = cfg.get("vad_parameters", {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        })

        # Reduce beam size under low VRAM to lower OOM risk
        if self._fw_device == "cuda" and _get_free_gpu_gb() < 2.0:
            beam_size = min(beam_size, 3)
            logger.info(
                f"  Low VRAM, beam_size reduced to {beam_size}"
            )

        transcribe_kwargs = {
            "beam_size": beam_size,
            "best_of": cfg.get("best_of", 5),
            "patience": cfg.get("patience", 1.0),
            "temperature": cfg.get(
                "temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            ),
            "vad_filter": vad_filter,
            "vad_parameters": (
                vad_parameters if vad_filter else None
            ),
            "word_timestamps": bool(
                cfg.get("word_timestamps", False)
                or self._subtitle_overlay_requests_word_timestamps()
            ),
            "without_timestamps": False,
        }
        if language:
            transcribe_kwargs["language"] = language

        segments_gen, info = self._faster_whisper_model.transcribe(
            audio_f32, **transcribe_kwargs,
        )

        detected_lang = info.language if info else "unknown"
        lang_prob = info.language_probability if info else 0.0
        duration = info.duration if info else 0.0
        logger.info(
            f"  fw: lang={detected_lang} (p={lang_prob:.2f}), "
            f"dur={duration:.1f}s"
        )

        # Materialize generator output so we can post-filter and free resources
        raw_segments = []
        try:
            for seg in segments_gen:
                word_items = []
                for word in getattr(seg, "words", []) or []:
                    word_text = str(
                        getattr(word, "word", getattr(word, "text", "")) or ""
                    )
                    if not word_text.strip():
                        continue
                    entry: Dict[str, Any] = {"text": word_text}
                    try:
                        if getattr(word, "start", None) is not None:
                            entry["start"] = float(word.start)
                    except Exception:
                        pass
                    try:
                        if getattr(word, "end", None) is not None:
                            entry["end"] = float(word.end)
                    except Exception:
                        pass
                    try:
                        if getattr(word, "probability", None) is not None:
                            entry["probability"] = float(word.probability)
                    except Exception:
                        pass
                    word_items.append(entry)
                raw_segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "words": word_items,
                })
        except Exception as e:
            logger.warning(f"Segment consumption error: {e}")

        del segments_gen
        gc.collect()

        if not raw_segments:
            return []

        # Drop empty / obvious no-speech segments
        filtered = []
        for seg in raw_segments:
            text = seg["text"].strip()
            if not text:
                continue
            if (
                seg.get("no_speech_prob", 0) > 0.9
                and seg.get("avg_logprob", 0) < -1.0
            ):
                continue
            filtered.append(seg)

        # Speaker mapping is assigned by NeMo MSDD (or default single-speaker).

        # Convert filtered raw segments to normalized segment objects
        result = []
        for seg in filtered:
            text = seg["text"].strip()
            if not text:
                continue

            speaker = "0"

            confidence = 0.0
            logprob = seg.get("avg_logprob", -1.0)
            if logprob is not None:
                confidence = min(1.0, max(0.0, math.exp(logprob)))

            result.append(TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=text,
                speaker=speaker,
                language=(language or detected_lang),
                confidence=confidence,
                words=seg.get("words", []),
            ))

        return result

    # Diarization (NeMo MSDD primary path)

    def _diarize_audio(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        segments: List[TranscriptionSegment],
        cfg: Dict[str, Any],
        file_name: str = "",
        prefer_structured_cfg: bool = False,
    ) -> List[Dict[str, Any]]:
        if not segments or sample_rate <= 0:
            return []
        if not self.use_nemo_msdd_pipeline():
            return []
        if not self._nemo_msdd_model_is_available(cfg):
            return []

        base_cfg = dict(cfg or {})
        strict = bool(cfg.get("strict", False))
        keep_temp = bool(cfg.get("keep_temp_files", False))
        run_dir: Optional[Path] = None
        out_dir: Optional[Path] = None
        wav_stem = ""

        self._ensure_diarization_dependency_compat()
        try:
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer
            from omegaconf import OmegaConf
            try:
                from nemo.collections.asr.models.configs.diarizer_config import (
                    NeuralDiarizerInferenceConfig,
                )
            except Exception:
                NeuralDiarizerInferenceConfig = None
        except Exception as e:
            err = (
                "NeMo MSDD dependencies missing. "
                "Install nemo_toolkit[asr] and omegaconf."
                f" detail={e}"
            )
            self._nemo_msdd_runtime_error = err
            if strict:
                raise RuntimeError(err) from e
            self._disable_nemo_msdd_for_session(err)
            return []

        try:
            runtime_cfg = dict(cfg or {})
            device_pref = str(runtime_cfg.get("device", "auto") or "auto").strip().lower()
            run_device = self._preferred_torch_device(device_pref, allow_mps=True)

            try:
                resolved_models = self._ensure_nemo_models_downloaded(runtime_cfg)
                if resolved_models:
                    runtime_cfg.update(resolved_models)
            except Exception as preload_err:
                preload_text = str(preload_err or "")
                if self._is_nemo_auth_error(preload_err):
                    self._nemo_msdd_runtime_error = preload_text
                    self._disable_nemo_msdd_for_session(preload_text)
                    return []
                logger.warning(
                    "NeMo MSDD model warmup failed; continuing with direct load: %s",
                    preload_err,
                )

            safe_stem = self._safe_stem(file_name)
            run_id = f"{safe_stem}_{int(time.time() * 1000)}_{os.getpid()}"
            run_dir = self._nemo_run_root() / run_id
            out_dir = run_dir / "nemo_out"
            run_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            wav_path = run_dir / f"{safe_stem}.wav"
            audio_f32 = self._write_wav_mono(wav_path, audio_np, sample_rate)
            wav_stem = str(wav_path.stem)
            duration = len(audio_f32) / float(sample_rate)

            manifest_path = run_dir / "manifest.json"
            num_speakers = max(0, self._safe_int(runtime_cfg.get("num_speakers", 0), 0))
            manifest_item: Dict[str, Any] = {
                "audio_filepath": str(wav_path),
                "offset": 0.0,
                "duration": float(duration),
                "label": "infer",
                "text": "-",
                "rttm_filepath": "",
                "uem_filepath": "",
            }
            if num_speakers > 0:
                manifest_item["num_speakers"] = num_speakers
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")

            external_vad_manifest = self._write_asr_segments_vad_manifest(
                manifest_path=run_dir / "external_vad_manifest.json",
                audio_path=wav_path,
                uniq_id=wav_stem,
                segments=segments,
                total_duration=duration,
                cfg=runtime_cfg,
            )
            diar_cfg = self._build_nemo_diarizer_cfg(
                manifest_path=manifest_path,
                out_dir=out_dir,
                sample_rate=sample_rate,
                run_device=run_device,
                cfg_override=runtime_cfg,
                external_vad_manifest=external_vad_manifest,
            )
            try:
                if prefer_structured_cfg and NeuralDiarizerInferenceConfig is not None:
                    diar_cfg_obj = OmegaConf.merge(
                        OmegaConf.structured(NeuralDiarizerInferenceConfig()),
                        OmegaConf.create(diar_cfg),
                    )
                else:
                    diar_cfg_obj = OmegaConf.create(diar_cfg)
            except Exception as cfg_merge_err:
                logger.debug(
                    "NeMo MSDD structured config merge skipped: "
                    f"{cfg_merge_err}"
                )
                diar_cfg_obj = OmegaConf.create(diar_cfg)

            diarizer = NeuralDiarizer(cfg=diar_cfg_obj)
            if run_device != "cpu" and hasattr(diarizer, "to"):
                diarizer = diarizer.to(run_device)
            diarizer.diarize()

            rttm_path = self._resolve_rttm_path(out_dir=out_dir, wav_stem=wav_path.stem)
            if rttm_path is None or not rttm_path.exists():
                raise RuntimeError("NeMo MSDD produced no RTTM output")

            diar_segments = self._load_rttm_segments(rttm_path)
            if not diar_segments:
                raise RuntimeError("NeMo MSDD RTTM is empty")

            unique_speakers = sorted({str(seg["speaker"]) for seg in diar_segments})
            logger.info(
                "  NeMo MSDD diarization complete: "
                f"{len(unique_speakers)} speaker(s), {len(diar_segments)} turn(s)"
            )
            self._nemo_msdd_runtime_error = ""
            return diar_segments

        except Exception as e:
            err = f"NeMo MSDD diarization failed: {e}"
            self._nemo_msdd_runtime_error = err
            err_lower = str(e).lower()
            default_num_workers = 0 if os.name == "nt" else 1
            current_workers = self._safe_int(cfg.get("num_workers", default_num_workers), default_num_workers)
            can_retry_relaxed_vad = (
                not strict
                and not self._safe_bool(cfg.get("__nemo_relaxed_vad_retry_done", False), False)
                and (
                    "contains silence" in err_lower
                    or "all files present in manifest contains silence" in err_lower
                )
            )
            can_retry_single_worker = (
                not strict
                and current_workers > 0
                and ("pickle" in err_lower or "speechlabelentity" in err_lower)
            )
            can_retry_unstructured = (
                not strict
                and prefer_structured_cfg
                and ("not in struct" in err_lower or "full_key" in err_lower)
            )
            can_retry_index_oob_safe_mode = (
                not strict
                and not self._safe_bool(cfg.get("__nemo_index_oob_retry_done", False), False)
                and self._is_nemo_msdd_index_oob_error(e)
            )
            if can_retry_single_worker:
                logger.warning(
                    "NeMo MSDD hit multiprocessing pickle issue; retrying once with num_workers=0."
                )
                retry_cfg = dict(cfg or {})
                retry_cfg["num_workers"] = 0
                return self._diarize_audio(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                    prefer_structured_cfg=prefer_structured_cfg,
                )
            if can_retry_unstructured:
                logger.warning(
                    "NeMo MSDD hit OmegaConf struct-key mismatch; retrying once with non-structured config."
                )
                return self._diarize_audio(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=dict(cfg or {}),
                    file_name=file_name,
                    prefer_structured_cfg=False,
                )
            if can_retry_index_oob_safe_mode:
                logger.warning(
                    "NeMo MSDD hit known index-out-of-bounds path; retrying once with safer MSDD settings."
                )
                retry_cfg = dict(cfg or {})
                retry_cfg["__nemo_index_oob_retry_done"] = True
                retry_cfg["split_infer"] = False
                retry_cfg["use_speaker_model_from_ckpt"] = False
                cur_num_speakers = self._safe_int(retry_cfg.get("num_speakers", 0), 0)
                if cur_num_speakers <= 1:
                    retry_cfg["num_speakers"] = 2
                    retry_cfg["min_speakers"] = max(
                        2, self._safe_int(retry_cfg.get("min_speakers", 1), 1)
                    )
                    retry_cfg["max_speakers"] = max(
                        self._safe_int(retry_cfg.get("max_speakers", 8), 8),
                        self._safe_int(retry_cfg.get("min_speakers", 2), 2),
                    )
                raw_th = retry_cfg.get("sigmoid_threshold", 0.7)
                if isinstance(raw_th, (list, tuple)):
                    th = self._safe_float(raw_th[0] if raw_th else 0.7, 0.7)
                else:
                    th = self._safe_float(raw_th, 0.7)
                retry_cfg["sigmoid_threshold"] = [th]
                retry_cfg["overlap_infer_spk_limit"] = max(
                    2,
                    self._safe_int(retry_cfg.get("overlap_infer_spk_limit", 5), 5),
                )
                return self._diarize_audio(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                    prefer_structured_cfg=prefer_structured_cfg,
                )
            if run_device == "mps" and not self._safe_bool(cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(cfg):
                    logger.warning(
                        "NeMo MSDD MPS path failed and CPU retry is disabled; "
                        "continuing with the next diarization route. error=%s",
                        e,
                    )
                    if strict:
                        raise RuntimeError(err) from e
                    logger.info("%s Falling back to secondary diarization pipeline.", err)
                    return []
                logger.info("NeMo MSDD MPS path failed; retrying once on CPU.")
                retry_cfg = dict(base_cfg)
                retry_cfg["device"] = "cpu"
                retry_cfg["__cpu_retry_done"] = True
                return self._diarize_audio(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                    prefer_structured_cfg=prefer_structured_cfg,
                )
            if can_retry_relaxed_vad:
                logger.info(
                    "NeMo MSDD reported manifest silence; retrying once with relaxed VAD thresholds."
                )
                retry_cfg = dict(cfg or {})
                retry_cfg["__nemo_relaxed_vad_retry_done"] = True
                retry_cfg["vad_onset"] = min(
                    0.55, self._safe_float(retry_cfg.get("vad_onset", 0.8), 0.8)
                )
                retry_cfg["vad_offset"] = min(
                    0.45, self._safe_float(retry_cfg.get("vad_offset", 0.6), 0.6)
                )
                retry_cfg["vad_min_duration_on"] = min(
                    0.05,
                    self._safe_float(retry_cfg.get("vad_min_duration_on", 0.1), 0.1),
                )
                retry_cfg["vad_min_duration_off"] = min(
                    0.1,
                    self._safe_float(retry_cfg.get("vad_min_duration_off", 0.2), 0.2),
                )
                retry_cfg["vad_pad_onset"] = max(
                    0.08,
                    self._safe_float(retry_cfg.get("vad_pad_onset", 0.05), 0.05),
                )
                retry_cfg["vad_pad_offset"] = max(
                    0.0,
                    self._safe_float(retry_cfg.get("vad_pad_offset", -0.05), -0.05),
                )
                retry_cfg["vad_filter_speech_first"] = False
                return self._diarize_audio(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                    prefer_structured_cfg=prefer_structured_cfg,
                )
            if out_dir is not None and wav_stem:
                try:
                    cached_rttm = self._resolve_rttm_path(out_dir=out_dir, wav_stem=wav_stem)
                    if cached_rttm is not None and cached_rttm.exists():
                        cached_segments = self._load_rttm_segments(cached_rttm)
                        if cached_segments:
                            logger.warning(
                                "NeMo MSDD post-processing failed but RTTM exists; using generated RTTM fallback "
                                "(%d turn(s)).",
                                len(cached_segments),
                            )
                            self._nemo_msdd_runtime_error = ""
                            return cached_segments
                except Exception:
                    pass
            if self._is_nemo_auth_error(e):
                self._disable_nemo_msdd_for_session(str(e))
            if strict:
                raise RuntimeError(err) from e
            logger.info("%s Falling back to secondary diarization pipeline.", err)
            return []
        finally:
            if run_dir is not None and run_dir.exists() and not keep_temp:
                shutil.rmtree(run_dir, ignore_errors=True)

    def _rank_speakers_for_span(
        self,
        speaker_map: List[Dict[str, Any]],
        start: float,
        end: float,
        *,
        preferred_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if end <= start:
            end = start + 0.04
        score_map: Dict[str, float] = {}
        source_map: Dict[str, set[str]] = {}
        span_mid = (start + end) * 0.5

        def _accumulate(
            items: List[Dict[str, Any]],
            *,
            base_weight: float,
            source_name: str,
        ) -> None:
            for item in items or []:
                speaker = self._normalize_speaker_id(str(item.get("speaker", "0")))
                item_start = float(item.get("start", 0.0) or 0.0)
                item_end = max(item_start, float(item.get("end", item_start) or item_start))
                overlap = self._segment_overlap_seconds(item_start, item_end, start, end)
                if overlap <= 0.0:
                    if item_start - 0.05 <= span_mid <= item_end + 0.05:
                        overlap = 0.05
                    else:
                        continue
                confidence = max(0.1, float(item.get("confidence", 1.0) or 1.0))
                score_map[speaker] = score_map.get(speaker, 0.0) + overlap * max(0.05, base_weight) * confidence
                source_map.setdefault(speaker, set()).add(source_name)

        _accumulate(list(preferred_tracks or []), base_weight=1.40, source_name="overlap_track")
        _accumulate(list(speaker_map or []), base_weight=1.0, source_name="diar")
        if not score_map and speaker_map:
            fallback = self._normalize_speaker_id(self._find_speaker(speaker_map, span_mid))
            return [{"speaker": fallback, "confidence": 1.0, "sources": ["diar"]}]

        ranked = sorted(score_map.items(), key=lambda item: (float(item[1]), str(item[0])), reverse=True)
        if not ranked:
            return []
        best_score = max(1e-6, float(ranked[0][1]))
        return [
            {
                "speaker": speaker,
                "confidence": min(1.0, float(score) / best_score),
                "sources": sorted(source_map.get(speaker, set())),
            }
            for speaker, score in ranked
        ]

    def _annotate_words_with_speakers(
        self,
        *,
        words: List[Dict[str, Any]],
        diar_segments: List[Dict[str, Any]],
        overlap_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        for raw_word in words or []:
            word = dict(raw_word)
            try:
                word_start = float(word.get("start"))
                word_end = float(word.get("end", word_start) or word_start)
            except Exception:
                annotated.append(word)
                continue
            if word_end <= word_start:
                word_end = word_start + 0.04
            candidates = self._rank_speakers_for_span(
                diar_segments,
                word_start,
                word_end,
                preferred_tracks=overlap_tracks,
            )
            if candidates:
                word["speaker"] = str(candidates[0]["speaker"])
                word["speaker_confidence"] = float(candidates[0].get("confidence", 0.0) or 0.0)
                word["speaker_candidates"] = [dict(item) for item in candidates[:3]]
                strong_overlap = [
                    str(item["speaker"])
                    for item in candidates
                    if float(item.get("confidence", 0.0) or 0.0) >= max(0.52, float(candidates[0].get("confidence", 0.0) or 0.0) * 0.72)
                ]
                if len(strong_overlap) > 1:
                    word["overlap_speakers"] = strong_overlap[:3]
            annotated.append(word)
        return annotated

    @staticmethod
    def _word_speaker_runs(
        words: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for word in words or []:
            speaker = str(word.get("speaker", "") or "").strip()
            if not speaker:
                continue
            try:
                start = float(word.get("start"))
                end = float(word.get("end", start) or start)
            except Exception:
                continue
            if end <= start:
                end = start + 0.04
            confidence = float(word.get("speaker_confidence", 0.0) or 0.0)
            if (
                runs
                and runs[-1]["speaker"] == speaker
                and start <= float(runs[-1]["end"]) + 0.16
            ):
                runs[-1]["end"] = max(float(runs[-1]["end"]), end)
                runs[-1]["words"].append(word)
                runs[-1]["confidence_sum"] = float(runs[-1]["confidence_sum"]) + confidence
                continue
            runs.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "words": [word],
                    "confidence_sum": confidence,
                }
            )

        index = 1
        while index < len(runs) - 1:
            cur = runs[index]
            prev = runs[index - 1]
            nxt = runs[index + 1]
            cur_duration = max(0.0, float(cur["end"]) - float(cur["start"]))
            cur_conf = float(cur["confidence_sum"]) / max(1, len(cur["words"]))
            cur_has_overlap = any(
                len(list(word.get("overlap_speakers") or [])) > 1
                for word in list(cur.get("words") or [])
                if isinstance(word, dict)
            )
            if (
                prev["speaker"] == nxt["speaker"]
                and prev["speaker"] != cur["speaker"]
                and len(cur["words"]) == 1
                and cur_duration <= 0.12
                and cur_conf < 0.60
                and not cur_has_overlap
            ):
                prev["end"] = max(float(prev["end"]), float(nxt["end"]))
                prev["words"].extend(cur["words"])
                prev["words"].extend(nxt["words"])
                prev["confidence_sum"] = float(prev["confidence_sum"]) + float(cur["confidence_sum"]) + float(nxt["confidence_sum"])
                del runs[index:index + 2]
                continue
            index += 1
        return runs

    def _find_speaker_for_span(
        self,
        speaker_map: List[Dict[str, Any]],
        start: float,
        end: float,
        *,
        preferred_tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not speaker_map and not preferred_tracks:
            return "0"
        if end <= start:
            end = start + 1e-3

        ranked = self._rank_speakers_for_span(
            speaker_map,
            start,
            end,
            preferred_tracks=preferred_tracks,
        )
        if ranked:
            return str(ranked[0].get("speaker", "0") or "0")

        best_speaker = "0"
        best_overlap = 0.0
        mid = (start + end) * 0.5
        nearest_dist = float("inf")

        for seg in speaker_map:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
            seg_mid = (seg_start + seg_end) * 0.5
            dist = abs(mid - seg_mid)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(seg.get("speaker", "0"))
                nearest_dist = dist
            elif overlap > 0 and abs(overlap - best_overlap) <= 1e-6 and dist < nearest_dist:
                best_speaker = str(seg.get("speaker", "0"))
                nearest_dist = dist

        if best_overlap > 0:
            return best_speaker
        return Transcriber._find_speaker(speaker_map, mid)

    @staticmethod
    def _find_speaker(
        speaker_map: List[Dict[str, Any]], time_point: float
    ) -> str:
        if not speaker_map:
            return "0"
        best_speaker = "0"
        best_dist = float("inf")
        for seg in speaker_map:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            seg_mid = (seg_start + seg_end) * 0.5
            dist = abs(float(time_point) - seg_mid)
            if dist < best_dist:
                best_dist = dist
                best_speaker = str(seg.get("speaker", "0"))
        return best_speaker

    @staticmethod
    def _normalize_speaker_id(raw: str) -> str:
        value = str(raw or "").strip()
        if not value:
            return "0"
        match = re.search(r"(\d+)$", value)
        if match:
            return str(int(match.group(1)))
        return value

    def _pyannote_diar_fallback_cfg(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        root = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        fb_cfg = root.get("pyannote_fallback", {}) if isinstance(root, dict) else {}
        return fb_cfg if isinstance(fb_cfg, dict) else {}

    def _torchaudio_fallback_cfg(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        root = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        fb_cfg = root.get("torchaudio_fallback", {}) if isinstance(root, dict) else {}
        return fb_cfg if isinstance(fb_cfg, dict) else {}

    @staticmethod
    def _pyannote_device_text(value: Any) -> str:
        if value is None:
            return ""
        try:
            return str(value).strip()
        except Exception:
            return ""

    @staticmethod
    def _is_accelerator_device(device_name: Any) -> bool:
        text = str(device_name or "").strip().lower()
        return text == "mps" or text.startswith("cuda")

    def _summarize_pyannote_pipeline_devices(self, pipeline: Any) -> Dict[str, str]:
        summary: Dict[str, str] = {}
        queue: List[Tuple[str, Any, int]] = [("pipeline", pipeline, 0)]
        visited: set[int] = set()
        attr_names = (
            "segmentation",
            "embedding",
            "clustering",
            "inference",
            "model",
            "model_",
            "segmentation_model_",
            "embedding_model_",
        )
        while queue:
            prefix, obj, depth = queue.pop(0)
            if obj is None:
                continue
            obj_id = id(obj)
            if obj_id in visited or depth > 2:
                continue
            visited.add(obj_id)
            try:
                device_text = self._pyannote_device_text(getattr(obj, "device", None))
            except Exception:
                device_text = ""
            if device_text:
                summary[prefix] = device_text
            if depth >= 2:
                continue
            for attr_name in attr_names:
                try:
                    child = getattr(obj, attr_name, None)
                except Exception:
                    continue
                if child is None:
                    continue
                queue.append((f"{prefix}.{attr_name}", child, depth + 1))
        return summary

    def _log_pyannote_pipeline_device_summary(
        self,
        *,
        pipeline: Any,
        runtime_device: str,
        model_name: str,
        file_name: str = "",
    ) -> None:
        device_summary = self._summarize_pyannote_pipeline_devices(pipeline)
        if device_summary:
            summary_text = ", ".join(f"{name}={device}" for name, device in sorted(device_summary.items()))
            logger.info(
                "  pyannote diarization device summary: requested=%s, model=%s, %s%s",
                runtime_device,
                model_name,
                summary_text,
                f" ({file_name})" if file_name else "",
            )
        model_lower = str(model_name or "").strip().lower()
        has_cpu_substage = any(str(device).strip().lower().startswith("cpu") for device in device_summary.values())
        if runtime_device == "mps" and ("community-1" in model_lower or has_cpu_substage):
            logger.warning(
                "  pyannote community-1 on MPS is only partially accelerated; embedding/clustering stages may still run on CPU%s",
                f" ({file_name})" if file_name else "",
            )

    def _diarize_audio_pyannote_fallback(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        segments: List[TranscriptionSegment],
        cfg: Optional[Dict[str, Any]] = None,
        file_name: str = "",
    ) -> List[Dict[str, Any]]:
        base_cfg = dict(cfg or self._nemo_msdd_cfg())
        fb_cfg = self._pyannote_diar_fallback_cfg(cfg)
        if not self._safe_bool(fb_cfg.get("enabled", True), True):
            return []
        if sample_rate <= 0 or audio_np is None or len(segments or []) < 2:
            return []

        provider = str(fb_cfg.get("provider", "pyannote.audio") or "pyannote.audio").strip().lower()
        if provider not in {"pyannote", "pyannote.audio", "auto"}:
            self._pyannote_diar_runtime_error = f"unsupported pyannote diar provider: {provider}"
            return []

        model_name = str(
            fb_cfg.get("model_name", "pyannote/speaker-diarization-3.1")
            or "pyannote/speaker-diarization-3.1"
        ).strip()
        if not model_name:
            model_name = "pyannote/speaker-diarization-3.1"
        model_candidates_raw = fb_cfg.get("model_candidates", [])
        model_candidates: List[str] = [model_name]
        if isinstance(model_candidates_raw, (list, tuple)):
            for item in model_candidates_raw:
                candidate = str(item or "").strip()
                if candidate:
                    model_candidates.append(candidate)
        deduped_candidates: List[str] = []
        seen_candidates: set[str] = set()
        for item in model_candidates:
            key = item.lower()
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            deduped_candidates.append(item)
        model_candidates = deduped_candidates or [model_name]
        incompatible_models = {
            str(item or "").strip().lower()
            for item in self._pyannote_incompatible_models
            if str(item or "").strip()
        }
        compatible_candidates = [
            candidate
            for candidate in model_candidates
            if str(candidate or "").strip().lower() not in incompatible_models
        ]
        if compatible_candidates:
            model_candidates = compatible_candidates

        device_pref = str(fb_cfg.get("device", "auto") or "auto").strip().lower()
        runtime_device = self._preferred_torch_device(device_pref, allow_mps=True)
        require_gpu = self._safe_bool(fb_cfg.get("require_gpu", False), False)
        if require_gpu and not self._is_accelerator_device(runtime_device):
            requested_device = self._normalize_device_pref(device_pref or "auto") or "auto"
            self._pyannote_diar_runtime_error = (
                "pyannote-diar-gpu-required: "
                f"requested={requested_device}, resolved={runtime_device}"
            )
            logger.warning(
                "pyannote diarization skipped: GPU required but unavailable "
                "(requested=%s, resolved=%s)%s",
                requested_device,
                runtime_device,
                f" ({file_name})" if file_name else "",
            )
            return []
        audio_duration_sec = (
            float(len(audio_np)) / float(max(1, sample_rate))
            if sample_rate > 0 and audio_np is not None
            else 0.0
        )
        setup = {
            "model_candidates": tuple(model_candidates),
            "device": runtime_device,
        }

        try:
            cached_candidates = tuple(self._pyannote_diar_setup.get("model_candidates", ()))
            cached_device = str(self._pyannote_diar_setup.get("device", "") or "")
            need_reload = (
                self._pyannote_diar_pipeline is None
                or cached_device != runtime_device
                or cached_candidates != tuple(model_candidates)
            )
            if need_reload:
                logger.info(
                    "  pyannote diarization init: device=%s, duration=%.1fs, candidates=%s%s",
                    runtime_device,
                    audio_duration_sec,
                    ",".join(model_candidates),
                    f" ({file_name})" if file_name else "",
                )
                token = str(fb_cfg.get("hf_token", "") or "").strip()
                if not token:
                    token = self._resolve_hf_token(self.asr_cfg.get("faster_whisper", {}) or {})
                pipeline = None
                selected_model = ""
                last_error: Optional[Exception] = None
                total_candidates = max(1, len(model_candidates))
                for candidate_index, candidate_model in enumerate(model_candidates, start=1):
                    try:
                        pipeline = self._load_pyannote_pipeline_with_retry(
                            model_name=candidate_model,
                            cfg=fb_cfg,
                            token=token,
                            purpose="pyannote-diar",
                        )
                        selected_model = candidate_model
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        msg = str(e or "")
                        hint = ""
                        msg_lower = msg.lower()
                        if any(
                            marker in msg_lower
                            for marker in ("restricted", "gated", "401", "403", "forbidden", "token")
                        ):
                            hint = " (check HF token and access approval for this model)"
                        if candidate_index < total_candidates:
                            logger.info(
                                "pyannote diarization init failed for %s (%d/%d), trying next candidate: %s%s",
                                candidate_model,
                                candidate_index,
                                total_candidates,
                                e,
                                hint,
                            )
                        else:
                            logger.debug(
                                "pyannote diarization init failed for %s (%d/%d): %s%s",
                                candidate_model,
                                candidate_index,
                                total_candidates,
                                e,
                                hint,
                            )
                        if self._is_pyannote_known_non_retryable_error(e):
                            self._pyannote_incompatible_models.add(
                                str(candidate_model or "").strip().lower()
                            )
                        if not self._is_pyannote_known_non_retryable_error(e):
                            logger.debug(traceback.format_exc())
                if pipeline is None:
                    raise RuntimeError(last_error or "pyannote diarization init failed")
                if hasattr(pipeline, "to"):
                    move_t0 = time.perf_counter()
                    logger.info(
                        "  pyannote diarization moving pipeline to %s%s",
                        runtime_device,
                        f" ({file_name})" if file_name else "",
                    )
                    if runtime_device == "mps":
                        smart_empty_cache(force=True)
                    pipeline = pipeline.to(torch.device(runtime_device))
                    if runtime_device == "mps":
                        _sync_mps()
                    logger.info(
                        "  pyannote diarization pipeline ready on %s in %.1fs%s",
                        runtime_device,
                        max(0.0, time.perf_counter() - move_t0),
                        f" ({file_name})" if file_name else "",
                    )
                    self._log_pyannote_pipeline_device_summary(
                        pipeline=pipeline,
                        runtime_device=runtime_device,
                        model_name=selected_model or model_candidates[0],
                        file_name=file_name,
                    )

                self._pyannote_diar_pipeline = pipeline
                setup = {
                    "model_candidates": tuple(model_candidates),
                    "active_model": selected_model or model_candidates[0],
                    "device": runtime_device,
                }
                self._pyannote_diar_setup = dict(setup)
                self._pyannote_diar_runtime_error = ""
        except Exception as e:
            self._pyannote_diar_runtime_error = f"pyannote-diar-init-failed: {e}"
            if self._is_pyannote_known_non_retryable_error(e):
                logger.info(f"pyannote diarization fallback init skipped due to known model/runtime issue: {e}")
            else:
                logger.warning(f"pyannote diarization fallback init failed: {e}")
                logger.debug(traceback.format_exc())
            if runtime_device == "mps" and not self._safe_bool(fb_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(fb_cfg):
                    logger.warning(
                        "pyannote diarization MPS init failed and CPU retry is disabled; "
                        "continuing with the next diarization route. error=%s",
                        e,
                    )
                    return []
                logger.info("pyannote diarization MPS init failed; retrying once on CPU.")
                retry_fb_cfg = dict(fb_cfg)
                retry_fb_cfg["device"] = "cpu"
                retry_fb_cfg["__cpu_retry_done"] = True
                retry_cfg = dict(base_cfg)
                retry_cfg["pyannote_fallback"] = retry_fb_cfg
                return self._diarize_audio_pyannote_fallback(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                )
            return []

        if self._pyannote_diar_pipeline is None:
            return []

        audio_f32 = self._audio_to_numpy(audio_np)
        if audio_f32.size == 0:
            return []
        waveform = torch.from_numpy(np.ascontiguousarray(audio_f32, dtype=np.float32)).unsqueeze(0)
        try:
            move_t0 = time.perf_counter()
            logger.info(
                "  pyannote diarization moving waveform to %s%s",
                runtime_device,
                f" ({file_name})" if file_name else "",
            )
            if runtime_device == "mps":
                smart_empty_cache(force=True)
            waveform = self._move_torch_waveform(waveform, runtime_device)
            if runtime_device == "mps":
                _sync_mps()
            logger.info(
                "  pyannote diarization waveform ready on %s in %.1fs%s",
                runtime_device,
                max(0.0, time.perf_counter() - move_t0),
                f" ({file_name})" if file_name else "",
            )
        except Exception as e:
            self._pyannote_diar_runtime_error = f"pyannote-diar-waveform-move-failed: {e}"
            if runtime_device == "mps" and not self._safe_bool(fb_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(fb_cfg):
                    logger.warning(
                        "pyannote diarization MPS waveform move failed and CPU retry is disabled; "
                        "continuing with the next diarization route. error=%s",
                        e,
                    )
                    return []
                logger.info("pyannote diarization MPS waveform move failed; retrying once on CPU.")
                retry_fb_cfg = dict(fb_cfg)
                retry_fb_cfg["device"] = "cpu"
                retry_fb_cfg["__cpu_retry_done"] = True
                retry_cfg = dict(base_cfg)
                retry_cfg["pyannote_fallback"] = retry_fb_cfg
                return self._diarize_audio_pyannote_fallback(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                )
            return []

        base_cfg = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        fixed_num = max(
            0,
            self._safe_int(
                fb_cfg.get("num_speakers", base_cfg.get("num_speakers", 0)),
                self._safe_int(base_cfg.get("num_speakers", 0), 0),
            ),
        )
        min_spk = max(
            1,
            self._safe_int(
                fb_cfg.get("min_speakers", base_cfg.get("min_speakers", 1)),
                self._safe_int(base_cfg.get("min_speakers", 1), 1),
            ),
        )
        max_spk = max(
            min_spk,
            self._safe_int(
                fb_cfg.get("max_speakers", base_cfg.get("max_speakers", 8)),
                self._safe_int(base_cfg.get("max_speakers", 8), 8),
            ),
        )

        infer_kwargs: Dict[str, Any] = {}
        if fixed_num > 0:
            infer_kwargs["num_speakers"] = int(fixed_num)
        else:
            infer_kwargs["min_speakers"] = int(min_spk)
            infer_kwargs["max_speakers"] = int(max_spk)

        try:
            infer_t0 = time.perf_counter()
            logger.info(
                "  pyannote diarization inference start: device=%s, duration=%.1fs, kwargs=%s%s",
                runtime_device,
                audio_duration_sec,
                infer_kwargs,
                f" ({file_name})" if file_name else "",
            )
            with torch.inference_mode():
                try:
                    output = self._pyannote_diar_pipeline(
                        {"waveform": waveform, "sample_rate": int(sample_rate)},
                        **infer_kwargs,
                    )
                except TypeError:
                    # Older pyannote pipeline versions may not accept speaker-count kwargs.
                    output = self._pyannote_diar_pipeline(
                        {"waveform": waveform, "sample_rate": int(sample_rate)}
                    )
            if runtime_device == "mps":
                _sync_mps()
            logger.info(
                "  pyannote diarization inference complete in %.1fs%s",
                max(0.0, time.perf_counter() - infer_t0),
                f" ({file_name})" if file_name else "",
            )
        except Exception as e:
            self._pyannote_diar_runtime_error = f"pyannote-diar-run-failed: {e}"
            logger.warning(f"pyannote diarization fallback failed: {e}")
            logger.debug(traceback.format_exc())
            if runtime_device == "mps" and not self._safe_bool(fb_cfg.get("__cpu_retry_done", False), False):
                if not self._allow_cpu_retry_after_mps_failure(fb_cfg):
                    logger.warning(
                        "pyannote diarization MPS inference failed and CPU retry is disabled; "
                        "continuing with the next diarization route. error=%s",
                        e,
                    )
                    return []
                logger.info("pyannote diarization MPS inference failed; retrying once on CPU.")
                retry_fb_cfg = dict(fb_cfg)
                retry_fb_cfg["device"] = "cpu"
                retry_fb_cfg["__cpu_retry_done"] = True
                retry_cfg = dict(base_cfg)
                retry_cfg["pyannote_fallback"] = retry_fb_cfg
                return self._diarize_audio_pyannote_fallback(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=retry_cfg,
                    file_name=file_name,
                )
            return []

        diar_segments: List[Dict[str, Any]] = []
        min_turn_sec = max(0.01, self._safe_float(fb_cfg.get("min_turn_sec", 0.05), 0.05))
        try:
            if hasattr(output, "itertracks"):
                for turn, _track, label in output.itertracks(yield_label=True):
                    start = float(getattr(turn, "start", 0.0))
                    end = float(getattr(turn, "end", start))
                    if end - start < min_turn_sec:
                        continue
                    diar_segments.append(
                        {
                            "start": start,
                            "end": end,
                            "speaker": str(label if label is not None else "0"),
                        }
                    )
            else:
                self._pyannote_diar_runtime_error = "pyannote-diar-parse-failed: unsupported output type"
                return []
        except Exception as e:
            self._pyannote_diar_runtime_error = f"pyannote-diar-parse-failed: {e}"
            logger.warning(f"pyannote diarization fallback parse failed: {e}")
            logger.debug(traceback.format_exc())
            return []

        if not diar_segments:
            return []

        merge_gap_sec = max(0.0, self._safe_float(fb_cfg.get("merge_gap_sec", 0.08), 0.08))
        merged_segments = self._merge_adjacent_speaker_turns(
            diar_segments,
            merge_gap_sec=merge_gap_sec,
        )
        logger.info(
            "  pyannote diarization raw output: %d speaker(s), %d turn(s)%s",
            len({str(seg["speaker"]) for seg in merged_segments}),
            len(merged_segments),
            f" ({file_name})" if file_name else "",
        )
        merged_segments = self._refine_pyannote_diar_segments(
            audio_np=audio_np,
            sample_rate=sample_rate,
            diar_segments=merged_segments,
            cfg=cfg,
            file_name=file_name,
        )

        unique_speakers = sorted({str(seg["speaker"]) for seg in merged_segments})
        logger.info(
            "  pyannote.audio diarization fallback complete: %d speaker(s), %d turn(s)%s",
            len(unique_speakers),
            len(merged_segments),
            f" ({file_name})" if file_name else "",
        )
        self._pyannote_diar_runtime_error = ""
        return merged_segments

    @staticmethod
    def _kmeans_numpy(
        features: np.ndarray,
        k: int,
        max_iter: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        x = np.asarray(features, dtype=np.float32)
        n = int(x.shape[0]) if x.ndim == 2 else 0
        if n <= 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0, 0), dtype=np.float32), 0.0

        k = max(1, min(int(k), n))
        if k == 1:
            center = np.mean(x, axis=0, keepdims=True)
            inertia = float(np.sum((x - center[0]) ** 2))
            return np.zeros((n,), dtype=np.int32), center.astype(np.float32), inertia

        centers = np.zeros((k, x.shape[1]), dtype=np.float32)
        centers[0] = x[0]
        chosen = [0]
        for ci in range(1, k):
            best_idx = 0
            best_dist = -1.0
            for i in range(n):
                if i in chosen:
                    continue
                d = min(float(np.sum((x[i] - centers[j]) ** 2)) for j in range(ci))
                if d > best_dist:
                    best_dist = d
                    best_idx = i
            centers[ci] = x[best_idx]
            chosen.append(best_idx)

        labels = np.zeros((n,), dtype=np.int32)
        for _ in range(max(2, int(max_iter))):
            dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(k):
                mask = labels == ci
                if not np.any(mask):
                    farthest_idx = int(np.argmax(np.min(dists, axis=1)))
                    centers[ci] = x[farthest_idx]
                    continue
                centers[ci] = x[mask].mean(axis=0)

        final_dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = float(np.sum(final_dists[np.arange(n), labels]))
        return labels, centers.astype(np.float32), inertia

    @staticmethod
    def _silhouette_score_numpy(features: np.ndarray, labels: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        n = int(x.shape[0]) if x.ndim == 2 else 0
        if n < 3:
            return -1.0
        unique = [int(v) for v in np.unique(y)]
        if len(unique) <= 1:
            return -1.0

        dist = np.sqrt(np.maximum(0.0, ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)))
        scores: List[float] = []
        for i in range(n):
            same_mask = (y == y[i])
            same_count = int(np.sum(same_mask))
            if same_count <= 1:
                scores.append(0.0)
                continue
            a = float(np.sum(dist[i, same_mask]) / max(1, same_count - 1))
            b = float("inf")
            for cls in unique:
                if cls == int(y[i]):
                    continue
                other_mask = (y == cls)
                if not np.any(other_mask):
                    continue
                b = min(b, float(np.mean(dist[i, other_mask])))
            if not np.isfinite(b):
                scores.append(0.0)
                continue
            denom = max(a, b, 1e-6)
            scores.append((b - a) / denom)
        if not scores:
            return -1.0
        return float(np.mean(scores))

    def _cluster_torchaudio_embeddings(
        self,
        features: np.ndarray,
        *,
        num_speakers: int,
        min_speakers: int,
        max_speakers: int,
        auto_silhouette_min: float,
    ) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32)

        # Standardize to reduce scale bias across MFCC dimensions.
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        x = (x - mean) / np.where(std < 1e-6, 1.0, std)

        n = int(x.shape[0])
        fixed_k = max(0, int(num_speakers))
        if fixed_k > 0:
            labels, _centers, _inertia = self._kmeans_numpy(x, min(fixed_k, n))
            return labels

        min_k = max(1, min(int(min_speakers), n))
        max_k = max(min_k, min(int(max_speakers), n))
        max_k = min(max_k, 6)  # keep fallback cheap/stable
        if max_k <= 1:
            return np.zeros((n,), dtype=np.int32)

        best_labels = np.zeros((n,), dtype=np.int32)
        best_score = -1.0
        best_k = 1
        for k in range(max(2, min_k), max_k + 1):
            labels, _centers, _inertia = self._kmeans_numpy(x, k)
            score = self._silhouette_score_numpy(x, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

        if min_k <= 1 and (best_k <= 1 or best_score < float(auto_silhouette_min)):
            return np.zeros((n,), dtype=np.int32)
        return best_labels

    def _assign_speakers_torchaudio_fallback(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        segments: List[TranscriptionSegment],
        cfg: Optional[Dict[str, Any]] = None,
        file_name: str = "",
    ) -> bool:
        fb_cfg = self._torchaudio_fallback_cfg(cfg)
        if not self._safe_bool(fb_cfg.get("enabled", True), True):
            return False
        if sample_rate <= 0 or audio_np is None or len(segments or []) < 2:
            return False

        try:
            target_sr = max(8000, self._safe_int(fb_cfg.get("sample_rate", 16000), 16000))
            n_mfcc = max(8, self._safe_int(fb_cfg.get("n_mfcc", 20), 20))
            min_seg_sec = max(0.20, self._safe_float(fb_cfg.get("min_segment_sec", 0.60), 0.60))
            max_seg_sec = max(min_seg_sec, self._safe_float(fb_cfg.get("max_segment_sec", 12.0), 12.0))
            max_probe_segments = max(8, self._safe_int(fb_cfg.get("max_probe_segments", 120), 120))
            auto_silhouette_min = self._safe_float(fb_cfg.get("auto_silhouette_min", 0.08), 0.08)
            min_samples = max(256, int(min_seg_sec * target_sr))

            base_cfg = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
            fixed_num = max(
                0,
                self._safe_int(
                    fb_cfg.get("num_speakers", base_cfg.get("num_speakers", 0)),
                    self._safe_int(base_cfg.get("num_speakers", 0), 0),
                ),
            )
            min_spk = max(
                1,
                self._safe_int(
                    fb_cfg.get("min_speakers", base_cfg.get("min_speakers", 1)),
                    self._safe_int(base_cfg.get("min_speakers", 1), 1),
                ),
            )
            max_spk = max(
                min_spk,
                self._safe_int(
                    fb_cfg.get("max_speakers", base_cfg.get("max_speakers", 8)),
                    self._safe_int(base_cfg.get("max_speakers", 8), 8),
                ),
            )

            audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32).reshape(-1)
            if audio_f32.size == 0:
                return False

            peak = float(np.max(np.abs(audio_f32))) if audio_f32.size else 0.0
            if peak > 1.0:
                audio_f32 = audio_f32 / max(peak, 1e-6)

            torchaudio = None
            mfcc = None
            feature_backend = "numpy"
            if not self._torchaudio_fallback_blocked:
                try:
                    import torchaudio as _torchaudio  # type: ignore

                    torchaudio = _torchaudio
                    mfcc = torchaudio.transforms.MFCC(
                        sample_rate=int(target_sr),
                        n_mfcc=int(n_mfcc),
                        melkwargs={
                            "n_fft": 400,
                            "hop_length": 160,
                            "n_mels": 40,
                        },
                    )
                    feature_backend = "torchaudio"
                except Exception as e:
                    err_text = str(e or "")
                    err_lower = err_text.lower()
                    if "libtorchcodec" in err_lower or "torchcodec" in err_lower:
                        self._torchaudio_fallback_blocked = True
                        self._torchaudio_fallback_reason = err_text
                        logger.info(
                            "torchaudio backend unavailable (libtorchcodec missing); "
                            "using numpy speaker fallback."
                        )
                    else:
                        logger.debug(f"torchaudio speaker features unavailable: {e}")
                    torchaudio = None
                    mfcc = None
                    feature_backend = "numpy"

            def _numpy_embedding(clip: np.ndarray) -> Optional[np.ndarray]:
                return self._numpy_speaker_embedding(
                    clip,
                    source_sr=int(sample_rate),
                    target_sr=int(target_sr),
                    min_samples=int(min_samples),
                )

            probe_indices: List[int] = []
            probe_embeddings: List[np.ndarray] = []
            segment_mids: List[float] = [float((seg.start + seg.end) * 0.5) for seg in segments]
            candidate_windows: List[Tuple[int, float, float]] = []
            for idx, seg in enumerate(segments):
                start = max(0.0, float(seg.start))
                end = max(start + 1e-3, float(seg.end))
                dur = end - start
                if dur < min_seg_sec * 0.5:
                    continue
                if dur > max_seg_sec:
                    mid = (start + end) * 0.5
                    half = max_seg_sec * 0.5
                    start = max(0.0, mid - half)
                    end = max(start + 1e-3, mid + half)
                candidate_windows.append((idx, start, end))

            if len(candidate_windows) < 2:
                return False

            if len(candidate_windows) > max_probe_segments:
                picks = np.linspace(
                    0,
                    len(candidate_windows) - 1,
                    num=max_probe_segments,
                    dtype=np.int32,
                ).tolist()
                seen = set()
                probe_windows = []
                for p in picks:
                    i = int(p)
                    if i in seen:
                        continue
                    seen.add(i)
                    probe_windows.append(candidate_windows[i])
            else:
                probe_windows = candidate_windows

            for idx, start, end in probe_windows:
                s_idx = max(0, min(audio_f32.size, int(start * sample_rate)))
                e_idx = max(s_idx + 1, min(audio_f32.size, int(end * sample_rate)))
                clip = audio_f32[s_idx:e_idx]
                if clip.size < 32:
                    continue

                emb: Optional[np.ndarray] = None
                if torchaudio is not None and mfcc is not None:
                    try:
                        wav = torch.from_numpy(clip).to(torch.float32).unsqueeze(0)
                        if int(sample_rate) != int(target_sr):
                            wav = torchaudio.functional.resample(
                                wav,
                                int(sample_rate),
                                int(target_sr),
                            )
                        if int(wav.shape[-1]) < min_samples:
                            pad = min_samples - int(wav.shape[-1])
                            wav = torch.nn.functional.pad(wav, (0, pad))

                        feat = mfcc(wav)  # [1, n_mfcc, T]
                        if feat.ndim == 3 and feat.shape[-1] > 1:
                            feat2d = feat.squeeze(0)
                            feat_mean = feat2d.mean(dim=1)
                            feat_std = feat2d.std(dim=1)
                            rms = torch.sqrt(torch.clamp((wav ** 2).mean(), min=1e-9)).reshape(1)
                            zcr = (
                                (torch.sign(wav[:, 1:]) != torch.sign(wav[:, :-1]))
                                .to(torch.float32)
                                .mean()
                                .reshape(1)
                            )
                            emb = torch.cat([feat_mean, feat_std, rms, zcr], dim=0)
                            emb = emb.cpu().numpy().astype(np.float32)
                    except Exception as e:
                        err_text = str(e or "")
                        if "libtorchcodec" in err_text.lower() or "torchcodec" in err_text.lower():
                            self._torchaudio_fallback_blocked = True
                            self._torchaudio_fallback_reason = err_text
                            torchaudio = None
                            mfcc = None
                            feature_backend = "numpy"
                            logger.info(
                                "torchaudio runtime failed (libtorchcodec missing); "
                                "switching to numpy speaker fallback."
                            )
                        else:
                            logger.debug(f"torchaudio embedding extraction failed: {e}")
                if emb is None:
                    emb = _numpy_embedding(clip)
                if emb is None:
                    continue
                probe_indices.append(int(idx))
                probe_embeddings.append(emb)

            if len(probe_embeddings) < 2:
                return False

            features = np.stack(probe_embeddings, axis=0)
            labels = self._cluster_torchaudio_embeddings(
                features,
                num_speakers=fixed_num,
                min_speakers=min_spk,
                max_speakers=max_spk,
                auto_silhouette_min=auto_silhouette_min,
            )
            if labels.size == 0:
                return False

            probe_label_map: Dict[int, int] = {
                int(seg_idx): int(lbl)
                for seg_idx, lbl in zip(probe_indices, labels.tolist())
            }
            unique_labels = sorted(set(probe_label_map.values()))
            if not unique_labels:
                return False

            # Assign missing/short segments by nearest labeled segment in time.
            for idx, seg in enumerate(segments):
                if idx in probe_label_map:
                    seg.speaker = str(int(probe_label_map[idx]))
                    continue
                best_label = 0
                best_dist = float("inf")
                target_mid = segment_mids[idx]
                for probe_idx in probe_indices:
                    dist = abs(target_mid - segment_mids[probe_idx])
                    if dist < best_dist:
                        best_dist = dist
                        best_label = int(probe_label_map.get(probe_idx, 0))
                seg.speaker = str(best_label)

            # Smooth isolated one-segment spikes.
            for i in range(1, len(segments) - 1):
                prev_lbl = str(segments[i - 1].speaker or "")
                cur_lbl = str(segments[i].speaker or "")
                next_lbl = str(segments[i + 1].speaker or "")
                cur_dur = max(0.0, float(segments[i].end) - float(segments[i].start))
                if prev_lbl and prev_lbl == next_lbl and cur_lbl != prev_lbl and cur_dur <= 1.2:
                    segments[i].speaker = prev_lbl

            logger.info(
                "  torchaudio speaker fallback assigned %d segment(s), %d speaker(s), backend=%s%s",
                len(segments),
                len({str(seg.speaker) for seg in segments if str(seg.speaker).strip()}),
                feature_backend,
                f" ({file_name})" if file_name else "",
            )
            return True
        except Exception as e:
            err_text = str(e or "")
            err_lower = err_text.lower()
            if "libtorchcodec" in err_lower or "torchcodec" in err_lower:
                self._torchaudio_fallback_blocked = True
                self._torchaudio_fallback_reason = err_text
                logger.info("torchaudio dependency issue detected; using non-torchaudio diar fallback.")
            else:
                logger.warning(f"torchaudio speaker fallback failed: {e}")
            logger.debug(traceback.format_exc())
            return False

    def assign_speakers(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        segments: List[TranscriptionSegment],
        file_name: str = "",
        map_speakers: bool = True,
    ) -> List[TranscriptionSegment]:
        if not segments:
            self.last_speaker_languages = {}
            self.last_diarization_route = "disabled(no-segments)"
            self.last_detected_speaker_ids = []
            self.last_detected_speaker_count = 0
            return segments

        nemo_cfg = self._nemo_msdd_cfg()
        diar_segments: List[Dict[str, Any]] = []
        diarization_route = "disabled"
        msdd_available = False
        msdd_model = ""
        hybrid_pyannote_attempted = False
        preferred_overlap_regions: List[Dict[str, float]] = []
        preferred_overlap_speaker_tracks: List[Dict[str, Any]] = []
        if map_speakers and self.use_nemo_msdd_pipeline():
            self._startup_preload_nemo_models_once(nemo_cfg)
            logger.info(
                "  Speaker constraints from config: mode=%s, num=%d, min=%d, max=%d%s",
                "manual" if self._safe_int(nemo_cfg.get("num_speakers", 0), 0) > 0 else "auto",
                self._safe_int(nemo_cfg.get("num_speakers", 0), 0),
                self._safe_int(nemo_cfg.get("min_speakers", 1), 1),
                self._safe_int(nemo_cfg.get("max_speakers", 8), 8),
                f" ({file_name})" if file_name else "",
            )
            sf_cfg = self._nemo_sortformer_cfg(nemo_cfg)
            final_strategy = self._nemo_final_diarization_strategy(nemo_cfg)
            sortformer_enabled = self._safe_bool(sf_cfg.get("enabled", True), True)
            sf_model = str(
                sf_cfg.get("model_name", "nvidia/diar_streaming_sortformer_4spk-v2.1")
                or "nvidia/diar_streaming_sortformer_4spk-v2.1"
            ).strip()
            msdd_model = str(
                nemo_cfg.get("model_path", "diar_msdd_telephonic")
                or "diar_msdd_telephonic"
            ).strip()
            msdd_available = self._nemo_msdd_model_is_available(nemo_cfg)
            msdd_reason = (
                self._nemo_msdd_model_unavailable_reason
                or self._nemo_msdd_disabled_reason
                or self._nemo_msdd_runtime_error
            )
            logger.info(
                "  NeMo diarization plan: sortformer=%s (%s), msdd=%s (%s), final=%s%s",
                "enabled" if sortformer_enabled else "disabled",
                sf_model,
                "ready" if msdd_available else "skip",
                msdd_model,
                final_strategy,
                f", reason={msdd_reason}" if (not msdd_available and msdd_reason) else "",
            )

            if final_strategy == "hybrid":
                audio_duration = (
                    float(len(audio_np)) / float(max(1, sample_rate))
                    if sample_rate > 0
                    else 0.0
                )
                hybrid_candidates: List[Dict[str, Any]] = []
                hybrid_candidate_map: Dict[str, List[Dict[str, Any]]] = {}
                fixed_num = max(0, self._safe_int(nemo_cfg.get("num_speakers", 0), 0))
                max_speakers = max(
                    1,
                    self._safe_int(
                        nemo_cfg.get("max_speakers", 8),
                        8,
                    ),
                    fixed_num,
                )

                if sortformer_enabled:
                    sortformer_segments = self._diarize_audio_nemo_sortformer(
                        audio_np=audio_np,
                        sample_rate=sample_rate,
                        segments=segments,
                        cfg=nemo_cfg,
                        file_name=file_name,
                    )
                    if sortformer_segments:
                        hybrid_candidate_map["sortformer"] = list(sortformer_segments)
                        hybrid_candidates.append(
                            {
                                "backend": "sortformer",
                                "route": "NeMo-Sortformer",
                                "segments": sortformer_segments,
                                "speaker_count": count_diar_speakers(sortformer_segments),
                                "turn_count": len(sortformer_segments),
                            }
                        )

                if msdd_available:
                    msdd_segments = self._diarize_audio(
                        audio_np=audio_np,
                        sample_rate=sample_rate,
                        segments=segments,
                        cfg=nemo_cfg,
                        file_name=file_name,
                    )
                    if msdd_segments:
                        hybrid_candidate_map["msdd"] = list(msdd_segments)
                        hybrid_candidates.append(
                            {
                                "backend": "msdd",
                                "route": "NeMo-MSDD",
                                "segments": msdd_segments,
                                "speaker_count": count_diar_speakers(msdd_segments),
                                "turn_count": len(msdd_segments),
                            }
                        )

                fb_cfg = self._pyannote_diar_fallback_cfg(nemo_cfg)
                pyannote_runtime_device = self._preferred_torch_device(
                    str(fb_cfg.get("device", "auto") or "auto"),
                    allow_mps=True,
                )
                should_probe_pyannote = should_probe_pyannote_hybrid(
                    hybrid_candidates,
                    audio_duration=audio_duration,
                    max_speakers=max_speakers,
                    requested_num_speakers=fixed_num,
                )

                if should_probe_pyannote:
                    logger.info(
                        "  Pyannote hybrid probe enabled: device=%s, duration=%.1fs, primary_candidates=%d%s",
                        pyannote_runtime_device,
                        audio_duration,
                        len(hybrid_candidates),
                        f" ({file_name})" if file_name else "",
                    )
                    hybrid_pyannote_attempted = True
                    pyannote_segments = self._diarize_audio_pyannote_fallback(
                        audio_np=audio_np,
                        sample_rate=sample_rate,
                        segments=segments,
                        cfg=nemo_cfg,
                        file_name=file_name,
                    )
                    if pyannote_segments:
                        hybrid_candidate_map["pyannote"] = list(pyannote_segments)
                        hybrid_candidates.append(
                            {
                                "backend": "pyannote",
                                "route": "pyannote.audio",
                                "segments": pyannote_segments,
                                "speaker_count": count_diar_speakers(pyannote_segments),
                                "turn_count": len(pyannote_segments),
                            }
                        )

                if hybrid_candidates:
                    fused_result = self._posterior_fusion_decoder.decode(
                        audio_duration=audio_duration,
                        candidates=hybrid_candidate_map,
                        cfg=nemo_cfg,
                        file_name=file_name,
                        output_root=Path(self.config["paths"]["output_dir"]),
                    )
                    if fused_result.segments:
                        diar_segments = list(fused_result.segments)
                        preferred_overlap_regions = list(fused_result.overlap_seed_regions or [])
                        preferred_overlap_speaker_tracks = list(fused_result.overlap_tracks or [])
                        diarization_route = str(fused_result.route or "")
                    else:
                        diar_segments, diarization_route = select_hybrid_candidate(
                            hybrid_candidates,
                            audio_duration=audio_duration,
                            max_speakers=max_speakers,
                            requested_num_speakers=fixed_num,
                        )
                        if diarization_route:
                            diarization_route = f"Hybrid[{diarization_route}]"
            elif final_strategy in {"msdd_primary", "msdd_only"} and msdd_available:
                logger.info(
                    "  NeMo final strategy is %s; invoking MSDD first (%s)%s",
                    final_strategy,
                    msdd_model,
                    f" ({file_name})" if file_name else "",
                )
                diar_segments = self._diarize_audio(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=nemo_cfg,
                    file_name=file_name,
                )
                if diar_segments:
                    diarization_route = "NeMo-MSDD"
                elif final_strategy == "msdd_primary" and sortformer_enabled:
                    logger.info(
                        "  NeMo MSDD produced no usable diarization; falling back to sortformer (%s)%s",
                        sf_model,
                        f" ({file_name})" if file_name else "",
                    )
                    diar_segments = self._diarize_audio_nemo_sortformer(
                        audio_np=audio_np,
                        sample_rate=sample_rate,
                        segments=segments,
                        cfg=nemo_cfg,
                        file_name=file_name,
                    )
                    if diar_segments:
                        diarization_route = "NeMo-Sortformer(fallback)"
            elif sortformer_enabled:
                diar_segments = self._diarize_audio_nemo_sortformer(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    segments=segments,
                    cfg=nemo_cfg,
                    file_name=file_name,
                )
                if diar_segments:
                    diarization_route = "NeMo-Sortformer"
                if diar_segments and msdd_available:
                    logger.info(
                        "  NeMo MSDD is available but remained on standby because sortformer already produced diarization%s.",
                        f" ({file_name})" if file_name else "",
                    )
        if (
            map_speakers
            and self.use_nemo_msdd_pipeline()
            and msdd_available
            and not diar_segments
        ):
            logger.info(
                "  NeMo sortformer produced no usable diarization; invoking MSDD pipeline (%s)%s",
                msdd_model,
                f" ({file_name})" if file_name else "",
            )
            diar_segments = self._diarize_audio(
                audio_np=audio_np,
                sample_rate=sample_rate,
                segments=segments,
                cfg=nemo_cfg,
                file_name=file_name,
            )
            if diar_segments:
                diarization_route = "NeMo"
        if map_speakers and not diar_segments and not hybrid_pyannote_attempted:
            diar_segments = self._diarize_audio_pyannote_fallback(
                audio_np=audio_np,
                sample_rate=sample_rate,
                segments=segments,
                cfg=nemo_cfg,
                file_name=file_name,
            )
            if diar_segments:
                diarization_route = "pyannote.audio"
        if map_speakers and not diar_segments:
            if self._assign_speakers_torchaudio_fallback(
                audio_np=audio_np,
                sample_rate=sample_rate,
                segments=segments,
                cfg=nemo_cfg,
                file_name=file_name,
            ):
                diarization_route = "torchaudio"
            else:
                diarization_route = "default-single-speaker"

        if diar_segments:
            segments = self._split_whisper_segments_by_diarization_turns(
                segments=segments,
                diar_segments=diar_segments,
                overlap_tracks=preferred_overlap_speaker_tracks,
                audio_np=audio_np,
                sample_rate=sample_rate,
                file_name=file_name,
            )
            for seg in segments:
                raw_speaker = self._find_speaker_for_span(
                    diar_segments,
                    float(seg.start),
                    float(seg.end),
                    preferred_tracks=preferred_overlap_speaker_tracks,
                )
                seg.speaker = self._normalize_speaker_id(raw_speaker)
        else:
            for seg in segments:
                if not seg.speaker:
                    seg.speaker = "0"

        overlap_cfg = self._nemo_overlap_cfg(nemo_cfg)
        overlap_enabled = (
            map_speakers
            and bool(diar_segments)
            and self._overlap_redecode_enabled(nemo_cfg)
        )
        if overlap_enabled:
            overlap_regions = self._detect_overlap_regions(
                audio_np=audio_np,
                sample_rate=sample_rate,
                diar_segments=diar_segments,
                cfg=nemo_cfg,
                file_name=file_name,
                seed_regions=preferred_overlap_regions,
            )
            if overlap_regions:
                overlap_segments = self._transcribe_overlap_regions(
                    audio_np=audio_np,
                    sample_rate=sample_rate,
                    base_segments=segments,
                    diar_segments=diar_segments,
                    overlap_regions=overlap_regions,
                    cfg=nemo_cfg,
                    file_name=file_name,
                    overlap_speaker_hints=preferred_overlap_speaker_tracks,
                )
                if overlap_segments:
                    segments = self._merge_overlap_redecoded_segments(
                        base_segments=segments,
                        overlap_segments=overlap_segments,
                        overlap_regions=overlap_regions,
                        cfg=nemo_cfg,
                    )
                else:
                    logger.info("  Overlap regions detected but no valid re-recognized segments were produced.")
        elif (
            map_speakers
            and bool(diar_segments)
            and self._safe_bool(overlap_cfg.get("enabled", False), False)
        ):
            logger.info(
                "  Overlap re-recognition is disabled because separation and mixed-audio fallback are both off."
            )

        for seg in segments:
            if not seg.speaker:
                seg.speaker = "0"
            else:
                seg.speaker = self._normalize_speaker_id(seg.speaker)

        segments.sort(key=lambda s: (float(s.start), float(s.end)))

        detected_speakers: List[str] = []
        seen_detected: set[str] = set()
        for collection in (diar_segments, preferred_overlap_speaker_tracks):
            for item in collection or []:
                speaker = self._normalize_speaker_id(str(item.get("speaker", "") or ""))
                if not speaker or speaker in seen_detected:
                    continue
                seen_detected.add(speaker)
                detected_speakers.append(speaker)
        for seg in segments:
            speaker = self._normalize_speaker_id(str(getattr(seg, "speaker", "") or ""))
            if not speaker or speaker in seen_detected:
                continue
            seen_detected.add(speaker)
            detected_speakers.append(speaker)

        if not map_speakers:
            diarization_route = "disabled(map_speakers=False)"
        speaker_count = max(
            len(detected_speakers),
            len({str(seg.speaker) for seg in segments if str(seg.speaker).strip()}),
        )
        logger.info(
            "  Speaker diarization route hit: %s, speakers=%d%s",
            diarization_route,
            speaker_count,
            f" ({file_name})" if file_name else "",
        )
        self.last_diarization_route = str(diarization_route or "")
        self.last_detected_speaker_ids = list(detected_speakers)
        self.last_detected_speaker_count = int(speaker_count)

        if map_speakers:
            segments = self._map_speaker_labels(segments)
            segments = self._cleanup_adjacent_speaker_fragments(
                segments,
                file_name=file_name,
            )

        speaker_languages: Dict[str, str] = {}
        for seg in segments:
            if seg.speaker and seg.language and seg.speaker not in speaker_languages:
                speaker_languages[seg.speaker] = seg.language
        self.last_speaker_languages = speaker_languages
        return segments

    # Speaker label remapping

    @staticmethod
    def _speaker_duration_stats(
        segments: List[TranscriptionSegment],
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for idx, seg in enumerate(segments):
            speaker = str(getattr(seg, "speaker", "") or "").strip()
            if not speaker:
                continue
            start = max(0.0, float(getattr(seg, "start", 0.0) or 0.0))
            end = max(start, float(getattr(seg, "end", start) or start))
            info = stats.setdefault(
                speaker,
                {
                    "duration": 0.0,
                    "count": 0.0,
                    "first_start": start,
                    "first_index": float(idx),
                },
            )
            info["duration"] += max(0.0, end - start)
            info["count"] += 1.0
            if start < float(info.get("first_start", start)):
                info["first_start"] = start
                info["first_index"] = float(idx)
        return stats

    @staticmethod
    def _clone_segments(
        segments: List[TranscriptionSegment],
    ) -> List[TranscriptionSegment]:
        cloned: List[TranscriptionSegment] = []
        for seg in segments:
            words = [
                dict(item)
                for item in list(getattr(seg, "words", []) or [])
                if isinstance(item, dict)
            ]
            cloned.append(
                TranscriptionSegment(
                    start=float(getattr(seg, "start", 0.0) or 0.0),
                    end=float(getattr(seg, "end", 0.0) or 0.0),
                    text=str(getattr(seg, "text", "") or ""),
                    speaker=str(getattr(seg, "speaker", "") or ""),
                    language=str(getattr(seg, "language", "") or ""),
                    confidence=float(getattr(seg, "confidence", 0.0) or 0.0),
                    words=words or None,
                )
            )
        return cloned

    @staticmethod
    def _normalize_embedding_vector(values: Any) -> Optional[Any]:
        if torch.is_tensor(values):
            arr = values.detach().reshape(-1).to(dtype=torch.float32)
            if int(arr.numel()) <= 0:
                return None
            if not bool(torch.all(torch.isfinite(arr)).detach().to("cpu").item()):
                return None
            norm = torch.linalg.vector_norm(arr)
            if not bool(torch.isfinite(norm).detach().to("cpu").item()):
                return None
            if float(norm.detach().to("cpu").item()) <= 1e-6:
                return None
            return arr / norm.clamp_min(1e-6)

        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size <= 0 or not np.all(np.isfinite(arr)):
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-6:
            return None
        return np.ascontiguousarray(arr / norm, dtype=np.float32)

    @staticmethod
    def _embedding_to_numpy(values: Any) -> Optional[np.ndarray]:
        normalized = Transcriber._normalize_embedding_vector(values)
        if normalized is None:
            return None
        if torch.is_tensor(normalized):
            try:
                return np.ascontiguousarray(
                    normalized.detach().to(device="cpu", dtype=torch.float32).numpy().reshape(-1),
                    dtype=np.float32,
                )
            except Exception:
                return None
        arr = np.asarray(normalized, dtype=np.float32).reshape(-1)
        if arr.size <= 0 or not np.all(np.isfinite(arr)):
            return None
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(
        left: Any,
        right: Any,
    ) -> float:
        if torch.is_tensor(left) or torch.is_tensor(right):
            try:
                target_device = left.device if torch.is_tensor(left) else right.device
                a = (
                    left.detach().reshape(-1).to(device=target_device, dtype=torch.float32)
                    if torch.is_tensor(left)
                    else torch.from_numpy(
                        np.ascontiguousarray(left, dtype=np.float32).reshape(-1)
                    ).to(device=target_device, dtype=torch.float32)
                )
                b = (
                    right.detach().reshape(-1).to(device=target_device, dtype=torch.float32)
                    if torch.is_tensor(right)
                    else torch.from_numpy(
                        np.ascontiguousarray(right, dtype=np.float32).reshape(-1)
                    ).to(device=target_device, dtype=torch.float32)
                )
                if int(a.numel()) <= 0 or int(a.numel()) != int(b.numel()):
                    return float("nan")
                if (
                    not bool(torch.all(torch.isfinite(a)).detach().to("cpu").item())
                    or not bool(torch.all(torch.isfinite(b)).detach().to("cpu").item())
                ):
                    return float("nan")
                denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
                if not bool(torch.isfinite(denom).detach().to("cpu").item()):
                    return float("nan")
                denom_value = float(denom.detach().to("cpu").item())
                if denom_value <= 1e-6:
                    return float("nan")
                sim = torch.clamp(torch.dot(a, b) / denom, -1.0, 1.0)
                return float(sim.detach().to("cpu").item())
            except Exception:
                pass

        a = Transcriber._normalize_embedding_vector(left)
        b = Transcriber._normalize_embedding_vector(right)
        if a is None or b is None:
            return float("nan")
        if torch.is_tensor(a) or torch.is_tensor(b):
            try:
                target_device = a.device if torch.is_tensor(a) else b.device
                a_tensor = (
                    a.detach().reshape(-1).to(device=target_device, dtype=torch.float32)
                    if torch.is_tensor(a)
                    else torch.from_numpy(
                        np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
                    ).to(device=target_device, dtype=torch.float32)
                )
                b_tensor = (
                    b.detach().reshape(-1).to(device=target_device, dtype=torch.float32)
                    if torch.is_tensor(b)
                    else torch.from_numpy(
                        np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
                    ).to(device=target_device, dtype=torch.float32)
                )
                sim = torch.clamp(torch.dot(a_tensor, b_tensor), -1.0, 1.0)
                return float(sim.detach().to("cpu").item())
            except Exception:
                return float("nan")
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    @staticmethod
    def _best_matching_speaker_from_centroids(
        source_speaker: str,
        candidate_speakers: List[str],
        speaker_centroids: Dict[str, np.ndarray],
    ) -> Tuple[str, float]:
        source_vec = speaker_centroids.get(str(source_speaker or "").strip())
        if source_vec is None:
            return "", float("nan")

        best_speaker = ""
        best_similarity = float("nan")
        for candidate in candidate_speakers:
            candidate_key = str(candidate or "").strip()
            if not candidate_key or candidate_key == str(source_speaker or "").strip():
                continue
            similarity = Transcriber._cosine_similarity(
                source_vec,
                speaker_centroids.get(candidate_key),
            )
            if not math.isfinite(similarity):
                continue
            if not best_speaker or similarity > best_similarity:
                best_speaker = candidate_key
                best_similarity = similarity
        return best_speaker, best_similarity

    @staticmethod
    def _weighted_embedding_centroid(
        items: List[Tuple[Optional[Any], float]],
    ) -> Optional[Any]:
        normalized_items: List[Tuple[Any, float]] = []
        target_device = None
        for vector, weight in items:
            normalized = Transcriber._normalize_embedding_vector(vector)
            if normalized is None:
                continue
            if target_device is None and torch.is_tensor(normalized):
                target_device = normalized.device
            normalized_items.append((normalized, max(1e-3, float(weight or 0.0))))

        if not normalized_items:
            return None

        if target_device is not None:
            vectors_t: List[torch.Tensor] = []
            weights_t: List[float] = []
            for vector, weight in normalized_items:
                tensor = (
                    vector.detach().reshape(-1).to(device=target_device, dtype=torch.float32)
                    if torch.is_tensor(vector)
                    else torch.from_numpy(
                        np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
                    ).to(device=target_device, dtype=torch.float32)
                )
                vectors_t.append(tensor)
                weights_t.append(weight)
            if len(vectors_t) == 1:
                return vectors_t[0]
            matrix = torch.stack(vectors_t, dim=0)
            weights_tensor = torch.tensor(
                weights_t,
                device=target_device,
                dtype=torch.float32,
            )
            centroid = torch.sum(matrix * weights_tensor.unsqueeze(1), dim=0)
            centroid = centroid / weights_tensor.sum().clamp_min(1e-6)
            return Transcriber._normalize_embedding_vector(centroid)

        vectors_np: List[np.ndarray] = []
        weights_np: List[float] = []
        for vector, weight in normalized_items:
            vectors_np.append(np.ascontiguousarray(vector, dtype=np.float32).reshape(-1))
            weights_np.append(weight)
        if len(vectors_np) == 1:
            return vectors_np[0]

        matrix = np.stack(vectors_np, axis=0)
        centroid = np.average(
            matrix,
            axis=0,
            weights=np.asarray(weights_np, dtype=np.float32),
        )
        return Transcriber._normalize_embedding_vector(centroid)

    def _resolve_speaker_embedding_runtime_device(
        self,
        requested_device: str,
        *,
        target_sr: int,
        min_samples: int,
    ) -> str:
        device_name = str(requested_device or "cpu").strip().lower() or "cpu"
        if not self._is_accelerator_device(device_name):
            return device_name

        cached = self._speaker_embedding_accel_status.get(device_name)
        if cached is False:
            return "cpu"
        if cached is True:
            return device_name

        probe_samples = max(512, int(min_samples), int(target_sr))
        probe = np.zeros((probe_samples,), dtype=np.float32)
        _ = self._torch_speaker_embedding(
            probe,
            source_sr=int(target_sr),
            target_sr=int(target_sr),
            min_samples=max(256, int(min_samples)),
            device=device_name,
        )
        return device_name if self._speaker_embedding_accel_status.get(device_name, False) else "cpu"

    @staticmethod
    def _choose_embedding_views(
        audio: np.ndarray,
        sample_rate: int,
        start_sec: float,
        end_sec: float,
        *,
        min_sec: float = 0.8,
        max_sec: float = 6.0,
        view_sec: float = 1.8,
        max_views: int = 3,
        edge_pad_sec: float = 0.15,
    ) -> List[np.ndarray]:
        if audio is None or sample_rate <= 0:
            return []

        wav = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if wav.size <= 0:
            return []

        total_samples = int(wav.size)
        min_sec = max(0.4, float(min_sec))
        max_sec = max(min_sec, float(max_sec))
        view_sec = max(min_sec, min(float(view_sec), max_sec))

        start_sec = max(0.0, float(start_sec))
        end_sec = max(start_sec + 1e-3, float(end_sec))
        local_start = max(0.0, start_sec - edge_pad_sec)
        local_end = min(float(total_samples) / float(sample_rate), end_sec + edge_pad_sec)
        local_s = max(0, min(total_samples, int(round(local_start * sample_rate))))
        local_e = max(local_s + 1, min(total_samples, int(round(local_end * sample_rate))))
        local_clip = np.ascontiguousarray(wav[local_s:local_e], dtype=np.float32)
        if local_clip.size <= 0:
            return []

        target_min_samples = max(256, int(round(min_sec * sample_rate)))
        target_view_samples = max(target_min_samples, int(round(view_sec * sample_rate)))
        target_max_samples = max(target_view_samples, int(round(max_sec * sample_rate)))

        if local_clip.size <= target_max_samples:
            clip = local_clip
            if clip.size < target_min_samples:
                clip = np.pad(clip, (0, int(target_min_samples - clip.size)), mode="constant")
            return [np.ascontiguousarray(clip, dtype=np.float32)]

        def _energy_score(samples: np.ndarray) -> float:
            if samples.size <= 0:
                return 0.0
            clip = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
            rms = float(np.sqrt(np.maximum(1e-9, np.mean(clip * clip))))
            if clip.size < 128:
                return rms

            frame = min(clip.size, 512)
            window = np.hanning(frame).astype(np.float32, copy=False)
            frag = (
                clip[:frame]
                if clip.size == frame
                else clip[(clip.size - frame) // 2 : (clip.size + frame) // 2]
            )
            spec = np.abs(np.fft.rfft(frag * window))
            spec = np.maximum(spec, 1e-8)
            flatness = float(
                np.exp(np.mean(np.log(spec))) / np.maximum(np.mean(spec), 1e-8)
            )
            return rms * max(0.05, 1.05 - flatness)

        candidates: List[Tuple[float, int, np.ndarray]] = []
        step = max(1, target_view_samples // 2)
        center_start = max(
            0,
            min(
                max(0, local_clip.size - target_view_samples),
                (local_clip.size - target_view_samples) // 2,
            ),
        )
        center_clip = local_clip[center_start : center_start + target_view_samples]
        candidates.append((_energy_score(center_clip) + 0.03, center_start, center_clip))

        upper = max(1, local_clip.size - target_view_samples + 1)
        for offset in range(0, upper, step):
            frag = local_clip[offset : offset + target_view_samples]
            score = _energy_score(frag)
            if score <= 0.0:
                continue
            candidates.append((score, offset, frag))

        chosen: List[Tuple[int, np.ndarray]] = []
        for _score, offset, frag in sorted(
            candidates,
            key=lambda item: (-item[0], abs(item[1] - center_start)),
        ):
            if any(
                abs(offset - existing_offset) < max(1, target_view_samples // 2)
                for existing_offset, _existing in chosen
            ):
                continue
            chosen.append((offset, np.ascontiguousarray(frag, dtype=np.float32)))
            if len(chosen) >= max(1, int(max_views)):
                break

        if not chosen:
            chosen = [(center_start, np.ascontiguousarray(center_clip, dtype=np.float32))]

        return [frag for _offset, frag in sorted(chosen, key=lambda item: item[0])]

    def _speaker_view_payloads(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        start_sec: float,
        end_sec: float,
        *,
        min_view_sec: float,
        max_view_sec: float,
        per_view_sec: float,
        max_views: int,
    ) -> List[Tuple[np.ndarray, float]]:
        views = self._choose_embedding_views(
            audio_np,
            sample_rate,
            start_sec,
            end_sec,
            min_sec=min_view_sec,
            max_sec=max_view_sec,
            view_sec=per_view_sec,
            max_views=max_views,
        )
        payloads: List[Tuple[np.ndarray, float]] = []
        for clip in views:
            clip_np = np.ascontiguousarray(clip, dtype=np.float32).reshape(-1)
            if clip_np.size <= 0:
                continue
            payloads.append(
                (
                    clip_np,
                    min(
                        6.0,
                        max(0.25, float(clip_np.size) / float(max(1, sample_rate))),
                    ),
                )
            )
        return payloads

    def _numpy_speaker_embedding(
        self,
        clip: np.ndarray,
        *,
        source_sr: int,
        target_sr: int,
        min_samples: int,
    ) -> Optional[np.ndarray]:
        wav = _resample_audio_linear_np(
            clip,
            source_sr=int(source_sr),
            target_sr=int(target_sr),
        )
        if wav.size <= 0:
            return None
        if wav.size < min_samples:
            wav = np.pad(wav, (0, int(min_samples - wav.size)), mode="constant")
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        frame = max(128, int(target_sr * 0.025))
        hop = max(64, int(target_sr * 0.010))
        if wav.size < frame:
            wav = np.pad(wav, (0, int(frame - wav.size)), mode="constant")
        starts = list(range(0, max(1, wav.size - frame + 1), hop))
        if not starts:
            starts = [0]
        window = np.hanning(frame).astype(np.float32, copy=False)
        frame_feats: List[List[float]] = []
        nyq = max(1.0, float(target_sr) * 0.5)
        for start in starts:
            frag = wav[start:start + frame]
            if frag.size < frame:
                frag = np.pad(frag, (0, int(frame - frag.size)), mode="constant")
            spec = np.abs(np.fft.rfft(frag * window))
            spec_sum = float(np.sum(spec))
            rms = float(np.sqrt(np.maximum(1e-9, np.mean(frag * frag))))
            zcr = float(np.mean(np.sign(frag[1:]) != np.sign(frag[:-1]))) if frag.size > 1 else 0.0
            if spec_sum <= 1e-9:
                centroid_hz = 0.0
                rolloff_hz = 0.0
                flatness = 0.0
            else:
                bins = np.arange(spec.shape[0], dtype=np.float32)
                centroid_bin = float(np.sum(bins * spec) / spec_sum)
                centroid_hz = centroid_bin * nyq / max(1.0, float(spec.shape[0] - 1))
                csum = np.cumsum(spec)
                ridx = int(np.searchsorted(csum, 0.85 * csum[-1], side="left"))
                rolloff_hz = float(ridx) * nyq / max(1.0, float(spec.shape[0] - 1))
                geom = float(np.exp(np.mean(np.log(spec + 1e-8))))
                arith = float(np.mean(spec))
                flatness = geom / max(arith, 1e-8)
            frame_feats.append([
                rms,
                zcr,
                centroid_hz / nyq,
                rolloff_hz / nyq,
                flatness,
            ])
        ff = np.asarray(frame_feats, dtype=np.float32)
        if ff.ndim != 2 or ff.shape[0] <= 0:
            return None
        emb = np.concatenate(
            [
                ff.mean(axis=0),
                ff.std(axis=0),
                np.array([float(wav.size) / float(max(1, target_sr))], dtype=np.float32),
            ],
            axis=0,
        )
        return self._normalize_embedding_vector(emb)

    def _torch_speaker_embedding_batch(
        self,
        clips: List[np.ndarray],
        *,
        source_sr: int,
        target_sr: int,
        min_samples: int,
        device: str,
        batch_size: int,
    ) -> List[Optional[np.ndarray]]:
        clip_list = [
            np.ascontiguousarray(item, dtype=np.float32).reshape(-1)
            if item is not None
            else np.zeros((0,), dtype=np.float32)
            for item in list(clips or [])
        ]
        if not clip_list:
            return []

        runtime_device = str(device or "").strip().lower()

        def _numpy_fallback() -> List[Optional[np.ndarray]]:
            results: List[Optional[np.ndarray]] = []
            for clip in clip_list:
                results.append(
                    self._embedding_to_numpy(
                        self._numpy_speaker_embedding(
                            clip,
                            source_sr=source_sr,
                            target_sr=target_sr,
                            min_samples=min_samples,
                        )
                    )
                )
            return results

        if not self._is_accelerator_device(runtime_device):
            return _numpy_fallback()
        if self._speaker_embedding_accel_status.get(runtime_device) is False:
            return _numpy_fallback()

        target_min_samples = max(1, int(min_samples))
        preprocessed: List[Optional[np.ndarray]] = []
        for clip in clip_list:
            wav_np = _resample_audio_linear_np(
                clip,
                source_sr=int(source_sr),
                target_sr=int(target_sr),
            )
            if wav_np.size <= 0:
                preprocessed.append(None)
                continue
            if wav_np.size < target_min_samples:
                wav_np = np.pad(
                    wav_np,
                    (0, int(target_min_samples - wav_np.size)),
                    mode="constant",
                )
            preprocessed.append(np.ascontiguousarray(wav_np, dtype=np.float32))

        results: List[Optional[np.ndarray]] = [None] * len(preprocessed)
        frame = max(128, int(target_sr * 0.025))
        hop = max(64, int(target_sr * 0.010))
        max_batch = max(1, int(batch_size))

        try:
            with torch.inference_mode():
                window = torch.hann_window(
                    frame,
                    periodic=False,
                    device=runtime_device,
                    dtype=torch.float32,
                )
                nyq = max(1.0, float(target_sr) * 0.5)
                for batch_start in range(0, len(preprocessed), max_batch):
                    batch_items = preprocessed[batch_start : batch_start + max_batch]
                    valid_positions = [
                        pos
                        for pos, clip in enumerate(batch_items)
                        if clip is not None and int(clip.size) > 0
                    ]
                    if not valid_positions:
                        continue

                    valid_wavs = [batch_items[pos] for pos in valid_positions]
                    lengths = [max(frame, int(item.size)) for item in valid_wavs]
                    max_len = max(lengths)
                    batch_np = np.zeros((len(valid_wavs), max_len), dtype=np.float32)
                    for local_idx, wav_np in enumerate(valid_wavs):
                        batch_np[local_idx, : wav_np.size] = wav_np

                    wav = torch.from_numpy(batch_np).to(
                        device=runtime_device,
                        dtype=torch.float32,
                    )
                    frames = wav.unfold(1, frame, hop)
                    if frames.ndim != 3 or int(frames.shape[1]) <= 0:
                        continue

                    frame_count = int(frames.shape[1])
                    valid_frame_counts = torch.tensor(
                        [
                            max(1, 1 + max(0, int(length) - frame) // hop)
                            for length in lengths
                        ],
                        device=runtime_device,
                        dtype=torch.int64,
                    )
                    frame_indices = torch.arange(
                        frame_count,
                        device=runtime_device,
                        dtype=torch.int64,
                    ).unsqueeze(0)
                    frame_mask = frame_indices < valid_frame_counts.unsqueeze(1)
                    mask_f = frame_mask.unsqueeze(-1).to(torch.float32)

                    windowed = frames * window.view(1, 1, -1)
                    spec = torch.abs(torch.fft.rfft(windowed, dim=-1))
                    spec_sum = torch.sum(spec, dim=-1)
                    spec_sum_safe = torch.clamp(spec_sum, min=1e-9)
                    rms = torch.sqrt(
                        torch.clamp(torch.mean(frames * frames, dim=-1), min=1e-9)
                    )
                    zcr = (
                        torch.ge(frames[:, :, 1:], 0.0)
                        != torch.ge(frames[:, :, :-1], 0.0)
                    ).to(torch.float32).mean(dim=-1)

                    bin_count = max(1.0, float(spec.shape[-1] - 1))
                    bins = torch.arange(
                        spec.shape[-1],
                        device=runtime_device,
                        dtype=torch.float32,
                    )
                    centroid_bin = torch.sum(spec * bins.view(1, 1, -1), dim=-1) / spec_sum_safe
                    centroid_hz = centroid_bin * nyq / bin_count

                    csum = torch.cumsum(spec, dim=-1)
                    rolloff_threshold = 0.85 * csum[:, :, -1:].contiguous()
                    rolloff_idx = torch.argmax(
                        (csum >= rolloff_threshold).to(torch.int32),
                        dim=-1,
                    ).to(torch.float32)
                    rolloff_hz = rolloff_idx * nyq / bin_count

                    geom = torch.exp(torch.mean(torch.log(spec + 1e-8), dim=-1))
                    arith = torch.mean(spec, dim=-1)
                    flatness = geom / torch.clamp(arith, min=1e-8)

                    valid_spec = spec_sum > 1e-9
                    centroid_hz = torch.where(
                        valid_spec,
                        centroid_hz,
                        torch.zeros_like(centroid_hz),
                    )
                    rolloff_hz = torch.where(
                        valid_spec,
                        rolloff_hz,
                        torch.zeros_like(rolloff_hz),
                    )
                    flatness = torch.where(
                        valid_spec,
                        flatness,
                        torch.zeros_like(flatness),
                    )

                    features = torch.stack(
                        [
                            rms,
                            zcr,
                            centroid_hz / nyq,
                            rolloff_hz / nyq,
                            flatness,
                        ],
                        dim=-1,
                    )
                    features = torch.where(mask_f > 0.0, features, torch.zeros_like(features))
                    counts = torch.clamp(mask_f.sum(dim=1), min=1.0)
                    mean = features.sum(dim=1) / counts
                    centered = (features - mean.unsqueeze(1)) * mask_f
                    var = torch.sum(centered * centered, dim=1) / counts
                    std = torch.sqrt(torch.clamp(var, min=0.0))
                    durations = torch.tensor(
                        [float(length) / float(max(1, target_sr)) for length in lengths],
                        device=runtime_device,
                        dtype=torch.float32,
                    ).unsqueeze(1)
                    emb = torch.cat([mean, std, durations], dim=1)

                    norms = torch.linalg.vector_norm(emb, dim=1, keepdim=True)
                    valid = (
                        torch.all(torch.isfinite(emb), dim=1)
                        & torch.isfinite(norms.squeeze(1))
                        & (norms.squeeze(1) > 1e-6)
                    )
                    emb = emb / norms.clamp_min(1e-6)

                    emb_cpu = emb.detach().to(device="cpu", dtype=torch.float32).numpy()
                    valid_cpu = valid.detach().to(device="cpu").tolist()
                    for local_idx, pos in enumerate(valid_positions):
                        if bool(valid_cpu[local_idx]):
                            results[batch_start + pos] = np.ascontiguousarray(
                                emb_cpu[local_idx].reshape(-1),
                                dtype=np.float32,
                            )

                if runtime_device == "mps":
                    _sync_mps()

            if any(item is not None for item in results):
                self._speaker_embedding_accel_status[runtime_device] = True
            return results
        except Exception as e:
            self._speaker_embedding_accel_status[runtime_device] = False
            logger.warning(
                "Accelerated batched speaker refinement embedding failed on %s; fallback to CPU numpy path: %s",
                runtime_device,
                e,
            )
            return _numpy_fallback()

    def _torch_speaker_embedding(
        self,
        clip: np.ndarray,
        *,
        source_sr: int,
        target_sr: int,
        min_samples: int,
        device: str,
    ) -> Optional[Any]:
        runtime_device = str(device or "").strip().lower()
        if not self._is_accelerator_device(runtime_device):
            return self._numpy_speaker_embedding(
                clip,
                source_sr=source_sr,
                target_sr=target_sr,
                min_samples=min_samples,
            )
        if self._speaker_embedding_accel_status.get(runtime_device) is False:
            return self._numpy_speaker_embedding(
                clip,
                source_sr=source_sr,
                target_sr=target_sr,
                min_samples=min_samples,
            )

        wav_np = _resample_audio_linear_np(
            clip,
            source_sr=int(source_sr),
            target_sr=int(target_sr),
        )
        if wav_np.size <= 0:
            return None

        try:
            with torch.no_grad():
                wav = torch.from_numpy(
                    np.ascontiguousarray(wav_np, dtype=np.float32).reshape(-1)
                ).to(device=runtime_device, dtype=torch.float32)
                target_min_samples = max(1, int(min_samples))
                if int(wav.numel()) < target_min_samples:
                    wav = torch.nn.functional.pad(
                        wav,
                        (0, target_min_samples - int(wav.numel())),
                    )

                frame = max(128, int(target_sr * 0.025))
                hop = max(64, int(target_sr * 0.010))
                if int(wav.numel()) < frame:
                    wav = torch.nn.functional.pad(
                        wav,
                        (0, frame - int(wav.numel())),
                    )

                frames = wav.unfold(0, frame, hop)
                if frames.ndim != 2 or int(frames.shape[0]) <= 0:
                    return None

                window = torch.hann_window(
                    frame,
                    periodic=False,
                    device=runtime_device,
                    dtype=torch.float32,
                )
                windowed = frames * window.unsqueeze(0)
                spec = torch.abs(torch.fft.rfft(windowed, dim=-1))
                spec_sum = torch.sum(spec, dim=-1)
                spec_sum_safe = torch.clamp(spec_sum, min=1e-9)
                rms = torch.sqrt(
                    torch.clamp(torch.mean(frames * frames, dim=-1), min=1e-9)
                )
                zcr = (
                    torch.ge(frames[:, 1:], 0.0) != torch.ge(frames[:, :-1], 0.0)
                ).to(torch.float32).mean(dim=-1)

                nyq = max(1.0, float(target_sr) * 0.5)
                bin_count = max(1.0, float(spec.shape[-1] - 1))
                bins = torch.arange(
                    spec.shape[-1],
                    device=runtime_device,
                    dtype=torch.float32,
                )
                centroid_bin = torch.sum(spec * bins.unsqueeze(0), dim=-1) / spec_sum_safe
                centroid_hz = centroid_bin * nyq / bin_count

                csum = torch.cumsum(spec, dim=-1)
                rolloff_threshold = 0.85 * csum[:, -1:].contiguous()
                rolloff_idx = torch.argmax(
                    (csum >= rolloff_threshold).to(torch.int32),
                    dim=-1,
                ).to(torch.float32)
                rolloff_hz = rolloff_idx * nyq / bin_count

                geom = torch.exp(torch.mean(torch.log(spec + 1e-8), dim=-1))
                arith = torch.mean(spec, dim=-1)
                flatness = geom / torch.clamp(arith, min=1e-8)

                valid_spec = spec_sum > 1e-9
                centroid_hz = torch.where(
                    valid_spec,
                    centroid_hz,
                    torch.zeros_like(centroid_hz),
                )
                rolloff_hz = torch.where(
                    valid_spec,
                    rolloff_hz,
                    torch.zeros_like(rolloff_hz),
                )
                flatness = torch.where(
                    valid_spec,
                    flatness,
                    torch.zeros_like(flatness),
                )

                features = torch.stack(
                    [
                        rms,
                        zcr,
                        centroid_hz / nyq,
                        rolloff_hz / nyq,
                        flatness,
                    ],
                    dim=-1,
                )
                emb = torch.cat(
                    [
                        torch.mean(features, dim=0),
                        torch.std(features, dim=0, unbiased=False),
                        torch.tensor(
                            [float(int(wav.numel())) / float(max(1, target_sr))],
                            device=runtime_device,
                            dtype=torch.float32,
                        ),
                    ],
                    dim=0,
                )
                if runtime_device == "mps":
                    _sync_mps()
                normalized = self._normalize_embedding_vector(
                    emb.detach().reshape(-1).to(dtype=torch.float32)
                )
            if normalized is not None:
                self._speaker_embedding_accel_status[runtime_device] = True
            return normalized
        except Exception as e:
            self._speaker_embedding_accel_status[runtime_device] = False
            logger.warning(
                "Accelerated speaker refinement embedding failed on %s; fallback to CPU numpy path: %s",
                runtime_device,
                e,
            )
            return self._numpy_speaker_embedding(
                clip,
                source_sr=source_sr,
                target_sr=target_sr,
                min_samples=min_samples,
            )

    def _specialized_longform_runtime_device(
        self,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> str:
        fb_cfg = cfg if isinstance(cfg, dict) else self._pyannote_diar_fallback_cfg()
        pref = fb_cfg.get("specialize_device", fb_cfg.get("device", "auto"))
        return self._preferred_torch_device(pref, allow_mps=True)

    def _speaker_span_embedding(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        start_sec: float,
        end_sec: float,
        *,
        target_sr: int,
        min_samples: int,
        min_view_sec: float = 0.8,
        max_view_sec: float = 6.0,
        per_view_sec: float = 1.8,
        max_views: int = 3,
        runtime_device: str = "cpu",
    ) -> Optional[Any]:
        embeddings = self._batched_speaker_span_embeddings(
            audio_np,
            sample_rate,
            [(start_sec, end_sec)],
            target_sr=target_sr,
            min_samples=min_samples,
            min_view_sec=min_view_sec,
            max_view_sec=max_view_sec,
            per_view_sec=per_view_sec,
            max_views=max_views,
            runtime_device=runtime_device,
            batch_size=1,
        )
        return embeddings[0] if embeddings else None

    def _batched_speaker_span_embeddings(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        spans: List[Tuple[float, float]],
        *,
        target_sr: int,
        min_samples: int,
        min_view_sec: float = 0.8,
        max_view_sec: float = 6.0,
        per_view_sec: float = 1.8,
        max_views: int = 3,
        runtime_device: str = "cpu",
        batch_size: int = 24,
    ) -> List[Optional[np.ndarray]]:
        if not spans:
            return []

        span_plans: List[List[Tuple[int, float]]] = []
        flat_views: List[np.ndarray] = []
        for start_sec, end_sec in spans:
            payloads = self._speaker_view_payloads(
                audio_np,
                sample_rate,
                start_sec,
                end_sec,
                min_view_sec=min_view_sec,
                max_view_sec=max_view_sec,
                per_view_sec=per_view_sec,
                max_views=max_views,
            )
            plan: List[Tuple[int, float]] = []
            for clip, weight in payloads:
                plan.append((len(flat_views), float(weight)))
                flat_views.append(clip)
            span_plans.append(plan)

        if not flat_views:
            return [None] * len(spans)

        device_name = str(runtime_device or "cpu").strip().lower()
        if self._is_accelerator_device(device_name):
            view_embeddings = self._torch_speaker_embedding_batch(
                flat_views,
                source_sr=int(sample_rate),
                target_sr=int(target_sr),
                min_samples=int(min_samples),
                device=device_name,
                batch_size=max(1, int(batch_size)),
            )
        else:
            view_embeddings = [
                self._embedding_to_numpy(
                    self._numpy_speaker_embedding(
                        clip,
                        source_sr=int(sample_rate),
                        target_sr=int(target_sr),
                        min_samples=int(min_samples),
                    )
                )
                for clip in flat_views
            ]

        results: List[Optional[np.ndarray]] = []
        for plan in span_plans:
            centroid = self._weighted_embedding_centroid(
                [
                    (view_embeddings[idx], weight)
                    for idx, weight in plan
                    if 0 <= idx < len(view_embeddings) and view_embeddings[idx] is not None
                ]
            )
            results.append(self._embedding_to_numpy(centroid))
        return results

    @staticmethod
    def _merge_adjacent_speaker_turns(
        diar_segments: List[Dict[str, Any]],
        merge_gap_sec: float,
    ) -> List[Dict[str, Any]]:
        if not diar_segments:
            return []

        merged: List[Dict[str, Any]] = []
        for item in sorted(
            diar_segments,
            key=lambda value: (
                float(value.get("start", 0.0)),
                float(value.get("end", 0.0)),
                str(value.get("speaker", "")),
            ),
        ):
            start = float(item.get("start", 0.0))
            end = max(start, float(item.get("end", start)))
            speaker = str(item.get("speaker", "0"))
            if (
                merged
                and str(merged[-1].get("speaker", "")) == speaker
                and start <= float(merged[-1].get("end", 0.0)) + max(0.0, merge_gap_sec)
            ):
                merged[-1]["end"] = max(float(merged[-1]["end"]), end)
                continue
            merged.append({"start": start, "end": end, "speaker": speaker})
        return merged

    def _refine_pyannote_diar_segments(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        diar_segments: List[Dict[str, Any]],
        cfg: Optional[Dict[str, Any]] = None,
        file_name: str = "",
    ) -> List[Dict[str, Any]]:
        fb_cfg = dict(self._pyannote_diar_fallback_cfg(cfg))
        if not self._safe_bool(fb_cfg.get("specialize_longform", True), True):
            return diar_segments
        if sample_rate <= 0 or audio_np is None or len(diar_segments) < 3:
            return diar_segments

        raw_speakers = sorted(
            {
                str(item.get("speaker", "") or "").strip()
                for item in diar_segments
                if str(item.get("speaker", "") or "").strip()
            }
        )
        raw_speaker_count = len(raw_speakers)
        if raw_speaker_count <= 2:
            return diar_segments

        audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32).reshape(-1)
        if audio_f32.size == 0:
            return diar_segments
        peak = float(np.max(np.abs(audio_f32))) if audio_f32.size else 0.0
        if peak > 1.0:
            audio_f32 = audio_f32 / max(peak, 1e-6)

        target_sr = 16000
        min_view_sec = max(0.4, self._safe_float(fb_cfg.get("specialize_min_view_sec", 0.8), 0.8))
        max_view_sec = max(min_view_sec, self._safe_float(fb_cfg.get("specialize_max_view_sec", 6.0), 6.0))
        per_view_sec = max(
            min_view_sec,
            min(max_view_sec, self._safe_float(fb_cfg.get("specialize_view_sec", 1.8), 1.8)),
        )
        max_views = max(1, self._safe_int(fb_cfg.get("specialize_max_views", 3), 3))
        short_turn_sec = max(0.8, self._safe_float(fb_cfg.get("specialize_short_turn_sec", 1.8), 1.8))
        strong_merge_similarity = max(
            self._safe_float(fb_cfg.get("specialize_merge_similarity", 0.91), 0.91),
            0.91,
        )
        short_merge_similarity = max(
            self._safe_float(fb_cfg.get("specialize_short_merge_similarity", 0.86), 0.86),
            0.86,
        )
        attach_similarity = max(
            self._safe_float(fb_cfg.get("specialize_attach_similarity", 0.84), 0.84),
            0.84,
        )
        merge_gap_sec = max(0.0, self._safe_float(fb_cfg.get("merge_gap_sec", 0.08), 0.08))
        min_samples = max(256, int(round(min_view_sec * target_sr)))
        requested_runtime_device = self._specialized_longform_runtime_device(fb_cfg)
        runtime_device = self._resolve_speaker_embedding_runtime_device(
            requested_runtime_device,
            target_sr=target_sr,
            min_samples=min_samples,
        )
        require_gpu = self._safe_bool(fb_cfg.get("require_gpu", False), False)
        if require_gpu and not self._is_accelerator_device(runtime_device):
            logger.warning(
                "  Specialized pyannote long-form refinement skipped: GPU required but unavailable (requested=%s, actual=%s)%s",
                requested_runtime_device,
                runtime_device,
                f" ({file_name})" if file_name else "",
            )
            return diar_segments
        if runtime_device != requested_runtime_device:
            logger.warning(
                "  Specialized pyannote long-form refinement downgraded: requested=%s, actual=%s%s",
                requested_runtime_device,
                runtime_device,
                f" ({file_name})" if file_name else "",
            )
        dense_turn_threshold = max(
            120,
            self._safe_int(fb_cfg.get("specialize_dense_turn_threshold", 240), 240),
        )
        ultra_dense_turn_threshold = max(
            dense_turn_threshold + 20,
            self._safe_int(
                fb_cfg.get("specialize_ultra_dense_turn_threshold", 420),
                420,
            ),
        )
        dense_max_views = max(
            1,
            min(
                max_views,
                self._safe_int(fb_cfg.get("specialize_dense_max_views", 2), 2),
            ),
        )
        ultra_dense_max_views = max(
            1,
            min(
                dense_max_views,
                self._safe_int(fb_cfg.get("specialize_ultra_dense_max_views", 1), 1),
            ),
        )
        dense_view_sec = max(
            min_view_sec,
            min(
                max_view_sec,
                self._safe_float(fb_cfg.get("specialize_dense_view_sec", 1.25), 1.25),
            ),
        )
        ultra_dense_view_sec = max(
            min_view_sec,
            min(
                dense_view_sec,
                self._safe_float(fb_cfg.get("specialize_ultra_dense_view_sec", 0.95), 0.95),
            ),
        )
        refinement_mode = "standard"
        turn_count = len(diar_segments)
        if turn_count >= ultra_dense_turn_threshold:
            max_views = min(max_views, ultra_dense_max_views)
            per_view_sec = min(per_view_sec, ultra_dense_view_sec)
            max_view_sec = min(max_view_sec, max(per_view_sec, ultra_dense_view_sec * 1.75))
            refinement_mode = "ultra_dense"
        elif turn_count >= dense_turn_threshold:
            max_views = min(max_views, dense_max_views)
            per_view_sec = min(per_view_sec, dense_view_sec)
            max_view_sec = min(max_view_sec, max(per_view_sec, dense_view_sec * 1.9))
            refinement_mode = "dense"
        batch_size = max(
            4,
            self._safe_int(
                fb_cfg.get(
                    "specialize_batch_size",
                    32 if runtime_device == "mps" else 24,
                ),
                32 if runtime_device == "mps" else 24,
            ),
        )
        logger.info(
            "  Specialized pyannote long-form refinement start: %d speaker(s), %d turn(s), duration=%.1fs, device=%s%s",
            raw_speaker_count,
            len(diar_segments),
            float(audio_f32.size) / float(max(1, sample_rate)),
            runtime_device,
            f" ({file_name})" if file_name else "",
        )
        if refinement_mode != "standard":
            logger.info(
                "  Specialized pyannote long-form refinement mode=%s: view_sec=%.2fs, max_view_sec=%.2fs, max_views=%d, batch_size=%d%s",
                refinement_mode,
                per_view_sec,
                max_view_sec,
                max_views,
                batch_size,
                f" ({file_name})" if file_name else "",
            )
        root_cfg = cfg if isinstance(cfg, dict) else self._nemo_msdd_cfg()
        fixed_num = max(
            0,
            self._safe_int(
                fb_cfg.get("num_speakers", root_cfg.get("num_speakers", 0)),
                self._safe_int(root_cfg.get("num_speakers", 0), 0),
            ),
        )

        span_specs: List[Tuple[int, float, float, str]] = []
        for idx, seg in enumerate(diar_segments):
            start = max(0.0, float(seg.get("start", 0.0)))
            end = max(start + 1e-3, float(seg.get("end", start)))
            speaker = str(seg.get("speaker", "0"))
            span_specs.append((idx, start, end, speaker))

        span_embeddings = self._batched_speaker_span_embeddings(
            audio_f32,
            sample_rate,
            [(start, end) for _idx, start, end, _speaker in span_specs],
            target_sr=target_sr,
            min_samples=min_samples,
            min_view_sec=min_view_sec,
            max_view_sec=max_view_sec,
            per_view_sec=per_view_sec,
            max_views=max_views,
            runtime_device=runtime_device,
            batch_size=batch_size,
        )

        runs: Dict[int, Dict[str, Any]] = {}
        for pos, (idx, start, end, speaker) in enumerate(span_specs):
            runs[idx] = {
                "id": idx,
                "segment_indices": [idx],
                "start": start,
                "end": end,
                "duration": end - start,
                "raw_speakers": {speaker},
                "embedding": span_embeddings[pos] if pos < len(span_embeddings) else None,
            }

        if sum(1 for item in runs.values() if item.get("embedding") is not None) < 2:
            return diar_segments

        def _cluster_from_run_ids(cluster_id: int, run_ids: List[int]) -> Dict[str, Any]:
            cluster_runs = [runs[run_id] for run_id in sorted(run_ids)]
            return {
                "id": cluster_id,
                "run_ids": [item["id"] for item in cluster_runs],
                "segment_indices": sorted(
                    seg_idx
                    for item in cluster_runs
                    for seg_idx in item["segment_indices"]
                ),
                "start": min(float(item["start"]) for item in cluster_runs),
                "end": max(float(item["end"]) for item in cluster_runs),
                "duration": sum(float(item["duration"]) for item in cluster_runs),
                "raw_speakers": set().union(*(item["raw_speakers"] for item in cluster_runs)),
                "embedding": self._weighted_embedding_centroid(
                    [
                        (
                            item.get("embedding"),
                            min(6.0, max(0.25, float(item["duration"]))),
                        )
                        for item in cluster_runs
                    ]
                ),
            }

        def _clusters_overlap(
            cluster_a: Dict[str, Any],
            cluster_b: Dict[str, Any],
            overlap_tol: float = 0.05,
        ) -> bool:
            indices_a = sorted(cluster_a["segment_indices"])
            indices_b = sorted(cluster_b["segment_indices"])
            ia = 0
            ib = 0
            while ia < len(indices_a) and ib < len(indices_b):
                seg_a = diar_segments[indices_a[ia]]
                seg_b = diar_segments[indices_b[ib]]
                a_start = float(seg_a.get("start", 0.0))
                a_end = float(seg_a.get("end", a_start))
                b_start = float(seg_b.get("start", 0.0))
                b_end = float(seg_b.get("end", b_start))
                overlap = min(a_end, b_end) - max(a_start, b_start)
                if overlap > overlap_tol:
                    return True
                if a_end <= b_end:
                    ia += 1
                else:
                    ib += 1
            return False

        def _cluster_neighbors(
            cluster_id: int,
            active: Dict[int, Dict[str, Any]],
        ) -> Tuple[Optional[int], Optional[int]]:
            segment_to_cluster: Dict[int, int] = {}
            for cid, item in active.items():
                for seg_index in item["segment_indices"]:
                    segment_to_cluster[seg_index] = cid

            indices = sorted(active[cluster_id]["segment_indices"])
            if not indices:
                return None, None

            left_cluster = None
            right_cluster = None
            for probe in range(indices[0] - 1, -1, -1):
                candidate = segment_to_cluster.get(probe)
                if candidate is not None and candidate != cluster_id:
                    left_cluster = candidate
                    break
            for probe in range(indices[-1] + 1, len(diar_segments)):
                candidate = segment_to_cluster.get(probe)
                if candidate is not None and candidate != cluster_id:
                    right_cluster = candidate
                    break
            return left_cluster, right_cluster

        active: Dict[int, Dict[str, Any]] = {
            run_id: _cluster_from_run_ids(run_id, [run_id])
            for run_id in sorted(runs)
        }
        next_cluster_id = max(active) + 1 if active else 0

        while True:
            best_choice: Optional[Tuple[float, float, int, int]] = None
            active_ids = sorted(active)
            for idx, cluster_id_a in enumerate(active_ids):
                cluster_a = active[cluster_id_a]
                for cluster_id_b in active_ids[idx + 1 :]:
                    cluster_b = active[cluster_id_b]
                    if _clusters_overlap(cluster_a, cluster_b):
                        continue
                    similarity = self._cosine_similarity(
                        cluster_a.get("embedding"),
                        cluster_b.get("embedding"),
                    )
                    if not math.isfinite(similarity):
                        continue
                    shared_raw = bool(cluster_a["raw_speakers"] & cluster_b["raw_speakers"])
                    threshold = strong_merge_similarity
                    if cluster_a["duration"] <= short_turn_sec or cluster_b["duration"] <= short_turn_sec:
                        threshold = min(threshold, short_merge_similarity)
                    if shared_raw:
                        threshold -= 0.04
                    else:
                        threshold += 0.02
                    if (
                        cluster_a["duration"] >= 8.0
                        and cluster_b["duration"] >= 8.0
                        and not shared_raw
                    ):
                        threshold = max(threshold, strong_merge_similarity + 0.02)
                    if similarity < threshold:
                        continue
                    margin = similarity - threshold
                    candidate = (margin, similarity, cluster_id_a, cluster_id_b)
                    if best_choice is None or candidate[:2] > best_choice[:2]:
                        best_choice = candidate

            if best_choice is None:
                break

            _margin, _score, cluster_id_a, cluster_id_b = best_choice
            cluster_a = active.pop(cluster_id_a)
            cluster_b = active.pop(cluster_id_b)
            active[next_cluster_id] = _cluster_from_run_ids(
                next_cluster_id,
                list(cluster_a["run_ids"]) + list(cluster_b["run_ids"]),
            )
            next_cluster_id += 1

        candidate_ids = [
            cluster_id
            for cluster_id, cluster in sorted(
                active.items(),
                key=lambda item: (float(item[1]["duration"]), float(item[1]["start"])),
            )
            if float(cluster["duration"]) <= short_turn_sec
            or (
                len(cluster["run_ids"]) == 1
                and float(cluster["duration"]) <= max(3.0, short_turn_sec * 1.6)
            )
        ]
        for cluster_id in candidate_ids:
            cluster = active.get(cluster_id)
            if cluster is None:
                continue

            left_context, right_context = _cluster_neighbors(cluster_id, active)
            best_target = None
            best_score = float("-inf")
            for target_id, target_cluster in active.items():
                if target_id == cluster_id or _clusters_overlap(cluster, target_cluster):
                    continue
                similarity = self._cosine_similarity(
                    cluster.get("embedding"),
                    target_cluster.get("embedding"),
                )
                if not math.isfinite(similarity):
                    continue
                threshold = attach_similarity
                if cluster["raw_speakers"] & target_cluster["raw_speakers"]:
                    threshold -= 0.04
                else:
                    threshold += 0.02
                bonus = 0.0
                if left_context == target_id:
                    bonus += 0.04
                if right_context == target_id:
                    bonus += 0.04
                total_score = similarity + bonus
                if total_score < threshold or total_score <= best_score:
                    continue
                best_target = target_id
                best_score = total_score

            if best_target is None:
                continue

            target_cluster = active.get(best_target)
            if target_cluster is None:
                continue

            active.pop(cluster_id, None)
            active.pop(best_target, None)
            active[next_cluster_id] = _cluster_from_run_ids(
                next_cluster_id,
                list(cluster["run_ids"]) + list(target_cluster["run_ids"]),
            )
            next_cluster_id += 1

        refined_count = len(active)
        if fixed_num > 0:
            if refined_count != fixed_num:
                return diar_segments
        else:
            if refined_count >= raw_speaker_count:
                return diar_segments
            if raw_speaker_count > 2 and refined_count < 2:
                return diar_segments

        ordered_clusters = sorted(
            active.values(),
            key=lambda item: (
                -float(item["duration"]),
                float(item["start"]),
                sorted(item["raw_speakers"]),
            ),
        )
        cluster_labels: Dict[int, str] = {
            int(item["id"]): str(idx)
            for idx, item in enumerate(ordered_clusters)
        }
        segment_to_cluster: Dict[int, int] = {}
        for cluster in active.values():
            for seg_index in cluster["segment_indices"]:
                segment_to_cluster[seg_index] = int(cluster["id"])

        refined_segments: List[Dict[str, Any]] = []
        for seg_index, seg in enumerate(diar_segments):
            cluster_id = segment_to_cluster.get(seg_index)
            speaker = cluster_labels.get(cluster_id, str(seg.get("speaker", "0")))
            refined_segments.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "speaker": speaker,
                }
            )

        refined_segments = self._merge_adjacent_speaker_turns(
            refined_segments,
            merge_gap_sec=merge_gap_sec,
        )
        logger.info(
            "  Specialized pyannote long-form refinement: %d -> %d speaker(s), %d -> %d turn(s)%s",
            raw_speaker_count,
            len({str(item['speaker']) for item in refined_segments}),
            len(diar_segments),
            len(refined_segments),
            f" ({file_name})" if file_name else "",
        )
        return refined_segments

    def _speaker_centroid_embeddings(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        segments: List[TranscriptionSegment],
        cfg: Optional[Dict[str, Any]] = None,
        *,
        max_segments_per_speaker: int = 10,
    ) -> Dict[str, np.ndarray]:
        if sample_rate <= 0 or audio_np is None or not segments:
            return {}

        fb_cfg = dict(self._torchaudio_fallback_cfg(cfg))
        target_sr = max(8000, self._safe_int(fb_cfg.get("sample_rate", 16000), 16000))
        min_seg_sec = max(0.20, self._safe_float(fb_cfg.get("min_segment_sec", 0.60), 0.60))
        max_seg_sec = max(min_seg_sec, self._safe_float(fb_cfg.get("max_segment_sec", 12.0), 12.0))
        min_samples = max(256, int(min_seg_sec * target_sr))

        audio_f32 = np.ascontiguousarray(audio_np, dtype=np.float32).reshape(-1)
        if audio_f32.size == 0:
            return {}
        peak = float(np.max(np.abs(audio_f32))) if audio_f32.size else 0.0
        if peak > 1.0:
            audio_f32 = audio_f32 / max(peak, 1e-6)

        windows_by_speaker: Dict[str, List[Tuple[float, int, float, float]]] = {}
        for idx, seg in enumerate(segments):
            speaker = str(getattr(seg, "speaker", "") or "").strip()
            if not speaker:
                continue
            start = max(0.0, float(getattr(seg, "start", 0.0) or 0.0))
            end = max(start + 1e-3, float(getattr(seg, "end", start) or start))
            duration = end - start
            if duration < min_seg_sec * 0.5:
                continue
            if duration > max_seg_sec:
                mid = (start + end) * 0.5
                half = max_seg_sec * 0.5
                start = max(0.0, mid - half)
                end = max(start + 1e-3, mid + half)
                duration = end - start
            windows_by_speaker.setdefault(speaker, []).append((duration, idx, start, end))

        speaker_centroids: Dict[str, np.ndarray] = {}
        for speaker, windows in windows_by_speaker.items():
            speaker_embeddings: List[np.ndarray] = []
            for _dur, _idx, start, end in sorted(
                windows,
                key=lambda item: (-item[0], item[1]),
            )[: max(1, int(max_segments_per_speaker))]:
                s_idx = max(0, min(audio_f32.size, int(start * sample_rate)))
                e_idx = max(s_idx + 1, min(audio_f32.size, int(end * sample_rate)))
                clip = audio_f32[s_idx:e_idx]
                if clip.size < 32:
                    continue
                emb = self._numpy_speaker_embedding(
                    clip,
                    source_sr=int(sample_rate),
                    target_sr=int(target_sr),
                    min_samples=int(min_samples),
                )
                if emb is not None:
                    speaker_embeddings.append(emb)
            if not speaker_embeddings:
                continue
            centroid = self._normalize_embedding_vector(
                np.mean(np.stack(speaker_embeddings, axis=0), axis=0)
            )
            if centroid is not None:
                speaker_centroids[speaker] = centroid
        return speaker_centroids

    @staticmethod
    def _split_text_by_char_weights(
        text: str,
        weights: List[float],
    ) -> List[str]:
        raw = str(text or "")
        if not raw or len(weights) <= 1:
            return [raw]

        chars = list(raw)
        total_chars = len(chars)
        if total_chars <= 1:
            return [raw] + [""] * max(0, len(weights) - 1)

        safe_weights = [max(0.0, float(w or 0.0)) for w in weights]
        total_weight = sum(safe_weights)
        if total_weight <= 0.0:
            safe_weights = [1.0] * len(weights)
            total_weight = float(len(weights))

        remaining_chars = total_chars
        remaining_weight = total_weight
        cursor = 0
        parts: List[str] = []
        for idx, weight in enumerate(safe_weights):
            remaining_slots = len(safe_weights) - idx
            if remaining_slots <= 1:
                parts.append("".join(chars[cursor:]))
                break

            target = int(round((weight / max(remaining_weight, 1e-6)) * remaining_chars))
            take = max(1, min(remaining_chars - (remaining_slots - 1), target))
            parts.append("".join(chars[cursor:cursor + take]))
            cursor += take
            remaining_chars -= take
            remaining_weight -= weight

        if len(parts) < len(weights):
            parts.extend([""] * (len(weights) - len(parts)))
        return parts[:len(weights)]

    @staticmethod
    def _split_text_into_delimited_units(
        text: str,
        delimiters: str,
    ) -> List[str]:
        source = str(text or "")
        if not source:
            return []

        units: List[str] = []
        start = 0
        closing_chars = "\"'”’)]}）】》」』"
        idx = 0
        while idx < len(source):
            if source[idx] not in delimiters:
                idx += 1
                continue
            end = idx + 1
            while end < len(source) and source[end] in closing_chars:
                end += 1
            while end < len(source) and source[end].isspace():
                end += 1
            units.append(source[start:end])
            start = end
            idx = end

        if start < len(source):
            units.append(source[start:])
        return [unit for unit in units if unit]

    @staticmethod
    def _split_text_into_token_units(text: str) -> List[str]:
        source = str(text or "")
        if not source:
            return []
        return [unit for unit in re.findall(r"\S+\s*", source, flags=re.UNICODE) if unit]

    @staticmethod
    def _split_units_by_weights(
        units: List[str],
        weights: List[float],
    ) -> List[str]:
        if not units or len(weights) <= 1:
            return ["".join(units)] if units else []

        def _char_len(item: str) -> int:
            return len([ch for ch in str(item or "") if not ch.isspace()])

        safe_weights = [max(0.0, float(w or 0.0)) for w in weights]
        total_weight = sum(safe_weights)
        if total_weight <= 0.0:
            safe_weights = [1.0] * len(weights)
            total_weight = float(len(weights))

        total_chars = sum(_char_len(unit) for unit in units)
        if total_chars <= 0:
            return ["".join(units)] if units else []

        targets: List[int] = []
        remaining_chars = total_chars
        remaining_weight = total_weight
        for idx, weight in enumerate(safe_weights):
            remaining_slots = len(safe_weights) - idx
            if remaining_slots <= 1:
                targets.append(max(1, remaining_chars))
                break
            target = int(round((weight / max(remaining_weight, 1e-6)) * remaining_chars))
            target = max(1, min(remaining_chars - (remaining_slots - 1), target))
            targets.append(target)
            remaining_chars -= target
            remaining_weight -= weight

        parts: List[str] = []
        cursor = 0
        for idx, target in enumerate(targets):
            remaining_slots = len(targets) - idx
            if remaining_slots <= 1:
                parts.append("".join(units[cursor:]).strip())
                break

            current_units: List[str] = []
            current_len = 0
            while cursor < len(units):
                remaining_units_after = len(units) - (cursor + 1)
                remaining_slots_after = remaining_slots - 1
                if current_units and remaining_units_after < remaining_slots_after:
                    break

                unit = units[cursor]
                unit_len = max(1, _char_len(unit))
                if current_units:
                    before_diff = abs(current_len - target)
                    after_diff = abs(current_len + unit_len - target)
                    if (
                        after_diff > before_diff
                        and current_len >= max(1, int(round(target * 0.65)))
                        and remaining_units_after >= remaining_slots_after
                    ):
                        break

                current_units.append(unit)
                current_len += unit_len
                cursor += 1

                if current_len >= target and remaining_units_after >= remaining_slots_after:
                    break

            if not current_units and cursor < len(units):
                current_units.append(units[cursor])
                cursor += 1

            parts.append("".join(current_units).strip())

        if len(parts) < len(weights):
            parts.extend([""] * (len(weights) - len(parts)))
        return parts[:len(weights)]

    @classmethod
    def _split_text_by_duration_weights(
        cls,
        text: str,
        weights: List[float],
        *,
        allow_hard_fallback: bool = False,
    ) -> List[str]:
        raw = str(text or "")
        if not raw or len(weights) <= 1:
            return [raw]

        safe_weights = [max(0.0, float(w or 0.0)) for w in weights]
        units = cls._split_text_into_delimited_units(raw, "。！？!?；;")
        if len(units) < len(weights):
            refined_units: List[str] = []
            for unit in units or [raw]:
                pieces = cls._split_text_into_delimited_units(unit, "，,、：:")
                if len(pieces) > 1:
                    refined_units.extend(pieces)
                else:
                    refined_units.append(unit)
            units = refined_units
        if len(units) < len(weights):
            refined_units = []
            for unit in units or [raw]:
                pieces = cls._split_text_into_token_units(unit)
                if len(pieces) > 1:
                    refined_units.extend(pieces)
                else:
                    refined_units.append(unit)
            units = refined_units

        meaningful_units = [
            unit for unit in units
            if any(not ch.isspace() for ch in str(unit or ""))
        ]
        if len(meaningful_units) >= len(weights):
            return cls._split_units_by_weights(meaningful_units, safe_weights)

        if not allow_hard_fallback:
            return []
        return cls._split_text_by_char_weights(raw, safe_weights)

    def _transcribe_clip_without_speakers(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        *,
        file_name: str = "",
        language_override: str = "",
    ) -> List[TranscriptionSegment]:
        preferred_lang = self._normalize_language_tag(language_override)
        if (
            preferred_lang
            and str(self.config.get("asr.engine", "auto") or "auto").strip().lower() == "auto"
        ):
            try:
                self.ensure_engine_for_language(preferred_lang)
            except Exception as e:
                logger.debug(
                    "Clip-level ASR engine switch skipped [%s -> %s]: %s",
                    file_name or "clip",
                    preferred_lang,
                    e,
                )

        segments: List[TranscriptionSegment] = []
        if self.engine_name == "funasr":
            segments = self._transcribe_funasr(audio_np, sample_rate, file_name)
        elif self.engine_name == "faster_whisper":
            segments = self._transcribe_faster_whisper(
                audio_np,
                sample_rate,
                file_name,
                language_override=language_override,
            )
        elif self.engine_name == "mlx_whisper":
            segments = self._transcribe_mlx_whisper(
                audio_np,
                sample_rate,
                file_name,
                language_override=language_override,
            )
        else:
            raise RuntimeError(f"Unknown engine: {self.engine_name}")

        if language_override:
            for seg in segments:
                seg.language = language_override or seg.language

        return self._rebalance_segment_granularity(
            segments,
            file_name=file_name,
        )

    def _retranscribe_diarization_piece_segments(
        self,
        *,
        audio_np: np.ndarray,
        sample_rate: int,
        seg: TranscriptionSegment,
        piece: Dict[str, Any],
        piece_index: int,
        file_name: str = "",
    ) -> List[TranscriptionSegment]:
        if sample_rate <= 0 or audio_np is None:
            return []

        audio_f32 = self._audio_to_numpy(audio_np)
        if audio_f32.size <= 0:
            return []

        total_duration = float(audio_f32.size) / float(max(1, sample_rate))
        core_start = max(0.0, float(piece.get("start", 0.0) or 0.0))
        core_end = min(
            total_duration,
            max(core_start, float(piece.get("end", core_start) or core_start)),
        )
        piece_duration = core_end - core_start
        if piece_duration < 0.45 or piece_duration > 18.0:
            return []

        speaker = self._normalize_speaker_id(piece.get("speaker", "0"))
        context_pad = min(0.10, max(0.02, piece_duration * 0.05))
        win_start = max(0.0, core_start - context_pad)
        win_end = min(total_duration, core_end + context_pad)
        s0 = int(max(0, round(win_start * sample_rate)))
        s1 = int(min(audio_f32.size, round(win_end * sample_rate)))
        if s1 <= s0:
            return []

        clip = np.ascontiguousarray(audio_f32[s0:s1], dtype=np.float32)
        if clip.size < int(max(1, sample_rate * 0.20)):
            return []

        piece_tag = f"turn-split-{piece_index + 1}"
        if file_name:
            piece_tag = f"{file_name}#{piece_tag}"

        try:
            decoded = self._transcribe_clip_without_speakers(
                clip,
                sample_rate,
                file_name=piece_tag,
                language_override=str(getattr(seg, "language", "") or ""),
            )
        except Exception as e:
            logger.debug("Diarization turn re-transcribe failed [%s]: %s", piece_tag, e)
            return []

        recovered: List[TranscriptionSegment] = []
        min_overlap_ratio = 0.12 if piece_duration < 1.0 else 0.20
        for item in decoded:
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue

            abs_start = float(getattr(item, "start", 0.0) or 0.0) + win_start
            abs_end = float(getattr(item, "end", abs_start) or abs_start) + win_start
            if abs_end <= abs_start:
                abs_end = abs_start + 0.05

            overlap = self._segment_overlap_seconds(
                abs_start,
                abs_end,
                core_start,
                core_end,
            )
            seg_len = max(1e-6, abs_end - abs_start)
            if overlap <= 0.0 or (overlap / seg_len) < min_overlap_ratio:
                continue

            clipped_start = max(core_start, abs_start)
            clipped_end = min(core_end, abs_end)
            if clipped_end <= clipped_start:
                continue

            words: List[Dict[str, Any]] = []
            for word in list(getattr(item, "words", []) or []):
                if not isinstance(word, dict):
                    continue
                shifted = dict(word)
                try:
                    if shifted.get("start", None) is not None:
                        shifted["start"] = float(shifted["start"]) + win_start
                    if shifted.get("end", None) is not None:
                        shifted["end"] = float(shifted["end"]) + win_start
                except Exception:
                    pass
                words.append(shifted)

            recovered.append(
                TranscriptionSegment(
                    start=clipped_start,
                    end=clipped_end,
                    text=text,
                    speaker=speaker,
                    language=str(getattr(item, "language", "") or getattr(seg, "language", "") or ""),
                    confidence=float(getattr(item, "confidence", 0.0) or 0.0),
                    words=words or None,
                )
            )

        recovered.sort(key=lambda item: (float(item.start), float(item.end)))
        return recovered

    def _split_whisper_segments_by_diarization_turns(
        self,
        *,
        segments: List[TranscriptionSegment],
        diar_segments: List[Dict[str, Any]],
        overlap_tracks: Optional[List[Dict[str, Any]]] = None,
        audio_np: Optional[np.ndarray] = None,
        sample_rate: int = 0,
        file_name: str = "",
    ) -> List[TranscriptionSegment]:
        if not segments or not diar_segments:
            return segments

        split_segments: List[TranscriptionSegment] = []
        split_count = 0
        expanded_count = 0
        redecoded_segment_count = 0
        min_piece_sec = 0.12

        for seg in segments:
            seg_start = float(getattr(seg, "start", 0.0) or 0.0)
            seg_end = float(getattr(seg, "end", seg_start) or seg_start)
            if seg_end - seg_start <= min_piece_sec:
                split_segments.append(seg)
                continue

            raw_pieces: List[Dict[str, Any]] = []
            for diar in diar_segments:
                piece_start = max(seg_start, float(diar.get("start", 0.0) or 0.0))
                piece_end = min(seg_end, float(diar.get("end", piece_start) or piece_start))
                if piece_end - piece_start < min_piece_sec:
                    continue
                speaker = self._normalize_speaker_id(diar.get("speaker", "0"))
                if (
                    raw_pieces
                    and raw_pieces[-1]["speaker"] == speaker
                    and piece_start <= float(raw_pieces[-1]["end"]) + 1e-3
                ):
                    raw_pieces[-1]["end"] = max(float(raw_pieces[-1]["end"]), piece_end)
                    continue
                raw_pieces.append(
                    {
                        "start": piece_start,
                        "end": piece_end,
                        "speaker": speaker,
                    }
                )

            segment_words = [
                dict(item)
                for item in list(getattr(seg, "words", []) or [])
                if isinstance(item, dict)
            ]
            annotated_words = self._annotate_words_with_speakers(
                words=segment_words,
                diar_segments=diar_segments,
                overlap_tracks=overlap_tracks,
            )
            word_runs = self._word_speaker_runs(annotated_words)
            unique_piece_speakers = {str(item.get("speaker", "") or "") for item in raw_pieces}
            unique_word_speakers = {str(item.get("speaker", "") or "") for item in word_runs}

            word_derived_pieces: List[Dict[str, Any]] = []
            if len(word_runs) >= 2 and len(unique_word_speakers) >= 2:
                for run in word_runs:
                    piece_start = max(seg_start, float(run.get("start", seg_start) or seg_start))
                    piece_end = min(seg_end, float(run.get("end", piece_start) or piece_start))
                    if piece_end - piece_start <= 0.02:
                        continue
                    word_derived_pieces.append(
                        {
                            "start": piece_start,
                            "end": piece_end,
                            "speaker": self._normalize_speaker_id(str(run.get("speaker", "0"))),
                            "words": [dict(word) for word in list(run.get("words") or []) if isinstance(word, dict)],
                        }
                    )

            if (
                len(raw_pieces) <= 1
                or len(unique_piece_speakers) <= 1
            ) and not word_derived_pieces:
                split_segments.append(seg)
                continue

            word_chunks: List[List[Dict[str, Any]]] = []
            assigned_words = 0
            pieces_for_assignment: List[Dict[str, Any]] = []
            using_word_derived_pieces = bool(word_derived_pieces)
            if using_word_derived_pieces:
                pieces_for_assignment = [
                    {
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "speaker": str(item["speaker"]),
                    }
                    for item in word_derived_pieces
                ]
                word_chunks = [
                    [dict(word) for word in list(item.get("words") or []) if isinstance(word, dict)]
                    for item in word_derived_pieces
                ]
                assigned_words = sum(len(words) for words in word_chunks)
            else:
                pieces_for_assignment = list(raw_pieces)
                word_chunks = [[] for _ in pieces_for_assignment]
                if annotated_words:
                    for word in annotated_words:
                        word_start = word.get("start", None)
                        word_end = word.get("end", None)
                        try:
                            if word_start is not None and word_end is not None:
                                word_start_f = float(word_start)
                                word_end_f = float(word_end)
                                if word_end_f <= word_start_f:
                                    word_end_f = word_start_f + 0.04
                                word_mid = (word_start_f + word_end_f) * 0.5
                            elif word_start is not None:
                                word_start_f = float(word_start)
                                word_end_f = word_start_f + 0.04
                                word_mid = word_start_f
                            elif word_end is not None:
                                word_end_f = float(word_end)
                                word_start_f = word_end_f - 0.04
                                word_mid = word_end_f
                            else:
                                continue
                        except Exception:
                            continue
                        best_idx = -1
                        best_score = 0.0
                        preferred_speaker = str(word.get("speaker", "") or "").strip()
                        for idx, piece in enumerate(pieces_for_assignment):
                            piece_start = float(piece["start"])
                            piece_end = float(piece["end"])
                            piece_speaker = str(piece.get("speaker", "") or "").strip()
                            overlap = self._segment_overlap_seconds(
                                word_start_f,
                                word_end_f,
                                piece_start,
                                piece_end,
                            )
                            if overlap <= 0.0 and not (piece_start - 1e-3 <= word_mid < piece_end + 1e-3):
                                continue
                            score = overlap if overlap > 0.0 else 0.02
                            if preferred_speaker and preferred_speaker == piece_speaker:
                                score *= 1.35
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                        if best_idx >= 0:
                            word_chunks[best_idx].append(word)
                            assigned_words += 1

            retranscribed_chunks: List[List[TranscriptionSegment]] = [[] for _ in raw_pieces]
            can_redecode = (
                (not using_word_derived_pieces)
                and
                assigned_words <= 0
                and audio_np is not None
                and sample_rate > 0
                and len(raw_pieces) <= 4
                and (seg_end - seg_start) <= 20.0
            )
            if can_redecode:
                for idx, piece in enumerate(raw_pieces):
                    retranscribed_chunks[idx] = self._retranscribe_diarization_piece_segments(
                        audio_np=audio_np,
                        sample_rate=sample_rate,
                        seg=seg,
                        piece=piece,
                        piece_index=idx,
                        file_name=file_name,
                    )

            text_chunks: List[str] = []
            if assigned_words > 0 and any(word_chunks):
                for words in word_chunks:
                    text_chunks.append(
                        "".join(str(item.get("text", "") or "") for item in words).strip()
                    )
            else:
                recovered_any = any(bool(items) for items in retranscribed_chunks)
                weights = [
                    max(0.0, float(piece["end"]) - float(piece["start"]))
                    for piece in pieces_for_assignment
                ]
                text_chunks = self._split_text_by_duration_weights(
                    seg.text,
                    weights,
                    allow_hard_fallback=recovered_any,
                )
                if not text_chunks and not recovered_any:
                    split_segments.append(seg)
                    continue

            produced_for_segment = 0
            for idx, piece in enumerate(pieces_for_assignment):
                recovered_segments = (
                    retranscribed_chunks[idx]
                    if idx < len(retranscribed_chunks)
                    else []
                )
                if recovered_segments:
                    split_segments.extend(recovered_segments)
                    produced_for_segment += len(recovered_segments)
                    redecoded_segment_count += len(recovered_segments)
                    continue

                text_value = str(text_chunks[idx] if idx < len(text_chunks) else "" or "").strip()
                words_value = word_chunks[idx] if idx < len(word_chunks) and word_chunks[idx] else None
                if not text_value and not words_value:
                    continue
                split_segments.append(
                    TranscriptionSegment(
                        start=float(piece["start"]),
                        end=float(piece["end"]),
                        text=text_value or str(seg.text or "").strip(),
                        speaker=str(piece["speaker"]),
                        language=str(getattr(seg, "language", "") or ""),
                        confidence=float(getattr(seg, "confidence", 0.0) or 0.0),
                        words=words_value,
                    )
                )
                produced_for_segment += 1

            if produced_for_segment >= 2:
                split_count += 1
                expanded_count += produced_for_segment - 1
            elif produced_for_segment == 0:
                split_segments.append(seg)

        if split_count > 0:
            logger.info(
                "  Split ASR segments by diarization turns: %d -> %d (%d expanded)%s",
                len(segments),
                len(split_segments),
                expanded_count,
                f" ({file_name})" if file_name else "",
            )
        if redecoded_segment_count > 0:
            logger.info(
                "  Re-transcribed %d diarization-aligned segment(s) without word timestamps%s",
                redecoded_segment_count,
                f" ({file_name})" if file_name else "",
            )
        return split_segments

    def _map_speaker_labels(
        self, segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        if not segments:
            return segments

        speaker_stats = self._speaker_duration_stats(segments)
        anchor_points: Dict[str, Dict[str, float]] = {}
        for idx, seg in enumerate(sorted(segments, key=lambda item: (float(item.start), float(item.end)))):
            raw = str(getattr(seg, "speaker", "") or "").strip()
            if not raw:
                continue
            start = max(0.0, float(getattr(seg, "start", 0.0) or 0.0))
            end = max(start, float(getattr(seg, "end", start) or start))
            duration = max(0.0, end - start)
            text = str(getattr(seg, "text", "") or "")
            compact_chars = len([ch for ch in text if not ch.isspace()])
            is_substantive = (
                duration >= 0.80
                or compact_chars >= 4
                or (duration >= 0.35 and compact_chars >= 2)
            )
            if is_substantive and raw not in anchor_points:
                anchor_points[raw] = {
                    "anchor_start": start,
                    "anchor_index": float(idx),
                }

        ordered_speakers = sorted(
            speaker_stats.keys(),
            key=lambda raw: (
                float(anchor_points.get(raw, {}).get("anchor_start", float(speaker_stats.get(raw, {}).get("first_start", float("inf"))))),
                float(anchor_points.get(raw, {}).get("anchor_index", float(speaker_stats.get(raw, {}).get("first_index", float("inf"))))),
                float(speaker_stats.get(raw, {}).get("first_start", float("inf"))),
                -float(speaker_stats.get(raw, {}).get("duration", 0.0) or 0.0),
                -float(speaker_stats.get(raw, {}).get("count", 0.0) or 0.0),
                str(raw),
            ),
        )

        speaker_map: Dict[str, str] = {}
        for idx, raw in enumerate(ordered_speakers):
            if idx < len(self.speaker_labels):
                speaker_map[raw] = self.speaker_labels[idx]
            else:
                speaker_map[raw] = f"S{idx}"

        next_index = len(speaker_map)
        for seg in segments:
            raw = str(getattr(seg, "speaker", "") or "").strip()
            if raw not in speaker_map:
                if next_index < len(self.speaker_labels):
                    speaker_map[raw] = self.speaker_labels[next_index]
                else:
                    speaker_map[raw] = f"S{next_index}"
                next_index += 1
            seg.speaker = speaker_map[raw]
            for word in list(getattr(seg, "words", []) or []):
                if not isinstance(word, dict):
                    continue
                raw_word_speaker = str(word.get("speaker", "") or "").strip()
                if raw_word_speaker in speaker_map:
                    word["speaker"] = speaker_map[raw_word_speaker]
                overlap_speakers = word.get("overlap_speakers")
                if isinstance(overlap_speakers, list):
                    word["overlap_speakers"] = [
                        speaker_map.get(str(item or "").strip(), str(item or "").strip())
                        for item in overlap_speakers
                        if str(item or "").strip()
                    ]
                speaker_candidates = word.get("speaker_candidates")
                if isinstance(speaker_candidates, list):
                    for candidate in speaker_candidates:
                        if not isinstance(candidate, dict):
                            continue
                        candidate_raw = str(candidate.get("speaker", "") or "").strip()
                        if candidate_raw in speaker_map:
                            candidate["speaker"] = speaker_map[candidate_raw]
        if speaker_map:
            logger.info(f"  Speakers: {speaker_map}")
        return segments

    @staticmethod
    def _compact_fragment_text(text: str) -> str:
        return re.sub(r"[\W_]+", "", str(text or ""), flags=re.UNICODE)

    def _cleanup_adjacent_speaker_fragments(
        self,
        segments: List[TranscriptionSegment],
        *,
        file_name: str = "",
    ) -> List[TranscriptionSegment]:
        if len(segments) < 2:
            return segments

        cleaned: List[TranscriptionSegment] = []
        dropped = 0
        replaced = 0
        max_gap_sec = 0.45

        for seg in segments:
            text = str(getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            if not cleaned:
                cleaned.append(seg)
                continue

            prev = cleaned[-1]
            prev_speaker = str(getattr(prev, "speaker", "") or "").strip()
            cur_speaker = str(getattr(seg, "speaker", "") or "").strip()
            gap = max(
                0.0,
                float(getattr(seg, "start", 0.0) or 0.0)
                - float(getattr(prev, "end", 0.0) or 0.0),
            )
            if not prev_speaker or prev_speaker != cur_speaker or gap > max_gap_sec:
                cleaned.append(seg)
                continue

            prev_text = str(getattr(prev, "text", "") or "").strip()
            prev_compact = self._compact_fragment_text(prev_text)
            cur_compact = self._compact_fragment_text(text)
            prev_punct_only = not prev_compact
            cur_punct_only = not cur_compact

            if prev_punct_only and not cur_punct_only:
                cleaned[-1] = seg
                replaced += 1
                continue
            if cur_punct_only:
                dropped += 1
                continue

            if cur_compact and len(cur_compact) <= 1 and cur_compact in prev_compact:
                dropped += 1
                continue
            if prev_compact and len(prev_compact) <= 1 and prev_compact in cur_compact:
                cleaned[-1] = seg
                replaced += 1
                continue
            if prev_compact and cur_compact and prev_compact == cur_compact and len(cur_compact) <= 2:
                choose_current = (
                    len(text) > len(prev_text)
                    or float(getattr(seg, "confidence", 0.0) or 0.0)
                    >= float(getattr(prev, "confidence", 0.0) or 0.0)
                )
                if choose_current:
                    cleaned[-1] = seg
                    replaced += 1
                else:
                    dropped += 1
                continue

            cleaned.append(seg)

        if dropped > 0 or replaced > 0:
            logger.info(
                "  Cleaned adjacent same-speaker text fragments: dropped=%d, replaced=%d%s",
                dropped,
                replaced,
                f" ({file_name})" if file_name else "",
            )
        return cleaned

    # Chunk-level transcription and overlap stitching

    def transcribe_chunks(
        self,
        chunks: List[tuple],
        sample_rate: int = 16000,
        file_name: str = "",
        file_path: str = "",
        map_speakers: bool = True,
        overlap_sec: float = 1.0,
        chunk_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[TranscriptionSegment]:
        all_segments = []
        dropped_by_boundary = 0
        overlap_hint = max(0.0, float(overlap_sec or 0.0))
        total_chunks = len(chunks)

        def _emit_chunk_progress(
            phase: str,
            chunk_index: int,
            *,
            error: str = "",
            kept_segments: int = 0,
        ) -> None:
            if chunk_progress_callback is None:
                return
            payload: Dict[str, Any] = {
                "phase": str(phase or "progress"),
                "chunk_index": int(chunk_index),
                "chunk_total": int(total_chunks),
                "file_name": str(file_name or ""),
                "file_path": str(file_path or ""),
                "kept_segments": int(max(0, kept_segments)),
            }
            if error:
                payload["error"] = str(error)
            try:
                chunk_progress_callback(payload)
            except Exception:
                pass

        for i, (chunk_audio, chunk_start, chunk_end) in enumerate(
            chunks
        ):
            _emit_chunk_progress("start", i + 1)
            logger.info(
                f"  Chunk {i+1}/{len(chunks)}: "
                f"[{chunk_start:.1f}s - {chunk_end:.1f}s]"
            )
            left_overlap = (
                max(0.0, float(chunks[i - 1][2]) - float(chunk_start))
                if i > 0
                else 0.0
            )
            right_overlap = (
                max(0.0, float(chunk_end) - float(chunks[i + 1][1]))
                if i + 1 < total_chunks
                else 0.0
            )
            if i > 0 and left_overlap <= 0:
                left_overlap = overlap_hint
            if i + 1 < total_chunks and right_overlap <= 0:
                right_overlap = overlap_hint
            keep_start = float(chunk_start) + left_overlap * 0.5
            keep_end = float(chunk_end) - right_overlap * 0.5
            if keep_end <= keep_start:
                keep_start = float(chunk_start)
                keep_end = float(chunk_end)

            try:
                detected_chunk_lang = ""
                detected_chunk_prob = 0.0
                if (
                    str(self.config.get("asr.engine", "auto") or "auto").strip().lower() == "auto"
                    and self._safe_bool(
                        self.config.get("language.chunk_level_engine_switch", True),
                        True,
                    )
                ):
                    chunk_duration = max(0.0, float(chunk_end) - float(chunk_start))
                    min_chunk_duration = max(
                        0.0,
                        self._safe_float(
                            self.config.get(
                                "language.chunk_level_engine_switch_min_duration_sec",
                                6.0,
                            ),
                            6.0,
                        ),
                    )
                    if chunk_duration >= min_chunk_duration:
                        detected_chunk_lang = self.detect_audio_language(
                            chunk_audio,
                            sample_rate=sample_rate,
                            file_name=f"{file_name}_chunk{i+1}",
                        )
                        chunk_probe_meta = dict(self.last_language_probe or {})
                        detected_chunk_prob = self._safe_float(
                            chunk_probe_meta.get("probability", 0.0),
                            0.0,
                        )
                        logger.info(
                            "  Chunk %d language: %s (p=%.2f)",
                            i + 1,
                            detected_chunk_lang or "unknown",
                            detected_chunk_prob,
                        )
                        min_switch_prob = self._safe_float(
                            self.config.get(
                                "language.chunk_engine_switch_min_probability",
                                self.config.get(
                                    "language.probe_engine_switch_min_probability",
                                    0.55,
                                ),
                            ),
                            0.60,
                        )
                        if detected_chunk_lang and detected_chunk_prob >= min_switch_prob:
                            switched = self.ensure_engine_for_language(detected_chunk_lang)
                            if switched:
                                logger.info(
                                    "  Chunk %d engine selected for %s: %s",
                                    i + 1,
                                    detected_chunk_lang,
                                    self.engine_name,
                                )
                        elif detected_chunk_lang:
                            logger.info(
                                "  Chunk %d skip engine switch due low confidence "
                                "(p=%.2f < %.2f).",
                                i + 1,
                                detected_chunk_prob,
                                min_switch_prob,
                            )

                # Defer diarization until merged full-audio segments are available.
                # Chunk-local speaker IDs are not stable across chunk boundaries.
                segs = self.transcribe(
                    chunk_audio, sample_rate,
                    file_name=f"{file_name}_chunk{i+1}",
                    map_speakers=False,
                    assign_speakers_enabled=False,
                    language_override=detected_chunk_lang,
                    cleanup_after=False,
                )

                absolute_segments: List[TranscriptionSegment] = []
                for seg in segs:
                    abs_start = float(seg.start) + float(chunk_start)
                    abs_end = float(seg.end) + float(chunk_start)
                    if abs_end <= abs_start:
                        abs_end = abs_start + 0.05
                    seg.start = abs_start
                    seg.end = abs_end
                    absolute_segments.append(seg)

                filtered: List[TranscriptionSegment] = []
                for seg in absolute_segments:
                    center = (float(seg.start) + float(seg.end)) * 0.5
                    if i > 0 and center < keep_start:
                        dropped_by_boundary += 1
                        continue
                    if i + 1 < total_chunks and center > keep_end:
                        dropped_by_boundary += 1
                        continue
                    filtered.append(seg)

                if filtered:
                    all_segments.extend(filtered)
                    _emit_chunk_progress("done", i + 1, kept_segments=len(filtered))
                else:
                    all_segments.extend(absolute_segments)
                    _emit_chunk_progress("done", i + 1, kept_segments=len(absolute_segments))

            except Exception as e:
                logger.error(f"  Chunk {i+1} failed: {e}")
                _emit_chunk_progress("failed", i + 1, error=str(e))
                self._maybe_force_cuda_cleanup(force=True)
                continue

        self._maybe_force_cuda_cleanup()

        if dropped_by_boundary > 0:
            logger.info(
                f"  Chunk boundary smoothing kept sentence-level timing "
                f"(trimmed {dropped_by_boundary} overlap segment(s))."
            )
        all_segments.sort(key=lambda s: s.start)
        if map_speakers:
            logger.info(
                "  Deferred speaker diarization until after chunk merge "
                "for stable cross-chunk speaker IDs."
            )
        return all_segments

    def cleanup(self):
        logger.info("Cleaning up transcriber...")
        self._unload_all()
        self._unload_language_probe()
        self._pyannote_diar_pipeline = None
        self._pyannote_diar_setup = {}
        self._pyannote_incompatible_models = set()
        self._nemo_startup_preload_attempted = False
        self._nemo_startup_preload_summary = ""
        self._nemo_msdd_model_unavailable = False
        self._nemo_msdd_model_unavailable_reason = ""
        self._nemo_sortformer_model = None
        self._nemo_sortformer_setup = {}
        self._overlap_osd_pipeline = None
        self._overlap_osd_setup = {}
        self._overlap_separator_model = None
        self._overlap_separator_setup = {}
        self._pyannote_separation_blocked = False
        self._pyannote_separation_block_reason = ""
        _force_cuda_cleanup()
